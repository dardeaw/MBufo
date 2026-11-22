#include "src/cache/disk_catalog.hpp"

#include <sys/stat.h>

#include <algorithm>
#include <map>
#include <set>
#include <stdexcept>

namespace mbufo::cache {
namespace {
template<class Visit>
void Dependencies(const DiskManifest& manifest, Visit visit) {
  for (const auto& component : manifest.components) {
    for (const auto& payload : component.chunks)
      visit(false, payload.file);
    for (const auto& payload : {component.tail, component.private_state})
      if (payload)
        visit(true, payload->file);
  }
}
}  // namespace
struct DiskCatalog::Impl {
  using Key = std::pair<bool, DiskFileId>;
  struct Record {
    ResourceCharge metadata;
    std::shared_ptr<const DiskDescription> description;
    std::uint64_t used{};
    bool invalid{};
  };
  struct Bucket {
    ResourceCharge metadata;
    std::map<std::uint64_t, ResourceCharge> checkpoints;
  };
  ResourceLedger* ledger;
  DiskStore* store;
  PrefixIndex* index;
  ResourceCharge metadata;
  std::map<std::uint64_t, Record> records;
  std::map<std::pair<std::uint64_t, std::uint64_t>, std::uint64_t> lru;
  std::map<Key, Bucket> reverse;
  std::uint64_t clock{};
  bool loading{true};
  Impl(ResourceLedger& resources, DiskStore& disk, PrefixIndex& prefixes,
       ResourceReservation reservation)
      : ledger(&resources),
        store(&disk),
        index(&prefixes),
        metadata(reservation.Convert()) {}
  ~Impl() {
    while (!records.empty())
      Erase(records.begin());
  }
  std::uint64_t Tick() {
    if (clock == UINT64_MAX)
      throw std::overflow_error("disk LRU clock exhausted");
    return ++clock;
  }
  void AddDependency(Key key, std::uint64_t id) {
    auto bucket = reverse.find(key);
    if (bucket == reverse.end()) {
      auto admission =
          ledger->Reserve(ResourceCategory::kMetadata,
                          sizeof(decltype(reverse)::value_type) + 64);
      bucket = reverse.emplace(key, Bucket{}).first;
      try {
        bucket->second.metadata = admission.Convert();
      } catch (...) {
        reverse.erase(bucket);
        throw;
      }
    }
    auto& references = bucket->second.checkpoints;
    if (references.contains(id))
      return;
    auto admission =
        ledger->Reserve(ResourceCategory::kMetadata,
                        sizeof(decltype(Bucket::checkpoints)::value_type) + 64);
    auto reference = references.emplace(id, ResourceCharge{}).first;
    try {
      reference->second = admission.Convert();
    } catch (...) {
      references.erase(reference);
      throw;
    }
  }
  void RemoveDependency(Key key, std::uint64_t id) {
    auto bucket = reverse.find(key);
    if (bucket == reverse.end())
      return;
    auto& references = bucket->second.checkpoints;
    if (auto it = references.find(id); it != references.end()) {
      auto charge = std::move(it->second);
      auto node = references.extract(it);
    }
    if (references.empty()) {
      auto charge = std::move(bucket->second.metadata);
      auto node = reverse.extract(bucket);
    }
  }
  void Erase(decltype(records)::iterator it) {
    const auto id = it->first;
    Dependencies(it->second.description->Manifest(), [&](bool priv, auto file) {
      RemoveDependency({priv, file}, id);
    });
    lru.erase({it->second.used, id});
    auto charge = std::move(it->second.metadata);
    auto node = records.extract(it);
  }
};
DiskCatalog::DiskCatalog(ResourceLedger& ledger, DiskStore& store,
                         PrefixIndex& index)
    : impl_(std::make_unique<Impl>(
          ledger, store, index,
          ledger.Reserve(ResourceCategory::kMetadata, sizeof(Impl)))) {
  for (const auto& entry : store.Entries())
    if (entry.Durable())
      Track(entry.Manifest().checkpoint);
  impl_->loading = false;
}
DiskCatalog::~DiskCatalog() = default;
void DiskCatalog::Track(CheckpointId id) {
  auto description = impl_->store->Describe(id);
  if (!description)
    throw std::invalid_argument("cannot track missing disk checkpoint");
  auto existing = impl_->records.find(id.value);
  if (existing != impl_->records.end()) {
    if (existing->second.description->Epoch() == description->Epoch())
      return;
    impl_->index->RemoveDurable(
        *existing->second.description,
        existing->second.invalid ||
            existing->second.description->Quarantined());
    impl_->Erase(existing);
  }
  const auto used = impl_->loading ? 0 : impl_->Tick();
  auto admission = impl_->ledger->Reserve(
      ResourceCategory::kMetadata,
      sizeof(decltype(impl_->records)::value_type) +
          sizeof(decltype(impl_->lru)::value_type) + 128);
  auto record =
      impl_->records
          .emplace(id.value,
                   Impl::Record{{}, std::move(description), used, false})
          .first;
  try {
    impl_->lru.emplace(std::make_pair(used, id.value), id.value);
    Dependencies(record->second.description->Manifest(),
                 [&](bool priv, auto file) {
                   impl_->AddDependency({priv, file}, id.value);
                 });
    record->second.metadata = admission.Convert();
  } catch (...) {
    impl_->Erase(record);
    throw;
  }
}
std::shared_ptr<const DiskDescription> DiskCatalog::Find(
    CheckpointId id) const {
  auto it = impl_->records.find(id.value);
  if (it == impl_->records.end() || it->second.invalid ||
      !it->second.description->Pin())
    return {};
  return it->second.description;
}
void DiskCatalog::Touch(CheckpointId id) {
  auto it = impl_->records.find(id.value);
  if (it == impl_->records.end() || it->second.invalid)
    return;
  const auto used = impl_->Tick();
  auto node = impl_->lru.extract({it->second.used, id.value});
  node.key() = {used, id.value};
  it->second.used = used;
  impl_->lru.insert(std::move(node));
}
bool DiskCatalog::EvictOne() {
  Reconcile();
  for (auto it = impl_->lru.begin(); it != impl_->lru.end();) {
    auto record = impl_->records.find(it->second);
    ++it;
    const auto& description = *record->second.description;
    auto result = impl_->store->TryRetire(description.Manifest().checkpoint,
                                          description.Epoch());
    if (result == DiskRetireResult::kBusy)
      return false;
    if (result == DiskRetireResult::kPinned)
      continue;
    impl_->index->RemoveDurable(description);
    impl_->Erase(record);
    if (result == DiskRetireResult::kRetired)
      return true;
  }
  return false;
}
void DiskCatalog::Invalidate(bool private_file, DiskFileId file) {
  impl_->store->InvalidateDependency(private_file, file);
  auto it = impl_->reverse.find({private_file, file});
  if (it == impl_->reverse.end())
    return;
  for (const auto& [id, charge] : it->second.checkpoints) {
    (void)charge;
    auto& record = impl_->records.at(id);
    record.invalid = true;
    const auto& description = *record.description;
    impl_->store->Invalidate(description.Manifest().checkpoint,
                             description.Epoch());
    impl_->index->RemoveDurable(description, true);
  }
}
void DiskCatalog::Reconcile() {
  for (auto& [id, record] : impl_->records) {
    (void)id;
    if (!record.invalid && record.description->Quarantined()) {
      record.invalid = true;
      impl_->index->RemoveDurable(*record.description, true);
    }
  }
}
std::size_t DiskCatalog::ReclaimInvalid() {
  Reconcile();
  std::size_t retired{};
  for (auto it = impl_->records.begin(); it != impl_->records.end();) {
    auto current = it++;
    if (!current->second.invalid)
      continue;
    const auto& description = *current->second.description;
    auto result = impl_->store->TryRetire(description.Manifest().checkpoint,
                                          description.Epoch());
    if (result == DiskRetireResult::kBusy)
      break;
    if (result == DiskRetireResult::kPinned)
      continue;
    retired += result == DiskRetireResult::kRetired;
    impl_->Erase(current);
  }
  return retired;
}
std::vector<DiskCatalogEntryInfo> DiskCatalog::ListEntries(
    const std::filesystem::path& directory,
    std::span<const std::uint8_t> active_compatibility) const {
  constexpr char kHex[] = "0123456789abcdef";
  std::vector<DiskCatalogEntryInfo> out;
  out.reserve(impl_->records.size());
  for (const auto& [id, record] : impl_->records) {
    (void)id;
    const auto& desc = *record.description;
    const auto& m = desc.Manifest();
    DiskCatalogEntryInfo info;
    info.checkpoint_id = m.checkpoint.value;
    info.lineage_id = m.lineage.value;
    info.manifest_file = DiskFileName(desc.ManifestFile());
    info.compatibility_hex.reserve(m.compatibility.size() * 2);
    for (std::uint8_t b : m.compatibility) {
      info.compatibility_hex.push_back(kHex[b >> 4]);
      info.compatibility_hex.push_back(kHex[b & 15]);
    }
    info.compatible =
        active_compatibility.size() == m.compatibility.size() &&
        std::equal(m.compatibility.begin(), m.compatibility.end(),
                   active_compatibility.begin());
    info.token_count = m.tokens.size();
    info.purpose = m.purpose == CheckpointPurpose::kLearned ? "learned" : "prompt";
    info.invalid = record.invalid || desc.Quarantined();

    std::set<DiskFileId> seen_chunks;
    std::set<DiskFileId> seen_private;
    for (const auto& c : m.components) {
      for (const auto& p : c.chunks) {
        if (seen_chunks.insert(p.file).second) {
          info.chunk_bytes += p.bytes;
        }
      }
      for (const auto& opt : {c.tail, c.private_state}) {
        if (opt && seen_private.insert(opt->file).second) {
          info.private_bytes += opt->bytes;
        }
      }
    }
    info.chunk_count = seen_chunks.size();

    if (!directory.empty()) {
      const auto mf_path =
          directory / ("v" + std::to_string(kDiskFormatVersion)) /
          "manifests" / info.manifest_file;
      struct stat st{};
      if (::stat(mf_path.c_str(), &st) == 0) {
        info.manifest_bytes = static_cast<std::uint64_t>(st.st_size);
        info.mtime_epoch = static_cast<std::int64_t>(st.st_mtime);
      }
    }
    out.push_back(std::move(info));
  }
  std::sort(out.begin(), out.end(),
            [](const DiskCatalogEntryInfo& a, const DiskCatalogEntryInfo& b) {
              if (a.mtime_epoch != b.mtime_epoch)
                return a.mtime_epoch > b.mtime_epoch;
              return a.checkpoint_id > b.checkpoint_id;
            });
  return out;
}

DiskRetireResult DiskCatalog::Retire(CheckpointId id) {
  Reconcile();
  auto record = impl_->records.find(id.value);
  if (record == impl_->records.end())
    return DiskRetireResult::kMissing;
  const auto& description = *record->second.description;
  const auto result = impl_->store->TryRetire(
      description.Manifest().checkpoint, description.Epoch());
  if (result == DiskRetireResult::kBusy || result == DiskRetireResult::kPinned)
    return result;
  impl_->index->RemoveDurable(description, record->second.invalid);
  impl_->Erase(record);
  return result;
}

std::size_t DiskCatalog::Size() const {
  return impl_->records.size();
}
}  // namespace mbufo::cache
