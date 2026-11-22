#ifndef MBUFO_CACHE_DISK_CATALOG_HPP_
#define MBUFO_CACHE_DISK_CATALOG_HPP_

#include <cstdint>
#include <filesystem>
#include <span>
#include <string>
#include <vector>

#include "src/cache/prefix_index.hpp"

namespace mbufo::cache {

struct DiskCatalogEntryInfo {
  std::uint64_t checkpoint_id{0};
  std::uint64_t lineage_id{0};
  std::string manifest_file;
  std::string compatibility_hex;
  std::size_t token_count{0};
  std::size_t chunk_count{0};
  std::uint64_t chunk_bytes{0};
  std::uint64_t private_bytes{0};
  std::uint64_t manifest_bytes{0};
  std::int64_t mtime_epoch{0};
  std::string purpose;
  bool compatible{false};
  bool invalid{false};
};
// One global LRU catalog per store, including identities not currently loaded.
// Construct at quiescence after startup discovery; Track every successful new
// publication before exposing it through lookup. Operations are caller
// serialized. Store, index and ledger outlive the catalog. Touch records a
// successful use; startup ties and untouched publications use checkpoint ID.
class DiskCatalog {
public:
  DiskCatalog(ResourceLedger&, DiskStore&, PrefixIndex&);
  ~DiskCatalog();
  DiskCatalog(const DiskCatalog&) = delete;
  DiskCatalog& operator=(const DiskCatalog&) = delete;
  void Track(CheckpointId);
  [[nodiscard]] std::shared_ptr<const DiskDescription> Find(CheckpointId) const;
  void Touch(CheckpointId);
  // Evict oldest unpinned entry. Busy writers cause an immediate false result;
  // capacity is credited only after DiskStore's durable retirement succeeds.
  bool EvictOne();
  // Reverse dependency quarantine is immediate, with no I/O-lock wait. Retire
  // quarantined manifests at a later idle boundary after reader pins settle.
  void Invalidate(bool private_file, DiskFileId);
  // Reconcile worker-discovered quarantine at a serialized caller boundary.
  // Lookup rejects quarantined descriptions immediately, even before this.
  void Reconcile();
  std::size_t ReclaimInvalid();
  [[nodiscard]] std::vector<DiskCatalogEntryInfo> ListEntries(
      const std::filesystem::path& directory,
      std::span<const std::uint8_t> active_compatibility) const;
  [[nodiscard]] DiskRetireResult Retire(CheckpointId);
  [[nodiscard]] std::size_t Size() const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
}  // namespace mbufo::cache
#endif
