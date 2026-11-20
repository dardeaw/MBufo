#include <atomic>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include "src/core/hip/detail/hipblaslt_plan_database.hpp"

namespace {

using mbufo::hip::detail::HipblasLtPlanDatabase;
using mbufo::hip::detail::HipblasLtPlanDatabaseKey;
using mbufo::hip::detail::HipblasLtPlanDatabaseLoadStatus;
using mbufo::hip::detail::HipblasLtPlanDataType;
using mbufo::hip::detail::HipblasLtPlanRecord;

void Expect(bool condition, const std::string& message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << '\n';
    std::exit(1);
  }
}

HipblasLtPlanDatabase MakeDatabase() {
  return {
      .key =
          {
              .hardware_fingerprint = "gfx1151-test",
              .hip_runtime_version = 70203000,
              .hipblaslt_version = 10202,
          },
      .records =
          {
              {
                  .batch_size = 128,
                  .m = 17408,
                  .k = 5120,
                  .data_type = HipblasLtPlanDataType::kBfloat16,
                  .algorithm_id = 42,
                  .algorithm_blob = {1, 2, 3, 4},
                  .workspace_bytes = 4 * 1024 * 1024,
                  .median_nanoseconds = 125000,
                  .solution_name = "solution-42",
                  .kernel_name = "kernel-42",
              },
          },
  };
}

}  // namespace

int main() {
  const auto temporary_directory =
      std::filesystem::temp_directory_path() / "mbufo-hipblaslt-cache-test";
  std::filesystem::create_directories(temporary_directory);
  const auto database_path = temporary_directory / "plans.bin";
  const auto truncated_path = temporary_directory / "truncated.bin";

  const auto expected = MakeDatabase();
  std::string error;
  Expect(mbufo::hip::detail::SaveHipblasLtPlanDatabase(database_path, expected,
                                                      &error),
         "save database: " + error);

  const auto loaded =
      mbufo::hip::detail::LoadHipblasLtPlanDatabase(database_path, expected.key);
  Expect(loaded.status == HipblasLtPlanDatabaseLoadStatus::kLoaded,
         "load compatible database");
  Expect(loaded.database.key == expected.key, "round-trip database key");
  Expect(loaded.database.records == expected.records,
         "round-trip database records");

  const auto* record = mbufo::hip::detail::FindHipblasLtPlanRecord(
      loaded.database, 128, 17408, 5120, HipblasLtPlanDataType::kBfloat16);
  Expect(record != nullptr && record->algorithm_id == 42,
         "lookup exact GEMM shape");
  Expect(mbufo::hip::detail::FindHipblasLtPlanRecord(
             loaded.database, 256, 17408, 5120,
             HipblasLtPlanDataType::kBfloat16) == nullptr,
         "reject unrecorded GEMM shape");

  auto incompatible_key = expected.key;
  ++incompatible_key.hip_runtime_version;
  const auto incompatible = mbufo::hip::detail::LoadHipblasLtPlanDatabase(
      database_path, incompatible_key);
  Expect(incompatible.status == HipblasLtPlanDatabaseLoadStatus::kIncompatible,
         "detect ROCm runtime change");

  {
    std::ofstream truncated(truncated_path, std::ios::binary);
    truncated.write("STRIXLT", 7);
  }
  const auto invalid = mbufo::hip::detail::LoadHipblasLtPlanDatabase(
      truncated_path, expected.key);
  Expect(invalid.status == HipblasLtPlanDatabaseLoadStatus::kInvalid,
         "reject truncated database");

  auto duplicate = expected;
  duplicate.records.push_back(duplicate.records.front());
  Expect(!mbufo::hip::detail::SaveHipblasLtPlanDatabase(
             temporary_directory / "duplicate.bin", duplicate, &error),
         "reject duplicate shape records");

  const auto concurrent_path = temporary_directory / "concurrent.bin";
  std::filesystem::remove(concurrent_path);
  std::atomic<bool> writers_ok{true};
  std::vector<std::thread> writers;
  for (unsigned i = 1; i <= 8; ++i)
    writers.emplace_back([&, i] {
      auto update = MakeDatabase();
      update.records[0].batch_size = i;
      std::string failure;
      if (!mbufo::hip::detail::SaveHipblasLtPlanDatabase(concurrent_path, update,
                                                        &failure))
        writers_ok = false;
    });
  for (auto& writer : writers)
    writer.join();
  const auto combined = mbufo::hip::detail::LoadHipblasLtPlanDatabase(
      concurrent_path, expected.key);
  Expect(writers_ok &&
             combined.status == HipblasLtPlanDatabaseLoadStatus::kLoaded &&
             combined.database.records.size() == 8,
         "concurrent publications retain every distinct tuning result");
  for (const auto& entry :
       std::filesystem::directory_iterator(temporary_directory))
    Expect(entry.path().filename().string().find(".tmp.") == std::string::npos,
           "publication leaves no temporary files");
  std::filesystem::remove_all(temporary_directory);
  std::cout << "PASS: hipBLASLt plan database tests\n";
  return 0;
}
