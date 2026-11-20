#ifndef MBUFO_MODELS_MINIMAX_H3_JSON_HPP_
#define MBUFO_MODELS_MINIMAX_H3_JSON_HPP_

#include <cstdint>
#include <filesystem>
#include <map>
#include <stdexcept>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace mbufo::minimax_h3::json {

class Error : public std::runtime_error {
public:
  using std::runtime_error::runtime_error;
};

class Value {
public:
  using Array = std::vector<Value>;
  using Object = std::map<std::string, Value, std::less<>>;

  Value() = default;
  explicit Value(bool value) : value_(value) {}
  explicit Value(std::int64_t value) : value_(value) {}
  explicit Value(std::uint64_t value) : value_(value) {}
  explicit Value(double value) : value_(value) {}
  explicit Value(std::string value) : value_(std::move(value)) {}
  explicit Value(Array value) : value_(std::move(value)) {}
  explicit Value(Object value) : value_(std::move(value)) {}

  [[nodiscard]] bool IsNull() const noexcept;
  [[nodiscard]] bool IsBool() const noexcept;
  [[nodiscard]] bool IsInteger() const noexcept;
  [[nodiscard]] bool IsNumber() const noexcept;
  [[nodiscard]] bool IsString() const noexcept;
  [[nodiscard]] bool IsArray() const noexcept;
  [[nodiscard]] bool IsObject() const noexcept;

  [[nodiscard]] bool AsBool() const;
  [[nodiscard]] std::int64_t AsInt64() const;
  [[nodiscard]] std::uint64_t AsUint64() const;
  [[nodiscard]] double AsDouble() const;
  [[nodiscard]] const std::string& AsString() const;
  [[nodiscard]] const Array& AsArray() const;
  [[nodiscard]] const Object& AsObject() const;
  [[nodiscard]] const Value* Find(std::string_view key) const noexcept;

private:
  using Storage =
      std::variant<std::monostate, bool, std::int64_t, std::uint64_t, double,
                   std::string, Array, Object>;
  Storage value_;
};

[[nodiscard]] Value Parse(std::string_view input);
[[nodiscard]] Value ParseFile(const std::filesystem::path& path,
                              std::size_t maximum_bytes);

}  // namespace mbufo::minimax_h3::json

#endif  // MBUFO_MODELS_MINIMAX_H3_JSON_HPP_
