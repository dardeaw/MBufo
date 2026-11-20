#include "src/models/minimax_h3/json.hpp"

#include <charconv>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <sstream>
#include <system_error>

namespace mbufo::minimax_h3::json {

namespace {

[[noreturn]] void TypeError(std::string_view expected) {
  throw Error("JSON value is not " + std::string(expected));
}

class Parser {
public:
  explicit Parser(std::string_view input) : input_(input) {}

  Value Run() {
    SkipSpace();
    Value result = ParseValue(0);
    SkipSpace();
    if (position_ != input_.size()) {
      Fail("trailing bytes");
    }
    return result;
  }

private:
  static constexpr std::size_t kMaximumDepth = 128;

  [[noreturn]] void Fail(std::string_view message) const {
    throw Error("JSON parse error at byte " + std::to_string(position_) + ": " +
                std::string(message));
  }

  void SkipSpace() {
    while (position_ < input_.size()) {
      const char value = input_[position_];
      if (value != ' ' && value != '\t' && value != '\n' && value != '\r') {
        break;
      }
      ++position_;
    }
  }

  bool Consume(char expected) {
    if (position_ < input_.size() && input_[position_] == expected) {
      ++position_;
      return true;
    }
    return false;
  }

  void RequireLiteral(std::string_view literal) {
    if (input_.substr(position_, literal.size()) != literal) {
      Fail("invalid literal");
    }
    position_ += literal.size();
  }

  // Recursive descent is bounded by kMaximumDepth.
  // NOLINTNEXTLINE(misc-no-recursion)
  Value ParseValue(std::size_t depth) {
    if (depth > kMaximumDepth) {
      Fail("maximum nesting depth exceeded");
    }
    if (position_ >= input_.size()) {
      Fail("unexpected end of input");
    }
    switch (input_[position_]) {
      case 'n':
        RequireLiteral("null");
        return Value();
      case 't':
        RequireLiteral("true");
        return Value(true);
      case 'f':
        RequireLiteral("false");
        return Value(false);
      case '"':
        return Value(ParseString());
      case '[':
        return ParseArray(depth + 1);
      case '{':
        return ParseObject(depth + 1);
      default:
        if (input_[position_] == '-' ||
            (input_[position_] >= '0' && input_[position_] <= '9')) {
          return ParseNumber();
        }
        Fail("unexpected token");
    }
  }

  static void AppendUtf8(std::string& output, std::uint32_t codepoint) {
    if (codepoint <= 0x7FU) {
      output.push_back(static_cast<char>(codepoint));
    } else if (codepoint <= 0x7FFU) {
      output.push_back(static_cast<char>(0xC0U | (codepoint >> 6U)));
      output.push_back(static_cast<char>(0x80U | (codepoint & 0x3FU)));
    } else if (codepoint <= 0xFFFFU) {
      output.push_back(static_cast<char>(0xE0U | (codepoint >> 12U)));
      output.push_back(static_cast<char>(0x80U | ((codepoint >> 6U) & 0x3FU)));
      output.push_back(static_cast<char>(0x80U | (codepoint & 0x3FU)));
    } else {
      output.push_back(static_cast<char>(0xF0U | (codepoint >> 18U)));
      output.push_back(static_cast<char>(0x80U | ((codepoint >> 12U) & 0x3FU)));
      output.push_back(static_cast<char>(0x80U | ((codepoint >> 6U) & 0x3FU)));
      output.push_back(static_cast<char>(0x80U | (codepoint & 0x3FU)));
    }
  }

  std::uint32_t ParseHex4() {
    if (position_ + 4 > input_.size()) {
      Fail("truncated unicode escape");
    }
    std::uint32_t value = 0;
    for (int index = 0; index < 4; ++index) {
      const char digit = input_[position_++];
      value <<= 4U;
      if (digit >= '0' && digit <= '9') {
        value |= static_cast<std::uint32_t>(digit - '0');
      } else if (digit >= 'a' && digit <= 'f') {
        value |= static_cast<std::uint32_t>(digit - 'a' + 10);
      } else if (digit >= 'A' && digit <= 'F') {
        value |= static_cast<std::uint32_t>(digit - 'A' + 10);
      } else {
        Fail("invalid unicode escape");
      }
    }
    return value;
  }

  std::string ParseString() {
    if (!Consume('"')) {
      Fail("expected string");
    }
    std::string output;
    while (position_ < input_.size()) {
      const unsigned char value =
          static_cast<unsigned char>(input_[position_++]);
      if (value == '"') {
        return output;
      }
      if (value < 0x20U) {
        Fail("unescaped control byte in string");
      }
      if (value != '\\') {
        output.push_back(static_cast<char>(value));
        continue;
      }
      if (position_ >= input_.size()) {
        Fail("truncated string escape");
      }
      const char escaped = input_[position_++];
      switch (escaped) {
        case '"':
        case '\\':
        case '/':
          output.push_back(escaped);
          break;
        case 'b':
          output.push_back('\b');
          break;
        case 'f':
          output.push_back('\f');
          break;
        case 'n':
          output.push_back('\n');
          break;
        case 'r':
          output.push_back('\r');
          break;
        case 't':
          output.push_back('\t');
          break;
        case 'u': {
          std::uint32_t codepoint = ParseHex4();
          if (codepoint >= 0xD800U && codepoint <= 0xDBFFU) {
            if (position_ + 2 > input_.size() || input_[position_] != '\\' ||
                input_[position_ + 1] != 'u') {
              Fail("high surrogate lacks low surrogate");
            }
            position_ += 2;
            const std::uint32_t low = ParseHex4();
            if (low < 0xDC00U || low > 0xDFFFU) {
              Fail("invalid low surrogate");
            }
            codepoint =
                0x10000U + ((codepoint - 0xD800U) << 10U) + (low - 0xDC00U);
          } else if (codepoint >= 0xDC00U && codepoint <= 0xDFFFU) {
            Fail("unpaired low surrogate");
          }
          AppendUtf8(output, codepoint);
          break;
        }
        default:
          Fail("invalid string escape");
      }
    }
    Fail("unterminated string");
  }

  // NOLINTNEXTLINE(misc-no-recursion)
  Value ParseArray(std::size_t depth) {
    Consume('[');
    Value::Array output;
    SkipSpace();
    if (Consume(']')) {
      return Value(std::move(output));
    }
    while (true) {
      SkipSpace();
      output.push_back(ParseValue(depth));
      SkipSpace();
      if (Consume(']')) {
        return Value(std::move(output));
      }
      if (!Consume(',')) {
        Fail("expected comma in array");
      }
    }
  }

  // NOLINTNEXTLINE(misc-no-recursion)
  Value ParseObject(std::size_t depth) {
    Consume('{');
    Value::Object output;
    SkipSpace();
    if (Consume('}')) {
      return Value(std::move(output));
    }
    while (true) {
      SkipSpace();
      if (position_ >= input_.size() || input_[position_] != '"') {
        Fail("expected object key");
      }
      std::string key = ParseString();
      SkipSpace();
      if (!Consume(':')) {
        Fail("expected colon after object key");
      }
      SkipSpace();
      auto [iterator, inserted] =
          output.emplace(std::move(key), ParseValue(depth));
      if (!inserted) {
        Fail("duplicate object key " + iterator->first);
      }
      SkipSpace();
      if (Consume('}')) {
        return Value(std::move(output));
      }
      if (!Consume(',')) {
        Fail("expected comma in object");
      }
    }
  }

  Value ParseNumber() {
    const std::size_t start = position_;
    const bool negative = Consume('-');
    if (position_ >= input_.size()) {
      Fail("truncated number");
    }
    if (Consume('0')) {
      if (position_ < input_.size() && input_[position_] >= '0' &&
          input_[position_] <= '9') {
        Fail("number has a leading zero");
      }
    } else {
      if (input_[position_] < '1' || input_[position_] > '9') {
        Fail("invalid number");
      }
      while (position_ < input_.size() && input_[position_] >= '0' &&
             input_[position_] <= '9') {
        ++position_;
      }
    }
    bool floating = false;
    if (Consume('.')) {
      floating = true;
      const std::size_t fraction_start = position_;
      while (position_ < input_.size() && input_[position_] >= '0' &&
             input_[position_] <= '9') {
        ++position_;
      }
      if (fraction_start == position_) {
        Fail("fraction has no digits");
      }
    }
    if (position_ < input_.size() &&
        (input_[position_] == 'e' || input_[position_] == 'E')) {
      floating = true;
      ++position_;
      if (position_ < input_.size() &&
          (input_[position_] == '+' || input_[position_] == '-')) {
        ++position_;
      }
      const std::size_t exponent_start = position_;
      while (position_ < input_.size() && input_[position_] >= '0' &&
             input_[position_] <= '9') {
        ++position_;
      }
      if (exponent_start == position_) {
        Fail("exponent has no digits");
      }
    }

    const std::string_view token = input_.substr(start, position_ - start);
    if (!floating) {
      if (negative) {
        std::int64_t value = 0;
        const auto result =
            std::from_chars(token.data(), token.data() + token.size(), value);
        if (result.ec != std::errc{} ||
            result.ptr != token.data() + token.size()) {
          Fail("signed integer is out of range");
        }
        return Value(value);
      }
      std::uint64_t value = 0;
      const auto result =
          std::from_chars(token.data(), token.data() + token.size(), value);
      if (result.ec != std::errc{} ||
          result.ptr != token.data() + token.size()) {
        Fail("unsigned integer is out of range");
      }
      return Value(value);
    }
    double value = 0.0;
    const auto result =
        std::from_chars(token.data(), token.data() + token.size(), value);
    if (result.ec != std::errc{} || result.ptr != token.data() + token.size() ||
        !std::isfinite(value)) {
      Fail("floating-point number is invalid");
    }
    return Value(value);
  }

  std::string_view input_;
  std::size_t position_{0};
};

}  // namespace

bool Value::IsNull() const noexcept {
  return std::holds_alternative<std::monostate>(value_);
}
bool Value::IsBool() const noexcept {
  return std::holds_alternative<bool>(value_);
}
bool Value::IsInteger() const noexcept {
  return std::holds_alternative<std::int64_t>(value_) ||
         std::holds_alternative<std::uint64_t>(value_);
}
bool Value::IsNumber() const noexcept {
  return IsInteger() || std::holds_alternative<double>(value_);
}
bool Value::IsString() const noexcept {
  return std::holds_alternative<std::string>(value_);
}
bool Value::IsArray() const noexcept {
  return std::holds_alternative<Array>(value_);
}
bool Value::IsObject() const noexcept {
  return std::holds_alternative<Object>(value_);
}

bool Value::AsBool() const {
  if (const auto* value = std::get_if<bool>(&value_)) {
    return *value;
  }
  TypeError("a boolean");
}

std::int64_t Value::AsInt64() const {
  if (const auto* value = std::get_if<std::int64_t>(&value_)) {
    return *value;
  }
  if (const auto* value = std::get_if<std::uint64_t>(&value_);
      value != nullptr &&
      *value <= static_cast<std::uint64_t>(
                    std::numeric_limits<std::int64_t>::max())) {
    return static_cast<std::int64_t>(*value);
  }
  TypeError("a signed integer");
}

std::uint64_t Value::AsUint64() const {
  if (const auto* value = std::get_if<std::uint64_t>(&value_)) {
    return *value;
  }
  if (const auto* value = std::get_if<std::int64_t>(&value_);
      value != nullptr && *value >= 0) {
    return static_cast<std::uint64_t>(*value);
  }
  TypeError("an unsigned integer");
}

double Value::AsDouble() const {
  if (const auto* value = std::get_if<double>(&value_)) {
    return *value;
  }
  if (const auto* value = std::get_if<std::int64_t>(&value_)) {
    return static_cast<double>(*value);
  }
  if (const auto* value = std::get_if<std::uint64_t>(&value_)) {
    return static_cast<double>(*value);
  }
  TypeError("a number");
}

const std::string& Value::AsString() const {
  if (const auto* value = std::get_if<std::string>(&value_)) {
    return *value;
  }
  TypeError("a string");
}

const Value::Array& Value::AsArray() const {
  if (const auto* value = std::get_if<Array>(&value_)) {
    return *value;
  }
  TypeError("an array");
}

const Value::Object& Value::AsObject() const {
  if (const auto* value = std::get_if<Object>(&value_)) {
    return *value;
  }
  TypeError("an object");
}

const Value* Value::Find(std::string_view key) const noexcept {
  const auto* object = std::get_if<Object>(&value_);
  if (object == nullptr) {
    return nullptr;
  }
  const auto iterator = object->find(key);
  return iterator == object->end() ? nullptr : &iterator->second;
}

Value Parse(std::string_view input) {
  return Parser(input).Run();
}

Value ParseFile(const std::filesystem::path& path, std::size_t maximum_bytes) {
  std::error_code error;
  const std::uintmax_t size = std::filesystem::file_size(path, error);
  if (error) {
    throw Error("cannot stat JSON file " + path.string() + ": " +
                error.message());
  }
  if (size > maximum_bytes) {
    throw Error("JSON file exceeds size limit: " + path.string());
  }
  std::ifstream stream(path, std::ios::binary);
  if (!stream) {
    throw Error("cannot open JSON file " + path.string());
  }
  std::string bytes(static_cast<std::size_t>(size), '\0');
  if (!bytes.empty()) {
    stream.read(bytes.data(), static_cast<std::streamsize>(bytes.size()));
  }
  if (!stream ||
      stream.gcount() != static_cast<std::streamsize>(bytes.size())) {
    throw Error("cannot read complete JSON file " + path.string());
  }
  return Parse(bytes);
}

}  // namespace mbufo::minimax_h3::json
