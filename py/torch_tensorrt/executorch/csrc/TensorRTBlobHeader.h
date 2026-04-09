#pragma once

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace torch_tensorrt {
namespace executorch {

static constexpr char kTensorRTMagic[4] = {'T', 'R', '0', '1'};
// Must match HEADER_SIZE = struct.calcsize(HEADER_FORMAT) in
// py/torch_tensorrt/executorch/_serialization.py
static constexpr uint32_t kHeaderSize = 32;

struct TensorRTBlobHeader {
  uint32_t metadata_offset;
  uint32_t metadata_size;
  uint32_t engine_offset;
  uint64_t engine_size;
  std::vector<std::string> input_binding_names;
  std::vector<std::string> output_binding_names;
  bool hw_compatible = false;
  int device_id = 0;

  static const void* engine_data(const void* blob, const TensorRTBlobHeader& h) {
    return static_cast<const uint8_t*>(blob) + h.engine_offset;
  }

  static bool parse(const void* data, size_t size, TensorRTBlobHeader& out) {
    if (size < kHeaderSize) {
      return false;
    }

    auto* bytes = static_cast<const uint8_t*>(data);

    if (std::memcmp(bytes, kTensorRTMagic, 4) != 0) {
      return false;
    }

    auto read_u32 = [&](size_t offset) -> uint32_t {
      uint32_t val;
      std::memcpy(&val, bytes + offset, sizeof(val));
      return val;
    };
    auto read_u64 = [&](size_t offset) -> uint64_t {
      uint64_t val;
      std::memcpy(&val, bytes + offset, sizeof(val));
      return val;
    };

    out.metadata_offset = read_u32(4);
    out.metadata_size = read_u32(8);
    out.engine_offset = read_u32(12);
    out.engine_size = read_u64(16);

    if (out.engine_offset % 16 != 0) {
      return false;
    }
    if (static_cast<size_t>(out.metadata_offset) + out.metadata_size > size) {
      return false;
    }
    if (static_cast<size_t>(out.engine_offset) + out.engine_size > size) {
      return false;
    }

    // Parse JSON metadata (hand-written for fixed schema)
    std::string json_str(
        reinterpret_cast<const char*>(bytes + out.metadata_offset),
        out.metadata_size);

    if (!parse_metadata_json(json_str, out)) {
      return false;
    }

    return true;
  }

 private:
  // Skip whitespace
  static size_t skip_ws(const std::string& s, size_t pos) {
    while (pos < s.size() && (s[pos] == ' ' || s[pos] == '\t' ||
                              s[pos] == '\n' || s[pos] == '\r')) {
      ++pos;
    }
    return pos;
  }

  // Parse a quoted string, return end position after closing quote
  static size_t parse_string(
      const std::string& s,
      size_t pos,
      std::string& out) {
    if (pos >= s.size() || s[pos] != '"') {
      return std::string::npos;
    }
    ++pos;
    out.clear();
    while (pos < s.size() && s[pos] != '"') {
      if (s[pos] == '\\' && pos + 1 < s.size()) {
        ++pos;
      }
      out += s[pos++];
    }
    if (pos >= s.size()) {
      return std::string::npos;
    }
    return pos + 1; // skip closing quote
  }

  // Skip a JSON value (string, number, bool, null, object, array)
  static size_t skip_value(const std::string& s, size_t pos) {
    pos = skip_ws(s, pos);
    if (pos >= s.size()) return std::string::npos;

    if (s[pos] == '"') {
      std::string dummy;
      return parse_string(s, pos, dummy);
    } else if (s[pos] == '{') {
      int depth = 1;
      ++pos;
      while (pos < s.size() && depth > 0) {
        if (s[pos] == '{') ++depth;
        else if (s[pos] == '}') --depth;
        else if (s[pos] == '"') {
          std::string dummy;
          pos = parse_string(s, pos, dummy);
          if (pos == std::string::npos) return pos;
          continue;
        }
        ++pos;
      }
      return pos;
    } else if (s[pos] == '[') {
      int depth = 1;
      ++pos;
      while (pos < s.size() && depth > 0) {
        if (s[pos] == '[') ++depth;
        else if (s[pos] == ']') --depth;
        else if (s[pos] == '"') {
          std::string dummy;
          pos = parse_string(s, pos, dummy);
          if (pos == std::string::npos) return pos;
          continue;
        }
        ++pos;
      }
      return pos;
    } else {
      // number, bool, null — read until delimiter
      while (pos < s.size() && s[pos] != ',' && s[pos] != '}' &&
             s[pos] != ']' && s[pos] != ' ' && s[pos] != '\n') {
        ++pos;
      }
      return pos;
    }
  }

  // IMPORTANT: The JSON metadata must emit fields in this order:
  // 1. io_bindings (array)  2. hardware_compatible (bool)  3. device_id (int)
  // The parser searches forward from the end of io_bindings for remaining fields.
  static bool parse_metadata_json(
      const std::string& json,
      TensorRTBlobHeader& out) {
    out.input_binding_names.clear();
    out.output_binding_names.clear();
    out.hw_compatible = false;
    out.device_id = 0;

    // Find "io_bindings" array
    size_t bindings_pos = json.find("\"io_bindings\"");
    if (bindings_pos == std::string::npos) {
      return false;
    }

    // Find the '[' after "io_bindings":
    size_t arr_start = json.find('[', bindings_pos);
    if (arr_start == std::string::npos) {
      return false;
    }

    size_t pos = arr_start + 1;
    while (true) {
      pos = skip_ws(json, pos);
      if (pos >= json.size()) return false;
      if (json[pos] == ']') break;
      if (json[pos] == ',') { ++pos; continue; }

      // Parse binding object
      if (json[pos] != '{') return false;
      ++pos;

      std::string name;
      bool is_input = false;

      while (true) {
        pos = skip_ws(json, pos);
        if (pos >= json.size()) return false;
        if (json[pos] == '}') { ++pos; break; }
        if (json[pos] == ',') { ++pos; continue; }

        std::string key;
        pos = parse_string(json, pos, key);
        if (pos == std::string::npos) return false;

        pos = skip_ws(json, pos);
        if (pos >= json.size() || json[pos] != ':') return false;
        ++pos;
        pos = skip_ws(json, pos);

        if (key == "name") {
          pos = parse_string(json, pos, name);
          if (pos == std::string::npos) return false;
        } else if (key == "is_input") {
          if (json.compare(pos, 4, "true") == 0) {
            is_input = true;
            pos += 4;
          } else if (json.compare(pos, 5, "false") == 0) {
            is_input = false;
            pos += 5;
          } else {
            return false;
          }
        } else {
          pos = skip_value(json, pos);
          if (pos == std::string::npos) return false;
        }
      }

      if (is_input) {
        out.input_binding_names.push_back(name);
      } else {
        out.output_binding_names.push_back(name);
      }
    }

    // Parse "hardware_compatible" and "device_id" from top level.
    // Search after io_bindings array (pos) to avoid matching inside nested values.
    // hw_compatible is parsed for forward-compatibility; not currently used at runtime.
    size_t hw_pos = json.find("\"hardware_compatible\"", pos);
    if (hw_pos != std::string::npos) {
      size_t colon = json.find(':', hw_pos);
      if (colon != std::string::npos) {
        size_t val = skip_ws(json, colon + 1);
        out.hw_compatible = (json.compare(val, 4, "true") == 0);
      }
    }

    size_t dev_pos = json.find("\"device_id\"", pos);
    if (dev_pos != std::string::npos) {
      size_t colon = json.find(':', dev_pos);
      if (colon != std::string::npos) {
        size_t val = skip_ws(json, colon + 1);
        out.device_id = 0;
        while (val < json.size() && json[val] >= '0' && json[val] <= '9') {
          out.device_id = out.device_id * 10 + (json[val] - '0');
          ++val;
        }
      }
    }

    return true;
  }
};

} // namespace executorch
} // namespace torch_tensorrt
