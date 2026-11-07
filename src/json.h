#pragma once
// Minimal JSON reader: enough for config.json and safetensors headers.
#include <cstdlib>
#include <cmath>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

struct Json {
    enum Type { Null, Bool, Num, Str, Arr, Obj } type = Null;
    bool b = false;
    double num = 0;
    std::string str;
    std::vector<Json> arr;
    std::map<std::string, Json> obj;

    const Json& operator[](const std::string& k) const {
        auto it = obj.find(k);
        if (it == obj.end()) throw std::runtime_error("json: missing key " + k);
        return it->second;
    }
    const Json& operator[](size_t i) const { return arr.at(i); }
    bool has(const std::string& k) const { return obj.count(k) != 0; }
    long long i() const { return (long long)num; }
    double f() const { return num; }

    static Json parse(const std::string& s) {
        size_t p = 0;
        Json j = parse_value(s, p, 0);
        ws(s, p);
        if (p != s.size()) throw std::runtime_error("json: trailing data");
        return j;
    }

private:
    static void ws(const std::string& s, size_t& p) {
        while (p < s.size() && (s[p] == ' ' || s[p] == '\n' || s[p] == '\r' || s[p] == '\t')) p++;
    }
    static char at(const std::string& s, size_t p) { return p < s.size() ? s[p] : '\0'; }
    static Json parse_value(const std::string& s, size_t& p, int depth) {
        if (depth > 64) throw std::runtime_error("json: nested too deep");
        ws(s, p);
        if (p >= s.size()) throw std::runtime_error("json: eof");
        Json j;
        const char c = s[p];
        if (c == '{' || c == '[') {
            const bool object = c == '{';
            const char close = object ? '}' : ']';
            j.type = object ? Obj : Arr;
            ++p;
            ws(s, p);
            if (at(s, p) == close) { ++p; return j; }
            while (true) {
                ws(s, p);
                if (object) {
                    std::string key = parse_string(s, p);
                    ws(s, p);
                    if (at(s, p) != ':') throw std::runtime_error("json: expected :");
                    ++p;
                    if (!j.obj.emplace(std::move(key), parse_value(s, p, depth + 1)).second)
                        throw std::runtime_error("json: duplicate key");
                } else {
                    j.arr.push_back(parse_value(s, p, depth + 1));
                }
                ws(s, p);
                if (at(s, p) == close) { ++p; return j; }
                if (at(s, p) != ',') throw std::runtime_error("json: expected comma or closing delimiter");
                ++p;
            }
        } else if (c == '"') {
            j.type = Str;
            j.str = parse_string(s, p);
        } else if (s.compare(p, 4, "true") == 0) {
            j.type = Bool; j.b = true; p += 4;
        } else if (s.compare(p, 5, "false") == 0) {
            j.type = Bool; p += 5;
        } else if (s.compare(p, 4, "null") == 0) {
            p += 4;
        } else {
            const size_t begin = p;
            if (at(s, p) == '-') ++p;
            if (at(s, p) == '0') ++p;
            else {
                if (at(s, p) < '1' || at(s, p) > '9') throw std::runtime_error("json: expected number");
                while (at(s, p) >= '0' && at(s, p) <= '9') ++p;
            }
            if (at(s, p) == '.') {
                ++p;
                if (at(s, p) < '0' || at(s, p) > '9') throw std::runtime_error("json: expected fraction");
                while (at(s, p) >= '0' && at(s, p) <= '9') ++p;
            }
            if (at(s, p) == 'e' || at(s, p) == 'E') {
                ++p;
                if (at(s, p) == '+' || at(s, p) == '-') ++p;
                if (at(s, p) < '0' || at(s, p) > '9') throw std::runtime_error("json: expected exponent");
                while (at(s, p) >= '0' && at(s, p) <= '9') ++p;
            }
            j.type = Num;
            j.num = strtod(s.c_str() + begin, nullptr);
            if (!std::isfinite(j.num)) throw std::runtime_error("json: number out of range");
        }
        return j;
    }
    static std::string parse_string(const std::string& s, size_t& p) {
        if (at(s, p) != '"') throw std::runtime_error("json: expected string");
        ++p;
        std::string out;
        while (true) {
            if (p >= s.size()) throw std::runtime_error("json: unterminated string");
            const unsigned char c = s[p++];
            if (c == '"') return out;
            if (c < 0x20) throw std::runtime_error("json: control character in string");
            if (c != '\\') { out += char(c); continue; }
            if (p >= s.size()) throw std::runtime_error("json: unterminated escape");
            const char e = s[p++];
            if (e == 'n') out += '\n';
            else if (e == 't') out += '\t';
            else if (e == 'r') out += '\r';
            else if (e == 'b') out += '\b';
            else if (e == 'f') out += '\f';
            else if (e == '"' || e == '\\' || e == '/') out += e;
            else if (e == 'u') {
                for (int i = 0; i < 4; ++i) {
                    const char h = at(s, p++);
                    if (!((h >= '0' && h <= '9') || (h >= 'a' && h <= 'f') || (h >= 'A' && h <= 'F')))
                        throw std::runtime_error("json: invalid unicode escape");
                }
                out += '?';  // Preserve the reader's existing Unicode escape representation.
            } else throw std::runtime_error("json: invalid escape");
        }
    }
};
