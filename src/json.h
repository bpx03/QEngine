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
        Json j = parse_value(s, p);
        return j;
    }

private:
    static void ws(const std::string& s, size_t& p) {
        while (p < s.size() && (s[p] == ' ' || s[p] == '\n' || s[p] == '\r' || s[p] == '\t')) p++;
    }
    static Json parse_value(const std::string& s, size_t& p, int depth = 0) {
        ws(s, p);
        if (p >= s.size()) throw std::runtime_error("json: eof");
        if (depth > 64) throw std::runtime_error("json: nested too deep");
        Json j;
        char c = s[p];
        if (c == '{') {
            j.type = Obj;
            p++;
            ws(s, p);
            if (s[p] == '}') { p++; return j; }
            while (true) {
                ws(s, p);
                std::string k = parse_string(s, p);
                ws(s, p);
                if (s[p] != ':') throw std::runtime_error("json: expected :");
                p++;
                j.obj.emplace(std::move(k), parse_value(s, p, depth + 1));
                ws(s, p);
                if (s[p] == ',') { p++; continue; }
                if (s[p] == '}') { p++; break; }
                throw std::runtime_error("json: expected , or }");
            }
        } else if (c == '[') {
            j.type = Arr;
            p++;
            ws(s, p);
            if (s[p] == ']') { p++; return j; }
            while (true) {
                j.arr.push_back(parse_value(s, p, depth + 1));
                ws(s, p);
                if (s[p] == ',') { p++; continue; }
                if (s[p] == ']') { p++; break; }
                throw std::runtime_error("json: expected , or ]");
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
            j.type = Num;
            char* end;
            j.num = strtod(s.c_str() + p, &end);
            p = end - s.c_str();
        }
        return j;
    }
    static std::string parse_string(const std::string& s, size_t& p) {
        if (p >= s.size() || s[p] != '"') throw std::runtime_error("json: expected string");
        p++;
        std::string out;
        while (true) {  // every read bounded: an unterminated string is an error, not a read past the end
            if (p >= s.size()) throw std::runtime_error("json: unterminated string");
            if (s[p] == '"') break;
            if (s[p] == '\\') {
                if (++p >= s.size()) throw std::runtime_error("json: unterminated string");
                char e = s[p];
                if (e == 'n') out += '\n';
                else if (e == 't') out += '\t';
                else if (e == 'u') {
                    if (p + 4 >= s.size()) throw std::runtime_error("json: unterminated string");
                    out += '?';
                    p += 4;
                }
                else out += e;
                p++;
            } else {
                out += s[p++];
            }
        }
        p++;
        return out;
    }

public:
    // Strict reader for checkpoint manifests, used only by flash/checkpoint.h. Additive: Json::parse above
    // keeps its historical lenient behaviour. Strict grammar: trailing data, duplicate keys, non-standard
    // numbers (leading '+', ".5", "5.", NaN/Infinity from strtod's leniency), unknown escapes and control
    // characters inside strings are all rejected.
    struct Strict {
        const std::string& s;
        size_t p = 0;
        explicit Strict(const std::string& text) : s(text) {}
        void fail(const std::string& what) const {
            throw std::runtime_error("json strict: " + what + " at offset " + std::to_string(p)
                                     + " near: " + s.substr(p > 8 ? p - 8 : 0, 16));
        }
        char at() const { return p < s.size() ? s[p] : '\0'; }
        void ws() { while (p < s.size() && (s[p] == ' ' || s[p] == '\n' || s[p] == '\r' || s[p] == '\t')) ++p; }
        Json value(int depth) {
            if (depth > 64) fail("nested too deep");
            ws();
            if (p >= s.size()) fail("eof");
            Json j;
            const char c = s[p];
            if (c == '{') {
                j.type = Json::Obj;
                ++p;
                ws();
                if (at() == '}') { ++p; return j; }
                while (true) {
                    ws();
                    const std::string key = string();
                    ws();
                    if (at() != ':') fail("expected :");
                    ++p;
                    if (!j.obj.emplace(key, value(depth + 1)).second) fail("duplicate key: " + key);
                    ws();
                    if (at() == '}') { ++p; return j; }
                    if (at() != ',') fail("expected , or }");
                    ++p;
                }
            }
            if (c == '[') {
                j.type = Json::Arr;
                ++p;
                ws();
                if (at() == ']') { ++p; return j; }
                while (true) {
                    j.arr.push_back(value(depth + 1));
                    ws();
                    if (at() == ',') { ++p; continue; }
                    if (at() == ']') { ++p; return j; }
                    fail("expected , or ]");
                }
            }
            if (c == '"') { j.type = Json::Str; j.str = string(); return j; }
            if (s.compare(p, 4, "true") == 0) { j.type = Json::Bool; j.b = true; p += 4; return j; }
            if (s.compare(p, 5, "false") == 0) { j.type = Json::Bool; p += 5; return j; }
            if (s.compare(p, 4, "null") == 0) { p += 4; return j; }
            j.type = Json::Num;
            number(j);
            return j;
        }
        void number(Json& j) {
            const size_t begin = p;
            if (at() == '-') ++p;
            if (at() == '0') ++p;
            else if (at() >= '1' && at() <= '9') { while (at() >= '0' && at() <= '9') ++p; }
            else fail("expected number");
            if (at() == '.') {
                ++p;
                if (at() < '0' || at() > '9') fail("expected fraction digits");
                while (at() >= '0' && at() <= '9') ++p;
            }
            if (at() == 'e' || at() == 'E') {
                ++p;
                if (at() == '+' || at() == '-') ++p;
                if (at() < '0' || at() > '9') fail("expected exponent digits");
                while (at() >= '0' && at() <= '9') ++p;
            }
            j.num = std::strtod(s.c_str() + begin, nullptr);
            if (!std::isfinite(j.num)) fail("number out of range");
        }
        std::string string() {
            if (at() != '"') fail("expected string");
            ++p;
            std::string out;
            while (true) {
                if (p >= s.size()) fail("unterminated string");
                const unsigned char c = s[p++];
                if (c == '"') return out;
                if (c < 0x20) fail("control character in string");
                if (c != '\\') { out += char(c); continue; }
                if (p >= s.size()) fail("unterminated escape");
                const char e = s[p++];
                if (e == 'n') out += '\n';
                else if (e == 't') out += '\t';
                else if (e == 'r') out += '\r';
                else if (e == 'b') out += '\b';
                else if (e == 'f') out += '\f';
                else if (e == '"' || e == '\\' || e == '/') out += e;
                else if (e == 'u') {
                    for (int i = 0; i < 4; ++i) {
                        const char h = at();
                        ++p;
                        if (!((h >= '0' && h <= '9') || (h >= 'a' && h <= 'f') || (h >= 'A' && h <= 'F')))
                            fail("invalid unicode escape");
                    }
                    out += '?';  // same representation as Json::parse; manifest fields never use escapes
                } else fail("invalid escape");
            }
        }
    };
    static Json parse_strict(const std::string& s) {
        Strict r(s);
        Json j = r.value(0);
        r.ws();
        if (r.p != s.size()) r.fail("trailing data");
        return j;
    }
};
