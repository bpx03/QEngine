#pragma once
// Minimal JSON reader: enough for config.json and safetensors headers.
#include <cstdlib>
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
    static Json parse_value(const std::string& s, size_t& p) {
        ws(s, p);
        if (p >= s.size()) throw std::runtime_error("json: eof");
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
                j.obj.emplace(std::move(k), parse_value(s, p));
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
                j.arr.push_back(parse_value(s, p));
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
        if (s[p] != '"') throw std::runtime_error("json: expected string");
        p++;
        std::string out;
        while (s[p] != '"') {
            if (s[p] == '\\') {
                p++;
                char e = s[p];
                if (e == 'n') out += '\n';
                else if (e == 't') out += '\t';
                else if (e == 'u') { out += '?'; p += 4; }
                else out += e;
                p++;
            } else {
                out += s[p++];
            }
        }
        p++;
        return out;
    }
};
