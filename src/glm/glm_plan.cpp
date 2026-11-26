// card plan and host slice layout of the GLM engine (host code shared with the CPU-lane test)
#include <sched.h>
#include <unistd.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <string>

#include "glm.h"

namespace glm {

std::vector<Plan> make_plan(int ngpu, const int* ecols) {
    CHECK(ngpu >= 1 && ngpu <= 4, "ngpu %d", ngpu);
    std::vector<Plan> v(ngpu);
    // heads: 22/21/21 (TP4: 16 each) ; expert columns in 128-groups: 6/5/5 ; dense 32 groups each ; vocab ~equal
    // [glm-ngpu4] rows 0-2 are the 1-3 card plans unchanged; row 3 is the balanced TP4 plan
    const int hs[4][4] = {{64, 0, 0, 0}, {32, 32, 0, 0}, {22, 21, 21, 0}, {16, 16, 16, 16}};
    // shared expert column groups per card (the 6th on the card with the fewest routed columns)
    int es[4][4] = {{16, 0, 0, 0}, {8, 8, 0, 0}, {5, 6, 5, 0}, {4, 4, 4, 4}};
    // "a,b[,c[,d]]" with exactly ngpu values (the override applies to this card count's row only)
    auto parse_n = [ngpu](const char* s, int* out) {
        int x[5] = {};
        const int n = sscanf(s, "%d,%d,%d,%d,%d", &x[0], &x[1], &x[2], &x[3], &x[4]);
        if (n != ngpu) return false;
        for (int r = 0; r < ngpu; r++) out[r] = x[r];
        return true;
    };
    if (const char* s = getenv("GLM_EGROUPS")) {  // e.g. "6,5,5" (3 cards) or "4,4,4,4" (4 cards)
        int g[4] = {};
        int sum = 0;
        bool ok = ngpu > 1 && parse_n(s, g);
        for (int r = 0; ok && r < ngpu; r++) { ok &= g[r] > 0; sum += g[r]; }
        if (ok && sum == 16) for (int r = 0; r < ngpu; r++) es[ngpu - 1][r] = g[r];
        else fprintf(stderr, "[glm] GLM_EGROUPS=%s ignored: %d positive values summing to 16 expected\n", s, ngpu);
    }
    // routed expert columns per card in 64-steps (default: by the free VRAM of this box: card 1 drives the desktop)
    int ec[4][4] = {{2048, 0, 0, 0}, {1024, 1024, 0, 0}, {704, 640, 704, 0}, {512, 512, 512, 512}};
    if (const char* s = getenv("GLM_ECOLS")) {  // e.g. "704,640,704" (3 cards) or "512,512,512,512" (4 cards)
        int c[4] = {};
        int sum = 0;
        bool ok = ngpu > 1 && parse_n(s, c);
        for (int r = 0; ok && r < ngpu; r++) { ok &= c[r] > 0 && c[r] % 64 == 0; sum += c[r]; }
        if (ok && sum == INTER) for (int r = 0; r < ngpu; r++) ec[ngpu - 1][r] = c[r];
        else fprintf(stderr, "[glm] GLM_ECOLS=%s ignored: %d multiples of 64 summing to %d expected\n", s, ngpu, INTER);
    } else if (getenv("GLM_EGROUPS")) {
        for (int r = 0; r < ngpu; r++) ec[ngpu - 1][r] = es[ngpu - 1][r] * 128;
    } else if (ecols && ngpu > 1) {
        int sum = 0;
        for (int r = 0; r < ngpu; r++) sum += ecols[r];
        if (sum == INTER) for (int r = 0; r < ngpu; r++) ec[ngpu - 1][r] = ecols[r];
    }
    int h = 0, e = 0, sc = 0;
    for (int r = 0; r < ngpu; r++) {
        Plan& p = v[r];
        p.rank = r;
        p.ngpu = ngpu;
        p.h0 = h; h += hs[ngpu - 1][r]; p.h1 = h;
        p.e0 = e; e += ec[ngpu - 1][r]; p.e1 = e;
        p.s0 = sc * 128; sc += es[ngpu - 1][r]; p.s1 = sc * 128;
        p.d0 = DINTER * r / ngpu; p.d1 = DINTER * (r + 1) / ngpu;
        p.v0 = (int)((long long)VOCAB * r / ngpu); p.v1 = (int)((long long)VOCAB * (r + 1) / ngpu);
    }
    return v;
}

HostCpus host_cpus(int npool) {
    HostCpus h;
    cpu_set_t cs;
    CPU_ZERO(&cs);
    std::vector<int> ok;
    if (sched_getaffinity(0, sizeof cs, &cs) == 0)
        for (int c = 0; c < CPU_SETSIZE; c++) if (CPU_ISSET(c, &cs)) ok.push_back(c);
    if (ok.empty()) for (int c = 0; c < (int)sysconf(_SC_NPROCESSORS_ONLN); c++) ok.push_back(c);
    // physical cores in order of their first allowed thread; sibling = the next allowed thread of the same core
    std::map<std::pair<int, int>, std::vector<int>> core;  // (package, core id) -> threads
    std::vector<std::pair<int, int>> order;
    for (int c : ok) {
        int pk = 0, id = c;
        std::ifstream a("/sys/devices/system/cpu/cpu" + std::to_string(c) + "/topology/physical_package_id");
        std::ifstream b("/sys/devices/system/cpu/cpu" + std::to_string(c) + "/topology/core_id");
        if (a.good()) a >> pk;
        if (b.good()) b >> id;
        auto k = std::make_pair(pk, id);
        if (!core.count(k)) order.push_back(k);
        core[k].push_back(c);
    }
    std::vector<int> first, sib;
    for (auto& k : order) {
        first.push_back(core[k][0]);
        sib.push_back(core[k].size() > 1 ? core[k][1] : -1);
    }
    const int P = (int)first.size();
    h.coord = first[0];
    h.caller = sib[0];
    // pool threads (one per card after the first): the last core and its sibling, then the cores before it
    int last = P - 1;
    for (int c = P - 1; c > 0 && (int)h.pool.size() < npool; c--) {
        h.pool.push_back(first[c]);
        last = c;
        if ((int)h.pool.size() < npool && sib[c] >= 0) h.pool.push_back(sib[c]);
    }
    for (int c = 1; c < last; c++) h.lane.push_back(first[c]);
    if (h.lane.size() < 2) {  // a small CPU: the lane takes every thread but the coordinator's (shared with the rest)
        h.lane.clear();
        for (int c : ok) if (c != h.coord) h.lane.push_back(c);
        if (h.lane.empty()) h.lane.push_back(h.coord);  // review rel L9: one allowed CPU
    }
    if (h.lane.size() > 64) h.lane.resize(64);
    char b[256];
    snprintf(b, sizeof b, "%d cores / %zu threads allowed: coordinator %d, caller %d, pool", P, ok.size(), h.coord, h.caller);
    h.why = b;
    for (int c : h.pool) h.why += " " + std::to_string(c);
    snprintf(b, sizeof b, ", CPU lane %zu threads %d-%d", h.lane.size(), h.lane.front(), h.lane.back());
    h.why += b;
    return h;
}

void HxLayout::set(const std::vector<Plan>& p, bool gu3, bool mix) {
    ns = (int)p.size();
    stride = 0;
    mixed = mix;
    for (int r = 0; r < ns; r++) {
        sl3[r].set(p[r].ne(), true);
        sl[r].set(p[r].ne(), gu3 && !mix);
        e0[r] = p[r].e0;
        off[r] = stride;
        stride += sl[r].bytes;
    }
    if (mixed) { flag_off = stride; stride += 256; }
}

void SliceL::set(int n, bool g3) {
    ni = n;
    gu3 = g3;
    const size_t gub = g3 ? (size_t)128 * 3 * n * 4 : (size_t)512 * n * 4;
    gq = 0;
    uq = gq + gub;
    gs = uq + gub;
    us = gs + (size_t)32 * n * 2;
    dq = us + (size_t)32 * n * 2;
    ds = dq + (size_t)(n / 8) * H * 4;
    bytes = ds + (size_t)(n / 64) * H * 2;
    bytes = (bytes + 255) & ~(size_t)255;
}

}  // namespace glm
