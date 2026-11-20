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
    CHECK(ngpu == 3 || ngpu == 2 || ngpu == 1, "ngpu %d", ngpu);
    std::vector<Plan> v(ngpu);
    // heads: 22/21/21 ; expert columns in 128-groups: 6/5/5 ; dense 32 groups each ; vocab ~equal
    const int hs[3][3] = {{64, 0, 0}, {32, 32, 0}, {22, 21, 21}};
    // shared expert column groups per card (the 6th on the card with the fewest routed columns)
    int es[3][3] = {{16, 0, 0}, {8, 8, 0}, {5, 6, 5}};
    if (const char* s = getenv("GLM_EGROUPS")) {  // e.g. "6,5,5"
        int a = 0, b = 0, c = 0;
        if (sscanf(s, "%d,%d,%d", &a, &b, &c) == 3 && a + b + c == 16) { es[2][0] = a; es[2][1] = b; es[2][2] = c; }
    }
    // routed expert columns per card in 64-steps (default: by the free VRAM of this box: card 1 drives the desktop)
    int ec[3][3] = {{2048, 0, 0}, {1024, 1024, 0}, {704, 640, 704}};
    if (const char* s = getenv("GLM_ECOLS")) {  // e.g. "704,640,704"
        int a = 0, b = 0, c = 0;
        if (sscanf(s, "%d,%d,%d", &a, &b, &c) == 3 && a + b + c == INTER && a % 64 == 0 && b % 64 == 0 && c % 64 == 0) {
            ec[2][0] = a; ec[2][1] = b; ec[2][2] = c;
        }
    } else if (getenv("GLM_EGROUPS")) {
        for (int r = 0; r < 3; r++) ec[2][r] = es[2][r] * 128;
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
