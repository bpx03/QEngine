// Batched speculative steps over several sequence slots (included at the end of flash.cu: the kernels live in its
// translation unit). Rows of segment s (slot bseg[s]) are [s * rps, (s + 1) * rps): the slot's pending token + k
// drafts. Row-independent parts (GEMV, MoE, HC, lm_head) run once over all rows; the sequence parts take per-row
// metadata (position, KV region start row, slot) or per-segment state from device pointer tables, so one graph per
// (segments, drafts, sampling) serves every set of slots.
#pragma once

namespace flash {
static_assert(XROWS == PFM, "rbcur_ is sized PFM rows (flash_seq.cu)");

// per-row metadata of a verify step: row t = segment t / rps, draft j = t % rps
__global__ void k_bmeta(const int* __restrict__ segs, int nseg, int rps, const int* __restrict__ svp, const int* __restrict__ svt,
                        const int* __restrict__ kvrb, int* __restrict__ rpos, int* __restrict__ rbase, int* __restrict__ rslot,
                        int* __restrict__ bvt, int* __restrict__ bpn) {
    const int t = threadIdx.x;
    if (t < nseg) bpn[t] = svp[segs[t] * 2 + 1];
    if (t >= nseg * rps) return;
    const int s = t / rps, j = t % rps, slot = segs[s];
    rpos[t] = svp[slot * 2] + j;
    rbase[t] = kvrb[slot];
    rslot[t] = slot;
    bvt[t] = svt[slot * (MAXM + 1) + j];
}
// chain rows of draft j (one per segment): position of the slot's next verify row + j - 1
__global__ void k_bmeta_chain(const int* __restrict__ segs, int nseg, const int* __restrict__ svp, const int* __restrict__ kvrb,
                              int j, int* __restrict__ cpos, int* __restrict__ cbase) {
    const int s = threadIdx.x;
    if (s >= nseg) return;
    cpos[s] = svp[segs[s] * 2] + j - 1;
    cbase[s] = kvrb[segs[s]];
}
__global__ void k_step_bump_b(unsigned* __restrict__ stepcs, const int* __restrict__ segs, int nseg) {
    if (threadIdx.x < nseg) stepcs[segs[threadIdx.x]] += 1u;
}
// n-gram rows of the verify rows (k_ngram per segment)
__global__ void k_ngram_b(const int* __restrict__ vtok, const int* __restrict__ rslot, int rps, const int* __restrict__ svh,
                          NgramDev ng, bf16* __restrict__ out) {
    const int t = blockIdx.x, tid = threadIdx.x;
    const int j = t % rps;
    const int* hist2 = svh + rslot[t] * 2;
    const long long t0 = vtok[t];
    const long long t1 = j >= 1 ? vtok[t - 1] : hist2[1];
    const long long t2r = j >= 2 ? vtok[t - 2] : (j == 1 ? hist2[1] : hist2[0]);
    const long long s1 = t1, s2 = t1 == EOS ? (long long)EOS : t2r;
    auto mul = [](long long a, long long b) { return (long long)((unsigned long long)a * (unsigned long long)b); };
    const long long big = mul(t0, ng.mult[0]) ^ mul(s1, ng.mult[1]);
    const long long tri = big ^ mul(s2, ng.mult[2]);
    const int hd = tid >> 4, sub = tid & 15;
    const long long mixed = hd < 8 ? big : tri;
    long long r = mixed % ng.vsz[hd];
    if (r < 0) r += ng.vsz[hd];
    const long long row = r + ng.voff[hd];
    if (ng.phase == 1) { if (sub == 0) ng.req[t * PLE_HEADS + hd] = row; return; }
    if (ng.phase == 2) { ngram_gather_q8(ng.stage + (size_t)(t * PLE_HEADS + hd) * NGRAM_Q8_ROW, sub, out + (size_t)t * H + hd * PLE_DIM); return; }
    ngram_gather(ng, row, sub, out + (size_t)t * H + hd * PLE_DIM);
}
// k_ple_conv per segment (grid (RW / 256, nseg)): history / pending rows of the segment's slot
__global__ void k_ple_conv_b(const float* __restrict__ gv, const float* __restrict__ gvn, const bf16* __restrict__ cw,
                             float* const* __restrict__ th, float* const* __restrict__ tp, const int* __restrict__ rslot,
                             const int* __restrict__ bpn, int rps, float* __restrict__ R) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x, s = blockIdx.y, r0 = s * rps;
    if (c >= RW) return;
    float* hist = th[rslot[r0]];
    float* pend = tp[rslot[r0]];
    float h[9];
#pragma unroll
    for (int j = 0; j < 9; j++) h[j] = hist[j * RW + c];
    const int n = bpn[s];
    for (int r = 0; r < n; r++) {
#pragma unroll
        for (int j = 0; j < 8; j++) h[j] = h[j + 1];
        h[8] = pend[(size_t)r * RW + c];
    }
#pragma unroll
    for (int j = 0; j < 9; j++) hist[j * RW + c] = h[j];
    const float w0 = __bfloat162float(cw[c * 4 + 0]), w1 = __bfloat162float(cw[c * 4 + 1]);
    const float w2 = __bfloat162float(cw[c * 4 + 2]), w3 = __bfloat162float(cw[c * 4 + 3]);
    for (int t = 0; t < rps; t++) {
        const size_t i = (size_t)(r0 + t) * RW + c;
        const float cur = gvn[i];
        pend[(size_t)t * RW + c] = cur;
        const float o = w0 * h[0] + w1 * h[3] + w2 * h[6] + w3 * cur;
        R[i] += gv[i] + silu(o);
#pragma unroll
        for (int j = 0; j < 8; j++) h[j] = h[j + 1];
        h[8] = cur;
    }
}
// k_dn_conv_d per segment (grid (ceil(ch / 256), nseg))
__global__ void k_dn_conv_b(const float* __restrict__ proj, int ld, int ch, const bf16* __restrict__ cw,
                            float* const* __restrict__ th, float* const* __restrict__ tp, const int* __restrict__ rslot,
                            const int* __restrict__ bpn, int rps, float* __restrict__ out) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x, s = blockIdx.y, r0 = s * rps;
    if (c >= ch) return;
    float* hist = th[rslot[r0]];
    float* pend = tp[rslot[r0]];
    float h0 = hist[c], h1 = hist[ch + c], h2 = hist[2 * ch + c];
    const int n = bpn[s];
    for (int r = 0; r < n; r++) {
        const float x = pend[(size_t)r * ch + c];
        h0 = h1; h1 = h2; h2 = x;
    }
    hist[c] = h0; hist[ch + c] = h1; hist[2 * ch + c] = h2;
    const float w0 = __bfloat162float(cw[c * 4]), w1 = __bfloat162float(cw[c * 4 + 1]);
    const float w2 = __bfloat162float(cw[c * 4 + 2]), w3 = __bfloat162float(cw[c * 4 + 3]);
    for (int t = 0; t < rps; t++) {
        const float x = proj[(size_t)(r0 + t) * ld + c];
        pend[(size_t)t * ch + c] = x;
        out[(size_t)(r0 + t) * ch + c] = silu(w0 * h0 + w1 * h1 + w2 * h2 + w3 * x);
        h0 = h1; h1 = h2; h2 = x;
    }
}
// k_dn_rec_d per segment (grid (vl, 16, nseg)): the slot's state, deferred-commit buffers and step counter
__global__ void __launch_bounds__(128) k_dn_rec_b(const float* __restrict__ cv, int ch, int kl, int vl, int k_first,
                                                  int v_first, const float* __restrict__ ab, const float* __restrict__ A,
                                                  const float* __restrict__ dtb, float* const* __restrict__ tS,
                                                  float* const* __restrict__ tp, const int* __restrict__ rslot,
                                                  const int* __restrict__ bpn, const unsigned* __restrict__ stepcs,
                                                  float* __restrict__ o, int rps) {
    __shared__ float qs[128], ks[128];
    __shared__ float red[32];
    const int hv = blockIdx.x, cg = blockIdx.y, sg = blockIdx.z, t = threadIdx.x;
    const int r0 = sg * rps, slot = rslot[r0];
    const int col = cg * 8 + (t >> 4), kb = (t & 15) * 8;
    const int kh = (v_first + hv) / (LNV / LNK) - k_first;
    const unsigned par = stepcs[slot] & 1u;
    float* pend = tp[slot];
    const float* pold = pend + (size_t)(1u - par) * MAXP * vl * DREC;
    float* pnew = pend + (size_t)par * MAXP * vl * DREC;
    float* sp = tS[slot] + ((size_t)hv * 128 + col) * 128 + kb;
    float s[8];
    {
        const float4 a = *reinterpret_cast<const float4*>(sp), b = *reinterpret_cast<const float4*>(sp + 4);
        s[0] = a.x; s[1] = a.y; s[2] = a.z; s[3] = a.w; s[4] = b.x; s[5] = b.y; s[6] = b.z; s[7] = b.w;
    }
    const int n = bpn[sg];
    for (int r = 0; r < n; r++) {
        const float* pr = pold + ((size_t)r * vl + hv) * DREC;
        const float beta = pr[256], decay = pr[257];
        float mem = 0.f;
#pragma unroll
        for (int i = 0; i < 8; i++) { s[i] *= decay; mem += s[i] * pr[kb + i]; }
#pragma unroll
        for (int off = 8; off > 0; off >>= 1) mem += __shfl_xor_sync(0xffffffffu, mem, off);
        const float delta = (pr[128 + col] - mem) * beta;
#pragma unroll
        for (int i = 0; i < 8; i++) s[i] += pr[kb + i] * delta;
    }
    *reinterpret_cast<float4*>(sp) = make_float4(s[0], s[1], s[2], s[3]);
    *reinterpret_cast<float4*>(sp + 4) = make_float4(s[4], s[5], s[6], s[7]);
    const float Ah = A[hv], dt = dtb[hv];
    for (int rr = 0; rr < rps; rr++) {
        const int r = r0 + rr;
        const float* row = cv + (size_t)r * ch;
        const float q = row[kh * 128 + t], k = row[kl * 128 + kh * 128 + t];
        const float qq = block_sum<128>(q * q, red), kk = block_sum<128>(k * k, red);
        qs[t] = q * rsqrtf(qq + 1e-6f) * rsqrtf(128.f);
        ks[t] = k * rsqrtf(kk + 1e-6f);
        __syncthreads();
        const float b = ab[(size_t)r * 2 * vl + hv], a = ab[(size_t)r * 2 * vl + vl + hv];
        const float beta = sigm(b);
        const float xa = a + dt;
        const float spl = xa > 20.f ? xa : log1pf(__expf(xa));
        const float decay = __expf(Ah * spl);
        const float v = row[2 * kl * 128 + hv * 128 + col];
        {
            float* pw = pnew + ((size_t)rr * vl + hv) * DREC;
            if (cg == 0) {
                pw[t] = ks[t];
                if (t == 0) { pw[256] = beta; pw[257] = decay; }
            }
            if ((t & 15) == 0) pw[128 + col] = v;
        }
        float mem = 0.f;
#pragma unroll
        for (int i = 0; i < 8; i++) { s[i] *= decay; mem += s[i] * ks[kb + i]; }
#pragma unroll
        for (int off = 8; off > 0; off >>= 1) mem += __shfl_xor_sync(0xffffffffu, mem, off);
        const float delta = (v - mem) * beta;
        float out = 0.f;
#pragma unroll
        for (int i = 0; i < 8; i++) { s[i] += ks[kb + i] * delta; out += s[i] * qs[kb + i]; }
#pragma unroll
        for (int off = 8; off > 0; off >>= 1) out += __shfl_xor_sync(0xffffffffu, out, off);
        if ((t & 15) == 0) o[((size_t)r * vl + hv) * 128 + col] = out;
        __syncthreads();
    }
}
// k_dn_rec_b with the per-row q / k normalization, beta and decay precomputed by k_dn_qk (qk [m][2][kl][128], gb [m][2][vl]):
// the row loop needs no block reductions or barriers (same formulas: bit-identical)
__global__ void __launch_bounds__(128) k_dn_rec_bq(const float* __restrict__ cv, int ch, int kl, int vl, int k_first,
                                                   int v_first, const float* __restrict__ qk, const float* __restrict__ gb,
                                                   float* const* __restrict__ tS, float* const* __restrict__ tp,
                                                   const int* __restrict__ rslot, const int* __restrict__ bpn,
                                                   const unsigned* __restrict__ stepcs, float* __restrict__ o, int rps) {
    const int hv = blockIdx.x, cg = blockIdx.y, sg = blockIdx.z, t = threadIdx.x;
    const int r0 = sg * rps, slot = rslot[r0];
    const int col = cg * 8 + (t >> 4), kb = (t & 15) * 8;
    const int kh = (v_first + hv) / (LNV / LNK) - k_first;
    const unsigned par = stepcs[slot] & 1u;
    float* pend = tp[slot];
    const float* pold = pend + (size_t)(1u - par) * MAXP * vl * DREC;
    float* pnew = pend + (size_t)par * MAXP * vl * DREC;
    float* sp = tS[slot] + ((size_t)hv * 128 + col) * 128 + kb;
    float s[8];
    {
        const float4 a = *reinterpret_cast<const float4*>(sp), b = *reinterpret_cast<const float4*>(sp + 4);
        s[0] = a.x; s[1] = a.y; s[2] = a.z; s[3] = a.w; s[4] = b.x; s[5] = b.y; s[6] = b.z; s[7] = b.w;
    }
    const int n = bpn[sg];
    for (int r = 0; r < n; r++) {
        const float* pr = pold + ((size_t)r * vl + hv) * DREC;
        const float beta = pr[256], decay = pr[257];
        float kv[8];
#pragma unroll
        for (int i = 0; i < 8; i++) kv[i] = pr[kb + i];  // DREC rows are not 16-byte aligned
        float mem = 0.f;
#pragma unroll
        for (int i = 0; i < 8; i++) { s[i] *= decay; mem += s[i] * kv[i]; }
#pragma unroll
        for (int off = 8; off > 0; off >>= 1) mem += __shfl_xor_sync(0xffffffffu, mem, off);
        const float delta = (pr[128 + col] - mem) * beta;
#pragma unroll
        for (int i = 0; i < 8; i++) s[i] += kv[i] * delta;
    }
    *reinterpret_cast<float4*>(sp) = make_float4(s[0], s[1], s[2], s[3]);
    *reinterpret_cast<float4*>(sp + 4) = make_float4(s[4], s[5], s[6], s[7]);
    for (int rr = 0; rr < rps; rr++) {
        const int r = r0 + rr;
        const float* qr = qk + (size_t)r * 2 * kl * 128 + kh * 128;
        const float* kr = qr + kl * 128;
        float qv[8], kv[8];
        *reinterpret_cast<float4*>(qv) = *reinterpret_cast<const float4*>(qr + kb);
        *reinterpret_cast<float4*>(qv + 4) = *reinterpret_cast<const float4*>(qr + kb + 4);
        *reinterpret_cast<float4*>(kv) = *reinterpret_cast<const float4*>(kr + kb);
        *reinterpret_cast<float4*>(kv + 4) = *reinterpret_cast<const float4*>(kr + kb + 4);
        const float decay = gb[(size_t)r * 2 * vl + hv], beta = gb[(size_t)r * 2 * vl + vl + hv];
        const float v = cv[(size_t)r * ch + 2 * kl * 128 + hv * 128 + col];
        {
            float* pw = pnew + ((size_t)rr * vl + hv) * DREC;
            if (cg == 0) {
                pw[t] = kr[t];
                if (t == 0) { pw[256] = beta; pw[257] = decay; }
            }
            if ((t & 15) == 0) pw[128 + col] = v;
        }
        float mem = 0.f;
#pragma unroll
        for (int i = 0; i < 8; i++) { s[i] *= decay; mem += s[i] * kv[i]; }
#pragma unroll
        for (int off = 8; off > 0; off >>= 1) mem += __shfl_xor_sync(0xffffffffu, mem, off);
        const float delta = (v - mem) * beta;
        float out = 0.f;
#pragma unroll
        for (int i = 0; i < 8; i++) { s[i] += kv[i] * delta; out += s[i] * qv[i]; }
#pragma unroll
        for (int off = 8; off > 0; off >>= 1) out += __shfl_xor_sync(0xffffffffu, out, off);
        if ((t & 15) == 0) o[((size_t)r * vl + hv) * 128 + col] = out;
    }
}
// k_idx_prep per segment (one block per segment, its rows in order): the slot's raw-key ring and pooled keys
__global__ void k_idx_prep_b(const float* __restrict__ ip, int ld, const bf16* __restrict__ qn, const bf16* __restrict__ kn,
                             const int* __restrict__ rpos, const int* __restrict__ rbase, const int* __restrict__ rslot,
                             float* __restrict__ iq, float* const* __restrict__ ttail, bf16* __restrict__ ikb0, Plan p, int rps) {
    __shared__ float red[32];
    __shared__ float xs[IDX_HD];
    const int d = threadIdx.x, r0 = blockIdx.x * rps;
    float* tail = ttail[rslot[r0]];
    bf16* ikb = ikb0 + (size_t)(rbase[r0] / IDX_R) * IDX_HD;
    auto norm_rope = [&](float x, const bf16* w, int pos) {
        const float ss = block_sum<128>(x * x, red);
        x = x * rsqrtf(ss / IDX_HD + EPS) * (1.f + __bfloat162float(w[d]));
        __syncthreads();
        xs[d] = x;
        __syncthreads();
        if (d < ROT) {
            const int i = d & 31;
            const float f = (float)pos * powf((float)THETA, -2.f * i / ROT);
            float sn, cs;
            sincosf(f, &sn, &cs);
            x = d < 32 ? xs[d] * cs - xs[d + 32] * sn : xs[d] * cs + xs[d - 32] * sn;
        }
        return x;
    };
    for (int tt = 0; tt < rps; tt++) {
        const int t = r0 + tt;
        const int pos = rpos[t];
        for (int h = 0; h < IDX_NH; h++) iq[((size_t)t * IDX_NH + h) * IDX_HD + d] = norm_rope(ip[(size_t)t * ld + h * IDX_HD + d], qn, pos);
        const float rk = __bfloat162float(__float2bfloat16(ip[(size_t)t * ld + IDX_NH * IDX_HD + d]));
        tail[(pos & (IDX_TAIL - 1)) * IDX_HD + d] = rk;
        __syncthreads();
        if ((pos & 3) == 3) {
            float sm = 0.f;
            for (int i = 0; i < 4; i++) sm += tail[((pos - i) & (IDX_TAIL - 1)) * IDX_HD + d];
            const float pooled = __bfloat162float(__float2bfloat16(sm * 0.25f));
            const float kr = norm_rope(pooled, kn, pos - 3);
            const int row = own_row(p, pos - 3);
            if (row >= 0) ikb[(size_t)(row >> 2) * IDX_HD + d] = __float2bfloat16(kr);
        }
        __syncthreads();
    }
}
// acceptance per segment (thread s): emitted tokens -> em[s][0 .. a + 1] (count first), the slot's position / commit
// rows / committed-token pair advance; bacc[s] = global row of the last accepted row. lim[s]: max drafts accepted.
__global__ void k_accept_b(const int* __restrict__ vtok, const int* __restrict__ otok, int k, int nseg, const int* __restrict__ segs,
                           const int* __restrict__ lim, int* __restrict__ svp, int* __restrict__ svh, int* __restrict__ bacc,
                           int* __restrict__ em) {
    const int s = threadIdx.x;
    if (s >= nseg) return;
    const int r0 = s * (k + 1), slot = segs[s], kl = min(k, lim[s]);
    int a = 0;
    while (a < kl && vtok[r0 + a + 1] == otok[r0 + a] && otok[r0 + a] != EOS && otok[r0 + a] != EOS + 2) a++;
    int* e = em + s * (MAXM + 2);
    e[0] = a + 1;
    for (int i = 0; i <= a; i++) e[1 + i] = otok[r0 + i];
    int h0 = svh[slot * 2], h1 = svh[slot * 2 + 1];
    for (int i = 0; i <= a; i++) { h0 = h1; h1 = vtok[r0 + i]; }
    svh[slot * 2] = h0;
    svh[slot * 2 + 1] = h1;
    svp[slot * 2] += a + 1;
    svp[slot * 2 + 1] = a + 1;
    bacc[s] = r0 + a;
}
// chain inputs: row rows[s] (or s) of R -> Rminb[s]; grid (RW / 512, nseg)
__global__ void k_copy_rows_b(float* __restrict__ dst, const float* __restrict__ src, const int* __restrict__ rows) {
    const int s = blockIdx.y, i = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = rows ? rows[s] : s;
    if (i < RW) dst[(size_t)s * RW + i] = src[(size_t)row * RW + i];
}
// drafts are j-major: bdr[j * MAXM + s]
__global__ void k_draft0_b(const int* __restrict__ motok, const int* __restrict__ bacc, int* __restrict__ bdr, int nseg) {
    if (threadIdx.x < nseg) bdr[threadIdx.x] = motok[bacc[threadIdx.x]];
}
__global__ void k_next_b(int* __restrict__ svt, const int* __restrict__ otok, const int* __restrict__ bacc,
                         const int* __restrict__ bdr, int k, int nseg, const int* __restrict__ segs) {
    const int s = threadIdx.x;
    if (s >= nseg) return;
    int* v = svt + segs[s] * (MAXM + 1);
    v[0] = otok[bacc[s]];
    for (int i = 0; i < k; i++) v[1 + i] = bdr[i * MAXM + s];
}

// lookup drafts in batched steps: per-slot token history hdb[slot][hstride] (prompt + emitted), the step's emitted
// tokens appended, the longest earlier match of each history's suffix (<= 32) -> best[s]; a match of >= mmin tokens
// replaces the segment's MTP drafts with the tokens that followed it (as many as the history has, up to k)
__global__ void k_hist_add_b(int* __restrict__ hdb, int* __restrict__ hnb, size_t hstride, const int* __restrict__ segs, int nseg,
                             const int* __restrict__ em) {
    const int s = threadIdx.x;
    if (s >= nseg) return;
    const int slot = segs[s];
    int* hd = hdb + (size_t)slot * hstride;
    const int* e = em + s * (MAXM + 2);
    const int n = hnb[slot], c = e[0];
    for (int i = 0; i < c && n + i < (int)hstride; i++) hd[n + i] = e[1 + i];
    hnb[slot] = min(n + c, (int)hstride);
}
__global__ void k_lk_scan_b(const int* __restrict__ hdb, const int* __restrict__ hnb, size_t hstride, const int* __restrict__ segs,
                            unsigned long long* __restrict__ best) {
    const int sg = blockIdx.y, slot = segs[sg];
    const int* hd = hdb + (size_t)slot * hstride;
    const int n = hnb[slot];
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (blockIdx.x * blockDim.x >= n) return;
    unsigned long long key = 0ull;
    if (j < n - 1 && hd[j] == hd[n - 1]) {
        int s = 1;
        while (s < 32 && j - s >= 0 && hd[j - s] == hd[n - 1 - s]) s++;
        key = ((unsigned long long)s << 32) | (unsigned)j;
    }
    for (int o = 16; o > 0; o >>= 1) { const unsigned long long t = __shfl_xor_sync(0xffffffffu, key, o); key = t > key ? t : key; }
    __shared__ unsigned long long wmax[8];
    if ((threadIdx.x & 31) == 0) wmax[threadIdx.x >> 5] = key;
    __syncthreads();
    if (threadIdx.x == 0) {
        unsigned long long b = 0ull;
        for (int w = 0; w < (int)(blockDim.x >> 5); w++) b = wmax[w] > b ? wmax[w] : b;
        if (b) atomicMax(best + sg, b);
    }
}
__global__ void k_lk_pick_b(const int* __restrict__ hdb, const int* __restrict__ hnb, size_t hstride, const int* __restrict__ segs,
                            int nseg, int k, int mmin, unsigned long long* __restrict__ best, int* __restrict__ svt,
                            int* __restrict__ cnt, const int* __restrict__ nolk) {
    const int s = threadIdx.x;
    if (s >= nseg) return;
    const unsigned long long b = best[s];
    best[s] = 0ull;
    const int len = (int)(b >> 32), j = (int)(b & 0xffffffffu), slot = segs[s];
    const int n = hnb[slot];
    if (len < mmin || (nolk && nolk[slot])) return;  // nolk: the slot's lookup drafts are off (loop guard)
    const int* hd = hdb + (size_t)slot * hstride;
    const int L = min(k, n - 1 - j);
    int* v = svt + slot * (MAXM + 1);
    for (int i = 0; i < L; i++) v[1 + i] = hd[j + 1 + i];
    if (cnt) atomicAdd(cnt, 1);
}

// ---------------------------------------------------------------- Worker: batched steps
float* const* Worker::tab(int kind, int li) const { return tab_ + ((size_t)kind * (LAYERS + 1) + li) * slots_.size(); }

void Worker::sv_stash() {  // the bound slot's speculative state -> its stash row
    if (!sv_live_) return;
    const int s = cur_slot_;
    CUDA_CHECK(cudaMemcpyAsync(svt_ + (size_t)s * (MAXM + 1), vtok_, MAXM * 4, cudaMemcpyDeviceToDevice, st_));
    CUDA_CHECK(cudaMemcpyAsync(svp_ + 2 * s, posd_, 8, cudaMemcpyDeviceToDevice, st_));
    CUDA_CHECK(cudaMemcpyAsync(svh_ + 2 * s, hist2_, 8, cudaMemcpyDeviceToDevice, st_));
    svpos_[s] = spos_;
    sv_live_ = false;
}

int Worker::next_drafts(int slot, int k, int* out) {
    CUDA_CHECK(cudaSetDevice(dev_));
    CUDA_CHECK(cudaStreamSynchronize(st_));
    const int* src = slot == cur_slot_ && sv_live_ ? vtok_ + 1 : svt_ + (size_t)slot * (MAXM + 1) + 1;
    k = std::min(k, MAXM);
    CUDA_CHECK(cudaMemcpy(out, src, (size_t)k * 4, cudaMemcpyDeviceToHost));
    return k;
}

void Worker::sv_load() {  // the bound slot's stash row -> the single-sequence speculative state
    CUDA_CHECK(cudaSetDevice(dev_));
    ws_ensure();
    if (sv_live_) return;
    const int s = cur_slot_;
    CUDA_CHECK(cudaMemcpyAsync(vtok_, svt_ + (size_t)s * (MAXM + 1), MAXM * 4, cudaMemcpyDeviceToDevice, st_));
    CUDA_CHECK(cudaMemcpyAsync(posd_, svp_ + 2 * s, 8, cudaMemcpyDeviceToDevice, st_));
    CUDA_CHECK(cudaMemcpyAsync(hist2_, svh_ + 2 * s, 8, cudaMemcpyDeviceToDevice, st_));
    spos_ = svpos_[s];
    sv_live_ = true;
}

void Worker::ple_b(Layer& L, int li, int m, int rps) {
    flush_y(m);
    float* key = gates_;
    float* val = proj_;
    xh_pack(pler_, H, H, m, pleh_, H, st_);
    gemv_w4h(L.ple_key, pleh_, H, key, RW, m, st_);
    gemv_w4h(L.ple_val, pleh_, H, val, H, m, st_);
    float* gv = proj2_;
    float* gvn = yd_;
    k_ple_gate<<<dim3(HC, m), 256, 0, st_>>>(key, val, R_, L.ple_nk, L.ple_nq, L.ple_nc, gv, gvn);
    k_ple_conv_b<<<dim3(RW / 256, m / rps), 256, 0, st_>>>(gv, gvn, L.ple_conv, tab(TB_PH, li), tab(TB_PP, li), rslot_, bpn_, rps, R_);
    KCHK("k_ple_conv_b");
}

void Worker::gdn_b(Layer& L, int li, int m, int rps) {
    const int vl = p_.v1 - p_.v0, kl = p_.k1 - p_.k0;
    const int ch = 2 * kl * DK + vl * DV, ld = ch + vl * DV;
    gemv_w4h(L.dn_in, xh_, H, proj_, ld, m, st_);
    gemv_w16(L.dn_ab, x_, H, proj2_, 2 * vl, m, st_);
    float* cv = qbuf_;
    k_dn_conv_b<<<dim3((ch + 255) / 256, m / rps), 256, 0, st_>>>(proj_, ld, ch, L.conv_w, tab(TB_CH, li), tab(TB_CP, li), rslot_, bpn_,
                                                                 rps, cv);
    float* o = gates_;
    static const int qkp = getenv("FLASH_DN_QK") ? atoi(getenv("FLASH_DN_QK")) : 9;  // precomputed q / k norms from m rows (0: off)
    if (qkp > 0 && m >= qkp) {
        k_dn_qk<<<dim3(m, kl + 1), 128, 0, st_>>>(cv, ch, kl, vl, p_.k0, p_.v0, proj2_, L.A, L.dtb, dnqk_, dngb_);
        k_dn_rec_bq<<<dim3(vl, 16, m / rps), 128, 0, st_>>>(cv, ch, kl, vl, p_.k0, p_.v0, dnqk_, dngb_, tab(TB_S, li), tab(TB_RP, li),
                                                            rslot_, bpn_, stepcs_, o, rps);
    } else
    k_dn_rec_b<<<dim3(vl, 16, m / rps), 128, 0, st_>>>(cv, ch, kl, vl, p_.k0, p_.v0, proj2_, L.A, L.dtb, tab(TB_S, li), tab(TB_RP, li),
                                                       rslot_, bpn_, stepcs_, o, rps);
    k_dn_out<<<dim3(vl, m), 128, 0, st_>>>(o, proj_ + ch, ld, L.dn_norm, vl, ah_, vl * DV);
    gemv_w4h(L.dn_out, ah_, vl * DV, part_, H, m, st_);
    KCHK("gdn_b");
}

// decode-path QSA (GQA-tile attention with block selection) over rows of several slots
void Worker::qsa_b(Layer& L, int li, int m, const int* rp, const int* rb, const int* rs, int rps) {
    const int ql = p_.q1 - p_.q0;
    const int ld = L.qkv.n;
    const KvBase& kb = kvb_.at(kvi_.at(li));
    gemv_w8_rows(L.qkv, x_, H, proj_, ld, m, st_);
    gemv_w16_rows(L.idx, x_, H, proj2_, (IDX_NH + 1) * IDX_HD, m, st_);
    const int nqp = ql, qh0 = p_.q0;
    k_qsa_prep<<<dim3(nqp + 4, m), 256, 0, st_>>>(proj_, ld, L.q_norm, L.k_norm, posd_, qbuf_, kb.kc, kb.vc, kb.ks, kb.vs, p_, nqp,
                                                   qh0, rp, rb);
    k_idx_prep_b<<<m / rps, IDX_HD, 0, st_>>>(proj2_, (IDX_NH + 1) * IDX_HD, L.iq_norm, L.ik_norm, rp, rb, rs, gates_, tab(TB_IT, li),
                                              kb.ikb, p_, rps);
    KCHK("qsa_b prep");
    const int lbmax = rows_ / IDX_R;
    Xchg xq;
    for (int g = 0; g < p_.ngpu; g++) { xq.recv[g] = xqpr_[g]; xq.flags[g] = xqpf_[g]; }
    xq.seq = xqseq_;
    xq.done = xqdone_;
    xq.scnt = xqscnt_;
    k_qxsend<<<dim3(p_.ngpu - 1, std::min(16, (m + 3) / 4)), 256, 0, st_>>>(qbuf_, m, xq, p_);
    Xchg xi;
    for (int g = 0; g < p_.ngpu; g++) { xi.recv[g] = xipr_[g]; xi.flags[g] = xipf_[g]; }
    xi.seq = xiseq_;
    xi.done = xidone_;
    k_idx_score<<<dim3((lbmax + 255) / 256, m), 256, 0, st_>>>(kb.ikb, gates_, posd_, p_, lbmax, iscore_, rp, rb);
    if (idx_gk_) k_idx_local<true><<<m, 1024, 0, st_>>>(iscore_, posd_, p_, lbmax, iloc_, rp);
    else k_idx_local<<<m, 1024, (size_t)lbmax * 4, st_>>>(iscore_, posd_, p_, lbmax, iloc_, rp);
    k_idx_send<<<p_.ngpu - 1, 256, 0, st_>>>(iloc_, m, xi, p_);
    k_idx_select<<<m, 1024, 0, st_>>>(iscore_, iloc_, posd_, p_, lbmax, xi, isel_, insel_, rp, idx_gk_);
    KCHK("qsa_b select");
    AttnDst d{};
    d.o[0] = attn_x_;
    for (int z = 1; z < dec_split_; z++) d.o[z] = attn_spl_ + (size_t)(z - 1) * XROWS * NQ * (HD + 2);
    k_qsa_dec<<<dim3(NKV, m, dec_split_), 128, QD_SMEM, st_>>>(qbuf_, kb.kc, kb.vc, kb.ks, kb.vs, posd_, d, p_, xq, isel_, insel_, lbmax,
                                                               rp, rb);
    if (dec_split_ > 1) k_attn_merge<<<dim3(NQ, m), HD, 0, st_>>>(d, dec_split_);
    KCHK("qsa_b attn");
    Xchg xc;
    for (int g = 0; g < p_.ngpu; g++) { xc.recv[g] = xpr_[g]; xc.flags[g] = xpf_[g]; }
    xc.seq = xseq_;
    xc.done = xdone_;
    xc.scnt = xscnt_;
    k_xsend<<<dim3(p_.ngpu - 1, std::min(16, (m + 3) / 4)), 256, 0, st_>>>(attn_x_, m, xc, p_);
    k_qsa_comb2<<<dim3(ql, m), HD, 0, st_>>>(attn_x_, m, proj_, ld, ab_, ql * HD, xc, p_, nqp * HD);
    KCHK("qsa_b comb");
    gemv_w8_rows(L.o, ab_, ql * HD, part_, H, m, st_);
}

// debug FLASH_BDUMP=prefix (FLASH_GRAPH=0): the residual of segment 0's rows after every mixer / MoE of a batched step
// -> prefix.r<rank>.s<step> ([48][2][rps][10240] floats)
void Worker::moe_dn_1w() {  // debug FLASH_MOE_DN_1W (see k_moe_dn)
    if (!getenv("FLASH_MOE_DN_1W") || !atoi(getenv("FLASH_MOE_DN_1W"))) return;
    CUDA_CHECK(cudaSetDevice(dev_));
    const int one = 1;
    CUDA_CHECK(cudaMemcpyToSymbol(g_moe_dn_1w, &one, 4));
}

void Worker::bdump(int li, int part, int m, int rps) {
    static const char* pre = getenv("FLASH_BDUMP");
    if (!pre) return;
    flush_y(m);
    CUDA_CHECK(cudaStreamSynchronize(st_));
    thread_local std::vector<float> buf;  // one worker thread per card
    thread_local int step = 0;
    buf.resize((size_t)LAYERS * 2 * rps * RW);
    CUDA_CHECK(cudaMemcpy(buf.data() + ((size_t)li * 2 + part) * rps * RW, R_, (size_t)rps * RW * 4, cudaMemcpyDeviceToHost));
    if (li == LAYERS - 1 && part == 1) {
        char fn[512];
        snprintf(fn, sizeof fn, "%s.r%d.s%d", pre, p_.rank, step++);
        if (FILE* f = fopen(fn, "wb")) { fwrite(buf.data(), 4, buf.size(), f); fclose(f); }
    }
}

void Worker::trunk_b(int m, const int* toks, int nseg, int rps) {
    ll_idx_ = 0;
    curpos_ = posd_;
    k_step_bump_b<<<1, 32, 0, st_>>>(stepcs_, bseg_d_, nseg);
    k_embed<<<m, 256, 0, st_>>>(embed_, toks, R_);
    for (int li = 0; li < LAYERS; li++) {
        Layer& L = L_[li];
        if (li == PLE_LAYER) ple_b(L, li, m, rps);
        hc_read(L.hca, m, true);
        if (L.full) qsa_b(L, li, m, rpos_, rbase_, rslot_, rps);
        else gdn_b(L, li, m, rps);
        hc_write(m);
        bdump(li, 0, m, rps);
        hc_read(L.hcm, m, true);
        moe(L, m);
        hc_write(m);
        steer_mark(li, rslot_);  // the layer output; per row its slot's vector
        bdump(li, 1, m, rps);
    }
    flush_y(m);
    CUDA_CHECK(cudaMemcpyAsync(Rsave_, R_, (size_t)m * RW * 4, cudaMemcpyDeviceToDevice, st_));
    hc_read(mixer_, m, false);
    if (bsamp_) head_sample_b(m, otok_);
    else head(m, otok_);
}

// MTP block over m rows (verify rows: chain = false, rows of the trunk; chain: one row per segment from Rminb_)
void Worker::mtp_core_b(int m, bool chain, const int* toks, int* out, const int* rp, const int* rb, const int* rs, int rps) {
    const float* hsrc = chain ? Rminb_ : Rsave_;
    k_mtp_prep<<<dim3(m, HC + 1), 256, 0, st_>>>(embed_, toks, hsrc, mtp_pre_e_, mtp_pre_h_, mtp_en_, mtp_hn_);
    gemv_w8_rows(mtp_fce_, mtp_en_, H, mtp_fe_, H, m, st_);
    for (int r0 = 0; r0 < m * HC; r0 += kMaxBatchRows)
        gemv_w8(mtp_fch_, mtp_hn_ + (size_t)r0 * H, H, mtp_fh_ + (size_t)r0 * H, H, std::min(kMaxBatchRows, m * HC - r0), st_);
    k_mtp_comb<<<m, 512, 0, st_>>>(mtp_fe_, mtp_fh_, Rm_);
    float* Rt = R_;
    float* Rt2 = R2_;
    R_ = Rm_;
    R2_ = Rm2_;
    hc_read(mtp_.hca, m, true);
    qsa_b(mtp_, LAYERS, m, rp, rb, rs, rps);
    hc_write(m);
    hc_read(mtp_.hcm, m, true);
    moe(mtp_, m);
    hc_write(m);
    flush_y(m);
    k_copy_rows_b<<<dim3(RW / 512, chain ? m : m / rps), 512, 0, st_>>>(Rminb_, R_, chain ? nullptr : bacc_);
    hc_read(mtp_mixer_, m, false);
    R_ = Rt;
    R2_ = Rt2;
    head(m, out, true);
    KCHK("mtp_core_b");
}

void Worker::batch_body(int nseg, int k) {
    const int rps = k + 1, m = nseg * rps;
    CUDA_CHECK(cudaMemcpyAsync(bseg_d_, bseg_h_, (size_t)2 * MAXM * 4, cudaMemcpyHostToDevice, st_));
    CUDA_CHECK(cudaMemcpyAsync(kvrb_d_, kvrb_h_, slots_.size() * 4, cudaMemcpyHostToDevice, st_));
    comm_begin_unit(ctr_, st_);
    k_bmeta<<<1, 32, 0, st_>>>(bseg_d_, nseg, rps, svp_, svt_, kvrb_d_, rpos_, rbase_, rslot_, bvt_, bpn_);
    if (ngp_.flags) {  // staged rows (table not pinned): ids -> host -> rows
        NgramDev g = ngp_;
        g.phase = 1;
        k_ngram_b<<<m, 256, 0, st_>>>(bvt_, rslot_, rps, svh_, g, pler_);
        k_ngram_handoff<<<1, 32, 0, st_>>>(g, m * PLE_HEADS);
        g.phase = 2;
        k_ngram_b<<<m, 256, 0, st_>>>(bvt_, rslot_, rps, svh_, g, pler_);
    } else
    k_ngram_b<<<m, 256, 0, st_>>>(bvt_, rslot_, rps, svh_, ngp_, pler_);
    KCHK("batch meta");
    trunk_b(m, bvt_, nseg, rps);
    k_accept_b<<<1, 32, 0, st_>>>(bvt_, otok_, k, nseg, bseg_d_, bseg_d_ + MAXM, svp_, svh_, bacc_, bem_);
    if (bsamp_) pen_commit_b(spslot_, bseg_d_, nseg, bem_, MAXM + 2, svp_, st_);
    if (has_mtp_) {
        mtp_core_b(m, false, otok_, motok_, rpos_, rbase_, rslot_, rps);
        k_draft0_b<<<1, 32, 0, st_>>>(motok_, bacc_, bdr_, nseg);
        for (int j = 1; j < k; j++) {
            k_bmeta_chain<<<1, 32, 0, st_>>>(bseg_d_, nseg, svp_, kvrb_d_, j, chpos_, chbase_);
            mtp_core_b(nseg, true, bdr_ + (size_t)(j - 1) * MAXM, bdr_ + (size_t)j * MAXM, chpos_, chbase_, bseg_d_, 1);
        }
    }
    k_next_b<<<1, 32, 0, st_>>>(svt_, otok_, bacc_, bdr_, has_mtp_ ? k : 0, nseg, bseg_d_);
    if (blk_ && k > 0) {  // lookup drafts replace the MTP drafts of segments whose history repeats
        const size_t hs = (size_t)max_ctx_ + 64;
        k_hist_add_b<<<1, 32, 0, st_>>>(hdb_, hnb_, hs, bseg_d_, nseg, bem_);
        k_lk_scan_b<<<dim3((unsigned)((hs + 255) / 256), nseg), 256, 0, st_>>>(hdb_, hnb_, hs, bseg_d_, lkbb_);
        k_lk_pick_b<<<1, 32, 0, st_>>>(hdb_, hnb_, hs, bseg_d_, nseg, k, blk_min_, lkbb_, svt_, blkcnt_, nolk_d_);
        KCHK("batch lookup");
    }
    KCHK("batch step");
}

void Worker::set_bhist(int slot, const int* ids, int n) {
    if (!blk_) return;
    CUDA_CHECK(cudaSetDevice(dev_));
    const size_t hs = (size_t)max_ctx_ + 64;
    n = std::min<int>(n, (int)hs);
    CUDA_CHECK(cudaMemcpyAsync(hdb_ + (size_t)slot * hs, ids, (size_t)n * 4, cudaMemcpyHostToDevice, st_));
    CUDA_CHECK(cudaMemcpyAsync(hnb_ + slot, &n, 4, cudaMemcpyHostToDevice, st_));
    CUDA_CHECK(cudaStreamSynchronize(st_));
}

void Worker::batch_launch(const int* slots, const int* bases, const int* lims, int nseg, int k) {
    CUDA_CHECK(cudaSetDevice(dev_));
    CHECK(nseg >= 1 && nseg * (k + 1) <= MAXM && !qsa_a_ && dec_k_ && dec_split_ <= 8, "flash batch: %d x %d rows", nseg, k + 1);
    sv_stash();
    ws_save();  // the bound slot's recurrent state back to its storage (the batch reads every slot's storage)
    for (int s = 0; s < (int)slots_.size(); s++) kvrb_h_[s] = bases[s] * p_.own;
    for (int i = 0; i < nseg; i++) { bseg_h_[i] = slots[i]; bseg_h_[MAXM + i] = lims ? lims[i] : k; }
    ec_tick(0, 0);
    bnseg_ = nseg;
    brps_ = k + 1;
    bslots_.assign(slots, slots + nseg);
    const bool con0 = con_;  // grammar masks belong to the bound slot's single steps (constrained slots never batch)
    con_ = false;
    // per-slot sampling: the bound slot's live parameters (counter) back to its row first, reloaded after the step
    bsamp_ = false;
    for (int i = 0; i < nseg; i++) bsamp_ |= sson_.at(slots[i]) != 0;
    if (bsamp_) CUDA_CHECK(cudaMemcpyAsync(spslot_ + cur_slot_, sp_, sizeof(SampleParams), cudaMemcpyDeviceToDevice, st_));
    const long long key = ((long long)nseg << 16) | ((long long)k << 1) | (bsamp_ ? 1 : 0);
    if (!ev0_) { CUDA_CHECK(cudaEventCreate(&ev0_)); CUDA_CHECK(cudaEventCreate(&ev1_)); }
    CUDA_CHECK(cudaEventRecord(ev0_, st_));
    // one graph per shape (not LRU-capped like the prefill caches: few shapes); eager the first time (lazy library
    // allocations); an instantiate OOM drops the prefill graphs and retries, else the step runs eagerly
    static const bool graphs = !getenv("FLASH_GRAPH") || atoi(getenv("FLASH_GRAPH"));
    auto it = bgraphs_.find(key);
    if (!graphs || dump_ || getenv("FLASH_BDUMP") || it == bgraphs_.end()) {
        if (graphs) bgraphs_[key] = nullptr;
        batch_body(nseg, k);
    } else {
        if (!it->second) {
            cudaGraph_t g;
            CUDA_CHECK(cudaStreamBeginCapture(st_, cudaStreamCaptureModeThreadLocal));
            batch_body(nseg, k);
            CUDA_CHECK(cudaStreamEndCapture(st_, &g));
            cudaError_t e = cudaGraphInstantiate(&it->second, g, 0);
            if (e == cudaErrorMemoryAllocation) {
                (void)cudaGetLastError();
                CUDA_CHECK(cudaStreamSynchronize(st_));
                for (auto* c : {&pgraphs_, &pbgraphs_}) {
                    for (auto& kv : *c)
                        if (kv.second) CUDA_CHECK(cudaGraphExecDestroy(kv.second));
                    c->clear();
                }
                e = cudaGraphInstantiate(&it->second, g, 0);
            }
            CUDA_CHECK(cudaGraphDestroy(g));
            if (e == cudaErrorMemoryAllocation) {
                (void)cudaGetLastError();
                it->second = nullptr;
                batch_body(nseg, k);
            } else {
                CUDA_CHECK(e);
            }
        }
        if (it->second) CUDA_CHECK(cudaGraphLaunch(it->second, st_));
    }
    if (bsamp_) CUDA_CHECK(cudaMemcpyAsync(sp_, spslot_ + cur_slot_, sizeof(SampleParams), cudaMemcpyDeviceToDevice, st_));
    con_ = con0;
    CUDA_CHECK(cudaEventRecord(ev1_, st_));
}

void Worker::batch_finish(int* out, int* cnt) {
    CUDA_CHECK(cudaSetDevice(dev_));
    CUDA_CHECK(cudaStreamSynchronize(st_));
    CUDA_CHECK(cudaGetLastError());
    {
        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, ev0_, ev1_));
        bgpu_ms_.push_back(ms);
    }
    for (int i = 0; i < bnseg_; i++) {
        const int* e = bem_h_ + i * (MAXM + 2);
        cnt[i] = e[0];
        for (int j = 0; j < e[0]; j++) out[i * 64 + j] = e[1 + j];
        svpos_[bslots_[i]] += e[0];
        set_slot_prev_m(bslots_[i], e[0]);
    }
    nf_check("batch step", 0);
    static const bool js = getenv("FLASH_JOBSTAT") && atoi(getenv("FLASH_JOBSTAT"));
    if (js && bgpu_ms_.size() % 50 == 0) {  // mean expert-group jobs per MoE call on this card, by row count
        unsigned long long st[MAXM + 1][2];
        CUDA_CHECK(cudaMemcpyFromSymbol(st, g_jobstat, sizeof st));
        std::string o;
        for (int i = 0; i <= MAXM; i++)
            if (st[i][1]) o += " m" + std::to_string(i) + ":" + std::to_string(st[i][0] / st[i][1]) + "x" + std::to_string(st[i][1]);
        fprintf(stderr, "[jobstat] card %d:%s\n", p_.rank, o.c_str());
    }
}

}  // namespace flash
