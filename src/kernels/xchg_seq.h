#pragma once
// The 27B KV-split exchange flag protocol (ops.cuh xchg_seq; tests/xchg_seq_unit.cpp): the sender raises its flag to
// seq = epoch * 4096 + xidx + 1 on every exchange, the receiver waits while (int)(flag - seq) < 0. The compare is
// right while the receiver's flag lags its expected value by less than 2^31. With one flag set per epoch counter
// (review round 4 S1) the lag is at most one exchange, at any context and MTP depth: the decode host set and the
// prefill device set (QE_DECODE_COMM host) each count the launch units that use them (xchg_set).
#ifdef __CUDACC__
#define QE_HD __host__ __device__ __forceinline__
#else
#define QE_HD inline
#endif
QE_HD unsigned xchg_seq_of(unsigned epoch, int xidx) { return epoch * 4096u + (unsigned)xidx + 1u; }
QE_HD bool xchg_arrived(unsigned flag, unsigned expected) { return (int)(flag - expected) >= 0; }
// the flag set (and its epoch counter) of a launch unit: 1 = the decode host set (both decode modes on host memory, a
// decode unit), 0 = the device set. With per-set counters (Worker::xep_) every unit bumps the counter of its set only,
// so each set's flags trail its expected value by at most one exchange, also after a set was idle for any time.
QE_HD int xchg_set(bool xh, bool in_pf) { return xh && !in_pf ? 1 : 0; }
#undef QE_HD
