"""GLU epilogue quantization (gemm_i8v_kernel, QE_I8_GLUQ): gate_up (GLU, var = the engine default) followed by
down on its activations must give bit-identical down outputs whether down takes the int8 rows + records from the
GLU epilogue (first call after gate_up) or quantizes them itself (pre-pass, second call); also times the pair.
Usage: CUDA_VISIBLE_DEVICES=1 python3 tests/test_gluq.py [--ts 77,256,1024,2048] [--var 100] [--real]"""
import argparse

import torch

import qe
import test_gemv as G
import test_gemm_i8 as I8


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ts", default="77,256,1000,1024,2048")
    ap.add_argument("--var", type=int, default=100)
    ap.add_argument("--real", action="store_true")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--ab", default="", help="also time gate_up+down for these variants interleaved (e.g. 7,100)")
    ap.add_argument("--reps", type=int, default=7)
    a = ap.parse_args()
    torch.cuda.set_device(0)
    nu, ku = I8.SHAPES["gate_up"]
    nd, kd = I8.SHAPES["down"]
    nu, ku, wu, dqu, dsu = I8.make_w(nu, ku, I8.REAL.get("gate_up") if a.real else None)
    nd, kd, wd, dqd, dsd = I8.make_w(nd, kd, I8.REAL.get("down") if a.real else None)
    assert nu // 2 == kd, (nu, kd)
    ok = True
    for t in [int(v) for v in a.ts.split(",")]:
        x = I8.make_x(t, ku)
        act = torch.zeros(t, kd, dtype=torch.bfloat16, device="cuda")
        y1 = torch.zeros(t, nd, device="cuda")
        y2 = torch.zeros(t, nd, device="cuda")
        qe.call("qe_gemm_w4_i8_glu", dqu, dsu, nu, ku, x, act, t, 64, 1, a.var)
        qe.call("qe_gemm_w4_i8", dqd, dsd, nd, kd, act, y1, t, 64, 1, a.var)  # consumes the GLU quantization
        qe.call("qe_gemm_w4_i8", dqd, dsd, nd, kd, act, y2, t, 64, 1, a.var)  # own pre-pass
        qe.sync()
        same = (y1.view(torch.int32) == y2.view(torch.int32)).float().mean().item()
        emul = I8.quant_emul(act, 64) @ wd.T
        em = I8.relrms(y1, emul)

        def pair():
            qe.call("qe_gemm_w4_i8_glu", dqu, dsu, nu, ku, x, act, t, 64, 1, a.var)
            qe.call("qe_gemm_w4_i8", dqd, dsd, nd, kd, act, y1, t, 64, 1, a.var)
        us = G.bench(pair, a.iters)
        print(f"t={t:5d}: down(GLU-quantized) == down(pre-pass) {100 * same:.4f}%  err vs emul {em:.1e}  "
              f"gate_up+down {us:7.1f} us", flush=True)
        ok &= same == 1.0 and em < 1e-5
        if a.ab:
            vs = [int(v) for v in a.ab.split(",")]
            fns = {v: (lambda v=v: (qe.call("qe_gemm_w4_i8_glu", dqu, dsu, nu, ku, x, act, t, 64, 1, v),
                                    qe.call("qe_gemm_w4_i8", dqd, dsd, nd, kd, act, y1, t, 64, 1, v))) for v in vs}
            tm = {v: [] for v in vs}
            for _ in range(a.reps):
                for v in vs:
                    tm[v].append(G.bench(fns[v], a.iters))
            print("    A/B gate_up+down: " + "  ".join(f"v{v} {sorted(tm[v])[len(tm[v]) // 2]:7.1f} us" for v in vs), flush=True)
    print("OK" if ok else "FAIL")
    assert ok


if __name__ == "__main__":
    main()
