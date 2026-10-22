"""bf16 output mode of the prefill GEMM (gemm_set_out_bf16): for every non-GLU shape and t (stream-K split
tiles included) the bf16 result must equal the fp32 result rounded to bf16 (RN) bit for bit, on the
gemm_i8v_kernel epilogue (engine variant 100) and on the fallback (old kernels: fp32 scratch + conversion).
Also times fp32 vs bf16 output. Usage: CUDA_VISIBLE_DEVICES=1 python3 tests/test_out16.py [--ts 77,256,1024]"""
import argparse

import torch

import qe
import test_gemv as G
import test_gemm_i8 as I8


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ts", default="77,256,1000,1024,2048")
    ap.add_argument("--shapes", default="down,o/out,o_split,qkv")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--reps", type=int, default=5)
    a = ap.parse_args()
    torch.cuda.set_device(0)
    ok = True
    for name in a.shapes.split(","):
        n, k = I8.SHAPES[name]
        n, k, wref, dq, ds = I8.make_w(n, k)
        for t in [int(v) for v in a.ts.split(",")]:
            x = I8.make_x(t, k)
            y32 = torch.zeros(t, n, device="cuda")
            y16 = torch.zeros(t, n, dtype=torch.bfloat16, device="cuda")
            line = f"{name:8s} t={t:5d}:"
            for var in (100,):  # engine kernel (the old kernels have no bf16 epilogue: fallback only through gemm_w4)
                qe.call("qe_gemm_w4_i8", dq, ds, n, k, x, y32, t, 64, 1, var)
                qe.call("qe_gemm_set_out_bf16", 1)
                qe.call("qe_gemm_w4_i8", dq, ds, n, k, x, y16, t, 64, 1, var)
                qe.call("qe_gemm_set_out_bf16", 0)
                qe.sync()
                same = (y32.bfloat16().view(torch.int16) == y16.view(torch.int16)).float().mean().item()
                line += f" v{var} {100 * same:.4f}%" if var == 100 else ""
                if var == 100:
                    ok &= same == 1.0
            # gemm_w4 (tuned engine entry): bf16 == rn(fp32)
            qe.call("qe_gemm_w4_t", dq, ds, n, k, x, y32, t)
            qe.call("qe_gemm_set_out_bf16", 1)
            qe.call("qe_gemm_w4_t", dq, ds, n, k, x, y16, t)
            qe.call("qe_gemm_set_out_bf16", 0)
            qe.sync()
            same = (y32.bfloat16().view(torch.int16) == y16.view(torch.int16)).float().mean().item()
            line += f" | gemm_w4 {100 * same:.4f}%"
            ok &= same == 1.0
            f32 = lambda: qe.call("qe_gemm_w4_i8", dq, ds, n, k, x, y32, t, 64, 1, 100)
            def f16():
                qe.call("qe_gemm_set_out_bf16", 1)
                qe.call("qe_gemm_w4_i8", dq, ds, n, k, x, y16, t, 64, 1, 100)
                qe.call("qe_gemm_set_out_bf16", 0)
            t32, t16 = [], []
            for _ in range(a.reps):
                t32.append(G.bench(f32, a.iters))
                t16.append(G.bench(f16, a.iters))
            m32, m16 = sorted(t32)[a.reps // 2], sorted(t16)[a.reps // 2]
            line += f" | fp32 out {m32:7.1f} us, bf16 out {m16:7.1f} us ({100 * (m16 / m32 - 1):+.1f} %)"
            print(line, flush=True)
    print("OK" if ok else "FAIL")
    assert ok


if __name__ == "__main__":
    main()
