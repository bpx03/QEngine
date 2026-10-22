"""Self-distillation of the MTP head on the engine's own answers (data from tools/mtp_data.py).

The MTP layer mirrors the engine (src/model.cpp mtp_forward): input = [norm_e(embed(x_{i+1})),
norm_h(h_i)] -> fc -> one gated-attention decoder layer -> norm; h_i is the main model's final normed
hidden state (what the engine feeds it), drafts are read through the shared lm_head. Norm weights
are zero-centred (x * (1 + w)). Only the MTP weights train; embeddings and lm_head are frozen.

  python tools/mtp_train.py eval  <data_dir> [mtp.safetensors]
  python tools/mtp_train.py train <data_dir> <out.safetensors> [--steps N] [--lr X] [--two-step] [--init head]
"""
import argparse, json, math, os, random, time
import torch.utils.checkpoint
import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
MODEL = os.path.join(ROOT, "models", "Qwen3.8-27B-INT4")
H, NQ, NKV, HD, ROT, EPS, THETA = 5120, 24, 4, 256, 64, 1e-6, 1e7
STEPS = int(os.environ.get("MTP_STEPS", "4"))  # draft steps trained (1 + recursive ones)


def rms1p(x, w):  # zero-centred RMSNorm in fp32
    xf = x.float()
    return xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + EPS) * (1.0 + w.float())


class MTP(torch.nn.Module):
    def __init__(self, sd):
        super().__init__()
        P = lambda n: torch.nn.Parameter(sd[n].float())
        self.fc = P("mtp.fc.weight")
        self.pre_e, self.pre_h = P("mtp.pre_fc_norm_embedding.weight"), P("mtp.pre_fc_norm_hidden.weight")
        a, l = "mtp.layers.0.self_attn.", "mtp.layers.0."
        self.q, self.k, self.v, self.o = P(a + "q_proj.weight"), P(a + "k_proj.weight"), P(a + "v_proj.weight"), P(a + "o_proj.weight")
        self.qn, self.kn = P(a + "q_norm.weight"), P(a + "k_norm.weight")
        self.in_norm, self.post_norm = P(l + "input_layernorm.weight"), P(l + "post_attention_layernorm.weight")
        self.gate, self.up, self.down = P(l + "mlp.gate_proj.weight"), P(l + "mlp.up_proj.weight"), P(l + "mlp.down_proj.weight")
        self.norm = P("mtp.norm.weight")
        inv = 1.0 / (THETA ** (torch.arange(0, ROT, 2, dtype=torch.float64) / ROT))
        self.register_buffer("inv_freq", inv.float(), persistent=False)

    qat = False  # fake-quantize the matrices like the engine (INT4 RTN, groups of 128, bf16 scale)

    def w(self, p):
        wb = p.to(torch.bfloat16)
        if not self.qat:
            return wb
        n, k = p.shape
        g = p.detach().float().view(n, k // 128, 128)
        sc = (g.abs().amax(-1, keepdim=True) / 7.0).to(torch.bfloat16).float()
        q = torch.clamp(torch.round(g / torch.where(sc > 0, sc, torch.ones_like(sc))), -8, 7) * sc
        fq = q.view(n, k).to(torch.bfloat16)
        return wb + (fq - wb).detach()     # straight-through estimator

    def state(self):
        n = {"fc": "mtp.fc.weight", "pre_e": "mtp.pre_fc_norm_embedding.weight", "pre_h": "mtp.pre_fc_norm_hidden.weight",
             "q": "mtp.layers.0.self_attn.q_proj.weight", "k": "mtp.layers.0.self_attn.k_proj.weight",
             "v": "mtp.layers.0.self_attn.v_proj.weight", "o": "mtp.layers.0.self_attn.o_proj.weight",
             "qn": "mtp.layers.0.self_attn.q_norm.weight", "kn": "mtp.layers.0.self_attn.k_norm.weight",
             "in_norm": "mtp.layers.0.input_layernorm.weight", "post_norm": "mtp.layers.0.post_attention_layernorm.weight",
             "gate": "mtp.layers.0.mlp.gate_proj.weight", "up": "mtp.layers.0.mlp.up_proj.weight",
             "down": "mtp.layers.0.mlp.down_proj.weight", "norm": "mtp.norm.weight"}
        return {v: getattr(self, k).detach().to(torch.bfloat16).contiguous().cpu() for k, v in n.items()}

    def rope(self, x, pos):  # x [T, heads, 256]: rotate_half over the first 64 dims
        ang = pos[:, None].float() * self.inv_freq[None]          # [T, 32]
        c, s = ang.cos()[:, None], ang.sin()[:, None]
        x1, x2 = x[..., :32], x[..., 32:64]
        return torch.cat([x1 * c - x2 * s, x2 * c + x1 * s, x[..., 64:]], -1)

    def qkv(self, x, pos):
        a_in = rms1p(x, self.in_norm).to(torch.bfloat16)
        qg = F.linear(a_in, self.w(self.q)).float().view(-1, NQ, 2 * HD)
        q, g = qg[..., :HD], qg[..., HD:]
        k = F.linear(a_in, self.w(self.k)).float().view(-1, NKV, HD)
        v = F.linear(a_in, self.w(self.v)).float().view(-1, NKV, HD)
        q = self.rope(rms1p(q, self.qn), pos)
        k = self.rope(rms1p(k, self.kn), pos)
        return q, k, v, g

    def after_attn(self, x, att, g):
        att = att * torch.sigmoid(g)
        x = x + F.linear(att.reshape(-1, NQ * HD).to(torch.bfloat16), self.w(self.o)).float()
        m_in = rms1p(x, self.post_norm).to(torch.bfloat16)
        act = F.silu(F.linear(m_in, self.w(self.gate)).float()) * F.linear(m_in, self.w(self.up)).float()
        x = x + F.linear(act.to(torch.bfloat16), self.w(self.down)).float()
        return rms1p(x, self.norm).to(torch.bfloat16)   # the engine's xn_ (bf16) -> lm_head / next draft

    def embed_in(self, e, h):
        cat = torch.cat([rms1p(e, self.pre_e).to(torch.bfloat16), rms1p(h, self.pre_h).to(torch.bfloat16)], -1)
        return F.linear(cat, self.w(self.fc)).float()

    def forward(self, e, h, pos):
        """One draft step at every position (teacher forcing): e [T,H] embeddings of x_{i+1},
        h [T,H] hidden states h_i -> MTP outputs [T,H] (bf16) and its K, V (for a second step)."""
        x = self.embed_in(e, h)
        q, k, v, g = self.qkv(x, pos)
        att = F.scaled_dot_product_attention(q.transpose(0, 1)[None].to(torch.bfloat16),
                                             k.transpose(0, 1)[None].to(torch.bfloat16),
                                             v.transpose(0, 1)[None].to(torch.bfloat16),
                                             is_causal=True, scale=HD ** -0.5, enable_gqa=True)
        out = self.after_attn(x, att[0].transpose(0, 1).float(), g)
        return out, k, v

    def forward_r(self, e, h, pos, Ks, Vs, r, off):
        """Draft step r >= 2 for query rows i (positions pos[i], i >= r - 1 in the step-1 index):
        input (x_{p+1}, the step r-1 output at p-1); attends to the step-1 keys of positions
        <= p - r + 1 and to the key of step j at position p - r + j for j = 2 .. r (the drafts of
        this draft loop), as the engine's draft loop does. Ks[j] / Vs[j]: step j's keys / values
        indexed like step 1 (row i = position pos0 + i); returns (outputs, k, v) for rows >= r - 1."""
        x = self.embed_in(e, h)
        q, k, v, g = self.qkv(x, pos)
        T = q.shape[0]                       # rows r-1 .. of the step-1 index
        rep = NQ // NKV
        sc = HD ** -0.5
        qh = q.transpose(0, 1).to(torch.bfloat16)                         # [NQ, T, HD]
        K1 = Ks[1].transpose(0, 1).to(torch.bfloat16).repeat_interleave(rep, 0)
        V1 = Vs[1].transpose(0, 1).to(torch.bfloat16).repeat_interleave(rep, 0)
        T1 = K1.shape[1]
        s1 = torch.matmul(qh, K1.transpose(1, 2)).float() * sc            # [NQ, T, T1]
        # query t sits at step-1 row b + r - 2 + t; step-1 keys allowed up to row t + off (off = b - 1)
        row = torch.arange(T, device=q.device)[:, None]
        col = torch.arange(T1, device=q.device)[None, :]
        s1 = s1.masked_fill(col > row + off, float("-inf"))               # positions <= p - r + 1
        extra_s, extra_v = [], []
        for jj in range(2, r + 1):
            # step jj key at position p - r + jj: row index (i + r - 1) - r + jj = i + jj - 1,
            # in Ks[jj] (indexed from row jj - 1) -> offset i
            kj = (k if jj == r else Ks[jj][: T]).transpose(0, 1).float().repeat_interleave(rep, 0)
            vj = (v if jj == r else Vs[jj][: T]).transpose(0, 1).float().repeat_interleave(rep, 0)
            extra_s.append((qh.float() * kj).sum(-1, keepdim=True) * sc)
            extra_v.append(vj)
        p = torch.softmax(torch.cat([s1] + extra_s, -1), -1)
        att = torch.matmul(p[..., :T1].to(torch.bfloat16), V1).float()
        for n, vj in enumerate(extra_v):
            att = att + p[..., T1 + n:T1 + n + 1] * vj
        return self.after_attn(x, att.transpose(0, 1), g), k, v


class Data:
    def __init__(self, d, ctx=6144):
        self.idx = json.load(open(os.path.join(d, "index.json")))
        self.tok = np.load(os.path.join(d, "tokens.npy"))
        self.hid = np.memmap(os.path.join(d, "hidden.bin"), dtype=np.uint16, mode="r").reshape(-1, H)
        self.ctx = ctx
        rnd = random.Random(0)
        order = list(range(len(self.idx)))
        rnd.shuffle(order)
        nv = max(8, len(order) // 20)
        self.val, self.train = order[:nv], order[nv:]
        if os.environ.get("MTP_ALL_VAL"):  # evaluate on every sequence (a separate eval set)
            self.val = order

    def get(self, i, dev):
        s = self.idx[i]
        a, n, pl = s["off"], s["n"], s["prompt"]
        b = max(0, n - self.ctx)  # full context (capped at ctx, which only cuts very long prompts)
        tok = torch.from_numpy(self.tok[a + b:a + n].astype(np.int64)).to(dev)
        hid = torch.from_numpy(np.ascontiguousarray(self.hid[a + b:a + n]).view(np.int16)).to(dev).view(torch.bfloat16)
        pos = torch.arange(b, n, device=dev)
        return tok, hid, pos, max(0, pl - b)


def load_frozen(dev):
    with safe_open(os.path.join(MODEL, "model.safetensors"), "pt", device="cpu") as f:
        emb = f.get_tensor("model.language_model.embed_tokens.weight")  # stays in host memory
        head = f.get_tensor("lm_head.weight").to(dev)
    rank = np.fromfile(os.path.join(MODEL, "token_rank.bin"), dtype=np.int32)[:65520]
    draft = torch.from_numpy(np.sort(rank).astype(np.int64)).to(dev)
    if os.environ.get("MTP_Q_HEAD"):  # the engine's draft lm_head is INT4 (RTN) too
        w = head[draft].float().view(len(draft), H // 128, 128)
        sc = (w.abs().amax(-1, keepdim=True) / 7.0).to(torch.bfloat16).float()
        q = torch.clamp(torch.round(w / torch.where(sc > 0, sc, torch.ones_like(sc))), -8, 7) * sc
        head = head.clone()
        head[draft] = q.view(len(draft), H).to(torch.bfloat16)
    return emb, head, draft


def load_mtp(path):
    with safe_open(path, "pt", device="cpu") as f:
        return {k: f.get_tensor(k) for k in f.keys()}


def seq_loss(model, emb, head, draft, tok, hid, pos, pl, two_step, chunk=1024):
    """Returns (loss sum, n, correct1, correct2, n2) over answer targets; draft-vocabulary top-1."""
    T = tok.shape[0]
    e = emb[tok[1:].cpu()].to(tok.device)          # x_{i+1} for i = 0 .. T-2
    if model.training:  # recompute the layer in backward (full-context sequences are long)
        out, k1, v1 = torch.utils.checkpoint.checkpoint(model, e, hid[:-1], pos[:-1], use_reentrant=False)
    else:
        out, k1, v1 = model(e, hid[:-1], pos[:-1])     # predicts x_{i+2}
    tgt = tok[2:]
    m = torch.arange(T - 2, device=tok.device) + 2 >= pl     # targets inside the answer
    out1 = out[:-1]
    def ce(o, t, mk):
        tot = torch.zeros((), device=o.device)
        corr = 0
        for c0 in range(0, o.shape[0], chunk):
            oc, tc, mc = o[c0:c0 + chunk], t[c0:c0 + chunk], mk[c0:c0 + chunk]
            if not mc.any():
                continue
            lg = F.linear(oc[mc], head).float()
            tot = tot + F.cross_entropy(lg, tc[mc], reduction="sum")
            corr += int((draft[lg[:, draft].argmax(-1)] == tc[mc]).sum())
        return tot, corr
    loss, c1 = ce(out1, tgt, m)
    n = int(m.sum())
    c2 = n2 = 0
    if two_step and T > 2 + STEPS:
        # steps r = 2 .. STEPS: rows i >= r - 1 of the step-1 index (position i, input x_{i+1}
        # with the step r-1 output at i-1, target x_{i+2}); the reported step-2 accuracy is the
        # mean over these steps
        # recursive steps only over the answer window (plus STEPS rows of lead-in), attending to
        # the step-1 keys of the whole sequence: step r computes rows b + r - 2 .. T - 2
        b = max(STEPS, pl - 2 - STEPS)
        Ks, Vs, prev = {1: k1}, {1: v1}, out[b - 1:]
        for r in range(2, STEPS + 1):
            r0 = b + r - 2
            if r0 >= T - 2:
                break
            e_r = e[r0:]                          # x_{i+1}, rows r0 .. T-2
            h_r = prev[:-1]                       # step r-1 output at rows r0-1 .. T-3
            if model.training:
                o_r, k_r, v_r = torch.utils.checkpoint.checkpoint(
                    model.forward_r, e_r, h_r, pos[r0:-1], Ks, Vs, r, b - 1, use_reentrant=False)
            else:
                o_r, k_r, v_r = model.forward_r(e_r, h_r, pos[r0:-1], Ks, Vs, r, b - 1)
            Ks[r], Vs[r], prev = k_r, v_r, o_r
            t_r = tok[r0 + 2:]                    # x_{i+2} for rows r0 .. T-3
            m_r = torch.arange(r0, T - 2, device=tok.device) + 2 >= pl
            l_r, c_r = ce(o_r[:-1], t_r, m_r)
            loss = loss + l_r
            c2 += c_r
            n2 += int(m_r.sum())
    return loss, n, c1, c2, n2


def evaluate(model, data, emb, head, draft, two_step, dev, maxn=60):
    model.eval()
    L = N = C1 = C2 = N2 = 0
    with torch.no_grad():
        for i in data.val[:maxn]:
            tok, hid, pos, pl = data.get(i, dev)
            l, n, c1, c2, n2 = seq_loss(model, emb, head, draft, tok, hid, pos, pl, two_step)
            L += float(l); N += n; C1 += c1; C2 += c2; N2 += n2
    model.train()
    return L / max(1, N + N2), C1 / max(1, N), (C2 / N2 if N2 else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode")
    ap.add_argument("data")
    ap.add_argument("out", nargs="?")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--two-step", action="store_true")
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--init", default=None, help="start from this MTP head (continue training)")
    ap.add_argument("--ctx", type=int, default=6144)
    ap.add_argument("--qat", action="store_true", help="train / evaluate with INT4 fake quantization")
    ap.add_argument("--val-data", default=None, help="separate evaluation set (all its sequences)")
    a = ap.parse_args()
    dev = "cuda"
    data = Data(a.data, a.ctx)
    vdata = data
    if a.val_data:
        vdata = Data(a.val_data, a.ctx)
        vdata.val = list(range(len(vdata.idx)))
        data.train = data.train + data.val  # all training sequences train
    emb, head, draft = load_frozen(dev)
    src = a.out if a.mode == "eval" and a.out else (a.init or os.path.join(MODEL, "model_mtp.safetensors"))
    model = MTP(load_mtp(src)).to(dev)
    model.qat = a.qat
    print(f"{len(data.train)} train / {len(data.val)} val sequences", flush=True)
    l, a1, a2 = evaluate(model, vdata, emb, head, draft, True, dev)
    print(f"[{src}] val loss {l:.4f}  draft top-1: step1 {a1:.4f} step2 {a2:.4f}", flush=True)
    if a.mode == "eval":
        return
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.98), weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / 50) * (0.5 * (1 + math.cos(math.pi * min(1.0, s / a.steps)))))
    rnd = random.Random(1)
    t0 = time.time()
    best = a1 + a2
    for step in range(a.steps):
        tot = 0
        for _ in range(a.accum):
            tok, hid, pos, pl = data.get(rnd.choice(data.train), dev)
            l, n, _, _, n2 = seq_loss(model, emb, head, draft, tok, hid, pos, pl, a.two_step)
            (l / max(1, (n + n2)) / a.accum).backward()
            tot += n
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); opt.zero_grad(set_to_none=True); sched.step()
        if (step + 1) % a.eval_every == 0 or step + 1 == a.steps:
            l, a1, a2 = evaluate(model, vdata, emb, head, draft, True, dev)
            print(f"step {step + 1}: val loss {l:.4f} step1 {a1:.4f} step2 {a2:.4f} "
                  f"({time.time() - t0:.0f}s, lr {sched.get_last_lr()[0]:.2e})", flush=True)
            if a1 + a2 > best:
                best = a1 + a2
                save_file(model.state(), a.out)
                print(f"  saved {a.out}", flush=True)


if __name__ == "__main__":
    main()
