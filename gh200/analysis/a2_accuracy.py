"""A2: accuracy of BF16 mantissa truncation versus direct FP8 quantization.

Formats compared (all simulated exactly in numpy, then GEMM in FP64):
  bf16-k    round to BF16, then keep k of its 7 fraction bits (truncation, as the
            HPL-MxP storage path does)
  fp8-e4m3-k  per-tensor scale to the E4M3 maximum (448) from the tensor's amax, as
            Transformer Engine does, round to E4M3 (RNE, subnormals, saturation),
            then keep k of its 3 fraction bits
  fp8-e5m2  same with E5M2 (range-oriented FP8)

Data: uniform U[-1,1), Gaussian, a wide log-uniform magnitude range (2^-24..2^24,
the "needs range" case), heavy-tailed Student-t activations, and real Llama-2-70B
weights read straight from the safetensors checkpoint.

Metrics: relative Frobenius error of the operand itself and of C = A @ B, and
the fraction of nonzero values that flush to zero.
"""
import argparse
import json
import struct
import sys

import numpy as np


def minifloat(x, ebits, mbits, bias, maxval):
    """Round x (float64) to a sign/ebits/mbits minifloat with subnormals, RNE, saturating."""
    a = np.abs(x)
    emin = 1 - bias
    with np.errstate(divide="ignore"):
        e = np.floor(np.log2(np.where(a > 0, a, 1.0)))
    e = np.maximum(e, emin)
    q = np.exp2(e - mbits)
    r = np.round(a / q) * q  # numpy rounds half to even
    r = np.minimum(r, maxval)
    return np.sign(x) * r


def keep_bits(v, total, k):
    """Truncate a value already on a `total`-fraction-bit grid to k fraction bits (toward zero)."""
    if k >= total:
        return v
    a = np.abs(v)
    with np.errstate(divide="ignore"):
        e = np.floor(np.log2(np.where(a > 0, a, 1.0)))
    q = np.exp2(e - k)
    return np.sign(v) * np.floor(a / q) * q


def bf16(x, k=7):
    # float32 then RNE to bf16 via the bit trick
    f = x.astype(np.float32)
    u = f.view(np.uint32).astype(np.uint64)
    u = (u + 0x7FFF + ((u >> 16) & 1)) >> 16 << 16
    b = u.astype(np.uint32).view(np.float32).astype(np.float64)
    if k < 7:  # truncate to k fraction bits on the raw pattern (MxP storage behaviour)
        bb = b.astype(np.float32).view(np.uint32)
        bb &= np.uint32(~((1 << (23 - k)) - 1) & 0xFFFFFFFF)
        b = bb.view(np.float32).astype(np.float64)
    return b


def fp8(x, kind="e4m3", k=3):
    if kind == "e4m3":
        ebits, mbits, bias, maxval = 4, 3, 7, 448.0
    else:
        ebits, mbits, bias, maxval = 5, 2, 15, 57344.0
    amax = np.max(np.abs(x))
    s = maxval / amax if amax > 0 else 1.0
    y = minifloat(x * s, ebits, mbits, bias, maxval)
    y = keep_bits(y, mbits, k)
    return y / s


def read_safetensor(path, name, rows=None):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
        meta = hdr[name]
        assert meta["dtype"] == "BF16", meta["dtype"]
        shape = meta["shape"]
        beg, end = meta["data_offsets"]
        f.seek(8 + n + beg)
        count = (end - beg) // 2
        if rows is not None:
            count = min(count, rows * shape[1])
            shape = [count // shape[1], shape[1]]
        raw = np.frombuffer(f.read(count * 2), dtype=np.uint16)
    return (raw.astype(np.uint32) << 16).view(np.float32).astype(np.float64).reshape(shape)


def find_tensor(model_dir, want):
    idx = json.load(open(f"{model_dir}/model.safetensors.index.json"))
    return f"{model_dir}/{idx['weight_map'][want]}"


def formats():
    out = [("fp64", lambda x: x)]
    out += [(f"bf16-{k}", (lambda k: lambda x: bf16(x, k))(k)) for k in (7, 5, 3, 1, 0)]
    out += [(f"fp8-e4m3-{k}", (lambda k: lambda x: fp8(x, "e4m3", k))(k)) for k in (3, 2, 1, 0)]
    out += [("fp8-e5m2", lambda x: fp8(x, "e5m2", 2))]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/data/mlperf/llama2_70b_lora/model")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    m, k, n = 512, 4096, 256
    data = {
        "uniform": (rng.uniform(-1, 1, (m, k)), rng.uniform(-1, 1, (k, n))),
        "gaussian": (rng.standard_normal((m, k)), rng.standard_normal((k, n))),
        "wide-range": tuple(rng.choice([-1, 1], s) * np.exp2(rng.uniform(-24, 24, s)) for s in [(m, k), (k, n)]),
        # rows of A (and columns of B) at scales 2^-24..2^24: each output element has its own
        # magnitude, as when one GEMM mixes physical quantities of very different size
        "row-scaled": (rng.uniform(-1, 1, (m, k)) * np.exp2(rng.uniform(-24, 24, (m, 1))),
                       rng.uniform(-1, 1, (k, n)) * np.exp2(rng.uniform(-24, 24, (1, n)))),
    }
    # Real weights: a q_proj and an MLP down_proj from layer 40, against heavy-tailed activations.
    acts = rng.standard_t(3, (8192, n))
    for tag, name in [("llama2-qkv_proj", "model.layers.40.self_attn.qkv_proj.weight"),
                      ("llama2-down_proj", "model.layers.40.mlp.down_proj.weight")]:
        try:
            w = read_safetensor(find_tensor(a.model, name), name, rows=m)
            data[tag] = (w[:, :8192], acts[: w.shape[1]])
        except Exception as e:  # keep going without real weights
            print(f"skip {tag}: {e}", file=sys.stderr)

    rows = []
    for dname, (A, B) in data.items():
        C = A @ B
        cn = np.linalg.norm(C)
        info = {"data": dname, "a_log2_min": float(np.log2(np.min(np.abs(A[A != 0])))),
                "a_log2_max": float(np.log2(np.max(np.abs(A))))}
        for fname, q in formats():
            Aq, Bq = q(A), q(B)
            Cq = Aq @ Bq
            nz = A != 0
            rows.append({**info, "format": fname,
                         "op_rel": float(np.linalg.norm(Aq - A) / np.linalg.norm(A)),
                         "gemm_rel": float(np.linalg.norm(Cq - C) / cn),
                         # per-element view: small outputs count as much as large ones
                         "elem_rel_median": float(np.median(np.abs(Cq - C) / np.abs(C))),
                         "elem_rel_p99": float(np.percentile(np.abs(Cq - C) / np.abs(C), 99)),
                         "flush_to_zero": float(np.mean((Aq == 0) & nz))})
            r = rows[-1]
            print(dname, fname, f"fro={r['gemm_rel']:.3e}", f"elem_med={r['elem_rel_median']:.3e}",
                  f"elem_p99={r['elem_rel_p99']:.3e}", f"ftz={r['flush_to_zero']:.3f}", flush=True)
    import csv
    with open(a.out, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)


if __name__ == "__main__":
    main()
