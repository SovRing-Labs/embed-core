#!/usr/bin/env python3
"""TERN-NOMIC-0: ternarize a GGUF's linear weights to BitNet I2_S (2 bits/weight).

Reads an f16 GGUF, writes a new GGUF with the same architecture and KV metadata in
which every 2-D *linear* weight tensor is replaced by an I2_S ternary tensor. The
packing and dequant formula are byte-exact with bitnet.cpp's
utils/convert-bitnet-embedding-to-gguf.py:quantize_to_i2_s and with
dequantize_row_i2_s in ggml/src/ggml-cpu/quants.c:

    abs_mean = mean(|w|)              # BitLinear weight_quant_minmax
    scale    = gamma * abs_mean       # gamma is our distillation gain knob
    q        = clip(round(w / (tau*abs_mean)), -1, +1)   # tau = sparsity knob
    code     = {0: -1, 1: 0, 2: +1}   # {-1,0,+1} -> I2_S nibble encoding
    block    = 128 values -> 32 bytes; byte[b][gp] holds values
               128*b + 0*32+gp, +1*32+gp, +2*32+gp, +3*32+gp at bits 6,4,2,0
    buffer   = packed(n/4 bytes) + 32 zero bytes, f32 scale at byte offset n/4

Tensors deliberately left dense (matching the shipped bitnet-embeddings-270m i2_s):
token_embd.weight, *_norm.weight, *_norm_in.weight and any 1-D tensor.

Usage:
  ternarize.py SRC.gguf DST.gguf [--gamma G] [--tau T] [--params p.json] [--report]
  --params is a JSON dict {"tensor.name": {"gamma": .., "tau": ..}} used by the
  distillation stage to override the knobs per tensor.
"""
import argparse
import json
import struct
import sys
import time

import numpy as np
from gguf import GGMLQuantizationType, GGUFReader, GGUFWriter

I2_S = GGMLQuantizationType.I2_S
BLK = 128

DENSE_SUFFIX = (".weight",)


def is_ternary_candidate(name: str, shape) -> bool:
    """2-D linear weight, row length a multiple of the 128-value block."""
    if len(shape) != 2:
        return False
    if name == "token_embd.weight":
        return False          # 23M-param embedding table stays f16
    if name == "token_types.weight":
        return False          # type-id lookup, not a matmul weight
    if name.endswith("_norm.weight") or name.endswith("_norm_in.weight"):
        return False
    if name.endswith("norm.weight"):
        return False
    ne0 = int(shape[0])       # ggml ne[0] = the row length
    nelem = ne0 * int(shape[1])
    return ne0 % BLK == 0 and nelem % BLK == 0


def quantize_to_i2_s(w_flat: np.ndarray, gamma: float = 1.0, tau: float = 1.0) -> np.ndarray:
    """w_flat: ggml row-major flatten (ne[0] fastest) of a 2-D weight."""
    w = np.asarray(w_flat, dtype=np.float32)
    n = w.size
    assert n % BLK == 0, f"nelem {n} not a multiple of {BLK}"

    abs_mean = max(float(np.mean(np.abs(w))), 1e-5)
    q_float = np.round(w / (tau * abs_mean)).clip(-1, 1)

    code = np.ones(n, dtype=np.uint8)      # 1 -> value 0
    code[q_float > 0.5] = 2                # +1
    code[q_float < -0.5] = 0                # -1

    blocks = code.reshape(-1, 4, 32)
    packed = ((blocks[:, 0, :] << 6) | (blocks[:, 1, :] << 4) |
              (blocks[:, 2, :] << 2) | (blocks[:, 3, :])).reshape(-1).astype(np.uint8)

    scale = np.float32(gamma * abs_mean)
    out = np.zeros(n // 4 + 32, dtype=np.uint8)
    out[: n // 4] = packed[: n // 4]
    out[n // 4: n // 4 + 4] = np.frombuffer(scale.tobytes(), dtype=np.uint8)
    return out


def to_py(v):
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, np.ndarray):
        return [to_py(x) for x in v.tolist()]
    if isinstance(v, (list, tuple)):
        return [to_py(x) for x in v]
    return v


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--params", default=None, help="JSON {tensor: {gamma, tau}}")
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()

    overrides = {}
    if args.params:
        with open(args.params) as fh:
            overrides = json.load(fh)

    t0 = time.time()
    src = GGUFReader(args.src)
    n_tern = 0
    stats = []
    src_bytes = 0
    dst_bytes = 0

    w = GGUFWriter(args.dst, arch=src.fields["general.architecture"].contents())
    for key, field in src.fields.items():
        if key == "general.architecture" or key.startswith("GGUF."):
            continue          # architecture is set by the ctor; GGUF.* are pseudo-fields
        w.add_key_value(key, to_py(field.contents()), raw_dtype=field.types[0])

    for t in src.tensors:
        shape = [int(x) for x in t.shape]
        nelem = int(np.prod(shape))
        src_bytes += t.data.nbytes
        if not is_ternary_candidate(t.name, shape):
            w.add_tensor(t.name, t.data, raw_shape=shape,
                         raw_dtype=GGMLQuantizationType(t.tensor_type))
            dst_bytes += t.data.nbytes
            continue

        p = overrides.get(t.name, {})
        gamma = float(p.get("gamma", args.gamma))
        tau = float(p.get("tau", args.tau))

        flat = np.asarray(t.data, dtype=np.float32).reshape(-1)   # ggml order
        packed = quantize_to_i2_s(flat, gamma, tau)
        w.add_tensor(t.name, packed, raw_shape=shape, raw_dtype=I2_S)

        abs_mean = max(float(np.mean(np.abs(flat))), 1e-5)
        nz = float(np.mean(np.round(flat / (tau * abs_mean)).clip(-1, 1) != 0))
        stats.append({"name": t.name, "shape": shape, "gamma": gamma, "tau": tau,
                      "abs_mean": abs_mean, "frac_nonzero": round(nz, 4),
                      "src_bytes": int(t.data.nbytes), "dst_bytes": int(packed.nbytes)})
        src_bytes += t.data.nbytes
        dst_bytes += packed.nbytes
        n_tern += 1

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()

    report = {
        "src": args.src, "dst": args.dst,
        "n_tensors_total": len(src.tensors), "n_ternarized": n_tern,
        "weight_bytes_f16": src_bytes, "weight_bytes_i2s": dst_bytes,
        "compression": round(src_bytes / max(dst_bytes, 1), 3),
        "bits_per_weight": round(dst_bytes * 8 / max(
            sum(s["shape"][0] * s["shape"][1] for s in stats), 1), 3),
        "secs": round(time.time() - t0, 1),
        "tensors": stats,
    }
    with open(args.dst + ".ternarize-report.json", "w") as fh:
        json.dump(report, fh, indent=1)
    summary = {k: v for k, v in report.items() if k != "tensors"}
    print(json.dumps(summary, indent=1))
    if args.report:
        for s in stats[:6]:
            print("  ", s)
    return 0


if __name__ == "__main__":
    sys.exit(main())