#!/usr/bin/env python3
"""End-to-end Relax model correctness and latency probe.

This script intentionally uses small, self-contained models so it can run in an
existing TVM checkout without external datasets.  It checks the compiled TVM
VirtualMachine output against a NumPy oracle and reports latency for each model,
batch size, and backend target.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import subprocess
import sys
import time
from collections import OrderedDict
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import tvm
from tvm import dlight as dl
from tvm import relax
from tvm.relax.frontend import nn
from tvm.runtime.ndarray import NDArray as TVMNDArray


SEED = 123
DEFAULT_ATOL = 2e-4
DEFAULT_RTOL = 2e-3


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return float("nan")
    xs = sorted(values)
    if len(xs) == 1:
        return float(xs[0])
    k = (len(xs) - 1) * q
    f = int(math.floor(k))
    c = int(math.ceil(k))
    if f == c:
        return float(xs[f])
    return float(xs[f] * (c - k) + xs[c] * (k - f))


def summarize(values_ms: Sequence[float]) -> Dict[str, float]:
    if not values_ms:
        return {}
    return {
        "latency_mean_ms": float(np.mean(values_ms)),
        "latency_p50_ms": percentile(values_ms, 0.50),
        "latency_p90_ms": percentile(values_ms, 0.90),
        "latency_p99_ms": percentile(values_ms, 0.99),
    }


def sync(dev) -> None:
    if hasattr(dev, "sync") and callable(dev.sync):
        dev.sync()


def to_numpy(arr_like) -> np.ndarray:
    if isinstance(arr_like, np.ndarray):
        return arr_like
    if isinstance(arr_like, TVMNDArray):
        return arr_like.numpy()
    if hasattr(arr_like, "numpy") and callable(arr_like.numpy):
        return arr_like.numpy()
    if hasattr(arr_like, "asnumpy") and callable(arr_like.asnumpy):
        return arr_like.asnumpy()
    return np.array(arr_like)


def extract_array(p):
    if isinstance(p, (TVMNDArray, np.ndarray)):
        return p
    for attr in ("value", "data", "array", "ndarray", "tensor", "_value", "_data", "_array"):
        if hasattr(p, attr):
            x = getattr(p, attr)
            if isinstance(x, (TVMNDArray, np.ndarray)):
                return x
    for meth in ("numpy", "asnumpy"):
        if hasattr(p, meth) and callable(getattr(p, meth)):
            try:
                return getattr(p, meth)()
            except Exception:
                pass
    return None


def init_param(name: str, shape: Tuple[int, ...], dtype: str, rng: np.random.Generator) -> np.ndarray:
    if not np.issubdtype(np.dtype(dtype), np.floating):
        return np.zeros(shape, dtype=dtype)
    lname = name.lower()
    if ("ln" in lname or "norm" in lname) and "weight" in lname:
        return np.ones(shape, dtype=dtype)
    if "bias" in lname:
        return np.zeros(shape, dtype=dtype)
    scale = 0.01 if "pos" in lname else 0.02
    return (rng.standard_normal(size=shape) * scale).astype(dtype)


def materialize_params(params, rng: np.random.Generator) -> "OrderedDict[str, np.ndarray]":
    named = OrderedDict()
    items = params.items() if isinstance(params, dict) else list(params)
    for name, holder in items:
        arr = extract_array(holder)
        if arr is None:
            shape = tuple(int(s) for s in holder.shape)
            dtype = str(holder.dtype)
            arr = init_param(str(name), shape, dtype, rng)
        else:
            arr = to_numpy(arr)
        named[str(name)] = arr.astype("float32", copy=False)
    return named


def get_param(named: Dict[str, np.ndarray], *tokens: str) -> np.ndarray:
    def match(name: str, token: str) -> bool:
        if token == "":
            return True
        parts = [p for p in re.split(r"[._]", name) if p]
        token_parts = [p for p in re.split(r"[._]", token) if p]
        if len(token_parts) > 1:
            return all(part in parts for part in token_parts)
        if len(token) == 1 or token.isdigit():
            return token in parts
        return token in parts or token in name

    matches = []
    for name, arr in named.items():
        if all(match(name, tok) for tok in tokens):
            matches.append((name, arr))
    if len(matches) != 1:
        available = ", ".join(named.keys())
        raise KeyError(f"Expected one param matching {tokens}, got {len(matches)}. Available: {available}")
    return matches[0][1]


def linear_np(x: np.ndarray, w: np.ndarray, b: np.ndarray | None = None) -> np.ndarray:
    y = np.matmul(x, np.swapaxes(w, -1, -2))
    if b is not None:
        y = y + b
    return y.astype("float32")


def conv2d_nchw_np(
    x: np.ndarray,
    w: np.ndarray,
    b: np.ndarray | None,
    stride: int,
    padding: int,
) -> np.ndarray:
    n, cin, h, width = x.shape
    cout, cin_w, kh, kw = w.shape
    assert cin == cin_w
    hp = h + 2 * padding
    wp = width + 2 * padding
    xp = np.zeros((n, cin, hp, wp), dtype="float32")
    xp[:, :, padding : padding + h, padding : padding + width] = x
    oh = (hp - kh) // stride + 1
    ow = (wp - kw) // stride + 1
    out = np.zeros((n, cout, oh, ow), dtype="float32")
    for yy in range(oh):
        ys = yy * stride
        for xx in range(ow):
            xs = xx * stride
            patch = xp[:, :, ys : ys + kh, xs : xs + kw]
            out[:, :, yy, xx] = np.einsum("nchw,ochw->no", patch, w, optimize=True)
    if b is not None:
        out += b.reshape(1, -1, 1, 1)
    return out.astype("float32")


def layer_norm_np(x: np.ndarray, w: np.ndarray, b: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.mean((x - mean) ** 2, axis=-1, keepdims=True)
    return ((x - mean) / np.sqrt(var + eps) * w + b).astype("float32")


def gelu_np(x: np.ndarray) -> np.ndarray:
    # Match the exact erf-based GELU lowering used by Relax.
    erf_vec = np.vectorize(math.erf)
    return (0.5 * x * (1.0 + erf_vec(x / math.sqrt(2.0)))).astype("float32")


def softmax_np(x: np.ndarray, axis: int = -1) -> np.ndarray:
    y = x - np.max(x, axis=axis, keepdims=True)
    exp = np.exp(y)
    return (exp / np.sum(exp, axis=axis, keepdims=True)).astype("float32")


class MLPModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 10, bias=False)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


class CNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2D(3, 16, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2 = nn.Conv2D(16, 32, kernel_size=3, stride=2, padding=1, bias=True)
        self.head = nn.Linear(32, 10, bias=True)

    def forward(self, x):
        x = nn.relu(self.conv1(x))
        x = nn.relu(self.conv2(x))
        x = nn.sum(x, axis=[2, 3], keepdims=False) / 256.0
        return self.head(x)


class TinySelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, seq_len: int):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.seq_len = seq_len
        self.q = nn.Linear(dim, dim, bias=True)
        self.k = nn.Linear(dim, dim, bias=True)
        self.v = nn.Linear(dim, dim, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)

    def _reshape_heads(self, x):
        x = nn.reshape(x, (-1, self.seq_len, self.num_heads, self.head_dim))
        return nn.permute_dims(x, axes=(0, 2, 1, 3))

    def forward(self, x):
        q = self._reshape_heads(self.q(x))
        k = self._reshape_heads(self.k(x))
        v = self._reshape_heads(self.v(x))
        kt = nn.permute_dims(k, axes=(0, 1, 3, 2))
        attn = nn.softmax(nn.matmul(q, kt) * (1.0 / (self.head_dim**0.5)), axis=-1)
        out = nn.matmul(attn, v)
        out = nn.permute_dims(out, axes=(0, 2, 1, 3))
        out = nn.reshape(out, (-1, self.seq_len, self.dim))
        return self.proj(out)


class TinyTransformer(nn.Module):
    def __init__(self, dim: int = 64, num_heads: int = 4, seq_len: int = 16, hidden: int = 128):
        super().__init__()
        self.seq_len = seq_len
        self.ln1 = nn.LayerNorm(dim)
        self.attn = TinySelfAttention(dim, num_heads, seq_len)
        self.ln2 = nn.LayerNorm(dim)
        self.ff1 = nn.Linear(dim, hidden, bias=True)
        self.ff2 = nn.Linear(hidden, dim, bias=True)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, 10, bias=True)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff2(nn.gelu(self.ff1(self.ln2(x))))
        x = self.norm(x)
        x = nn.sum(x, axis=[1], keepdims=False) / float(self.seq_len)
        return self.head(x)


class TinyViT(nn.Module):
    def __init__(self, dim: int = 64, num_heads: int = 4, seq_len: int = 16, hidden: int = 128):
        super().__init__()
        self.seq_len = seq_len
        self.dim = dim
        self.embed = nn.Linear(784, seq_len * dim, bias=True)
        self.pos = nn.Parameter((1, seq_len, dim), "float32")
        self.block = TinyTransformer(dim=dim, num_heads=num_heads, seq_len=seq_len, hidden=hidden)

    def forward(self, x):
        x = self.embed(x)
        x = nn.reshape(x, (-1, self.seq_len, self.dim))
        x = x + self.pos
        return self.block(x)


class PatchLinearEmbed(nn.Module):
    def __init__(self, in_dim: int = 784, seq_len: int = 49, embed_dim: int = 256):
        super().__init__()
        self.seq_len = seq_len
        self.embed_dim = embed_dim
        self.fc = nn.Linear(in_dim, seq_len * embed_dim, bias=True)

    def forward(self, x):
        x = self.fc(x)
        return nn.reshape(x, (-1, self.seq_len, self.embed_dim))


class TinyMLPBlock(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        return self.fc2(nn.gelu(self.fc1(x)))


class TinyTransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, seq_len: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = TinySelfAttention(dim, num_heads, seq_len)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = TinyMLPBlock(dim, int(dim * mlp_ratio))

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class TinyViTFull(nn.Module):
    """ViT shape used by the existing tuning_vit_logs experiment."""

    def __init__(
        self,
        num_classes: int = 10,
        d_model: int = 256,
        nhead: int = 8,
        depth: int = 2,
        seq_len: int = 49,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.embed = PatchLinearEmbed(784, seq_len, d_model)
        self.pos = nn.Parameter((1, seq_len, d_model), "float32")
        self.blocks = nn.ModuleList(
            [TinyTransformerBlock(d_model, nhead, seq_len, mlp_ratio) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes, bias=False)

    def forward(self, x):
        x = self.embed(x) + self.pos
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        x = nn.sum(x, axis=[1], keepdims=False) / float(self.seq_len)
        return self.head(x)


class TinyViTTokens(nn.Module):
    """ViT encoder/classifier with pre-tokenized patch embeddings as input."""

    def __init__(
        self,
        num_classes: int = 10,
        d_model: int = 256,
        nhead: int = 8,
        depth: int = 2,
        seq_len: int = 49,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pos = nn.Parameter((1, seq_len, d_model), "float32")
        self.blocks = nn.ModuleList(
            [TinyTransformerBlock(d_model, nhead, seq_len, mlp_ratio) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes, bias=False)

    def forward(self, x):
        x = x + self.pos
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        x = nn.sum(x, axis=[1], keepdims=False) / float(self.seq_len)
        return self.head(x)


def ref_mlp(x: np.ndarray, p: Dict[str, np.ndarray]) -> np.ndarray:
    y = linear_np(x, get_param(p, "fc1", "weight"), get_param(p, "fc1", "bias"))
    y = np.maximum(y, 0.0)
    return linear_np(y, get_param(p, "fc2", "weight"), None)


def ref_cnn(x: np.ndarray, p: Dict[str, np.ndarray]) -> np.ndarray:
    y = conv2d_nchw_np(x, get_param(p, "conv1", "weight"), get_param(p, "conv1", "bias"), 1, 1)
    y = np.maximum(y, 0.0)
    y = conv2d_nchw_np(y, get_param(p, "conv2", "weight"), get_param(p, "conv2", "bias"), 2, 1)
    y = np.maximum(y, 0.0)
    y = np.sum(y, axis=(2, 3)) / 256.0
    return linear_np(y, get_param(p, "head", "weight"), get_param(p, "head", "bias"))


def attention_np(x: np.ndarray, p: Dict[str, np.ndarray], prefix: str, seq_len: int, dim: int, heads: int) -> np.ndarray:
    head_dim = dim // heads
    q = linear_np(x, get_param(p, prefix, "attn", "q", "weight"), get_param(p, prefix, "attn", "q", "bias"))
    k = linear_np(x, get_param(p, prefix, "attn", "k", "weight"), get_param(p, prefix, "attn", "k", "bias"))
    v = linear_np(x, get_param(p, prefix, "attn", "v", "weight"), get_param(p, prefix, "attn", "v", "bias"))
    bsz = x.shape[0]
    q = q.reshape(bsz, seq_len, heads, head_dim).transpose(0, 2, 1, 3)
    k = k.reshape(bsz, seq_len, heads, head_dim).transpose(0, 2, 1, 3)
    v = v.reshape(bsz, seq_len, heads, head_dim).transpose(0, 2, 1, 3)
    attn = softmax_np(np.matmul(q, np.swapaxes(k, -1, -2)) * (1.0 / math.sqrt(head_dim)), axis=-1)
    y = np.matmul(attn, v).transpose(0, 2, 1, 3).reshape(bsz, seq_len, dim)
    return linear_np(
        y,
        get_param(p, prefix, "attn", "proj", "weight"),
        get_param(p, prefix, "attn", "proj", "bias"),
    )


def transformer_core_np(
    x: np.ndarray,
    p: Dict[str, np.ndarray],
    prefix: str,
    seq_len: int = 16,
    dim: int = 64,
    heads: int = 4,
) -> np.ndarray:
    y = layer_norm_np(x, get_param(p, prefix, "ln1", "weight"), get_param(p, prefix, "ln1", "bias"))
    x = x + attention_np(y, p, prefix, seq_len, dim, heads)
    y = layer_norm_np(x, get_param(p, prefix, "ln2", "weight"), get_param(p, prefix, "ln2", "bias"))
    y = linear_np(y, get_param(p, prefix, "ff1", "weight"), get_param(p, prefix, "ff1", "bias"))
    y = gelu_np(y)
    y = linear_np(y, get_param(p, prefix, "ff2", "weight"), get_param(p, prefix, "ff2", "bias"))
    x = x + y
    y = layer_norm_np(x, get_param(p, prefix, "norm", "weight"), get_param(p, prefix, "norm", "bias"))
    y = np.sum(y, axis=1) / float(seq_len)
    return linear_np(y, get_param(p, prefix, "head", "weight"), get_param(p, prefix, "head", "bias"))


def transformer_block_np(
    x: np.ndarray,
    p: Dict[str, np.ndarray],
    prefix: str,
    seq_len: int,
    dim: int,
    heads: int,
) -> np.ndarray:
    y = layer_norm_np(x, get_param(p, prefix, "ln1", "weight"), get_param(p, prefix, "ln1", "bias"))
    x = x + attention_np(y, p, prefix, seq_len, dim, heads)
    y = layer_norm_np(x, get_param(p, prefix, "ln2", "weight"), get_param(p, prefix, "ln2", "bias"))
    y = linear_np(y, get_param(p, prefix, "mlp", "fc1", "weight"), get_param(p, prefix, "mlp", "fc1", "bias"))
    y = gelu_np(y)
    y = linear_np(y, get_param(p, prefix, "mlp", "fc2", "weight"), get_param(p, prefix, "mlp", "fc2", "bias"))
    return x + y


def ref_transformer(x: np.ndarray, p: Dict[str, np.ndarray]) -> np.ndarray:
    return transformer_core_np(x, p, prefix="")


def ref_vit(x: np.ndarray, p: Dict[str, np.ndarray]) -> np.ndarray:
    seq_len = 16
    dim = 64
    y = linear_np(x, get_param(p, "embed", "weight"), get_param(p, "embed", "bias"))
    y = y.reshape(x.shape[0], seq_len, dim) + get_param(p, "pos")
    return transformer_core_np(y, p, prefix="block", seq_len=seq_len, dim=dim, heads=4)


def ref_vit_full(x: np.ndarray, p: Dict[str, np.ndarray]) -> np.ndarray:
    seq_len = 49
    dim = 256
    y = linear_np(x, get_param(p, "embed", "fc", "weight"), get_param(p, "embed", "fc", "bias"))
    y = y.reshape(x.shape[0], seq_len, dim) + get_param(p, "pos")
    for i in range(2):
        y = transformer_block_np(y, p, f"blocks.{i}", seq_len=seq_len, dim=dim, heads=8)
    y = layer_norm_np(y, get_param(p, "norm", "weight"), get_param(p, "norm", "bias"))
    y = np.sum(y, axis=1) / float(seq_len)
    return linear_np(y, get_param(p, "head", "weight"), None)


def ref_vit_tokens(x: np.ndarray, p: Dict[str, np.ndarray]) -> np.ndarray:
    seq_len = 49
    dim = 256
    y = x + get_param(p, "pos")
    for i in range(2):
        y = transformer_block_np(y, p, f"blocks.{i}", seq_len=seq_len, dim=dim, heads=8)
    y = layer_norm_np(y, get_param(p, "norm", "weight"), get_param(p, "norm", "bias"))
    y = np.sum(y, axis=1) / float(seq_len)
    return linear_np(y, get_param(p, "head", "weight"), None)


def ref_vit_tokens_small(x: np.ndarray, p: Dict[str, np.ndarray]) -> np.ndarray:
    seq_len = 49
    dim = 64
    y = x + get_param(p, "pos")
    y = transformer_block_np(y, p, "blocks.0", seq_len=seq_len, dim=dim, heads=4)
    y = layer_norm_np(y, get_param(p, "norm", "weight"), get_param(p, "norm", "bias"))
    y = np.sum(y, axis=1) / float(seq_len)
    return linear_np(y, get_param(p, "head", "weight"), None)


def make_case(model_name: str, batch: int):
    if model_name == "mlp":
        return MLPModel(), (batch, 784), ref_mlp
    if model_name == "cnn":
        return CNNModel(), (batch, 3, 32, 32), ref_cnn
    if model_name == "transformer":
        return TinyTransformer(), (batch, 16, 64), ref_transformer
    if model_name == "vit":
        return TinyViT(), (batch, 784), ref_vit
    if model_name == "vit_full":
        return TinyViTFull(), (batch, 784), ref_vit_full
    if model_name == "vit_tokens":
        return TinyViTTokens(), (batch, 49, 256), ref_vit_tokens
    if model_name == "vit_tokens_small":
        return TinyViTTokens(d_model=64, nhead=4, depth=1, seq_len=49), (batch, 49, 64), ref_vit_tokens_small
    raise ValueError(f"Unknown model {model_name}")


def timed_runs(fn: Callable[[], object], dev, warmup: int, measure: int) -> List[float]:
    for _ in range(warmup):
        fn()
    sync(dev)
    values = []
    for _ in range(measure):
        t0 = time.perf_counter_ns()
        fn()
        sync(dev)
        values.append((time.perf_counter_ns() - t0) / 1e6)
    return values


def apply_pipeline(mod, target, args):
    if args.pipeline == "default":
        return mod, relax.get_default_pipeline(target)
    if args.pipeline == "static_shape_tuning":
        kwargs = {"target": target, "total_trials": args.total_trials}
        if args.work_dir:
            kwargs["work_dir"] = args.work_dir
        pipe = relax.get_pipeline("static_shape_tuning", **kwargs)
        mod = pipe(mod)
        with target:
            mod = dl.ApplyDefaultSchedule(
                dl.gpu.Matmul(),
                dl.gpu.GEMV(),
                dl.gpu.Reduction(),
                dl.gpu.GeneralReduction(),
                dl.gpu.Fallback(),
            )(mod)
        return mod, None
    raise ValueError(f"Unknown pipeline {args.pipeline}")


def make_target(target_kind: str, dev, target_string: str):
    if target_string:
        return tvm.target.Target(target_string)
    try:
        return tvm.target.Target.from_device(dev)
    except (KeyError, ValueError, RuntimeError, tvm.error.TVMError):
        return tvm.target.Target(target_kind)


def runtime_device_kind(target_kind: str) -> str:
    if target_kind != "sycl":
        return target_kind
    if tvm.get_global_func("device_api.oneapi", allow_missing=True) is not None:
        return "oneapi"
    return "sycl"


def run_one(model_name: str, batch: int, target_kind: str, warmup: int, measure: int, args) -> Dict[str, object]:
    seed_target = args.seed_target_alias or target_kind
    case_seed = SEED + batch * 17 + sum(ord(c) for c in (model_name + seed_target))
    rng = np.random.default_rng(case_seed)
    device_kind = runtime_device_kind(target_kind)
    dev = tvm.device(device_kind, args.device_id)
    target = make_target(target_kind, dev, args.target_string)
    model, input_shape, ref_fn = make_case(model_name, batch)
    started = time.perf_counter()
    mod, params = model.export_tvm({"forward": {"x": nn.spec.Tensor(input_shape, "float32")}})
    named_np = materialize_params(params, rng)
    x_np = (rng.standard_normal(size=input_shape) * 0.1).astype("float32")
    try:
        compile_t0 = time.perf_counter()
        mod, compile_pipeline = apply_pipeline(mod, target, args)
        if compile_pipeline is None:
            ex = tvm.compile(mod, target=target)
        else:
            ex = tvm.compile(mod, target=target, relax_pipeline=compile_pipeline)
        compile_s = time.perf_counter() - compile_t0
        vm = relax.VirtualMachine(ex, dev)
        x_dev = tvm.nd.array(x_np, dev)
        tvm_params = [tvm.nd.array(arr, dev) for arr in named_np.values()]
        out = vm["forward"](x_dev, *tvm_params).numpy()
        ref = ref_fn(x_np, named_np).astype("float32")
        if out.shape != ref.shape:
            return {
                "model": model_name,
                "batch": batch,
                "target": target_kind,
                "runtime_device": device_kind,
                "device_id": args.device_id,
                "seed_target_alias": seed_target,
                "status": "fail_shape",
                "pipeline": args.pipeline,
                "work_dir": args.work_dir or "",
                "resolved_target": str(target),
                "output_shape": str(tuple(out.shape)),
                "reference_shape": str(tuple(ref.shape)),
            }
        abs_err = np.abs(out - ref)
        max_abs = float(np.max(abs_err)) if abs_err.size else 0.0
        max_rel = float(np.max(abs_err / np.maximum(np.abs(ref), 1e-6))) if abs_err.size else 0.0
        allclose = bool(np.allclose(out, ref, atol=args.atol, rtol=args.rtol))
        latency = timed_runs(lambda: vm["forward"](x_dev, *tvm_params), dev, warmup=warmup, measure=measure)
        row = {
            "model": model_name,
            "batch": batch,
            "target": target_kind,
            "runtime_device": device_kind,
            "device_id": args.device_id,
            "seed_target_alias": seed_target,
            "status": "pass" if allclose else "fail_tolerance",
            "pipeline": args.pipeline,
            "work_dir": args.work_dir or "",
            "resolved_target": str(target),
            "input_shape": str(tuple(input_shape)),
            "output_shape": str(tuple(out.shape)),
            "num_params": len(named_np),
            "max_abs_err": max_abs,
            "max_rel_err": max_rel,
            "atol": args.atol,
            "rtol": args.rtol,
            "compile_s": compile_s,
            "total_s": time.perf_counter() - started,
        }
        row.update(summarize(latency))
        return row
    except Exception as exc:  # Keep the whole experiment moving.
        return {
            "model": model_name,
            "batch": batch,
            "target": target_kind,
            "runtime_device": device_kind,
            "device_id": args.device_id,
            "seed_target_alias": seed_target,
            "status": "runtime_error",
            "input_shape": str(tuple(input_shape)),
            "pipeline": args.pipeline,
            "work_dir": args.work_dir or "",
            "resolved_target": str(target),
            "error": repr(exc),
            "total_s": time.perf_counter() - started,
        }


def parse_csv_list(text: str, cast: Callable[[str], object] = str) -> List[object]:
    return [cast(x.strip()) for x in text.split(",") if x.strip()]


def run_case_subprocess(script_path: Path, model_name: str, batch: int, target_kind: str, args):
    case_work_dir = ""
    if args.work_dir:
        case_work_dir = str(Path(args.work_dir) / f"{target_kind}_{model_name}_b{batch}")
        Path(case_work_dir).mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(script_path),
        "--worker",
        "--models",
        model_name,
        "--batches",
        str(batch),
        "--targets",
        target_kind,
        "--device-id",
        str(args.device_id),
        "--warmup",
        str(args.warmup),
        "--measure",
        str(args.measure),
        "--pipeline",
        args.pipeline,
        "--total-trials",
        str(args.total_trials),
        "--atol",
        str(args.atol),
        "--rtol",
        str(args.rtol),
    ]
    if args.target_string:
        cmd.extend(["--target-string", args.target_string])
    if args.seed_target_alias:
        cmd.extend(["--seed-target-alias", args.seed_target_alias])
    if case_work_dir:
        cmd.extend(["--work-dir", case_work_dir])
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=args.case_timeout_s,
            env=os.environ.copy(),
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "model": model_name,
            "batch": batch,
            "target": target_kind,
            "device_id": args.device_id,
            "status": "timeout",
            "error": f"case exceeded {args.case_timeout_s} s: {exc}",
        }
    if proc.returncode != 0:
        return {
            "model": model_name,
            "batch": batch,
            "target": target_kind,
            "device_id": args.device_id,
            "status": "runtime_error",
            "error": (proc.stderr or proc.stdout)[-4000:],
        }
    for line in reversed(proc.stdout.splitlines()):
        if line.startswith("RESULT_JSON="):
            return json.loads(line[len("RESULT_JSON=") :])
    return {
        "model": model_name,
        "batch": batch,
        "target": target_kind,
        "device_id": args.device_id,
        "status": "parse_error",
        "error": f"worker produced no result: {proc.stdout[-4000:]}",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--models", default="mlp,cnn,transformer,vit")
    parser.add_argument("--batches", default="1,8")
    parser.add_argument("--targets", default="sycl")
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument(
        "--target-string",
        default="",
        help="Optional complete TVM target string; otherwise detect properties from the device.",
    )
    parser.add_argument(
        "--seed-target-alias",
        default="",
        help="Use another target name when deriving random inputs for paired runs.",
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--measure", type=int, default=30)
    parser.add_argument("--pipeline", choices=["default", "static_shape_tuning"], default="default")
    parser.add_argument("--total-trials", type=int, default=0)
    parser.add_argument("--work-dir", default="")
    parser.add_argument("--atol", type=float, default=DEFAULT_ATOL)
    parser.add_argument("--rtol", type=float, default=DEFAULT_RTOL)
    parser.add_argument("--case-timeout-s", type=int, default=1800)
    parser.add_argument("--out-json", default="relax_model_probe.json")
    parser.add_argument("--out-csv", default="relax_model_probe.csv")
    args = parser.parse_args()

    models = parse_csv_list(args.models)
    batches = parse_csv_list(args.batches, int)
    targets = parse_csv_list(args.targets)

    if args.worker:
        if len(models) != 1 or len(batches) != 1 or len(targets) != 1:
            raise ValueError("worker mode requires exactly one model, batch, and target")
        row = run_one(models[0], batches[0], targets[0], args.warmup, args.measure, args)
        print("RESULT_JSON=" + json.dumps(row, sort_keys=True))
        return

    results = []
    script_path = Path(__file__).resolve()
    for target_kind in targets:
        for model_name in models:
            for batch in batches:
                row = run_case_subprocess(script_path, model_name, batch, target_kind, args)
                results.append(row)
                msg = (
                    f"{target_kind:>4s} {model_name:>11s} B{batch:<3d} "
                    f"{row.get('status')} abs={row.get('max_abs_err', 'NA')} "
                    f"p50={row.get('latency_p50_ms', 'NA')}"
                )
                if "error" in row:
                    msg += f" err={str(row['error'])[:180]}"
                print(msg, flush=True)

    out_json = Path(args.out_json)
    out_csv = Path(args.out_csv)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "protocol": {
                    "seed": SEED,
                    "seed_target_alias": args.seed_target_alias or None,
                    "warmup": args.warmup,
                    "measure": args.measure,
                    "atol": args.atol,
                    "rtol": args.rtol,
                    "pipeline": args.pipeline,
                    "target_string": args.target_string or None,
                    "device_id": args.device_id,
                    "total_trials": args.total_trials,
                },
                "results": results,
            },
            f,
            indent=2,
            sort_keys=True,
        )
    fieldnames = sorted({k for row in results for k in row.keys()})
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


if __name__ == "__main__":
    main()
