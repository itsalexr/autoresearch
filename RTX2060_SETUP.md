# RTX 2060 Setup Notes

This branch (`autoresearch/mar17`) forks karpathy/autoresearch at commit
`32a1460f626e28479d427c033ee485bf5f86875a` and adapts it to run on an RTX 2060
(Turing, sm75, 12 GB VRAM). Below is a summary of every deviation from the
upstream `train.py` and why it exists.

---

## 1. Flash Attention 3 → PyTorch SDPA fallback

**Upstream** unconditionally loads FA3 via the `kernels` hub and uses it for all
attention layers, including the sliding-window ("S") layers defined by
`WINDOW_PATTERN = "SSSL"`.

**This branch** gates FA3 on `cap >= (8, 0)` (Ampere+). On sm75 it falls back
to `F.scaled_dot_product_attention` with `is_causal=True`:

```python
_USE_FA3 = cap >= (8, 0)
if _USE_FA3:
    fa3 = get_kernel(repo).flash_attn_interface
else:
    fa3 = None

# inside CausalSelfAttention.forward:
if _USE_FA3:
    y = fa3.flash_attn_func(q, k, v, causal=True, window_size=window_size)
else:
    q_t, k_t, v_t = q.transpose(1,2), k.transpose(1,2), v.transpose(1,2)
    y_t = F.scaled_dot_product_attention(q_t, k_t, v_t, is_causal=True)
    y = y_t.transpose(1, 2).contiguous()
```

### Quality impact

**Sliding-window attention is silently dropped.** The upstream model with
`WINDOW_PATTERN = "SSSL"` and `DEPTH = 8` gives:

| Layer index (mod 4) | Pattern char | Window size |
|---|---|---|
| 0, 1, 2 | S | 1024 tokens (half context) |
| 3 | L | 2048 tokens (full) |

With the SDPA fallback every layer uses full 2048-token causal attention.
The model is architecturally different from what runs on H100. Relative
hyperparameter comparisons within this branch are still valid; absolute
`val_bpb` values and optimal hyperparameter magnitudes are not directly
comparable to upstream H100 results.

---

## 2. `bfloat16` → `float16`

**Upstream** uses `torch.amp.autocast(..., dtype=torch.bfloat16)`. The RTX 2060
does not support bfloat16 hardware acceleration.

**This branch** uses `dtype=torch.float16`. The code also hardcodes
`.bfloat16()` casts on embeddings and RoPE buffers in upstream; those are
removed in the `train_gpuN.py` copies (they default to the autocast dtype).

### Quality impact

fp16 has a narrower dynamic range than bfloat16 (5 vs 8 exponent bits).
Without a `GradScaler`, gradient underflow to zero or activation overflow to
`inf`/`nan` is possible under aggressive learning rates or late-training loss
spikes. In practice the runs completed without visible instability at the
default learning rates, but there is latent fragility compared to bfloat16.

---

## 3. `DEVICE_BATCH_SIZE` and `EVAL_BATCH_SIZE`

**Upstream default:** `DEVICE_BATCH_SIZE = 128`. `val_bpb` evaluation reuses
`DEVICE_BATCH_SIZE`.

**This branch:** `DEVICE_BATCH_SIZE = 16`, `EVAL_BATCH_SIZE = 64`.

`TOTAL_BATCH_SIZE = 2**19 = 524,288` tokens per optimizer step is unchanged.
Gradient accumulation compensates: with `DEVICE_BATCH_SIZE=16` and
`MAX_SEQ_LEN=2048`, `grad_accum_steps = 524288 / (16 × 2048) = 16`.

The separate `EVAL_BATCH_SIZE = 64` exists because eval has no backward pass
and can safely use a larger batch. Using `DEVICE_BATCH_SIZE=16` for eval would
make each validation run ~29 minutes on this hardware.

### Quality impact

None. The effective batch size seen by the optimizer is identical. Gradient
accumulation with loss divided by `grad_accum_steps` before each `.backward()`
produces numerically equivalent gradient estimates.

The final settled value went through several iterations during round 1 bringup:
128 (OOM) → 16 (OOM with explicit SDPA mask) → 4 (OOM on backward) → 2
(worked but too slow) → **16 with `is_causal=True`** (no mask materialization,
fits in 12 GB).

---

## 4. Multi-GPU experiment harness

Upstream provides a single `train.py`. This branch adds:

- **`train_gpu0.py` – `train_gpu5.py`**: six copies of `train.py`, each with a
  different hyperparameter variant for one research round. All six copies
  include the RTX 2060 fixes above.
- **`run_parallel.sh`**: orchestrates a research round over SSH to a remote
  6-GPU machine (`turtle.local`). Workflow:
  1. `rsync` repo to remote
  2. `uv sync`, check data cache
  3. Launch GPU 0 first; wait for `torch.compile` to write its inductor cache
  4. Launch GPUs 1–5 reading from the warm cache (avoids 6× parallel
     recompilations)
  5. Kill any stale GPU-holding processes (`nvidia-smi --query-compute-apps`)
     and clear old logs before each run

---

## Time budget

Each training run is **5 minutes** wall-clock (`TIME_BUDGET = 300` seconds in
`prepare.py`), excluding startup and `torch.compile` time. The time budget is
the same as upstream — the metric (`val_bpb`) is therefore comparable within
this branch across rounds, but not across platforms.
