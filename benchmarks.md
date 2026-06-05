# PIDM-ME training-throughput benchmarks

All runs on a Vast.ai container with a single **NVIDIA RTX PRO 6000 Blackwell Workstation Edition** (97.9 GB VRAM, sm_120), torch nightly cu128 (`2.12.0.dev20260407+cu128`), Python 3.12. Variant `darcy_pidm_me` (config `configs/darcy_pidm_me.yaml`), batch size 64 unless noted.

Throughput numbers come from a 30 s polling loop reading the tqdm iter counter; "mean ± std" excludes the first 1-2 windows (warmup / compile capture). Loss numbers are validation test-loss from `print(f'test loss at iteration {iteration}: {loss_test:.3e}')`.

## Summary table

| Run | Steady it/s | Δ vs vanilla | GPU util | VRAM | Test loss @ 15k | Test loss @ end | End iter | Wandb run |
|---|---|---|---|---|---|---|---|---|
| **vanilla** | **15.22 ± 0.09** | — | 89% | 14.8 GB | (no wandb) | (20-min bench only, no test_loss logged) | n/a | — |
| **opt3** (`opt3full_1780444688`) | **22.4** (~22-26 first hour) | **+47%** | 57-68% | 5.7-8.9 GB | TBD | TBD | https://wandb.ai/glowpauul/pi_diffusion/runs/cvq1lfc3 |
| **opt4** (scoped autocast) | pending | pending | pending | pending | TBD | TBD | pending | pending |

(See per-run details below for the *reasons* behind the differences.)

## Vanilla baseline (`baseline_1780436811`)

- **Patches applied:** none. Stock `main.py` with `name`+config-path sed only.
- **Throughput** (30 s polling, 20 min, 38 windows): mean 15.22 it/s, std 0.0924, std/mean 0.6%. Range 15.07–15.33. No drift.
- **GPU:** 89% time-util, 14.8 GB VRAM resident.
- **Diagnosis:** dispatch-bound. Combined throughput of two parallel jobs on one GPU ≈ identical to solo → the GPU is idle in dispatch gaps that the second process fills.

## opt3 (`opt3full_1780444688`, in progress at 300k iters)

- **Patches applied** (vs vanilla):
  1. `optim.Adam(..., fused=True)`  — single CUDA kernel for param updates.
  2. `cur_batch.to(device, non_blocking=True)` (+ `cur_test_batch`)  — async H2D when paired with pinned memory.
  3. `optimizer.zero_grad(set_to_none=True)`  — skip the zero-fill kernel.
  4. `wandb_track = True`  — metrics logged to W&B.
  5. `DataLoader(..., num_workers=4, pin_memory=True, persistent_workers=True)` for train; `num_workers=2` for valid.
  6. `model = torch.compile(model, mode='reduce-overhead')`  — Inductor + CUDA Graphs.
  7. `torch.set_float32_matmul_precision('high')`  — TF32 for fp32 matmuls.
  8. `torch.compiler.cudagraph_mark_step_begin()` at the top of each training iteration  — fixes CUDA Graph capture-and-replay across steps.
  9. **Global** `torch.autocast(device_type='cuda', dtype=torch.bfloat16).__enter__()`  inserted before the training loop — UNet forward + residual finite-diffs both run under bf16.
- **Throughput** (tqdm smoothed): ~22-26 it/s post-warmup (compile takes ~2 min). Settles around **22.4 it/s**.
- **GPU:** 57-68% time-util (kernels finish much faster in bf16, so dispatch gaps are a bigger fraction of total time → utilization metric *drops* even though true work-rate is up).
- **VRAM:** 5.7-8.9 GB resident (bf16 halves the activation memory).
- **Observation re convergence (live, mid-run):** opt3's early-iters loss drops *faster* than vanilla's but is showing a **higher convergence floor** at later iters. Suspected cause: the **global autocast** is applying bf16 to the residual finite-difference computation in `src/residuals_darcy.py`, where small differences of similar-magnitude numbers fall below bf16's 7-bit mantissa precision floor. Vanilla in fp32 doesn't hit that floor until ~1e-7.

### Failure modes seen during opt3 development
- `cur_batch = next(dl).to(device, non_blocking=True)` without `pin_memory=True` in DataLoader → the async hint is silently ignored.
- Several CUDA-illegal-memory-access crashes at iter 0 validation when training with `bf16 autocast + torch.compile(reduce-overhead) + EMA weight-swap (line 183) + train/eval mode toggling (line 182)`. Workaround that worked once but stochastic: skip iter 0 validation (`if iteration > 0 and iteration % test_eval_freq == 0`). Workaround that's reliable: drop `torch.compile`, or drop autocast.
- `RuntimeError: GET was unable to find an engine to execute this computation` — cuDNN couldn't dispatch a bf16 conv shape on Blackwell sm_120 in this nightly. No clean workaround; switch to fp32 conv.

## opt4 (planned, post-opt3-completion)

Target diagnosis: confirm the bf16 floor on the residual is the cap, by scoping autocast narrower.

- **Patches** (vs opt3):
  - **Remove** global `torch.autocast(...).__enter__()` at line 158.
  - **Add** scoped autocast wrapper inserted **before** `torch.compile`:
    ```python
    class _BF16Wrapper(torch.nn.Module):
        def __init__(self, m): super().__init__(); self.m = m
        def forward(self, *a, **k):
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                return self.m(*a, **k)
    model = _BF16Wrapper(model)
    model = torch.compile(model, mode='reduce-overhead')
    ```
    The UNet's conv2d/matmul ops still hit tensor cores in bf16, but the **residual finite-difference operators in `src/residuals_darcy.py`** run on the fp32 output of the model — no bf16 quantization on the gradient signal.
  - **Disable** the per-20k-iter sample artifacts to eliminate the ~10-min CPU/I-O pause: `save_output = False`, `create_gif = False`. The wandb residual metric still gets logged from `p_sample_loop`'s output; only the PNG/CSV writes are skipped.
- Expected: similar throughput to opt3 (~22 it/s), but test-loss should keep descending past opt3's plateau — matching or beating vanilla's late-iter loss.

(Numbers filled in after the run.)

## Methodology notes

- **`it/s` mean is sampled, not derived from tqdm's smoothing.** Polling reads the latest tqdm iter counter every 30 s; we compute `Δiter / 30`. Tqdm's own `XX.XX it/s` is an exponentially smoothed rate which lags real-time changes — both numbers usually agree to within ~2%.
- **"std/mean < 1% means no jumping"** is a rough rule. We saw 0.6% on vanilla, ~2-3% on opt-runs. Sample-step pauses (every 20k iters in opt3) show up as a single 0-window followed by a catch-up window; pre/post they don't affect the steady-state estimate.
- **VRAM peak** quoted is `nvidia-smi --query-gpu=memory.used --format=csv` at steady state, not the peak across the run. We're using the workstation-edition Blackwell where the 97.9 GB VRAM is far above what any of these configs need.
- **bf16 ≠ fp16.** bf16 has fp32's exponent range with 7 mantissa bits; fp16 has narrower range with 10 mantissa bits. We use bf16 because it doesn't need a loss-scaler.
