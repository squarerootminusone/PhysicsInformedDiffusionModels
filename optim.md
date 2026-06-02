# Optimization landscape for PIDM training

Notes from profiling PIDM-ME training on Blackwell (RTX PRO 6000, sm_120). The workload is **dispatch-bound** — small UNet (dim=32), batch 64, 64×64 spatial, many small kernels per step, Python loop and CUDA launch overhead dominate. nvidia-smi reports ~89% util solo, but combined throughput when two jobs share one GPU is essentially identical to solo throughput, which is the textbook signature of "second process fills the dispatch gaps left by the first."

Three tiers ordered by effort-to-payoff ratio.

## Tier 1 — likely big wins, low effort

### 1. `torch.compile(model, mode='reduce-overhead')`
- One line after `model = Unet3D(...).to(device)` in `main.py:124`.
- `reduce-overhead` uses CUDA Graphs to capture-and-replay the kernel sequence without Python overhead.
- Especially good for small models with many tiny kernels (our exact situation).
- First few iterations slower (compilation), then steady-state much faster.
- Expected: 2-5× on dispatch-bound code. Less if the model already spends meaningful time in matmul.

### 2. `optim.Adam(..., fused=True)`
- One-flag change at `main.py:143`.
- Single CUDA kernel for all param updates instead of one per parameter tensor.
- Trivial win. Requires all params on CUDA, which they are.

### 3. Dispatch sync cleanup
- `cur_batch = next(dl).to(device, non_blocking=True)` at `main.py:159`. Combined with pinned-memory DataLoader, lets the host-to-device copy overlap with the previous iteration.
- `optimizer.zero_grad(set_to_none=True)` at `main.py:163`. Skips the zero-fill kernel — sets `.grad = None` instead.
- `pbar.set_description(loss.item())` is **already** gated by `iteration % log_freq == 0` (lines 167-169), so the `.item()` sync only happens every 20 iters. No change needed.

### 4. bf16 autocast on the model forward
- `with torch.autocast(device_type='cuda', dtype=torch.bfloat16):` around the UNet forward call.
- Halves memory traffic; Blackwell has strong bf16 throughput.
- **Risk for PIDM specifically:** the residual computation in `src/residuals_darcy.py` uses finite differences via `findiff`. fp32 is safer there. Scope the autocast to the UNet's forward inside `src/denoising_utils.py:model_estimation_loss` — leave the residual path in fp32.

## Tier 2 — medium effort

### 5. Manual CUDA Graphs
- Only beats `torch.compile` reduce-overhead when shapes are dynamic. Our training shapes are fixed (batch=64, 64×64), so `torch.compile` should already capture this.
- Worth trying if `torch.compile` regresses or has graph breaks.

### 6. Larger batch size
- We use 64; peak VRAM ~15 GB on 97 GB Blackwell.
- Going to 128-256 cuts iterations needed for same gradient signal. Need lr adjustment (linear or sqrt scaling).
- Diminishing returns past a point because the model's compute time grows linearly with batch but launch overhead doesn't, so the dispatch-bound region shrinks.

### 7. Channels-last memory format
- `model.to(memory_format=torch.channels_last)` and inputs the same.
- Historically 10-30% for image conv models on Ampere+.
- Compose with #1 — `torch.compile` should handle the layout change cleanly.

## Tier 3 — speculative or out-of-scope

### 8. `nsys profile` / `ncu`
- Actually quantify dispatch gaps before guessing.
- The most rigorous next step if Tier 1 results are ambiguous.

### 9. Diffusion-step distillation
- Train PIDM-ME, distill to consistency model or DDIM with ~10 steps.
- Huge inference win, but doesn't help training throughput. Out of scope for this benchmark.

### 10. Smaller backbone
- Defeats the reproduction goal. Skip.

## Suggested experiment order

1. Vanilla 20-min benchmark → steady-state it/s, mean±std.
2. Apply Tier 1 items 1+2+3 together (they don't interact) → 20-min benchmark → compare.
3. If that's < 2× speedup, profile with `nsys` to see what's still leaving gaps.
4. If 2× or more, also try Tier 1 item 4 (bf16 autocast) carefully — verify residual MAE didn't regress.
5. Consider Tier 2 items 6 (larger batch) only if you also care about wall-clock to convergence, not raw it/s.

## Reference points

| Configuration | it/s (solo, Blackwell Server Ed.) | Notes |
|---|---|---|
| Vanilla, batch 64 | ~14.5 | dispatch-bound at 89% util |
| Two jobs sharing 1 GPU | ~7.3 each, ~14.6 combined | additivity confirms dispatch is the constraint |

Update this with Workstation-edition numbers after benchmarking.
