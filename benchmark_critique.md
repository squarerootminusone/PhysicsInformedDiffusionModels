# Benchmarks in physics-informed generative modeling (2023–2026): survey, failure modes, and a critique/improvement proposal for PIDM

Compiled 2026-06-11 from a verified deep-research pass (18 sources fetched, 90 claims extracted, 25 adversarially
verified: 24 confirmed / 1 refuted). Every claim below traces to a verbatim-verified source statement; caveats at the end.

## 1. Benchmark landscape, categorized

### By physics-residual type

| Category | Canonical benchmarks | Ground truth | Iteration cost | Upsides | Downsides |
|---|---|---|---|---|---|
| **Elliptic / smooth static** (Darcy ∇·(K∇p)=f, Poisson, Helmholtz) | PIDM Darcy 64×64; PDEBench Darcy | Exact FD/solver solution | PIDM defaults: 300k iters ≈ 23–26 h on A100 (≈3 h on optimized Blackwell stack) | Exact residual operator; stable training; well understood | Smooth and "easy"; residual gameable by amplitude shrink (PBFM); PIDM evals with the *same* fd_acc=2 scheme that generated the data → circularity |
| **FEM equilibrium / solid mechanics** (Ku−f topology opt) | PIDM mechanics (Mazé & Ahmed dataset), TopoDiff | Exact algebraic residual + FEM solve for compliance | 600k iters @ batch 6 ≈ 60 h A100 (matrix-free residual makes batch 32+ feasible) | Exact residual; engineering-relevant metrics (compliance, VF, floating material) | Slowest iteration of all; metric-dependent comparisons (CE vs VFE trade-off); baselines need post-processing parity (DOM compared without SIMP) |
| **Hyperbolic / advective** (Burgers, advection, KS) | APEBench (procedural, JAX ETDRK), PDEBench | Deterministic pseudo-spectral solver; data regenerated in seconds | <1 h single-GPU ablations possible (50 trajectories of 2D KS 128² generated in <1 s) | Fast iteration; rollout + Sobolev/Fourier-band metrics built in; discriminative (shocks, instability) | Periodic domains, semi-linear PDEs only; designed for autoregressive emulation — needs adaptation for unconditional generation |
| **Chaotic incompressible NS** (Kolmogorov flow, decaying turbulence) | Shu et al. 2023 data, PG-Diff/SG-Diff suite, PDEBench NS | Converged solver (no analytic GT); statistical targets (spectra) | Medium: 256² data heavy but public; training hours-scale | Discriminative for small scales; spectra/statistics natural metrics | Residual noisy under chaos; one-step metrics hide rollout divergence (APEBench's emulator beat the solver for ~13 steps then diverged) |
| **Solver-generated coarse-input reconstruction** | SG-Diff (formerly PG-Diff, arXiv:2504.04375) | High-fid 4096²→256² targets; low-fid from *actual coarse solver rollouts* | Medium (public, 0.44 s/frame data gen at 64²) | Realistic task; exposes models that only work on downsampled inputs | Newer, less adopted |

### By task type
- **Unconditional/joint generation** (PIDM Darcy: joint (p,K)) — needs distributional metrics, most exposed to residual gaming.
- **Super-resolution/reconstruction** (Shu et al., SG-Diff) — input-representativeness pitfall (downsampled vs solver-generated).
- **Inverse/coefficient inference** (DiffusionPDE, PISD) — residual + observation consistency; guidance cost dominates.
- **Forward emulation/rollout** (PDEBench, PDEArena, APEBench, The Well) — rollout stability is the metric that matters; one-step MSE misleads.

## 2. Documented benchmark/metric failure modes (all citation-backed)

1. **Residual hackable by amplitude shrinkage** — PBFM (arXiv:2506.08604, ICLR 2026): PIDM-ME attains the lowest Darcy residual (0.022) with pressure "constrained within the narrow range ±0.2", "artificially reduc[ing] the residual due to the lower magnitude of P, rather than improved fidelity"; Wasserstein distance 3.103 vs PBFM's 0.838-residual/0.138-WD Pareto point. (Caveat: competitor reimplementation; targets the deterministic PIDM-ME variant; ±0.2 phrasing only in arXiv v1, v4 uses WD numbers.)
2. **Coefficient-shrinkage degeneracy in joint (p,K) objectives** — arXiv:2508.09156 (Appendix C, stated for Darcy): "a joint optimiser can reduce the loss merely by driving the parameter toward smaller values"; they normalize each residual by ∫K over its support. Directly applicable to PIDM's joint Darcy setup.
3. **Physics constraints collapse diversity if unregularized** — arXiv:2508.09156 Table 2: permeability ensemble diversity 7.01e-1 → 4.29e-1 without a diversity regularizer (recovered at 6.97e-1 with λ_f=0.1).
4. **Grid-based FD residuals structurally ill-defined under diffusion noise** — PISD (arXiv:2602.09708, ICML 2026): diffusion noise → spatial white noise in the continuum limit; FD residuals blow up, forcing late-stage-only guidance; explicitly places PIDM in this residual class (training-time variant). (Do NOT cite PISD's benchmark composition — that claim was refuted 0-3.)
5. **Tweedie/Jensen's-gap bias in training-time residuals** — arXiv:2508.09156: estimating the expected residual of the denoised sample needs multiple reverse trajectories; PIDM's single-pass x0-prediction "introduces bias, particularly in the final denoising steps". PBFM addresses the same gap with unrolling + conflict-free gradients.
6. **Downsampled low-fidelity inputs are unrepresentative** — SG-Diff: downsampled inputs "inherently ha[ve] more information compared to solver-generated low-fidelity data in reality"; SOTA models degrade on genuine coarse-solver inputs.
7. **One-step metrics hide rollout instability; MSE hides spectral bias** — APEBench (arXiv:2411.00180): rollout metrics + Sobolev/Fourier-band metrics; PDEBench: "RMSE on test data is not a good proxy … in particular in turbulent and non-smooth regimes."
8. **Weak baselines + reporting bias are the field norm** — McGreivy & Hakim (Nature MI 2024): 79% (60/76) of ML-for-fluid-PDE papers claiming wins used weak baselines (rules: compare at equal accuracy *or* equal runtime, against an *efficient* numerical method); 94.8% of abstracts report only positive results.

## 3. Proposal (NOT actioned): how to criticize the PIDM paper

1. **Residual-metric hackability.** The headline "two orders of magnitude" residual reduction (reproduced locally: R_MAE 0.0127 vs CoCoGen 1.08, ~85×) is an unnormalized residual. PBFM's PIDM-ME evidence + the coefficient-shrinkage degeneracy show the metric can be moved by amplitude/coefficient compression without better physics. Test: re-report residuals normalized by field amplitude (R/std(p)) and by ∫K; measure generated p,K amplitude statistics vs data. (Our local check on the 50k dim64 checkpoint is the first datapoint; pending the 300k run.)
2. **Evaluation circularity.** model.yaml: "fd_acc: 2 … keep at 2 to be consistent with training data" — the eval residual uses the data-generating discretization. (Local evidence agrees: fd_acc=4/6-trained models look 10× worse at the fd2 reference.) A fair instrument would be an independent/higher-order or spectral discretization, or solver re-solve error: solve the PDE with generated K and compare p.
3. **Missing distributional metrics.** No Wasserstein/spectra/diversity/coverage numbers in the paper — exactly the gap PBFM exploited. Any residual claim should be a (residual, distribution) Pareto point, not a scalar.
4. **Baseline strength (McGreivy–Hakim audit).** CoCoGen comparison: different enforcement point (sampling vs training) at unequal compute? TopoDiff-G (239M) and DOM (121M) vs PIDM: DOM compared *without* its SIMP post-processing; topology-opt "win" is metric-selective (median CE 0.06 vs 0.83 won; VFE 2.25 vs 1.49/1.52 lost).
5. **Representativeness & iteration cost.** Both benchmarks are smooth/static (elliptic, equilibrium) — the easiest residual classes — and need ~24 h / ~60 h default training, discouraging ablation. Neither stresses advective/chaotic dynamics where physics-informed claims would be most discriminative.

## 4. Proposal (NOT actioned): how to improve the paper/method

**Metrics (cheap, additive):**
- Amplitude-normalized residual (R/std(p)) and coefficient-normalized residual (∫K-normalized per arXiv:2508.09156), always alongside raw R.
- Sliced-Wasserstein (already implemented in this repo) or full WD on field distributions + energy spectra + ensemble diversity; report residual-vs-distribution **Pareto plots** (PBFM format).
- Solver re-solve consistency: FEM/FD solve with generated K → compare to generated p (decouples eval from the training residual operator entirely).
- For mechanics: report CE *jointly* with VFE/floating-material; rerun DOM with SIMP post-processing.

**Method (ranked by effort):**
- Diversity regularizer (λ_f-style, arXiv:2508.09156) — guards the collapse failure mode; ablate at 50k.
- Conflict-free gradient combination (ConFIG, used by PBFM) replacing the hand-tuned c_residual — removes the manual balance that creates the hackability incentive.
- Unrolled final denoising steps when computing the training residual — mitigates the documented Tweedie bias.

**Benchmarks (additive, fast-iteration):**
- One APEBench-procedural case (2D KS or Kolmogorov snapshots at 64–128², data generated in seconds) as a distribution-targeted generation benchmark with spectra + rollout-consistency metrics; trains a PIDM-class model <1 h on one GPU with this repo's optimized stack.
- Optionally an SG-Diff-style solver-generated coarse-input reconstruction variant to address representativeness.

## Caveats (verbatim from verification)
- The hackability evidence is from PBFM's own reimplementation of the PIDM-ME variant; no independent replication or author rebuttal found; do not conflate PIDM-ME with full sampling-based PIDM.
- Three critique sources (PBFM, 2508.09156, PISD) are competitor method papers.
- PIDM's zero-mean pressure correction is mean-invariant for the residual — it is NOT the shrinkage mechanism (that constituent claim passed only 2-1).
- McGreivy–Hakim surveyed ML-vs-numerical-solver claims (pre-2024, fluids); applying it to PIDM's ML-vs-ML baselines is an analogy.
- No surviving verified claims on PDEArena, The Well, BubbleML; PISD's benchmark composition claim was refuted (0-3) — don't cite it.

## Open questions
1. Does full sampling-based PIDM (not just ME) show amplitude compression? → directly testable here with `eval_amplitude.py` (pending the 300k run).
2. What does the 2-orders-of-magnitude claim look like under normalized residuals and an independent discretization?
3. Can an APEBench-procedural Kolmogorov/KS generation benchmark with McGreivy-Hakim-clean baselines train in <1 h on this repo's stack?
4. Do the topology-opt comparisons survive DOM+SIMP and joint CE/VFE reporting?

## Key sources
PIDM arXiv:2403.14404 · PBFM arXiv:2506.08604 · SG-Diff arXiv:2504.04375 · arXiv:2508.09156 · PISD arXiv:2602.09708 · McGreivy & Hakim, Nature MI 2024 (arXiv:2407.07218) · APEBench arXiv:2411.00180 · PDEBench arXiv:2210.07182 · The Well arXiv:2412.00568 · PDEArena arXiv:2209.15616 · BubbleML arXiv:2307.14623
