import os, yaml
import threading, queue
from contextlib import nullcontext
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from tqdm import tqdm
from src.data_utils import *
from torch.utils.data import DataLoader
from src.denoising_utils import *
from src.unet_model import Unet3D
from src.residuals_darcy import ResidualsDarcy
from src.residuals_mechanics_K import ResidualsMechanics

try:
    import optuna
except ImportError:
    optuna = None

# --- opt3-no-bf16 recipe: full-fp32 matmuls (no TF32, no bf16 autocast) for maximum numerical precision ---
torch.set_float32_matmul_precision('highest')


# Default run/hyperparameters. An Optuna objective (or any caller) overrides any of these
# by passing an `overrides` dict to train(). `None` means "use the per-gov_eqs / yaml default".
DEFAULTS = dict(
    name='run_1',
    config_path='model.yaml',
    wandb_track=False,
    compile_model=True,
    compile_mode='default',     # 'default' (Inductor, no cudagraphs) is safe across many trials;
                                # 'reduce-overhead' (cudagraphs) is faster but breaks on the 2nd
                                # compiled model in one process (RNG offset / graph-capture errors)
    bf16_train=False,           # wrap the training forward in bf16 autocast (eval stays fp32)
    async_eval=True,            # run the periodic validation eval off the training critical path
    # --- tunable hyperparameters ---
    lr=1.0e-4,
    lr_schedule=None,           # None or 'plateau': ReduceLROnPlateau on a rolling loss average
    lr_plateau_factor=0.5,      # lr *= factor when the rolling loss plateaus
    lr_plateau_patience=5,      # scheduler steps (of lr_sched_freq each) w/o improvement before decay
    lr_plateau_threshold=1e-3,  # relative-improvement threshold for "no improvement"
    lr_min=1e-6,
    lr_window=50,               # rolling average over this many logged-loss samples
    lr_sched_freq=500,          # iterations between scheduler.step(rolling_avg)
    grad_clip=1.0,
    ema_decay=0.99,
    c_data=None,                # None -> from yaml config
    c_residual=None,            # None -> from yaml config
    diff_steps=None,            # None -> from yaml config
    fd_acc=None,                # None -> from yaml config
    model_dim=None,             # None -> per-gov_eqs default (darcy 32 / mechanics 128)
    batch_size=None,            # None -> per-gov_eqs default
    batch_schedule=None,        # optional {iter: batch_size} ramp (e.g. {0:64, 22000:128, 35000:256})
    fd_acc_schedule=None,       # optional {iter: fd_acc} ramp, darcy only (e.g. {25000: 4})
    c_residual_schedule=None,   # optional {iter: c_residual} (e.g. recalibrate at the fd_acc switch)
    train_iterations=None,      # None -> per-gov_eqs default (HPO callers pass a small value)
    # --- evaluation cadence ---
    test_eval_freq=500,
    sample_freq=20000,          # HPO: set >= train_iterations to skip the heavy GPU sampler
    final_sample=True,          # run one sample+checkpoint at the last iteration (HPO: False)
    save_final_checkpoint=False, # save the final EMA model checkpoint (for later fp64 eval / reuse)
    sample_eval_freq=None,      # if set, async p_sample_loop residual_mean_abs_samples every N steps
    ema_start=1000,
)


class AsyncEvaluator:
    """Runs the periodic validation eval on a *separate* (uncompiled) model copy and its own
    CUDA stream, inside a background thread, so the training loop never blocks on it.

    Training submits an EMA-weight snapshot + a CUDA event; the worker waits the event,
    loads the snapshot into its eval model, computes residual_mean_abs_test, logs the
    metrics, reports to the Optuna trial, and raises a prune flag. The training thread polls
    that flag (cheap bool) and raises optuna.TrialPruned itself.

    The heavy 20k sampler is intentionally NOT routed here: it is GPU-bound, so overlapping
    it with training on a single GPU buys no wall-clock and only complicates correctness.
    """

    def __init__(self, eval_model, eval_residuals, eval_diffusion, dl_valid,
                 loss_kwargs, log_fn):
        self.eval_model = eval_model.eval()
        self.eval_residuals = eval_residuals
        self.diffusion = eval_diffusion
        self.dl_valid = dl_valid
        self.loss_kwargs = loss_kwargs
        self.log_fn = log_fn
        self.stream = torch.cuda.Stream() if torch.cuda.is_available() else None
        self.q = queue.Queue(maxsize=1)     # at most one eval in flight -> training never waits
        self.results = queue.Queue()        # worker -> main thread (worker never touches wandb/Optuna)
        self.best = float('inf')
        self.skipped = 0
        self._exc = None
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def submit(self, iteration, snapshot, ready_event):
        """Non-blocking: enqueue an eval job, or skip it if the worker is still busy."""
        try:
            self.q.put_nowait((iteration, snapshot, ready_event))
        except queue.Full:
            self.skipped += 1               # keep training non-blocking; a later eval still runs

    def _load_snapshot(self, snapshot):
        # EMA shadow keys carry torch.compile's '_orig_mod.' prefix; the eval model is uncompiled.
        clean = {k.replace('_orig_mod.', ''): v for k, v in snapshot.items()}
        self.eval_model.load_state_dict(clean, strict=False)

    def _worker(self):
        try:
            while True:
                item = self.q.get()
                if item is None:
                    break
                iteration, snapshot, ready_event = item
                ctx = torch.cuda.stream(self.stream) if self.stream is not None else nullcontext()
                with ctx:
                    if ready_event is not None:
                        ready_event.wait(self.stream)       # order after the training-thread clone
                    self._load_snapshot(snapshot)
                    batch = next(self.dl_valid).to(device, non_blocking=True)
                    # fp32 eval: residual stencils must not be cast to bf16
                    with torch.autocast(device_type='cuda', enabled=False):
                        loss_test, data_loss_test, residual_loss_test, _, _ = \
                            self.diffusion.model_estimation_loss(
                                batch, residual_func=self.eval_residuals, **self.loss_kwargs)
                if self.stream is not None:
                    self.stream.synchronize()
                residual_loss_test = float(residual_loss_test)
                # compute-only: hand results to the main thread; never log/report from here
                self.results.put((iteration, {'loss_test': float(loss_test.detach()),
                                              'loss_data_test': float(data_loss_test),
                                              'residual_mean_abs_test': residual_loss_test},
                                  residual_loss_test))
        except Exception as e:        # surface to the main thread via drain()
            self._exc = e

    def drain(self):
        """Called from the MAIN thread: log queued eval results, update best, re-raise errors."""
        if self._exc is not None:
            raise self._exc
        while True:
            try:
                iteration, data, rlt = self.results.get_nowait()
            except queue.Empty:
                break
            self.log_fn(data, step=iteration)
            self.best = min(self.best, rlt)

    def close(self):
        try:
            self.q.put_nowait(None)
        except queue.Full:
            self.q.put(None)
        self.thread.join(timeout=120)


class SampleEvaluator:
    """Async generative-sample eval (Darcy). Periodically runs p_sample_loop on a separate
    uncompiled model copy + its own CUDA stream in a background thread, fed an EMA weight
    snapshot, and logs residual_mean_abs_samples / residual_median_abs_samples -- the residual
    on actually-generated samples (catches overfitting that the cheap forward eval misses).
    Tracks the best (min) residual_mean_abs_samples for use as the study objective.

    GPU-bound on a single card, so it overlaps but does not run for free; the bounded queue
    skips a trigger only if a previous sampling pass is still running."""

    def __init__(self, eval_model, eval_residuals, eval_diffusion, sample_shape, log_fn,
                 use_dynamic_threshold=False, M_correction=0, N_correction=0, correction_mode='xt'):
        self.eval_model = eval_model.eval()
        self.eval_residuals = eval_residuals
        self.diffusion = eval_diffusion
        self.sample_shape = sample_shape
        self.log_fn = log_fn
        self.use_dynamic_threshold = use_dynamic_threshold
        self.M_correction = M_correction
        self.N_correction = N_correction
        self.correction_mode = correction_mode
        self.stream = torch.cuda.Stream() if torch.cuda.is_available() else None
        self.q = queue.Queue(maxsize=1)
        self.results = queue.Queue()        # worker -> main thread (worker never touches wandb)
        self.best = float('inf')
        self.skipped = 0
        self._exc = None
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def submit(self, iteration, snapshot, ready_event):
        try:
            self.q.put_nowait((iteration, snapshot, ready_event))
        except queue.Full:
            self.skipped += 1

    def _load_snapshot(self, snapshot):
        clean = {k.replace('_orig_mod.', ''): v for k, v in snapshot.items()}
        self.eval_model.load_state_dict(clean, strict=False)

    def _worker(self):
        try:
            while True:
                item = self.q.get()
                if item is None:
                    break
                iteration, snapshot, ready_event = item
                ctx = torch.cuda.stream(self.stream) if self.stream is not None else nullcontext()
                with ctx:
                    if ready_event is not None:
                        ready_event.wait(self.stream)
                    self._load_snapshot(snapshot)
                    # fp32 sampling + residual: bf16 stencils would inflate the metric
                    with torch.autocast(device_type='cuda', enabled=False):
                        output = self.diffusion.p_sample_loop(
                            None, self.sample_shape, save_output=True, surpress_noise=True,
                            use_dynamic_threshold=self.use_dynamic_threshold,
                            residual_func=self.eval_residuals, eval_residuals=True,
                            return_optimizer=False, return_inequality=False,
                            M_correction=self.M_correction, N_correction=self.N_correction,
                            correction_mode=self.correction_mode)
                        residual = output[1]['residual']
                        residual = residual.abs().mean(dim=tuple(range(1, residual.ndim)))
                if self.stream is not None:
                    self.stream.synchronize()
                arr = residual.detach().cpu().numpy()
                mean_abs = float(np.nanmean(arr))
                # compute-only: hand results to the main thread; never log from here
                self.results.put((iteration, {'residual_mean_abs_samples': mean_abs,
                                              'residual_median_abs_samples': float(np.nanmedian(arr))},
                                  mean_abs))
        except Exception as e:
            self._exc = e

    def drain(self):
        """Called from the MAIN thread: log queued sample-eval results, update best, re-raise errors."""
        if self._exc is not None:
            raise self._exc
        while True:
            try:
                iteration, data, mean_abs = self.results.get_nowait()
            except queue.Empty:
                break
            self.log_fn(data, step=iteration)
            self.best = min(self.best, mean_abs)
            print(f'sample-eval at iteration {iteration}: residual_mean_abs_samples={mean_abs:.3e}')

    def close(self):
        try:
            self.q.put_nowait(None)
        except queue.Full:
            self.q.put(None)
        self.thread.join(timeout=600)


def train(overrides=None, trial=None):
    """Train PIDM and return the best validation residual_mean_abs_test (to minimize).

    overrides: dict of DEFAULTS keys to override (hyperparameters / run config).
    trial:     an optuna.Trial for pruning + reporting (or None for a plain run).
    """
    p = {**DEFAULTS, **(overrides or {})}
    name = p['name']
    wandb_track = p['wandb_track']

    config = yaml.safe_load(Path(p['config_path']).read_text())
    # fold tunable overrides back into the yaml-sourced config where provided
    for key in ('c_data', 'c_residual', 'diff_steps', 'fd_acc'):
        if p[key] is not None:
            config[key] = p[key]

    # diffusion parameters
    if config['x0_estimation'] == 'mean':
        use_ddim_x0 = False
    elif config['x0_estimation'] == 'sample':
        use_ddim_x0 = True
    ddim_steps = config['ddim_steps']
    residual_grad_guidance = config['residual_grad_guidance']
    correction_mode = config['correction_mode']
    M_correction = config['M_correction']
    N_correction = config['N_correction']
    gov_eqs = config['gov_eqs']
    if gov_eqs != 'darcy' and (residual_grad_guidance or N_correction > 0 or M_correction > 0):
        raise ValueError('Gradient guidance and CoCoGen only implemented for Darcy flow study.')
    fd_acc = config['fd_acc']
    c_data = config['c_data']
    c_residual = config['c_residual']
    c_ineq = config['c_ineq']
    lambda_opt = config['lambda_opt']
    diff_steps = config['diff_steps']
    use_dynamic_threshold = False
    self_condition = False

    # evaluation params
    test_eval_freq = p['test_eval_freq']
    sample_freq = p['sample_freq']
    ema_start = p['ema_start']
    ema = EMA(p['ema_decay'])
    topopt_eval = True
    use_double = False
    no_samples = 8
    save_output = True
    eval_residuals = True
    create_gif = False

    # training parameters and datasets
    if gov_eqs == 'darcy':
        input_dim = 2
        output_dim = 2
        pixels_at_boundary = True
        domain_length = 1.
        reverse_d1 = True
        data_paths = ('./data/darcy/train/p_data.csv', './data/darcy/train/K_data.csv')
        data_paths_valid = ('./data/darcy/valid/p_data.csv', './data/darcy/valid/K_data.csv')
        bcs = 'none'
        pixels_per_dim = 64
        return_optimizer = False
        return_inequality = False
        ds = Dataset(data_paths, use_double=use_double)
        ds_valid = Dataset(data_paths_valid, use_double=use_double)
        default_batch_size = 16 if use_ddim_x0 else 64
        sigmoid_last_channel = False
        default_iterations = 300000
        default_dim = 32
    elif gov_eqs == 'mechanics':
        input_dim = 2
        output_dim = 3
        pixels_at_boundary = True
        reverse_d1 = True
        data_paths = ('./data/mechanics/train/fields/')
        data_paths_valid = ('./data/mechanics/test/valid/fields/')
        bcs = 'none'
        pixels_per_dim = 64
        return_optimizer = True
        return_inequality = True
        ds = Dataset_Paths(data_paths, use_double=use_double)
        ds_valid = Dataset_Paths(data_paths_valid, use_double=use_double)
        default_batch_size = 4 if use_ddim_x0 else 6
        sigmoid_last_channel = True
        default_iterations = 600000
        default_dim = 128
    else:
        raise ValueError('Unknown governing equations.')

    train_batch_size = p['batch_size'] or default_batch_size
    train_iterations = p['train_iterations'] or default_iterations
    model_dim = p['model_dim'] or default_dim
    config['model_dim'] = model_dim   # record in the saved model.yaml so eval can rebuild correctly

    if use_double:
        torch.set_default_dtype(torch.float64)

    # optional schedules: JSON may stringify dict keys -> normalize to int
    batch_schedule = {int(k): v for k, v in (p['batch_schedule'] or {}).items()}
    fd_acc_schedule = {int(k): int(v) for k, v in (p['fd_acc_schedule'] or {}).items()}
    c_residual_schedule = {int(k): float(v) for k, v in (p['c_residual_schedule'] or {}).items()}

    def make_train_dl(bs):
        return cycle(DataLoader(ds, batch_size=bs, shuffle=False,
                                num_workers=4, pin_memory=True, persistent_workers=True))

    cur_bs = batch_schedule.get(0, train_batch_size)
    dl = make_train_dl(cur_bs)
    dl_valid = cycle(DataLoader(ds_valid, batch_size=train_batch_size, shuffle=False,
                                num_workers=2, pin_memory=True, persistent_workers=True))

    # diffusion utils
    diffusion_utils = DenoisingDiffusion(diff_steps, device, residual_grad_guidance)

    def build_model():
        if gov_eqs == 'darcy':
            m = Unet3D(dim=model_dim, channels=output_dim,
                       sigmoid_last_channel=sigmoid_last_channel)
        else:
            m = Unet3D(dim=model_dim, channels=output_dim + 3 + 4, out_dim=output_dim,
                       sigmoid_last_channel=sigmoid_last_channel)
        return m.to(device)

    def build_residuals(m, fd=None):
        fd = fd if fd is not None else fd_acc
        if gov_eqs == 'darcy':
            return ResidualsDarcy(model=m, fd_acc=fd, pixels_per_dim=pixels_per_dim,
                                  pixels_at_boundary=pixels_at_boundary, reverse_d1=reverse_d1,
                                  device=device, bcs=bcs, domain_length=domain_length,
                                  residual_grad_guidance=residual_grad_guidance,
                                  use_ddim_x0=use_ddim_x0, ddim_steps=ddim_steps)
        return ResidualsMechanics(model=m, pixels_per_dim=pixels_per_dim,
                                  pixels_at_boundary=pixels_at_boundary, device=device, bcs=bcs,
                                  no_BC_folder='./data/mechanics/solidspy_k_no_BC/',
                                  topopt_eval=topopt_eval, use_ddim_x0=use_ddim_x0,
                                  ddim_steps=ddim_steps)

    # model
    model = build_model()
    # --- opt3-no-bf16: torch.compile (Inductor + CUDA Graphs) ---
    if p['compile_model']:
        model = torch.compile(model, mode=p['compile_mode'])
    ema.register(model)
    num_params = sum(pp.numel() for pp in model.parameters() if pp.requires_grad)
    print(f'Number of trainable parameters: {num_params}')

    # residual computation based on governing equations
    residuals = build_residuals(model)
    optimizer = optim.Adam(model.parameters(), lr=p['lr'], fused=True)  # opt3-no-bf16: fused Adam

    # optional LR schedule: ReduceLROnPlateau driven by a rolling average of the training loss
    scheduler = None
    loss_window = None
    if p['lr_schedule'] == 'plateau':
        from collections import deque
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=p['lr_plateau_factor'],
            patience=p['lr_plateau_patience'], threshold=p['lr_plateau_threshold'],
            min_lr=p['lr_min'])
        loss_window = deque(maxlen=p['lr_window'])

    if wandb_track:
        import wandb
        wandb.init(project='pi_diffusion', name=name)
        # Log ALL hyperparameters. `config={...}` passed to init was not persisting (empty
        # config in the run), so set it explicitly via config.update on the resolved params:
        # p (run/hyperparams) + config (yaml-sourced, with the actually-used c_residual etc.)
        # + derived values. Also mirror the key tunables into summary as a recovery backstop.
        full_hparams = {**p, **config,
                        'train_batch_size': train_batch_size,
                        'train_iterations_resolved': train_iterations,
                        'model_dim_resolved': model_dim,
                        'num_params': num_params}
        wandb.config.update(full_hparams, allow_val_change=True)   # often empty in this env...
        for _k, _v in full_hparams.items():                        # ...so summary is the reliable store
            try:
                wandb.run.summary[f'hp/{_k}'] = _v
            except Exception:
                pass
        # Use an explicit 'iteration' x-axis so the async eval's (possibly out-of-order)
        # logs are not dropped by wandb's monotonic internal step counter.
        wandb.define_metric('iteration')
        wandb.define_metric('*', step_metric='iteration')

        # wandb.log is NOT safe to call concurrently from multiple threads; the main loop and
        # the async eval workers all log, so serialize every call behind one lock.
        _wandb_lock = threading.Lock()

        def log_fn(data, step):
            with _wandb_lock:
                wandb.log({**data, 'iteration': step})
    else:
        def log_fn(data, step=None):
            pass
    log_freq = 20

    output_save_dir = f'./trained_models/{name}'
    os.makedirs(output_save_dir, exist_ok=True)

    loss_kwargs = dict(c_data=c_data, c_residual=c_residual, c_ineq=c_ineq, lambda_opt=lambda_opt)

    # --- async validation evaluator (separate uncompiled model copy + its own CUDA stream) ---
    evaluator = None
    if p['async_eval']:
        eval_model = build_model()
        eval_residuals = build_residuals(eval_model)
        eval_diffusion = DenoisingDiffusion(diff_steps, device, residual_grad_guidance)
        dl_valid_async = cycle(DataLoader(ds_valid, batch_size=train_batch_size, shuffle=False,
                                          num_workers=1, pin_memory=True, persistent_workers=True))
        evaluator = AsyncEvaluator(eval_model, eval_residuals, eval_diffusion, dl_valid_async,
                                   loss_kwargs, log_fn)

    # --- async generative-sample evaluator (residual_mean_abs_samples every sample_eval_freq) ---
    sample_evaluator = None
    if p['sample_eval_freq'] and gov_eqs == 'darcy':
        s_model = build_model()
        s_residuals = build_residuals(s_model)
        s_diffusion = DenoisingDiffusion(diff_steps, device, residual_grad_guidance)
        sample_shape = (no_samples, output_dim, pixels_per_dim, pixels_per_dim)
        sample_evaluator = SampleEvaluator(s_model, s_residuals, s_diffusion, sample_shape, log_fn,
                                           use_dynamic_threshold=use_dynamic_threshold,
                                           M_correction=M_correction, N_correction=N_correction,
                                           correction_mode=correction_mode)
    elif p['sample_eval_freq']:
        print('[sample-eval] only implemented for gov_eqs=="darcy"; disabled')

    def sample_and_checkpoint(iteration):
        """Heavy periodic sampler + checkpoint. Synchronous & EMA-swapped on the training
        model (only at sample_freq cadence). Unchanged from the original sample block."""
        nonlocal no_samples
        model.eval()
        ema.ema(residuals.model)
        if gov_eqs == 'darcy':
            conditioning_input = None
            sample_shape = (no_samples, output_dim, pixels_per_dim, pixels_per_dim)
        elif gov_eqs == 'mechanics':
            cur_batch = next(dl_valid).to(device)
            if cur_batch.shape[0] < no_samples:
                no_samples = cur_batch.shape[0] # reduce no_samples to batch size
            sample_shape = (no_samples, output_dim, pixels_per_dim+1, pixels_per_dim+1)
            cur_batch = cur_batch[torch.randperm(cur_batch.shape[0], device = device)[:no_samples]]
            conditioning, x_0, bcs = torch.tensor_split(cur_batch, (3, 6), dim=1)
            conditioning_input = (conditioning, bcs, x_0)
            # save conditioning data for later evaluation
            cond_data = torch.cat((conditioning, x_0, bcs), dim=1)
            for cur_sample in range(no_samples):
                for channel_idx in range(cond_data.shape[1]):
                    os.makedirs(output_save_dir + f'/training/step_{iteration}/sample_{cur_sample}', exist_ok=True)
                    np.savetxt(output_save_dir + f'/training/step_{iteration}/sample_{cur_sample}/cond_channel_{channel_idx}.csv', cond_data[cur_sample, channel_idx].detach().cpu().numpy(), delimiter=',')

        output = diffusion_utils.p_sample_loop(conditioning_input, sample_shape,
                                save_output=save_output, surpress_noise=True,
                                use_dynamic_threshold=use_dynamic_threshold,
                                residual_func=residuals, eval_residuals = eval_residuals,
                                return_optimizer = return_optimizer, return_inequality = return_inequality,
                                M_correction = M_correction, N_correction = N_correction, correction_mode = correction_mode)

        if eval_residuals:
            seqs = output[0]
            residual = output[1]['residual']
            residual = residual.abs().mean(dim=tuple(range(1, residual.ndim))) # reduce to batch dim
            if return_optimizer:
                optimized_quant = output[1]['optimized_quant']
            if return_inequality:
                ineq = output[1]['inequality_quant']
        else:
            seqs = output

        output_save_dir_step = output_save_dir + f'/training/step_{iteration}/'
        os.makedirs(output_save_dir_step, exist_ok=True)

        labels = ['sample', 'model_output']
        for seq_idx, seq in enumerate(seqs):

            # NOTE: We here only evaluate the sample at the final timestep and skip model_output as this is identical (since no noise is applied in last step).
            if seq_idx == 1:
                continue

            seq = torch.stack(seq, dim=0)

            if len(seq.shape) == 6:
                seq = seq.squeeze(-3)

            last_preds = seq[-1].numpy()
            sel_samples = np.arange(len(last_preds))
            channels = np.arange(output_dim)

            for sel_sample in sel_samples:
                for sel_channel in channels:
                    last_pred = last_preds[sel_sample, sel_channel]
                    last_pred_normalized = (last_pred - last_pred.min()) / (last_pred.max() - last_pred.min()) # normalize to [0,1]

                    image = np.uint8(last_pred_normalized * 255)
                    fig, ax = plt.subplots()
                    ax.imshow(image, cmap='gray', vmin=0, vmax=255)
                    ax.axis('off')
                    if eval_residuals:
                        title = f'eq: {residual[sel_sample]:.2e}'
                        if return_optimizer:
                            title += f'\nopt: {optimized_quant[sel_sample]:.2f}'
                        if return_inequality:
                            title += f'\nineq: {ineq[sel_sample]:.2e}'
                        plt.title(title, color='green')
                    filename = labels[seq_idx] + '_sample_' + str(sel_sample) + '_' + str(sel_channel) + '.png'
                    plt.savefig(output_save_dir_step + filename, bbox_inches='tight', pad_inches=0)
                    plt.close(fig)

                    os.makedirs(output_save_dir_step + f'/sample_{sel_sample}/', exist_ok=True)
                    np.savetxt(output_save_dir_step + f'/sample_{sel_sample}/' + labels[seq_idx] + '_' + str(sel_channel) + '.csv', last_pred, delimiter=',')

                    if create_gif:
                        sel_seq = seq[:, sel_sample, sel_channel].detach().cpu().numpy()
                        image_array_to_gif(sel_seq, output_save_dir_step + f'/sample_{sel_sample}/' + labels[seq_idx] + '_' + str(sel_channel) + '.gif')


        if eval_residuals:
            residuals_array = residual.detach().cpu().numpy()
            ineq_array = ineq.detach().cpu().numpy() if return_inequality else None
            optimized_quant_array = optimized_quant.detach().cpu().numpy() if return_optimizer else None

            # logging
            log_fn({'residual_mean_abs_samples': np.nanmean(residuals_array)}, step=iteration)
            log_fn({'residual_median_abs_samples': np.nanmedian(residuals_array)}, step=iteration)
            df_data = {'Sample Index': list(range(no_samples)) + ['Mean'],
                    'Residuals (abs)': list(residuals_array)}
            if return_inequality:
                df_data['Inequality'] = list(ineq_array)
            if return_optimizer:
                df_data['Optimized quantity'] = list(optimized_quant_array)
            df_data['Residuals (abs)'].append(np.nanmean(residuals_array))
            if return_optimizer:
                df_data['Optimized quantity'].append(np.nanmean(optimized_quant_array))
            if return_inequality:
                df_data['Inequality'].append(np.nanmean(ineq_array))
            df = pd.DataFrame(df_data)
            csv_path = os.path.join(output_save_dir_step, 'sample_statistics.csv')
            df.to_csv(csv_path, index=False)

        if topopt_eval and gov_eqs == 'mechanics':
            log_fn({'rel_CE_error': np.nanmean(output[1]['rel_CE_error_full_batch'].detach().cpu().numpy())}, step=iteration)
            log_fn({'rel_vf_error': np.nanmean(output[1]['vf_error_full_batch'].detach().cpu().numpy())}, step=iteration)
            log_fn({'fm_error': np.nanmean(output[1]['fm_error_full_batch'].detach().cpu().numpy())}, step=iteration)

        if iteration > 0:
            save_model(config, model, iteration, output_save_dir)

        ema.restore(residuals.model)
        model.train()

    best_residual_test = float('inf')
    try:
        pbar = tqdm(range(train_iterations + 1))
        for iteration in pbar:
            if iteration in batch_schedule and iteration > 0:
                cur_bs = batch_schedule[iteration]
                dl = make_train_dl(cur_bs)   # new shape -> one-time torch.compile recapture
                print(f'batch_size -> {cur_bs} at iter {iteration}', flush=True)
            if iteration in fd_acc_schedule and iteration > 0:
                residuals = build_residuals(model, fd=fd_acc_schedule[iteration])  # swap stencil order
                print(f'fd_acc -> {fd_acc_schedule[iteration]} at iter {iteration}', flush=True)
            if iteration in c_residual_schedule and iteration > 0:
                loss_kwargs['c_residual'] = c_residual_schedule[iteration]          # recalibrate physics weight
                log_fn({'c_residual': loss_kwargs['c_residual']}, step=iteration)
                print(f'c_residual -> {loss_kwargs["c_residual"]:.3e} at iter {iteration}', flush=True)
            torch.compiler.cudagraph_mark_step_begin()  # safe CUDA Graph replay
            model.train()
            cur_batch = next(dl).to(device, non_blocking=True)
            train_ctx = torch.autocast(device_type='cuda', dtype=torch.bfloat16) \
                if p['bf16_train'] else nullcontext()
            with train_ctx:
                loss, data_loss, residual_loss, ineq_loss, opt_loss = diffusion_utils.model_estimation_loss(
                    cur_batch, residual_func=residuals, **loss_kwargs)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), p['grad_clip'])
            optimizer.step()

            # logging
            if iteration % log_freq == 0:
                loss_val = loss.item()
                pbar.set_description(f'training loss: {loss_val:.3e}')
                log_fn({'loss': loss_val}, step=iteration)
                log_fn({'loss_data': data_loss}, step=iteration)
                log_fn({'residual_mean_abs': residual_loss}, step=iteration)
                if c_ineq > 0:
                    log_fn({'loss_inequality': ineq_loss}, step=iteration)
                if lambda_opt > 0:
                    log_fn({'loss_optimization': opt_loss}, step=iteration)
                if scheduler is not None:
                    loss_window.append(loss_val)
                    log_fn({'lr': optimizer.param_groups[0]['lr']}, step=iteration)  # dense for plotting
                if batch_schedule:
                    log_fn({'batch_size': cur_bs}, step=iteration)

            # LR plateau schedule driven by the rolling loss average
            if scheduler is not None and iteration % p['lr_sched_freq'] == 0 and loss_window:
                rolling = sum(loss_window) / len(loss_window)
                scheduler.step(rolling)
                log_fn({'loss_rolling': rolling}, step=iteration)

            # ema update
            if iteration > ema_start:
                ema.update(model)

            # drain async eval results in the MAIN thread (all wandb/best handled here, not in workers)
            if evaluator is not None:
                evaluator.drain()
            if sample_evaluator is not None:
                sample_evaluator.drain()

            # periodic eval -- validation (every test_eval_freq) + generative-sample
            # (every sample_eval_freq); both async, sharing one EMA snapshot. Never blocks training.
            val_fires = (iteration % test_eval_freq == 0) and exists(dl_valid)
            sample_fires = (sample_evaluator is not None) and (iteration > 0) \
                and (iteration % p['sample_eval_freq'] == 0)
            if val_fires or sample_fires:
                snapshot = ready_event = None
                if (val_fires and evaluator is not None) or sample_fires:
                    snapshot = {k: v.detach().clone() for k, v in ema.shadow.items()}
                    if torch.cuda.is_available():
                        ready_event = torch.cuda.Event()
                        ready_event.record()                 # marks completion of the clones above
                if val_fires and evaluator is not None:
                    evaluator.submit(iteration, snapshot, ready_event)
                elif val_fires:
                    model.eval()
                    ema.ema(residuals.model)
                    cur_test_batch = next(dl_valid).to(device, non_blocking=True)
                    # NOTE: no torch.no_grad() since residual gradient may be needed for classifier-free guidance
                    loss_test, data_loss_test, residual_loss_test, ineq_loss_test, opt_loss_test = \
                        diffusion_utils.model_estimation_loss(
                            cur_test_batch, residual_func=residuals, **loss_kwargs)
                    print(f'test loss at iteration {iteration}: {loss_test:.3e}')
                    log_fn({'loss_test': loss_test.item(),
                            'loss_data_test': data_loss_test,
                            'residual_mean_abs_test': residual_loss_test}, step=iteration)
                    if c_ineq > 0:
                        log_fn({'loss_inequality_test': ineq_loss_test}, step=iteration)
                    if lambda_opt > 0:
                        log_fn({'loss_optimization_test': opt_loss_test}, step=iteration)
                    best_residual_test = min(best_residual_test, float(residual_loss_test))
                    ema.restore(residuals.model)
                    model.train()
                if sample_fires:
                    sample_evaluator.submit(iteration, snapshot, ready_event)

            # heavy sampler + checkpoint (GPU-bound; gated by sample_freq; kept synchronous)
            if (iteration % sample_freq == 0) or (p['final_sample'] and iteration == train_iterations):
                sample_and_checkpoint(iteration)

        # final EMA checkpoint (so the trained model can be reloaded, e.g. for an fp64 eval)
        if p['save_final_checkpoint']:
            model.eval()
            ema.ema(residuals.model)
            save_model(config, model, train_iterations, output_save_dir)
            ema.restore(residuals.model)
            model.train()
            print(f'saved final checkpoint: {output_save_dir}/model/checkpoint_{train_iterations}.pt', flush=True)
    finally:
        if evaluator is not None:
            evaluator.close()
            evaluator.drain()                  # log any results the worker produced before exit
            best_residual_test = min(best_residual_test, evaluator.best)
            if evaluator.skipped:
                print(f'[async-eval] skipped {evaluator.skipped} eval(s) that overlapped a busy worker')
        if sample_evaluator is not None:
            sample_evaluator.close()
            sample_evaluator.drain()
            if sample_evaluator.skipped:
                print(f'[sample-eval] skipped {sample_evaluator.skipped} sampling pass(es)')
        if wandb_track:
            import wandb
            wandb.finish()

    # Objective: best generative-sample residual when sample-eval is on (overfitting-robust:
    # this is the min over the periodic samples, i.e. best-checkpoint selection). Else the
    # best validation residual.
    if sample_evaluator is not None and sample_evaluator.best < float('inf'):
        return sample_evaluator.best
    return best_residual_test


if __name__ == '__main__':
    train()
