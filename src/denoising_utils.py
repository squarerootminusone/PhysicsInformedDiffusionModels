import os, yaml
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib as mpl
import matplotlib.pyplot as plt
import imageio
from einops import reduce, rearrange
from src.residuals_mechanics_K import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# fancy plots
def hdr_plot_style():
    plt.style.use('dark_background')
    mpl.rcParams.update({'font.size': 18, 'lines.linewidth': 3, 'lines.markersize': 15})
    mpl.rcParams['ps.useafm'] = True
    mpl.rcParams['pdf.use14corefonts'] = True
    mpl.rcParams['text.usetex'] = False
    plt.rc('legend', facecolor='#666666EE', edgecolor='white', fontsize=16)
    plt.rc('grid', color='white', linestyle='solid')
    plt.rc('text', color='white')
    plt.rc('xtick', direction='out', color='white')
    plt.rc('ytick', direction='out', color='white')
    plt.rc('patch', edgecolor='#E6E6E6')
hdr_plot_style()

def fix_seeds(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

def image_to_b_xy_c(tensor):
    """
    Transpose the tensor from [batch, channels, x, y] to [batch, x*y, channels].
    """
    assert len(tensor.shape) == 4, 'Input tensor must have shape [batch, channels, x, y].'
    batch_size, channels, pixels_x, pixels_y = tensor.shape
    return torch.permute(tensor, (0,2,3,1)).view(batch_size, pixels_x*pixels_y, channels)

def b_xy_c_to_image(tensor, pixels_x = None, pixels_y = None):
    """
    Transpose the tensor from [batch, x*y, channels] to [batch, channels, x, y].
    """
    assert len(tensor.shape) == 3, 'Input tensor must have shape [batch, x*y, channels].'
    batch_size, pixels_x_times_y, channels = tensor.shape
    if pixels_x is None and pixels_y is None:
        assert np.sqrt(pixels_x_times_y) % 1 == 0, 'Number of pixels must be a perfect square.'
        pixels_x = pixels_y = int(np.sqrt(pixels_x_times_y))
    else:
        assert pixels_x*pixels_y == pixels_x_times_y, 'Number of given pixels must match dim 1 of input tensor.'
    return torch.permute(tensor.view(batch_size, pixels_x, pixels_y, channels), (0,3,1,2))

def resize_image(tensor, target_size):
    """
    Transpose the tensor from [batch, channels, ..., pixel_x, pixel_y] to [batch, pixel_x*pixel_y, channels, ...]. We assume two pixel dimensions.
    """
    assert len(tensor.shape) > 3, f"Expected image, got {tensor.shape}"
    original_shape = tensor.shape
    batch_size = original_shape[0]
    num_dims = len(tensor.shape) - 3  # Subtracting batch and pixel dimensions
    pattern = 'b ' + ' '.join([f'c{i}' for i in range(num_dims)]) + ' x y -> b' + ' (' + ' '.join([f'c{i}' for i in range(num_dims)]) + ') ' + 'x y'
    tensor = rearrange(tensor, pattern)
    # equivalent of torchvision transforms.Resize(antialias=False) on tensors (drops the torchvision dep)
    tensor = F.interpolate(tensor, size=(target_size, target_size), mode='bilinear', align_corners=False).view(batch_size, *original_shape[1:-2], target_size, target_size)
    return tensor

def noop(*args, **kwargs):
    pass

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def right_pad_dims_to(x, t):
    padding_dims = x.ndim - t.ndim
    if padding_dims <= 0:
        return t
    return t.view(*t.shape, *((1,) * padding_dims))

def plot_data(data):
    plt.figure(figsize=(16, 12))
    plt.scatter(*data, s=10)
    plt.title('Ground truth $q(\mathbf{x}_{0})$')
    plt.show()
    exit()

def sample_zeros(size):
    return np.zeros((size, 2))

def sample_gaussian(size, dim=2):
    return np.random.randn(size, dim)

def sample_circle(size):
    # sample points from a circle
    theta = np.random.uniform(0, 2*np.pi, size)
    x = np.stack([np.cos(theta), np.sin(theta)], axis=1)
    return x

def sample_hypersphere(size, dim):
    # sample points from a hypersphere surface in `dim` dimensions
    x = np.random.normal(0, 1, (size, dim))
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    x_normalized = x / norm    
    return x_normalized

def sample_two_points(size):
    # two points in 2D
    x = np.array([[-0.5, -0.5], [0.5, 0.5]])
    # random selection of these two points
    return x[np.random.randint(2, size=size)]

def sample_four_points(size):
    # four points in 2D
    x = np.array([[-1., -1.], [-1., 1.], [1., -1.], [1., 1.]])
    # random selection of these four points
    return x[np.random.randint(4, size=size)]

def sample_images_with_squares(no_points, pixels_per_dim, dim, frame_dim = False, use_double = True):

    if use_double:
        dtype = np.float64
    else:
        dtype = np.float32

    # Define the size of the square (e.g., a quarter of the image dimension)
    square_size = pixels_per_dim // 4

    # Initialize an array to store the images
    # Shape: (no_points, pixels_per_dim, pixels_per_dim, dim)
    if frame_dim:
        images = np.zeros((no_points, dim, 1, pixels_per_dim, pixels_per_dim), dtype=dtype)
    else:
        images = np.zeros((no_points, dim, pixels_per_dim, pixels_per_dim), dtype=dtype)

    for i in range(no_points):
        # Randomly choose the top-left corner of the square
        x_start = np.random.randint(0, pixels_per_dim - square_size)
        y_start = np.random.randint(0, pixels_per_dim - square_size)

        for j in range(dim):
            # Draw the square in each channel of the image
            # You can modify the pattern per channel as needed
            if frame_dim:
                images[i, j, :, x_start:x_start + square_size, y_start:y_start + square_size] = 1.
            else:
                images[i, j, x_start:x_start + square_size, y_start:y_start + square_size] = 1.
                
    return images

def sample_ones(no_points, pixels_per_dim, dim, frame_dim = False):
    if frame_dim:
        return np.ones((no_points, dim, 1, pixels_per_dim, pixels_per_dim), dtype=float)
    else:
        return np.ones((no_points, dim, pixels_per_dim, pixels_per_dim), dtype=float)

class EMA(object):
    def __init__(self, mu=0.999):
        self.mu = mu
        self.shadow = {}
        self.backup = {}
        # opt8e: cache tensor lists for fused _foreach_ ops (set via _cache_lists)
        self._opt8e_foreach = os.environ.get('OPT8_EMA_FOREACH', '') == '1'
        self._lists_cached = False

    def register(self, module):
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def _cache_lists(self, module):
        self._param_names = []
        self._shadow_list = []
        self._param_list = []
        for name, param in module.named_parameters():
            if param.requires_grad:
                self._param_names.append(name)
                self._shadow_list.append(self.shadow[name])
                self._param_list.append(param.data)
        self._lists_cached = True

    def update(self, module):
        if self._opt8e_foreach:
            if not self._lists_cached:
                self._cache_lists(module)
            # lerp(shadow, param, 1-mu) = shadow + (1-mu)*(param - shadow) = mu*shadow + (1-mu)*param
            torch._foreach_lerp_(self._shadow_list, self._param_list, 1.0 - self.mu)
            return
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name].data = (1. - self.mu) * param.data + self.mu * self.shadow[name].data

    def ema(self, module, backup=True):
        if self._opt8e_foreach:
            if not self._lists_cached:
                self._cache_lists(module)
            if backup:
                # cache backup tensor list once
                if not hasattr(self, '_backup_list'):
                    self._backup_list = [p.clone() for p in self._param_list]
                else:
                    torch._foreach_copy_(self._backup_list, self._param_list)
                # populate dict for legacy access compatibility
                for n, t in zip(self._param_names, self._backup_list):
                    self.backup[n] = t
            torch._foreach_copy_(self._param_list, self._shadow_list)
            return
        for name, param in module.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                if backup:
                    self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name].data)

    def restore(self, module):
        if self._opt8e_foreach:
            assert hasattr(self, '_backup_list')
            torch._foreach_copy_(self._param_list, self._backup_list)
            self.backup = {}
            return
        assert hasattr(self, 'backup')
        for name, param in module.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data.copy_(self.backup[name])
        self.backup = {}

    def ema_copy(self, module):
        module_copy = type(module)(module.config).to(module.config.device)
        module_copy.load_state_dict(module.state_dict())
        self.ema(module_copy)
        return module_copy

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state_dict):
        self.shadow = state_dict

def remove_outliers(data, percentile = 0.01, also_lower_bound = False):    
    percentile *= 100    
    if data.size == 0:
        return data  # Return the empty array as is
    
    norms = np.linalg.norm(data, axis=1)
    
    # compute the lower and upper bounds for filtering based on norms
    lower_bound = np.percentile(norms, percentile) if also_lower_bound else 0.
    upper_bound = np.percentile(norms, 100 - percentile)
    mask = (norms > lower_bound) & (norms < upper_bound)
    return data[mask]

# tensor of shape (channels, frames, height, width) -> gif
def array_to_gif(data, output_save_dir, x_lim, y_lim, label = None, duration = 0.05):    
    # Create a GIF writer object
    with imageio.get_writer(output_save_dir, mode='I', duration = duration, loop=1) as writer:
        for step in range(data.shape[0]):
            fig, ax = plt.subplots()
            ax.scatter(data[step, :, 0], data[step, :, 1], s=10)
            ax.set_xlim(x_lim)
            ax.set_ylim(y_lim)
            if label is not None:
                if label == 'sampled':
                    ax.set_title('$p(\mathbf{x}_{' + str(len(data)-step-1)+'})$')
                elif label == 'pred':
                    ax.set_title('Model pred. step ' + str(len(data)-step-1))
                else:
                    ax.set_title(label)
            # Save the current figure directly to the GIF, without an intermediate file
            gif_frame_path = output_save_dir[:-4] + f'_gif_frame_{step}.png'
            plt.savefig(gif_frame_path)
            plt.close(fig)
            writer.append_data(imageio.imread(gif_frame_path))
            # remove the temporary image file after adding to the GIF
            os.remove(gif_frame_path)

def image_array_to_gif(image_array, output_file, frame_duration=0.05, normalization_mode='final_pred', given_min_max=None):
    """
    Create a GIF from a numpy array of images with different normalization modes.

    :param image_array: Numpy array of shape (frames, pixel, pixel)
    :param output_file: Output file path for the GIF
    :param frame_duration: Duration of each frame in the GIF
    :param normalization_mode: Mode of normalization ('given', 'global', 'individual', 'none')
    :param given_min_max: Tuple of (min, max) values for 'given' normalization mode
    """
    if normalization_mode == 'final_pred':
        min_val, max_val = image_array[-1].min(), image_array[-1].max()
    elif normalization_mode == 'global':
        min_val, max_val = image_array.min(), image_array.max()
    elif normalization_mode == 'given':
        if given_min_max is None:
            raise ValueError("Please provide min and max values for 'given' normalization mode.")
        min_val, max_val = given_min_max

    with imageio.get_writer(output_file, mode='I', duration=frame_duration) as writer:
        for frame in image_array:
            if normalization_mode == 'individual':
                min_val, max_val = frame.min(), frame.max()
            if normalization_mode != 'none':
                # Normalize the frame
                frame = (frame - min_val) / (max_val - min_val)
                frame = (frame * 255).astype(np.uint8)
            writer.append_data(frame)

def save_model(config, model, train_iterations, output_save_dir):

    os.makedirs(Path(output_save_dir, 'model/'), exist_ok=True)

    # save yaml file for later runs
    with open(output_save_dir + '/model/model.yaml', 'w') as yaml_file:
        yaml.dump(dict(config), yaml_file, default_flow_style=False)    
    save_dir_model = output_save_dir + '/model/checkpoint_' + str(train_iterations) + '.pt'
    save_obj = dict(
        model = model.state_dict()
    )
    # save model to path
    with open(save_dir_model, 'wb') as f:
        torch.save(save_obj, f)
    print(f'\ncheckpoint saved to {output_save_dir}/.')

def load_model(path, model, strict = True):

    # to avoid extra GPU memory usage in main process when using Accelerate
    with open(path, 'rb') as f:
        loaded_obj = torch.load(f, map_location='cpu')
    try:
        model.load_state_dict(loaded_obj['model'], strict = strict)
    except RuntimeError:
        print('Failed loading state dict.')
    print('\nCheckpoint loaded from {}'.format(path))

    return model

def extract(input, t, x):
    shape = x.shape
    out = torch.gather(input, 0, t.to(input.device))
    reshape = [t.shape[0]] + [1] * (len(shape) - 1)
    return out.reshape(*reshape)

class LossSecondMomentResampler:
    """Importance-sample diffusion timesteps by the running second moment of the
    per-timestep loss (Nichol & Dhariwal, 'Improved DDPM', arXiv:2102.09672 Sec 3.3).

    Draw t ~ p(t) ∝ sqrt(E[L_t^2]) instead of uniform, then reweight each term by
    w(t) = (1/T) / p(t) so the estimator of the *uniform-t* objective stays unbiased
    (the model sees the same loss in expectation; only the gradient variance drops).
    Falls back to uniform sampling until every t has `history_per_term` observations."""
    def __init__(self, n_steps, history_per_term=10, uniform_prob=0.001):
        self.n_steps = n_steps
        self.history_per_term = history_per_term
        self.uniform_prob = uniform_prob
        self._loss_history = np.zeros([n_steps, history_per_term], dtype=np.float64)
        self._loss_counts = np.zeros([n_steps], dtype=np.int64)

    def _warmed_up(self):
        return bool((self._loss_counts == self.history_per_term).all())

    def weights(self):
        if not self._warmed_up():
            return np.ones([self.n_steps], dtype=np.float64) / self.n_steps
        w = np.sqrt(np.mean(self._loss_history ** 2, axis=1))
        w = w / w.sum()
        w = (1.0 - self.uniform_prob) * w + self.uniform_prob / self.n_steps
        return w

    def sample(self, batch_size, device):
        p = self.weights()
        p_t = torch.from_numpy(p).to(device=device, dtype=torch.float32)
        t = torch.multinomial(p_t, batch_size, replacement=True)
        # importance weight = uniform(t) / p(t) = (1/T) / p(t)
        imp_weights = 1.0 / (self.n_steps * p_t[t])
        return t, imp_weights

    def update_with_local_losses(self, ts, losses):
        ts = ts.detach().cpu().numpy()
        losses = losses.detach().to(torch.float64).cpu().numpy()
        for t, l in zip(ts, losses):
            if self._loss_counts[t] == self.history_per_term:
                self._loss_history[t, :-1] = self._loss_history[t, 1:]
                self._loss_history[t, -1] = l
            else:
                self._loss_history[t, self._loss_counts[t]] = l
                self._loss_counts[t] += 1

class DenoisingDiffusion(nn.Module):
    def __init__(self, n_steps, device, residual_grad_guidance = False):
        self.n_steps = n_steps
        self.device = device
        self.diff_dict = self.create_diff_dict()
        self.residual_grad_guidance = residual_grad_guidance

    def create_diff_dict(self): 
        diff_dict = {
            'betas': self.make_beta_schedule(schedule='cosine', n_timesteps=self.n_steps, start=1e-5, end=1e-2).to(self.device),
        }
        diff_dict['alphas'] = 1. - diff_dict['betas']
        diff_dict['sqrt_recip_alphas'] = torch.sqrt(1. / diff_dict['alphas'])
        diff_dict['alphas_prod'] = torch.cumprod(diff_dict['alphas'], 0)
        diff_dict['alphas_prod_p'] = torch.cat([torch.tensor([1], device=device).float(), diff_dict['alphas_prod'][:-1]], 0)
        diff_dict['alphas_bar_sqrt'] = torch.sqrt(diff_dict['alphas_prod'])
        diff_dict['sqrt_recip_alphas_cumprod'] = torch.sqrt(1. / diff_dict['alphas_prod'])
        diff_dict['sqrt_recipm1_alphas_cumprod'] = torch.sqrt(1. / diff_dict['alphas_prod'] - 1)
        diff_dict['one_minus_alphas_bar_log'] = torch.log(1 - diff_dict['alphas_prod'])
        diff_dict['one_minus_alphas_bar_sqrt'] = torch.sqrt(1 - diff_dict['alphas_prod'])
        diff_dict['alphas_prod_prev'] = F.pad(diff_dict['alphas_prod'][:-1], (1, 0), value=1.)
        diff_dict['posterior_mean_coef1'] = diff_dict['betas'] * torch.sqrt(diff_dict['alphas_prod_prev']) / (1. - diff_dict['alphas_prod'])
        diff_dict['posterior_mean_coef2'] = (1. - diff_dict['alphas_prod_prev']) * torch.sqrt(diff_dict['alphas']) / (1. - diff_dict['alphas_prod'])

        diff_dict['noise_mean_coeff'] = torch.sqrt(1. / diff_dict['alphas']) * (1. - diff_dict['alphas']) / torch.sqrt(1. - diff_dict['alphas_prod'])

        # posterior variance
        diff_dict['posterior_variance'] = diff_dict['betas'] * (1. - diff_dict['alphas_prod_prev']) / (1. - diff_dict['alphas_prod'])
        # clip this since it is 0 at the beginning
        diff_dict['posterior_variance_clipped'] = diff_dict['posterior_variance'].clone()
        diff_dict['posterior_variance_clipped'][0] = diff_dict['posterior_variance'][1]

        # NOTE Ho et al. also have a version that clips the log to 1.e-20.
        diff_dict['posterior_log_variance_clipped'] = torch.log(diff_dict['posterior_variance_clipped'])

        use_constant_p2_weight = False
        if use_constant_p2_weight:
            p2_loss_weight_gamma = 1.0  # p2 loss weight, from https://arxiv.org/abs/2204.00227 - 0 is equivalent to weight of 1 across time - 1. is recommended
            p2_loss_weight_k = 1.0
            diff_dict['p2_loss_weight'] = (p2_loss_weight_k + diff_dict['alphas_prod'] / (1. - diff_dict['alphas_prod'])) ** -p2_loss_weight_gamma
        else:
            snr = diff_dict['alphas_prod'] / (1. - diff_dict['alphas_prod'])
            diff_dict['p2_loss_weight'] = torch.minimum(snr, torch.ones_like(snr) * 5.0) # from https://arxiv.org/pdf/2303.09556.pdf

        return diff_dict

    def make_beta_schedule(self, schedule='linear', n_timesteps=1000, start=1e-5, end=1e-2):
        if schedule == 'linear':
            betas = torch.linspace(start, end, n_timesteps)
        elif schedule == "quad":
            betas = torch.linspace(start ** 0.5, end ** 0.5, n_timesteps) ** 2
        elif schedule == 'sigmoid':
            betas = torch.linspace(-6, 6, n_timesteps)
            betas = torch.sigmoid(betas) * (end - start) + start
        elif schedule == 'cosine':
            s = 0.008
            steps = n_timesteps + 1
            x = torch.linspace(0, n_timesteps, steps)
            alphas_cumprod = torch.cos(((x / n_timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            betas = torch.clip(betas, 0, 0.999)
        return betas

    # Sampling function
    def q_sample(self, x_0, t, alphas_bar_sqrt, one_minus_alphas_bar_sqrt, noise=None):
        if noise is None:
            noise = torch.randn_like(x_0)
        alphas_t = extract(alphas_bar_sqrt, t, x_0)
        alphas_1_m_t = extract(one_minus_alphas_bar_sqrt, t, x_0)
        return (alphas_t * x_0 + alphas_1_m_t * noise)

    def plot_diffusion(self, dataset, alphas_bar_sqrt, one_minus_alphas_bar_sqrt):
        fig, axs = plt.subplots(1, 10, figsize=(18, 2))
        for i in range(10):
            q_i = self.q_sample(dataset, torch.tensor([i * 10]), alphas_bar_sqrt, one_minus_alphas_bar_sqrt)
            axs[i].scatter(q_i[:, 0], q_i[:, 1], s=10)
            axs[i].set_axis_off(); axs[i].set_title('$q(\mathbf{x}_{'+str(i*10)+'})$', fontsize=10)
        plt.show()

    def p_sample(self, x, conditioning_input, t, 
                save_output = False, surpress_noise = False, 
                use_dynamic_threshold = False, residual_func = None, eval_residuals = False,
                return_optimizer = False, return_inequality = False, residual_correction = False,
                correction_mode = 'none'):
        
        x_init = x.clone().detach()        
        if conditioning_input is not None:
            conditioning, bcs, solution = conditioning_input 
            x = torch.cat((x, conditioning), dim = 1)
        batch_size = len(x)
        assert correction_mode in ['x0', 'xt'] or not residual_correction, 'Correction mode unknown or not given.'
        
        t = torch.tensor([t], device=x.device)
        model_input = image_to_b_xy_c(x) # we reshape this later to an image in U-net model class but let's be consistent here with the operator model
        model_input = (model_input, t.repeat(batch_size))

        model_intermediate = None
        
        # model output
        # evaluate residuals at last timestep if required
        extra_kwargs = {}
        if residual_func.gov_eqs == 'darcy':
            residual_input = (model_input, )
            sample = True
            # opt15: the FD residual at intermediate sampling steps is discarded — only compute
            # it at the final step when it is actually consumed (corrections re-evaluate anyway)
            extra_kwargs['skip_residual'] = not (t[0] == 0 and eval_residuals)
        if residual_func.gov_eqs == 'mechanics':
            vf = conditioning[:,0,0,0]
            # vf = x_0[:,2].mean((1,2))
            residual_input = (model_input, bcs, vf, solution)
            if t[0] == 0:
                sample = True
            else:
                sample = False
        out_dict = residual_func.compute_residual(  residual_input,
                                                    reduce='per-batch',
                                                    return_model_out = True,
                                                    return_optimizer = return_optimizer,
                                                    return_inequality = return_inequality,
                                                    sample = sample,
                                                    ddim_func = self.ddim_sample_x0,
                                                    **extra_kwargs)
        
        output, residual = out_dict['model_out'], out_dict['residual']
        model_out = output
        if len(model_out.shape) == 3:
            # convert to image [batch_size, channels, pixels, pixels]
            model_out = generalized_b_xy_c_to_image(model_out)

        if residual_correction and correction_mode == 'x0':
            model_out, residual = residual_func.residual_correction(generalized_image_to_b_xy_c(model_out))
            model_out = generalized_b_xy_c_to_image(model_out)
            
        if save_output:
            model_intermediate = model_out.clone().detach()
        x0_pred = model_out
        mean = (
            extract(self.diff_dict['posterior_mean_coef1'], t, x_init) * x0_pred +
            extract(self.diff_dict['posterior_mean_coef2'], t, x_init) * x_init
        )

        # Generate z
        z = torch.randn_like(x_init, device=x.device)
        # Fixed sigma
        sigma_t = extract(self.diff_dict['betas'], t, x_init).sqrt()
        # no noise when t == 0
        if surpress_noise:
            nonzero_mask = (1. - (t == 0).float())
        else:
            nonzero_mask = 1.
        sample = mean + nonzero_mask * sigma_t * z

        if residual_correction and correction_mode == 'xt':
            sample, residual = residual_func.residual_correction(generalized_image_to_b_xy_c(sample))
            sample = generalized_b_xy_c_to_image(sample)

        dynamic_thres_percentile = 0.9
        if use_dynamic_threshold:
            def maybe_clip(x):
                s = torch.quantile(
                    rearrange(x.float(), "b ... -> b (...)").abs(),
                    dynamic_thres_percentile,
                    dim=-1,
                )
                s.clamp_(min=1.0)
                s = right_pad_dims_to(x, s)
                x = x.clamp(-s, s) / s
                return x
            sample = maybe_clip(sample)
        
        if (t[0] == 0 and eval_residuals):
            aux_out = {}
            aux_out['residual'] = residual
            if return_optimizer:
                aux_out['optimized_quant'] = out_dict['optimizer']
            if return_inequality:
                aux_out['inequality_quant'] = out_dict['inequality']
            if residual_func.gov_eqs == 'mechanics':
                if residual_func.topopt_eval:
                    aux_out['rel_CE_error_full_batch'] = out_dict['rel_CE_error_full_batch']
                    aux_out['vf_error_full_batch'] = out_dict['vf_error_full_batch']
                    aux_out['fm_error_full_batch'] = out_dict['fm_error_full_batch']

            return (sample, model_intermediate), aux_out
        else:
            return (sample, model_intermediate), None

    # NOTE we do not use @torch.inference_mode() since we need gradients to obtain residual
    # to free up memory, we manually call .detach() where appropriate
    def p_sample_loop(self,
                    conditioning_input,
                    shape,
                    save_output = False, 
                    surpress_noise = True,
                    use_dynamic_threshold = False,
                    residual_func = None,
                    eval_residuals = False,
                    return_optimizer = False,
                    return_inequality = False,
                    M_correction = 0,
                    N_correction = 0,
                    correction_mode = 'none'):

        cur_x = torch.randn(shape, device=self.diff_dict['alphas'].device)
        x_seq = [cur_x.detach().cpu()]

        if save_output:
            interm_imgs = [torch.zeros(shape)]
        else:
            interm_imgs = []

        interm_img = None
        for i in reversed(range(self.n_steps)):
            
            # CoCoGen correction
            residual_correction = False
            if i < N_correction:
                residual_correction = True
                eval_residuals = True
                
            output = self.p_sample(cur_x.detach(), conditioning_input, i, save_output, surpress_noise, use_dynamic_threshold,
                                        residual_func = residual_func, eval_residuals = eval_residuals,
                                        return_optimizer = return_optimizer, return_inequality = return_inequality,
                                        residual_correction = residual_correction, correction_mode = correction_mode)
            cur_x, interm_img = output[0]

            x_seq.append(cur_x.detach().cpu())
            interm_imgs.append(interm_img.detach().cpu())

        # CoCoGen correction
        for i in range(M_correction):
            cur_x, residual = residual_func.residual_correction(generalized_image_to_b_xy_c(cur_x))
            cur_x = generalized_b_xy_c_to_image(cur_x)
            x_seq.append(cur_x.detach().cpu())  
            if eval_residuals and i == M_correction - 1:
                output[1]['residual'] = residual
            
        if eval_residuals:
            return (x_seq, interm_imgs), output[1]
        else:
            return x_seq, interm_imgs

    def normal_kl(self, mean1, logvar1, mean2, logvar2):
        """
        KL divergence between normal distributions parameterized by mean and log-variance.
        """
        kl = 0.5 * (-1.0 + logvar2 - logvar1 + torch.exp(logvar1 - logvar2) + ((mean1 - mean2) ** 2) * torch.exp(-logvar2))
        return kl

    def gaussian_log_likelihood(self, x, means, variance):
        centered_x = x - means    
        squared_diffs = (centered_x ** 2) / variance
        log_probs = -0.5 * squared_diffs
        return log_probs

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.diff_dict['sqrt_recip_alphas_cumprod'], t, x_t) * x_t -
            extract(self.diff_dict['sqrt_recipm1_alphas_cumprod'], t, x_t) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return (
            extract(self.diff_dict['sqrt_recip_alphas_cumprod'], t, x_t) * x_t - x0
        ) / extract(self.diff_dict['sqrt_recipm1_alphas_cumprod'], t, x_t)

    def predict_noise_from_mean(self, x_t, t, mean_t):
        return (
            extract(self.diff_dict['sqrt_recip_alphas'], t, mean_t) * x_t - mean_t
        ) / extract(self.diff_dict['noise_mean_coeff'], t, mean_t)

    def loss_variational(self, output, x_0, x_t, t, base_2 = False):    
        batch_size = x_0.shape[0]

        # Compute the true mean and variance
        true_mean = (
        extract(self.diff_dict['posterior_mean_coef1'], t, x_t) * x_0 +
        extract(self.diff_dict['posterior_mean_coef2'], t, x_t) * x_t
        )
        
        true_var = extract(self.diff_dict['posterior_variance_clipped'], t, x_t)
        model_var = true_var

        # Infer the mean and variance with our model
        model_mean = output

        # Compute the KL loss
        true_var_log = torch.log(true_var)
        model_var_log = torch.log(model_var)
        kl = self.normal_kl(true_mean, true_var_log, model_mean, model_var_log)
        kl = torch.mean(kl.view(batch_size, -1), dim=1)
        if base_2:
            kl = kl / np.log(2.)

        # define p(x_0|x_1) simply as a gaussian
        log_likelihood = self.gaussian_log_likelihood(x_0, means=model_mean, variance=model_var)
        log_likelihood = torch.mean(log_likelihood.view(batch_size, -1), dim=1) 
        if base_2:
            log_likelihood = log_likelihood / np.log(2.)

        # At the first timestep return the log likelihood, otherwise return KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))
        # BUG (imo) nan/inf values in tensor that is not considered in torch.where() still affects gradients. Thus check for this before.
        assert log_likelihood.isnan().any() == False, 'Log likelihood is nan.'
        assert log_likelihood.isinf().any() == False, 'Log likelihood is inf.'

        loss_log_likelihood = -1. * log_likelihood # since we minimize loss (instead of maximizing likelihood)

        loss = torch.where(t == 0, loss_log_likelihood, kl)

        return loss.mean(-1)

    def _dwt_hf_weight(self, target, lam):
        """B2 (PG-Diff-style): per-pixel data-loss weight from single-level Haar high-frequency
        energy of the target — upweights sharp interfaces (where the Darcy residual error
        concentrates). Per-sample-per-channel normalized to mean 1 so the overall data-loss
        scale (and thus the c_residual balance) is unchanged."""
        B, C, H, W = target.shape
        if not hasattr(self, '_haar_hf_k') or self._haar_hf_k.device != target.device:
            lh = [[0.5, 0.5], [-0.5, -0.5]]
            hl = [[0.5, -0.5], [0.5, -0.5]]
            hh = [[0.5, -0.5], [-0.5, 0.5]]
            self._haar_hf_k = torch.tensor([[lh], [hl], [hh]], device=target.device)
        k = self._haar_hf_k.to(target.dtype)
        e = F.conv2d(target.detach().reshape(B * C, 1, H, W), k, stride=2)
        e = (e ** 2).sum(dim=1, keepdim=True)
        e = F.interpolate(e, size=(H, W), mode='nearest').reshape(B, C, H, W)
        e = e / e.mean(dim=(2, 3), keepdim=True).clamp_min(1e-12)
        w = 1. + lam * e
        return w / w.mean(dim=(2, 3), keepdim=True)

    def model_estimation_loss(self,
                              input,
                              residual_func = None,
                              c_data = 1.,
                              c_residual = 0.,
                              c_ineq = 0.,
                              lambda_opt = 0.,
                              hf_loss_weight = 0.,
                              allow_importance = True):

        batch_size = len(input)
        # importance-sampling of t (env-gated, TRAINING ONLY): draw t from the loss-second-moment
        # distribution and carry per-sample importance weights to keep the objective unbiased.
        # allow_importance=False (eval/test) keeps uniform t so loss_test stays a clean, comparable
        # metric and eval batches don't pollute the resampler history.
        _imp_sample = allow_importance and os.environ.get('IMPORTANCE_SAMPLE_T', '') == '1'
        if _imp_sample:
            if getattr(self, '_t_sampler', None) is None or self._t_sampler.n_steps != self.n_steps:
                self._t_sampler = LossSecondMomentResampler(self.n_steps)
            t, _imp_weights = self._t_sampler.sample(batch_size, input.device)
        else:
            t = torch.randint(0, self.n_steps, size=(batch_size,), device=input.device)
            _imp_weights = None

        if residual_func.gov_eqs == 'darcy':
            x_0 = input
        if residual_func.gov_eqs == 'mechanics':            
            conditioning, x_0, bcs = torch.tensor_split(input, (3, 6), dim=1) # vf_arr, strain_energy_density_fem, von_mises_stress, disp_x, disp_y, E_field, BC_node_x, BC_node_y, load_x_img, load_y_img

        # x0 multiplier
        a = extract(self.diff_dict['alphas_bar_sqrt'], t, x_0)
        # eps multiplier
        am1 = extract(self.diff_dict['one_minus_alphas_bar_sqrt'], t, x_0)
        e = torch.randn_like(x_0, device=x_0.device)
        # model input
        x = x_0 * a + e * am1 # previous x_t

        if residual_func.gov_eqs == 'mechanics':
            x = torch.cat((x, conditioning), dim=1)
            
        x = image_to_b_xy_c(x) # we reshape this later to an image in U-net model class but let's be consistent here with the operator model
        model_input = (x, t)

        return_inequality = False
        return_optimizer = False
        if c_ineq > 0.:
            return_inequality = True
        if lambda_opt > 0. or residual_func.gov_eqs == 'mechanics':
            return_optimizer = True

        if residual_func.gov_eqs == 'darcy':
            residual_input = (model_input, )
        if residual_func.gov_eqs == 'mechanics':
            vf = conditioning[:,0,0,0]
            residual_input = (model_input, bcs, vf, x_0)

        out_dict = residual_func.compute_residual(residual_input,
                                                  reduce='per-batch', 
                                                  return_model_out = True, 
                                                  return_optimizer = return_optimizer, 
                                                  return_inequality = return_inequality,
                                                  ddim_func = self.ddim_sample_x0)
        
        residual, output = out_dict['residual'], out_dict['model_out']

        # reshape output to image (batch_size, channels, pixels, pixels)
        if len(output.shape) == 3:
            output = b_xy_c_to_image(output)
            
        target = x_0
        loss_fn = nn.MSELoss(reduction='none')
        loss = loss_fn(target, output)
        if hf_loss_weight > 0 and target.ndim == 4:
            loss = loss * self._dwt_hf_weight(target, hf_loss_weight)
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss * extract(self.diff_dict['p2_loss_weight'], t, loss)

        # per-sample data loss (shape [b]): reduce 'b ... -> b (...)' only flattens, so mean the
        # flattened pixel dim here (the original code deferred this to the trailing .mean())
        data_loss_ps = c_data * (loss.mean(dim=1) if loss.ndim > 1 else loss)
        # opt14: detached 0-dim tensors instead of .item() — each .item() forces a device
        # sync every iteration on this dispatch-bound workload; callers convert at log freq.
        data_loss_track = data_loss_ps.detach().mean()

        # per-sample negative residual log-likelihood, i.e., - log p(r|x_0_pred(x_0))
        var = extract(self.diff_dict['posterior_variance_clipped'], t, residual)

        residual_loss_track = residual.detach().abs().mean()

        residual_log_likelihood = self.gaussian_log_likelihood(torch.zeros_like(residual), means=residual, variance=var)
        residual_loss = c_residual * -1. * residual_log_likelihood
        residual_loss_ps = residual_loss.flatten(1).mean(dim=1) if residual_loss.ndim > 1 else residual_loss

        total_loss_ps = data_loss_ps + residual_loss_ps

        ineq_loss_track = 0.
        if return_inequality:
            # add negative inequality residual log-likelihood, i.e., - log p(r_ineq|x_0_pred(x_0)) (similar to above)
            ineq_log_likelihood = self.gaussian_log_likelihood(torch.zeros_like(out_dict['inequality']), means=out_dict['inequality'], variance=var)
            ineq_loss = c_ineq * -1. * ineq_log_likelihood
            ineq_loss_ps = ineq_loss.flatten(1).mean(dim=1) if ineq_loss.ndim > 1 else ineq_loss
            total_loss_ps = total_loss_ps + ineq_loss_ps
            ineq_loss_track = out_dict['inequality'].detach().mean()

        opt_loss_track = 0.
        if return_optimizer:
            # add optimization log-likelihood, i.e., log p(c=c_min|x_0_pred(x_0)) (where p is Expon. distribution)
            opt_log_likelihood = -1. * out_dict['optimizer']
            opt_loss = -1. * lambda_opt * opt_log_likelihood
            opt_loss_ps = opt_loss.flatten(1).mean(dim=1) if opt_loss.ndim > 1 else opt_loss
            total_loss_ps = total_loss_ps + opt_loss_ps
            opt_loss_track = out_dict['optimizer'].detach().mean()

        # combine: importance-weighted mean keeps the uniform-t objective unbiased; feed the
        # per-sample loss back to the resampler for its second-moment estimate.
        if _imp_weights is not None:
            loss = (_imp_weights * total_loss_ps).mean()
            self._t_sampler.update_with_local_losses(t, total_loss_ps.detach())
        else:
            loss = total_loss_ps.mean()

        return loss, data_loss_track, residual_loss_track, ineq_loss_track, opt_loss_track
    
    def ddim_sample_x0(self, xt, t, model, shape, reduced_n_steps, ddim_sampling_eta, gov_eqs = None, self_cond = None):

        batch, device, sample_timesteps, eta = shape[0], self.diff_dict['alphas'].device, reduced_n_steps, ddim_sampling_eta

        if len(t) == 1:
            batch_t = torch.ones(batch, device=device, dtype=torch.long)*t
        else:
            batch_t = t

        # opt10b: pure-GPU equivalent of the original .cpu().numpy() + np.linspace Python loop.
        # Replaces a forced sync barrier per training iter with one GPU-side construction.
        ts_frac = torch.arange(sample_timesteps + 2, device=device, dtype=torch.float32) / float(sample_timesteps + 1)  # (steps+2,) in [0, 1]
        seqs_forward = (batch_t.to(torch.float32).unsqueeze(1) * ts_frac.unsqueeze(0)).long()  # (batch, steps+2): [0, step, ..., batch_t[i]]
        seqs_rev = seqs_forward.flip(1)  # (batch, steps+2): [batch_t[i], ..., step, 0]
        seqs_next_unrev = torch.cat([
            torch.full((batch_t.shape[0], 1), -1, device=device, dtype=torch.long),
            seqs_forward[:, :-1],
        ], dim=1)  # (batch, steps+2): [-1, 0, ..., batch_t[i]-step]
        seqs_next_rev = seqs_next_unrev.flip(1)  # (batch, steps+2): [batch_t[i]-step, ..., 0, -1]
        cur_times = seqs_rev.T   # (steps+2, batch)
        next_times = seqs_next_rev.T  # (steps+2, batch)

        time_pairs = list(zip(cur_times, next_times)) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        if len(xt.shape) == 3:            
            xt = generalized_b_xy_c_to_image(xt)

        if gov_eqs == 'mechanics':
            model_input = xt
            cur_x = xt[:, :3] # consider only solution fields
        else:
            model_input = xt
            cur_x = xt
        x0_pred = None

        model_output = None
        for fwd_idx, (t, t_next) in enumerate(time_pairs):

            # create mask for those timesteps that are equal
            mask = (t == t_next).float().view(-1, 1, 1, 1)
            model_output = model(model_input, t, self_cond)
            x0_pred = model_output            
            mean = (
                extract(self.diff_dict['posterior_mean_coef1'], t, cur_x) * x0_pred +
                extract(self.diff_dict['posterior_mean_coef2'], t, cur_x) * cur_x
            )
            # noise estimate
            eps_theta = self.predict_noise_from_mean(cur_x, t, mean)

            if fwd_idx == 0:
                model_out = model_output

            if t_next[0] < 0: # this happens when we predict x0, should never happen during training
                # assert that all next timesteps are equal
                assert torch.all(t_next == -1), 'Next timesteps should be -1, otherwise this is inconsistent.'
                cur_x = x0_pred
                continue

            alpha = extract(self.diff_dict['alphas_prod'], t, cur_x)
            alpha_next = extract(self.diff_dict['alphas_prod'], t_next, cur_x)

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(cur_x)

            # only update cur_x where t != t_next
            # NOTE: this is the standard update (not using larger variance)
            cur_x_ = x0_pred * alpha_next.sqrt() + \
                     c * eps_theta + \
                     sigma * noise
            
            cur_x = mask * cur_x + (1 - mask) * cur_x_

        assert model_out is not None, 'Model output not given.'
        return cur_x, model_out