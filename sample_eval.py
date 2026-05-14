"""sample_eval.py — parametrized version of sample.py for batch evaluation.

Usage:
    python sample_eval.py \
        --directory_path ./trained_models/darcy/ \
        --name PIDM-ME \
        --load_model_step 300000 \
        --output_dir ./results/reproduced/darcy/PIDM-ME
"""
import argparse, os, yaml, time
import matplotlib.pyplot as plt
import pandas as pd
import torch
from pathlib import Path
from src.data_utils import *
from torch.utils.data import DataLoader
from src.denoising_utils import *
from src.unet_model import Unet3D
from src.residuals_darcy import ResidualsDarcy
from src.residuals_mechanics_K import ResidualsMechanics

parser = argparse.ArgumentParser()
parser.add_argument('--directory_path', required=True)
parser.add_argument('--name', required=True)
parser.add_argument('--load_model_step', type=int, required=True)
parser.add_argument('--output_dir', required=True)
args = parser.parse_args()

directory_path = args.directory_path
name = args.name
load_model_step = args.load_model_step
output_base_dir = args.output_dir

no_samples = 3
create_gif = False
topopt_eval = True
eval_test_sets = True
test_batches = -1

load_path = directory_path + name
config = yaml.safe_load(Path(load_path, 'model', 'model.yaml').read_text())

use_ddim_x0 = False
ddim_steps = 0

residual_grad_guidance = config['residual_grad_guidance']
correction_mode = config['correction_mode']
M_correction = config['M_correction']
N_correction = config['N_correction']

gov_eqs = config['gov_eqs']
if gov_eqs != 'darcy' and residual_grad_guidance:
    raise ValueError('Gradient guidance only implemented for Darcy equation.')
fd_acc = config['fd_acc']
diff_steps = config['diff_steps']
use_dynamic_threshold = False
self_condition = False
use_double = False

save_output = True
eval_residuals = True

data_paths = None
if gov_eqs == 'darcy':
    input_dim = 2
    output_dim = 2
    pixels_at_boundary = True
    domain_length = 1.
    reverse_d1 = True
    bcs = 'none'
    pixels_per_dim = 64
    return_optimizer = False
    return_inequality = False
    train_batch_size = 32
    sigmoid_last_channel = False
elif gov_eqs == 'mechanics':
    input_dim = 2
    output_dim = 3
    pixels_at_boundary = True
    domain_length = 64.
    reverse_d1 = True
    data_paths_valid = ('./data/mechanics/test/valid/fields/')
    data_paths_test_level_1 = ('./data/mechanics/test/test_level_1/fields/')
    data_paths_test_level_2 = ('./data/mechanics/test/test_level_2/fields/')
    bcs = 'none'
    pixels_per_dim = 64
    return_optimizer = True
    return_inequality = True
    ds_valid = Dataset_Paths(data_paths_valid, use_double=use_double)
    ds_test_level_1 = Dataset_Paths(data_paths_test_level_1, use_double=use_double)
    ds_test_level_2 = Dataset_Paths(data_paths_test_level_2, use_double=use_double)
    train_batch_size = 5
    dl_valid = cycle(DataLoader(ds_valid, batch_size=train_batch_size, shuffle=False))
    dl_test_level_1 = DataLoader(ds_test_level_1, batch_size=train_batch_size, shuffle=True)
    dl_test_level_2 = DataLoader(ds_test_level_2, batch_size=train_batch_size, shuffle=True)
    sigmoid_last_channel = True
else:
    raise ValueError('Unknown governing equations.')

# Output dirs rooted at output_base_dir (no auto-increment — path is fully caller-controlled)
output_save_dir_validation = os.path.join(output_base_dir, f'validation/step_{load_model_step}/')
os.makedirs(output_save_dir_validation, exist_ok=True)

if use_double:
    torch.set_default_dtype(torch.float64)

diffusion_utils = DenoisingDiffusion(diff_steps, device, residual_grad_guidance)

if gov_eqs == 'darcy':
    model = Unet3D(dim=32, channels=output_dim, sigmoid_last_channel=sigmoid_last_channel).to(device)
elif gov_eqs == 'mechanics':
    model = Unet3D(dim=128, channels=output_dim+3+4, out_dim=output_dim, sigmoid_last_channel=sigmoid_last_channel).to(device)

load_model(Path(load_path, 'model', 'checkpoint_' + str(load_model_step) + '.pt'), model)

if gov_eqs == 'darcy':
    residuals = ResidualsDarcy(model=model, fd_acc=fd_acc, pixels_per_dim=pixels_per_dim,
                               pixels_at_boundary=pixels_at_boundary, reverse_d1=reverse_d1,
                               device=device, bcs=bcs, domain_length=domain_length,
                               residual_grad_guidance=residual_grad_guidance,
                               use_ddim_x0=use_ddim_x0, ddim_steps=ddim_steps)
elif gov_eqs == 'mechanics':
    residuals = ResidualsMechanics(model=model, pixels_per_dim=pixels_per_dim,
                                   pixels_at_boundary=pixels_at_boundary, device=device,
                                   bcs=bcs, no_BC_folder='./data/mechanics/solidspy_k_no_BC/',
                                   topopt_eval=topopt_eval, use_ddim_x0=use_ddim_x0,
                                   ddim_steps=ddim_steps)

num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Number of trainable parameters: {num_params}')

if gov_eqs == 'darcy':
    conditioning_input = None
    sample_shape = (no_samples, output_dim, pixels_per_dim, pixels_per_dim)
elif gov_eqs == 'mechanics':
    cur_batch = next(dl_valid).to(device)
    if cur_batch.shape[0] < no_samples:
        no_samples = cur_batch.shape[0]
    sample_shape = (no_samples, output_dim, pixels_per_dim+1, pixels_per_dim+1)
    cur_batch = cur_batch[torch.randperm(cur_batch.shape[0], device=device)[:no_samples]]
    conditioning, x_0, bcs = torch.tensor_split(cur_batch, (3, 6), dim=1)
    conditioning_input = (conditioning, bcs, x_0)

output = diffusion_utils.p_sample_loop(conditioning_input, sample_shape,
                        save_output=save_output, surpress_noise=True,
                        use_dynamic_threshold=use_dynamic_threshold,
                        residual_func=residuals, eval_residuals=eval_residuals,
                        return_optimizer=return_optimizer, return_inequality=return_inequality,
                        M_correction=M_correction, N_correction=N_correction,
                        correction_mode=correction_mode)

if eval_residuals:
    seqs = output[0]
    residual = output[1]['residual']
    residual = residual.abs().mean(dim=tuple(range(1, residual.ndim)))
    if return_optimizer:
        optimized_quant = output[1]['optimized_quant']
    if return_inequality:
        ineq = output[1]['inequality_quant']
else:
    seqs = output

if gov_eqs == 'mechanics':
    cond_data = torch.cat((conditioning, x_0, bcs), dim=1)
    for cur_sample in range(no_samples):
        for channel_idx in range(cond_data.shape[1]):
            os.makedirs(output_save_dir_validation + f'sample_{cur_sample}/', exist_ok=True)
            np.savetxt(output_save_dir_validation + f'sample_{cur_sample}/cond_channel_{channel_idx}.csv',
                       cond_data[cur_sample, channel_idx].detach().cpu().numpy(), delimiter=',')

labels = ['sample', 'model_output']
for seq_idx, seq in enumerate(seqs):
    if seq_idx == 1:
        continue
    seq = torch.stack(seq, dim=0)
    if len(seq.shape) == 6:
        seq = seq.squeeze(-3)
    last_preds = seq[-1].numpy()
    sel_samples = np.arange(no_samples)
    channels = np.arange(output_dim)
    for sel_sample in sel_samples:
        for sel_channel in channels:
            last_pred = last_preds[sel_sample, sel_channel]
            last_pred_normalized = (last_pred - last_pred.min()) / (last_pred.max() - last_pred.min())
            image = np.uint8(last_pred_normalized * 255)
            fig, ax = plt.subplots()
            ax.imshow(image, cmap='gray', vmin=0, vmax=255)
            ax.axis('off')
            if eval_residuals:
                title = f'residual: {residual[sel_sample]:.2e}'
                if return_optimizer:
                    title += f'\nopt: {optimized_quant[sel_sample]:.2f}'
                if return_inequality:
                    title += f'\nineq: {ineq[sel_sample]:.2e}'
                plt.title(title, color='green')
            filename = labels[seq_idx] + '_sample_' + str(sel_sample) + '_' + str(sel_channel) + '.png'
            plt.savefig(output_save_dir_validation + filename, bbox_inches='tight', pad_inches=0)
            plt.close(fig)
            os.makedirs(output_save_dir_validation + f'/sample_{sel_sample}/', exist_ok=True)
            np.savetxt(output_save_dir_validation + f'/sample_{sel_sample}/' + labels[seq_idx] + '_' + str(sel_channel) + '.csv',
                       last_pred, delimiter=',')

if eval_residuals:
    residuals_array = residual.detach().cpu().numpy()
    ineq_array = ineq.detach().cpu().numpy() if return_inequality else None
    optimized_quant_array = optimized_quant.detach().cpu().numpy() if return_optimizer else None

    df_data = {'Sample Index': list(range(no_samples)) + ['Mean'],
               'Residuals (abs)': list(residuals_array)}
    if return_optimizer:
        df_data['Optimized quantity'] = list(optimized_quant_array)
    if return_inequality:
        df_data['Inequality'] = list(ineq_array)
    df_data['Residuals (abs)'].append(np.nanmean(residuals_array))
    if return_optimizer:
        df_data['Optimized quantity'].append(np.nanmean(optimized_quant_array))
    if return_inequality:
        df_data['Inequality'].append(np.nanmean(ineq_array))
    df = pd.DataFrame(df_data)
    df.to_csv(os.path.join(output_save_dir_validation, 'sample_statistics.csv'), index=False)

# Full test-set evaluation for mechanics (in-distribution = test_level_1, OOD = test_level_2)
with torch.no_grad():
    start_time = time.time()
    if eval_test_sets and gov_eqs == 'mechanics':
        test_datasets = [dl_test_level_1, dl_test_level_2]
        test_datasets_names = ['test_level_1', 'test_level_2']
        for ds_test_idx, dl_test in enumerate(test_datasets):
            residual_mean_abs_list, rel_CE_error_list, rel_vf_error_list, fm_error_list = [], [], [], []
            for batch_idx, batch in enumerate(dl_test):
                cur_batch = batch.to(device)
                sample_shape = (cur_batch.shape[0], output_dim, pixels_per_dim+1, pixels_per_dim+1)
                conditioning, x_0, bcs = torch.tensor_split(cur_batch, (3, 6), dim=1)
                conditioning_input = (conditioning, bcs, x_0)
                output = diffusion_utils.p_sample_loop(conditioning_input, sample_shape,
                                        save_output=save_output, surpress_noise=True,
                                        use_dynamic_threshold=use_dynamic_threshold,
                                        residual_func=residuals, eval_residuals=eval_residuals,
                                        return_optimizer=return_optimizer, return_inequality=return_inequality,
                                        M_correction=M_correction, N_correction=N_correction,
                                        correction_mode=correction_mode)
                if eval_residuals:
                    seqs = output[0]
                    residual = output[1]['residual']
                    residual = residual.abs().mean(dim=tuple(range(1, residual.ndim)))
                    if return_optimizer:
                        optimized_quant = output[1]['optimized_quant']
                    if return_inequality:
                        ineq = output[1]['inequality_quant']
                else:
                    seqs = output
                output_save_dir_tests = os.path.join(output_base_dir, test_datasets_names[ds_test_idx]) + '/'
                os.makedirs(output_save_dir_tests, exist_ok=True)
                if batch_idx == 0:
                    labels = ['sample', 'model_output']
                    for seq_idx, seq in enumerate(seqs):
                        if seq_idx == 1:
                            continue
                        seq = torch.stack(seq, dim=0)
                        if len(seq.shape) == 6:
                            seq = seq.squeeze(-3)
                        last_preds = seq[-1].numpy()
                        sel_samples = np.arange(len(last_preds))
                        channels = np.arange(output_dim)
                        for sel_sample in sel_samples:
                            cond_data = torch.cat((conditioning, x_0, bcs), dim=1)[sel_sample]
                            for channel_idx in range(cond_data.shape[0]):
                                os.makedirs(output_save_dir_tests + f'/sample_{sel_sample}/', exist_ok=True)
                                np.savetxt(output_save_dir_tests + f'/sample_{sel_sample}/cond_channel_{channel_idx}.csv',
                                           cond_data[channel_idx].detach().cpu().numpy(), delimiter=',')
                            for sel_channel in channels:
                                last_pred = last_preds[sel_sample, sel_channel]
                                last_pred_normalized = (last_pred - last_pred.min()) / (last_pred.max() - last_pred.min())
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
                                plt.savefig(output_save_dir_tests + filename, bbox_inches='tight', pad_inches=0)
                                plt.close(fig)
                                os.makedirs(output_save_dir_tests + f'/sample_{sel_sample}/', exist_ok=True)
                                np.savetxt(output_save_dir_tests + f'/sample_{sel_sample}/' + labels[seq_idx] + '_' + str(sel_channel) + '.csv',
                                           last_pred, delimiter=',')

                if eval_residuals:
                    residuals_array = residual.detach().cpu().numpy()
                    residual_mean_abs_list.append(residuals_array)
                if topopt_eval:
                    rel_CE_error = output[1]['rel_CE_error_full_batch'].detach().cpu().numpy()
                    rel_vf_error = output[1]['vf_error_full_batch'].detach().cpu().numpy()
                    fm_error = output[1]['fm_error_full_batch'].detach().cpu().numpy()
                    rel_CE_error_list.append(rel_CE_error)
                    rel_vf_error_list.append(rel_vf_error)
                    fm_error_list.append(fm_error)

                if test_batches != -1 and batch_idx > test_batches:
                    break

            if eval_residuals:
                residuals_array = np.concatenate(residual_mean_abs_list, axis=0)
                np.savetxt(output_save_dir_tests + 'residuals.csv', residuals_array, delimiter=',')
            if topopt_eval:
                rel_CE_error = np.concatenate(rel_CE_error_list, axis=0)
                rel_vf_error = np.concatenate(rel_vf_error_list, axis=0)
                fm_error = np.concatenate(fm_error_list, axis=0)
                np.savetxt(output_save_dir_tests + 'rel_CE_error.csv', rel_CE_error, delimiter=',')
                np.savetxt(output_save_dir_tests + 'rel_vf_error.csv', rel_vf_error, delimiter=',')
                np.savetxt(output_save_dir_tests + 'fm_error.csv', fm_error, delimiter=',')

            print(f'Evaluation of {name}: on {test_datasets_names[ds_test_idx]}.')
            print('CE median error:', np.median(rel_CE_error),
                  'VF mean error:', np.mean(rel_vf_error),
                  'FM mean error:', np.mean(fm_error),
                  'Mean residual:', np.mean(residuals_array),
                  'Median residual:', np.median(residuals_array))

    end_time = time.time()
    print(f'Evaluation for model {name} done (time: {time.strftime("%H:%M:%S", time.gmtime(end_time - start_time))}).')
