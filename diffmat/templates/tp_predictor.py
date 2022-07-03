import os
import sys
import random
import logging
import imageio
import configargparse

import numpy as np
import torch as th
import torch.nn as nn
import matplotlib.pyplot as plt
from torchsummary import summary
from skimage.transform import resize
from torchvision.utils import make_grid

from GRAPH_NAME_util import *
from diffmat.sbs_core.util import *
from diffmat.util.render import render
from diffmat.util.model import ParamPredNet
from diffmat.util.descriptor import TextureDescriptor
# ---------------------------------------------------------------------------- #


# load configuration
current_path = os.path.dirname(os.path.abspath(__file__))
p = configargparse.ArgumentParser(default_config_files = [os.path.join(current_path, 'configs', 'predictor.conf')])
p.add('-c', '--config_filepath', required=False, is_config_file=True, help='Path to config file')

p.add_argument('--use_real_image', action='store_true', help='Evaluate real image input')
p.add_argument('--real_image_paths', type=str, nargs='+', default=[], help='Real image path')

p.add_argument('--eval_mode', action='store_true', help='Evaluation mode')
p.add_argument('--load_model', action='store_true', help='Load pretrained model')
p.add_argument('--load_epoch', type=int, default=0, help='Epoch of the loaded model')

p.add_argument('--load_grid_parameter', action='store_true', help='Load target from saved grid')
p.add_argument('--grid_id', type=int, default='0', help='Grid index')
p.add_argument('--grid_example_ids', type=int, nargs='+', default=[0], help='Examples indices in the selected grid; -1 sets all examples')
p.add_argument('--grid_param_pkl_path', type=str, default=os.path.join('.','grid', 'sampling_grid_%d.pkl'), help='Path to load the saved grid parameter')

p.add_argument('--save_png', action='store_true', help='Save results as png images instead of exr images')
p.add_argument('--log_debug', action='store_true', help='Set log output to debug mode')
p.add_argument('--use_cpu', action='store_true', help='Use CPU as computing device')

p.add_argument('--params_ptb_max', type=float, default=0.05, help='Max perturbation (as percentage/100) to node parameters')
p.add_argument('--params_ptb_min', type=float, default=-0.05, help='Min perturbation (as percentage/100) to node parameters')
p.add_argument('--params_ptb_mu', type=float, default=0.0, help='Mean shift of perturbation (sampling with normal distribution)')
p.add_argument('--params_ptb_sigma', type=float, default=0.03, help='Standard deviation of perturbation (sampling with normal distribution)')
p.add_argument('--params_normal_sampling_func', action='store_true', help='Sampling function for node parameters')

p.add_argument('--lambertian_BRDF', action='store_true', help='Use lambertian BRDF')
p.add_argument('--render_ptb_max', type=float, default=0.05, help='Max perturbation (as percentage/100) to render parameters.')
p.add_argument('--render_ptb_min', type=float, default=-0.05, help='Min perturbation (as percentage/100) to render parameters.')
p.add_argument('--render_ptb_mu', type=float, default=0.0, help='Mean shift of perturbation (sampling with normal distribution)')
p.add_argument('--render_ptb_sigma', type=float, default=0.03, help='Standard deviation of perturbation (sampling with normal distribution)')
p.add_argument('--render_normal_sampling_func', action='store_true', help='Sampling function for render parameters')

p.add_argument('--in_width', type=int, default=RES_W, help='Input image width')
p.add_argument('--in_height', type=int, default=RES_H, help='Input image height')
p.add_argument('--num_train_pyramid', type=int, default=0, help='Number of layers in image pyramid')
p.add_argument('--num_graph_parameters', type=int, default=GRAPH_NAME_num_trainable_params[GRAPH_NAME_optimization_level], help='Number of trainable graph parameters')
p.add_argument('--fix_vgg', action='store_true', help='Fix vgg network')

p.add_argument('--param_batch', type=int, default=5, help='Number of samples in the batch when only use parameter loss')
p.add_argument('--td_batch', type=int, default=1, help='Number of samples in the batch when use parameter loss plus td loss')
p.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
p.add_argument('--lr_decay_exp', type=float, default=0.97, help='Learning rate decay')
p.add_argument('--lr_decay_div_multi', type=float, default=0.5, help='Learning rate decay division multiplier')
p.add_argument('--num_epoch', type=int, default=150, help='Number of epochs')
p.add_argument('--switch_epoch', type=int, default=100, help='The epoch at when to switch to parameter loss plus td loss')
p.add_argument('--per_epoch_iter', type=int, default=100, help='Iterations per epoch')
p.add_argument('--num_validation_batch', type=int, default=1, help='Number of validation batches')
p.add_argument('--num_loss_pyramid', type=int, default=2, help='Number of pyramid levels used in the td loss')
p.add_argument('--img_loss_weight', type=float, default=0.1, help='td loss weight')
p.add_argument('--param_loss_weight', type=float, default=1.0, help='parameter loss weight')
p.add_argument('--render_param_weight', type=float, default=1.0, help='render parameter loss weight')
p.add_argument('--use_l1', action='store_true', help='Use l1 loss')
p.add_argument('--use_td_loss', action='store_true', help='Use td loss')
p.add_argument('--use_same_dict', action='store_true', help='Use same input noise seed')
p.add_argument('--save_itv', type=int, default=10, help='Save interval (epoch) of model')

p.add_argument('--model_name', type=str, default='epoch_%d.pth', help='Model name')
p.add_argument('--model_dir', type=str, default=os.path.join('.', 'models'), help='Directory to save the model')

p.add_argument('--save_validation_results', action='store_true', help='Save validation results.')
p.add_argument('--show_validation_results', action='store_true', help='Show validation results')
p.add_argument('--show_time', type=int, default=20, help='Show time for validation results')
p.add_argument('--num_imgs_per_grid', type=int, default=10, help='Number of images per validation grid')
p.add_argument('--num_cols', type=int, default=10, help='Number of columns in validation grid')
p.add_argument('--padding', type=int, default=15, help='Paddings between images')
p.add_argument('--validation_output_name', type=str, default='epoch_%d', help='Validation image grid output name')
p.add_argument('--validation_params_name', type=str, default='params_epoch_%d', help='Validation parameters grid output name')

locals().update(p.parse_args().__dict__)
# ---------------------------------------------------------------------------- #

# setup computing device
logging.basicConfig(stream=sys.stderr, level=logging.DEBUG if log_debug else logging.INFO)
# enable GPU
if th.cuda.is_available() and not use_cpu:
    device_name = 'cuda'
    device_full_name = 'cuda:DEVICE_COUNT'
    device = th.device(device_full_name)
    th.set_default_tensor_type('torch.cuda.FloatTensor')
    logging.info('Use GPU')
else:
    device_name = 'cpu'
    device_full_name = 'cpu'
    device = th.device(device_full_name)
    th.set_default_tensor_type('torch.FloatTensor')
    logging.info('Use CPU')


# parameter sampling specs
params_spec = {
    'total_var' : GRAPH_NAME_num_nodes,
    'num_free_nodes' : GRAPH_NAME_num_nodes,
    'ptb_max' : params_ptb_max,
    'ptb_min' : params_ptb_min,
    'mu': params_ptb_mu,
    'sigma': params_ptb_sigma,
}

render_params_spec = {
    'ptb_max' : render_ptb_max,
    'ptb_min' : render_ptb_min,
    'mu': render_ptb_mu,
    'sigma': render_ptb_sigma,
}
params_sampling_func = param_sampling_normal if params_normal_sampling_func else param_sampling_uniform
render_sampling_func = param_sampling_normal if render_normal_sampling_func else param_sampling_uniform
num_random_seed = NOISE_COUNT


# create some useful parameters
grid_param_pkl_path = grid_param_pkl_path % grid_id


# create model
sample_input_shape = (3, in_width, in_height)
num_out_parameters = num_graph_parameters+2 if GRAPH_NAME_render_params_trainable else num_graph_parameters
net = ParamPredNet(3, in_width, in_height, num_out_parameters)
td = net.build_fc_after_td(device, num_train_pyramid, fix_vgg)
if not fix_vgg:
    td = TextureDescriptor(device)
    # fix parameters for evaluation
    for param in td.parameters():
        param.requires_grad = False
summary(net, sample_input_shape, device=device_name)
net.to(device)


# create optimizer and loss criterion
optimizer = th.optim.Adam(net.parameters(), lr=lr, eps=1e-6)
criterion = nn.functional.l1_loss if use_l1 else nn.functional.mse_loss


# create model save/load path
if fix_vgg:
    postfix = 'fix_vgg'
else:
    postfix = 'trainable_vgg'
loss_key = 'num_params_%d_opt_mode_%d_%s' % (num_out_parameters, GRAPH_NAME_optimization_level, postfix)
model_dir = os.path.join(model_dir, loss_key)
# ---------------------------------------------------------------------------- #


# load pre-trained model
load_model = load_model or eval_mode
if load_model:
    if os.path.exists(os.path.join(model_dir, model_name % (load_epoch))):
        checkpoint = th.load(os.path.join(model_dir, model_name % (load_epoch)))
        # check if this if else is necessary
        if fix_vgg:
            net.fc_net.load_state_dict(checkpoint['model_state_dict'])
        else:
            net.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = load_epoch + 1
        print('Epoch %d loaded' % (load_epoch))
    else:
        if eval_mode:
            raise ValueError('unfound model during evaluation mode')
        else:
            start_epoch = 0
else:
    start_epoch = 0
# ---------------------------------------------------------------------------- #


def get_one_input_dict_batch(batch):
    """Get one batch of input noises lists

    Args:
        batch (int): batch size

    Returns:
        list: list of input noises lists
    """
    input_dict_batch = []
    # create batch
    for i in range(batch):
        input_dict = get_input_dict(random.choices(range(num_random_seed),
                                    k=len(input_names)),
                                    source_random_base_dir,
                                    input_names,
                                    device,
                                    GRAPH_NAME_use_alpha)
        input_dict_batch.append(input_dict)
        return input_dict_batch


def get_one_batch(render_params, batch):
    """Generate one batch of graph parameters

    Args:
        render_params (dict): render parameters dictionary
        batch (int): batch size

    Returns:
        list: list of rendered images tensors
        list: list of graph parameters tensors
        list: list of input noises lists
    """    
    in_batch = []
    out_batch = []
    input_dict_batch = []
    # create batch
    for i in range(batch):
        GRAPH_NAME_params_rand, GRAPH_NAME_exposed_params_rand, GRAPH_NAME_swapped_params_rand = gen_rand_params(GRAPH_NAME_params,
                                                        params_spec,
                                                        GRAPH_NAME_trainable_list,
                                                        GRAPH_NAME_keys,
                                                        params_sampling_func,
                                                        res = RES_H,
                                                        exposed_params = None if GRAPH_NAME_optimization_level == 0 else GRAPH_NAME_exposed_params,
                                                        params_tp = None if GRAPH_NAME_optimization_level == 0 else GRAPH_NAME_params_tp,
                                                        gen_exposed_only = True if GRAPH_NAME_optimization_level == 2 else False,
                                                        output_level=2
                                                        )
        input_dict = get_input_dict(random.choices(range(num_random_seed),
                                    k=len(input_names)),
                                    source_random_base_dir,
                                    input_names,
                                    device,
                                    GRAPH_NAME_use_alpha)
        sample = GRAPH_NAME_forward(GRAPH_NAME_swapped_params_rand, **input_dict)
        sample_render_img = render((sample[1] - 0.5) * 2.0,
                                        sample[0],
                                        sample[3],
                                        sample[2],
                                        render_params['light_color'].to(device) * render_params['light_color_max'].to(device),
                                        render_params['f0'].to(device),
                                        render_params['size'].to(device) * render_params['size_max'].to(device),
                                        render_params['camera'].to(device),
                                        lambertian_BRDF
                                    )
        nan_test(sample_render_img, True)

        if GRAPH_NAME_optimization_level == 0:
            params = convert_params_dict_to_list(GRAPH_NAME_params_rand, GRAPH_NAME_keys, GRAPH_NAME_trainable_list)
        elif GRAPH_NAME_optimization_level == 1:
            params = convert_params_dict_to_list(GRAPH_NAME_swapped_params_rand, GRAPH_NAME_keys, GRAPH_NAME_trainable_list)
            params = np.append(params, convert_exposed_params_dict_to_list(GRAPH_NAME_exposed_params_rand))
        elif GRAPH_NAME_optimization_level == 2:
            params = convert_exposed_params_dict_to_list(GRAPH_NAME_exposed_params_rand)

        params = th.tensor(params, dtype=th.float32).to(device)

        in_batch.append(sample_render_img)
        out_batch.append(params)
        input_dict_batch.append(input_dict)

    in_batch = th.stack(in_batch, 0)
    out_batch = th.stack(out_batch, 0)
    return in_batch, out_batch, input_dict_batch


def convert_estimate_out_for_forward_eval(estimate_out_batch, keep_tensor, for_save=False):
    """Convert network prediction to parameter dictionaries

    Args:
        estimate_out_batch (tensor): 2-d tensor of network prediction
        keep_tensor (bool): keep the output value as tensor. If false, convert to list.
        for_save (bool, optional): set return values for saving checkpoints. If false, only return swapped parameters for training.

    Returns:
        for_save == False:
            list : list of swapped parameter dictionaries
        for_save == True:
            list : list of swapped parameter dictionaries
            list : list of node parameter dictionaries
            list : list of exposed parameter dictionary
    """
    if GRAPH_NAME_optimization_level == 0:
        estimate_out_dict_batch = convert_params_list_to_dict(estimate_out_batch, GRAPH_NAME_params, GRAPH_NAME_trainable_list, GRAPH_NAME_keys, keep_tensor=keep_tensor)
        if for_save:
            return estimate_out_dict_batch, estimate_out_dict_batch, [GRAPH_NAME_exposed_params]*len(estimate_out_dict_batch)
    elif GRAPH_NAME_optimization_level == 1:
        estimate_out_batch_trimmed_params = estimate_out_batch[:, :GRAPH_NAME_num_trainable_params[1]-GRAPH_NAME_num_trainable_params[2]]
        estimate_out_batch_exposed_params = estimate_out_batch[:, -GRAPH_NAME_num_trainable_params[2]:]
        estimate_out_trimmed_dict_batch = convert_params_list_to_dict(estimate_out_batch_trimmed_params, GRAPH_NAME_params, GRAPH_NAME_trainable_list, GRAPH_NAME_keys, GRAPH_NAME_params_tp, keep_tensor=keep_tensor)
        estimate_out_exposed_dict_batch = convert_exposed_params_list_to_dict(estimate_out_batch_exposed_params, GRAPH_NAME_exposed_params, keep_tensor=keep_tensor)
        estimate_out_dict_batch = []
        for estimate_out_exposed_dict, estimate_out_trimmed_dict in zip(estimate_out_exposed_dict_batch, estimate_out_trimmed_dict_batch):
            estimate_out_dict_batch.append(swap_tp(estimate_out_trimmed_dict, GRAPH_NAME_params_tp, estimate_out_exposed_dict))
        if for_save:
            estimate_out_trimmed_dict_batch_copy = convert_params_list_to_dict(estimate_out_batch_trimmed_params, GRAPH_NAME_params, GRAPH_NAME_trainable_list, GRAPH_NAME_keys, GRAPH_NAME_params_tp, keep_tensor=keep_tensor)
            return estimate_out_dict_batch, estimate_out_trimmed_dict_batch_copy, estimate_out_exposed_dict_batch
    elif GRAPH_NAME_optimization_level == 2:
        estimate_out_exposed_dict_batch = convert_exposed_params_list_to_dict(estimate_out_batch, GRAPH_NAME_exposed_params, keep_tensor=keep_tensor)
        estimate_out_dict_batch = []
        for estimate_out_exposed_dict in estimate_out_exposed_dict_batch:
            estimate_out_dict_batch.append(swap_tp(copy.deepcopy(GRAPH_NAME_params), GRAPH_NAME_params_tp, estimate_out_exposed_dict))
        if for_save:
            return estimate_out_dict_batch, [GRAPH_NAME_params]*len(estimate_out_exposed_dict_batch), estimate_out_exposed_dict_batch

    return estimate_out_dict_batch

def convert_render_params_list_to_dict(render_params_list):
    """Convert predicted render parameters list to dictionary

    Args:
        render_params_list (list): a list of two parameters, first 'light color', second 'size'

    Returns:
        Dict: render parameters dictionary
    """
    render_params = copy.deepcopy(GRAPH_NAME_render_params)
    render_params['light_color'] = render_params_list[0]
    render_params['size'] = render_params_list[1]
    return render_params

def eval_for_grid_parameter():
    """Evaluate example from saved grid
    """
    grid_results_dir = os.path.join(model_dir, 'grid_results')
    if not os.path.exists(grid_results_dir):
        os.makedirs(grid_results_dir)

    out_params_matrix = []    
    with open(grid_param_pkl_path, 'rb') as f:
        in_params_matrix = th.load(f)

    if grid_example_ids == [-1]:
        example_ids = range(len(in_params_matrix))
    else:
        example_ids = grid_example_ids

    for j in example_ids:
        visualization_list = []
        out_params_matrix.append(in_params_matrix[j])
        GRAPH_NAME_swapped_params_rand = swap_tp(in_params_matrix[j][0], GRAPH_NAME_params_tp, in_params_matrix[j][1])
        if GRAPH_NAME_render_params_trainable:
            GRAPH_NAME_render_params_rand = in_params_matrix[j][2]
        input_dict_batch = get_one_input_dict_batch(1)[0]
        in_sample = GRAPH_NAME_forward(GRAPH_NAME_swapped_params_rand, **input_dict_batch)
        in_sample_render_img = render((in_sample[1] - 0.5) * 2.0,
                                                in_sample[0],
                                                in_sample[3],
                                                in_sample[2],
                                                GRAPH_NAME_render_params_rand['light_color'].to(device)*GRAPH_NAME_render_params['light_color_max'].to(device) if GRAPH_NAME_render_params_trainable \
                                                    else GRAPH_NAME_render_params['light_color'].to(device)*GRAPH_NAME_render_params['light_color_max'].to(device),
                                                GRAPH_NAME_render_params['f0'].to(device),
                                                GRAPH_NAME_render_params_rand['size'].to(device)*GRAPH_NAME_render_params['size_max'].to(device) if GRAPH_NAME_render_params_trainable \
                                                    else GRAPH_NAME_render_params['size'].to(device)*GRAPH_NAME_render_params['size_max'].to(device),
                                                GRAPH_NAME_render_params['camera'].to(device),
                                                lambertian_BRDF
                                                )
        visualization_list.append(in_sample_render_img)
        estimate_out_batch = net.forward(in_sample_render_img.unsqueeze(0))
        if GRAPH_NAME_render_params_trainable:
            estimate_render_params = estimate_out_batch[:,num_graph_parameters:]
            estimate_out_batch = estimate_out_batch[:,:num_graph_parameters]
        estimate_out_dict_batch, estimate_params_dict_batch, estimate_exposed_params_dict_batch = convert_estimate_out_for_forward_eval(estimate_out_batch, False, True)
        estimated_sample = GRAPH_NAME_forward(estimate_out_dict_batch[0], **input_dict_batch)
        estimated_sample_render_img = render((estimated_sample[1] - 0.5) * 2.0,
                                                estimated_sample[0],
                                                estimated_sample[3],
                                                estimated_sample[2],
                                                estimate_render_params[0,0]*GRAPH_NAME_render_params['light_color_max'].to(device) if GRAPH_NAME_render_params_trainable \
                                                    else GRAPH_NAME_render_params['light_color'].to(device)*GRAPH_NAME_render_params['light_color_max'].to(device),
                                                GRAPH_NAME_render_params['f0'].to(device),
                                                estimate_render_params[0,1]*GRAPH_NAME_render_params['size_max'].to(device) if GRAPH_NAME_render_params_trainable \
                                                    else GRAPH_NAME_render_params['size'].to(device)*GRAPH_NAME_render_params['size_max'].to(device),
                                                GRAPH_NAME_render_params['camera'].to(device),
                                                lambertian_BRDF
                                                )
        visualization_list.append(estimated_sample_render_img)

        # add to out_params_matrix
        if GRAPH_NAME_render_params_trainable:
            estimate_render_params = estimate_render_params.squeeze().detach().cpu()
            estimated_entry = [estimate_params_dict_batch[0], estimate_exposed_params_dict_batch[0], convert_render_params_list_to_dict(estimate_render_params)]
        else:
            estimated_entry = [estimate_params_dict_batch[0], estimate_exposed_params_dict_batch[0]]

        # first save estimated parameters, then save target parameters
        out_params_matrix.append(estimated_entry)
        out_params_matrix.append(in_params_matrix[j])

        grid = make_grid(visualization_list, nrow=2, padding=padding, normalize=False, range=None, scale_each=False, pad_value=0)
        np_grid = np.transpose(grid.detach().cpu().numpy(), [1,2,0])
        if save_png:
            imageio.imwrite(os.path.join(grid_results_dir, validation_output_name % (load_epoch) + '_grid_%d_example_%d.png' % (grid_id, j)), (np_grid * 255.0).astype(np.uint8))
        else:
            imageio.imwrite(os.path.join(grid_results_dir, validation_output_name % (load_epoch) + '_grid_%d_example_%d.exr' % (grid_id, j)), np_grid)

    # save output matrix
    with open(os.path.join(grid_results_dir, validation_params_name % (load_epoch) + '_grid_%d.pkl' % (grid_id)), 'wb') as f:
        th.save(out_params_matrix,f)


def eval_real_world_image():
    """ Evaludate real world image
    """
    real_world_results_dir = os.path.join(model_dir, 'real_world_results')
    if not os.path.exists(real_world_results_dir):
        os.makedirs(real_world_results_dir)

    for real_image_path in real_image_paths:
        visualization_list = []
        real_image_postfix = os.path.splitext(os.path.basename(real_image_path))[0]
        in_batch = th.tensor(resize(read_image(real_image_path), (in_height, in_width), mode='constant', anti_aliasing=True), dtype=th.float32).permute(2,0,1).unsqueeze(0).to(device)
        input_dict_batch = get_one_input_dict_batch(1)[0]
        estimate_out_batch = net.forward(in_batch)
        if GRAPH_NAME_render_params_trainable:
            estimate_render_params = estimate_out_batch[:,num_graph_parameters:]
            estimate_out_batch = estimate_out_batch[:,:num_graph_parameters]
        estimate_out_dict_batch, estimate_params_dict_batch, estimate_exposed_params_dict_batch = convert_estimate_out_for_forward_eval(estimate_out_batch, False, True)
        estimated_sample = GRAPH_NAME_forward(estimate_out_dict_batch[0], **input_dict_batch)
        estimated_sample_render_img = render((estimated_sample[1] - 0.5) * 2.0,
                                                    estimated_sample[0],
                                                    estimated_sample[3],
                                                    estimated_sample[2],
                                                    estimate_render_params[0,0]*GRAPH_NAME_render_params['light_color_max'].to(device) if GRAPH_NAME_render_params_trainable \
                                                        else GRAPH_NAME_render_params['light_color'].to(device)*GRAPH_NAME_render_params['light_color_max'].to(device),
                                                    GRAPH_NAME_render_params['f0'].to(device),
                                                    estimate_render_params[0,1]*GRAPH_NAME_render_params['size_max'].to(device) if GRAPH_NAME_render_params_trainable \
                                                        else GRAPH_NAME_render_params['size'].to(device)*GRAPH_NAME_render_params['size_max'].to(device),
                                                    GRAPH_NAME_render_params['camera'].to(device),
                                                    lambertian_BRDF
                                                    )
        visualization_list.append(in_batch.squeeze())
        visualization_list.append(estimated_sample_render_img)
        grid = make_grid(visualization_list, nrow=2, padding=padding, normalize=False, range=None, scale_each=False, pad_value=0)
        np_grid = np.transpose(grid.detach().cpu().numpy(), [1,2,0])
        if save_png:
            imageio.imwrite(os.path.join(real_world_results_dir, validation_output_name % (load_epoch) + '_%s.png' % (real_image_postfix)), (np_grid * 255.0).astype(np.uint8))
        else:
            imageio.imwrite(os.path.join(real_world_results_dir, validation_output_name % (load_epoch) + '_%s.exr' % (real_image_postfix)), np_grid)

        # save output matrix
        with open(os.path.join(real_world_results_dir, validation_params_name % (load_epoch) + '_%s.pkl' % (real_image_postfix)), 'wb') as f:
            if GRAPH_NAME_render_params_trainable:
                estimate_render_params = estimate_render_params.squeeze().detach().cpu()
                th.save([[estimate_params_dict_batch[0], estimate_exposed_params_dict_batch[0], convert_render_params_list_to_dict(estimate_render_params)]], f)
            else:
                th.save([[estimate_params_dict_batch[0], estimate_exposed_params_dict_batch[0]]],f)


def train():
    """ Train parameter prediction network
    """
    add_td_loss = False
    for cur_epoch in range(start_epoch, num_epoch):
        # swtich loss by adding td loss
        if cur_epoch >= switch_epoch:
            batch = td_batch
            add_td_loss = True
        else:
            batch = param_batch

        # learning rate decay
        for g in optimizer.param_groups:
            g['lr'] = 1.0 / (1.0 + cur_epoch * lr_decay_div_multi) * lr

        # train the epoch
        for cur_iter in range(per_epoch_iter):
            # start online training
            net.zero_grad()

            # get lighting
            if GRAPH_NAME_render_params_trainable:
                render_params = gen_rand_light_params(GRAPH_NAME_render_params, render_params_spec, render_sampling_func)
            else:
                render_params = GRAPH_NAME_render_params

            # get one batch
            in_batch, out_batch, input_dict_batch = get_one_batch(render_params, batch)

            # feedforward the network
            estimate_out_batch = net.forward(in_batch.to(device))
            if GRAPH_NAME_render_params_trainable:
                estimate_render_params = estimate_out_batch[:,num_graph_parameters:]
                estimate_out_batch = estimate_out_batch[:,:num_graph_parameters]

            # compute error and back-propagation
            err = criterion(estimate_out_batch, out_batch) * param_loss_weight
            if GRAPH_NAME_render_params_trainable:
                err = err + criterion(estimate_render_params, th.cat([render_params['light_color'].to(device), render_params['size'].to(device)], dim=0).expand(batch, 2)) * render_param_weight

            # compute rendering error
            if use_td_loss or add_td_loss:
                if use_same_dict:
                    estimate_input_dict_batch = input_dict_batch
                else:
                    estimate_input_dict_batch = get_one_input_dict_batch(batch)

                # convert output to parameters dictionary for forward evaluation
                estimate_out_dict_batch = convert_estimate_out_for_forward_eval(estimate_out_batch, True)

                estimate_in_batch = []
                for j in range(batch):
                    estimated_sample = GRAPH_NAME_forward(estimate_out_dict_batch[j], **estimate_input_dict_batch[j])
                    estimated_sample_render_img = render((estimated_sample[1] - 0.5) * 2.0,
                                                                estimated_sample[0],
                                                                estimated_sample[3],
                                                                estimated_sample[2],
                                                                estimate_render_params[j,0]*GRAPH_NAME_render_params['light_color_max'].to(device) if GRAPH_NAME_render_params_trainable \
                                                                    else GRAPH_NAME_render_params['light_color'].to(device)*GRAPH_NAME_render_params['light_color_max'].to(device),
                                                                GRAPH_NAME_render_params['f0'].to(device),
                                                                estimate_render_params[j,1]*GRAPH_NAME_render_params['size_max'].to(device) if GRAPH_NAME_render_params_trainable \
                                                                    else GRAPH_NAME_render_params['size'].to(device)*GRAPH_NAME_render_params['size_max'].to(device),
                                                                GRAPH_NAME_render_params['camera'].to(device),
                                                                lambertian_BRDF
                                                            )
                    nan_test(estimated_sample, True)
                    estimate_in_batch.append(estimated_sample_render_img)
                estimate_in_batch = th.stack(estimate_in_batch, 0)


                in_batch_rendering = in_batch if use_rendering_only else in_batch[:,8:,:,:]
                in_batch_td= td.eval_CHW_tensor(in_batch_rendering)
                for scale in range(num_loss_pyramid):
                    in_batch_td_= td.eval_CHW_tensor(nn.functional.interpolate(in_batch_rendering, scale_factor = 1.0/(2.0**(scale+1)), mode='bilinear', align_corners=True))
                    in_batch_td = th.cat([in_batch_td, in_batch_td_], dim=1)

                estimate_in_batch_rendering = estimate_in_batch if use_rendering_only else estimate_in_batch[:,8:,:,:]
                estimate_in_batch_td= td.eval_CHW_tensor(estimate_in_batch_rendering)
                for scale in range(num_loss_pyramid):
                    estimate_in_batch_td_= td.eval_CHW_tensor(nn.functional.interpolate(estimate_in_batch_rendering, scale_factor = 1.0/(2.0**(scale+1)), mode='bilinear', align_corners=True))
                    estimate_in_batch_td = th.cat([estimate_in_batch_td, estimate_in_batch_td_], dim=1)

                img_err = criterion(in_batch_td, estimate_in_batch_td)
                err = err + img_err * img_loss_weight

            err.backward()
            optimizer.step()

            # print error
            print('Epoch %d iter %d, total err: %.4f' % (cur_epoch, cur_iter, err))

        # one epoch finishes
        # 1) save model
        if cur_epoch % save_itv == 0:
            print('Epoch %d ends, save model' % (cur_epoch))
            if not os.path.exists(os.path.join(model_dir)):
                os.makedirs(os.path.join(model_dir))
            th.save({'model_state_dict': net.fc_net.state_dict() if fix_vgg else net.state_dict(),
                        'epoch': cur_epoch},
                    os.path.join(model_dir, model_name % (cur_epoch))
                    )
        else:
            print('Epoch %d ends' % (cur_epoch))

        # 2) do validation
        visualization_list = []
        out_params_matrix = []
        eval_params_list = []
        grid_count = 0
        mean_err = 0
        count = 0

        for i in range(num_validation_batch):
            # get lighting
            if GRAPH_NAME_render_params_trainable:
                render_params = gen_rand_light_params(GRAPH_NAME_render_params, render_params_spec, render_sampling_func)
            else:
                render_params = GRAPH_NAME_render_params

            # get one batch
            in_batch, out_batch, input_dict_batch = get_one_batch(render_params, batch)
            estimate_out_batch = net.forward(in_batch.to(device))

            if GRAPH_NAME_render_params_trainable:
                estimate_render_params = estimate_out_batch[:,num_graph_parameters:]
                estimate_out_batch = estimate_out_batch[:,:num_graph_parameters]

            # save output and estimated output params
            out_dict_batch, params_dict_batch, exposed_params_dict_batch = convert_estimate_out_for_forward_eval(out_batch, False, True)
            estimate_out_dict_batch, estimate_params_dict_batch, estimate_exposed_params_dict_batch = convert_estimate_out_for_forward_eval(estimate_out_batch, False, True)
            for j in range(batch):
                if GRAPH_NAME_render_params_trainable:
                    # first save estimated parameters, then save target parameters
                    out_params_matrix.append([estimate_params_dict_batch[j], estimate_exposed_params_dict_batch[j], convert_render_params_list_to_dict(estimate_render_params[j])])
                    out_params_matrix.append([params_dict_batch[j], exposed_params_dict_batch[j], render_params])
                else:
                    out_params_matrix.append([estimate_params_dict_batch[j], estimate_exposed_params_dict_batch[j]])
                    out_params_matrix.append([params_dict_batch[j], exposed_params_dict_batch[j]])
                eval_params_list.append(estimate_out_dict_batch[j])
                eval_params_list.append(out_dict_batch[j])

            if save_validation_results:
                # detach to prevent unnecessary memory comsumption
                in_batch = in_batch.detach().cpu()

                for j in range(batch):
                    visualization_list.append(in_batch[j,:,:,:])
                    estimated_sample = GRAPH_NAME_forward(eval_params_list[2*(i*batch+j)], **input_dict_batch[j])
                    estimated_sample_render_img = render((estimated_sample[1] - 0.5) * 2.0,
                                                            estimated_sample[0],
                                                            estimated_sample[3],
                                                            estimated_sample[2],
                                                            estimate_render_params[j,0]*GRAPH_NAME_render_params['light_color_max'].to(device) if GRAPH_NAME_render_params_trainable \
                                                                else GRAPH_NAME_render_params['light_color'].to(device)*GRAPH_NAME_render_params['light_color_max'].to(device),
                                                            GRAPH_NAME_render_params['f0'].to(device),
                                                            estimate_render_params[j,1]*GRAPH_NAME_render_params['size_max'].to(device) if GRAPH_NAME_render_params_trainable \
                                                                else GRAPH_NAME_render_params['size'].to(device)*GRAPH_NAME_render_params['size_max'].to(device),
                                                            GRAPH_NAME_render_params['camera'].to(device),
                                                            lambertian_BRDF
                                                            )
                    # del estimated_sample
                    estimated_sample_render_img = estimated_sample_render_img.detach().cpu()
                    visualization_list.append(estimated_sample_render_img)
                    count += 2
                    if count == num_imgs_per_grid or count == 2 * num_validation_batch * batch:
                        grid = make_grid(visualization_list, nrow=num_cols, padding=padding, normalize=False, range=None, scale_each=False, pad_value=0)
                        np_grid = np.transpose(grid.detach().cpu().numpy(), [1,2,0])
                        if show_validation_results:
                            plt.imshow(np_grid)
                            plt.draw()
                            plt.pause(show_time) # pause how many seconds
                            plt.close()
                        if save_png:
                            imageio.imwrite(os.path.join(model_dir, validation_output_name % (cur_epoch) + '_img_%d.png' % (grid_count)), (np_grid * 255.0).astype(np.uint8))
                        else:
                            imageio.imwrite(os.path.join(model_dir, validation_output_name % (cur_epoch) + '_img_%d.exr' % (grid_count)), np_grid)
                        count = 0
                        grid_count += 1
                        visualization_list = []

            # detach error to prevent hugh gpu memory comsumption due requirement of gradient
            mean_err += criterion(estimate_out_batch, out_batch).detach().cpu()

            # save output matrix
            with open(os.path.join(model_dir, validation_params_name % (cur_epoch) + '.pkl'), 'wb') as f:
                th.save(out_params_matrix,f)

            mean_err = mean_err / num_validation_batch
            print('Validation parameters error after epoch %d: %.4f' % (cur_epoch, mean_err))
# ---------------------------------------------------------------------------- #


if __name__ == "__main__":
    if eval_mode:
        if load_grid_parameter:
            eval_for_grid_parameter()
        if use_real_image:
            eval_real_world_image()
    else:
        train()