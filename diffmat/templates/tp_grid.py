import os
import sys
import random
import logging
import imageio
import numpy as np
import torch as th
import configargparse
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

from GRAPH_NAME_util import *
from diffmat.sbs_core.util import *
from diffmat.util.render import render
# ---------------------------------------------------------------------------- #


# load configuration
current_path = os.path.dirname(os.path.abspath(__file__))
p = configargparse.ArgumentParser(default_config_files = [os.path.join(current_path, 'configs', 'grid.conf')])
p.add('-c', '--config_filepath', required=False, is_config_file=True, help='Path to config file')

# parameters
p.add_argument('--eval_checkpoint', action='store_true', help='Evaluated saved checkpoint')
p.add_argument('--grid_ids', type=int, nargs='+', default=[0], help='Index of saved checkpoint')

p.add_argument('--num_grids', type=int, default=1, help='Number of grids')
p.add_argument('--num_imgs_per_grid', type=int, default=40, help='Number of images in a grid')
p.add_argument('--num_cols', type=int, default=8, help='Number of cols in the grid')
p.add_argument('--padding', type=int, default=15, help='Paddings between images')
p.add_argument('--show_grid', action='store_true', help='Grid visualization flag')
p.add_argument('--show_time', type=int, default=20, help='Show time of the grid')

p.add_argument('--log_debug', action='store_true', help='Set log output to debug mode')
p.add_argument('--use_cpu', action='store_true', help='Use CPU as computing device')
p.add_argument('--save_png', action='store_true', help='Save results as png images instead of exr images')

p.add_argument('--save_params', action='store_true', help='Save generated parameters')
p.add_argument('--output_dir', type=str, default=os.path.join('.','grid'), help='Output directory')
p.add_argument('--grid_image_prefix', type=str, default='sampling_grid_%d', help='Grid image output prefix')
p.add_argument('--grid_params_output_name', type=str, default='sampling_grid_%d.pkl', help='Grid parameter output file name')

p.add_argument('--params_ptb_max', type=float, default=0.05, help='Max perturbation (as percentage/100) to node parameters')
p.add_argument('--params_ptb_min', type=float, default=-0.05, help='Min perturbation (as percentage/100) to node parameters')
p.add_argument('--params_ptb_mu', type=float, default=0.0, help='Mean shift of perturbation (sampling with normal distribution)')
p.add_argument('--params_ptb_sigma', type=float, default=0.03, help='Standard deviation of perturbation (sampling with normal distribution)')
p.add_argument('--params_normal_sampling_func', action='store_true', help='Sampling function for node parameters')

p.add_argument('--lambertian_BRDF', action='store_true', help='Use lambertian BRDF')
p.add_argument('--render_ptb_max', type=float, default=0.05, help='Max perturbation (as percentage/100) to render parameters')
p.add_argument('--render_ptb_min', type=float, default=-0.05, help='Min perturbation (as percentage/100) to render parameters')
p.add_argument('--render_ptb_mu', type=float, default=0.0, help='Mean shift of perturbation (sampling with normal distribution)')
p.add_argument('--render_ptb_sigma', type=float, default=0.03, help='Standard deviation of perturbation (sampling with normal distribution)')
p.add_argument('--render_normal_sampling_func', action='store_true', help='Sampling function for render parameters')

locals().update(p.parse_args().__dict__)
# ---------------------------------------------------------------------------- #


# create output dirs
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

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
# ---------------------------------------------------------------------------- #


def gen_sample():
    """Sample a new set of graph parameters

    Returns:
        Dict: node parameters dictionary
        Dict: exposed parameters dictionary
        Dict: swapped node parameters dictionary
    """    
    GRAPH_NAME_params_rand, GRAPH_NAME_exposed_params_rand, GRAPH_NAME_swapped_params_rand = gen_rand_params(GRAPH_NAME_params,
                    params_spec,
                    GRAPH_NAME_trainable_list, 
                    GRAPH_NAME_keys, 
                    params_sampling_func, 
                    res=RES_H,
                    exposed_params = None if GRAPH_NAME_optimization_level == 0 else GRAPH_NAME_exposed_params, 
                    params_tp = None if GRAPH_NAME_optimization_level == 0 else GRAPH_NAME_params_tp,
                    gen_exposed_only = True if GRAPH_NAME_optimization_level == 2 else False, 
                    output_level=2
                    )
    if GRAPH_NAME_render_params_trainable:
        GRAPH_NAME_render_params_rand = gen_rand_light_params(GRAPH_NAME_render_params, render_params_spec, render_sampling_func)
        return GRAPH_NAME_params_rand, GRAPH_NAME_exposed_params_rand, GRAPH_NAME_swapped_params_rand, GRAPH_NAME_render_params_rand
    else:
        return GRAPH_NAME_params_rand, GRAPH_NAME_exposed_params_rand, GRAPH_NAME_swapped_params_rand


def render_sample():
    """Render material from the graph parameters

    Returns:
        Tensor: a 3D tensor of a rendered image
    """    
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
                                    GRAPH_NAME_render_params_rand['light_color'].to(device) * GRAPH_NAME_render_params['light_color_max'].to(device) if GRAPH_NAME_render_params_trainable else GRAPH_NAME_render_params['light_color'].to(device) * GRAPH_NAME_render_params['light_color_max'].to(device),
                                    GRAPH_NAME_render_params['f0'].to(device),
                                    GRAPH_NAME_render_params_rand['size'].to(device) * GRAPH_NAME_render_params['size_max'].to(device) if GRAPH_NAME_render_params_trainable else GRAPH_NAME_render_params['size'].to(device) * GRAPH_NAME_render_params['size_max'].to(device),
                                    GRAPH_NAME_render_params['camera'].to(device),
                                    lambertian_BRDF
                                )
    sample_render_img = sample_render_img.detach().cpu()
    return sample_render_img


def export_grid(visualization_list, i):
    """Make and export the grid

    Args:
        visualization_list (list): a list of rendered images
        i (int): grid ID
    """    
    grid = make_grid(visualization_list, nrow=num_cols, padding=padding, normalize=False, range=None, scale_each=False, pad_value=0)
    np_grid = np.transpose(grid.detach().cpu().numpy(), [1,2,0])
    if show_grid:
        plt.imshow(np_grid)
        plt.draw()
        plt.pause(show_time) # pause how many seconds
        plt.close()
    if save_png:
        imageio.imwrite(os.path.join(output_dir, grid_image_prefix % (i) + '.png'), (np_grid * 255.0).astype(np.uint8))
    else: 
        imageio.imwrite(os.path.join(output_dir, grid_image_prefix % (i) + '.exr'), np_grid)
# ---------------------------------------------------------------------------- #


if __name__ == "__main__":
    if eval_checkpoint:
        # evaluate an existing grid checkpoint
        for i in grid_ids:
            visualization_list = []
            pkl_path = os.path.join(output_dir, grid_params_output_name % (i))
            with open(pkl_path, 'rb') as f: 
                out_params_matrix = th.load(f)
            for j in range(num_imgs_per_grid):
                GRAPH_NAME_swapped_params_rand = swap_tp(out_params_matrix[j][0], GRAPH_NAME_params_tp, out_params_matrix[j][1])
                if GRAPH_NAME_render_params_trainable:
                    GRAPH_NAME_render_params_rand = out_params_matrix[j][2]
                sample_render_img = render_sample()
                visualization_list.append(sample_render_img)
            export_grid(visualization_list, i)
    else:
        # generate a new grid
        for i in range(num_grids):
            visualization_list = []
            out_params_matrix = []
            for j in range(num_imgs_per_grid):
                if GRAPH_NAME_render_params_trainable:
                    GRAPH_NAME_params_rand, GRAPH_NAME_exposed_params_rand, GRAPH_NAME_swapped_params_rand, GRAPH_NAME_render_params_rand = gen_sample()
                else:
                    GRAPH_NAME_params_rand, GRAPH_NAME_exposed_params_rand, GRAPH_NAME_swapped_params_rand = gen_sample()
                sample_render_img = render_sample()
                visualization_list.append(sample_render_img)
                if GRAPH_NAME_render_params_trainable:
                    # only save the two perturbed parameters
                    out_params_matrix.append([GRAPH_NAME_params_rand, GRAPH_NAME_exposed_params_rand, GRAPH_NAME_render_params_rand])
                else:
                    out_params_matrix.append([GRAPH_NAME_params_rand, GRAPH_NAME_exposed_params_rand])
            export_grid(visualization_list, i)

            # save parameters
            if save_params:
                with open(os.path.join(output_dir, grid_params_output_name % i), 'wb') as f:
                    th.save(out_params_matrix, f)