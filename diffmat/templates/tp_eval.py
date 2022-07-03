import os
import time
import copy
import random
import logging
import configargparse

from GRAPH_NAME_util import *
from diffmat.sbs_core.util import *
from diffmat.sbs_core.nodes import *
from diffmat.util.render import render
import diffmat.sbs_core.functional as F
# ---------------------------------------------------------------------------- #

# load configuration
current_path = os.path.dirname(os.path.abspath(__file__))
p = configargparse.ArgumentParser(default_config_files = [os.path.join(current_path, 'configs', 'eval.conf')])
p.add('-c', '--config_filepath', required=False, is_config_file=True, help='Path to config file')

p.add_argument('--eval_checkpoint', action='store_true', help='Evaluated saved checkpoint')
p.add_argument('--example_id', type=int, default=0, help='Example index of the stored checkpoint')
p.add_argument('--iteration', type=int, default=0, help='Iteration of the stored checkpoint')
p.add_argument('--checkpoint_name', type=str, default="param_dict_%d_%d.pkl", help='Checkpoint name')
# p.add_argument('--pkl_path_prefix', type=str, default="./results/GRAPH_NAME/optim/neg_optim_imgs_default", help='Folder that contains the saved checkpoint')
p.add_argument('--checkpoint_dir', type=str, default=os.path.join('.','optim','neg_optim_imgs_default'), help='Checkpoint folder')
p.add_argument('--output_dir', type=str, default=os.path.join('.','maps'), help='Output directory')

p.add_argument('--no_ptb', action='store_true', help='No perturbation')
p.add_argument('--params_ptb_max', type=float, default=0.05, help='Max perturbation (as percentage/100) to node parameters')
p.add_argument('--params_ptb_min', type=float, default=-0.05, help='Min perturbation (as percentage/100) to node parameters')
p.add_argument('--params_ptb_mu', type=float, default=0.0, help='Mean shift of perturbation (sampling with normal distribution)')
p.add_argument('--params_ptb_sigma', type=float, default=0.03, help='Standard deviation of perturbation (sampling with normal distribution)')
p.add_argument('--params_normal_sampling_func', action='store_true', help='Sampling function for node parameters')

p.add_argument('--render_ptb_max', type=float, default=0.0, help='Max perturbation (as percentage/100) to render parameters')
p.add_argument('--render_ptb_min', type=float, default=-0.0, help='Min perturbation (as percentage/100) to render parameters')
p.add_argument('--render_ptb_mu', type=float, default=0.0, help='Mean shift of perturbation (sampling with normal distribution)')
p.add_argument('--render_ptb_sigma', type=float, default=0.0, help='Standard deviation of perturbation (sampling with normal distribution)')
p.add_argument('--render_normal_sampling_func', action='store_true', help='Sampling function for render parameters')

p.add_argument('--log_debug', action='store_true', help='Set log output to debug mode')
p.add_argument('--use_cpu', action='store_true', help='Use CPU as computing device')
p.add_argument('--save_png', action='store_true', help='Save results as png images instead of exr images')

locals().update(p.parse_args().__dict__)
# ---------------------------------------------------------------------------- #
checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name % (example_id, iteration))

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
    'ptb_max' : params_ptb_max if not no_ptb else 0.0,
    'ptb_min' : params_ptb_min if not no_ptb else 0.0,
    'mu': params_ptb_mu if not no_ptb else 0.0,
    'sigma': params_ptb_sigma if not no_ptb else 0.0,
}

render_params_spec = {
    'ptb_max' : render_ptb_max if not no_ptb else 0.0,
    'ptb_min' : render_ptb_min if not no_ptb else 0.0,
    'mu': render_ptb_mu if not no_ptb else 0.0,
    'sigma': render_ptb_sigma if not no_ptb else 0.0,
}

params_sampling_func = param_sampling_normal if params_normal_sampling_func else param_sampling_uniform
render_sampling_func = param_sampling_normal if render_normal_sampling_func else param_sampling_uniform
num_random_seed = NOISE_COUNT
# ---------------------------------------------------------------------------- #


if __name__ == "__main__":
    # generate random inputs
    input_dict = get_input_dict(random.choices(range(num_random_seed),
                                k=len(input_names)),
                                source_random_base_dir,
                                input_names,
                                device,
                                GRAPH_NAME_use_alpha)
    logging.debug(input_dict)

    # timing log
    logging.info('start processing example')
    start = time.time()

    if not eval_checkpoint:
        # generate a random example
        GRAPH_NAME_params_rand, GRAPH_NAME_exposed_params_rand, GRAPH_NAME_params_swap = gen_rand_params(GRAPH_NAME_params,
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
        if GRAPH_NAME_render_params_trainable:
            GRAPH_NAME_render_params_rand = gen_rand_light_params(GRAPH_NAME_render_params, render_params_spec, render_sampling_func)
    else:
        # load existing dataset
        with open(checkpoint_path, 'rb') as f: 
            # stopped here
            if GRAPH_NAME_render_params_trainable:
                GRAPH_NAME_params_rand, GRAPH_NAME_exposed_params_rand, GRAPH_NAME_render_params_rand = th.load(f) 
            else:
                GRAPH_NAME_params_rand, GRAPH_NAME_exposed_params_rand = th.load(f)

            GRAPH_NAME_params_swap = swap_tp(copy.deepcopy(GRAPH_NAME_params_rand), GRAPH_NAME_params_tp, GRAPH_NAME_exposed_params_rand)

    # evaluate the graph
    sample = GRAPH_NAME_forward(GRAPH_NAME_params_swap, **input_dict)
    render_img = render((sample[1] - 0.5) * 2.0,
                                sample[0],
                                sample[3],
                                sample[2],
                                GRAPH_NAME_render_params_rand['light_color'].to(device)*GRAPH_NAME_render_params['light_color_max'].to(device) if eval_checkpoint and GRAPH_NAME_render_params_trainable else GRAPH_NAME_render_params['light_color'].to(device)*GRAPH_NAME_render_params['light_color_max'].to(device),
                                GRAPH_NAME_render_params['f0'].to(device),
                                GRAPH_NAME_render_params_rand['size'].to(device)*GRAPH_NAME_render_params['size_max'].to(device) if eval_checkpoint and GRAPH_NAME_render_params_trainable else GRAPH_NAME_render_params['size'].to(device)*GRAPH_NAME_render_params['size_max'].to(device),
                                GRAPH_NAME_render_params['camera'].to(device),
                                False
                                )
    sample.append(render_img)
    logging.debug(sample)

    # save maps and rendering
    save_output_dict(sample, output_dir, output_names, 0, save_png)

    # timing log
    logging.info('finish processing sample')
    end = time.time()
    logging.info('runtime: %f' % (end - start))
