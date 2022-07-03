# all functions expect Pytorch 4D tensor 
import os 
import sys
import time 
import random 
import logging 
import configargparse

import numpy as np  
import torch as th 
import torch.nn as nn 
from skimage.transform import resize

from GRAPH_NAME_util import * 
from diffmat.sbs_core.util import * 
from diffmat.util.render import render
from diffmat.util.descriptor import TextureDescriptor 
# ---------------------------------------------------------------------------- #


# load configuration
current_path = os.path.dirname(os.path.abspath(__file__))
p = configargparse.ArgumentParser(default_config_files = [os.path.join(current_path, 'configs', 'opt.conf')])
p.add('-c', '--config_filepath', required=False, is_config_file=True, help='Path to config file.')

p.add_argument('--save_ids', type=int, nargs='+', default=[], help='Indices for saving the results')

p.add_argument('--use_real_image', action='store_true', help='Evaluate real image input.')
p.add_argument('--real_image_paths', type=str, nargs='+', default=[], help='Real image paths')

p.add_argument('--load_network_prediction_real', action='store_true', help='Load network prediction for real image')
p.add_argument('--load_network_prediction_syn', action='store_true', help='Load network prediction for synthetic image')
p.add_argument('--num_graph_parameters', type=int, default=GRAPH_NAME_num_trainable_params[GRAPH_NAME_optimization_level], help='Number of trainable parameters in the graph')
p.add_argument('--num_load_epoch', type=int, default=0, help='The epoch of the loaded results')
p.add_argument('--fix_vgg', action='store_true', help='Load the result from the network trained with fixed vgg')
p.add_argument('--net_prediction_pkl_path', type=str, default=os.path.join('.','models','num_params_%d_opt_mode_%d_%s','%s','params_epoch_%d%s.pkl'), help='Path to the saved network prediction file')
p.add_argument('--net_prediction_real_image_postfix', type=str, nargs='+', default=[], help='Postfix to the saved network prediction file path')
p.add_argument('--net_prediction_grid_id', type=int, default=0, help='Grid index of the saved network prediction file')
p.add_argument('--net_prediction_example_ids', type=int, nargs='+', default=[0], help='Example indices of the saved network prediction file')


p.add_argument('--load_grid_parameter', action='store_true', help='Load example from the grid parameter')
p.add_argument('--grid_id', type=int, default=0, help='ID of the grid to be loaded')
p.add_argument('--grid_pkl_path', type=str, default=os.path.join('.','grid','sampling_grid_%d.pkl'), help='Path to the saved grid file')
p.add_argument('--grid_example_ids',  type=int, nargs='+', default=[0], help='Example indices of the saved grid file')

# question
p.add_argument('--num_examples', type=int, default=1, help='Number of examples to test on the fly (when none of the load option is set)')


p.add_argument('--save_png', action='store_true', help='Save results as png images instead of exr images')
p.add_argument('--log_debug', action='store_true', help='Set log output to debug mode')
p.add_argument('--use_cpu', action='store_true', help='Use CPU as computing device')


p.add_argument('--params_ptb_max', type=float, default=0.05, help='Max perturbation (as percentage/100) to node parameters.')
p.add_argument('--params_ptb_min', type=float, default=-0.05, help='Min perturbation (as percentage/100) to node parameters.')
p.add_argument('--params_ptb_mu', type=float, default=0.0, help='Mean shift of perturbation (sampling with normal distribution)')
p.add_argument('--params_ptb_sigma', type=float, default=0.03, help='Standard deviation of perturbation (sampling with normal distribution)')
p.add_argument('--params_normal_sampling_func', action='store_true', help='Sampling function for node parameters')


p.add_argument('--lambertian_BRDF', action='store_true', help='Use lambertian BRDF')
p.add_argument('--render_ptb_max', type=float, default=0.05, help='Max perturbation (as percentage/100) to render parameters.')
p.add_argument('--render_ptb_min', type=float, default=-0.05, help='Min perturbation (as percentage/100) to render parameters.')
p.add_argument('--render_ptb_mu', type=float, default=0.0, help='Mean shift of perturbation (sampling with normal distribution)')
p.add_argument('--render_ptb_sigma', type=float, default=0.03, help='Standard deviation of perturbation (sampling with normal distribution)')
p.add_argument('--render_normal_sampling_func', action='store_true', help='Sampling function for render parameters')


p.add_argument('--pos_base_dir', type=str, default=os.path.join('.','optim','%d','pos_imgs'), help='Directory to save positive results')
p.add_argument('--neg_init_base_dir', type=str, default=os.path.join('.','optim','%d','neg_init_imgs'), help='Directory to save initialization results')
p.add_argument('--neg_optim_base_dir', type=str, default=os.path.join('.','optim','%d','neg_optim_imgs'), help='Directory to save optimization results')
p.add_argument('--ptb_name', type=str, default='default', help='Perturbation name')


p.add_argument('--num_iter', type=int, default=1000, help='Number of optimization iterations')
p.add_argument('--save_itv', type=int, default=10, help='Saving interval')
p.add_argument('--use_l1', action='store_true', help='Use l1 loss')
p.add_argument('--num_pyramid', type=int, default=2, help='Number of pyramid levels used in the td loss')


p.add_argument('--load_exist_dataset', action='store_true', help='Load existing dataset')
p.add_argument('--use_same_dict', action='store_true', help='Use same input seed')
p.add_argument('--start_iter', type=int, default=0, help='Start iteration')
p.add_argument('--optimizer_type', type=int, default=0, help='Optimization method [0: th.optim.Adam|1: th.optim.Adadelta|2: th.optim.Adagrad|3: th.optim.LBFGS]')

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

# update grid pkl path
grid_pkl_path = grid_pkl_path % grid_id

# network parameters
num_params = num_graph_parameters+2 if GRAPH_NAME_render_params_trainable else num_graph_parameters

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

# loss function 
loss_func = nn.functional.l1_loss if use_l1 else nn.functional.mse_loss 

# optimizer
optimizer_method_list = [th.optim.Adam, th.optim.Adadelta, th.optim.Adagrad, th.optim.LBFGS]

# create texture descriptor
net = TextureDescriptor(device) 
# fix parameters for evaluation 
for param in net.parameters(): 
    param.requires_grad = False 
# ---------------------------------------------------------------------------- #


def compute_td_pyramid(img, num_pyramid):
    """compute texture descriptor pyramid

    Args:
        img (tensor): 4D tensor of image (NCHW)
        num_pyramid (int): pyramid level]

    Returns:
        Tensor: 2-d tensor of texture descriptor
    """    
    td = net.eval_CHW_tensor(pos_render) 
    for scale in range(num_pyramid):
        td_ = net.eval_CHW_tensor(nn.functional.interpolate(img, scale_factor = 1.0/(2.0**(scale+1)), mode='bilinear', align_corners=True))
        td = th.cat([td, td_], dim=1) 
    return td

def eval_graph(graph):
    """Evaluate graph

    Args:
        graph (DiffMatGraphModule): procedural material graph

    Returns:
        list: a list of material maps and rendered image
    """    
    samples = graph.forward()
    sample_rendered_img = render((samples[1] - 0.5) * 2.0,  
                                   samples[0],  
                                   samples[3],  
                                   samples[2],  
                                   graph.render_params['light_color'].to(device) * graph.render_params['light_color_max'].to(device),
                                   graph.render_params['f0'].to(device),
                                   graph.render_params['size'].to(device) * graph.render_params['size_max'].to(device),
                                   graph.render_params['camera'].to(device),
                                   lambertian_BRDF
                                ) 
    samples.append(sample_rendered_img)
    return samples

def init_graph(save_id=0, load_example_id=0, input_seed=None):
    """Initialized material graph

    Args:
        load_example_id (int, optional): example id to load from the checkpoint. Defaults to 0.
        input_seed (list, optional): a list of input noise seeds. Defaults to None.

    Returns:
        DiffMatGraphModule: procedural material graph
    """
    neg_init_dir = neg_init_base_dir % save_id + '_' + ptb_name 
    if not os.path.exists(neg_init_dir):
        os.makedirs(neg_init_dir)

    if load_network_prediction_syn or load_network_prediction_real or load_exist_dataset:
        if load_network_prediction_syn:
            pkl_path = net_prediction_pkl_path % (num_params, GRAPH_NAME_optimization_level, 'fix_vgg' if fix_vgg else 'trainable_vgg', 'real_world_results' if load_network_prediction_real else 'grid_results', num_load_epoch, '_grid_%d' % net_prediction_grid_id)
            load_example_id = load_example_id * 2 + 1
        elif load_network_prediction_real:
            pkl_path = net_prediction_pkl_path % (num_params, GRAPH_NAME_optimization_level, 'fix_vgg' if fix_vgg else 'trainable_vgg', 'real_world_results' if load_network_prediction_real else 'grid_results', num_load_epoch, '_%s' % net_prediction_real_image_postfix[load_example_id])
        else:
            pkl_path = os.path.join(neg_init_dir, 'param_dict.pkl')

        with open(pkl_path, 'rb') as f: 
            if GRAPH_NAME_render_params_trainable:
                neg_init_GRAPH_NAME_params, neg_init_GRAPH_NAME_exposed_params, neg_init_GRAPH_NAME_render_params = th.load(f)[load_example_id] 
            else:
                neg_init_GRAPH_NAME_params, neg_init_GRAPH_NAME_exposed_params = th.load(f)[load_example_id]
                neg_init_GRAPH_NAME_render_params = GRAPH_NAME_render_params
    else:
        neg_init_GRAPH_NAME_params = GRAPH_NAME_params
        neg_init_GRAPH_NAME_exposed_params = GRAPH_NAME_exposed_params
        neg_init_GRAPH_NAME_render_params = GRAPH_NAME_render_params

    # save 
    with open(os.path.join(neg_init_dir, 'param_dict.pkl'), 'wb') as f: 
        if GRAPH_NAME_render_params_trainable:
            th.save([[neg_init_GRAPH_NAME_params, neg_init_GRAPH_NAME_exposed_params, neg_init_GRAPH_NAME_render_params]],f) 
        else:
            th.save([[neg_init_GRAPH_NAME_params, neg_init_GRAPH_NAME_exposed_params]],f) 
                
    # generate input dictionary for negative examples
    if input_seed == None:
        neg_input_seed = random.choices(range(num_random_seed), k=len(input_names)) 
    else:
        neg_input_seed = input_seed
    
    neg_input_dict = get_input_dict(neg_input_seed,  
                                    source_random_base_dir,  
                                    input_names,  
                                    device,
                                    GRAPH_NAME_use_alpha)

    # create GRAPH_NAME graph that starts with negative example  
    graph = CLASS_NAME(neg_init_GRAPH_NAME_params, GRAPH_NAME_params_tp, neg_init_GRAPH_NAME_exposed_params, GRAPH_NAME_trainable_list, GRAPH_NAME_ops_list, GRAPH_NAME_init_list, GRAPH_NAME_call_list, neg_init_GRAPH_NAME_render_params, GRAPH_NAME_render_params_trainable, GRAPH_NAME_optimization_level, device)
    graph.setup_input(neg_input_dict) 
    neg_init_samples = graph.forward() 
    neg_init_render_img = render((neg_init_samples[1] - 0.5) * 2.0,  
                                      neg_init_samples[0],  
                                      neg_init_samples[3],  
                                      neg_init_samples[2],  
                                      graph.render_params['light_color'].to(device) * graph.render_params['light_color_max'].to(device),
                                      graph.render_params['f0'].to(device),
                                      graph.render_params['size'].to(device) * graph.render_params['size_max'].to(device),
                                      graph.render_params['camera'].to(device),
                                      lambertian_BRDF
                                      ) 
    neg_init_samples.append(neg_init_render_img)

    # save initial negative results  
    save_output_dict(neg_init_samples, neg_init_dir, output_names, png_flag=save_png) 
    
    return graph

def gen_positive_img(save_id=0, load_example_id=0):
    """Generate a target (positive) image

    Args:
        load_example_id (int, optional): example id to load from the checkpoint. Defaults to 0.

    Returns:
        Tensor: rendered target image
        List: a list of input noise seeds
    """
    pos_dir = pos_base_dir % save_id + '_' + ptb_name
    if not os.path.exists(pos_dir):
        os.makedirs(pos_dir)

    # generate input dictionary for positive examples
    pos_input_seed = random.choices(range(num_random_seed), k=len(input_names)) 
    pos_input_dict = get_input_dict(pos_input_seed,  
                                    source_random_base_dir,  
                                    input_names,  
                                    device,
                                    GRAPH_NAME_use_alpha)
                                    
    if use_real_image:
        pos_render = resize(read_image(real_image_paths[load_example_id]), (RES_H,RES_W), anti_aliasing=True)
        if save_png:
            imageio.imwrite(os.path.join(pos_dir,'render.png'), pos_render * 255.0)
        else:
            imageio.imwrite(os.path.join(pos_dir,'render.exr'), pos_render)
        
        pos_render = th.tensor(pos_render, dtype=th.float32).permute(2,0,1)
    else:
        if load_network_prediction_syn:
            pkl_path = net_prediction_pkl_path % (num_params, GRAPH_NAME_optimization_level, 'fix_vgg' if fix_vgg else 'trainable_vgg', 'real_world_results' if load_network_prediction_real else 'grid_results', num_load_epoch, '_grid_%d' % net_prediction_grid_id)
            load_example_id = load_example_id * 2
        elif load_grid_parameter:
            pkl_path = grid_pkl_path
        elif load_exist_dataset:
            pkl_path = os.path.join(pos_dir, 'param_dict.pkl')

        # load positive parameters dict
        if load_network_prediction_syn or load_grid_parameter or load_exist_dataset:
            with open(pkl_path, 'rb') as f: 
                if GRAPH_NAME_render_params_trainable:
                    pos_GRAPH_NAME_params, pos_GRAPH_NAME_exposed_params, pos_GRAPH_NAME_render_params = th.load(f)[load_example_id] 
                else:
                    pos_GRAPH_NAME_params, pos_GRAPH_NAME_exposed_params = th.load(f)[load_example_id]
                    pos_GRAPH_NAME_render_params = GRAPH_NAME_render_params
            pos_GRAPH_NAME_params_swap = pos_GRAPH_NAME_params if GRAPH_NAME_optimization_level == 0 else swap_tp(pos_GRAPH_NAME_params, GRAPH_NAME_params_tp, pos_GRAPH_NAME_exposed_params)
        else:
            # randomly genrate a positive example
            pos_GRAPH_NAME_params, pos_GRAPH_NAME_exposed_params, pos_GRAPH_NAME_params_swap = gen_rand_params(GRAPH_NAME_params, 
                                        params_spec, 
                                        GRAPH_NAME_trainable_list,  
                                        GRAPH_NAME_keys,  
                                        params_sampling_func,
                                        res = RES_H,
                                        exposed_params = None if GRAPH_NAME_optimization_level == 0 else GRAPH_NAME_exposed_params, 
                                        params_tp = None if GRAPH_NAME_optimization_level == 0 else GRAPH_NAME_params_tp,
                                        gen_exposed_only = True if GRAPH_NAME_optimization_level == 2 else False, 
                                        output_level=2)
            
            if GRAPH_NAME_render_params_trainable:
                pos_GRAPH_NAME_render_params = gen_rand_light_params(GRAPH_NAME_render_params, render_params_spec, render_sampling_func)
            else:
                pos_GRAPH_NAME_render_params = GRAPH_NAME_render_params

        # save 
        with open(os.path.join(pos_dir, 'param_dict.pkl'), 'wb') as f: 
            if GRAPH_NAME_render_params_trainable:
                th.save([[pos_GRAPH_NAME_params, pos_GRAPH_NAME_exposed_params, pos_GRAPH_NAME_render_params]],f) 
            else:
                th.save([[pos_GRAPH_NAME_params, pos_GRAPH_NAME_exposed_params]],f) 

        # recreate positive example
        pos_samples = GRAPH_NAME_forward(pos_GRAPH_NAME_params_swap, **pos_input_dict)
        pos_render = render((pos_samples[1] - 0.5) * 2.0,
                                pos_samples[0],
                                pos_samples[3],
                                pos_samples[2],
                                pos_GRAPH_NAME_render_params['light_color'].to(device) * pos_GRAPH_NAME_render_params['light_color_max'].to(device),
                                pos_GRAPH_NAME_render_params['f0'].to(device),
                                pos_GRAPH_NAME_render_params['size'].to(device) * pos_GRAPH_NAME_render_params['size_max'].to(device),
                                pos_GRAPH_NAME_render_params['camera'].to(device),
                                lambertian_BRDF
                                )
        pos_samples.append(pos_render)

        # save initial negative results  
        save_output_dict(pos_samples, pos_dir, output_names, png_flag=save_png) 

    return pos_render.unsqueeze(0), pos_input_seed

def optimize(pos_render, graph, save_id=0):
    """Optimize material graph

    Args:
        pos_render (tensor): target (positive) image
        graph (DiffMatGraphModule): procedural material graph
    """
    neg_optim_dir = neg_optim_base_dir % save_id + '_' + ptb_name 
    if not os.path.exists(neg_optim_dir):
        os.makedirs(neg_optim_dir)
    err_optim_path = os.path.join(neg_optim_dir, 'error.txt') 

    optimizer = optimizer_method_list[optimizer_type](graph.trainable_params, lr=5e-4) 

    pos_td = compute_td_pyramid(pos_render, num_pyramid)
    err_list = []

    # log start time
    opt_start = time.time()

    # start optimization
    for i in range(start_iter, num_iter): 
        iter_start = time.time() 
        def closure(): 
            optimizer.zero_grad() 
            neg_optim_samples = eval_graph(graph)

            # save intermediate result 
            if (i % save_itv) == 0 and i != 0: 
                save_output_dict(neg_optim_samples, neg_optim_dir, output_names, i, save_png) 

            neg_optim_render = neg_optim_samples[4].unsqueeze(0)
            neg_optim_td = compute_td_pyramid(neg_optim_render, num_pyramid)

            # compute error
            err = (pos_td - neg_optim_td).abs().mean() 

            # log error 
            err_list.append(err.cpu().detach().numpy().item()) 

            print('iter %d; error: %.4f;' % (i, err)) 

            err.backward() 
            return err 

        optimizer.step(closure) 
        graph.save_params(neg_optim_dir, i) 
        iter_end = time.time() 
        print('per iter time: %f ms' % ((iter_end-iter_start)*1000.0)) 

    # print time cost for the entire optimization
    opt_end = time.time() 
    logging.info('runtime: %f' % (opt_end - opt_start)) 

    # save final image result 
    neg_optim_samples = eval_graph(graph)
    save_output_dict(neg_optim_samples, neg_optim_dir, output_names, png_flag=save_png) 

    # save final parameters 
    graph.save_params(neg_optim_dir) 

    # save the record of error 
    np.savetxt(err_optim_path, np.array(err_list), fmt='%f') 

if __name__ == '__main__':
    if use_real_image:
        if save_ids == []:
            save_ids=list(range(len(real_image_paths)))
        else:
            assert len(save_ids) == len(real_image_paths), "real_image_paths and net_prediction_example_ids should have same length"
        if load_network_prediction_real:
            assert len(real_image_paths) == len(net_prediction_real_image_postfix), "real_image_paths and net_prediction_example_ids should have same length"
        for i in range(len(save_ids)):
            pos_render, pos_input_seed = gen_positive_img(save_ids[i],i)
            graph = init_graph(save_ids[i], input_seed=pos_input_seed if use_same_dict else None)
            optimize(pos_render, graph, save_ids[i])    
    elif load_grid_parameter or load_network_prediction_syn:
        load_ids = grid_example_ids if load_grid_parameter else net_prediction_example_ids
        if save_ids == []:
            save_ids=list(range(len(load_ids)))
        else:
            assert len(save_ids) == len(load_ids), "real_image_paths and net_prediction_example_ids should have same length"
        for i in range(len(save_ids)):
            pos_render, pos_input_seed = gen_positive_img(save_ids[i], load_ids[i])
            graph = init_graph(save_ids[i], load_ids[i], input_seed=pos_input_seed if use_same_dict else None)
            optimize(pos_render, graph, save_ids[i])    
    else:
        if save_ids == []:
            save_ids=list(range(num_examples))
        else:
            assert len(save_ids) == num_examples, "real_image_paths and net_prediction_example_ids should have same length"
        for i in range(num_examples):
            pos_render, pos_input_seed = gen_positive_img(save_ids[i])
            graph = init_graph(save_ids[i], input_seed=pos_input_seed if use_same_dict else None)
            optimize(pos_render, graph, save_ids[i])