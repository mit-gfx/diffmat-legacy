import os
import sys
# import exr
import time
import copy
import random
import imageio
import logging
import torch as th
import numpy as np
from numbers import Number
from functools import partial
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# for substance node implementation
def input_check(num_tensor_input, class_method=False, profile=False):
    """Decorator that checks if the input is a 4D pytorch tensor

    Args:
        num_tensor_input (int): number of tensor inputs that require checking 
        class_method (bool, optional): The decorated method is a class method. Defaults to False.
        profile (bool, optional): Profile the code, will report the runtime if longer than 0.3s. Defaults to False.
    """    
    def decorator(func):
        def wrapper(*args, **kwargs):
            if 'trainable' in kwargs:
                del kwargs['trainable']
            for idx in range(num_tensor_input):
                if class_method:
                    assert (args[idx+1] is None) or (th.is_tensor(args[idx+1]) and len(args[idx+1].shape) == 4), "%d-th input is not a 4D Pytorch tensor" % (idx+1)
                else:
                    assert (args[idx] is None) or (th.is_tensor(args[idx]) and len(args[idx].shape) == 4), "%d-th input is not a 4D Pytorch tensor" % (idx+1)
            # Evaluate any partial argument before feed into the filter function
            # the partial in the class wrapper should have been evaluated before 
            # feeding to the filter function, otherwise the following command will
            #  break the gradient
            for key, val in kwargs.items():
                if isinstance(val, partial):
                    kwargs[key] = val()
            if profile:
                t_start = time.time()
                retvals = func(*args, **kwargs)
                t_end = time.time()
                if t_end - t_start >= 0.3:
                    print("[PROFILE] {:16s}: {:.3f}s".format(func.__name__, t_end - t_start))
            else:
                retvals = func(*args, **kwargs)
            return retvals
        return wrapper
    return decorator


@input_check(1)
def roll_row(img_in, n):
    """Roll the row of a 4D image tensor
    """    
    return img_in.roll(-n, 2)


@input_check(1)
def roll_col(img_in, n):
    """Roll the column of a 4D image tensor
    """    
    return img_in.roll(-n, 3)


@input_check(1)
def normalize(img_in):
    """Normalize along the color channel
    """    
    return img_in / th.sqrt((img_in ** 2).sum(1, keepdim=True))


def color_input_check(tensor_input, err_var_name):
    """Check if the input tensor is a 3 or 4 channel tensor
    """    
    assert tensor_input.shape[1] in [3,4], '%s should be a color image' % err_var_name
    

def grayscale_input_check(tensor_input, err_var_name):
    """Check if the input tensor is a 1 channel tensor
    """    
    assert tensor_input.shape[1] == 1, '%s should be a grayscale image' % err_var_name


def read_image(filename: str):
    """Read image
    """
    img = imageio.imread(filename)
    if img.dtype == np.float32:
        return img
    if img.dtype == np.uint8:
        return img.astype(np.float32) / 255.0
    elif img.dtype == np.uint16:
        return img.astype(np.float32) / 65535.0


def get_input_dict(seed_idx_list, source_random_base_dir, input_names, device, use_alpha=False):
    """Get input dictionary

    Args:
        seed_idx_list (list): a list of random seeds
        source_random_base_dir (str): path to the random noise directory
        input_names (list): a list of noise names
        device (torch.device): device on which a torch tensor will be allocated
        use_alpha (bool, optional): Add alpha channel to the RGB image. Defaults to False.

    Returns:
        List : a list of random noise images
    """    
    logging.debug("Reading input files ...")
    count = 0
    input_dict = {}
    logging.debug(seed_idx_list)
    for key, val in input_names.items():
        input_dict[key] = th.from_numpy(read_image(os.path.join(source_random_base_dir, "%d/%s" % (seed_idx_list[count], val)))) 
        if len(input_dict[key].shape) == 3:
            input_dict[key] = input_dict[key].permute(2,0,1)
            if use_alpha:
                if input_dict[key].shape[0] == 3:
                    input_dict[key] = th.cat([input_dict[key], th.ones(1,input_dict[key].shape[1],input_dict[key].shape[2], device='cpu')], dim=0)
            else:
                if input_dict[key].shape[0] == 4:
                    input_dict[key] = input_dict[key][:3,:,:]
            input_dict[key] = input_dict[key].unsqueeze(0).to(device)
        else:
            input_dict[key] = input_dict[key].unsqueeze(0).unsqueeze(0).to(device)
        count = count + 1
    return input_dict


def save_output_dict(imgs, base_dir, output_names, idx=None, png_flag=False):
    """Save output images

    Args:
        imgs (list): a list of maps and rendered image
        base_dir (str): base save path
        output_names (list): a list of output names
        idx (int): example ID
        png_flag (bool, optional): Save image in png format, otherwise exr format. Defaults to False.
        postfix_idx (int, optional): postfix index, typically the iteration. Defaults to None.
    """    
    # save generated output maps and rendering
    logging.debug("Saving output ...")
    for output_name, img in zip(output_names, imgs):
        img = img.detach().cpu().numpy()

        if not os.path.exists(os.path.join(base_dir, output_name)):
            os.makedirs(os.path.join(base_dir, output_name))
        
        if idx is None: 
            filename = os.path.join(base_dir, output_name, output_name + ".%s" % ("png" if png_flag else "exr"))
        else:
            filename = os.path.join(base_dir, output_name, output_name + "_%d.%s" % (idx, "png" if png_flag else "exr"))

        logging.debug(filename)
        if len(img.shape) == 3:
            out_img = np.transpose(img, [1, 2, 0])
            if png_flag:
                imageio.imwrite(filename, (out_img * 255.0).astype(np.uint8))
            else:
                imageio.imwrite(filename, out_img)
        else:
            out_img = img
            if png_flag:
                imageio.imwrite(filename, (out_img * 65535.0).astype(np.uint16))
            else:
                imageio.imwrite(filename, out_img)

def param_sampling_uniform(x_in, ptb_min_, ptb_max_, lb, ub):
    """sample parameter with uniform distribution

    Args:
        x_in (float): parameter to be sampled
        ptb_min_ (float): min perturbation (as percentage/100)
        ptb_max_ (float): max perturbation (as percentage/100)
        lb (float): lower bound
        ub (float): upper bound

    Returns:
        Float : perturbed parameter
    """    
    epsilon = 0.0
    if isinstance(x_in, th.Tensor):
        return np.clip(x_in * (1.0 + random.uniform(ptb_min_, ptb_max_)) + epsilon, lb, ub).float()
    else:
        return np.clip(x_in * (1.0 + random.uniform(ptb_min_, ptb_max_)) + epsilon, lb, ub).astype(np.float32)


def param_sampling_normal(x_in, mu, sigma, lb, ub):
    """smaple parameter with normal distribution

    Args:
        x_in (float): parameter to be sampled
        mu (float): mean of perturbation
        sigma (float): standard deviation of perturbation
        lb (float): lower bound
        ub (float): upper bound

    Returns:
        Float: perturbed parameter
    """    
    epsilon = 0.0
    if isinstance(x_in, th.Tensor):
        return np.clip(np.random.normal(mu, sigma) + x_in + epsilon, lb, ub).float()
    else:
        return np.clip(np.random.normal(mu, sigma) + x_in + epsilon, lb, ub).astype(np.float32)


def gen_rand_params(params, params_spec, trainable_list, keys, 
    param_sampling_func, node_list=None, res=512, exposed_params=None, params_tp=None,
    gen_exposed_only=False, output_level=0):
    """Given a dictionary of template parameters, this function generates a perturbed
    dictionary of parameters by adding a random offset to each of variable

    Args:
        params (dict): node parameters dictionary
        params_spec (dict): sampling function specs
        trainable_list (list): a list of trainable variables lists
        keys (list): a list of parameter keys
        param_sampling_func (func): parameter sampling function (uniform/normal)
        node_list (list, optional): a list of nodes to be perturbed. Defaults to None.
        res (int, optional): image resolution. Defaults to 512.
        exposed_params (dict, optional): exposed parameters. Defaults to None.
        params_tp (dict, optional): to partial function dictionary. Defaults to None.
        gen_exposed_only (bool, optional): perturb only exposed parameters. Defaults to False.
        output_level (int, optional): optimization level. Defaults to 0.
    Raises:
        ValueError: [description]

    Returns:
        optimization level:
            0: output parameters (swap params_tp into parameters if params_tp and exposed_params are not None)
            1: output parameters and exposed parameters, no swap
            2: output parameters, exposed parameters, and swapped parameters
    """    
    if not node_list:
        node_list = random.sample(range(params_spec['total_var']), params_spec['num_free_nodes'])
    params_rand = copy.deepcopy(params)
    
    if param_sampling_func is param_sampling_uniform:
        param_a = params_spec['ptb_min']
        param_b = params_spec['ptb_max']
    else:
        param_a = params_spec['mu']
        param_b = params_spec['sigma']

    def sample(var):
        if isinstance(var, list) or \
        (isinstance(var, np.ndarray) and len(var.shape) == 1) or \
        (isinstance(var, th.Tensor) and len(var.shape) == 1):
            for idx, item in enumerate(var):
                var[idx] = param_sampling_func(item, param_a, param_b, 0.0, 1.0)
        elif isinstance(var, Number) or \
            (isinstance(var, np.ndarray) and len(var.shape) == 0) or \
            (isinstance(var, th.Tensor) and len(var.shape) == 0):
            var = param_sampling_func(var, param_a, param_b, 0.0, 1.0)
        var = to_float32(var)
        return var

    if exposed_params is not None:
        exposed_params_rand = copy.deepcopy(exposed_params)
        # generate random exposed parameters
        for key, val in exposed_params_rand.items():
            if key != "trainable":
                exposed_params_rand[key] = sample(val)

    if gen_exposed_only == False:
        for node_idx in node_list:
            node_name = keys[node_idx]
            trainable_var_name_list = trainable_list[node_idx]
            node_prefix = node_name.split('_')[0]
            # deal with special nodes
            # gradient map & curve
            if (node_prefix == "uniform" or node_prefix == "normal_color" or node_name == "normal_color") and res != 512:
                params_rand[keys[node_idx]]["res_h"] = res
                params_rand[keys[node_idx]]["res_w"] = res
            if (node_prefix == "gradient" or node_prefix == "curve") and trainable_var_name_list[0] == "anchors":
                # sample each parameter in the 2-d matrix
                var = params_rand[keys[node_idx]]["anchors"]
                if var is None:
                    multiplier = 6 if node_prefix == "curve" else \
                                4 + params_rand[keys[node_idx]]["use_alpha"] if params_rand[keys[node_idx]]["mode"] == "color" else 2
                    var = [[0.0] * multiplier, [1.0] * multiplier]
                if isinstance(var, list) or \
                isinstance(var, np.ndarray) or \
                isinstance(var, th.Tensor):
                    # replace the list
                    for row_idx, row in enumerate(var):
                        for col_idx, item in enumerate(row):
                            var[row_idx][col_idx] = param_sampling_func(item, param_a, param_b, 0.0, 1.0)
                    params_rand[keys[node_idx]]["anchors"] = to_float32(var)
                else:
                    raise ValueError("Unrecognized input type")
            else:
                for var_key in trainable_var_name_list:
                    var = params_rand[keys[node_idx]][var_key]
                    params_rand[keys[node_idx]][var_key] = sample(var)

    if output_level == 0:
        if exposed_params is not None and params_tp is not None:
            # swap params_tp into params
            for key, val in params_tp.items():
                for key_, val_ in val.items():
                    params_rand[key][key_] = partial(val_.func, exposed_params_rand)
            return params_rand
        else:
            return params_rand
    elif output_level == 1:
        if exposed_params is not None:
            return params_rand, exposed_params_rand
        else:
            return params_rand, None
    elif output_level == 2:
        if exposed_params is not None and params_tp is not None:
            # swap params_tp into params
            swapped_params_rand = swap_tp(copy.deepcopy(params_rand), params_tp, exposed_params_rand)
            return params_rand, exposed_params_rand, swapped_params_rand
        else:
            return params_rand, None, params_rand


def swap_tp(params, params_tp, exposed_params):
    """Swap exposed parameters into the node parameters

    Args:
        params (dict): node parameters dictionary
        params_tp (dict): to partial function dictionary
        exposed_params (dict): exposed parameters dictionary

    Returns:
        Dict: swapped node parameters
    """    
    if exposed_params is not None:
        for key, val in params_tp.items():
            for key_, val_ in val.items():
                params[key][key_] = partial(val_.func, exposed_params)
    return params


def gen_rand_light_params(render_params, params_spec, param_sampling_func):
    """Generate random lighting parameters

    Args:
        render_params (dict): render parameters dictionary
        params_spec (dict): sampling function specs
        param_sampling_func (func): parameter sampling function

    Returns:
        Dict: perturbed lighting parameters
    """    
    ptb_max = params_spec['ptb_max']
    ptb_min = params_spec['ptb_min']

    render_params_rand = copy.deepcopy(render_params)
    render_params_rand['size'] = param_sampling_func(render_params_rand['size'], ptb_min, ptb_max, 0.0, 1.0)
    render_params_rand['light_color'] = param_sampling_func(render_params_rand['light_color'], ptb_min, ptb_max, 0.0, 1.0)
    
    return render_params_rand


def convert_params_dict_to_list(params, keys, trainable_list):
    """Convert node parameters dictionary to a list

    Args:
        params (dict): node parameters dictionary
        keys (list): a list of parameter keys
        trainable_list (list): a list of trainable variables lists

    Returns:
        List: node parameters list
    """
    params_list = []
    # return a 1d numpy array
    for idx, node_key in enumerate(keys):
        node_params = params[node_key]
        for var_key in trainable_list[idx]:
            if isinstance(node_params[var_key], th.Tensor):
                one_param = node_params[var_key].cpu().detach().numpy().flatten().astype(np.float32)
            elif isinstance(node_params[var_key], np.ndarray):
                one_param = node_params[var_key].flatten().astype(np.float32)
            # [NEW] partial branch, add nothing for the moment
            elif isinstance(node_params[var_key], partial):
                one_param = []
            else:
                one_param = np.array(node_params[var_key], dtype=np.float32)
                one_param = one_param.flatten().astype(np.float32)
            params_list = np.hstack((params_list, one_param))

    return params_list                       


def preprocess_network_prediction(params_list, keep_tensor=True):
    """Pre-process network prediction

    Args:
        params_list (tensor): a batch of flattened parameters (2D tensor)
        keep_tensor (bool, optional): keep numerical values as pytorch tensor. Defaults to True.

    Returns:
        Tensor or List: Reshaped network prediction
    """    
    if isinstance(params_list, th.Tensor):
        if not keep_tensor:
            params_list = params_list.detach().cpu().numpy()
        if len(params_list.shape) == 1:
            params_list = params_list.unsqueeze(0)
    if isinstance(params_list, list) or isinstance(params_list, np.ndarray):
        params_list = np.array(params_list)
        if len(params_list.shape) == 1:
            params_list = params_list[np.newaxis, :]
    return params_list


def convert_params_list_to_dict(params_list, template_params, trainable_list, keys, params_tp=None, keep_tensor=True):
    """Convert a batch of flattened node parameters into a batch of node parameters dictionaries

    Args:
        params_list (tensor): a batch of flattend parameters
        template_params (dict): a template node parameters dictionary
        trainable_list (list): a list of trainable variables lists
        keys (list): a list of parameter keys
        params_tp (dict, optional): to partial function dictionary.. Defaults to None.
        keep_tensor (bool, optional): keep numerical values as pytorch tensor. Defaults to True.

    Returns:
        List : a batch of node parameters dictionary
    """
    params_list = preprocess_network_prediction(params_list, keep_tensor)

    # if params_tp is not None, filter out to partial entries in the trainable_list
    if params_tp is not None:
        trainable_list_temp = copy.deepcopy(trainable_list)
        for key, val in params_tp.items():
            for key_, _ in val.items():
                temp = copy.deepcopy(trainable_list_temp[keys.index(key)])
                del temp[trainable_list_temp[keys.index(key)].index(key_)]
                trainable_list_temp[keys.index(key)] = temp
        trainable_list = trainable_list_temp
        
    # create empty list of dict output
    params_dict = []
    for idx in range(params_list.shape[0]):
        params_dict_item = copy.deepcopy(template_params)
        count = 0
        for node_idx, trainable_var_name_list in enumerate(trainable_list):
            node_name = keys[node_idx]
            node_prefix = node_name.split('_')[0]
            for var_key in trainable_var_name_list:
                if isinstance(params_dict_item[node_name][var_key], Number):
                    params_dict_item[node_name][var_key] = params_list[idx, count]
                    if not keep_tensor:
                        params_dict_item[node_name][var_key] = params_dict_item[node_name][var_key].item()
                    count += 1
                elif isinstance(params_dict_item[node_name][var_key], list):
                    temp = np.array(params_dict_item[node_name][var_key])
                    num_elem = temp.size
                    shape = temp.shape
                    params_dict_item[node_name][var_key] = params_list[idx, count:count+num_elem].reshape(shape)
                    if not keep_tensor:
                        params_dict_item[node_name][var_key] = params_dict_item[node_name][var_key].astype(np.float32).tolist()
                    count += num_elem
                elif isinstance(params_dict_item[node_name][var_key], np.ndarray):
                    num_elem = template_params[node_name][var_key].size
                    shape = template_params[node_name][var_key].shape
                    params_dict_item[node_name][var_key] = params_list[idx, count:count+num_elem].reshape(shape)
                    if not keep_tensor:
                        params_dict_item[node_name][var_key] = params_dict_item[node_name][var_key].astype(np.float32).tolist()
                    count += num_elem
                elif isinstance(params_dict_item[node_name][var_key], th.Tensor):
                    num_elem = template_params[node_name][var_key].numel()
                    shape = template_params[node_name][var_key].shape
                    params_dict_item[node_name][var_key] = params_list[idx, count:count+num_elem].reshape(shape)
                    if not keep_tensor:
                        params_dict_item[node_name][var_key] = params_dict_item[node_name][var_key].tolist()
                    count += num_elem
                
        params_dict.append(params_dict_item)
        
    return params_dict
                

def count_trainable_params(params, params_tp, exposed_params, keys, trainable_list):
    """Count number of trainable parameters for different optimization modes

    Args:
        params (orderedDict): node parameters dictonary
        params_tp (orderedDict): to partial function dictionary
        exposed_params (orderedDict): exposed parameters dictionary
        keys (list):  a list of parameter keys
        trainable_list (list): a list of trainable parameters lists

    Returns:
        A list of number of optimizable parameters 
            - all node parameters 
            - exposed parameters + independent node parameters
            - only exposed parameters
    """
    # compute length of full params
    params_list_len = len(convert_params_dict_to_list(params, keys, trainable_list))

    # compute length of trimmed parameters
    trim_count = 0
    for key, val in params_tp.items():
        for key_, _ in val.items():
            trim_count += len(np.array([params[key][key_]]).flatten())

    # compute length of exposed params
    exposed_params_len = 0
    for key, val in exposed_params.items():
        if key != "trainable":
            exposed_params_len += len(np.array([val]).flatten())
    
    return [params_list_len, params_list_len - trim_count + exposed_params_len, exposed_params_len]


def convert_exposed_params_list_to_dict(exposed_params_list, template_exposed_params, keep_tensor=True):
    """Convert a batch of flattened exposed parameters into a batch of exposed parameters dictionaries

    Args:
        exposed_params_list (tensor): a batch of flattened exposed parameters
        template_exposed_params (list): a template exposed parameters dictionary
        keep_tensor (bool, optional): keep numerical values as pytorch tensor. Defaults to True.

    Returns:
        List : a batch of exposed parameters dictionary
    """     
    exposed_params_list = preprocess_network_prediction(exposed_params_list, keep_tensor)
    exposed_params_dict = []

    for idx in range(exposed_params_list.shape[0]):
        exposed_params_dict_item = copy.deepcopy(template_exposed_params)
        count = 0
        for key, val in template_exposed_params.items():
            if key != 'trainable':
                if isinstance(val, Number):
                    val_ = exposed_params_list[idx, count]
                    exposed_params_dict_item[key] = val_ if keep_tensor else val_.item()
                    count += 1
                elif isinstance(val, list) or isinstance(val, np.ndarray) or isinstance(val, th.Tensor):
                    if_tensor = isinstance(val, th.Tensor)
                    if not if_tensor:
                        val = np.array(val)    
                    num_elem = val.numel() if if_tensor else val.size
                    shape = val.shape
                    val_ = exposed_params_list[idx, count:count+num_elem].reshape(shape)
                    if not if_tensor:
                        exposed_params_dict_item[key] = val_ if keep_tensor else val_.astype(np.float32).tolist()
                    else:
                        exposed_params_dict_item[key] = val_ if keep_tensor else val_.tolist()
                    count += num_elem

        exposed_params_dict.append(exposed_params_dict_item)

    return exposed_params_dict   


def convert_exposed_params_dict_to_list(exposed_params):
    """Convert an exposed parameters dictionary to a list

    Args:
        exposed_params (orderedDict): exposed parameters dictionary

    Returns:
        Numpy array: exposed parameters list (1D)
    """
    exposed_params_list = []
    # return a 1d numpy array
    for key, val in exposed_params.items():
        if key != "trainable":
            if isinstance(val, th.Tensor):
                val_ = val.cpu().detach().numpy().flatten().astype(np.float32)
            elif isinstance(val, np.ndarray):
                val_ = val.flatten().astype(np.float32)
            else:
                val_ = np.array(val, dtype=np.float32).flatten()
            exposed_params_list.append(val_)

    exposed_params_list = np.hstack(exposed_params_list)
    return exposed_params_list    


def nan_test(maps, rendering_only=True):
    """Test if any of the input has nan

    Args:
        maps (list): a list of maps and rendering
        rendering_only (bool): 
    """    
    if rendering_only:
        if th.sum(maps != maps) > 0:
            raise ValueError("rendering has nan")
    else:
        # order: albedo, normal, metallic, roughness, render
        if th.sum(maps[:,:3,:,:] != maps[:,:3,:,:]) > 0:
            raise ValueError("albedo has nan")
        if th.sum(maps[:,3:6,:,:] != maps[:,3:6,:,:]) > 0:
            raise ValueError("normal has nan")
        if th.sum(maps[:,6,:,:] != maps[:,6,:,:]) > 0:
            raise ValueError("metallic has nan")
        if th.sum(maps[:,7,:,:] != maps[:,7,:,:]) > 0:
            raise ValueError("roughness has nan")
        if th.sum(maps[:,8,:,:] != maps[:,8,:,:]) > 0:
            raise ValueError("rendered image has nan")                


@input_check(1)
def bin_statistics(img, bin_shape='circular', num_bins=5, num_row_bins=2, num_col_bins=2):
    """Compute statistics within each bin shape

    Args:
        img (tensor): an image tensor (4D)
        bin_shape (str, optional): shape of the bins 'circular' or 'grid'. Defaults to 'circular'.
        num_bins (int, optional): number of bins for 'circular'. Defaults to 5.
        num_row_bins (int, optional): num of bins along the row for 'grid'. Defaults to 2.
        num_col_bins (int, optional): num of bins along the column for 'grid'. Defaults to 2.

    Returns:
        Tensor : bin statistics 
    """    
    res_h = img.shape[2]
    res_w = img.shape[3]

    X,Y = th.meshgrid(th.linspace(-res_h//2,res_h//2-1, res_h), th.linspace(-res_w//2,res_w//2-1, res_w))
    dist = th.sqrt(X*X + Y*Y).expand(img.shape[0],res_h,res_w)

    if bin_shape == 'circular':
        dist_step = (th.max(dist)+1.0) / num_bins
        mean = th.zeros(img.shape[1], num_bins)
        var = th.zeros(img.shape[1], num_bins)
        for j in range(img.shape[1]):
            for i in range(num_bins):
                temp = img[:,j,:,:]
                mask = (dist >= dist_step*i) * (dist < dist_step*(i+1))
                mean[j,i] = th.mean(temp[mask])
                var[j,i] = th.var(temp[mask])
    elif bin_shape == 'grid':
        row_step = res_h // num_row_bins
        col_step = res_w // num_col_bins
        mean = th.zeros(img.shape[1], num_row_bins, num_col_bins)
        var = th.zeros(img.shape[1], num_row_bins, num_col_bins)
        for k in range(img.shape[1]):
            temp = img[:,k,:,:]
            for i in range(num_row_bins):
                for j in range(num_col_bins):
                    row_indices = th.arange(row_step*i, row_step*(i+1))
                    col_indices = th.arange(col_step*j, col_step*(j+1))
                    mean[k,i,j] = th.mean(temp[:,row_indices,col_indices])
                    var[k,i,j] = th.var(temp[:,row_indices,col_indices])

    return mean, var
    

def correct_normal(normal):
    """Make the mean of r,g to be 0.5

    Args:
        normal (tensor): normal tensor (3D, CHW)

    Returns:
        Tensor: corrected normal
    """
    r = normal[0]
    g = normal[1]
    b = th.ones_like(r)
    r = th.clamp(0.5 + r - r.sum() / r.numel(), 0.0, 1.0)
    g = th.clamp(0.5 + g - g.sum() / g.numel(), 0.0, 1.0)
    if normal.shape[0] == 4:
        return th.stack([r, g, b, normal[3]])
    else:
        return th.stack([r, g, b])


def display_img(img_in):
    """Display a 4D tensor image

    Args:
        img_in (tensor): input image
    """
    if not isinstance(img_in, th.Tensor):
        return
    
    if len(img_in.shape) == 4:
        img_in_np = img_in.clone()
        if img_in.shape[1] >= 3:
            img_in_np = img_in_np.squeeze().permute(1,2,0).cpu().numpy()
        else:
            img_in_np = img_in_np.squeeze().cpu().numpy()
    
        plt.imshow(img_in_np)
        plt.show()

# ------------------------ Other math helper functions ----------------------- #

# Convert [-1, 1] to [0, 1]
to_zero_one = lambda a: a / 2.0 + 0.5

# Convert to float32
to_float32 = lambda a: a.astype(np.float32) if isinstance(a, np.ndarray) else \
                       a.to(th.float32) if isinstance(a, th.Tensor) else a

# Convert to float tensor
to_tensor = lambda a: th.as_tensor(a, dtype=th.float)

# Swizzle operation
def swizzle(x, d):
    x, d = to_tensor(x), to_tensor(d)
    x = x[None] if not x.ndim else x
    return th.where(d < len(x), x[th.min(d, to_tensor(len(x) - 1))], th.zeros_like(d, dtype=x.dtype)).squeeze()

# Vector operation
def vector(x, y):
    x, y = to_tensor(x), to_tensor(y)
    x = x[None] if not x.ndim else x
    y = y[None] if not y.ndim else y
    return th.cat((x, y))

# Get an item and convert it into torch tensor
def itemgetter_torch(key):
    return lambda params: to_tensor(params[key])
