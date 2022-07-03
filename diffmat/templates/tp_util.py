'''
Automatically generated utility file for graph 'GRAPH_NAME'
'''
import os
import torch as th
from collections import OrderedDict
from functools import partial
import diffmat.sbs_core.functional as F
from diffmat.sbs_core.nodes import get_nodes, DiffMatGraphModule
from diffmat.sbs_core.util import convert_params_dict_to_list, correct_normal, count_trainable_params
from diffmat.sbs_core.util import swizzle, vector, to_zero_one, itemgetter_torch

# Exposed parameters
GRAPH_NAME_optimization_level = 0

# Exposed parameters
GRAPH_EXPOSED_PARAMS

# Dynamic functions and parameter converters
GRAPH_DYNAMIC_FUNCTIONS

# The global dictionary of to partial parameters
GRAPH_NAME_params_tp = OrderedDict([
    GRAPH_TP_PARAMS
])

# The global dictionary of node parameters
GRAPH_NAME_params = OrderedDict([
    GRAPH_PARAMS
])

# Class name of each node in program order
node_list = [
    NODE_LIST
]

# Get the list of trainable parameters and node classes
GRAPH_NAME_keys = list(GRAPH_NAME_params.keys())
GRAPH_NAME_trainable_list, GRAPH_NAME_ops_list, GRAPH_NAME_init_list, GRAPH_NAME_call_list = get_nodes(node_list)
GRAPH_NAME_num_trainable_params = count_trainable_params(GRAPH_NAME_params, GRAPH_NAME_params_tp, GRAPH_NAME_exposed_params, GRAPH_NAME_keys, GRAPH_NAME_trainable_list)
GRAPH_NAME_num_nodes = len(GRAPH_NAME_keys)
GRAPH_NAME_use_alpha = USE_ALPHA

# Labels of input files and output images
input_names = {
    INPUT_NAMES
}
output_names = ['basecolor', 'normal', 'metallic', 'roughness', 'render']

# The directory of noise images with various random seeds
# source_random_base_dir = './random_seeds'
source_random_base_dir = os.path.join('.','random_seeds')

# Rendering parameters
GRAPH_NAME_render_params = {
    'size': th.tensor([0.10], dtype=th.float32),
    'size_max': th.tensor(300.0, dtype=th.float32),
    'camera': th.tensor([0.0, 0.0, 25.0], dtype=th.float32).view(3, 1, 1),
    'light_color' : th.tensor([0.33], dtype=th.float32),
    'light_color_max' : th.tensor([10000.0, 10000.0, 10000.0], dtype=th.float32).view(3, 1, 1),
    'f0' : th.tensor(0.04, dtype=th.float32)
}
GRAPH_NAME_render_params_trainable = False

def GRAPH_NAME_forward(FORWARD_ARGS):
    '''
    Forward pass of creating GRAPH_NAME texture with specified input images.
    '''
    FORWARD_PROGRAM

class CLASS_NAME(DiffMatGraphModule):
    '''
    Computation graph for GRAPH_NAME texture.
    '''
    def __init__(self, params, params_tp, exposed_params, trainable_list, ops_list, init_list, call_list, render_params, render_params_trainable=False,
                 optimization_levels=GRAPH_NAME_optimization_level, device=th.device('cpu')):
        super(CLASS_NAME, self).__init__(params, params_tp, exposed_params, trainable_list, ops_list, init_list, call_list, render_params, 
                                         render_params_trainable, optimization_levels, device)

    def forward(self):
        '''
        Forward evalution of computation graph with auto-differentiation.
        '''
        FORWARD_CLASS

