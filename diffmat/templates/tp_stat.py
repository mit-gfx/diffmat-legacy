import os
import time
import GPUtil
import configargparse

import numpy as np
import torch as th

from GRAPH_NAME_util import *
from diffmat.sbs_core.util import *
# ---------------------------------------------------------------------------- #


# load configuration
current_path = os.path.dirname(os.path.abspath(__file__))
p = configargparse.ArgumentParser(default_config_files = [os.path.join(current_path, 'configs', 'stat.conf')])
p.add('-c', '--config_filepath', required=False, is_config_file=True, help='Path to config file')

p.add_argument('--num_skip_runs_start', type=int, default=10, help='Number of initial iterations to be skipped')
p.add_argument('--num_eval_prof_runs', type=int, default=10, help='Number of runtime evaluation runs')
p.add_argument('--num_mem_prof_runs', type=int, default=10, help='Number of memory profiling runs')
p.add_argument('--print_result', action='store_true', help='Print result in profiler')
p.add_argument('--format_output', action='store_true', help='Print formatted output in the end')
p.add_argument('--mem_prof_type', type=str, default='mem_th', help="Memory profiling type. 'mem_th'|'mem_gpu'")

locals().update(p.parse_args().__dict__)
# ---------------------------------------------------------------------------- #


# Other parameters
num_random_seed = NOISE_COUNT

# Set default GPU
os.environ['CUDA_VISIBLE_DEVICES'] = 'DEVICE_COUNT'

# Profiler class (context manager)
class Profiler:
    '''
    A simple profiler that supports multiple runs.
    '''
    def __init__(self, tag, prof_type='time', device=None, gpu_no=DEVICE_COUNT, print_result=True):
        self.tag = tag
        self.print_result = print_result
        self.prof_type = prof_type
        self.device = device
        self.gpu_no = gpu_no
        self.history = []

        # Collect memory usage prior to the code section
        if prof_type == 'mem_th':
            self.start_val = th.cuda.memory_reserved(device) // 1048576
        elif prof_type == 'mem_gpu':
            self.start_val = GPUtil.getGPUs()[gpu_no].memoryUsed

    def __enter__(self):
        # Start timer
        if self.prof_type == 'time':
            self.start_val = time.time()

    def __exit__(self, type, value, traceback):
        # Gather data at the end of profiling
        if self.prof_type == 'time':
            end_val = time.time()
        elif self.prof_type == 'mem_th':
            end_val = th.cuda.memory_reserved(self.device) // 1048576
        elif self.prof_type == 'mem_gpu':
            end_val = GPUtil.getGPUs()[self.gpu_no].memoryUsed

        # Record profiling results
        diff = end_val - self.start_val
        self.history.append(diff)
        if self.print_result:
            if self.prof_type == 'time':
                print(f'[PROFILER] {self.tag}: {diff:.3f}s')
            else:
                print(f'[PROFILER] {self.tag}: {int(diff)}MiB')

    def average(self, skip=0):
        # Calculate the average of metrics
        if skip >= len(self.history):
            print('Warning: all previous profiling runs are skipped.')
            return 0.0
        elif not len(self.history):
            print('Warning: no profiling runs on record.')
            return 0.0

        return sum(self.history[skip:]) / (len(self.history) - skip)

if __name__ == "__main__":
    use_cuda = True
    # enable GPU
    if th.cuda.is_available() and use_cuda:
        device_name = 'cuda'
        device_full_name = 'cuda:0'
        device = th.device(device_full_name)
        th.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        device_name = 'cpu'
        device_full_name = 'cpu'
        device = th.device(device_full_name)
        th.set_default_tensor_type('torch.FloatTensor')

    # Count the number of nodes
    num_nodes = len(GRAPH_NAME_params)
    if 'special_transform' in GRAPH_NAME_params:
        num_nodes -= 1

    # Count the number of trainable parameters
    params = convert_params_dict_to_list(GRAPH_NAME_params, GRAPH_NAME_keys, GRAPH_NAME_trainable_list)
    num_params = np.array(params).size

    # Read input images
    prof = Profiler('read input', prof_type=mem_prof_type, gpu_no=DEVICE_COUNT, print_result=print_result)
    with prof:
        input_dict = get_input_dict(random.choices(range(num_random_seed), k=len(input_names)), source_random_base_dir, input_names,
                                    device, GRAPH_NAME_use_alpha)
    input_mem = int(prof.average())

    # Measure forward evaluation mem usage
    prof = Profiler('forward evalution', prof_type=mem_prof_type, gpu_no=DEVICE_COUNT, print_result=print_result)
    for _ in range(num_mem_prof_runs):
        with prof:
            sample = GRAPH_NAME_forward(GRAPH_NAME_params, **input_dict)
    avg_mem = int(prof.average(2))

    # Measure forward evaluation time
    prof = Profiler('forward evaluation', print_result=print_result)
    for _ in range(num_eval_prof_runs + num_skip_runs_start):
        with prof:
            sample = GRAPH_NAME_forward(GRAPH_NAME_params, **input_dict)
    avg_time = prof.average(num_skip_runs_start)

    # Header
    if format_output:
        print('--- Stats for graph "GRAPH_NAME" ---')
        print('# nodes: ', num_nodes)
        print('# params:', num_params)
        print('Avg. eval time:', f'{avg_time:.3f}s')
        print('Avg. mem ({}) usage:'.format('PyTorch' if mem_prof_type == 'mem_th' else 'GPU'),
              f'{input_mem + avg_mem}MiB (input {input_mem}MiB; eval {avg_mem}MiB)')
    else:
        print(f'GRAPH_NAME,{num_nodes},{num_params},{avg_time},{input_mem + avg_mem},{input_mem},{avg_mem}')
