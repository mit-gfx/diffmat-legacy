'''
Node and parameter definitions for SBS file parser.
'''
import os
import sys
import abc
import types
import itertools
import numpy as np

from diffmat.sbs_converter.sbs_functions import SBSFunctionGraphParser
from diffmat.sbs_core.util import to_zero_one



# Name of the function that converts dynamic parameter functions to partial objects
NAME_TO_PARTIAL_FUNC = 'TP'

class SBSParameter:
    '''The wrapper class of an exposed parameter.
    '''
    def __init__(self, name, uid, param_type=1, param_val=None, parent=None):
        # Basic information
        self.name = name
        self.uid = uid
        self.type = param_type
        self.val = param_val
        self.reachable = False
        self.parent = parent

        # Whether the parameter is used in a dynamic function
        self.used = False

    def is_trainable(self):
        # Image and bool inputs are treated as untrainable
        return self.type >= 256

class SBSOutput:
    '''The wrapper class of an output image.
    '''
    def __init__(self, name, uid, usage, group, parent=None):
        # Basic information
        self.name = 'output_' + name
        self.uid = uid
        self.usage = usage
        self.group = group
        self.parent = parent

    def is_optimizable(self, lambertian=False):
        # Only basecolor, normal, roughness, and metallic are considered for now
        return self.group == 'Material' and \
              (self.usage in ['baseColor', 'normal', 'roughness', 'metallic'] if not lambertian else \
               self.usage in ['baseColor', 'normal'])

class SBSOutputBridge:
    '''The wrapper class of an output bridge inside a SBS node.
    '''
    def __init__(self, uid, name='', bridge_type=1):
        self.name = name
        self.var_name = name
        self.uid = uid
        self.type = bridge_type
        self.targets = []
        self.reachable = False

    def get_variable_name(self):
        return self.var_name if self.reachable else '_'

class SBSNodeParameter:
    '''The wrapper class of a generic SBS node parameter. The definition also includes inputs and outputs.
    '''
    def __init__(self, name, trans, param_type, param_val=None, convert=None, global_name=''):
        # Name(s) of the relevant SBS parameters
        self.name = name

        # The global name of the variable (usually including the node name)
        self.global_name = global_name if global_name else trans

        # The local name of the variable
        self.trans = trans

        self.type = param_type
        self.val = param_val() if callable(param_val) else param_val
        self.convert = convert

        # Dynamic parameter
        self.dynamic_params = []

    # Register a dynamic SBS parameter (in the form of SBSFunctionGraphParser)
    def register_dynamic_parameter(self, graph):
        self.dynamic_params.append(graph)

    # Check if a node parameter is dependent on dynamic parameters
    def is_dynamic(self):
        return any([param.trainable for param in self.dynamic_params]) and \
               not (hasattr(self.convert, 'trainable') and not self.convert.trainable)

    # Mark used exposed parameters
    def mark_exposed_params(self, param_dict):
        for param in self.dynamic_params:
            if param.trainable:
                param.mark_exposed_params(param_dict)

    # Print dynamic parameter function
    def get_func_str(self):
        # Skip if the parameter is non-dynamic
        if not self.is_dynamic():
            return ''

        # Beichen Li: conversion from multiple trainable SBS parameters is not supported for now
        if len(self.dynamic_params) > 1:
            raise NotImplementedError('Conversion from multiple trainable SBS parameters is not supported.')

        # Special case if no conversion is needed
        if not self.convert:
            return f'{self.global_name} = {self.dynamic_params[0].global_name}'

        # Obtain converter function
        # Beichen Li: debugging here
        if hasattr(self.convert, 'template'):
            convert_func_str = self.convert.template
        else:
            print(f"Error: The converter of node parameter '{self.global_name}' is not defined yet")
            convert_func_str = 'convert = None  # Undefined'

        # Write the definition of the dynamic function
        if convert_func_str:
            str_list = [f'def {self.global_name}(params):']
            write = lambda s: str_list.append(s)
            write(f'    x = {self.dynamic_params[0].global_name}(params)')
            for line in convert_func_str.split('\n'):
                write(f'    {line}')
            write('    return convert(x)')
            write('')
            return '\n'.join(str_list)
        # Special case of dummy conversion (See 'Make It Tile Patch')
        else:
            return f'{self.global_name} = {self.dynamic_params[0].global_name}'

class SBSNodeParameterConverter:
    '''A converter from SBS parameters to node parameters
    '''
    def __init__(self, func, func_template='', trainable=True):
        self.func = func

        # Beichen Li: Some templates need to be decided at runtime
        self.template = '' if callable(func_template) else func_template
        self.gen_template = func_template if callable(func_template) else None

        # Converters for max_intensity are non-trainable
        self.trainable = trainable

    def __call__(self, *args, **kwargs):
        if self.gen_template:
            self.template = self.gen_template(*args, **kwargs)
        return self.func(*args, **kwargs)

class SBSNode(abc.ABC):
    '''The base class of a generic SBS node.
    '''
    def __init__(self, name, uid, node_class, node_func, output_res=(9, 9), use_alpha=False):
        # Basic information
        self.name = name
        self.uid = uid
        self.type = node_class
        self.func = node_func
        self.res = output_res

        # Input connections
        self.connections = []

        # Output bridges (uid -> SBSNode)
        self.output_bridges = {}

        # Alpha channel switch
        self.use_alpha = use_alpha

        # Set default parameters
        self.params_list = []
        self.params = {}
        self.internal_params = {}
        self.reset_params()

    # If the node is reachable from an optimizable output
    def is_reachable(self):
        for _, bridge in self.output_bridges.items():
            if bridge.reachable:
                return True
        return False

    # Add an input connection
    def add_connection(self, conn_name, node_ref=None, node_output_ref=None):
        if node_output_ref is None and node_ref is not None:
            node_output_ref = node_ref.get_output_bridge_by_name().uid
        self.connections.append((conn_name, node_ref, node_output_ref))

    # Get the connection which has the target name
    def get_connection(self, conn_name):
        if isinstance(conn_name, list):
            conn_list = [self.get_connection(name) for name in conn_name]
            return None if all([x is None for x in conn_list]) else conn_list
        for conn in self.connections:
            if conn[0] == conn_name:
                return conn

    # Add an output bridge
    def add_output_bridge(self, uid, bridge):
        self.output_bridges[uid] = bridge

    # Update an output bridge with new target node
    def update_output_bridge(self, uid, node_ref):
        bridge = self.get_output_bridge_by_name() if uid is None else self.output_bridges[uid]
        bridge.targets.append(node_ref)

    # Obtain the output bridge associated with a name
    def get_output_bridge_by_name(self, name=''):
        for _, bridge in self.output_bridges.items():
            if not name or bridge.name == name:
                return bridge

    # Set an output bridge as reachable
    def set_output_bridge_reachable(self, uid):
        self.output_bridges[uid].reachable = True

    # Get all reachable output bridges
    def get_reachable_output_bridges(self):
        return [bridge for _, bridge in self.output_bridges.items() if bridge.reachable]

    # Update the name of output variables inside output bridges
    def update_output_variable_names(self):
        # Count the number of output bridges
        num_bridges = len(self.output_bridges)
        for _, bridge in self.output_bridges.items():
            var_name = self.name if num_bridges == 1 else f'{self.name}_{bridge.name}'
            bridge.var_name = '_'.join(var_name.split('-'))

    # Get the output variable name linked to an output bridge
    def get_output_variable_name(self, uid):
        return self.output_bridges[uid].var_name

    # Abstract method: get default parameter rules and values (must be manually specified)
    @abc.abstractmethod
    def get_default_params(self):
        return

    # Reset parameters
    def reset_params(self):
        for param_entry in self.get_default_params():
            # Add a new parameter
            param_name = param_entry[0]
            param_global_name = f'{self.name}_{param_entry[1]}'
            new_param = SBSNodeParameter(*param_entry, global_name=param_global_name)
            self.params_list.append(new_param)

            # Build mapping from SBS tag to parameter item
            if isinstance(param_name, str):
                if param_name not in self.params:
                    self.params[param_name] = []
                self.params[param_name].append(new_param)
            else:
                for param_name_item in param_name:
                    if param_name_item not in self.params:
                        self.params[param_name_item] = []
                    self.params[param_name_item].append(new_param)

        # Internal parameters
        self.internal_params.update({
            '$size': [1 << self.res[0], 1 << self.res[1]],
            '$sizelog2': self.res,
            '$normalformat': 0,
        })

    # Update node parameter
    def update_param(self, key, val):
        if key not in self.params:
            return

        # Evaluate the dynamic parameter first
        is_dynamic = isinstance(val, SBSFunctionGraphParser)
        new_val = val.eval(self.internal_params) if is_dynamic else val

        for param in self.params[key]:
            # Update the default values of the associated node parameters
            if param.convert is None:
                param.val = new_val
            elif isinstance(param.name, list):
                param.val = param.convert(new_val, param.val, key)
            else:
                param.val = param.convert(new_val)

            # Register the current SBS parameter if it is dynamic and trainable (requires grad)
            if is_dynamic and val.trainable:
                param.register_dynamic_parameter(val)

    # Print parameter description ("('name': {'param_name': param_value}, ...),")
    # Note: please avoid numpy arrays inside lists, tuples, or dicts
    # def get_params_str(self, round_float=6, calc_dynamic_params=True, disable_node_params=False):
    #     # Convert a parameter value to string format
    #     def to_str(param, val):
    #         if val is None:
    #             val_str = 'None'
    #         elif isinstance(val, float):
    #             val_str = str(round(val, round_float))
    #         elif isinstance(val, (int, bool, tuple, list, dict)):
    #             val_str = str(val)
    #         elif isinstance(val, str):
    #             val_str = f"'{val}'"
    #         elif isinstance(val, np.ndarray):
    #             val_str = str(val.tolist())
    #         # Beichen Li: check what this case means
    #         elif isinstance(val, types.MethodType):
    #             val_str = str(param.val())
    #         else:
    #             raise TypeError(f"Unknown parameter value type '{type(val)}'")

    #         return val_str

    #     # Append each parameter as an entry of the dictionary
    #     params_str = ''
    #     counter = 0
    #     for param in self.params_list:
    #         if param.trans and param.type == 'keyword' and self.get_connection(param.name) is None:
    #             val = param.val
    #             # Special case: trainable list
    #             if param.trans == 'trainable' and disable_node_params:
    #                 val = [False for _ in val]

    #             # Convert the parameter to string format
    #             if not param.is_dynamic() or calc_dynamic_params:
    #                 param_val_str = to_str(param, val)
    #             else:
    #                 param_val_str = f'{NAME_TO_PARTIAL_FUNC}({param.global_name})'
    #             params_str += (', ' if counter else '') + f"'{param.trans}': {param_val_str}"
    #             counter += 1

    #     return f"('{self.name}', {{{params_str}}})"

    def get_params_str(self, round_float=6, only_dynmaic_params=True):
        # Convert a parameter value to string format
        def to_str(param, val):
            if val is None:
                val_str = 'None'
            elif isinstance(val, float):
                val_str = str(round(val, round_float))
            elif isinstance(val, (int, bool, tuple, list, dict)):
                val_str = str(val)
            elif isinstance(val, str):
                val_str = f"'{val}'"
            elif isinstance(val, np.ndarray):
                val_str = str(val.tolist())
            # Beichen Li: check what this case means
            elif isinstance(val, types.MethodType):
                val_str = str(param.val())
            else:
                raise TypeError(f"Unknown parameter value type '{type(val)}'")

            return val_str

        # Append each parameter as an entry of the dictionary
        params_str = ''
        counter = 0
        for param in self.params_list:
            if param.trans and param.type == 'keyword' and self.get_connection(param.name) is None:
                val = param.val
                if only_dynmaic_params:
                    if param.is_dynamic():
                        param_val_str = f'{NAME_TO_PARTIAL_FUNC}({param.global_name})'
                        params_str += (', ' if counter else '') + f"'{param.trans}': {param_val_str}"
                        counter += 1
                else:
                    param_val_str = to_str(param, val)
                    params_str += (', ' if counter else '') + f"'{param.trans}': {param_val_str}"
                    counter += 1

        return None if params_str=='' and only_dynmaic_params else f"('{self.name}', {{{params_str}}})"

    # Print program description
    # ("output_variable, ... = op_func(input_variable, ..., global variable, **params[param_name])")
    def get_op_str(self, in_class=False):
        # Add output variables
        output_str = ''
        counter = 0
        for param in self.params_list:
            if param.type == 'output':
                bridge = self.get_output_bridge_by_name(param.name)
                output_str += (', ' if counter else '') + bridge.get_variable_name()
                counter += 1

        # Add arguments
        arg_str = ''
        counter = 0
        for param in self.params_list:
            if param.type != 'output':
                conn = self.get_connection(param.name)
                if conn is not None:
                    arg_str += ', ' if counter else ''
                    # Positional arguments
                    if param.type == 'input':
                        arg_str += conn[1].get_output_variable_name(conn[2])
                    # Keyword arguments
                    elif param.trans:
                        # Special case for multi-switch node (adjustable list of inputs)
                        if isinstance(conn, list):
                            assert self.type == 'MultiSwitch'
                            list_str = ', '.join([conn_item[1].get_output_variable_name(conn_item[2]) if conn_item is not None else None \
                                                for conn_item in conn[:self.params['input_number'][1].val]])
                            arg_str += f'{param.trans}=[{list_str}]'
                        # Ordinary cases
                        else:
                            arg_str += f'{param.trans}={conn[1].get_output_variable_name(conn[2])}'
                    counter += 1
                elif param.type == 'input':
                    raise RuntimeError(f"Missing connection '{param.name}' in node {self.name}")

        # Other static keyword arguments from dictionary
        if in_class:
            return f"{output_str} = self.eval_op('{self.name}'{', ' + arg_str if arg_str else ''})"
        else:
            arg_str += (', ' if counter else '') + f"**params['{self.name}']"
            return f'{output_str} = {self.func}({arg_str})'

    # Print dynamic node parameter description
    def get_dynamic_functions_str(self):
        # Collect dynamic parameters and quit if none
        dparams = [param for param in self.params_list if param.is_dynamic()]
        if not dparams:
            return ''

        # Header
        str_list = [f"# Node '{self.name}'"]
        write = lambda s: str_list.append(s)

        # SBS dynamic parameters
        sbs_dparams = set(itertools.chain(*[param.dynamic_params for param in dparams]))
        for sbs_dp in sbs_dparams:
            write(sbs_dp.get_func_str())

        # Dynamic node parameters
        for dp in dparams:
            write(dp.get_func_str())
        if not str_list[-1].endswith('\n'):
            write('')

        return '\n'.join(str_list)

### Special nodes

class SBSInputNode(SBSNode):
    '''SBS input node (only the reference of an input parameter; no implementation).
    '''
    def __init__(self, name, uid, param_ref):
        super().__init__(name, uid, 'Input', None)
        self.ref = param_ref
        param_ref.parent = self

    def get_default_params(self):
        return []

    # Set the input parameter as reachable
    def set_output_bridge_reachable(self, uid):
        super().set_output_bridge_reachable(uid)
        self.ref.reachable = True

class SBSOutputNode(SBSNode):
    '''SBS output node (only the reference of an output; no implementation).
    '''
    def __init__(self, name, uid, output_ref):
        super().__init__(name, uid, 'Output', None)
        self.ref = output_ref
        output_ref.parent = self

    def get_default_params(self):
        return []

    # Get the variable reference of the output
    def get_variable_reference(self):
        _, conn_ref, conn_output_ref = self.connections[0]
        return conn_ref.get_output_variable_name(conn_output_ref)

    def is_reachable(self):
        return self.ref.is_optimizable()

class SBSGeneratorNode(SBSNode):
    '''SBS generator node for noises and other inputs (no implementation).
    A generator may contain multiple untrainable parameter inputs.
    '''
    def __init__(self, name, uid, output_res=None, use_alpha=False):
        super().__init__(name, uid, 'Generator', None, output_res, use_alpha)
        self.refs = {}

    def get_default_params(self):
        return []

    # Set the relevant input parameter as reachable
    def set_output_bridge_reachable(self, uid):
        super().set_output_bridge_reachable(uid)
        self.refs[uid].reachable = True

    # Note: this function should be called after 'update_output_variable_names'
    def init_sbs_params(self):
        for uid, bridge in self.output_bridges.items():
            self.refs[uid] = SBSParameter(bridge.var_name, uid, bridge.type, parent=self)

    def get_sbs_param(self, uid):
        return self.refs[uid]

    def get_sbs_params_list(self):
        return [param for _, param in self.refs.items()]

### Helpers for node parameter converters

PC = SBSNodeParameterConverter

def normalizer(max_val):
    func_template = f'convert = lambda x: x / {max_val}'
    return PC(lambda p: p / max_val, func_template=func_template)

def to_zero_one_normalizer(max_val):
    func_template = f'convert = lambda x: to_zero_one(x / {max_val})'
    return PC(lambda p: to_zero_one(p / max_val), func_template=func_template)

def to_zero_one_helper():
    return PC(to_zero_one, func_template='convert = to_zero_one')

def to_zero_one_helper_getitem(index):
    func_template = f'convert = lambda x: to_zero_one(x[{index}])'
    return PC(lambda p: to_zero_one(p[index]), func_template=func_template)

def remainder_helper(denom):
    func_template = f'convert = lambda x: th.remainder(x, {denom})'
    return PC(lambda p: np.remainder(p, denom), func_template=func_template)

def intensity_helper(max_intensity):
    func_template = lambda intensity: f'convert = lambda x: x / {max(max_intensity, intensity * 2.0)}'
    return PC(lambda intensity: min(intensity / max_intensity, 0.5), func_template=func_template)

def max_intensity_helper(max_intensity):
    return PC(lambda intensity: max(max_intensity, intensity * 2.0), trainable=False)

def intensity_helper_zero_one(max_intensity):
    func_template = lambda intensity: f'convert = lambda x: to_zero_one(x / {max(max_intensity, abs(intensity * 2.0))})'
    return PC(lambda intensity: to_zero_one(intensity / max(max_intensity, abs(intensity * 2.0))), func_template=func_template)

def max_intensity_helper_zero_one(max_intensity):
    return PC(lambda intensity: max(max_intensity, abs(intensity * 2.0)), trainable=False)

def intensity_helper_getitem(max_intensity, index):
    func_template = lambda p: f'convert = lambda p: to_zero_one(p[{index}] / {max(max_intensity, abs(p[index] * 2.0))})'
    return PC(lambda p: to_zero_one(p[index] / max(max_intensity, abs(p[index] * 2.0))), func_template=func_template)

def max_intensity_helper_getitem(max_intensity, index):
    return PC(lambda p: max(max_intensity, abs(p[index] * 2.0)), trainable=False)

### Filter nodes

class SBSBlendNode(SBSNode):
    '''SBS blend node.

    Format for each parameter: `(sbs_name, trans_name, type, default_value=None, convert_func=None)`
      - `sbs_name`: name or a list of names as appeared in the `*.sbs` file.
      - `trans_name`: name used in the node implementation in functional.py.
      - `type`: parameter type which can be `'input'`, `'output'`, or `'keyword'`.
      - `default_value` (optional): default parameter value from `*.sbs` file.
      - `convert_func` (optional): the function for converting values in `*.sbs` to our implementation.
        It takes two arguments - new parameter and the existing parameter. By default, the parameter is
        copied directly without conversion.

    Notes:
      1. [CRUCIAL] Write all parameters according to their orders in functional.py.
      2. [CRUCIAL] Make sure that sbs_name and trans_name match.
      3. trans_name can be arbitrary for `'output'` parameters.
      4. sbs_name is not unique for extra parameters in our implementation, e.g. max_intensity. However,
         make sure that it does not have any conflict with other parameters.
      5. convert_func can be any callable object which takes only one positional argument. Two additional
         arguments is required if the parameter is related to multiple parameters in SBS.
      6. For atomic nodes that do not have output identifiers, set sbs_name as '' by default.
    '''
    blending_mode_list = [
        'copy', 'add', 'subtract', 'multiply', 'add_sub',
        'max', 'min', 'switch', 'divide', 'overlay', 'screen', 'soft_light'
    ]
    def get_default_params(self):
        mode_func = lambda p: self.blending_mode_list[p]
        return [
            ('source', 'img_fg', 'keyword'),
            ('destination', 'img_bg', 'keyword'),
            ('opacity', 'blend_mask', 'keyword'),
            ('blendingmode', 'blending_mode', 'keyword', 'copy', mode_func),
            ('maskrectangle', 'cropping', 'keyword', [0.0,1.0,0.0,1.0]),
            ('opacitymult', 'opacity', 'keyword', 1.0),
            ('', 'trainable', 'keyword', [True]),
            ('', '', 'output'),
        ]

class SBSBlurNode(SBSNode):
    '''SBS blur node.
    '''
    def get_default_params(self):
        default_intensity = 0.5
        max_intensity = 20.0
        return [
            ('input1', 'img_in', 'input'),
            ('intensity', 'intensity', 'keyword', default_intensity, intensity_helper(max_intensity)),
            ('intensity', 'max_intensity', 'keyword', max_intensity, max_intensity_helper(max_intensity)),
            ('', 'trainable', 'keyword', [True]),            
            ('', '', 'output'),
        ]

class SBSChannelShuffleNode(SBSNode):
    '''SBS channel shuffle node.
    '''
    def update_shuffle_indices(self, new_val, exist_val, sbs_name):
        channel_dict = {'channelred': 0, 'channelgreen': 1, 'channelblue': 2, 'channelalpha': 3}
        exist_val[channel_dict[sbs_name]] = new_val - (new_val > 3)
        return exist_val

    def get_default_params(self):
        return [
            ('input1', 'img_in', 'input'),
            ('input2', 'img_in_aux', 'keyword'),
            ('','use_alpha', 'keyword', self.use_alpha),
            (['channelred', 'channelgreen', 'channelblue', 'channelalpha'], 'shuffle_idx', 'keyword',
             [0, 1, 2, 3], self.update_shuffle_indices),
            ('', '', 'output'),
        ]

class SBSCurveNode(SBSNode):
    '''SBS curve node.
    '''
    def get_anchors(self, new_val):
        assert len(new_val) >= 2
        anchors = []
        for cell in new_val:
            pos = cell['position']
            cpl = cell['position'] if cell['isLeftBroken'] else cell['left']
            cpr = cell['position'] if cell['isRightBroken'] else cell['right']
            anchors.append(pos + cpl + cpr)
        return anchors

    def get_default_params(self):
        return [
            ('input1', 'img_in', 'input'),
            ('curveluminance', 'num_anchors', 'keyword', 2, len),
            ('curveluminance', 'anchors', 'keyword', [[0.0] * 6, [1.0] * 6], self.get_anchors),
            ('', 'trainable', 'keyword', [True]),
            ('', '', 'output'),
        ]

class SBSDBlurNode(SBSNode):
    '''SBS directional blur node.
    '''
    def get_default_params(self):
        default_intensity = 0.5
        max_intensity = 20.0
        return [
            ('input1', 'img_in', 'input'),
            ('intensity', 'intensity', 'keyword', default_intensity, intensity_helper(max_intensity)),
            ('intensity', 'max_intensity', 'keyword', max_intensity, max_intensity_helper(max_intensity)),
            ('mblurangle', 'angle', 'keyword', 0.0, remainder_helper(1.0)),
            ('', 'trainable', 'keyword', [True, True]),
            ('', '', 'output'),
        ]

class SBSDWarpNode(SBSNode):
    '''SBS directional warp node.
    '''
    def get_default_params(self):
        default_intensity = 0.5
        max_intensity = 20.0
        return [
            ('input1', 'img_in', 'input'),
            ('inputintensity', 'intensity_mask', 'input'),
            ('intensity', 'intensity', 'keyword', default_intensity, intensity_helper(max_intensity)),
            ('intensity', 'max_intensity', 'keyword', max_intensity, max_intensity_helper(max_intensity)),
            ('warpangle', 'angle', 'keyword', 0.0, remainder_helper(1.0)),
            ('', 'trainable', 'keyword', [True, True]),
            ('', '', 'output'),
        ]

class SBSDistanceNode(SBSNode):
    '''SBS distance node.
    '''
    def get_default_params(self):
        max_distance = 256.0
        default_distance = 10.0 / 256.0
        return [
            ('mask', 'img_mask', 'input'),
            ('source', 'img_source', 'keyword'),
            ('colorswitch', 'mode', 'keyword', 'gray', lambda p: 'color' if p else 'gray'),
            ('combinedistance', 'combine', 'keyword', True),
            ('', 'use_alpha', 'keyword', self.use_alpha),
            ('distance', 'dist', 'keyword', default_distance, intensity_helper(max_distance)),
            ('distance', 'max_dist', 'keyword', max_distance, max_intensity_helper(max_distance)),
            ('', 'trainable', 'keyword', [True]),
            ('', '', 'output')
        ]

class SBSEmbossNode(SBSNode):
    '''SBS emboss node.
    '''
    def get_default_params(self):
        default_intensity = 0.5
        max_intensity = 10.0
        return [
            ('input1', 'img_in', 'input'),
            ('inputgradient', 'height_map', 'input'),
            ('intensity', 'intensity', 'keyword', default_intensity, intensity_helper(max_intensity)),
            ('intensity', 'max_intensity', 'keyword', max_intensity, max_intensity_helper(max_intensity)),
            ('lightangle', 'light_angle', 'keyword', 0.0, remainder_helper(1.0)),
            ('highlightcolor', 'highlight_color', 'keyword', [1.0, 1.0, 1.0]),
            ('shadowcolor', 'shadow_color', 'keyword', [0.0, 0.0, 0.0]),
            ('', 'trainable', 'keyword', [True, True, True, True]),
            ('', '', 'output')
        ]

class SBSGradientMapNode(SBSNode):
    '''SBS gradient map node.
    '''
    def update_anchors(self, new_val, exist_val, sbs_name):
        if sbs_name == 'colorswitch':
            return self.update_default_anchors()
        if self.params['colorswitch'][0].val == 'gray':
            anchors = np.array([[cell['position'], (cell['value'][0]+cell['value'][1]+cell['value'][2])/3.0] for cell in new_val])
        elif self.use_alpha:
            anchors = np.array([[cell['position']] + cell['value'] for cell in new_val])
        else:
            anchors = np.array([[cell['position']] + cell['value'][:3] for cell in new_val])
        return sorted(anchors.tolist(), key=lambda p: p[0])

    def update_default_anchors(self):
        num_cols = 2 if self.params['colorswitch'][0].val == 'gray' else 4 + self.use_alpha
        anchors = [[0.0] * num_cols, [1.0] * num_cols]
        if num_cols >= 4 and self.use_alpha:
            anchors[0][4] = 1.0
        return anchors

    def update_interpolate_flag(self, new_val):
        return any([cell['midpoint'] < 0 for cell in new_val])

    def get_default_params(self):
        mode_func = lambda p: 'color' if p else 'gray'
        return [
            ('input1', 'img_in', 'input'),
            ('gradientrgba', 'interpolate', 'keyword', True, self.update_interpolate_flag),
            ('colorswitch', 'mode', 'keyword', 'color', mode_func),
            ('', 'use_alpha', 'keyword', self.use_alpha),
            ('gradientrgba', 'num_anchors', 'keyword', 2, len),
            (['colorswitch', 'gradientrgba'], 'anchors', 'keyword', self.update_default_anchors, self.update_anchors),
            ('', 'trainable', 'keyword', [True]),
            ('', '', 'output'),
        ]

class SBSGradientMapDynNode(SBSNode):
    '''SBS gradient map node.
    '''

    def get_default_params(self):
        mode_func = lambda p: 'color' if p else 'gray'
        return [
            ('input1', 'img_in', 'input'),
            ('input2', 'img_gradient', 'input'),
            ('uvselector', 'orientation', 'keyword', 'horizontal', lambda p: 'vertical' if p else 'horizontal'),
            ('', 'use_alpha', 'keyword', self.use_alpha),
            ('coordinate', 'position', 'keyword', 0.0),
            ('', '', 'output'),
        ]

class SBSC2GNode(SBSNode):
    '''SBS grayscale conversion node.
    '''
    def get_default_params(self):
        return [
            ('input1', 'img_in', 'input'),
            ('alphamult', 'flatten_alpha', 'keyword', False),
            ('channelsweights', 'rgba_weights', 'keyword', [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 0.0]),
            ('mattelevel', 'bg', 'keyword', 1.0),
            ('', 'trainable', 'keyword', [True, True]),
            ('', '', 'output'),
        ]

class SBSHSLNode(SBSNode):
    '''SBS hsl node.
    '''
    def get_default_params(self):
        return [
            ('input1', 'img_in', 'input'),
            ('hue', 'hue', 'keyword', 0.5),
            ('saturation', 'saturation', 'keyword', 0.5),
            ('luminosity', 'lightness', 'keyword', 0.5),
            ('', 'trainable', 'keyword', [True, True, True]),
            ('', '', 'output')
        ]

class SBSLevelsNode(SBSNode):
    '''SBS levels node.
    '''
    def update_levels_anchors(self, new_val):
        if isinstance(new_val, (int, float)):
            return [new_val]
        elif new_val[0] == new_val[1] and new_val[0] == new_val[2]:
            # return a single value list
            return [new_val[0]]
        elif not self.use_alpha:
            # return a length 3 vector
            return new_val[:3]
        else:
            # return the full length (4) vector
            return new_val

    def update_levels_anchors_template(self):
        p_str = 'p' if self.use_alpha else 'p[:3]'
        return f"convert = lambda p: {p_str} if p.ndim else p[None]"

    def get_default_params(self):
        func_template = self.update_levels_anchors_template()
        levels_anchors_helper = lambda: PC(self.update_levels_anchors, func_template=func_template)
        return [
            ('input1', 'img_in', 'input'),
            ('levelinlow', 'in_low', 'keyword', [0.0], levels_anchors_helper()),
            ('levelinmid', 'in_mid', 'keyword', [0.5], levels_anchors_helper()),
            ('levelinhigh', 'in_high', 'keyword', [1.0], levels_anchors_helper()),
            ('leveloutlow', 'out_low', 'keyword', [0.0], levels_anchors_helper()),
            ('levelouthigh', 'out_high', 'keyword', [1.0], levels_anchors_helper()),
            ('', 'trainable', 'keyword', [True, True, True, True, True]),
            ('', '', 'output'),
        ]

class SBSNormalNode(SBSNode):
    '''SBS normal node.
    '''
    def get_default_params(self):
        max_intensity = 3.0
        default_intensity = 1.0 / max_intensity
        return [
            ('input1', 'img_in', 'input'),
            ('', 'mode', 'keyword', 'tangent_space'),
            ('inversedy', 'normal_format', 'keyword', 'dx', lambda p: 'gl' if p else 'dx'),
            ('input2alpha', 'use_input_alpha', 'keyword', True, lambda p: True if p else False),
            ('', 'use_alpha', 'keyword', self.use_alpha),
            # ('intensity', 'intensity', 'keyword', default_intensity, intensity_helper(max_intensity)),
            # ('intensity', 'max_intensity', 'keyword', max_intensity, max_intensity_helper(max_intensity)),
            ('intensity', 'intensity', 'keyword', to_zero_one(default_intensity), intensity_helper_zero_one(max_intensity)),
            ('intensity', 'max_intensity', 'keyword', max_intensity, max_intensity_helper_zero_one(max_intensity)),
            ('', 'trainable', 'keyword', [True]),
            ('', '', 'output'),
        ]

class SBSSharpenNode(SBSNode):
    '''
    SBS sharpen node.
    '''
    def get_default_params(self):
        max_intensity = 3.0
        default_intensity = 1.0 / max_intensity
        return [
            ('input1', 'img_in', 'input'),
            ('intensity', 'intensity', 'keyword', default_intensity, intensity_helper(max_intensity)),
            ('intensity', 'max_intensity', 'keyword', max_intensity, max_intensity_helper(max_intensity)),
            ('', 'trainable', 'keyword', [True]),
            ('', '', 'output'),
        ]

class SBSTransform2dNode(SBSNode):
    '''SBS transform 2d node.
    '''
    def get_default_params(self):
        max_intensity = 1.0
        return [
            ('input1', 'img_in', 'input'),
            ('tiling', 'tile_mode', 'keyword', 3),
            ('filtering', 'sample_mode', 'keyword', 'bilinear', lambda p: 'nearest' if p else 'bilinear'),
            ('mipmapmode', 'mipmap_mode', 'keyword', 'auto', lambda p: 'manual' if p else 'auto'),
            ('manualmiplevel', 'mipmap_level', 'keyword', 0),
            ('matrix22', 'x1', 'keyword', to_zero_one(1.0), intensity_helper_getitem(max_intensity, 0)),
            ('matrix22', 'x1_max', 'keyword', max_intensity, max_intensity_helper_getitem(max_intensity, 0)),
            ('matrix22', 'x2', 'keyword', to_zero_one(0.0), intensity_helper_getitem(max_intensity, 1)),
            ('matrix22', 'x2_max', 'keyword', max_intensity, max_intensity_helper_getitem(max_intensity, 1)),
            ('offset', 'x_offset', 'keyword', to_zero_one(0.0), intensity_helper_getitem(max_intensity, 0)),
            ('offset', 'x_offset_max', 'keyword', max_intensity, max_intensity_helper_getitem(max_intensity, 0)),
            ('matrix22', 'y1', 'keyword', to_zero_one(0.0), intensity_helper_getitem(max_intensity, 2)),
            ('matrix22', 'y1_max', 'keyword', max_intensity, max_intensity_helper_getitem(max_intensity, 2)),
            ('matrix22', 'y2', 'keyword', to_zero_one(1.0), intensity_helper_getitem(max_intensity, 3)),
            ('matrix22', 'y2_max', 'keyword', max_intensity, max_intensity_helper_getitem(max_intensity, 3)),
            ('offset', 'y_offset', 'keyword', to_zero_one(0.0), intensity_helper_getitem(max_intensity, 1)),
            ('offset', 'y_offset_max', 'keyword', max_intensity, max_intensity_helper_getitem(max_intensity, 1)),
            ('mattecolor', 'matte_color', 'keyword', [0.0, 0.0, 0.0, 1.0]),
            ('', 'trainable', 'keyword', [True, True, True, True, True, True, True]),
            ('', '', 'output'),
        ]

class SBSUniformColorNode(SBSNode):
    '''SBS uniform color node.
    '''
    def get_default_params(self):
        color_func = lambda p: p if isinstance(p, list) else [p, p, p, 1.0]
        color_func_template = 'convert = lambda p: p if p.ndim else th.stack((p, p, p, th.tensor(1.0)))'
        return [
            ('colorswitch', 'mode', 'keyword', 'color', lambda p: 'color' if p else 'gray'),
            ('', 'num_imgs', 'keyword', 1),
            ('outputsize', 'res_h', 'keyword', 1 << self.res[0], lambda p: 1 << p[0]),
            ('outputsize', 'res_w', 'keyword', 1 << self.res[1], lambda p: 1 << p[1]),
            ('', 'use_alpha', 'keyword', self.use_alpha),
            ('outputcolor', 'rgba', 'keyword', [0.0, 0.0, 0.0, 1.0], PC(color_func, func_template=color_func_template)),
            ('', 'trainable', 'keyword', [True]),
            ('', '', 'output'),
        ]

class SBSWarpNode(SBSNode):
    '''SBS warp node.
    '''
    def get_default_params(self):
        max_intensity = 2.0
        default_intensity = 1.0 / max_intensity
        return [
            ('input1', 'img_in', 'input'),
            ('inputgradient', 'intensity_mask', 'input'),
            ('intensity', 'intensity', 'keyword', default_intensity, intensity_helper(max_intensity)),
            ('intensity', 'max_intensity', 'keyword',  max_intensity, max_intensity_helper(max_intensity)),
            ('', 'trainable', 'keyword', [True]),
            ('', '', 'output')
        ]

class SBSPassthroughNode(SBSNode):
    '''SBS linear to srgb node.
    '''
    def get_default_params(self):
        return [
            ('input', 'img_in', 'input'),
            ('', '', 'output'),
        ]

class SBSL2SRGBNode(SBSNode):
    '''SBS linear to srgb node.
    '''
    def get_default_params(self):
        return [
            ('input', 'img_in', 'input'),
            ('', '', 'output'),
        ]

class SBSSRGB2LNode(SBSNode):
    '''SBS srgb to linear node.
    '''
    def get_default_params(self):
        return [
            ('input', 'img_in', 'input'),
            ('', '', 'output'),
        ]


class SBSCurvatureNode(SBSNode):
    '''SBS curvature node.
    '''
    def get_default_params(self):
        default_intensity = 0.1
        max_intensity = 10.0
        return [
            ('Input', 'normal', 'input'),
            ('normal_format', 'normal_format', 'keyword', 'dx', lambda p: 'gl' if p else 'dx'),
            ('intensity', 'emboss_intensity', 'keyword', default_intensity, intensity_helper(max_intensity)),
            ('intensity', 'emboss_max_intensity', 'keyword', max_intensity, max_intensity_helper(max_intensity)),
            ('', 'trainable', 'keyword', [True]),
            ('Output', '', 'output'),
        ]

class SBSInvertGrayscaleNode(SBSNode):
    '''SBS invert node (color or grayscale).
    '''
    def get_default_params(self):
        return [
            ('Source', 'img_in', 'input'),
            ('invert', 'invert_switch', 'keyword', True),
            ('Invert_Grayscale', '', 'output'),
        ]

class SBSInvertColorNode(SBSNode):
    '''SBS invert node (color or grayscale).
    '''
    def get_default_params(self):
        return [
            ('Source', 'img_in', 'input'),
            ('invert', 'invert_switch', 'keyword', True),
            ('Invert_Color', '', 'output'),
        ]

class SBSHistogramScanNode(SBSNode):
    '''SBS histogram scan node.
    '''
    def get_default_params(self):
        return [
            ('Input_1', 'img_in', 'input'),
            ('Invert_Position', 'invert_position', 'keyword', False),
            ('Position', 'position', 'keyword', 0.0),
            ('Contrast', 'contrast', 'keyword', 0.0),
            ('', 'trainable', 'keyword', [True, True]),
            ('Output', '', 'output'),
        ]

class SBSHistogramRangeNode(SBSNode):
    '''SBS histogram range node.
    '''
    def get_default_params(self):
        return [
            ('input', 'img_in', 'input'),
            ('range', 'ranges', 'keyword', 0.5),
            ('position', 'position', 'keyword', 0.5),
            ('', 'trainable', 'keyword', [True, True]),
            ('output', '', 'output'),
        ]

class SBSHistogramSelectNode(SBSNode):
    '''SBS histogram s,elect node.
    '''
    def get_default_params(self):
        return [
            ('input', 'img_in', 'input'),
            ('position', 'position', 'keyword', 0.5),
            ('range', 'ranges', 'keyword', 0.25),
            ('constrast', 'contrast', 'keyword', 0.0),
            ('', 'trainable', 'keyword', [True, True]),
            ('output', '', 'output'),
        ]

class SBSEdgeDetectNode(SBSNode):
    '''SBS edge detect node.
    '''
    def get_default_params(self):
        max_width = 16.0
        default_width = 2.0/max_width
        max_roundness = 16.0
        default_roundness = 4.0/max_roundness
        return [
            ('input', 'img_in', 'input'),
            ('invert', 'invert_flag', 'keyword', False),
            ('edge_width', 'edge_width', 'keyword', default_width, intensity_helper(max_width)),
            ('edge_width', 'max_edge_width', 'keyword', max_width, max_intensity_helper(max_width)),
            ('edge_roundness', 'edge_roundness', 'keyword', default_roundness, intensity_helper(max_roundness)),
            ('edge_roundness', 'max_edge_roundness', 'keyword', max_roundness, max_intensity_helper(max_roundness)),
            ('tolerance', 'tolerance', 'keyword', 0.0),
            ('', 'trainable', 'keyword', [True, True, True]),
            ('output', '', 'output'),
        ]

class SBSSafeTransformNode(SBSNode):
    '''SBS safe transform node (color or grayscale).
    '''
    def get_default_params(self):
        symmetry_vals = ['none', 'X', 'Y', 'X+Y']
        return [
            ('input', 'img_in', 'input'),
            ('tile', 'tile', 'keyword', 1),
            ('tile_safe_rotation', 'tile_safe_rot', 'keyword', True),
            ('symmetry', 'symmetry', 'keyword', 'none', lambda p: symmetry_vals[p]),
            ('tiling', 'tile_mode', 'keyword', 3),
            ('mipmapmode', 'mipmap_mode', 'keyword', 'auto', lambda p: 'manual' if p else 'auto'),
            ('manualmiplevel', 'mipmap_level', 'keyword', 0),
            ('offset', 'offset_x', 'keyword', 0.5, to_zero_one_helper_getitem(0)),
            ('offset', 'offset_y', 'keyword', 0.5, to_zero_one_helper_getitem(1)),
            ('rotation', 'angle', 'keyword', 0.0, remainder_helper(1.0)),
            ('', 'trainable', 'keyword', [True, True, True]),
            ('output', '', 'output')
        ]

class SBSBlurHQNode(SBSNode):
    '''SBS blur hq node (color or grayscale).
    '''
    def get_default_params(self):
        max_intensity = 16.0
        default_intensity = 10.0 / max_intensity
        return [
            ('Source', 'img_in', 'input'),
            ('Quality', 'high_quality', 'keyword', False),
            ('Intensity', 'intensity', 'keyword', default_intensity, intensity_helper(max_intensity)),
            ('Intensity', 'max_intensity', 'keyword',  max_intensity, max_intensity_helper(max_intensity)),
            ('', 'trainable', 'keyword', [True]),
            ('Blur_HQ', '', 'output')
        ]

class SBSNonUniformBlurNode(SBSNode):
    '''SBS non-uniform blur node (color or grayscale).
    '''
    def get_default_params(self):
        default_intensity = 0.2
        max_intensity = 50.0
        return [
            ('Source', 'img_in', 'input'),
            ('Effect', 'img_mask', 'input'),
            ('Samples', 'samples', 'keyword', 4),
            ('Blades', 'blades', 'keyword', 5),
            ('Intensity', 'intensity', 'keyword', default_intensity, intensity_helper(max_intensity)),
            ('Intensity', 'max_intensity', 'keyword', max_intensity, max_intensity_helper(max_intensity)),
            ('Anisotropy', 'anisotropy', 'keyword', 0.0),
            ('Asymmetry', 'asymmetry', 'keyword', 0.0),
            ('Angle', 'angle', 'keyword', 0.0, remainder_helper(1.0)),
            ('', 'trainable', 'keyword', [True, True, True, True]),
            ('Non_Uniform_Blur', '', 'output')
        ]

class SBSBevelNode(SBSNode):
    '''SBS bevel node.
    '''
    def raise_corner_type_error(self, new_val):
        if new_val:
            raise NotImplementedError('Angular corner type is not supprted')
        return new_val

    def raise_custom_type_error(self, new_val):
        if new_val:
            raise NotImplementedError('Custom curve is not supprted')
        return new_val

    def get_default_params(self):
        default_smoothing = 0.0
        max_smoothing = 5.0
        default_normal_intensity = 0.2
        max_normal_intensity = 50.0
        max_dist = 1.0
        return [
            ('input', 'img_in', 'input'),
            ('non_uniform_blur', 'non_uniform_blur_flag', 'keyword', True),
            ('', 'use_alpha', 'keyword', self.use_alpha),
            ('bevel_mode', '', 'keyword', None, self.raise_corner_type_error),
            ('Use_Custom_Curve', '', 'keyword', None, self.raise_custom_type_error),
            ('distance', 'dist', 'keyword', to_zero_one(0.5), intensity_helper_zero_one(max_dist)),
            ('distance', 'max_dist', 'keyword', max_dist, max_intensity_helper_zero_one(max_dist)),
            ('smoothing', 'smoothing', 'keyword', default_smoothing, intensity_helper(max_smoothing)),
            ('smoothing', 'max_smoothing', 'keyword', max_smoothing, max_intensity_helper(max_smoothing)),
            ('normal_intensity', 'normal_intensity', 'keyword', default_normal_intensity, intensity_helper(max_normal_intensity)),
            ('normal_intensity', 'max_normal_intensity', 'keyword', max_normal_intensity, intensity_helper(max_normal_intensity)),
            ('', 'trainable', 'keyword', [True, True, True]),
            ('height', '', 'output'),
            ('normal', '', 'output')
        ]

class SBSSlopeBlurNode(SBSNode):
    '''SBS slope blur node (color or grayscale).
    '''
    def get_default_params(self):
        max_intensity = 16.0
        default_intensity = 10.0 / max_intensity
        return [
            ('Source', 'img_in', 'input'),
            ('Effect', 'img_mask', 'input'),
            ('Samples', 'samples', 'keyword', 8),
            ('mode', 'mode', 'keyword', 'blur', lambda p: 'min' if p == 6 else 'max'),
            ('Intensity', 'intensity', 'keyword', default_intensity, intensity_helper(max_intensity)),
            ('Intensity', 'max_intensity', 'keyword', max_intensity, max_intensity_helper(max_intensity)),
            ('', 'trainable', 'keyword', [True]),
            ('Slope_Blur', '', 'output')
        ]

class SBSMosaicNode(SBSNode):
    '''SBS mosaic node (color or grayscale).
    '''
    def get_default_params(self):
        default_intensity = 0.5
        max_intensity = 1.0
        return [
            ('Source', 'img_in', 'input'),
            ('Effect', 'img_mask', 'input'),
            ('Samples', 'samples', 'keyword', 8),
            ('Intensity', 'intensity', 'keyword', default_intensity, intensity_helper(max_intensity)),
            ('Intensity', 'max_intensity', 'keyword', max_intensity, max_intensity_helper(max_intensity)),
            ('', 'trainable', 'keyword', [True]),
            ('Mosaic', '', 'output')
        ]

class SBSAutoLevelsNode(SBSNode):
    '''SBS auto levels node.
    '''
    def get_default_params(self):
        return [
            ('Input', 'img_in', 'input'),
            ('Output', '', 'output')
        ]

class SBSAmbientOcclusionNode(SBSNode):
    '''SBS ambient occlusion node.
    '''
    def get_default_params(self):
        max_spreading = 1.0
        return [
            ('Source', 'img_in', 'input'),
            ('spreading', 'spreading', 'keyword', 0.15, intensity_helper(max_spreading)),
            ('spreading', 'max_spreading', 'keyword', max_spreading, max_intensity_helper(max_spreading)),
            ('equalizer', 'equalizer', 'keyword', [0.0, 0.0, 0.0]),
            ('levels', 'levels_param', 'keyword', [0.0, 0.5, 1.0]),
            ('', 'trainable', 'keyword', [True, True, True]),
            ('ambient_occlusion', '', 'output')
        ]

class SBSHBAONode(SBSNode):
    '''SBS hbao node.
    '''
    def get_default_params(self):
        return [
            ('input', 'img_in', 'input'),
            ('samples', 'quality', 'keyword', 8),
            ('height_depth', 'depth', 'keyword', 0.1),
            ('radius', 'radius', 'keyword', 1.0),
            ('', 'trainable', 'keyword', [True, True]),
            ('output', '', 'output')
        ]

class SBSHighpassNode(SBSNode):
    '''SBS highpass node.
    '''
    def get_default_params(self):
        max_radius = 64.0
        default_radius = 6.0 / max_radius
        return [
            ('Source', 'img_in', 'input'),
            ('Radius', 'radius' ,'keyword', default_radius, intensity_helper(max_radius)),
            ('Radius', 'max_radius' ,'keyword', max_radius, max_intensity_helper(max_radius)),
            ('', 'trainable', 'keyword', [True]),
            ('Highpass', '', 'output')
        ]

class SBSNormalNormalizeNode(SBSNode):
    '''SBS normal combine node.
    '''
    def get_default_params(self):
        return [
            ('Normal', 'normal', 'input'),
            ('Normalise', '', 'output')
        ]

class SBSNormalCombineNode(SBSNode):
    '''SBS normal combine node.
    '''
    def get_default_params(self):
        combine_mode = ['whiteout', 'channel_mixer', 'detail_oriented']
        return [
            ('Input', 'normal_one', 'input'),
            ('Input_1', 'normal_two', 'input'),
            ('blend_quality', 'mode', 'keyword', 'whiteout', lambda p: combine_mode[p]),
            ('normal', '', 'output')
        ]

class SBSChannelMixerNode(SBSNode):
    '''SBS normal combine node.
    '''
    def get_default_params(self):
        scale_func = lambda p: [i/400.0+0.5 for i in p]
        return [
            ('Input', 'img_in', 'input'),
            ('monochrome', 'monochrome', 'keyword', False),
            ('red_channel', 'red', 'keyword', [0.75,0.5,0.5,0.5], scale_func),
            ('green_channel', 'green', 'keyword', [0.5,0.75,0.5,0.5], scale_func),
            ('blue_channel', 'blue', 'keyword', [0.5,0.5,0.75,0.5], scale_func),
            ('', 'trainable', 'keyword', [True, True, True]),
            ('Output', '', 'output')
        ]

class SBSRGBASplitNode(SBSNode):
    '''SBS rgba_split node.
    '''
    def get_default_params(self):
        return [
            ('RGBA', 'rgba', 'input'),
            ('R', '', 'output'),
            ('G', '', 'output'),
            ('B', '', 'output'),
            ('A', '', 'output')
        ]

class SBSRGBAMergeNode(SBSNode):
    '''SBS rgba_merge node.
    '''
    def get_default_params(self):
        return [
            ('R', 'r', 'input'),
            ('G', 'g', 'input'),
            ('B', 'b', 'input'),
            ('A', 'a', 'keyword', None),
            ('', 'use_alpha', 'keyword', self.use_alpha),
            ('RGBA_Merge', '', 'output')
        ]

class SBSMultiSwitchNode(SBSNode):
    '''SBS multi switch node.
    '''
    default_img_list = [None] * 20

    def update_img_list(self, new_val, exist_val, sbs_name):
        if sbs_name == 'input_number':
            exist_val = exist_val[:new_val]
        else:
            idx = int(sbs_name.split('_')[1])
            exist_val[idx-1] = new_val
        return exist_val

    def get_default_params(self):
        input_list = ['input_%d' % d for d in range(1,21)]
        input_list.append('input_number')
        return [
            (input_list, 'img_list', 'keyword', [None] * 20, self.update_img_list),
            ('input_number', 'input_number', 'keyword', 2),
            ('input_selection', 'input_selection', 'keyword', 1),
            ('output', '', 'output')
        ]

class SBSPbrConverterNode(SBSNode):
    '''SBS pbr_converter node.
    '''
    def get_default_params(self):
        return [
            ('basecolor', 'base_color', 'input'),
            ('metallic', 'metallic', 'input'),
            ('roughness', 'roughness', 'input'),
            ('diffuse', '', 'output'),
            ('specular', '', 'output'),
            ('glossiness', '', 'output')
        ]

class SBSAlphaSplitNode(SBSNode):
    '''SBS alpha_split node.
    '''
    def get_default_params(self):
        return [
            ('RGBA', 'rgba', 'input'),
            ('RGB', '', 'output'),
            ('A', '', 'output')
        ]

class SBSAlphaMergeNode(SBSNode):
    '''SBS alpha_blend node.
    '''
    def get_default_params(self):
        return [
            ('RGB', 'rgb', 'input'),
            ('A', 'a', 'input'),
            ('RGB-A_Merge', '', 'output')
        ]

class SBSSwitchNode(SBSNode):
    '''SBS switch node.
    '''
    def get_default_params(self):
        return [
            ('input_1', 'img_1', 'input'),
            ('input_2', 'img_2', 'input'),
            ('switch', 'flag', 'keyword', True),
            ('output', '', 'output')
        ]

class SBSNormalBlendNode(SBSNode):
    '''SBS normal blend node.
    '''
    def get_default_params(self):
        return [
            ('NormalFG', 'normal_fg', 'input'),
            ('NormalBG', 'normal_bg', 'input'),
            ('Mask', 'mask', 'keyword'),
            ('Use_Mask', 'use_mask', 'keyword', True, lambda p: True if int(p) else False),
            ('Opacity', 'opacity', 'keyword', 1.0),
            ('', 'trainable', 'keyword', [True]),            
            ('Normal_Blend', '', 'output'),
        ]

class SBSMirrorNode(SBSNode):
    '''SBS normal blend node.
    '''
    def get_default_params(self):
        return [
            ('input', 'img_in', 'input'),
            ('mirror_type', 'mirror_axis', 'keyword'),
            (['axis_x', 'axis_y'], 'offset', 'keyword', 0.5),
            (['invert_x', 'invert_y'], 'invert_axis', 'keyword', False),
            ('corner_type', 'corner_type', 'keyword', 0),
            ('', 'trainable', 'keyword', [True]),
            ('output', '', 'output')
        ]

class SBSHeightToNormalNode(SBSNode):
    '''SBS height to normal world units node.
    '''
    def get_default_params(self):
        max_surface_size = 1000.0
        default_surface_size = 300.0 / max_surface_size
        max_height_depth = 100.0
        default_height_depth = 16.0 / max_height_depth
        return [
            ('input', 'img_in', 'input'),
            ('normal_format', 'normal_format', 'keyword', 'gl', lambda p: 'gl' if p else 'dx'),
            ('sampling', 'sampling_mode', 'keyword', 'standard', lambda p: 'sobel' if p else 'standard'),
            ('', 'use_alpha', 'keyword', self.use_alpha),
            ('surface_size', 'surface_size', 'keyword', default_surface_size, intensity_helper(max_surface_size)),
            ('surface_size', 'max_surface_size', 'keyword', max_surface_size, max_intensity_helper(max_surface_size)),
            ('height_depth', 'height_depth', 'keyword', default_height_depth, intensity_helper(max_height_depth)),
            ('height_depth', 'max_height_depth', 'keyword', max_height_depth, max_intensity_helper(max_height_depth)),
            ('', 'trainable', 'keyword', [True, True]),
            ('output', '', 'output'),
        ]

class SBSNormalToHeightNode(SBSNode):
    '''SBS normal to height node.
    '''
    def get_default_params(self):
        max_opacity = 1.0
        default_opacity = 0.36 / max_opacity
        return [
            ('Input', 'img_in', 'input'),
            ('normal_format', 'normal_format', 'keyword', 'dx', lambda p: 'gl' if p else 'dx'),
            ('Relief_Balance', 'relief_balance', 'keyword', [0.5, 0.5, 0.5]),
            ('global_opacity', 'opacity', 'keyword', default_opacity, intensity_helper(max_opacity)),
            ('global_opacity', 'max_opacity', 'keyword', max_opacity, max_intensity_helper(max_opacity)),
            ('', 'trainable', 'keyword', [True, True, True]),
            ('height', '', 'output')
        ]

class SBSCurvatureSmoothNode(SBSNode):
    '''SBS curvature smooth node.
    '''
    def get_default_params(self):
        return [
            ('input', 'img_in', 'input'),
            ('normal_format', 'normal_format', 'keyword', 'dx', lambda p: 'gl' if p else 'dx'),
            ('height', '', 'output')
        ]

class SBSMakeItTilePatchNode(SBSNode):
    '''SBS make it tile patch node.
    '''
    def get_default_params(self):
        max_mask_size = 1.0
        max_mask_precision = 1.0
        max_mask_warping = 100.0
        max_pattern_size = 1000.0
        max_disorder = 1.0
        max_size_variation = 100.0
        return [
            ('Source', 'img_in', 'input'),
            ('Octave', 'octave', 'keyword', 3),
            ('randomseed', 'seed', 'keyword', 0),
            ('', 'use_alpha', 'keyword', self.use_alpha),
            ('Mask_Size', 'mask_size', 'keyword', 1.0, intensity_helper(max_mask_size)),
            ('Mask_Size', 'max_mask_size', 'keyword', max_mask_size, max_intensity_helper(max_mask_size)),
            ('Mask_Precision', 'mask_precision', 'keyword', 0.5, intensity_helper(max_mask_precision)),
            ('Mask_Precision', 'max_mask_precision', 'keyword', max_mask_precision, max_intensity_helper(max_mask_precision)),
            ('Mask_Warping', 'mask_warping', 'keyword', to_zero_one(0.0), intensity_helper_zero_one(max_mask_warping)),
            ('Mask_Warping', 'max_mask_warping', 'keyword', max_mask_warping, max_intensity_helper_zero_one(max_mask_warping)),
            ('Pattern_size_width', 'pattern_width', 'keyword', 0.2, intensity_helper(max_pattern_size)),
            ('Pattern_size_width', 'max_pattern_width', 'keyword', max_pattern_size, max_intensity_helper(max_pattern_size)),
            ('Pattern_size_height', 'pattern_height', 'keyword', 0.2, intensity_helper(max_pattern_size)),
            ('Pattern_size_height', 'max_pattern_height', 'keyword', max_pattern_size, max_intensity_helper(max_pattern_size)),
            ('Disorder', 'disorder', 'keyword', 0.0, intensity_helper(max_disorder)),
            ('Disorder', 'max_disorder', 'keyword', max_disorder, max_intensity_helper(max_disorder)),
            ('Size_Variation', 'size_variation', 'keyword', 0.0, intensity_helper(max_size_variation)),
            ('Size_Variation', 'max_size_variation', 'keyword', max_size_variation, max_intensity_helper(max_size_variation)),
            ('Rotation', 'rotation', 'keyword', to_zero_one(0.0), to_zero_one_normalizer(360.0)),
            ('Rotation_Variation', 'rotation_variation', 'keyword', 0.0, normalizer(360.0)),
            ('Background_Color', 'background_color', 'keyword', [0.0, 0.0, 0.0, 1.0]),
            # Beichen Li: a dummy converter is used since the two SBS parameters do not co-exist
            (['Color_Variation', 'Luminosity_Variation'], 'color_variation', 'keyword', 0.0, PC(lambda x, y, s: x)),
            ('', 'trainable', 'keyword', [True]*11),
            ('Make_It_Tile_Patch', '', 'output')
        ]

class SBSMakeItTilePhotoNode(SBSNode):
    '''SBS make it tile photo node.
    '''
    def get_default_params(self):
        max_mask_warping = 100.0
        max_mask_size = 1.0
        max_mask_precision = 1.0
        return [
            ('Source', 'img_in', 'input'),
            ('Mask_Warping_H', 'mask_warping_x', 'keyword', to_zero_one(0.0), intensity_helper_zero_one(max_mask_warping)),
            ('Mask_Warping_H', 'max_mask_warping_x', 'keyword', max_mask_warping, max_intensity_helper_zero_one(max_mask_warping)),
            ('Mask_Warping_V', 'mask_warping_y', 'keyword', to_zero_one(0.0), intensity_helper_zero_one(max_mask_warping)),
            ('Mask_warping_V', 'max_mask_warping_y', 'keyword', max_mask_warping, max_intensity_helper_zero_one(max_mask_warping)),
            ('Mask_Size_H', 'mask_size_x', 'keyword', 0.1, intensity_helper(max_mask_size)),
            ('Mask_Size_H', 'max_mask_size_x', 'keyword', max_mask_size, max_intensity_helper(max_mask_size)),
            ('Mask_Size_V', 'mask_size_y', 'keyword', 0.1, intensity_helper(max_mask_size)),
            ('Mask_Size_V', 'max_mask_size_y', 'keyword', max_mask_size, max_intensity_helper(max_mask_size)),
            ('Mask_Precision_H', 'mask_precision_x', 'keyword', 0.5, intensity_helper(max_mask_precision)),
            ('Mask_Precision_H', 'max_mask_precision_x', 'keyword', max_mask_precision, max_intensity_helper(max_mask_precision)),
            ('Mask_Precision_V', 'mask_precision_y', 'keyword', 0.5, intensity_helper(max_mask_precision)),
            ('Mask_Precision_V', 'max_mask_precision_y', 'keyword', max_mask_precision, max_intensity_helper(max_mask_precision)),
            ('', 'trainable', 'keyword', [True]*6),
            ('Make_It_Tile_Photo', '', 'output')
        ]

class SBSReplaceColorNode(SBSNode):
    '''SBS replace color node.
    '''
    def get_default_params(self):
        return [
            ('Input', 'img_in', 'input'),
            ('SourceColor', 'source_color', 'keyword', [0.5,0.5,0.5]),
            ('TargetColor', 'target_color', 'keyword', [0.5,0.5,0.5]),
            ('', 'trainable', 'keyword', [True, True]),
            ('ToTargetColor','','output')
        ]

class SBSNormalColorNode(SBSNode):
    '''SBS normal color node.
    '''
    def get_default_params(self):
        return [
            ('Invert_Y', 'normal_format', 'keyword', 'dx', lambda p: 'gl' if p else 'dx'),
            ('', 'num_imgs', 'keyword', 1),
            ('outputsize', 'res_h', 'keyword', 1 << self.res[0], lambda p: 1 << p[0]),
            ('outputsize', 'res_w', 'keyword', 1 << self.res[1], lambda p: 1 << p[1]),
            ('', 'use_alpha', 'keyword', self.use_alpha),
            ('Direction', 'direction', 'keyword', 0.0),
            ('Slope_Angle', 'slope_angle', 'keyword', 0.0),
            ('', 'trainable', 'keyword', [True, True]),
            ('', '', 'output'),
        ]

class SBSVectorMorphNode(SBSNode):
    '''SBS vector morph node.
    '''
    def get_default_params(self):
        max_amount = 1.0
        return [
            ('input', 'img_in', 'input'),
            ('vector_field', 'vector_field', 'keyword'),
            ('amount', 'amount', 'keyword', 1.0, intensity_helper(max_amount)),
            ('amount', 'max_amount', 'keyword', max_amount, max_intensity_helper(max_amount)),
            ('', 'trainable', 'keyword', [True]),
            ('output', '', 'output')
        ]

class SBSVectorWarpNode(SBSNode):
    '''SBS vector warp node.
    '''
    def get_default_params(self):
        max_intensity = 1.0
        return [
            ('input', 'img_in', 'input'),
            ('vector_map', 'vector_map', 'keyword'),
            ('vector_format', 'vector_format', 'keyword', 'dx', lambda p: 'gl' if p else 'dx'),
            ('intensity', 'intensity', 'keyword', 1.0, intensity_helper(max_intensity)),
            ('intensity', 'max_intensity', 'keyword', max_intensity, max_intensity_helper(max_intensity)),
            ('', 'trainable', 'keyword', [True]),
            ('output', '', 'output')
        ]

class SBSContrastLuminosityNode(SBSNode):
    '''SBS contrast luminosity node
    '''
    def get_default_params(self):
        return [
            ('Source', 'img_in', 'input'),
            ('Contrast', 'contrast', 'keyword', to_zero_one(0.0), to_zero_one_helper()),
            ('Luminosity', 'luminosity', 'keyword', to_zero_one(0.0), to_zero_one_helper()),
            ('', 'trainable', 'keyword', [True, True]),
            ('Contrast_Luminosity', '', 'output'),
        ]

class SBSP2SNode(SBSNode):
    '''SBS pre-multiplied to straight node
    '''
    def get_default_params(self):
        return [
            ('input', 'img_in', 'input'),
            ('output', '', 'output')
        ]

class SBSS2PNode(SBSNode):
    '''SBS straight to pre-multiplied node
    '''
    def get_default_params(self):
        return [
            ('input', 'img_in', 'input'),
            ('output', '', 'output')
        ]

class SBSClampNode(SBSNode):
    '''SBS clamp node
    '''
    def get_default_params(self):
        return [
            ('input', 'img_in', 'input'),
            ('apply_to_alpha', 'clamp_alpha', 'keyword', True),
            ('min', 'low', 'keyword', 0.0),
            ('max', 'high', 'keyword', 0.0),
            ('', 'trainable', 'keyword', [True, True]),
            ('output', '', 'output')
        ]

class SBSPowNode(SBSNode):
    '''SBS pow node
    '''
    def get_default_params(self):
        max_exponent = 10.0
        return [
            ('input', 'img_in', 'input'),
            ('exponent', 'exponent', 'keyword', 0.4, intensity_helper(max_exponent)),
            ('exponent', 'max_exponent', 'keyword', max_exponent, max_intensity_helper(max_exponent)),
            ('', 'trainable', 'keyword', [True]),
            ('output', '', 'output')
        ]

class SBSQuantizeNode(SBSNode):
    '''SBS quantize node
    '''
    def update_quantize_number(self, new_val, exist_val, sbs_name):
        channel_dict = {'R': 0, 'G': 1, 'B': 2, 'A': 3}
        if sbs_name == 'Quantize':
            return new_val
        else:
            idx = channel_dict[sbs_name[-1]]
            exist_val[idx] = new_val
            return exist_val

    def get_default_params(self):
        return [
            ('Input', 'img_in', 'input'),
            (['Quantize', 'Quantize_R', 'Quantize_G', 'Quantize_B', 'Quantize_A'],
             'quantize_number', 'keyword', 3 if 'grayscale' in self.name else [4, 4, 4, 4], self.update_quantize_number),
            ('Quantize', '', 'output'),
        ]

class SBSAnisotropicBlurNode(SBSNode):
    '''SBS anisotropic blur node
    '''
    def get_default_params(self):
        max_intensity = 16.0
        return [
            ('Source', 'img_in', 'input'),
            ('Quality', 'high_quality', 'keyword', False, bool),
            ('Intensity', 'intensity', 'keyword', 10.0/16.0, intensity_helper(max_intensity)),
            ('Intensity', 'max_intensity', 'keyword', max_intensity, max_intensity_helper(max_intensity)),
            ('Anisotropy', 'anisotropy', 'keyword', 0.5),
            ('Angle', 'angle', 'keyword', 0.0),
            ('', 'trainable', 'keyword', [True, True, True]),
            ('Anisotropic_Blur', '', 'output'),
        ]

class SBSGlowNode(SBSNode):
    '''SBS glow node
    '''
    def get_default_params(self):
        max_size = 20.0
        return [
            ('Source', 'img_in', 'input'),
            ('Glow_Amount', 'glow_amount', 'keyword', 0.5),
            ('Clear_Amount', 'clear_amount', 'keyword', 0.5),
            ('Glow_Size', 'size', 'keyword', 0.5, intensity_helper(max_size)),
            ('Glow_Size', 'max_size', 'keyword', max_size, max_intensity_helper(max_size)),
            ('Glow_Color', 'color', 'keyword', [1.0, 1.0, 1.0, 1.0]),
            ('', 'trainable', 'keyword', [True, True, True, True]),
            ('Glow', '', 'output'),
        ]

class SBSCar2PolNode(SBSNode):
    '''SBS cartesian to polar node
    '''
    def get_default_params(self):
        return [
            ('input', 'img_in', 'input'),
            ('output', '', 'output'),
        ]

class SBSPol2CarNode(SBSNode):
    '''SBS polar to cartesian node
    '''
    def get_default_params(self):
        return [
            ('input', 'img_in', 'input'),
            ('output', '', 'output'),
        ]

class SBSNormalSobelNode(SBSNode):
    '''SBS normal sobel node
    '''
    def get_default_params(self):
        max_intensity = 3.0
        return [
            ('input', 'img_in', 'input'),
            ('normal_format', 'normal_format', 'keyword', 'gl', lambda p: 'gl' if p else 'dx'),
            ('', 'use_alpha', 'keyword', self.use_alpha),
            ('intensity', 'intensity', 'keyword', to_zero_one(1.0/3.0), intensity_helper_zero_one(max_intensity)),
            ('intensity', 'max_intensity', 'keyword', max_intensity, max_intensity_helper_zero_one(max_intensity)),
            ('', 'trainable', 'keyword', [True]),
            ('output', '', 'output'),
        ]

class SBSNormalVectorRotationNode(SBSNode):
    '''SBS normal vector rotation node
    '''
    def get_default_params(self):
        return [
            ('Normal', 'img_in', 'input'),
            ('rotation_map', 'img_map', 'keyword'),
            ('normal_format', 'normal_format', 'keyword', 'dx', lambda p: 'gl' if p else 'dx'),
            ('rotation_angle', 'rotation', 'keyword', to_zero_one(0.0), intensity_helper_zero_one(1.0)),
            ('rotation_angle', 'rotation_max', 'keyword', 1.0, max_intensity_helper_zero_one(1.0)),
            ('', 'trainable', 'keyword', [True]),
            ('Normal', '', 'output'),
        ]

class SBSNonSquareTransformNode(SBSNode):
    '''SBS non-square transform node
    '''
    def get_default_params(self):
        max_intensity = 1.0
        return [
            ('input', 'img_in', 'input'),
            ('tiling', 'tiling', 'keyword', 3),
            ('tile_mode', 'tile_mode', 'keyword', 'automatic', lambda p: 'manual' if p else 'automatic'),
            ('tile', 'x_tile', 'keyword', 1, lambda p: p[0]),
            ('tile', 'y_tile', 'keyword', 1, lambda p: p[1]),
            ('tile_safe_rotation', 'tile_safe_rotation', 'keyword', True),
            ('offset', 'x_offset', 'keyword', to_zero_one(0.0), intensity_helper_getitem(max_intensity, 0)),
            ('offset', 'x_offset_max', 'keyword', max_intensity, max_intensity_helper_getitem(max_intensity, 0)),
            ('offset', 'y_offset', 'keyword', to_zero_one(0.0), intensity_helper_getitem(max_intensity, 1)),
            ('offset', 'y_offset_max', 'keyword', max_intensity, max_intensity_helper_getitem(max_intensity, 1)),
            ('rotation', 'rotation', 'keyword', to_zero_one(0.0), intensity_helper_zero_one(max_intensity)),
            ('rotation', 'rotation_max', 'keyword', max_intensity, max_intensity_helper_zero_one(max_intensity)),
            ('background_color', 'background_color', 'keyword', [0.0, 0.0, 0.0, 1.0]),
            ('', 'trainable', 'keyword', [True, True, True, True]),
            ('output', '', 'output'),
        ]

class SBSQuadTransformNode(SBSNode):
    '''SBS quad transform node
    '''
    def get_default_params(self):
        return [
            ('input', 'img_in', 'input'),
            ('culling', 'culling', 'keyword', 'f-b', lambda p: ['f', 'b', 'f-b', 'b-f'][p]),
            ('enable_tiling', 'enable_tiling', 'keyword', False),
            ('sampling', 'sampling', 'keyword', 'bilinear', lambda p: 'nearest' if p else 'bilinear'),
            ('p00', 'p00', 'keyword', [0.0, 0.0]),
            ('p01', 'p01', 'keyword', [0.0, 1.0]),
            ('p10', 'p10', 'keyword', [1.0, 0.0]),
            ('p11', 'p11', 'keyword', [1.0, 1.0]),
            ('background_color', 'background_color', 'keyword', [0.0, 0.0, 0.0, 1.0]),
            ('', 'trainable', 'keyword', [True, True, True, True, True]),
            ('output', '', 'output'),
        ]

class SBSChrominanceExtractNode(SBSNode):
    '''SBS chrominance extract node
    '''
    def get_default_params(self):
        return [
            ('Source', 'img_in', 'input'),
            ('Chrominance_Extract', '', 'output'),
        ]

class SBSHistogramShiftNode(SBSNode):
    '''SBS histogram shift node
    '''
    def get_default_params(self):
        return [
            ('input', 'img_in', 'input'),
            ('position', 'position', 'keyword', 0.5),
            ('', 'trainable', 'keyword', [True]),
            ('output', '', 'output'),
        ]

class SBSHeightMapFrequenciesMapperNode(SBSNode):
    '''SBS height map frequencies mapper node
    '''
    def get_default_params(self):
        max_relief = 32.0
        return [
            ('Height', 'img_in', 'input'),
            ('Relief', 'relief', 'keyword', 0.5, intensity_helper(max_relief)),
            ('Relief', 'max_relief', 'keyword', max_relief, max_intensity_helper(max_relief)),
            ('', 'trainable', 'keyword', [True]),
            ('Displacement', '', 'output'),
            ('Relief_Parallax', '', 'output'),
        ]

class SBSLuminanceHighpassNode(SBSNode):
    '''SBS luminance highpass node
    '''
    def get_default_params(self):
        max_radius = 64.0
        return [
            ('Source', 'img_in', 'input'),
            ('Radius', 'radius', 'keyword', 6.0/max_radius, intensity_helper(max_radius)),
            ('Radius', 'max_radius', 'keyword', max_radius, max_intensity_helper(max_radius)),
            ('', 'trainable', 'keyword', [True]),
            ('Luminance_Highpass', '', 'output'),
        ]

class SBSReplaceColorRangeNode(SBSNode):
    '''SBS replace color range node
    '''
    def get_default_params(self):
        return [
            ('Input', 'img_in', 'input'),
            ('SourceColor', 'source_color', 'keyword', [0.501961]*3),
            ('TargetColor', 'target_color', 'keyword', [0.501961]*3),
            ('SourceRange', 'source_range', 'keyword', 0.5),
            ('threshold', 'threshold', 'keyword', 1.0),
            ('', 'trainable', 'keyword', [True,True,True,True]),
            ('ToTargetColorRange', '', 'output'),
        ]

class SBSDissolveNode(SBSNode):
    '''SBS dissolve node
    '''
    def get_default_params(self):
        return [
            ('foreground', 'img_fg', 'keyword'),
            ('background', 'img_bg', 'keyword'),
            ('mask', 'mask', 'keyword'),
            ('opacity', 'opacity', 'keyword', 1.0),
            ('Alpha_Blending', 'alpha_blending', 'keyword', True),
            ('', 'trainable', 'keyword', [True]),
            ('output', '', 'output'),
        ]

class SBSColorBlendNode(SBSNode):
    '''SBS color blend node
    '''
    def get_default_params(self):
        return [
            ('Foreground', 'img_fg', 'keyword'),
            ('Background', 'img_bg', 'keyword'),
            ('Mask', 'mask', 'keyword'),
            ('Opacity', 'opacity', 'keyword', 1.0),
            ('Alpha_Blending', 'alpha_blending', 'keyword', True),
            ('', 'trainable', 'keyword', [True]),
            ('Color', '', 'output'),
        ]

class SBSColorBurnNode(SBSNode):
    '''SBS color burn node
    '''
    def get_default_params(self):
        return [
            ('Foreground', 'img_fg', 'keyword'),
            ('Background', 'img_bg', 'keyword'),
            ('Mask', 'mask', 'keyword'),
            ('Opacity', 'opacity', 'keyword', 1.0),
            ('Alpha_Blending', 'alpha_blending', 'keyword', True),
            ('', 'trainable', 'keyword', [True]),
            ('Color_Burn', '', 'output'),
        ]

class SBSColorDodgeNode(SBSNode):
    '''SBS color dodge node
    '''
    def get_default_params(self):
        return [
            ('Foreground', 'img_fg', 'keyword'),
            ('Background', 'img_bg', 'keyword'),
            ('Mask', 'mask', 'keyword'),
            ('Opacity', 'opacity', 'keyword', 1.0),
            ('Alpha_Blending', 'alpha_blending', 'keyword', True),
            ('', 'trainable', 'keyword', [True]),
            ('Color_Dodge', '', 'output'),
        ]

class SBSDifferenceNode(SBSNode):
    '''SBS difference node
    '''
    def get_default_params(self):
        return [
            ('Foreground', 'img_fg', 'keyword'),
            ('Background', 'img_bg', 'keyword'),
            ('Mask', 'mask', 'keyword'),
            ('Opacity', 'opacity', 'keyword', 1.0),
            ('Alpha_Blending', 'alpha_blending', 'keyword', True),
            ('', 'trainable', 'keyword', [True]),
            ('Difference', '', 'output'),
        ]

class SBSLinearBurnNode(SBSNode):
    '''SBS linear burn node
    '''
    def get_default_params(self):
        return [
            ('Foreground', 'img_fg', 'keyword'),
            ('Background', 'img_bg', 'keyword'),
            ('Mask', 'mask', 'keyword'),
            ('Opacity', 'opacity', 'keyword', 1.0),
            ('Alpha_Blending', 'alpha_blending', 'keyword', True),
            ('', 'trainable', 'keyword', [True]),
            ('Linear_Burn', '', 'output'),
        ]

class SBSLuminosityNode(SBSNode):
    '''SBS luminosity node
    '''
    def get_default_params(self):
        return [
            ('Foreground', 'img_fg', 'keyword'),
            ('Background', 'img_bg', 'keyword'),
            ('Mask', 'mask', 'keyword'),
            ('Opacity', 'opacity', 'keyword', 1.0),
            ('Alpha_Blending', 'alpha_blending', 'keyword', True),
            ('', 'trainable', 'keyword', [True]),
            ('Exclusion', '', 'output'),  # not our typo
        ]

# Done
class SBSMultiDirWarpNode(SBSNode):
    '''SBS multi directional warp node
    '''
    blending_map = {
        0: "copy",
        5: "max",
        6: "min",
        10: "chain"
    }
    def get_default_params(self):
        mode_func = lambda p: self.blending_map[p]
        max_intensity=20.0
        return [
            ('input', 'img_in', 'input'),
            ('intensity_input', 'intensity_mask', 'input'),
            ('mode', 'mode', 'keyword', 'average', mode_func),
            ('directions', 'directions', 'keyword', 4),
            ('intensity', 'intensity', 'keyword', 10.0/max_intensity, intensity_helper(max_intensity)),
            ('intensity', 'max_intensity', 'keyword', max_intensity, max_intensity_helper(max_intensity)),
            ('warp_angle', 'angle', 'keyword', 0.0),
            ('', 'trainable', 'keyword', [True, True]),
            ('output', '', 'output'),
        ]

# Done
class SBSShapeDropShadowNode(SBSNode):
    '''SBS shape drop shadow node
    '''
    def get_default_params(self):
        max_dist = 0.5
        max_size = 1.0
        return [
            ('input', 'img_in', 'input'),
            ('input_is_premult', 'input_is_pre_multiplied', 'keyword', True),
            ('pre_multiply_output', 'pre_multiplied_output', 'keyword', False),
            ('angle', 'angle', 'keyword', 0.25),
            ('distance', 'dist', 'keyword', to_zero_one(0.02/0.5), intensity_helper(max_dist)),
            ('distance', 'max_dist', 'keyword', max_dist, max_intensity_helper(max_dist)),
            ('size', 'size', 'keyword', 0.15/max_size, intensity_helper(max_size)),
            ('size', 'max_size', 'keyword', max_size, max_intensity_helper(max_size)),
            ('spread', 'spread', 'keyword', 0.0),
            ('opacity', 'opacity', 'keyword', 0.5),
            ('mask_color', 'mask_color', 'keyword', [1.0,1.0,1.0]),
            ('shadow_color', 'shadow_color', 'keyword', [0.0,0.0,0.0]),
            ('', 'trainable', 'keyword', [True, True, True, True, True, True, True]),
            ('output', '', 'output'),
            ('mask', '', 'output'),
        ]

# Done
class SBSShapeGlowNode(SBSNode):
    '''SBS shape glow node
    '''
    def get_default_params(self):
        mode_func = lambda p: self.mode_list[p]
        return [
            ('input', 'img_in', 'input'),
            ('input_is_premult', 'input_is_pre_multiplied', 'keyword', True),
            ('pre_multiply_output', 'pre_multiplied_output', 'keyword', False),
            ('mode', 'mode', 'keyword', 'soft', lambda p: 'precise' if p else 'soft'),
            ('width', 'width', 'keyword', to_zero_one(0.25)),
            ('spread', 'spread', 'keyword', 0.0),
            ('opacity', 'opacity', 'keyword', 1.0),
            ('mask_color', 'mask_color', 'keyword', [1.0,1.0,1.0]),
            ('glow_color', 'glow_color', 'keyword', [1.0,1.0,1.0]),
            ('', 'trainable', 'keyword', [True, True, True, True, True]),
            ('output', '', 'output'),
            ('mask', '', 'output'),
        ]

# Done
class SBSSwirlNode(SBSNode):
    '''SBS swirl node
    '''
    def get_default_params(self):
        max_intensity_x1_y2 = 2.0
        max_intensity_rest = 1.0
        max_amount=16.0
        return [
            ('input', 'img_in', 'input'),
            ('tiling', 'tile_mode', 'keyword', 3),
            ('amount', 'amount', 'keyword', to_zero_one(0.5), intensity_helper_zero_one(max_amount)),
            ('amount', 'max_amount', 'keyword', max_amount, max_intensity_helper_zero_one(max_amount)),
            ('matrix22', 'x1', 'keyword', to_zero_one(1.0), intensity_helper_getitem(max_intensity_x1_y2, 0)),
            ('matrix22', 'x1_max', 'keyword', max_intensity_x1_y2, max_intensity_helper_getitem(max_intensity_x1_y2, 0)),
            ('matrix22', 'x2', 'keyword', to_zero_one(0.0), intensity_helper_getitem(max_intensity_rest, 1)),
            ('matrix22', 'x2_max', 'keyword', max_intensity_rest, max_intensity_helper_getitem(max_intensity_rest, 1)),
            ('offset', 'x_offset', 'keyword', to_zero_one(0.0), intensity_helper_getitem(max_intensity_rest, 0)),
            ('offset', 'x_offset_max', 'keyword', max_intensity_rest, max_intensity_helper_getitem(max_intensity_rest, 0)),
            ('matrix22', 'y1', 'keyword', to_zero_one(0.0), intensity_helper_getitem(max_intensity_rest, 2)),
            ('matrix22', 'y1_max', 'keyword', max_intensity_rest, max_intensity_helper_getitem(max_intensity_rest, 2)),
            ('matrix22', 'y2', 'keyword', to_zero_one(1.0), intensity_helper_getitem(max_intensity_x1_y2, 3)),
            ('matrix22', 'y2_max', 'keyword', max_intensity_x1_y2, max_intensity_helper_getitem(max_intensity_x1_y2, 3)),
            ('offset', 'y_offset', 'keyword', to_zero_one(0.0), intensity_helper_getitem(max_intensity_rest, 1)),
            ('offset', 'y_offset_max', 'keyword', max_intensity_rest, max_intensity_helper_getitem(max_intensity_rest, 1)),
            ('', 'trainable', 'keyword', [True, True, True, True, True, True, True]),
            ('output', '', 'output'),
        ]

# Done
class SBSCurvatureSobelNode(SBSNode):
    '''SBS curvature sobel node
    '''
    def get_default_params(self):
        max_intensity = 1.0
        return [
            ('normal', 'img_in', 'input'),
            ('input_normal_type', 'normal_format', 'keyword', 'dx', lambda p: 'gl' if p else 'dx'),
            ('intensity', 'intensity', 'keyword', to_zero_one(0.5), intensity_helper_zero_one(max_intensity)),
            ('intensity', 'max_intensity', 'keyword', max_intensity, max_intensity_helper_zero_one(max_intensity)),
            ('', 'trainable', 'keyword', [True]),
            ('output', '', 'output'),
        ]

# Not done!!
class SBSEmbossWithGlossNode(SBSNode):
    '''SBS emboss with gloss node
    '''
    def get_default_params(self):
        max_intensity = 10.0
        max_gloss = 1.0
        return [
            ('Source', 'img_in', 'input'),
            ('Height', 'height', 'input'),
            ('Intensity', 'intensity', 'keyword', 0.5, intensity_helper(max_intensity)),
            ('Intensity', 'max_intensity', 'keyword', max_intensity, max_intensity_helper(max_intensity)),
            ('LightAngle', 'light_angle', 'keyword', 0.0),
            ('Gloss', 'gloss', 'keyword', to_zero_one(0.25), intensity_helper_zero_one(max_gloss)),
            ('Gloss', 'max_gloss', 'keyword', max_gloss, max_intensity_helper_zero_one(max_gloss)),
            ('Highlightcolor', 'highlight_color', 'keyword', [1.0,1.0,1.0]),
            ('ShadowColor', 'shadow_color', 'keyword', [0.0,0.0,0.0]),
            ('', 'trainable', 'keyword', [True, True, True, True, True]),
            ('Emboss_With_Gloss', '', 'output'),
        ]

# Done
class SBSFacingNormalNode(SBSNode):
    '''SBS facing normal node
    '''
    def get_default_params(self):
        return [
            ('Normal', 'img_in', 'input'),
            ('Facing_Normal', '', 'output'),
        ]

# Done
class SBSHeightNormalBlendNode(SBSNode):
    '''SBS height normal blend node
    '''
    def get_default_params(self):
        max_normal_intensity=1.0
        return [
            ('Height', 'img_height', 'input'),
            ('Normal', 'img_in', 'input'),
            ('Normal_Format', 'normal_format', 'keyword', 'dx', lambda p: 'gl' if p else 'dx'),
            ('Normal_Intensity', 'normal_intensity', 'keyword', to_zero_one(0.0), intensity_helper_zero_one(max_normal_intensity)),
            ('Normal_Intensity', 'max_normal_intensity', 'keyword', max_normal_intensity, max_intensity_helper_zero_one(max_normal_intensity)),
            ('', 'trainable', 'keyword', [True]),
            ('Normal', '', 'output')
        ]

# Done
class SBSNormalInvertNode(SBSNode):
    '''SBS normal invert node
    '''
    def get_default_params(self):
        return [
            ('Normal', 'img_in', 'input'),
            ('invert_red', 'invert_red', 'keyword', False),
            ('invert_green', 'invert_green', 'keyword', True),
            ('invert_blue', 'invert_blue', 'keyword', False),
            ('invert_alpha', 'invert_alpha', 'keyword', False),
            ('normal', '', 'output')
        ]

# Done
class SBSSkewNode(SBSNode):
    '''SBS skew node
    '''
    align_mode=['center', 'top_left', 'bottom_right']
    def get_default_params(self):
        align_func = lambda p: self.align_mode[p]
        max_amount = 1.0
        return [
            ('Source', 'img_in', 'input'),
            ('Axis', 'axis', 'keyword', 'horizontal', lambda p: 'vertical' if p else 'horizontal'),
            ('Align', 'align', 'keyword', 'top_left', align_func),
            ('Amount', 'amount', 'keyword', to_zero_one(0.0), intensity_helper_zero_one(max_amount)),
            ('Amount', 'max_amount', 'keyword', max_amount, max_intensity_helper_zero_one(max_amount)),
            ('', 'trainable', 'keyword', [True]),
            ('Skew', '', 'output')
        ]

# Done
class SBSTrapezoidTransformNode(SBSNode):
    '''SBS trapezoid transform node
    '''
    def get_default_params(self):
        max_top_stretch = 1.0
        max_bottom_stretch = 1.0
        return [
            ('input', 'img_in', 'input'),
            ('sampling', 'sampling', 'keyword', 'bilinear', lambda p: 'nearest' if p else 'bilinear'),
            ('tiling', 'tile_mode', 'keyword', 3),
            ('top_stretch', 'top_stretch', 'keyword', to_zero_one(0.0), intensity_helper_zero_one(max_top_stretch)),
            ('top_stretch', 'max_top_stretch', 'keyword', max_top_stretch, max_intensity_helper_zero_one(max_top_stretch)),
            ('bottom_stretch', 'bottom_stretch', 'keyword', to_zero_one(0.0), intensity_helper_zero_one(max_bottom_stretch)),
            ('bottom_stretch', 'max_bottom_stretch', 'keyword', max_bottom_stretch, max_intensity_helper_zero_one(max_bottom_stretch)),
            ('background_color', 'bg_color', 'keyword', [0.0,0.0,0.0,1.0]),
            ('', 'trainable', 'keyword', [True,True,True]),
            ('output', '', 'output')
        ]

# Done
class SBSColorToMaskNode(SBSNode):
    '''SBS color to mask node
    '''
    keying_mode=['rgb', 'chrominance', 'luminance']
    def get_default_params(self):
        keying_func = lambda p: self.keying_mode[p]
        return [
            ('Input', 'img_in', 'input'),
            ('flatten_alpha', 'flatten_alpha', 'keyword', False),
            ('keying_type', 'keying_type', 'keyword', 'rgb', keying_func),
            ('color', 'rgb', 'keyword', [0.0,1.0,0.0]),
            ('mask_range', 'mask_range','keyword', 0.0),
            ('mask_softness', 'mask_softness','keyword', 0.0),
            ('', 'trainable', 'keyword', [True,True,True]),
            ('output', '', 'output')
        ]

# Done
class SBSC2GAdvancedNode(SBSNode):
    '''SBS grayscale conversion advanced node
    '''
    grayscale_mode=['desaturation', 'luma', 'average', 'max', 'min']
    def get_default_params(self):
        grayscale_func = lambda p: self.grayscale_mode[p]
        return [
            ('input', 'img_in', 'input'),
            ('grayscale_type', 'grayscale_type', 'keyword', 'desaturation', grayscale_func),
            ('output', '', 'output')
        ]
