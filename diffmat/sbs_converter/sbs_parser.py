'''
XML Parser for sbs files, providing empirical analysis on graph structures.
'''
import os
import sys
import copy
import random
import platform
import argparse
from pathlib import Path
from collections import deque
import xml.etree.ElementTree as ET
from ordered_set import OrderedSet

from diffmat.sbs_converter.sbs_nodes import *
from diffmat.sbs_converter.sbs_functions import SBSFunctionGraphParser



# Format: node name in SBS -> (class name in sbs_nodes.py, function name)
type_dict = {
    # Atomic nodes
    'blend':                    ('Blend', 'blend'),
    'blur':                     ('Blur', 'blur'),
    'shuffle':                  ('ChannelShuffle', 'channel_shuffle'),
    'curve':                    ('Curve', 'curve'),
    'dirmotionblur':            ('DBlur', 'd_blur'),
    'directionalwarp':          ('DWarp', 'd_warp'),
    'distance':                 ('Distance', 'distance'),
    'emboss':                   ('Emboss', 'emboss'),
    'gradient':                 ('GradientMap', 'gradient_map'),
    'dyngradient':              ('GradientMapDyn', 'gradient_map_dyn'),
    'grayscaleconversion':      ('C2G', 'c2g'),
    'hsl':                      ('HSL', 'hsl'),
    'levels':                   ('Levels', 'levels'),
    'normal':                   ('Normal', 'normal'),
    'sharpen':                  ('Sharpen', 'sharpen'),
    'transformation':           ('Transform2d', 'transform_2d'),
    'uniform':                  ('UniformColor', 'uniform_color'),
    'warp':                     ('Warp', 'warp'),

    # Non-atomic nodes (add as needed)
    "convert_to_srgb":            ('L2SRGB', 'linear_to_srgb'),
    "convert_to_linear":          ('SRGB2L', 'srgb_to_linear'),
    'curvature':                  ('Curvature', 'curvature'),
    'invert_grayscale':           ('InvertGrayscale', 'invert'),
    'invert':                     ('InvertColor', 'invert'),
    'histogram_scan':             ('HistogramScan', 'histogram_scan'),
    'histogram_range':            ('HistogramRange', 'histogram_range'),
    'histogram_select':           ('HistogramSelect', 'histogram_select'),
    'edge_detect':                ('EdgeDetect', 'edge_detect'),
    'safe_transform_grayscale':   ('SafeTransform', 'safe_transform'),
    'safe_transform':             ('SafeTransform', 'safe_transform'),
    'blur_hq_grayscale':          ('BlurHQ', 'blur_hq'),
    'blur_hq':                    ('BlurHQ', 'blur_hq'),
    'non_uniform_blur_grayscale': ('NonUniformBlur', 'non_uniform_blur'),
    'non_uniform_blur':           ('NonUniformBlur', 'non_uniform_blur'),
    'bevel':                      ('Bevel', 'bevel'),
    'slope_blur_grayscale_2':     ('SlopeBlur', 'slope_blur'),
    'slope_blur_grayscale':       ('SlopeBlur', 'slope_blur'),  # The difference from '_2' is unknown
    'slope_blur':                 ('SlopeBlur', 'slope_blur'),
    'mosaic_grayscale':           ('Mosaic', 'mosaic'),
    'mosaic':                     ('Mosaic', 'mosaic'),
    'auto_levels':                ('AutoLevels', 'auto_levels'),
    'ambient_occlusion_2':        ('AmbientOcclusion', 'ambient_occlusion'),
    'hbao':                       ('HBAO', 'hbao'),
    'highpass_grayscale':         ('Highpass', 'highpass'),
    'highpass':                   ('Highpass', 'highpass'),
    'normal_normalise':           ('NormalNormalize', 'normal_normalize'),
    'channel_mixer':              ('ChannelMixer', 'channel_mixer'),
    'normal_combine':             ('NormalCombine', 'normal_combine'),
    'multi_switch_grayscale':     ('MultiSwitch', 'multi_switch'),
    'multi_switch':               ('MultiSwitch', 'multi_switch'),
    'rgba_split':                 ('RGBASplit', 'rgba_split'),
    'rgba_merge':                 ('RGBAMerge', 'rgba_merge'),
    'basecolor_metallic_roughness_to_diffuse_specular_glossiness': ('PbrConverter', 'pbr_converter'),
    'height_to_normal_world_units_2': ('HeightToNormal', 'height_to_normal_world_units'),
    'normal_to_height':           ('NormalToHeight', 'normal_to_height'),
    'curvature_smooth':           ('CurvatureSmooth', 'curvature_smooth'),
    'make_it_tile_patch_grayscale': ('MakeItTilePatch', 'make_it_tile_patch'),
    'make_it_tile_patch':         ('MakeItTilePatch', 'make_it_tile_patch'),
    'make_it_tile_photo_grayscale': ('MakeItTilePhoto', 'make_it_tile_photo'),
    'make_it_tile_photo':         ('MakeItTilePhoto', 'make_it_tile_photo'),
    'rgb-a_split':                ('AlphaSplit', 'alpha_split'),
    'rgb-a_merge':                ('AlphaMerge', 'alpha_merge'),
    'switch_grayscale':           ('Switch', 'switch'),
    'switch':                     ('Switch', 'switch'),
    'normal_blend':               ('NormalBlend', 'normal_blend'),
    'mirror_grayscale':           ('Mirror', 'mirror'),
    'mirror':                     ('Mirror', 'mirror'),
    'vector_morph_grayscale':     ('VectorMorph', 'vector_morph'),
    'vector_morph':               ('VectorMorph', 'vector_morph'),
    'vector_warp_grayscale':      ('VectorWarp', 'vector_warp'),
    'vector_warp':                ('VectorWarp', 'vector_warp'),
    'passthrough':                ('Passthrough', 'passthrough'),
    'replace_color':              ('ReplaceColor', 'replace_color'),
    'normal_color':               ('NormalColor', 'normal_color'),
    'contrast_luminosity_grayscale': ('ContrastLuminosity', 'contrast_luminosity'),
    'contrast_luminosity':        ('ContrastLuminosity', 'contrast_luminosity'),
    'premult_to_straight':        ('P2S', 'p2s'),
    'straight_to_premult':        ('S2P', 's2p'),
    'clamp':                      ('Clamp', 'clamp'),
    'clamp_grayscale':            ('Clamp', 'clamp'),
    'pow':                        ('Pow', 'pow'),
    'pow_grayscale':              ('Pow', 'pow'),
    'quantize_grayscale':         ('Quantize', 'quantize'),
    'quantize':                   ('Quantize', 'quantize'),
    'anisotropic_blur_grayscale': ('AnisotropicBlur', 'anisotropic_blur'),
    'anisotropic_blur':           ('AnisotropicBlur', 'anisotropic_blur'),
    'glow_grayscale':             ('Glow', 'glow'),
    'glow':                       ('Glow', 'glow'),
    'cartesian_to_polar_grayscale': ('Car2Pol', 'car2pol'),
    'cartesian_to_polar':         ('Car2Pol', 'car2pol'),
    'polar_to_cartesian_grayscale': ('Pol2Car', 'pol2car'),
    'polar_to_cartesian':         ('Pol2Car', 'pol2car'),
    'normal_sobel_2':             ('NormalSobel', 'normal_sobel'),
    'normal_vector_rotation':     ('NormalVectorRotation', 'normal_vector_rotation'),
    'non_square_transform_greyscale': ('NonSquareTransform', 'non_square_transform'),
    'non_square_transform':       ('NonSquareTransform', 'non_square_transform'),
    'quad_transform_grayscale':   ('QuadTransform', 'quad_transform'),
    'quad_transform':             ('QuadTransform', 'quad_transform'),
    'chrominance_extract':        ('ChrominanceExtract', 'chrominance_extract'),
    'histogram_shift':            ('HistogramShift', 'histogram_shift'),
    'height_map_frequencies_mapper': ('HeightMapFrequenciesMapper', 'height_map_frequencies_mapper'),
    'luminance_highpass':         ('LuminanceHighpass', 'luminance_highpass'),
    'replace_color_range':        ('ReplaceColorRange', 'replace_color_range'),
    'dissolve_2':                 ('Dissolve', 'dissolve'),
    'color':                      ('ColorBlend', 'color_blend'),
    'color_burn':                 ('ColorBurn', 'color_burn'),
    'color_dodge':                ('ColorDodge', 'color_dodge'),
    'difference':                 ('Difference', 'difference'),
    'linear_burn':                ('LinearBurn', 'linear_burn'),
    'luminosity':                 ('Luminosity', 'luminosity'),
    'multi_directional_warp_grayscale': ('MultiDirWarp', 'multi_dir_warp'),
    'multi_directional_warp_color': ('MultiDirWarp', 'multi_dir_warp'),
    'shape_drop_shadow_color':    ('ShapeDropShadow', 'shape_drop_shadow'),
    'shape_drop_shadow_grayscale':('ShapeDropShadow', 'shape_drop_shadow'),
    'shape_glow_color':           ('ShapeGlow', 'shape_glow'),
    'shape_glow_grayscale':       ('ShapeGlow', 'shape_glow'),
    'swirl':                      ('Swirl', 'swirl'),
    'swirl_grayscale':            ('Swirl', 'swirl'),
    'curvature_sobel':            ('CurvatureSobel', 'curvature_sobel'),
    'emboss_with_gloss':          ('EmbossWithGloss', 'emboss_with_gloss'),
    'height_normal_blend':        ('HeightNormalBlend', 'height_normal_blend'),
    'skew':                       ('Skew', 'skew'),
    'skew_grayscale':             ('Skew', 'skew'),
    'trapezoid_transform':        ('TrapezoidTransform', 'trapezoid_transform'),
    'trapezoid_transform_grayscale': ('TrapezoidTransform', 'trapezoid_transform'),
    'color_to_mask':              ('ColotToMask', 'color_to_mask'),
    'facing_normal':              ('FacingNormal', 'facing_normal'),
    'normal_invert':              ('NormalInvert', 'normal_invert'),
    'grayscale_conversion_advanced': ('C2GAdvanced', 'c2g_advanced'),
}


class SBSGraphConverter:
    """An automatic converter that translates SBS files into PyTorch reimplementations.
    """
    def __init__(self, file_name, toolkit_path=None, output_path=None, output_graph_name=None,output_res=9, use_alpha=False, save_noise=False, 
                 noise_format='exr', noise_count=5, lambertian=False, device_count=0, scale=False, max_scale=4.0, correct_normal=False):
        """
        Args:
            file_name (str): input file name
            output_res (int, optional): resolution of the output texture (after log2). Defaults to 9.
            use_alpha (bool, optional): switch for the alpha channel output. Defaults to False.
            save_noise (bool, optional): switch for saving generator noises as images. Defaults to False.
            noise_format (str, optional): format of saved generator noises. Defaults to 'exr'.
            noise_count (int, optional): groups of saved generator noises. Defaults to 5.
            lambertian (bool, optional): switch for lambertian material. Defaults to False.
            device_count (int, optional): GPU device number to use in the output files. Defaults to 0.
            scale (bool, optional): switch that allows additional scaling of the output texture. Defaults to False.
            max_scale (float, optional): maximum scale multiplier. Defaults to 4.0.
            correct_normal (bool, optional):
                switch for eliminating the statistical bias in the output normal map that results in tilted specular rendering.
                Only necessary in some rare cases. Defaults to False.
        """
        # Input file name
        self.file_name = file_name

        # Toolkit path
        if toolkit_path == None:
            # set toolkit path as home directory for Linux and Mac
            if platform.system() == 'Linux' or platform.system() == 'Darwin':
                self.toolkit_path = os.environ['HOME']
            # set toolkit path as home directory for Windows
            elif platform.system() == 'Windows':
                self.toolkit_path = os.path.join(os.path.join(os.environ['USERPROFILE']), 'Desktop')
        else:
            self.toolkit_path = toolkit_path

        # Output path
        self.output_path = output_path if output_path is not None else '.'

        # Input only sbs graph namee
        self.input_output_file_name = os.path.splitext(file_name)[0] + '_input.sbs'

        # Output graph resolution
        self.output_res = output_res

        # Basic information: graph name and resolution
        self.name = file_name[file_name.rfind('/') + 1: file_name.rfind('.')]
        
        # Output graph name
        self.output_graph_name = output_graph_name if output_path is not None else self.name

        self.res = [8, 8]   # Default: [256, 256]

        # The dictionary of SBS nodes in the graph (uid -> SBSNode)
        self.nodes = {}

        # List of inputs and outputs
        self.inputs = []
        self.outputs = []
        self.input_dict = {}

        # Node counters for naming purposes
        self.node_count = {}

        # Build the type dictionary (type -> (class, func))
        self.init_type_dict()

        # Alpha channel switch (default: off)
        self.use_alpha = use_alpha

        # Lambertian material switch (default: off)
        self.lambertian = lambertian
        
        # device count
        self.device_count = device_count

        # scale offset
        self.scale = scale
        self.max_scale = max_scale

        # Automatic normal correction
        self.correct_normal = correct_normal

        # Parse the input file
        self.parse(file_name)

        # Generate input noise images for training and evaluation
        if save_noise:
            self.gen_noise(noise_count, noise_format)
        self.noise_count = noise_count
        self.noise_format = noise_format

        # Data flow analysis
        self.analyze()

    def get_input_param(self, key):
        """Return the input parameter specified by key (name or uid).
        """
        if isinstance(key, str):
            for param in self.inputs:
                if param.name == key:
                    return param
        elif isinstance(key, int):
            for param in self.inputs:
                if param.uid == key:
                    return param
        return None

    def get_output_param(self, key):
        """Return the output specified by uid.
        """
        for output in self.outputs:
            if output.uid == key:
                return output
        return None

    def gen_node_name(self, node_id):
        """Generate a new variable name for a certain node specified by its type.
        """
        if node_id not in self.node_count:
            self.node_count[node_id] = 1
            return f'{node_id}'

        count = self.node_count[node_id]
        self.node_count[node_id] = count + 1
        return f'{node_id}_{count}'

    def init_type_dict(self):
        """Initialize the type dictionary for classes and functions.
        """
        self.type_dict = type_dict
        # Save a copy of global classes for lookup
        self.class_dict = globals()

    def lookup_type(self, node_type):
        """Return the node information specified by its type.
        """
        if node_type not in self.type_dict:
            print(f'Warning: node \'{node_type}\' is not supported and substituted by passthrough.')
            node_type = 'passthrough'
        module_name = 'F'
        node_class, node_func = self.type_dict[node_type]
        return self.class_dict[f'SBS{node_class}Node'], node_class, f'{module_name}.{node_func}'

    def resolve_param_val(self, param_value_node, round_float=6):
        """Extract the default value of a parameter input according to its type ID.

        Args:
            param_value_node (ET.Element): XML element of the parameter value
            round_float (int, optional): round any float-valued parameter to certain precision. Defaults to 6.

        Raises:
            TypeError: Unknown parameter type

        Returns:
            Any: parameter value (could be an function graph object if the parameter is dynamic)
        """
        param_value_ = param_value_node
        param_tag = param_value_.tag
        if param_tag in ['constantValueInt32', 'constantValueInt1']:
            param_val = int(param_value_.get('v'))
        elif param_tag in ['constantValueInt2', 'constantValueInt3', 'constantValueInt4']:
            param_val = [int(i) for i in param_value_.get('v').strip().split()]
        elif param_tag == 'constantValueFloat1':
            param_val = round(float(param_value_.get('v')), round_float)
        elif param_tag in ['constantValueFloat2', 'constantValueFloat3', 'constantValueFloat4']:
            param_val = [round(float(i), round_float) for i in param_value_.get('v').strip().split()]
        elif param_tag == 'constantValueBool':
            param_val = bool(int(param_value_.get('v')))
        elif param_tag == 'constantValueString':
            param_val = param_value_.get('v')
        elif param_tag == 'dynamicValue':
            param_val = SBSFunctionGraphParser(param_value_, self.input_dict)
        else:
            raise TypeError('Unknown parameter type')
        return param_val

    def resolve_single_param(self, node_param, dynamic_switch=False):
        """Analyze a single parameter in SBS.

        Args:
            node_param (ET.Element): XML element of the node parameter
            dynamic_switch (bool, optional): switch for processing dynamic parameters. Defaults to False.

        Returns:
            Str: parameter name
            Int or None: relativity of the parameter value
            Any: parameter value
        """
        param_ = node_param
        param_name = param_.find('name').get('v')
        param_rel_to_ = param_.find('relativeTo')
        param_rel_to = int(param_rel_to_.get('v')) if param_rel_to_ is not None else None
        param_val_ = param_.find('paramValue')[0]
        if (param_val_.tag == 'dynamicValue') == dynamic_switch:
            param_val = self.resolve_param_val(param_val_)
        else:
            param_val = None
        return param_name, param_rel_to, param_val

    def resolve_node_params(self, node_imp, node_obj):
        """Analyze manually specified parameters and update their values in SBS nodes accordingly.

        Args:
            node_imp (ET.Element): XML element of the implementation of a node
            node_obj (derived classes of SBSNode): node instance
        """
        def update_node_param(node_obj, param_name, param_rel_to, param_val):
            """A smaller wrapper function which deals with some special cases (like output size).
            """
            if param_val is None:
                return

            # Special case for output size
            if param_name == 'outputsize':
                return
            #     if param_rel_to == 2 and param_val != [0, 0]:
            #         raise NotImplementedError('Output size relative to input is not supported')
            #     if param_rel_to == 1:
            #         param_val = [self.res[0] + param_val[0], self.res[1] + param_val[1]]

            # Special case for dynamic parameters
            # Assign a global variable name
            if isinstance(param_val, SBSFunctionGraphParser):
                param_val.global_name = '_'.join(['sbs', node_obj.name, param_name])

            # Update parameter
            node_obj.update_param(param_name, param_val)

        # Detect and update non-array parameters
        #   - First pass: identify regular parameters
        #   - Second pass: process dynamic parameters (which might depend on the first pass)
        params_ = node_imp.find('parameters')
        if params_ is not None:
            for dynamic_switch in (False, True):
                for param_ in params_.iter('parameter'):
                    update_node_param(node_obj, *self.resolve_single_param(param_, dynamic_switch))

        # Analyze parameter arrays
        param_arrays_ = node_imp.find('paramsArrays')
        if param_arrays_ is not None:
            for param_array_ in param_arrays_.iter('paramsArray'):
                param_array_name = param_array_.find('name').get('v')
                param_array = []

                # Extract parameter cells
                for param_array_cell_ in param_array_.iter('paramsArrayCell'):
                    param_cell = {}
                    for param_ in param_array_cell_.iter('parameter'):
                        param_name, _, param_val = self.resolve_single_param(param_)
                        param_cell[param_name] = param_val
                    param_array.append(param_cell)

                if len(param_array):
                    node_obj.update_param(param_array_name, param_array)

    def parse(self, file_name):
        """Parse a *.sbs file into a material graph and generate auxiliary files for saving generator noises.

        Args:
            file_name (str): path to the source SBS document

        Raises:
            NotImplementedError: when an unsupported node type is found
        """
        # Build XML tree
        tree = ET.parse(file_name)
        root = tree.getroot()

        # Copy and preprocess the current tree for creating the input to output graph
        input_output_tree = copy.deepcopy(tree)
        input_output_root = input_output_tree.getroot()
        # Clean up
        input_output_root.find('dependencies').clear()
        input_output_root.find('content/graph/graphOutputs').clear()
        input_output_root.find('content/graph/compNodes').clear()
        input_output_root.find('content/graph/root/rootOutputs').clear()
        # Set output resolution
        for param_ in input_output_root.find('content/graph/baseParameters').iter('parameter'):
            if param_.find('name').get('v') == 'outputsize':
                param_.find('relativeTo').set('v', '0')
                param_.find('paramValue/constantValueInt2').set('v', f'{self.output_res} {self.output_res}')

        # Retrieve graph name
        name = root.find('content/graph').find('identifier').get('v')
        if name != self.name:
            root.find('content/graph').find('identifier').set('v', self.name)
            tree.write(file_name)

        # Retrieve global resolution
        if self.output_res >= 1:
            self.res = [self.output_res, self.output_res]
        # else:
        #     for param_ in root.find('content/graph/baseParameters').iter('parameter'):
        #         if param_.find('name').get('v') == 'outputsize':
        #             relative_to = int(param_.find('relativeTo').get('v'))
        #             res_str = param_.find('paramValue/constantValueInt2').get('v')
        #             res = [int(i) for i in res_str.strip().split()]
        #             self.res = [self.res[0] + res[0], self.res[1] + res[1]] if relative_to else res

        # Scan parameter inputs and build the dictionary for those with default values
        for input_ in root.iter('paraminput'):
            input_name = input_.find('identifier').get('v')
            input_uid = int(input_.find('uid').get('v'))
            input_type = input_.find('type').get('v')
            if input_type is None:
                input_type = input_.find('type/value').get('v')
            input_type = int(input_type)
            input_val = None if input_type in (1, 2) else \
                self.resolve_param_val(input_.find('defaultValue')[0])
            new_param = SBSParameter(input_name, input_uid, input_type, input_val)
            self.inputs.append(new_param)

            # Add non-image parameters to global parameter dictionary
            if input_type not in (1, 2):
                self.input_dict[input_name] = new_param

        # Scan graph outputs
        tobe_removed_graphoutput = []
        tobe_removed_compNode = []
        tobe_removed_rootOutput = []
        for output_ in root.iter('graphoutput'):
            output_name = output_.find('identifier').get('v')
            output_uid = int(output_.find('uid').get('v'))
            output_group_ = output_.find('group')
            output_group = output_group_.get('v') if output_group_ is not None else ''
            output_usage_ = output_.find('usages/usage')
            output_usage = output_usage_.find('name').get('v') if output_usage_ else ''
            self.outputs.append(SBSOutput(output_name, output_uid, output_usage, output_group))

        # remove recorded nodes
        for idx, node in enumerate(tobe_removed_graphoutput):
            root.find('content/graph/graphOutputs').remove(node)
            root.find('content/graph/compNodes').remove(tobe_removed_compNode[idx])
            root.find('content/graph/root/rootOutputs').remove(tobe_removed_rootOutput[idx])

        def fix_output(original_node):
            '''
            Clone and change the output parameter of a node (in ET)
            '''
            node = copy.deepcopy(original_node)
            for params_ in node.iter('parameters'):
                for param_ in params_.iter('parameter'):
                    if param_.find('name').get('v') == 'outputsize':
                        params_.remove(param_)
            return node

        # Scan graph nodes
        for node in root.iter('compNode'):
            node_uid = int(node.find('uid').get('v'))

            # Identify node type and create node object
            node_imp = node.find('compImplementation')[0]
            # Input node
            if node_imp.tag == 'compInputBridge':
                input_uid = int(node_imp.find('entry').get('v'))
                node_name = self.gen_node_name('input')
                node_obj = SBSInputNode(node_name, node_uid, self.get_input_param(input_uid))
            # Output node
            elif node_imp.tag == 'compOutputBridge':
                output_uid = int(node_imp.find('output').get('v'))
                node_name = self.gen_node_name('output')
                node_obj = SBSOutputNode(node_name, node_uid, self.get_output_param(output_uid))
            # Non-atomic node
            elif node_imp.tag == 'compInstance':
                path_ = node_imp.find('path')
                path = path_.get('v')
                if path is None:
                    path = path_.find('value').get('v')
                instance_type = path[path.rfind('/') + 1: path.rfind('?')]
                node_name = self.gen_node_name(instance_type)
                # The node is a filter which requires input
                if node.find('connections') or instance_type == 'normal_color':
                    node_class, node_type, node_func = self.lookup_type(instance_type)
                    node_obj = node_class(node_name, node_uid, node_type, node_func, self.res, self.use_alpha)
                # This node is a generator that only provides inputs
                else:
                    node_obj = SBSGeneratorNode(node_name, node_uid, self.res, self.use_alpha)

                    # Build input to output graph
                    # Add this node to input_output_root
                    compNodes = input_output_root.find('content/graph/compNodes')
                    compNodes.append(fix_output(node))

                    # Add dependency node
                    path = node.find('compImplementation/compInstance/path').get('v')
                    path_ref = path.split("=")[1]
                    for dependency_node in root.iter('dependency'):
                        if dependency_node.find('uid').get('v') == path_ref:
                            exist_flag = False
                            for io_dependency_node in input_output_root.iter('dependency'):
                                if io_dependency_node.find('uid').get('v') == path_ref:
                                    exist_flag = True
                            if exist_flag == False:
                                input_output_root.find('dependencies').append(dependency_node)

                    # Add output node
                    GUILayout = node.find('GUILayout')
                    gpos = GUILayout.find('gpos').get('v')
                    gpos = [float(i) for i in gpos.strip().split()]
                    
                    compOutputs = node.find('compOutputs')
                    append_postfix_flag = len(list(compOutputs)) > 1
                    # iterate all compOutput
                    for compOutput in compOutputs.iter('compOutput'):
                        if append_postfix_flag:
                            # append output names
                            output_bridges = node.find('compImplementation/compInstance//outputBridgings')
                            for output_bridge in output_bridges.iter('outputBridging'):
                                if int(output_bridge.find('uid').get('v')) == int(compOutput.find('uid').get('v')):
                                    output_name = node_name + '_' + output_bridge.find('identifier').get('v')
                        else:
                            output_name = node_name

                        compOutput_uid = int(compOutput.find('uid').get('v'))

                        offset = 500000000
                        rand_offset = 100000000
                        new_gpos = [gpos[0] - 100, gpos[1] - 100, 0]
                        new_node_uid = node_uid + offset + random.randint(0, rand_offset)
                        new_node_output = compOutput_uid + offset + random.randint(0, rand_offset)

                        new_node_elem = ET.SubElement(compNodes, 'compNode')
                        # Create uid
                        new_node_uid_elem = ET.SubElement(new_node_elem, 'uid')
                        new_node_uid_elem.set('v', str(new_node_uid))
                        # Create gpos
                        new_GUILayout_elem = ET.SubElement(new_node_elem, 'GUILayout')
                        new_gpos_elem = ET.SubElement(new_GUILayout_elem, 'gpos')
                        new_gpos_elem.set('v', ' '.join(map(str, new_gpos)))
                        # Create connections
                        new_connections_elem = ET.SubElement(new_node_elem, 'connections')
                        new_connection_elem = ET.SubElement(new_connections_elem, 'connection')
                        new_identifier_elem = ET.SubElement(new_connection_elem, 'identifier')
                        new_identifier_elem.set('v', "inputNodeOutput")
                        new_connRef_elem = ET.SubElement(new_connection_elem, 'connRef')
                        new_connRef_elem.set('v', str(node_uid))
                        new_connRefOutput_elem = ET.SubElement(new_connection_elem, 'connRefOutput')
                        new_connRefOutput_elem.set('v', str(compOutput_uid))
                        # Create compImplementation
                        new_compImplementation_elem = ET.SubElement(new_node_elem, 'compImplementation')
                        new_compOutputBridge_elem = ET.SubElement(new_compImplementation_elem, 'compOutputBridge')
                        new_output_elem = ET.SubElement(new_compOutputBridge_elem, 'output')
                        new_output_elem.set('v', str(new_node_output))

                        # Add graphoutput
                        graphOutputs = input_output_root.find('content/graph/graphOutputs')
                        new_graphOutput_elem = ET.SubElement(graphOutputs, 'graphoutput')
                        new_go_identifier_elem = ET.SubElement(new_graphOutput_elem, 'identifier')
                        new_go_identifier_elem.set('v', output_name)
                        new_go_uid_elem = ET.SubElement(new_graphOutput_elem, 'uid')
                        new_go_uid_elem.set('v', str(new_node_output))
                        new_go_channels_elem = ET.SubElement(new_graphOutput_elem, 'channels')
                        new_go_channels_elem.set('v', str(2))
                        new_go_group_elem = ET.SubElement(new_graphOutput_elem, 'group')
                        new_go_group_elem.set('v', 'input')

                        # Add rootOutput
                        rootOutputs_elem = input_output_root.find('content/graph/root/rootOutputs')
                        new_rootOutput_elem = ET.SubElement(rootOutputs_elem, 'rootOutput')
                        new_ro_output_elem = ET.SubElement(new_rootOutput_elem, 'output')
                        new_ro_output_elem.set('v', str(new_node_output))
                        new_ro_format_elem = ET.SubElement(new_rootOutput_elem, 'format')
                        new_ro_format_elem.set('v', "0")
                        new_ro_usertag_elem = ET.SubElement(new_rootOutput_elem, 'usertag')
                        new_ro_usertag_elem.set('v', "")

            # Atomic node
            elif node_imp.tag == 'compFilter':
                filter_type = node_imp.find('filter').get('v')
                if filter_type in ['fxmaps', 'pixelprocessor', 'svg']:  # debug purposes
                    if not node.find('connections'):
                        node_name = self.gen_node_name(filter_type)
                        node_obj = SBSGeneratorNode(node_name, node_uid, self.res, self.use_alpha)

                        # Build input to output graph
                        # Add this node to input_output_root
                        compNodes = input_output_root.find('content/graph/compNodes')
                        compNodes.append(fix_output(node))

                        # Add output node
                        GUILayout = node.find('GUILayout')
                        gpos = GUILayout.find('gpos').get('v')
                        gpos = [float(i) for i in gpos.strip().split()]
                        
                        compOutputs = node.find('compOutputs')
                        compOutput = compOutputs.find('compOutput')
                        compOutput_uid = int(compOutput.find('uid').get('v'))

                        offset = 500000000
                        rand_offset = 100000000
                        new_gpos = [gpos[0] - 100, gpos[1] - 100, 0]
                        new_node_uid = node_uid + offset + random.randint(0, rand_offset)
                        new_node_output = compOutput_uid + offset + random.randint(0, rand_offset)

                        new_node_elem = ET.SubElement(compNodes, 'compNode')
                        # Create uid
                        new_node_uid_elem = ET.SubElement(new_node_elem, 'uid')
                        new_node_uid_elem.set('v', str(new_node_uid))
                        # Create gpos
                        new_GUILayout_elem = ET.SubElement(new_node_elem, 'GUILayout')
                        new_gpos_elem = ET.SubElement(new_GUILayout_elem, 'gpos')
                        new_gpos_elem.set('v', ' '.join(map(str, new_gpos)))
                        # Create connections
                        new_connections_elem = ET.SubElement(new_node_elem, 'connections')
                        new_connection_elem = ET.SubElement(new_connections_elem, 'connection')
                        new_identifier_elem = ET.SubElement(new_connection_elem, 'identifier')
                        new_identifier_elem.set('v', "inputNodeOutput")
                        new_connRef_elem = ET.SubElement(new_connection_elem, 'connRef')
                        new_connRef_elem.set('v', str(node_uid))
                        new_connRefOutput_elem = ET.SubElement(new_connection_elem, 'connRefOutput')
                        new_connRefOutput_elem.set('v', str(compOutput_uid))
                        # Create compImplementation
                        new_compImplementation_elem = ET.SubElement(new_node_elem, 'compImplementation')
                        new_compOutputBridge_elem = ET.SubElement(new_compImplementation_elem, 'compOutputBridge')
                        new_output_elem = ET.SubElement(new_compOutputBridge_elem, 'output')
                        new_output_elem.set('v', str(new_node_output))

                        # Add graphoutput
                        graphOutputs = input_output_root.find('content/graph/graphOutputs')
                        new_graphOutput_elem = ET.SubElement(graphOutputs, 'graphoutput')
                        new_go_identifier_elem = ET.SubElement(new_graphOutput_elem, 'identifier')
                        new_go_identifier_elem.set('v', node_name)
                        new_go_uid_elem = ET.SubElement(new_graphOutput_elem, 'uid')
                        new_go_uid_elem.set('v', str(new_node_output))
                        new_go_channels_elem = ET.SubElement(new_graphOutput_elem, 'channels')
                        new_go_channels_elem.set('v', str(2))
                        new_go_group_elem = ET.SubElement(new_graphOutput_elem, 'group')
                        new_go_group_elem.set('v', 'input')

                        # Add rootOutput
                        rootOutputs_elem = input_output_root.find('content/graph/root/rootOutputs')
                        new_rootOutput_elem = ET.SubElement(rootOutputs_elem, 'rootOutput')
                        new_ro_output_elem = ET.SubElement(new_rootOutput_elem, 'output')
                        new_ro_output_elem.set('v', str(new_node_output))
                        new_ro_format_elem = ET.SubElement(new_rootOutput_elem, 'format')
                        new_ro_format_elem.set('v', "0")
                        new_ro_usertag_elem = ET.SubElement(new_rootOutput_elem, 'usertag')
                        new_ro_usertag_elem.set('v', "")    
                else:
                    node_name = self.gen_node_name(filter_type)
                    node_class, node_type, node_func = self.lookup_type(filter_type)
                    node_obj = node_class(node_name, node_uid, node_type, node_func, self.res, self.use_alpha)
            else:
                raise NotImplementedError('Unrecognized node type: {}'.format(node_imp.tag))

            # Add output bridges
            if not isinstance(node_obj, SBSOutputNode):
                node_outputs_ = node.findall('compOutputs/compOutput')
                node_output_bridges_ = node_imp.findall('outputBridgings/outputBridging')
                if node_output_bridges_:
                    for node_output_, node_output_bridge_ in zip(node_outputs_, node_output_bridges_):
                        node_output_uid = int(node_output_.find('uid').get('v'))
                        node_output_type = int(node_output_.find('comptype').get('v'))
                        node_output_name = node_output_bridge_.find('identifier').get('v')
                        node_obj.add_output_bridge(node_output_uid,
                                                   SBSOutputBridge(node_output_uid, node_output_name, node_output_type))
                else:
                    for node_output_ in node_outputs_:
                        node_output_uid = int(node_output_.find('uid').get('v'))
                        node_output_type = int(node_output_.find('comptype').get('v'))
                        node_obj.add_output_bridge(node_output_uid,
                                                   SBSOutputBridge(node_output_uid, bridge_type=node_output_type))

                # Update output variable names based on the names of output bridges
                node_obj.update_output_variable_names()

                # Create extra parameters for generators
                if isinstance(node_obj, SBSGeneratorNode):
                    node_obj.init_sbs_params()
                    self.inputs.extend(node_obj.get_sbs_params_list())

            # Resolve node parameters
            if node_obj.type not in ('Passthrough', 'Generator'):
                self.resolve_node_params(node_imp, node_obj)

            # Save the current node into the global library
            self.nodes[node_uid] = node_obj

        # Scan graph connectivity
        for node in root.iter('compNode'):
            uid = int(node.find('uid').get('v'))
            if uid not in self.nodes:
                continue
            node_obj = self.nodes[uid]

            # Check each input connection
            for conn_ in node.findall('connections/connection'):
                conn_name = conn_.find('identifier').get('v')
                conn_ref = int(conn_.find('connRef').get('v'))
                conn_output_ref_ = conn_.find('connRefOutput')
                conn_output_ref = int(conn_.find('connRefOutput').get('v')) \
                                  if conn_output_ref_ is not None else None
                if conn_ref not in self.nodes:
                    node_obj.add_connection(conn_name)
                # Also update the output bridge of the previous node
                else:
                    node_prev = self.nodes[conn_ref]
                    node_obj.add_connection(conn_name, node_prev, conn_output_ref)
                    node_prev.update_output_bridge(conn_output_ref, node_obj)
        
        # save input to output graph
        input_output_tree.write(self.input_output_file_name)

    def gen_noise(self, noise_count, noise_format):
        """Generate groups of random input noise patterns.

        Args:
            noise_count (int): number of noise pattern groups
            noise_format (str): format of the saved noise pattern files (like 'png'|'jpg')
        """
        import sys
        import subprocess
        import numpy as np

        # Global parameters
        sbs_file_name = self.input_output_file_name
        sbsar_file_name = self.input_output_file_name[self.input_output_file_name.rfind('/') + 1:] + 'ar'
        cooker_path = os.path.join(self.toolkit_path, 'Substance_Automation_Toolkit', 'sbscooker')
        render_path = os.path.join(self.toolkit_path, 'Substance_Automation_Toolkit', 'sbsrender')
        packages_dir = os.path.join(self.toolkit_path, 'Substance_Automation_Toolkit', 'resources', 'packages')
        res_folder = os.path.join(self.output_path, self.output_graph_name, 'random_seeds')

        # Assemble cooker and render commands
        if platform.system() == 'Windows':
            print("using windows")
            command_cooker = '\"{}\" \"{}\" --includes \"{}\"'.format(cooker_path, sbs_file_name, packages_dir)
            command_render = '\"{}\" render \"{}\" --output-format \"{}\" --output-name \"{{outputNodeName}}\"'.format(render_path, sbsar_file_name, noise_format)
            command_move = 'move *.{} \"{}\"'
            command_clean = 'del \"{}\"'.format(sbsar_file_name)
        else:
            command_cooker = '\"{}\" \"{}\" --includes \"{}\"'.format(cooker_path, sbs_file_name, packages_dir)
            command_render = '\"{}\" render \"{}\" --output-format \"{}\" --output-name \"{{outputNodeName}}\" 1> /dev/null 2>&1'.format(render_path, sbsar_file_name, noise_format)
            command_move = 'mv *.{} \"{}\"'
            command_clean = 'rm \"{}\"'.format(sbsar_file_name)

        # Create result folder
        os.makedirs(res_folder, mode=0o775, exist_ok=True)

        # Parse XML file
        tree = ET.parse(sbs_file_name)
        root = tree.getroot()

        # Generate dataset
        for data_num in range(noise_count):
            print('Random seed:', data_num)
            res_path = os.path.join(res_folder, str(data_num))
            os.makedirs(res_path, mode=0o775, exist_ok=True)

            # Traverse all compute nodes
            for comp_node in root.iter('compNode'):
                comp_inst = comp_node.find('compImplementation').find('compInstance')
                if comp_inst is not None:
                    # Retrieve random seed parameter
                    set_seed = False
                    for comp_param in comp_inst.iter('parameter'):
                        if comp_param.find('name').get('v') == 'randomseed':
                            val = comp_param.find('paramValue').find('constantValueInt32')
                            val.set('v', str(data_num))
                            set_seed = True

                    # Manually add a random seed entry
                    if not set_seed:
                        comp_params = comp_inst.find('parameters')
                        comp_param = ET.SubElement(comp_params, 'parameter')
                        ET.SubElement(comp_param, 'name').set('v', 'randomseed')
                        ET.SubElement(comp_param, 'relativeTo').set('v', '1')
                        ET.SubElement(ET.SubElement(comp_param, 'paramValue'), 'constantValueInt32').set('v', str(data_num))

            # Save XML file
            tree.write(sbs_file_name)

            # Render output images
            subprocess.run(command_cooker, shell=True)
            subprocess.run(command_render, shell=True)
            subprocess.run(command_move.format(noise_format, res_path), shell=True)

        # Clean up
        subprocess.run(command_clean, shell=True)

    def analyze(self):
        """Run data flow analysis to find active nodes and determine node operation sequence.
        """
        # Calculate reachable nodes from output
        active_output_nodes = [output.parent for output in self.outputs if output.is_optimizable(self.lambertian)]
        queue = deque(active_output_nodes)
        visited_uids = {node.uid for node in active_output_nodes}

        # Run backward BFS
        while queue:
            node = queue.popleft()
            for _, conn_ref, conn_output_ref in node.connections:
                if conn_ref is not None:
                    conn_ref.set_output_bridge_reachable(conn_output_ref)
                    if conn_ref.uid not in visited_uids:
                        queue.append(conn_ref)
                        visited_uids.add(conn_ref.uid)

        # Save visited nodes
        visited_nodes = [self.nodes[uid] for uid in visited_uids]

        # Initialize the set of active nodes with zero indegrees
        active_init_nodes = OrderedSet([param.parent for param in self.inputs if param.reachable])
        for node in visited_nodes:
            if node.type in ('UniformColor', 'NormalColor'):
                active_init_nodes.add(node)

        # Determine other active nodes and count indegrees
        queue.extend(active_init_nodes)
        active_uids = {node.uid for node in active_init_nodes}
        indegrees = {uid: 0 for uid in active_uids}

        # Run forward BFS
        while queue:
            node = queue.popleft()
            for bridge in node.get_reachable_output_bridges():
                for target in bridge.targets:
                    if target.uid in visited_uids:
                        if target.uid not in active_uids:
                            queue.append(target)
                            active_uids.add(target.uid)
                            indegrees[target.uid] = 1
                        else:
                            indegrees[target.uid] += 1

        # Check if all optimizable outputs are covered
        for node in active_output_nodes:
            if node.uid not in active_uids:
                raise RuntimeError('An optimizable output is not connected')

        # Run topology sorting to compute node sequence
        queue.extend(active_init_nodes)
        self.node_seq = []

        while queue:
            node = queue.popleft()
            if node.type not in ['Input', 'Output', 'Generator']:
                self.node_seq.append(node)
            for bridge in node.get_reachable_output_bridges():
                for target in bridge.targets:
                    if target.uid in active_uids:
                        indegrees[target.uid] -= 1
                        if not indegrees[target.uid]:
                            queue.append(target)

        # Mark used exposed parameters
        for node in self.node_seq:
            for param in node.params_list:
                if param.is_dynamic():
                    param.mark_exposed_params(self.input_dict)

    def convert(self):
        """Convert the graph and save the target PyTorch program into a designated file.

        Args:
            output_path (str, optional): output folder path. Defaults to '' (the current directory).
            graph_name (str, optional): name of the graph to use. Defaults to '' (read from the source file).
        """
        graph_name = self.output_graph_name

        # Generate the definition of forward function
        def gen_forward_function(in_class=False):
            # Initialize string list and write function
            str_list = []
            write = lambda s: str_list.append(s)

            # Input assignment (only for class method)
            if in_class:
                for param in self.inputs:
                    if param.reachable:
                        write(f"{param.name} = self.input_dict['{param.name}']")
                write('')

            # Function body
            for node in self.node_seq:
                write(f'{node.get_op_str(in_class=in_class)}')
            write('')

            # Output variable assignment
            output_usages = ['baseColor', 'normal', 'metallic', 'roughness']
            for usage in output_usages:
                has_variable = False
                for output in self.outputs:
                    if output.usage == usage and output.is_optimizable(self.lambertian):
                        has_variable = True
                        variable = output.parent.get_variable_reference()
                        if self.scale:
                            if in_class:
                                write(f"{usage.lower()} = self.eval_op('special_transform', {variable}).squeeze()")
                            else:
                                write(f"{usage.lower()} = F.special_transform({variable}, **params['special_transform']).squeeze()")
                        else:
                            write(f'{usage.lower()} = {variable}.squeeze()')
                        if usage == 'normal' and self.correct_normal:
                            write(f'{usage.lower()} = correct_normal({usage.lower()})')
                if not has_variable:
                    print(usage)
                    if usage is 'metallic':
                        if self.scale:
                            if in_class:
                                write(f"{usage.lower()} = self.eval_op('special_transform', th.zeros(1,1,%d, %d)).squeeze()" % (1<<self.res[0], 1<<self.res[1]))
                            else:
                                write(f"{usage.lower()} = F.special_transform(th.zeros(1,1,%d, %d), **params['special_transform']).squeeze()" % (1<<self.res[0], 1<<self.res[1]))
                        else:
                            write(f'{usage.lower()} = th.zeros(%d, %d)' % (1<<self.res[0], 1<<self.res[1]))
                    elif usage is 'roughness':
                        if self.scale:
                            if in_class:
                                write(f"{usage.lower()} = self.eval_op('special_transform', th.ones(1,1,%d, %d)).squeeze()" % (1<<self.res[0], 1<<self.res[1]))
                            else:
                                write(f"{usage.lower()} = F.special_transform(th.ones(1,1,%d, %d), **params['special_transform']).squeeze()" % (1<<self.res[0], 1<<self.res[1]))
                        else:
                            write(f'{usage.lower()} = th.ones(%d, %d)' % (1<<self.res[0], 1<<self.res[1]))
                    elif usage is 'baseColor':
                        if self.scale:
                            if in_class:
                                write(f"{usage.lower()} = self.eval_op('special_transform', th.ones(1, %d, %d, %d)).squeeze()" % (4 if self.use_alpha else 3, 1<<self.res[0], 1<<self.res[1]))
                            else:
                                write(f"{usage.lower()} = F.special_transform(th.ones(1, %d, %d, %d), **params['special_transform']).squeeze()" % (4 if self.use_alpha else 3, 1<<self.res[0], 1<<self.res[1]))
                        else:
                            write(f'{usage.lower()} = th.ones(%d, %d, %d)' % (4 if self.use_alpha else 3, 1<<self.res[0], 1<<self.res[1]))
                    elif usage is 'normal':
                        if self.scale:
                            if in_class:
                                write(f"{usage.lower()} = self.eval_op('special_transform', th.ones(1, %d, %d, %d)).squeeze()" % (4 if self.use_alpha else 3, 1<<self.res[0], 1<<self.res[1]))
                                write(f'{usage.lower()}[:2,:,:] =  {usage.lower()}[:2,:,:] / 2.0')
                            else:
                                write(f"{usage.lower()} = F.special_transform(th.ones(1, %d, %d, %d), **params['special_transform']).squeeze()" % (4 if self.use_alpha else 3, 1<<self.res[0], 1<<self.res[1]))
                                write(f'{usage.lower()}[:2,:,:] =  {usage.lower()}[:2,:,:] / 2.0')
                        else:
                            write(f'{usage.lower()} = th.ones(%d, %d, %d)' % (4 if self.use_alpha else 3, 1<<self.res[0], 1<<self.res[1]))
                            write(f'{usage.lower()}[:2,:,:] =  {usage.lower()}[:2,:,:] / 2.0')
                        if self.correct_normal:
                            write(f'{usage.lower()} = correct_normal({usage.lower()})')

            # Return statement
            output_str = ', '.join([usage.lower() for usage in output_usages])
            write(f'return [{output_str}]')

            return str_list

        # Generate the graph params
        def gen_graph_params(only_dynmaic_params=False):
            # Initialize string list and write function
            str_list = []
            write = lambda s: str_list.append(s)

            # Append node parameters
            if not only_dynmaic_params:
                for node in self.node_seq:
                    write(f'{node.get_params_str(only_dynmaic_params=False)},')
                # Add scale and offset parameters
                if self.scale:
                    write("('special_transform', {'tile_mode': 3, 'sample_mode': 'bilinear', 'scale': %.4f , 'scale_max': %.4f, 'x_offset': 0.5, 'x_offset_max': 2.0, 'y_offset': 0.5, 'y_offset_max': 2.0, 'trainable': [True]*3})," % (1.0/self.max_scale, self.max_scale))
            else:
                for node in self.node_seq:
                    line = node.get_params_str(only_dynmaic_params=True)
                    if line is not None:
                        write(f'{line},')

            return str_list
            
        # Generate node list
        def gen_node_list():
            # Initialize string list and write function
            str_list = []
            write = lambda s: str_list.append(s)

            # Append node list
            node_types = [f"'{node.type}'," for node in self.node_seq]
            if self.scale:
                node_types.append("'SpecialTransform',")
            line_str = ''
            for node_type in node_types:
                if len(line_str) == 0:
                    line_str += node_type
                elif len(line_str) + len(node_type) + 2 <= 96:
                    line_str += ' ' + node_type
                else:
                    write(line_str)
                    line_str = node_type
            if len(line_str) > 0:
                write(line_str)

            return str_list

        # Generate input dictionary
        def gen_input_names():
            # Append input entries
            str_list = []
            for param in self.inputs:
                if param.reachable:
                    str_list.append(f"'{param.name}': '{param.name}.{self.noise_format}',")

            return str_list

        # Generate exposed parameters
        def gen_exposed_params():
            # Write exposed parameter dictionary
            str_list = [f'{graph_name}_exposed_params = OrderedDict([']
            write = lambda s: str_list.append(s)

            # Skip if no exposed parameters or they are not considered
            if not self.input_dict:
                write('])')
                return str_list

            to_str = lambda v: f"'{v}'" if isinstance(v, str) else str(v)
            used_pairs = [(key, param) for key, param in self.input_dict.items() if param.used]
            for key, param in used_pairs:
                write(f"    ('{key}', {to_str(param.val)}),")
            trainable_list = [param.is_trainable() for _, param in used_pairs]
            write(f"    ('trainable', {trainable_list}),")
            write('])')

            # Write partial object constructor
            write('')
            write('# Represent each dynamic node parameter by a partial object')
            write(f'{NAME_TO_PARTIAL_FUNC} = lambda func: partial(func, {graph_name}_exposed_params)')

            return str_list

        # Generate dynamic functions
        def gen_dynamic_functions():
            # Collect dynamic functions from each node
            str_list = []
            for node in self.node_seq:
                func_str = node.get_dynamic_functions_str()
                if func_str:
                    str_list.append(func_str)

            return str_list if str_list else ''

        # Output a utility script according to a given template file
        def write_script(output_file_name, template_file_name, macs):
            # Read the template file
            with open(template_file_name, 'r') as f:
                lines = f.readlines()

            # Filter blank lines
            filter_blank = lambda s: '\n'.join(['' if not v.strip() else v for v in s.split('\n')])

            # Write the output file while substituting macros by their content
            with open(output_file_name, 'w') as f:
                for line in lines:
                    # Count the number of macros
                    num_macros = 0
                    for macro in macs:
                        num_macros += line.count(macro)

                    # Apply macros
                    for macro, content in macs.items():
                        if isinstance(content, str):
                            line = line.replace(macro, content)
                    for macro, content in macs.items():
                        if macro in line and isinstance(content, list):
                            assert num_macros == 1, 'Only one multi-line macro is allowed in the template per line.'
                            line = ''.join([filter_blank(line.replace(macro, content_line)) for content_line in content])

                    # Write out the current line
                    f.write(line)

        # Construct template macro dictionary
        # !! Note: macros are not allowed to contain one another !!
        macs = {}
        macs['GRAPH_NAME'] = graph_name
        macs['CLASS_NAME'] = ''.join([s.capitalize() for s in graph_name.split('_')])
        macs['GRAPH_EXPOSED_PARAMS'] = gen_exposed_params()
        macs['GRAPH_DYNAMIC_FUNCTIONS'] = gen_dynamic_functions()
        macs['GRAPH_TP_PARAMS'] = gen_graph_params(True)
        macs['GRAPH_PARAMS'] = gen_graph_params()
        macs['NODE_LIST'] = gen_node_list()
        macs['INPUT_NAMES'] = gen_input_names()
        macs['FORWARD_ARGS'] = ', '.join(['params'] + [param.name for param in self.inputs if param.reachable])
        macs['FORWARD_PROGRAM'] = gen_forward_function()
        macs['FORWARD_CLASS'] = gen_forward_function(True)
        macs['DEVICE_COUNT'] = str(self.device_count)
        macs['NOISE_COUNT'] = str(self.noise_count)
        macs['USE_ALPHA'] = str(self.use_alpha)
        macs['RES_H'] = str(1 << self.res[0])
        macs['RES_W'] = str(1 << self.res[1])

        # Write script files in template
        project_dir = os.path.abspath(os.path.join(__file__, "..", ".."))
        template_dir = os.path.join(project_dir, 'templates')
        config_dir = os.path.join(self.output_path, graph_name, 'configs')
        os.makedirs(config_dir, mode=0o775, exist_ok=True)
        for file_name in os.listdir(template_dir):
            if file_name.startswith('tp_')  and file_name.endswith('.py'):
                output_name = os.path.join(self.output_path, graph_name, graph_name + '_' + file_name.split('_', 1)[1])
                write_script(output_name, os.path.join(template_dir, file_name), macs)
                Path(os.path.join(config_dir, file_name[file_name.find('tp_') + 3: file_name.rfind('.')]+'.conf')).touch()
                

def main():
    # Set up command line argument parser
    parser = argparse.ArgumentParser(description='Automatic converter of SBS documents to PyTorch modules for optimization.')
    parser.add_argument('input_file', metavar='FILE', help='Path to input *.sbs file.')
    parser.add_argument('-t', '--toolkit_path', default=None, help='Path to the Substance Automation Toolkit')
    parser.add_argument('-o', '--output_path', default=None, help='Path to the output PyTorch module file')
    parser.add_argument('-r', '--output_res', type=int, default=9, help='Specify the output graph resolution')
    parser.add_argument('-g', '--graph_name', default=None, help='Name of the output graph')
    parser.add_argument('-a', '--use_alpha', action='store_true', help='Enable alpha channel')
    parser.add_argument('-n', '--save_noise', default='exr', help='Generate input noises in specified format', choices=('exr', 'jpg', 'png'))
    parser.add_argument('-c', '--noise_count', default=0, type=int, help='Number of random seeds for generating noises')
    parser.add_argument('-l', '--lambertian', action='store_true', help='Force the output roughness to be 1 and matallic to be 0')
    parser.add_argument('-d', '--device_count', default=0, type=int, help='GPU device number')
    parser.add_argument('-s', '--scale', action='store_true', help='Add special transform at the end of graph to scale and offset output')
    parser.add_argument('-m', '--max_scale', default=4.0, type=float, help='Max scale of the special transform')
    parser.add_argument('-k', '--correct_normal', action='store_true', help='Add automatic normal correction to eliminate normal bias')

    # Parse arguments
    namespace = parser.parse_args()
    input_file_path = namespace.input_file
    output_path = namespace.output_path
    toolkit_path = namespace.toolkit_path
    graph_name = namespace.graph_name

    # Automatic conversion
    converter = SBSGraphConverter(input_file_path,
                                  output_path=output_path,
                                  output_graph_name=graph_name,
                                  output_res=namespace.output_res,
                                  use_alpha=namespace.use_alpha,
                                  save_noise=namespace.noise_count > 0,
                                  noise_format=namespace.save_noise,
                                  noise_count=namespace.noise_count,
                                  lambertian=namespace.lambertian,
                                  device_count=namespace.device_count,
                                  scale=namespace.scale,
                                  max_scale=namespace.max_scale,
                                  correct_normal=namespace.correct_normal)
    converter.convert()

if __name__ == '__main__':
    main()