import sys
import os
import abc
import copy
import types
import inspect
import torch as th
import numpy as np
from functools import partial
import matplotlib.pyplot as plt

import diffmat.sbs_core.functional as F
from diffmat.sbs_core.util import input_check, convert_params_dict_to_list, convert_params_list_to_dict, to_tensor


class BaseNode(abc.ABC):
    '''Base node class to be inherited by a specific node class implementation.
    This class implements helper functions for creating trainable node parameters.
    '''
    TRAINABLE_INPUT = []
    DEFAULT_VARS = []
    HARDTANH_FLAG = []
    RELU_FLAG = []

    def __init__(self):
        self.parameters = []

    def prepare_args(self, args, trainable):
        default_vals = dict(zip(self.TRAINABLE_INPUT, self.DEFAULT_VARS))
        self.trainable = dict(zip(self.TRAINABLE_INPUT, trainable))
        self.hardtanh_flag = dict(zip(self.TRAINABLE_INPUT,
                                      self.HARDTANH_FLAG))
        self.relu_flag = dict(zip(self.TRAINABLE_INPUT, self.RELU_FLAG))
        self.args = self.create_trainable_params(args, self.TRAINABLE_INPUT,
                                                 self.trainable, default_vals)

    def update_args(self):
        """Apply hardtanh to any parameter that should fall in between [0.0, 1.0],
        apply relu to any parameter that should fall in between [0.0, +oo].
        """
        args = self.args.copy()

        for k in self.TRAINABLE_INPUT:
            # evaluate partial
            switch = lambda val: val() if isinstance(val, partial) else val
            if self.hardtanh_flag[k]:
                args[k] = th.nn.functional.hardtanh(switch(self.args[k]), 0.0,
                                                    1.0)
            elif self.relu_flag[k]:
                args[k] = th.nn.functional.relu(swtich(self.args[k]))

        return args

    def create_trainable_params(self, args, trainable_input, trainable,
                                default_vals):
        """Convert all numerical parameters to pytorch tensors, append the trainable 
        tensors to self.parameters.
        """
        for key in trainable_input:
            if args[key] is None:
                args[key] = to_tensor(default_vals[key])
            elif isinstance(args[key], partial):
                pass
            else:
                args[key] = to_tensor(args[key])

            if trainable[key] and (not isinstance(args[key], partial)):
                args[key].requires_grad = True
                self.parameters.append(args[key])

        # delete useless keys
        del args['self']
        del args['trainable']
        del args['__class__']

        return args

    @abc.abstractmethod
    def __call__(self, **args):
        return


class Blend(BaseNode):
    '''Wrapper class for atomic node "Blend".
    '''
    TRAINABLE_INPUT = ['opacity']
    DEFAULT_VARS = [1.0]
    HARDTANH_FLAG = [True]
    RELU_FLAG = [False]

    def __init__(self, opacity=None, trainable=[True]):
        super(Blend, self).__init__()
        self.prepare_args(locals(), trainable)

    def __call__(self,
                 img_fg=None,
                 img_bg=None,
                 blend_mask=None,
                 blending_mode='copy',
                 cropping=[0.0, 1.0, 0.0, 1.0]):
        args = self.update_args()
        return F.blend(img_fg, img_bg, blend_mask, blending_mode, cropping,
                       **args)


class Blur(BaseNode):
    '''Wrapper class for atomic node "Blur".
    '''
    TRAINABLE_INPUT = ['intensity']
    DEFAULT_VARS = [0.5]
    HARDTANH_FLAG = [True]
    RELU_FLAG = [False]

    def __init__(self, intensity=None, max_intensity=20.0, trainable=[True]):
        super(Blur, self).__init__()
        self.prepare_args(locals(), trainable)

    def __call__(self, img_in):
        args = self.update_args()
        return F.blur(img_in, **args)


class Curve(BaseNode):
    '''Wrapper class for atomic node "Curve".
    '''
    TRAINABLE_INPUT = ['anchors']
    DEFAULT_VARS = []
    HARDTANH_FLAG = [True, True, True]
    RELU_FLAG = [False, False, False]

    def __init__(self, num_anchors=2, anchors=None, trainable=[True]):
        super(Curve, self).__init__()
        if anchors is not None:
            if isinstance(anchors, list):
                anchors = np.array(anchors, dtype=np.float32)
            assert num_anchors == anchors.shape[
                0], "number of anchors should match the anchors.shape[0]"
        self.DEFAULT_VARS.append(th.stack([th.zeros(6), th.ones(6)]))
        self.prepare_args(locals(), trainable)

    def __call__(self, img_in):
        args = self.update_args()
        return F.curve(img_in, **args)


class DBlur(BaseNode):
    '''Wrapper class for atmoic node "Directional Blur".
    '''
    TRAINABLE_INPUT = ['intensity', 'angle']
    DEFAULT_VARS = [0.5, 0.0]
    HARDTANH_FLAG = [True, True]
    RELU_FLAG = [False, False]

    def __init__(self,
                 intensity=None,
                 max_intensity=20.0,
                 angle=None,
                 trainable=[True, True]):
        super(DBlur, self).__init__()
        self.prepare_args(locals(), trainable)

    def __call__(self, img_in):
        args = self.update_args()
        return F.d_blur(img_in, **args)


class DWarp(BaseNode):
    '''Wrapper class for atomic node "Directional Wrap".
    '''
    TRAINABLE_INPUT = ['intensity', 'angle']
    DEFAULT_VARS = [0.5, 0.0]
    HARDTANH_FLAG = [True, True]
    RELU_FLAG = [False, False]

    def __init__(self,
                 intensity=None,
                 max_intensity=20.0,
                 angle=None,
                 trainable=[True, True]):
        super(DWarp, self).__init__()
        self.prepare_args(locals(), trainable)

    def __call__(self, img_in, intensity_mask):
        args = self.update_args()
        return F.d_warp(img_in, intensity_mask, **args)


class Distance(BaseNode):
    '''Wrapper class for atomic node "Distance".
    '''
    TRAINABLE_INPUT = ['dist']
    DEFAULT_VARS = [10.0 / 256.0]
    HARDTANH_FLAG = [True]
    RELU_FLAG = [False]

    def __init__(self, dist=None, max_dist=256.0, trainable=[True]):
        super(Distance, self).__init__()
        self.prepare_args(locals(), trainable)

    def __call__(self,
                 img_mask,
                 img_source=None,
                 mode='gray',
                 combine=True,
                 use_alpha=False):
        args = self.update_args()
        return F.distance(img_mask, img_source, mode, combine, use_alpha,
                          **args)


class Emboss(BaseNode):
    '''Wrapper class for atomic node "Emboss".
    '''
    TRAINABLE_INPUT = [
        'intensity', 'light_angle', 'highlight_color', 'shadow_color'
    ]
    DEFAULT_VARS = [0.5, 0.0, [1.0, 1.0, 1.0], [0.0, 0.0, 0.0]]
    HARDTANH_FLAG = [True, True, True, True]
    RELU_FLAG = [False, False, False, False]

    def __init__(self,
                 intensity=None,
                 max_intensity=10.0,
                 light_angle=None,
                 highlight_color=None,
                 shadow_color=None,
                 trainable=[True, True, True, True]):
        super(Emboss, self).__init__()
        self.prepare_args(locals(), trainable)

    def __call__(self, img_in, height_map):
        args = self.update_args()
        return F.emboss(img_in, height_map, **args)


class GradientMap(BaseNode):
    '''Wrapper class for atomic node "Gradient map".
    '''
    TRAINABLE_INPUT = ['anchors']
    DEFAULT_VARS = []
    HARDTANH_FLAG = [True]
    RELU_FLAG = [False]

    def __init__(self,
                 mode='color',
                 use_alpha=False,
                 num_anchors=2,
                 anchors=None,
                 trainable=[True]):
        super(GradientMap, self).__init__()
        assert num_anchors >= 2, "number of anchors should be no less than 2"
        if mode == 'color' and use_alpha:
            default_channel = 5
        elif mode == 'color' and (not use_alpha):
            default_channel = 4
        elif mode == 'gray':
            default_channel = 2

        if anchors is not None:
            if isinstance(anchors, list):
                anchors = np.array(anchors, dtype=np.float32)
            assert num_anchors == anchors.shape[
                0], "number of anchors should match anchors.shape[0]"
            assert anchors.shape[
                1] == default_channel, "the number of anchors' channel does match the mode of color output"
        self.DEFAULT_VARS.append(
            th.linspace(0.0, 1.0,
                        num_anchors).view(num_anchors,
                                          1).repeat(1, default_channel))
        del default_channel
        self.prepare_args(locals(), trainable)

    def __call__(self, img_in, interpolate=True):
        args = self.update_args()
        return F.gradient_map(img_in, interpolate, **args)


class C2G(BaseNode):
    '''Wrapper class for atomic node "grayscale conversion (c2g)".
    '''
    TRAINABLE_INPUT = ['rgba_weights', 'bg']
    DEFAULT_VARS = [[1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 0.0], 1.0]
    HARDTANH_FLAG = [True, True]
    RELU_FLAG = [False, False]

    def __init__(self, rgba_weights=None, bg=None, trainable=[True, True]):
        super(C2G, self).__init__()
        self.prepare_args(locals(), trainable)

    def __call__(self, img_in, flatten_alpha=False):
        args = self.update_args()
        return F.c2g(img_in, flatten_alpha, **args)


class HSL(BaseNode):
    '''Wrapper class for atomic node "HSL".
    '''
    TRAINABLE_INPUT = ['hue', 'saturation', 'lightness']
    DEFAULT_VARS = [0.5, 0.5, 0.5]
    HARDTANH_FLAG = [True, True, True]
    RELU_FLAG = [False, False]

    def __init__(self,
                 hue=None,
                 saturation=None,
                 lightness=None,
                 trainable=[True, True, True]):
        super(HSL, self).__init__()
        self.prepare_args(locals(), trainable)

    def __call__(self, img_in):
        args = self.update_args()
        return F.hsl(img_in, **args)


class Levels(BaseNode):
    '''Wrapper class for atomic node "Levels".
    '''
    TRAINABLE_INPUT = ['in_low', 'in_mid', 'in_high', 'out_low', 'out_high']
    DEFAULT_VARS = [[0.0], [0.5], [1.0], [0.0], [1.0]]
    HARDTANH_FLAG = [True, True, True, True, True]
    RELU_FLAG = [False, False, False, False, False]

    def __init__(self,
                 in_low=None,
                 in_mid=None,
                 in_high=None,
                 out_low=None,
                 out_high=None,
                 trainable=[True, True, True, True, True]):
        super(Levels, self).__init__()
        self.prepare_args(locals(), trainable)

    def __call__(self, img_in):
        args = self.update_args()
        return F.levels(img_in, **args)


class Normal(BaseNode):
    '''Wrapper class for atmoic node "Normal".
    '''
    TRAINABLE_INPUT = ['intensity']
    DEFAULT_VARS = [1.0 / 3.0]
    HARDTANH_FLAG = [True]
    RELU_FLAG = [False]

    def __init__(self, intensity=None, max_intensity=3.0, trainable=[True]):
        super(Normal, self).__init__()
        self.prepare_args(locals(), trainable)

    def __call__(self,
                 img_in,
                 mode='tangent_space',
                 normal_format='dx',
                 use_input_alpha=False,
                 use_alpha=False):
        args = self.update_args()
        return F.normal(img_in, mode, normal_format, use_input_alpha,
                        use_alpha, **args)


class Sharpen(BaseNode):
    '''Wrapper class for atomic node "Sharpen".
    '''
    TRAINABLE_INPUT = ['intensity']
    DEFAULT_VARS = [0.1]
    HARDTANH_FLAG = [True]
    RELU_FLAG = [False]

    def __init__(self, intensity=None, max_intensity=20.0, trainable=[True]):
        super(Sharpen, self).__init__()
        self.prepare_args(locals(), trainable)

    def __call__(self, img_in):
        args = self.update_args()
        return F.sharpen(img_in, **args)


class Transform2d(BaseNode):
    '''Wrapper class for atomic node "Transformation 2D".
    '''
    TRAINABLE_INPUT = ['x1', 'x2', 'x_offset', 'y1', 'y2', 'y_offset', 'matte_color']
    DEFAULT_VARS = [1.0, 0.5, 0.5, 0.5, 1.0, 0.5, [0.0, 0.0, 0.0, 1.0]]
    HARDTANH_FLAG = [True, True, True, True, True, True, True]
    RELU_FLAG = [False, False, False, False, False, False, False]

    def __init__(self,
                 x1=None,
                 x1_max=1.0,
                 x2=None,
                 x2_max=1.0,
                 x_offset=None,
                 x_offset_max=1.0,
                 y1=None,
                 y1_max=1.0,
                 y2=None,
                 y2_max=1.0,
                 y_offset=None,
                 y_offset_max=1.0,
                 matte_color=None,
                 trainable=[True, True, True, True, True, True, True]):
        super(Transform2d, self).__init__()
        self.prepare_args(locals(), trainable)

    def __call__(self,
                 img_in,
                 tile_mode=3,
                 sample_mode='bilinear',
                 mipmap_mode='auto',
                 mipmap_level=0):
        args = self.update_args()
        return F.transform_2d(img_in, tile_mode, sample_mode, mipmap_mode,
                              mipmap_level, **args)


class SpecialTransform(BaseNode):
    '''Wrapper class for special node "Special Transform".
    '''
    TRAINABLE_INPUT = ['scale', 'x_offset', 'y_offset']
    DEFAULT_VARS = [0.25, 0.5, 0.5]
    HARDTANH_FLAG = [True, True, True]
    RELU_FLAG = [False, False, False]

    def __init__(self,
                 scale=None,
                 scale_max=4.0,
                 x_offset=None,
                 x_offset_max=1.0,
                 y_offset=None,
                 y_offset_max=1.0,
                 trainable=[True, True, True]):
        super(SpecialTransform, self).__init__()
        self.prepare_args(locals(), trainable)

    def __call__(self, img_in, tile_mode=3, sample_mode='bilinear'):
        args = self.update_args()
        return F.special_transform(img_in, tile_mode, sample_mode, **args)


class UniformColor(BaseNode):
    '''Wrapper class for atomic node "Uniform Color".
    '''
    TRAINABLE_INPUT = ['rgba']
    DEFAULT_VARS = [[0.0, 0.0, 0.0, 1.0]]
    HARDTANH_FLAG = [True]
    RELU_FLAG = [False]

    def __init__(self, rgba=None, trainable=[True]):
        super(UniformColor, self).__init__()
        self.prepare_args(locals(), trainable)

    def __call__(self,
                 mode='gray',
                 num_imgs=1,
                 res_h=512,
                 res_w=512,
                 use_alpha=False):
        args = self.update_args()
        return F.uniform_color(mode, num_imgs, res_h, res_w, use_alpha, **args)


class Warp(BaseNode):
    '''Wrapper class for atomic node "Warp".
    '''
    TRAINABLE_INPUT = ['intensity']
    DEFAULT_VARS = [0.5]
    HARDTANH_FLAG = [True]
    RELU_FLAG = [False]

    def __init__(self, intensity=None, max_intensity=2.0, trainable=[True]):
        super(Warp, self).__init__()
        self.prepare_args(locals(), trainable)

    def __call__(self, img_in, intensity_mask):
        args = self.update_args()
        return F.warp(img_in, intensity_mask, **args)


class Curvature(BaseNode):
    '''Wrapper class for atomic node "Curvature".
    '''
    TRAINABLE_INPUT = ['emboss_intensity']
    DEFAULT_VARS = [0.1]
    HARDTANH_FLAG = [True]
    RELU_FLAG = [False]

    def __init__(self,
                 emboss_intensity=None,
                 emboss_max_intensity=10.0,
                 trainable=[True]):
        super(Curvature, self).__init__()
        self.prepare_args(locals(), trainable)

    def __call__(self, normal, normal_format='dx'):
        args = self.update_args()
        return F.curvature(normal, normal_format, **args)


class HistogramScan(BaseNode):
    '''Wrapper class for non-atomic node "Histogram Scan".
    '''
    TRAINABLE_INPUT = ['position', 'contrast']
    DEFAULT_VARS = [0.0, 0.0]
    HARDTANH_FLAG = [True, True]
    RELU_FLAG = [False, False]

    def __init__(self, position=None, contrast=None, trainable=[True, True]):
        super(HistogramScan, self).__init__()
        self.prepare_args(locals(), trainable)

    def __call__(self, img_in, invert_position=False):
        args = self.update_args()
        return F.histogram_scan(img_in, invert_position, **args)


class HistogramRange(BaseNode):
    '''Wrapper class for non-atomic node "Histogram Range".
    '''
    TRAINABLE_INPUT = ['ranges', 'position']
    DEFAULT_VARS = [0.5, 0.5]
    HARDTANH_FLAG = [True, True]
    RELU_FLAG = [False, False]

    def __init__(self, ranges=None, position=None, trainable=[True, True]):
        super(HistogramRange, self).__init__()
        self.prepare_args(locals(), trainable)

    def __call__(self, img_in):
        args = self.update_args()
        return F.histogram_range(img_in, **args)


class HistogramSelect(BaseNode):
    '''Wrapper class for non-atomic node "Histogram Select".
    '''
    TRAINABLE_INPUT = ['position', 'ranges', 'contrast']
    DEFAULT_VARS = [0.5, 0.25, 0.0]
    HARDTANH_FLAG = [True, True, True]
    RELU_FLAG = [False, False, False]

    def __init__(self,
                 position=None,
                 ranges=None,
                 contrast=None,
                 trainable=[True, True, True]):
        super(HistogramSelect, self).__init__()
        self.prepare_args(locals(), trainable)

    def __call__(self, img_in):
        args = self.update_args()
        return F.histogram_select(img_in, **args)


class EdgeDetect(BaseNode):
    '''Wrapper class for non-atomic node "Edge Detect".
    '''
    TRAINABLE_INPUT = ['edge_width', 'edge_roundness', 'tolerance']
    DEFAULT_VARS = [2.0 / 16.0, 4.0 / 16.0, 0.0]
    HARDTANH_FLAG = [True, True, True]
    RELU_FLAG = [False, False, False]

    def __init__(self,
                 edge_width=None,
                 max_edge_width=16.0,
                 edge_roundness=None,
                 max_edge_roundness=16.0,
                 tolerance=None,
                 trainable=[True, True, True]):
        super(EdgeDetect, self).__init__()
        self.prepare_args(locals(), trainable)

    def __call__(self, img_in, invert_flag=False):
        args = self.update_args()
        return F.edge_detect(img_in, invert_flag, **args)


class SafeTransform(BaseNode):
    '''Wrapper class for non-atomic node "Safe Transform".
    '''
    TRAINABLE_INPUT = ['offset_x', 'offset_y', 'angle']
    DEFAULT_VARS = [0.0, 0.0, 0.0]
    HARDTANH_FLAG = [True, True, True]
    RELU_FLAG = [False, False, False]

    def __init__(self,
                 offset_x=None,
                 offset_y=None,
                 angle=None,
                 trainable=[True, True, True]):
        super(SafeTransform, self).__init__()
        self.prepare_args(locals(), trainable)

    def __call__(self,
                 img_in,
                 tile=1,
                 tile_safe_rot=False,
                 symmetry='none',
                 tile_mode=3,
                 mipmap_mode='auto',
                 mipmap_level=0):
        args = self.update_args()
        return F.safe_transform(img_in, tile, tile_safe_rot, symmetry,
                                tile_mode, mipmap_mode, mipmap_level, **args)


class BlurHQ(BaseNode):
    '''Wrapper class for non-atomic node "Blur HQ".
    '''
    TRAINABLE_INPUT = ['intensity']
    DEFAULT_VARS = [10.0 / 16.0]
    HARDTANH_FLAG = [True]
    RELU_FLAG = [False]

    def __init__(self, intensity=None, max_intensity=16.0, trainable=[True]):
        super(BlurHQ, self).__init__()
        self.prepare_args(locals(), trainable)

    def __call__(self, img_in, high_quality=False):
        args = self.update_args()
        return F.blur_hq(img_in, high_quality, **args)


class NonUniformBlur(BaseNode):
    '''Wrapper class for non-atomic node "Non-Uniform Blur".
    '''
    TRAINABLE_INPUT = ['intensity', 'anisotropy', 'asymmetry', 'angle']
    DEFAULT_VARS = [0.2, 0.0, 0.0, 0.0]
    HARDTANH_FLAG = [True, True, True, True]
    RELU_FLAG = [False, False, False, False]

    def __init__(self,
                 intensity=None,
                 max_intensity=50.0,
                 anisotropy=None,
                 asymmetry=None,
                 angle=None,
                 trainable=[True, True, True, True]):
        super(NonUniformBlur, self).__init__()
        self.prepare_args(locals(), trainable)

    def __call__(self, img_in, img_mask, samples=4, blades=5):
        args = self.update_args()
        return F.non_uniform_blur(img_in, img_mask, samples, blades, **args)


class Bevel(BaseNode):
    '''Wrapper class for non-atomic node "Bevel".
    '''
    TRAINABLE_INPUT = ['dist', 'smoothing', 'normal_intensity']
    DEFAULT_VARS = [0.75, 0.0, 0.2]
    HARDTANH_FLAG = [True, True, True]
    RELU_FLAG = [False, False, False]

    def __init__(self,
                 dist=None,
                 max_dist=1.0,
                 smoothing=None,
                 max_smoothing=5.0,
                 normal_intensity=None,
                 max_normal_intensity=50.0,
                 trainable=[True, True, True]):
        super(Bevel, self).__init__()
        self.prepare_args(locals(), trainable)

    def __call__(self, img_in, non_uniform_blur_flag=True, use_alpha=False):
        args = self.update_args()
        return F.bevel(img_in, non_uniform_blur_flag, use_alpha, **args)


class SlopeBlur(BaseNode):
    '''Wrapper class for non-atomic node "Slope Blur".
    '''
    TRAINABLE_INPUT = ['intensity']
    DEFAULT_VARS = [10.0 / 16.0]
    HARDTANH_FLAG = [True]
    RELU_FLAG = [False]

    def __init__(self, intensity=None, max_intensity=16.0, trainable=[True]):
        super(SlopeBlur, self).__init__()
        self.prepare_args(locals(), trainable)

    def __call__(self, img_in, img_mask, samples=1, mode='blur'):
        args = self.update_args()
        return F.slope_blur(img_in, img_mask, samples, mode, **args)


class Mosaic(BaseNode):
    '''Wrapper class for non-atomic node "Mosaic".
    '''
    TRAINABLE_INPUT = ['intensity']
    DEFAULT_VARS = [0.5]
    HARDTANH_FLAG = [True]
    RELU_FLAG = [False]

    def __init__(self, intensity=None, max_intensity=1.0, trainable=[True]):
        super(Mosaic, self).__init__()
        self.prepare_args(locals(), trainable)

    def __call__(self, img_in, img_mask, samples=1):
        args = self.update_args()
        return F.mosaic(img_in, img_mask, samples, **args)


class AmbientOcclusion(BaseNode):
    '''Wrapper class for non-atomic node "Ambient Occlusion (Deprecated)".
    '''
    TRAINABLE_INPUT = ['spreading', 'equalizer', 'levels_param']
    DEFAULT_VARS = [0.15, [0.0, 0.0, 0.0], [0.0, 0.5, 1.0]]
    HARDTANH_FLAG = [True, True, True]
    RELU_FLAG = [False, False, False]

    def __init__(self,
                 spreading=None,
                 max_spreading=1.0,
                 equalizer=None,
                 levels_param=None,
                 trainable=[True, True, True]):
        super(AmbientOcclusion, self).__init__()
        self.prepare_args(locals(), trainable)

    def __call__(self, img_in):
        args = self.update_args()
        return F.ambient_occlusion(img_in, **args)


class HBAO(BaseNode):
    '''Wrapper class for non-atomic node "HBAO".
    '''
    TRAINABLE_INPUT = ['depth', 'radius']
    DEFAULT_VARS = [0.1, 1.0]
    HARDTANH_FLAG = [True, True]
    RELU_FLAG = [False, False]

    def __init__(self, depth=None, radius=None, trainable=[True, True]):
        super(HBAO, self).__init__()
        self.prepare_args(locals(), trainable)

    def __call__(self, img_in, quality=4):
        args = self.update_args()
        return F.hbao(img_in, quality, **args)


class Highpass(BaseNode):
    '''Wrapper class for non-atomic node "Highpass".
    '''
    TRAINABLE_INPUT = ['radius']
    DEFAULT_VARS = [6.0 / 64.0]
    HARDTANH_FLAG = [True]
    RELU_FLAG = [False]

    def __init__(self, radius=None, max_radius=64.0, trainable=[True]):
        super(Highpass, self).__init__()
        self.prepare_args(locals(), trainable)

    def __call__(self, img_in):
        args = self.update_args()
        return F.highpass(img_in, **args)


class ChannelMixer(BaseNode):
    '''Wrapper class for non-atomic node "Channal Mixer".
    '''
    TRAINABLE_INPUT = ['red', 'green', 'blue']
    DEFAULT_VARS = [[0.75, 0.5, 0.5, 0.5], [0.5, 0.75, 0.5, 0.5],
                    [0.5, 0.5, 0.75, 0.5]]
    HARDTANH_FLAG = [True, True, True]
    RELU_FLAG = [False, False, False]

    def __init__(self,
                 red=None,
                 green=None,
                 blue=None,
                 trainable=[True, True, True]):
        super(ChannelMixer, self).__init__()
        self.prepare_args(locals(), trainable)

    def __call__(self, img_in, monochrome=False):
        args = self.update_args()
        return F.channel_mixer(img_in, monochrome, **args)


class NormalBlend(BaseNode):
    '''Wrapper class for non-atomic node "Normal Blend".
    '''
    TRAINABLE_INPUT = ['opacity']
    DEFAULT_VARS = [1.0]
    HARDTANH_FLAG = [True]
    RELU_FLAG = [False]

    def __init__(self, opacity=None, trainable=[True]):
        super(NormalBlend, self).__init__()
        self.prepare_args(locals(), trainable)

    def __call__(self, normal_fg, normal_bg, mask=None, use_mask=True):
        args = self.update_args()
        return F.normal_blend(normal_fg, normal_bg, mask, use_mask, **args)


class Mirror(BaseNode):
    '''Wrapper class for non-atomic node "Mirror".
    '''
    TRAINABLE_INPUT = ['offset']
    DEFAULT_VARS = [0.5]
    HARDTANH_FLAG = [True]
    RELU_FLAG = [False]

    def __init__(self, offset=None, trainable=[True]):
        super(Mirror, self).__init__()
        self.prepare_args(locals(), trainable)

    def __call__(self,
                 img_in,
                 mirror_axis='x',
                 invert_axis=False,
                 corner_type='tl'):
        args = self.update_args()
        return F.mirror(img_in, mirror_axis, invert_axis, corner_type, **args)


class HeightToNormal(BaseNode):
    '''Wrapper class for non-atomic node "Height to Normal World Units".
    '''
    TRAINABLE_INPUT = ['surface_size', 'height_depth']
    DEFAULT_VARS = [0.3, 0.16]
    HARDTANH_FLAG = [True, True]
    RELU_FLAG = [False, False]

    def __init__(self,
                 surface_size=None,
                 max_surface_size=1000.0,
                 height_depth=None,
                 max_height_depth=100.0,
                 trainable=[True, True]):
        super(HeightToNormal, self).__init__()
        self.prepare_args(locals(), trainable)

    def __call__(self,
                 img_in,
                 normal_format='gl',
                 sampling_mode='standard',
                 use_alpha=False):
        args = self.update_args()
        return F.height_to_normal_world_units(img_in, normal_format,
                                              sampling_mode, use_alpha, **args)


class NormalToHeight(BaseNode):
    '''Wrapper class for non-atomic node "Normal to Height".
    '''
    TRAINABLE_INPUT = ['relief_balance', 'opacity']
    DEFAULT_VARS = [[0.5, 0.5, 0.5], 0.36]
    HARDTANH_FLAG = [True, True, True]
    RELU_FLAG = [False, False, False]

    def __init__(self,
                 relief_balance=None,
                 opacity=None,
                 max_opacity=1.0,
                 trainable=[True, True]):
        super(NormalToHeight, self).__init__()
        self.prepare_args(locals(), trainable)

    def __call__(self, img_in, normal_format='dx'):
        args = self.update_args()
        return F.normal_to_height(img_in, normal_format, **args)


class MakeItTilePatch(BaseNode):
    '''Wrapper class for non-atomic node "Make It Tile Patch".
    '''
    TRAINABLE_INPUT = [
        'mask_size', 'mask_precision', 'mask_warping', 'pattern_width',
        'pattern_height', 'disorder', 'size_variation', 'rotation',
        'rotation_variation', 'background_color', 'color_variation'
    ]
    DEFAULT_VARS = [
        1.0, 0.5, 0.5, 0.2, 0.2, 0.0, 0.0, 0.5, 0.0, [0.0, 0.0, 0.0, 1.0], 0.0
    ]
    HARDTANH_FLAG = [True for _ in range(len(TRAINABLE_INPUT))]
    RELU_FLAG = [False for _ in range(len(TRAINABLE_INPUT))]

    def __init__(self,
                 mask_size=None,
                 max_mask_size=1.0,
                 mask_precision=None,
                 max_mask_precision=1.0,
                 mask_warping=None,
                 max_mask_warping=100.0,
                 pattern_width=None,
                 max_pattern_width=1000.0,
                 pattern_height=None,
                 max_pattern_height=1000.0,
                 disorder=None,
                 max_disorder=1.0,
                 size_variation=None,
                 max_size_variation=100.0,
                 rotation=None,
                 rotation_variation=None,
                 background_color=None,
                 color_variation=None,
                 trainable=[True] * 11):
        super(MakeItTilePatch, self).__init__()
        self.prepare_args(locals(), trainable)

    def __call__(self, img_in, octave=3, seed=0, use_alpha=False):
        args = self.update_args()
        return F.make_it_tile_patch(img_in, octave, seed, use_alpha, **args)


class MakeItTilePhoto(BaseNode):
    '''Wrapper class for non-atomic node "Make It Tile Photo".
    '''
    TRAINABLE_INPUT = [
        'mask_warping_x', 'mask_warping_y', 'mask_size_x', 'mask_size_y',
        'mask_precision_x', 'mask_precision_y'
    ]
    DEFAULT_VARS = [0.5, 0.5, 0.1, 0.1, 0.5, 0.5]
    HARDTANH_FLAG = [True, True, True, True, True, True]
    RELU_FLAG = [False, False, False, False, False, False]

    def __init__(self,
                 mask_warping_x=None,
                 max_mask_warping_x=100.0,
                 mask_warping_y=None,
                 max_mask_warping_y=100.0,
                 mask_size_x=None,
                 max_mask_size_x=1.0,
                 mask_size_y=None,
                 max_mask_size_y=1.0,
                 mask_precision_x=None,
                 max_mask_precision_x=1.0,
                 mask_precision_y=None,
                 max_mask_precision_y=1.0,
                 trainable=[True, True, True, True, True, True]):
        super(MakeItTilePhoto, self).__init__()
        self.prepare_args(locals(), trainable)

    def __call__(self, img_in):
        args = self.update_args()
        return F.make_it_tile_photo(img_in, **args)


class ReplaceColor(BaseNode):
    '''Wrapper class for non-atomic node "Replace Color".
    '''
    TRAINABLE_INPUT = ['source_color', 'target_color']
    DEFAULT_VARS = [[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]]
    HARDTANH_FLAG = [True, True]
    RELU_FLAG = [False, False]

    def __init__(self,
                 source_color=None,
                 target_color=None,
                 trainable=[True, True]):
        super(ReplaceColor, self).__init__()
        self.prepare_args(locals(), trainable)

    def __call__(self, img_in):
        args = self.update_args()
        return F.replace_color(img_in, **args)


class NormalColor(BaseNode):
    '''Wrapper class for non-atomic node "Normal Color".
    '''
    TRAINABLE_INPUT = ['direction', 'slope_angle']
    DEFAULT_VARS = [0.0, 0.0]
    HARDTANH_FLAG = [True, True]
    RELU_FLAG = [False, False]

    def __init__(self,
                 direction=None,
                 slope_angle=None,
                 trainable=[True, True]):
        super(NormalColor, self).__init__()
        self.prepare_args(locals(), trainable)

    def __call__(self,
                 normal_format='dx',
                 num_imgs=1,
                 res_h=512,
                 res_w=512,
                 use_alpha=False):
        args = self.update_args()
        return F.normal_color(normal_format, num_imgs, res_h, res_w, use_alpha,
                              **args)


class VectorMorph(BaseNode):
    '''Wrapper class for non-atomic node "Vector Morph".
    '''
    TRAINABLE_INPUT = ['amount']
    DEFAULT_VARS = [1.0]
    HARDTANH_FLAG = [True]
    RELU_FLAG = [False]

    def __init__(self, amount=None, max_amount=1.0, trainable=[True]):
        super(VectorMorph, self).__init__()
        self.prepare_args(locals(), trainable)

    def __call__(self, img_in, vector_field=None):
        args = self.update_args()
        return F.vector_morph(img_in, vector_field, **args)


class VectorWarp(BaseNode):
    '''Wrapper class for non-atomic node "Vector Warp".
    '''
    TRAINABLE_INPUT = ['intensity']
    DEFAULT_VARS = [1.0]
    HARDTANH_FLAG = [True]
    RELU_FLAG = [False]

    def __init__(self, intensity=None, max_intensity=1.0, trainable=[True]):
        super(VectorWarp, self).__init__()
        self.prepare_args(locals(), trainable)

    def __call__(self, img_in, vector_map=None, vector_format='dx'):
        args = self.update_args()
        return F.vector_warp(img_in, vector_map, vector_format, **args)


class ContrastLuminosity(BaseNode):
    '''Wrapper class for non-atomic node "Contrast/Luminosity".
    '''
    TRAINABLE_INPUT = ['contrast', 'luminosity']
    DEFAULT_VARS = [0.5, 0.5]
    HARDTANH_FLAG = [True, True]
    RELU_FLAG = [False, False]

    def __init__(self, contrast=None, luminosity=None, trainable=[True, True]):
        super(ContrastLuminosity, self).__init__()
        self.prepare_args(locals(), trainable)

    def __call__(self, img_in):
        args = self.update_args()
        return F.contrast_luminosity(img_in, **args)


class Clamp(BaseNode):
    '''Wrapper class for non-atomic node "Clamp".
    '''
    TRAINABLE_INPUT = ['low', 'high']
    DEFAULT_VARS = [0.0, 1.0]
    HARDTANH_FLAG = [True, True]
    RELU_FLAG = [False, False]

    def __init__(self, low=None, high=None, trainable=[True, True]):
        super(Clamp, self).__init__()
        self.prepare_args(locals(), trainable)

    def __call__(self, img_in, clamp_alpha=True):
        args = self.update_args()
        return F.clamp(img_in, clamp_alpha, **args)


class Pow(BaseNode):
    '''Wrapper class for non-atomic node "Pow".
    '''
    TRAINABLE_INPUT = ['exponent']
    DEFAULT_VARS = [0.4]
    HARDTANH_FLAG = [True]
    RELU_FLAG = [False]

    def __init__(self, exponent=None, max_exponent=10.0, trainable=[True]):
        super(Pow, self).__init__()
        self.prepare_args(locals(), trainable)

    def __call__(self, img_in):
        args = self.update_args()
        return F.pow(img_in, **args)


class AnisotropicBlur(BaseNode):
    '''Wrapper class for non-atomic node "Anisotropic Blur".
    '''
    TRAINABLE_INPUT = ['intensity', 'anisotropy', 'angle']
    DEFAULT_VARS = [10.0 / 16.0, 0.5, 0.0]
    HARDTANH_FLAG = [True, True, True]
    RELU_FLAG = [False, False, False]

    def __init__(self,
                 intensity=None,
                 max_intensity=16.0,
                 anisotropy=None,
                 angle=None,
                 trainable=[True, True, True]):
        super(AnisotropicBlur, self).__init__()
        self.prepare_args(locals(), trainable)

    def __call__(self, img_in, high_quality=True):
        args = self.update_args()
        return F.anisotropic_blur(img_in, high_quality, **args)


class Glow(BaseNode):
    '''Wrapper class for non-atomic node "Glow".
    '''
    TRAINABLE_INPUT = ['glow_amount', 'clear_amount', 'size', 'color']
    DEFAULT_VARS = [0.5, 0.5, 0.5, [1.0, 1.0, 1.0, 1.0]]
    HARDTANH_FLAG = [True, True, True, True]
    RELU_FLAG = [False, False, False, False]

    def __init__(self,
                 glow_amount=None,
                 clear_amount=None,
                 size=None,
                 max_size=20.0,
                 color=None,
                 trainable=[True, True, True, True]):
        super(Glow, self).__init__()
        self.prepare_args(locals(), trainable)

    def __call__(self, img_in):
        args = self.update_args()
        return F.glow(img_in, **args)


class NormalSobel(BaseNode):
    '''Wrapper class for non-atomic node "Normal Sobel"
    '''
    TRAINABLE_INPUT = ['intensity']
    DEFAULT_VARS = [2.0 / 3.0]
    HARDTANH_FLAG = [True, True, True, True]
    RELU_FLAG = [False, False, False, False]

    def __init__(self, intensity=None, max_intensity=3.0, trainable=[True]):
        super(NormalSobel, self).__init__()
        self.prepare_args(locals(), trainable)

    def __call__(self, img_in, normal_format='dx', use_alpha=False):
        args = self.update_args()
        return F.normal_sobel(img_in, normal_format, use_alpha, **args)


class NormalVectorRotation(BaseNode):
    '''Wrapper class for non-atomic node "Normal Vector Rotation"
    '''
    TRAINABLE_INPUT = ['rotation']
    DEFAULT_VARS = [0.5]
    HARDTANH_FLAG = [True]
    RELU_FLAG = [False]

    def __init__(self, rotation=None, rotation_max=1.0, trainable=[True]):
        super(NormalVectorRotation, self).__init__()
        self.prepare_args(locals(), trainable)

    def __call__(self, img_in, img_map=None, normal_format='dx'):
        args = self.update_args()
        return F.normal_vector_rotation(img_in, img_map, normal_format, **args)


class NonSquareTransform(BaseNode):
    '''Wrapper class for non-atomic node "Non-Square Transform"
    '''
    TRAINABLE_INPUT = ['x_offset', 'y_offset', 'rotation', 'background_color']
    DEFAULT_VARS = [0.5, 0.5, 0.5, [0.0, 0.0, 0.0, 1.0]]
    HARDTANH_FLAG = [True, True, True, True]
    RELU_FLAG = [False, False, False, False]

    def __init__(self,
                 x_offset=None, x_offset_max=1.0,
                 y_offset=None, y_offset_max=1.0,
                 rotation=None, rotation_max=1.0,
                 background_color=None,
                 trainable=[True, True, True, True]):
        super(NonSquareTransform, self).__init__()
        self.prepare_args(locals(), trainable)

    def __call__(self, img_in, tile_mode='automatic', x_tile=1, y_tile=1, tile_safe_rotation=True):
        args = self.update_args()
        return F.non_square_transform(img_in, tile_mode, x_tile, y_tile, tile_safe_rotation, **args)


class QuadTransform(BaseNode):
    '''Wrapper class for non-atomic node "Quad Transform".
    '''
    TRAINABLE_INPUT = ['p00', 'p01', 'p10', 'p11', 'background_color']
    DEFAULT_VARS = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0], [0.0, 0.0, 0.0, 1.0]]
    HARDTANH_FLAG = [True, True, True, True]
    RELU_FLAG = [False, False, False, False]

    def __init__(self, p00=None, p01=None, p10=None, p11=None, background_color=None,
                 trainable=[True, True, True, True, True]):
        super(QuadTransform, self).__init__()
        self.prepare_args(locals(), trainable)

    def __call__(self, img_in, culling='f-b', enable_tiling=False, sampling='bilinear'):
        args = self.update_args()
        return F.quad_transform(img_in, culling, enable_tiling, sampling, **args)


class HistogramShift(BaseNode):
    '''Wrapper class for non-atomic node "Histogram Shift".
    '''
    TRAINABLE_INPUT = ['position']
    DEFAULT_VARS = [0.5]
    HARDTANH_FLAG = [True]
    RELU_FLAG = [False]

    def __init__(self, position=None, trainable=[True]):
        super(HistogramShift, self).__init__()
        self.prepare_args(locals(), trainable)

    def __call__(self, img_in):
        args = self.update_args()
        return F.histogram_shift(img_in, **args)


class HeightMapFrequenciesMapper(BaseNode):
    '''Wrapper class for non-atomic node "Height Map Frequencies Mapper"
    '''
    TRAINABLE_INPUT = ['relief']
    DEFAULT_VARS = [0.5]
    HARDTANH_FLAG = [True]
    RELU_FLAG = [False]

    def __init__(self, relief=None, max_relief=32.0, trainable=[True]):
        super(HeightMapFrequenciesMapper, self).__init__()
        self.prepare_args(locals(), trainable)

    def __call__(self, img_in):
        args = self.update_args()
        return F.height_map_frequencies_mapper(img_in, **args)


class LuminanceHighpass(BaseNode):
    '''Wrapper class for non-atomic node "Luminance Highpass"
    '''
    TRAINABLE_INPUT = ['radius']
    DEFAULT_VARS = [6.0/64.0]
    HARDTANH_FLAG = [True]
    RELU_FLAG = [False]

    def __init__(self, radius=None, max_radius=64.0, trainable=[True]):
        super(LuminanceHighpass, self).__init__()
        self.prepare_args(locals(), trainable)

    def __call__(self, img_in):
        args = self.update_args()
        return F.luminance_highpass(img_in, **args)


class ReplaceColorRange(BaseNode):
    '''Wrapper class for non-atomic node "Replace Color Range"
    '''
    TRAINABLE_INPUT = ['source_color', 'target_color', 'source_range', 'threshold']
    DEFAULT_VARS = [[0.501961]*3, [0.501961]*3, 0.5, 1.0]
    HARDTANH_FLAG = [True, True, True, True]
    RELU_FLAG = [False, False, False, False]

    def __init__(self, source_color=None, target_color=None, source_range=None, threshold=None,
                 trainable=[True, True, True, True]):
        super(ReplaceColorRange, self).__init__()
        self.prepare_args(locals(), trainable)

    def __call__(self, img_in):
        args = self.update_args()
        return F.replace_color_range(img_in, **args)


class Dissolve(BaseNode):
    '''Wrapper class for non-atomic node "Dissolve"
    '''
    TRAINABLE_INPUT = ['opacity']
    DEFAULT_VARS = [1.0]
    HARDTANH_FLAG = [True]
    RELU_FLAG = [False]

    def __init__(self, opacity=None, trainable=[True]):
        super(Luminosity, self).__init__()
        super(Dissolve, self).__init__()
        self.prepare_args(locals(), trainable)

    def __call__(self, img_fg=None, img_bg=None, mask=None, alpha_blending=True):
        args = self.update_args()
        return F.dissolve(img_fg, img_bg, mask, alpha_blending, **args)

class ColorBlend(BaseNode):
    '''Wrapper class for non-atomic node "Color Blend"
    '''
    TRAINABLE_INPUT = ['opacity']
    DEFAULT_VARS = [1.0]
    HARDTANH_FLAG = [True]
    RELU_FLAG = [False]

    def __init__(self, opacity=None, trainable=[True]):
        super(ColorBlend, self).__init__()
        self.prepare_args(locals(), trainable)

    def __call__(self, img_fg=None, img_bg=None, mask=None, alpha_blending=True):
        args = self.update_args()
        return F.color_blend(img_fg, img_bg, mask, alpha_blending, **args)


class ColorBurn(BaseNode):
    '''Wrapper class for non-atomic node "Color Burn"
    '''
    TRAINABLE_INPUT = ['opacity']
    DEFAULT_VARS = [1.0]
    HARDTANH_FLAG = [True]
    RELU_FLAG = [False]

    def __init__(self, opacity=None, trainable=[True]):
        super(ColorBurn, self).__init__()
        self.prepare_args(locals(), trainable)

    def __call__(self, img_fg=None, img_bg=None, mask=None, alpha_blending=True):
        args = self.update_args()
        return F.color_burn(img_fg, img_bg, mask, alpha_blending, **args)


class ColorDodge(BaseNode):
    '''Wrapper class for non-atomic node "Color Dodge"
    '''
    TRAINABLE_INPUT = ['opacity']
    DEFAULT_VARS = [1.0]
    HARDTANH_FLAG = [True]
    RELU_FLAG = [False]

    def __init__(self, opacity=None, trainable=[True]):
        super(ColorDodge, self).__init__()
        self.prepare_args(locals(), trainable)

    def __call__(self, img_fg=None, img_bg=None, mask=None, alpha_blending=True):
        args = self.update_args()
        return F.color_dodge(img_fg, img_bg, mask, alpha_blending, **args)


class Difference(BaseNode):
    '''Wrapper class for non-atomic node "Difference"
    '''
    TRAINABLE_INPUT = ['opacity']
    DEFAULT_VARS = [1.0]
    HARDTANH_FLAG = [True]
    RELU_FLAG = [False]

    def __init__(self, opacity=None, trainable=[True]):
        super(Difference, self).__init__()
        self.prepare_args(locals(), trainable)

    def __call__(self, img_fg=None, img_bg=None, mask=None, alpha_blending=True):
        args = self.update_args()
        return F.difference(img_fg, img_bg, mask, alpha_blending, **args)


class LinearBurn(BaseNode):
    '''Wrapper class for non-atomic node "Linear Burn"
    '''
    TRAINABLE_INPUT = ['opacity']
    DEFAULT_VARS = [1.0]
    HARDTANH_FLAG = [True]
    RELU_FLAG = [False]

    def __init__(self, opacity=None, trainable=[True]):
        super(LinearBurn, self).__init__()
        self.prepare_args(locals(), trainable)

    def __call__(self, img_fg=None, img_bg=None, mask=None, alpha_blending=True):
        args = self.update_args()
        return F.linear_burn(img_fg, img_bg, mask, alpha_blending, **args)


class Luminosity(BaseNode):
    '''Wrapper class for non-atomic node "Luminosity"
    '''
    TRAINABLE_INPUT = ['opacity']
    DEFAULT_VARS = [1.0]
    HARDTANH_FLAG = [True]
    RELU_FLAG = [False]

    def __init__(self, opacity=None, trainable=[True]):
        super(Luminosity, self).__init__()
        self.prepare_args(locals(), trainable)

    def __call__(self, img_fg=None, img_bg=None, mask=None, alpha_blending=True):
        args = self.update_args()
        return F.luminosity(img_fg, img_bg, mask, alpha_blending, **args)


class MultiDirWarp(BaseNode):
    '''Wrapper class for non-atomic node "Multi Directional Warp".
    '''
    TRAINABLE_INPUT = ['intensity', 'angle']
    DEFAULT_VARS = [0.5, 0.0]
    HARDTANH_FLAG = [True, True]
    RELU_FLAG = [False, False]

    def __init__(self, intensity=None, max_intensity=20.0, angle=None, trainable=[True, True]):
        super(MultiDirWarp, self).__init__()
        self.prepare_args(locals(), trainable)

    def __call__(self, img_in=None, intensity_mask=None, mode='average', directions=4):
        args = self.update_args()
        return F.multi_dir_warp(img_in, intensity_mask, mode, directions, **args)


class ShapeDropShadow(BaseNode):
    '''Wrapper class for non-atomic node "ShapeDropShadow".
    '''
    TRAINABLE_INPUT = ['angle', 'dist', 'size', 'spread', 'opacity', 'mask_color', 'shadow_color']
    DEFAULT_VARS = [0.25, 0.52, 0.15, 0.0, 0.5, [1.0,1.0,1.0], [0.0,0.0,0.0]]
    HARDTANH_FLAG = [True, True, True, True, True, True, True]
    RELU_FLAG = [False, False, False, False, False, False, False]

    def __init__(self, angle=None, dist=None, max_dist=0.5, size=None, max_size=1.0, spread=None, opacity=None, mask_color=None, shadow_color=None, trainable=[True, True, True, True, True, True, True]):
        super(ShapeDropShadow, self).__init__()
        self.prepare_args(locals(), trainable)

    def __call__(self, img_in, input_is_pre_multiplied=True, pre_multiplied_output=False):
        args = self.update_args()
        return F.shape_drop_shadow(img_in, input_is_pre_multiplied, pre_multiplied_output, **args)


class ShapeGlow(BaseNode):
    '''Wrapper class for non-atomic node "ShapeGlow".
    '''
    TRAINABLE_INPUT = ['width', 'spread', 'opacity', 'mask_color', 'glow_color']
    DEFAULT_VARS = [0.625, 0.0, 1.0, [1.0,1.0,1.0], [1.0,1.0,1.0]]
    HARDTANH_FLAG = [True, True, True, True, True]
    RELU_FLAG = [False, False, False, False, False]

    def __init__(self, width=None, spread=None, opacity=None, mask_color=None, glow_color=None, 
    trainable=[True, True, True, True, True]):
        super(ShapeGlow, self).__init__()
        self.prepare_args(locals(), trainable)

    def __call__(self, img_in, input_is_pre_multiplied=True, pre_multiplied_output=False, mode='soft'):
        args = self.update_args()
        return F.shape_glow(img_in, input_is_pre_multiplied, pre_multiplied_output, mode, **args)


class Swirl(BaseNode):
    '''Wrapper class for non-atomic node "Swirl".
    '''
    TRAINABLE_INPUT = ['amount', 'x1', 'x2', 'x_offset', 'y1', 'y2', 'y_offset']
    DEFAULT_VARS = [0.75, 1.0, 0.5, 0.5, 0.5, 1.0, 0.5]
    HARDTANH_FLAG = [True, True, True, True, True, True, True]
    RELU_FLAG = [False, False, False, False, False, False, False]

    def __init__(self, amount=None, max_amount=16.0, x1=None, x1_max=2.0, x2=None, x2_max=1.0, x_offset=None, 
        x_offset_max=1.0, y1=None, y1_max=1.0, y2=None, y2_max=2.0, y_offset=None, y_offset_max=1.0, 
        trainable=[True, True, True, True, True, True, True]):
        super(Swirl, self).__init__()
        self.prepare_args(locals(), trainable)

    def __call__(self, img_in, tile_mode=3):
        args = self.update_args()
        return F.swirl(img_in, tile_mode, **args)


class CurvatureSobel(BaseNode):
    '''Wrapper class for non-atomic node "Curvature Sobel".
    '''
    TRAINABLE_INPUT = ['intensity']
    DEFAULT_VARS = [0.75]
    HARDTANH_FLAG = [True]
    RELU_FLAG = [False]

    def __init__(self, intensity=None, max_intensity=1.0, trainable=[True]):
        super(CurvatureSobel, self).__init__()
        self.prepare_args(locals(), trainable)

    def __call__(self, img_in, normal_format='dx'):
        args = self.update_args()
        return F.curvature_sobel(img_in, normal_format, **args)


class EmbossWithGloss(BaseNode):
    '''Wrapper class for non-atomic node "Emboss with Gloss".
    '''
    TRAINABLE_INPUT = ['intensity', 'light_angle', 'gloss', 'highlight_color', 'shadow_color']
    DEFAULT_VARS = [0.5, 0.0, 0.625, [1.0,1.0,1.0], [0.0,0.0,0.0]]
    HARDTANH_FLAG = [True, True, True, True, True]
    RELU_FLAG = [False, False, False, False, False]

    def __init__(self, intensity=None, max_intensity=10.0, light_angle=None, gloss=None, max_gloss=1.0, highlight_color=None, shadow_color=None, trainable=[True, True, True, True, True]):
        super(EmbossWithGloss, self).__init__()
        self.prepare_args(locals(), trainable)

    def __call__(self, img_in, height):
        args = self.update_args()
        return F.emboss_with_gloss(img_in, height, **args)


class HeightNormalBlend(BaseNode):
    '''Wrapper class for non-atomic node "Height Normal Blender".
    '''
    TRAINABLE_INPUT = ['normal_intensity']
    DEFAULT_VARS = [0.5]
    HARDTANH_FLAG = [True]
    RELU_FLAG = [False]

    def __init__(self, normal_intensity=None, max_normal_intensity=1.0, trainable=[True]):
        super(HeightNormalBlend, self).__init__()
        self.prepare_args(locals(), trainable)

    def __call__(self, img_height, img_normal, normal_format='dx'):
        args = self.update_args()
        return F.height_normal_blend(img_height, img_normal, normal_format, **args)


class Skew(BaseNode):
    '''Wrapper class for non-atomic node "Skew".
    '''
    TRAINABLE_INPUT = ['amount']
    DEFAULT_VARS = [0.5]
    HARDTANH_FLAG = [True]
    RELU_FLAG = [False]

    def __init__(self, amount=None, max_amount=1.0, trainable=[True]):
        super(Skew, self).__init__()
        self.prepare_args(locals(), trainable)

    def __call__(self, img_in, axis="horizontal", align="top_left"):
        args = self.update_args()
        return F.skew(img_in, axis, align, **args)


class TrapezoidTransform(BaseNode):
    '''Wrapper class for non-atomic node "Trapezoid Transform".
    '''
    TRAINABLE_INPUT = ['top_stretch', 'bottom_stretch', 'bg_color']
    DEFAULT_VARS = [0.5,0.5,[0.0,0.0,0.0,1.0]]
    HARDTANH_FLAG = [True,True,True]
    RELU_FLAG = [False,False,False]

    def __init__(self, top_stretch=None, max_top_stretch=1.0, bottom_stretch=None, max_bottom_stretch=1.0, bg_color=None, trainable=[True, True, True]):
        super(TrapezoidTransform, self).__init__()
        self.prepare_args(locals(), trainable)

    def __call__(self, img_in, sampling="bilinear", tile_mode=3):
        args = self.update_args()
        return F.trapezoid_transform(img_in, sampling, tile_mode, **args)


class ColorToMask(BaseNode):
    '''Wrapper class for non-atomic node "Color to Mask".
    '''
    TRAINABLE_INPUT = ['rgb', 'mask_range', 'mask_softness']
    DEFAULT_VARS = [[0.0,1.0,0.0],0.0,0.0]
    HARDTANH_FLAG = [True,True,True]
    RELU_FLAG = [False,False,False]

    def __init__(self, rgb=None, mask_range=None, mask_softness=None, trainable=[True, True, True]):
        super(ColorToMask, self).__init__()
        self.prepare_args(locals(), trainable)

    def __call__(self, img_in, flatten_alpha=False, keying_type='RGB'):
        args = self.update_args()
        return F.color_to_mask(img_in, flatten_alpha, keying_type, **args)


class DiffMatGraphModule(abc.ABC):
    """Base class to be inherited by a specific procedural graph class implementation.
    """
    def __init__(self,
                 params,
                 params_tp, 
                 exposed_params,
                 trainable_list,
                 ops_list,
                 init_list,
                 call_list,
                 render_params,
                 render_params_trainable=False,
                 optimization_levels=0,
                 device=th.device('cpu')):
        """
        Args:
            params (orderedDict): an ordered dictionary that holds {node name: node parameters} pairs.
            params_tp (orderedDict): an ordered dictionary that holds 
                {node name: to partial functions} pairs.
            exposed_params (orderedDict): an ordered dictionary that holds 
                {exposed parameter name: exposed parameter} pairs.
            trainable_list (list): a list of trainable node parameter lists for all nodes. 
            ops_list (list): a list of wrapper class or functional handles (in order) used by 
                the target procedural graph.
            init_list (list): a list of list where every inner list consists of parameters names for 
                the __init__ function of the corresponding wrapper class.
            call_list (list): a list of list where every inner list consists of parameters names for 
                the __call__ function of the corresponding wrapper class.
            render_params (dict): rendering parameters set for this particular graph.
            render_params_trainable (bool, optional): set the rendering parameters to be trainable. 
                Defaults to False.
            optimization_levels (int, optional): 
                - 0: optimize all node parameters, ignore exposed parameters
                - 1: optimize exposed parameters and node parameters not linked to the exposed parameters
                - 2: optimize only exposed parameters
                Defaults to 0.
            device (torch.device, optional): device on which a torch tensor will be allocated. 
                Defaults to th.device('cpu').
        """        
        self.parameters = []
        self.ops = {}
        self.keys = list(params.keys())
        self.init_params = copy.deepcopy(params)
        self.exposed_params = self.exposed_params_to_torch(copy.deepcopy(exposed_params)) if (exposed_params != {} and exposed_params != None) else None
        self.trainable_list = trainable_list
        self.init_list = init_list
        self.call_list = call_list
        self.call_params = {}
        self.render_params = copy.deepcopy(render_params)
        self.render_params_trainable = render_params_trainable
        self.optimization_levels = optimization_levels

        # modify params based on optimization level
        # replace converted exposed parameters to "to partial" evaluator
        if self.optimization_levels >= 1 and self.exposed_params != None:
            params = swap_tp(params, params_tp, exposed_params)

        # replace trainable to all False
        if self.optimization_levels == 2 and self.exposed_params != None:
            for key, val in params.items():
                params[key]["trainable"] = [False for _ in val]

        # setup renderer params
        for k, _ in render_params.items():
            self.render_params[k] = self.render_params[k].to(device)
        if self.render_params_trainable:
            self.render_params['size'].requires_grad = True
            self.render_params['light_color'].requires_grad = True

        # Construct nodes and parameter dictionaries for '__call__' function
        for idx, key in enumerate(self.keys):
            if not isinstance(ops_list[idx], types.FunctionType):
                switch = lambda val: partial(val.func, self.exposed_params) if isinstance(val, partial) else val
                self.ops[key] = ops_list[idx](*[switch(params[key][val]) for val in init_list[idx]])
                self.call_params[key] = {p_key: params[key][p_key] for p_key in params[key] if p_key in call_list[idx]}
            else:
                self.ops[key] = ops_list[idx]
                self.call_params[key] = params[key]
        self.update_params()

    def exposed_params_to_torch(self, exposed_params):
        # turn exposed parameters to tensors
        trainable_list = exposed_params['trainable']
        del exposed_params['trainable']
        for (key, value), trainable in zip(exposed_params.items(), trainable_list):
            exposed_params[key] = to_tensor(value)
            exposed_params[key].requires_grad = trainable
        return exposed_params

    def update_params(self):
        # add node parameters
        for key, ops in self.ops.items():
            if not isinstance(ops, types.FunctionType):
                self.parameters.extend(ops.parameters)

        # add exposed parameters
        if self.exposed_params != None:
            self.parameters.extend([val for _, val in self.exposed_params.items() if val.requires_grad])
        self.trainable_params = self.parameters
        
        # add rendering parameters to trainable parameters
        if self.render_params_trainable:
            self.trainable_params.append(self.render_params['size'])
            self.trainable_params.append(self.render_params['light_color'])

    def setup_input(self, input_dict):
        self.input_dict = input_dict

    def save_params(self, save_path, iter=None):
        # save parameter as a list
        params_list = []
        
        # save both params and exposed parameters
        # all tp in params should be evaluated and saved as values

        # extract params and evaluate all tp
        for idx, key in enumerate(self.keys):
            if not isinstance(self.ops[key], types.FunctionType):
                op = self.ops[key]
                for op_key in self.trainable_list[idx]:
                    arg = op.args[op_key]() if isinstance(op.args[op_key], partial) else op.args[op_key]
                    arg = arg.cpu().detach().numpy().flatten().astype(np.float32)
                    params_list = np.hstack((params_list, arg))
        params_dict = convert_params_list_to_dict(params_list, self.init_params, self.trainable_list, self.keys, keep_tensor=False)[0]

        if not os.path.exists(save_path):
            os.makedirs(save_path, mode=0o775, exist_ok=True)
        save_path = os.path.join(save_path, "param_dict.pkl") if iter is None else \
            os.path.join(save_path, "param_dict_%d.pkl" % iter)

        with open(save_path, 'wb') as f:
            if self.render_params_trainable:
                th.save([[params_dict, self.exposed_params, self.render_params]], f)
            else:
                th.save([[params_dict, self.exposed_params]], f)


    def eval_op(self, name, *args, **kwargs):
        return self.ops[name](*args, **kwargs, **self.call_params[name])

    @abc.abstractmethod
    def forward(self):
        '''Child class has to define the forward computational graph
        '''
        return


# trainable list for every node
node_dict = {
    "Blend": Blend,
    "Blur": Blur,
    "ChannelShuffle": F.channel_shuffle,
    "Curve": Curve,
    "DBlur": DBlur,
    "DWarp": DWarp,
    "Distance": Distance,
    "Emboss": Emboss,
    "GradientMap": GradientMap,
    "GradientMapDyn": F.gradient_map_dyn,
    "C2G": C2G,
    "HSL": HSL,
    "Levels": Levels,
    "Normal": Normal,
    "Sharpen": Sharpen,
    "Transform2d": Transform2d,
    "SpecialTransform": SpecialTransform,
    "UniformColor": UniformColor,
    "Warp": Warp,
    "L2SRGB": F.linear_to_srgb,
    "SRGB2L": F.srgb_to_linear,
    "Curvature": Curvature,
    "InvertGrayscale": F.invert,
    "InvertColor": F.invert,
    "HistogramScan": HistogramScan,
    "HistogramRange": HistogramRange,
    "HistogramSelect": HistogramSelect,
    "EdgeDetect": EdgeDetect,
    "SafeTransform": SafeTransform,
    "BlurHQ": BlurHQ,
    "NonUniformBlur": NonUniformBlur,
    "Bevel": Bevel,
    "SlopeBlur": SlopeBlur,
    "Mosaic": Mosaic,
    "AutoLevels": F.auto_levels,
    "AmbientOcclusion": AmbientOcclusion,
    "HBAO": HBAO,
    "Highpass": Highpass,
    "NormalNormalize": F.normal_normalize,
    "ChannelMixer": ChannelMixer,
    "NormalCombine": F.normal_combine,
    "HeightToNormal": HeightToNormal,
    "NormalToHeight": NormalToHeight,
    "CurvatureSmooth": F.curvature_smooth,
    "MultiSwitch": F.multi_switch,
    "RGBASplit": F.rgba_split,
    "RGBAMerge": F.rgba_merge,
    "MakeItTilePatch": MakeItTilePatch,
    "MakeItTilePhoto": MakeItTilePhoto,
    "PbrConverter": F.pbr_converter,
    "AlphaSplit": F.alpha_split,
    "AlphaMerge": F.alpha_merge,
    "Switch": F.switch,
    "NormalBlend": NormalBlend,
    "Mirror": Mirror,
    "VectorMorph": VectorMorph,
    "VectorWarp": VectorWarp,
    "Passthrough": F.passthrough,
    "ReplaceColor": ReplaceColor,
    "NormalColor": NormalColor,
    "ContrastLuminosity": ContrastLuminosity,
    "P2S": F.p2s,
    "S2P": F.s2p,
    "Clamp": Clamp,
    "Pow": Pow,
    "Quantize": F.quantize,
    "AnisotropicBlur": AnisotropicBlur,
    "Glow": Glow,
    "Car2Pol": F.car2pol,
    "Pol2Car": F.pol2car,
    "NormalSobel": NormalSobel,
    "NormalVectorRotation": NormalVectorRotation,
    "NonSquareTransform": NonSquareTransform,
    "QuadTransform": QuadTransform,
    "ChrominanceExtract": F.chrominance_extract,
    "HistogramShift": HistogramShift,
    "HeightMapFrequenciesMapper": HeightMapFrequenciesMapper,
    "LuminanceHighpass": LuminanceHighpass,
    "ReplaceColorRange": ReplaceColorRange,
    "Dissolve": Dissolve,
    "ColorBlend": ColorBlend,
    "ColorBurn": ColorBurn,
    "ColorDodge": ColorDodge,
    "Difference": Difference,
    "LinearBurn": LinearBurn,
    "Luminosity": Luminosity,
    "MultiDirWarp": MultiDirWarp,
    "ShapeDropShadow": ShapeDropShadow,
    "ShapeGlow": ShapeGlow,
    "Swirl": Swirl,
    "CurvatureSobel": CurvatureSobel,
    "EmbossWithGloss": EmbossWithGloss,
    "HeightNormalBlend": HeightNormalBlend,
    "Skew": Skew,
    "TrapezoidTransform": TrapezoidTransform,
    "ColotToMask": ColorToMask,
    "FacingNormal": F.facing_normal,
    "NormalInvert": F.normal_invert,
    "C2GAdvanced": F.c2g_advanced,
}


def get_nodes(node_list):
    """Generate lists necessary for initializing the graph class

    Args:
        node_list (list): a list of all nodes classes' names (ordered) used by a graph

    Returns:
        List : a list of trainable node parameter lists for all nodes. 
        List : a list of wrapper class or functional handles (in order) used by 
            the target procedural graph.
        List : a list of list where every inner list consists of parameters names for 
            the __init__ function of the corresponding wrapper class.
        List : a list of list where every inner list consists of parameters names for 
            the __call__ function of the corresponding wrapper class.
    """
    trainable_list = []
    ops_list = []
    init_list = []
    call_list = []
    for ops in node_list:
        ops_list.append(node_dict[ops])
        if not isinstance(node_dict[ops], types.FunctionType):
            trainable_list.append(node_dict[ops].TRAINABLE_INPUT)
            init_list.append(
                inspect.getfullargspec(node_dict[ops].__init__).args[1:])
            call_list.append(
                inspect.getfullargspec(node_dict[ops].__call__).args[1:])
        else:
            trainable_list.append([])
            init_list.append([])
            call_list.append([])

    return trainable_list, ops_list, init_list, call_list
