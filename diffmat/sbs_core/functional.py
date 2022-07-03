import os
import sys
import time
import math
import numpy as np
import torch as th
import scipy.ndimage as spi
import matplotlib.pyplot as plt

from diffmat.sbs_core.util import input_check, roll_row, roll_col, normalize, color_input_check, grayscale_input_check, to_zero_one, to_tensor

'''
Atomic functions:
    - Blend
    - Blur
    - Channels Shuffle
    - Curve
    - Directional Blur
    - Directional Warp
    - Distance
    - Emboss
    - Gradient Dynamic
    - Gradient Map
    - Grayscale Conversion (c2g)
    - HSL
    - Levels
    - Normal
    - Sharpen
    - Transformation 2D
    - Uniform Color
    - Warp

Non-atomic functions:
    - Alpha Merge
    - Alpha split
    - Ambient Occlusion (legacy)
    - Anisotropic Blur*
    - Auto Levels
    - Bevel
    - Blur HQ*
    - Cartesian to Polar*
    - Channel Mixer
    - Chrominance Extract
    - Clamp*
    - Color to Mask
    - Contrast/Luminosity*
    - Convert to Linear*
    - Convert to sRGB*
    - Curvature
    - Curvature Smooth
    - Difference 
    - Dissolve
    - Dot (passthrough)
    - Edge Detect
    - Emboss with Gloss
    - Facing Normal
    - Glow*
    - Grayscale Conversion Advanced
    - HBAO
    - Height Map Frequencies Mapper
    - Height Normal Blender
    - Height to Normal World Units
    - Highpass*
    - Histogram Range
    - Histogram Scan
    - Histogram Scan Non-Uniform
    - Histogram Select
    - Histogram Shift
    - Invert*
    - Linear Burn
    - Luminance Highpass
    - Luminosity
    - Make it Tile Patch*
    - Make it Tile Photo*
    - Mirror*
    - Mosaic*
    - Multi Switch*
    - Non-Square Transform*
    - Non-uniform Blur*
    - Normal Blend
    - Normal Color
    - Normal Combine
    - Normal Invert
    - Normal Normalize
    - Normal Sobel
    - Normal to Height
    - Normal Vector Rotation
    - PBR Converter
    - Polar to Cartesian*
    - Pow*
    - Pre-Multiplied to Straight
    - Quad Transform*
    - Quantize*
    - RGBA Merge
    - RGBA Split
    - Replace Color
    - Replace Color Range
    - Safe Transform*
    - Skew*
    - Slope Blur*
    - Straight to Pre-Multiplied
    - Switch*
    - Trapezoid Transform*
    - Vector Morph*
    - Vector Warp*
'''

@input_check(0)
def blend(img_fg=None, img_bg=None, blend_mask=None, blending_mode='copy', cropping=[0.0,1.0,0.0,1.0], opacity=1.0):
    """Atomic function: Blend (https://docs.substance3d.com/sddoc/blending-modes-description-132120605.html)

    Args:
        img_fg (tensor, optional): Foreground image (G or RGB(A)). Defaults to None.
        img_bg (tensor, optional): Background image (G or RGB(A)). Defaults to None.
        blend_mask (tensor, optional): Blending mask (G only). Defaults to None.
        blending_mode (str, optional): 
            copy|add|subtract|multiply|add_sub|max|min|divide|switch|overlay|screen|soft_light. 
            Defaults to 'copy'.
        cropping (list, optional): [left, right, top, bottom]. Defaults to [0.0,1.0,0.0,1.0].
        opacity (float, optional): Alpha mask. Defaults to 1.0.

    Returns:
        Tensor: Blended image.
    """
    if img_fg is not None:
        img_fg = to_tensor(img_fg)
    else:
        img_fg = to_tensor(0.0)
        img_fg_alpha = 0.0
    if img_bg is not None:
        img_bg = to_tensor(img_bg)
        if len(img_fg.shape):
            assert img_fg.shape[1] == img_bg.shape[1], 'foreground and background image type does not match' 
    else:
        img_bg = to_tensor(0.0)
        img_bg_alpha = 0.0
    if blend_mask is not None:
        blend_mask = to_tensor(blend_mask)
        grayscale_input_check(blend_mask, 'blend mask')
        weight = blend_mask * opacity
    else:
        weight = opacity

    # compute output alpha channel
    use_alpha = False
    if len(img_fg.shape) and img_fg.shape[1] == 4:
        img_fg_alpha = img_fg[:,[3],:,:]
        img_fg = img_fg[:,:3,:,:]
        use_alpha = True
    if len(img_bg.shape) and img_bg.shape[1] == 4:
        img_bg_alpha = img_bg[:,[3],:,:]
        img_bg = img_bg[:,:3,:,:]
        use_alpha = True
    if use_alpha:
        if blending_mode == 'switch':
            img_out_alpha = img_fg_alpha * weight + img_bg_alpha * (1.0 - weight)
        else:
            weight = weight * img_fg_alpha
            img_out_alpha = weight + img_bg_alpha * (1.0 - weight)

    clamp_max = 1.0
    clamp_min = 0.0
    if blending_mode == 'copy': 
        img_out = th.clamp(img_fg * weight + img_bg * (1.0 - weight), clamp_min, clamp_max)
    elif blending_mode == 'add': 
        img_out = th.clamp(img_fg * weight + img_bg, clamp_min, clamp_max)
    elif blending_mode == 'subtract': 
        img_out = th.clamp(-img_fg * weight + img_bg, clamp_min, clamp_max)
    elif blending_mode == 'multiply': 
        img_fg = img_fg * weight + (1.0 - weight)
        img_out = th.clamp(img_fg * img_bg, clamp_min, clamp_max)
    elif blending_mode == 'add_sub': 
        img_fg = (img_fg - 0.5) * 2.0
        img_out = th.clamp(img_fg * weight + img_bg, clamp_min, clamp_max)
    elif blending_mode == 'max': 
        img_fg = th.clamp(th.max(img_fg, img_bg), clamp_min, clamp_max)
        img_out = img_fg * weight + img_bg * (1.0 - weight)
    elif blending_mode == 'min': 
        img_fg = th.clamp(th.min(img_fg, img_bg), clamp_min, clamp_max)
        img_out = img_fg * weight + img_bg * (1.0 - weight)
    elif blending_mode == 'divide': 
        img_fg = img_bg / (img_fg + 1e-15)
        img_out = th.clamp(img_fg * weight + img_bg * (1.0 - weight), clamp_min, clamp_max)
    elif blending_mode == 'switch':
        if blend_mask is None and weight == 1.0:
            img_out = img_fg
        elif blend_mask is None and weight == 0.0:
            img_out = img_bg
        else:
            img_out = th.clamp(img_fg * weight + img_bg * (1.0 - weight), clamp_min, clamp_max)
    elif blending_mode == 'overlay': 
        img_out = th.zeros_like(img_fg)
        mask = img_bg < 0.5
        img_out[mask] = th.clamp(2.0 * img_fg * img_bg, clamp_min, clamp_max)[mask]
        img_out[~mask] = th.clamp(1.0 - 2.0 * (1.0 - img_fg) * (1.0 - img_bg), clamp_min, clamp_max)[~mask]
        img_out = img_out * weight + img_bg * (1.0 - weight)
    elif blending_mode == 'screen': 
        img_fg = th.clamp(1.0 - (1.0 - img_fg) * (1.0 - img_bg), clamp_min, clamp_max)
        img_out = img_fg * weight + img_bg * (1.0 - weight)
    elif blending_mode == 'soft_light':
        img_fg = th.clamp(img_bg + (img_fg * 2.0 - 1.0) * img_bg * (1.0 - img_bg), clamp_min, clamp_max)
        img_out = img_fg * weight + img_bg * (1.0 - weight)
    else:
        raise 'unknown blending_mode'
    
    # apply cropping
    if cropping[0] == 0.0 and cropping[1] == 1.0 and cropping[2] == 0.0 and cropping[3] == 1.0:
        img_out_crop = img_out
    else:    
        start_row = math.floor(cropping[2] * img_out.shape[2])
        end_row = math.floor(cropping[3] * img_out.shape[2])
        start_col = math.floor(cropping[0] * img_out.shape[3])
        end_col = math.floor(cropping[1] * img_out.shape[3])
        img_out_crop = img_bg.clone()
        img_out_crop[:,:,start_row:end_row, start_col:end_col] = img_out[:,:,start_row:end_row, start_col:end_col]

    if use_alpha == True:
        img_out_crop = th.cat([img_out_crop, img_out_alpha], dim=1)

    return img_out_crop


@input_check(1)
def blur(img_in, intensity=0.5, max_intensity=20.0):
    """Atomic function: Blur (simple box blur: https://docs.substance3d.com/sddoc/blur-172825121.html)

    Args:
        img_in (tensor): Input image.
        intensity (float, optional): Normalized box filter side length [0,1], 
            need to multiply max_intensity. Defaults to 0.5.
        max_intensity (float, optional): Maximum blur intensity. Defaults to 20.0.

    Returns:
        Tensor: Blurred image.
    """
    num_group = img_in.shape[1]
    img_size = img_in.shape[2]
    intensity = to_tensor(intensity * img_size / 256.0) * max_intensity
    kernel_len = (th.ceil(intensity + 0.5) * 2.0 - 1.0).type(th.int)
    if kernel_len <= 1:
        return img_in.clone()
    # create 2d kernel
    blur_idx = -th.abs(th.linspace(-kernel_len//2, kernel_len//2, kernel_len))
    blur_1d  = th.clamp(blur_idx + intensity + 0.5, 0.0, 1.0)
    blur_row = blur_1d.view(1,1,1,kernel_len).expand(num_group,1,1,kernel_len)
    blur_col = blur_1d.view(1,1,kernel_len,1).expand(num_group,1,kernel_len,1)
    # manually pad input image
    p2d = [kernel_len//2, kernel_len//2, kernel_len//2, kernel_len//2]
    img_in  = th.nn.functional.pad(img_in, p2d, mode='circular')
    # perform depth-wise convolution without implicit padding
    img_out = th.nn.functional.conv2d(img_in, blur_row, groups=num_group, padding = 0)
    img_out = th.nn.functional.conv2d(img_out, blur_col, groups=num_group, padding = 0)
    img_out = th.clamp(img_out / (intensity ** 2 * 4.0), to_tensor(0.0), to_tensor(1.0))
    return img_out


@input_check(1)
def channel_shuffle(img_in, img_in_aux=None, use_alpha=False, shuffle_idx=[0,1,2,3]):
    """Atomic functionL: Channel Shuffle (https://docs.substance3d.com/sddoc/channel-shuffle-172825126.html)

    Args:
        img_in (tensor): Input image.
        img_in_aux (tensor, optional): Auxiliary input image for swapping channels between images. Defaults to None.
        use_alpha (bool, optional): Enable the alpha channel. Defaults to False.
        shuffle_idx (list, optional): Shuffle pattern containing indices of source channels. Defaults to [0,1,2,3].

    Returns:
        Tensor: Channel shuffled images.
    """
    # tensor conversion
    shuffle_idx = th.tensor(shuffle_idx)

    # img_in_aux type and shape check
    if img_in_aux is not None:
        img_in_aux = to_tensor(img_in_aux)
        assert len(img_in_aux.shape) == 4, 'img_in_aux is not a 4D Pytorch tensor'
        assert img_in.shape[0] == img_in_aux.shape[0] and \
                img_in.shape[2] == img_in_aux.shape[2] and \
                img_in.shape[3] == img_in_aux.shape[3], 'the shape of img_in and img_in_aux does not match'

    if img_in.shape[1] == 1:
        img_in = img_in.expand(img_in.shape[0], 4 if use_alpha else 3, img_in.shape[2], img_in.shape[3])

    if img_in_aux.shape[1] == 1:
        img_in_aux = img_in_aux.expand(img_in_aux.shape[0], 4 if use_alpha else 3, img_in_aux.shape[2], img_in_aux.shape[3])

    # output has the same shape as the input
    img_out = img_in.clone()
    for i in range(4 if use_alpha else 3):
        if i != shuffle_idx[i]:
            if shuffle_idx[i] <= 3 and shuffle_idx[i] >= 0:
                # check input types
                if img_in.shape[1] == 1:
                    out = img_in 
                elif img_in.shape[1] == 3:
                    out = 1.0 if shuffle_idx[i] == 3 else img_in[:,shuffle_idx[i],:,:]
                elif img_in.shape[1] == 4:
                    out = img_in[:,shuffle_idx[i],:,:]
            elif shuffle_idx[i] >=4 and shuffle_idx[i] <=7 and img_in_aux is not None:
                # check input types
                if img_in.shape[1] == 1:
                    out = img_in_aux
                elif img_in.shape[1] == 3:
                    out = 1.0 if shuffle_idx[i] == 7 else img_in_aux[:,shuffle_idx[i] - 3,:,:]
                elif img_in.shape[1] == 4:
                    out = img_in_aux[:,shuffle_idx[i] - 3,:,:]
            else:
                raise ValueError('shuffle idx is invalid')

            img_out[:,i,:,:] = out 

    return img_out


@input_check(1)
def curve(img_in, num_anchors=2, anchors=None):
    """Atomic function: Curve (https://docs.substance3d.com/sddoc/curve-172825175.html)

    Args:
        img_in (tensor): Input image.
        num_anchors (int, optional): Number of anchors. Defaults to 2.
        anchors (list, optional): Anchors. Defaults to None.

    Returns:
        Tensor: Curved image.

    TODO:
        - support per channel adjustment
        - support alpha adjustment
    """
    if img_in.shape[1] == 4:
        img_in_alpha = img_in[:,3,:,:].unsqueeze(1)
        img_in = img_in[:,:3,:,:]
        use_alpha = True
    else:
        use_alpha = False

    # Process input anchor table
    if anchors is None:
        anchors = th.stack([th.zeros(6), th.ones(6)])
    else:
        anchors = to_tensor(anchors)
        assert anchors.shape == (num_anchors, 6), 'shape of anchors is not [num_anchors, 6]'
        # Sort input anchors based on [:,0] in ascendng order
        anchors = anchors[th.argsort(anchors[:, 0]), :]

    # Determine the size of the sample grid
    res_h, res_w = img_in.shape[2], img_in.shape[3]
    sample_size_t = max(res_h, res_w) * 2
    sample_size_x = sample_size_t

    # First sampling pass (parameter space)
    p1 = anchors[:-1, :2].t()
    p2 = anchors[:-1, 4:].t()
    p3 = anchors[1:, 2:4].t()
    p4 = anchors[1:, :2].t()
    A = p4 - p1 + (p2 - p3) * 3.0
    B = (p1 + p3 - p2 * 2.0) * 3.0
    C = (p2 - p1) * 3.0
    D = p1

    t = th.linspace(0.0, 1.0, sample_size_t)
    inds = th.sum((t.unsqueeze(0) >= anchors[:, [0]]), 0)
    inds = th.clamp(inds - 1, 0, num_anchors - 2)
    t_ = (t - p1[0, inds]) / (p4[0, inds] - p1[0, inds] + 1e-8)
    bz_t = ((A[:, inds] * t_ + B[:, inds]) * t_ + C[:, inds]) * t_ + D[:, inds]
    bz_t = th.where((t <= p1[0, 0]).unsqueeze(0), th.stack([t, p1[1, 0].expand_as(t)]), bz_t)
    bz_t = th.where((t >= p4[0, -1]).unsqueeze(0), th.stack([t, p4[1, -1].expand_as(t)]), bz_t)

    # Second sampling pass (x space)
    x = th.linspace(0.0, 1.0, sample_size_x)
    inds = th.sum((x.unsqueeze(0) >= bz_t[0].view(sample_size_t, 1)), 0)
    inds = th.clamp(inds - 1, 0, sample_size_t - 2)
    x_ = (x - bz_t[0, inds]) / (bz_t[0, inds + 1] - bz_t[0, inds] + 1e-8)
    bz_x = bz_t[1, inds] * (1 - x_) + bz_t[1, inds + 1] * x_

    # Third sampling pass (color space)
    bz_x = bz_x.expand(img_in.shape[0] * img_in.shape[1], 1, 1, sample_size_x)
    col_grid = img_in.view(img_in.shape[0] * img_in.shape[1], res_h, res_w, 1) * 2.0 - 1.0
    sample_grid = th.cat([col_grid, th.zeros_like(col_grid)], 3)
    img_out = th.nn.functional.grid_sample(bz_x, sample_grid, align_corners=True)
    img_out = img_out.view_as(img_in)

    # Append the original alpha channel
    if use_alpha:
        img_out = th.cat([img_out, img_in_alpha], dim=1)

    return img_out

# update
@input_check(1)
def d_blur(img_in, intensity=0.5, max_intensity=20.0, angle=0.0):
    """Atomic function: Directional Blur (1d line blur filter: https://docs.substance3d.com/sddoc/directional-blur-172825181.html)

    Args:
        img_in (Tensor): Input image.
        intensity (float, optional): Normalized filter length. Defaults to 0.5.
        max_intensity (float, optional): Maximum blur intensity. Defaults to 20.0.
        angle (float, optional): Filter angle. Defaults to 0.0.

    Returns:
        Tensor: Directional blurred image.
    """
    num_group = img_in.shape[1]
    num_row = img_in.shape[2]
    num_col = img_in.shape[3]
    gs_interp_mode = 'bilinear'
    gs_padding_mode = 'zeros'
    res_angle = th.remainder(to_tensor(angle), 0.5)
    angle = to_tensor(angle) * np.pi * 2.0
    intensity = to_tensor(intensity * num_row / 512) * max_intensity
    if intensity <= 0.25:
        return img_in.clone()

    # compute convolution kernel
    kernel_len = (th.ceil(2*intensity+0.5)*2-1).type(th.int)
    blur_idx = -th.abs(th.linspace(-kernel_len//2, kernel_len//2, kernel_len))
    blur_1d  = th.nn.functional.hardtanh(blur_idx + intensity*2.0 + 0.5, 0.0, 1.0)
    kernel_1d  = blur_1d / th.sum(blur_1d)
    kernel_1d  = kernel_1d.view(1,1,1,kernel_len).expand(num_group, 1, 1, kernel_len)

    # Special case for small intensity
    ab_cos = th.abs(th.cos(angle))
    ab_sin = th.abs(th.sin(angle))
    sc_max, sc_min = th.max(ab_cos, ab_sin), th.min(ab_cos, ab_sin)
    dist_1 = (intensity * 2.0 - 0.5) * sc_max
    dist_2 = (intensity * 2.0 - 0.5) * sc_min
    if dist_1 <= 1.0:
        kernel_len = 3

    # circularly pad the image & update num_row, num_col
    conv_p2d = [kernel_len//2, kernel_len//2, kernel_len//2, kernel_len//2]
    img_in  = th.nn.functional.pad(img_in, conv_p2d, mode='circular')

    # Compute directional motion blur in different algorithms
    # Special condition (3x3 kernel) when intensity is small
    if dist_1 <= 1.0:
        k_00 = th.where(res_angle < 0.25, dist_2, to_tensor(0.0))
        k_01 = th.where(res_angle > 0.125 and res_angle < 0.375, dist_1 - dist_2, to_tensor(0.0))
        k_02 = th.where(res_angle > 0.25, dist_2, to_tensor(0.0))
        k_10 = th.where(res_angle < 0.125 or res_angle > 0.375, dist_1 - dist_2, to_tensor(0.0))
        k_11 = to_tensor(1.0)
        kernel_2d = th.stack([th.stack([k_00, k_01, k_02]),
                              th.stack([k_10, k_11, k_10]),
                              th.stack([k_02, k_01, k_00])])
        kernel_2d = (kernel_2d / th.sum(kernel_2d)).expand(num_group, 1, 3, 3)
        img_out = th.nn.functional.conv2d(img_in, kernel_2d, groups=num_group, padding=0)
        img_out = th.clamp(img_out, to_tensor(0.0), to_tensor(1.0))
    # Compute kernel from rotated small kernels
    elif intensity <= 1.1:
        assert kernel_len == 5
        kernel_2d = th.zeros(kernel_len, kernel_len)
        kernel_2d[[kernel_len//2],:] = blur_1d
        kernel_2d = kernel_2d.expand(num_group, 1, kernel_len, kernel_len)
        sin_res_angle = th.sin(res_angle * np.pi * 2.0)
        cos_res_angle = th.cos(res_angle * np.pi * 2.0)
        kernel_2d = transform_2d(kernel_2d, tile_mode=0, mipmap_mode='manual', x1=to_zero_one(cos_res_angle), x2=to_zero_one(-sin_res_angle),
                                 y1=to_zero_one(sin_res_angle), y2=to_zero_one(cos_res_angle))
        kernel_2d = kernel_2d / th.sum(kernel_2d[0,0])
        img_out = th.nn.functional.conv2d(img_in, kernel_2d, groups=num_group, padding=0)
        img_out = th.clamp(img_out, to_tensor(0.0), to_tensor(1.0))
    # Rotation -> convolution -> reversed rotation
    else:
        # compute rotation padding
        num_row = img_in.shape[2]
        num_col = img_in.shape[3]
        num_row_new = th.abs(th.cos(-angle))*num_row + th.abs(th.sin(-angle))*num_col
        num_col_new = th.abs(th.cos(-angle))*num_col + th.abs(th.sin(-angle))*num_row
        row_pad = ((num_row_new - num_row) / 2.0).ceil().type(th.int)
        col_pad = ((num_col_new - num_col) / 2.0).ceil().type(th.int)
        rot_p2d  = [row_pad, row_pad, col_pad, col_pad]
        img_in  = th.nn.functional.pad(img_in, rot_p2d, mode='constant')
        num_row = img_in.shape[2]
        num_col = img_in.shape[3]
        # rotate the image
        row_grid, col_grid = th.meshgrid(th.linspace(0, num_row-1, num_row), th.linspace(0, num_col-1, num_col))
        row_grid = (row_grid + 0.5) / num_row * 2.0 - 1.0
        col_grid = (col_grid + 0.5) / num_col * 2.0 - 1.0
        col_grid_rot = th.cos(-angle) * col_grid + th.sin(-angle) * row_grid
        row_grid_rot = -th.sin(-angle) * col_grid + th.cos(-angle) * row_grid
        # sample grid
        sample_grid = th.stack([col_grid_rot, row_grid_rot], 2).expand(img_in.shape[0], num_row, num_col, 2)
        img_rot = th.nn.functional.grid_sample(img_in, sample_grid, mode=gs_interp_mode, padding_mode=gs_padding_mode, align_corners=False)
        # perform depth-wise convolution without implicit padding
        img_blur = th.nn.functional.conv2d(img_rot, kernel_1d, groups=num_group, padding = 0)
        # pad back the columns comsumed by conv2d
        img_blur = th.nn.functional.pad(img_blur, [conv_p2d[0], conv_p2d[1], 0, 0], mode='constant')
        # unrotate the image
        col_grid_unrot = th.cos(angle) * col_grid + th.sin(angle) * row_grid
        row_grid_unrot = -th.sin(angle) * col_grid + th.cos(angle) * row_grid
        sample_grid = th.stack([col_grid_unrot, row_grid_unrot], 2).expand(img_in.shape[0], num_row, num_col, 2)
        img_out = th.nn.functional.grid_sample(img_blur, sample_grid, mode=gs_interp_mode, padding_mode=gs_padding_mode, align_corners=False)
        # remove padding
        full_pad = th.tensor(conv_p2d) + th.tensor(rot_p2d)
        img_out = img_out[:,:,full_pad[0]:img_out.shape[2]-full_pad[1], full_pad[2]:img_out.shape[3]-full_pad[3]]

    return img_out


@input_check(2)
def d_warp(img_in, intensity_mask, intensity=0.5, max_intensity=20.0, angle = 0.0):
    """Atomic function: Directional Warp (https://docs.substance3d.com/sddoc/directional-warp-172825190.html)

    Args:
        img_in (tensor): Input image.
        intensity_mask (tensor): Intensity mask for computing displacement (G only).
        intensity (float, optional): Normalized intensity_mask multiplier. Defaults to 0.5.
        max_intensity (float, optional): Maximum intensity_mask multiplier. Defaults to 20.0.
        angle (float, optional): Direction to shift, 0 degree points to the left. Defaults to 0.0.

    Returns:
        Tensor: Directional warpped image.
    """
    grayscale_input_check(intensity_mask, 'input mask')

    # tensor conversion
    angle = to_tensor(angle) * np.pi * 2.0
    intensity = to_tensor(intensity) * max_intensity
    gs_interp_mode = 'bilinear'
    gs_padding_mode = 'zeros'
    num_row = img_in.shape[2]
    num_col = img_in.shape[3]
    row_scale = num_row / 256.0 # magic number
    col_scale = num_col / 256.0 # magic number
    intensity_mask = intensity_mask * intensity
    row_shift = intensity_mask * th.sin(angle) * row_scale
    col_shift = intensity_mask * th.cos(angle) * col_scale
    row_grid, col_grid = th.meshgrid(th.linspace(0, num_row-1, num_row), th.linspace(0, num_col-1, num_col))
    # mod the index to behavior as tiling 
    row_grid = th.remainder((row_grid + row_shift + 0.5) / num_row * 2.0, 2.0) - 1.0
    col_grid = th.remainder((col_grid + col_shift + 0.5) / num_col * 2.0, 2.0) - 1.0
    row_grid = row_grid * num_row / (num_row + 2)
    col_grid = col_grid * num_col / (num_col + 2)
    # sample grid
    sample_grid = th.cat([col_grid, row_grid], 1).permute(0,2,3,1).expand(intensity_mask.shape[0], num_row, num_col, 2)
    in_pad = th.nn.functional.pad(img_in, [1, 1, 1, 1], mode='circular')
    img_out = th.nn.functional.grid_sample(in_pad, sample_grid, mode=gs_interp_mode, padding_mode=gs_padding_mode, align_corners=False)
    return img_out

@input_check(1)
def distance(img_mask, img_source=None, mode='gray', combine=True, use_alpha=False, dist=10.0 / 256.0, max_dist=256.0):
    """Atomic function: Distance (https://docs.substance3d.com/sddoc/distance-172825194.html)

    Args:
        img_mask (tensor): A mask image that will be binarized by a threshold of 0.5 (G only).
        img_source (tensor, optional): Colors will be fetched using img_mask. Defaults to None.
        mode (str, optional): 'gray' or 'color', determine the format of output when img_source
            is not provided. Defaults to 'gray'.
        combine (bool, optional): Combine image mask with img_source. Defaults to True.
        use_alpha (bool, optional): Enable the alpha channel. Defaults to False.
        dist (float, optional): Normalized propagation distance (euclidean distance, will be 
            multiplied by 3.0 to match substance designer's behavior). Defaults to 10.0/256.0.
        max_dist (float, optional): Maximum propagation distance. Defaults to 256.0.

    Returns:
        Tensor: Distanced image.
    """
    assert mode in ('color', 'gray')
    grayscale_input_check(img_mask, 'image mask')
    num_rows = img_mask.shape[2]
    num_cols = img_mask.shape[3]

    if img_source is not None:
        # img_source type and shape check
        img_source = to_tensor(img_source)
        assert len(img_source.shape) == 4, 'img_source is must be a 4D Pytorch tensor'
        assert img_mask.shape[0] == img_source.shape[0] and \
               num_rows == img_source.shape[2] and \
               num_cols == img_source.shape[3], 'the shape of img_mask and img_source does not match'

        # Manually add an alpha channel if necessary
        if combine and img_source.shape[1] > 1:
            assert use_alpha, 'Alpha channel must be enabled for this case.'
            if img_source.shape[1] == 3:
                img_source = th.cat([img_source, th.ones_like(img_mask)], dim=1)

    # Rescale distance
    dist = to_tensor(dist) * max_dist * num_rows / 256

    # Special cases for small distances
    if dist <= 1.0:
        img_mask = th.zeros_like(img_mask) if dist == 0.0 else to_tensor(img_mask > 0.5)
        if img_source is None:
            return img_mask.expand(img_mask.shape[0], 1 if mode == 'gray' else 3 if not use_alpha else 4, num_rows, num_cols)
        elif not combine:
            return img_source
        elif img_source.shape[1] == 1:
            return img_source * img_mask
        else:
            img_out = img_source.clone()
            img_out[:,[3],:,:] = img_out[:,[3],:,:] * img_mask
            return img_out

    # Calculate padding
    pad_dist = int(np.ceil(dist.item() if isinstance(dist, th.Tensor) else dist)) + 1
    pad_cols = min(num_cols // 2, pad_dist)
    pad_rows = min(num_rows // 2, pad_dist)
    p2d = [pad_cols, pad_cols, pad_rows, pad_rows]

    if img_source is not None:
        img_out = th.zeros_like(img_source)
    else:
        img_out = th.zeros(img_mask.shape[0], 1 if mode == 'gray' else 3 if not use_alpha else 4, num_rows, num_cols)

    # loop through batch
    for i in range(img_mask.shape[0]):
        # compute mask
        binary_mask = (img_mask[i,0,:,:] <= 0.5).unsqueeze(0).unsqueeze(0)
        binary_mask = th.nn.functional.pad(binary_mask, p2d, mode='circular')
        binary_mask_np = binary_mask[0, 0].detach().cpu().numpy()
        # compute manhattan distance, closest point indices, non-zero mask
        # !! speed bottleneck !!
        dist_mtx, indices = spi.morphology.distance_transform_edt(binary_mask_np, return_distances=True, return_indices=True)
        dist_mtx = to_tensor(dist_mtx.astype(np.float32))
        dist_mtx = dist_mtx[p2d[2]:p2d[2]+num_rows, p2d[0]:p2d[0]+num_cols].unsqueeze(0).unsqueeze(0)
        dist_weights = th.clamp(1.0 - dist_mtx / dist, 0.0, 1.0)
        indices = to_tensor(indices[::-1, p2d[2]:p2d[2]+num_rows, p2d[0]:p2d[0]+num_cols].astype(np.float32))

        if img_source is None:
            img_out[i,:,:,:] = dist_weights
        else:
            # normalize to screen coordinate
            indices[0,:,:] = (th.remainder(indices[0,:,:] - p2d[0], num_cols) + 0.5) / num_cols * 2.0 - 1.0
            indices[1,:,:] = (th.remainder(indices[1,:,:] - p2d[2], num_rows) + 0.5) / num_rows * 2.0 - 1.0
            # reshape to (1, num_rows, num_cols, 2) and convert to torch tensor
            sample_grid = indices.permute(1,2,0).unsqueeze(0)
            # sample grid and apply distance operator
            cur_img = th.nn.functional.grid_sample(img_source[[i],:,:,:], sample_grid, mode='nearest', padding_mode='zeros', align_corners=False)
            if not combine:
                dist_mask = (dist_mtx >= dist).expand_as(cur_img)
                cur_img[dist_mask] = img_source[dist_mask]
            elif img_source.shape[1] == 1:
                cur_img = cur_img * dist_weights
            else:
                cur_img[:,[3],:,:] = cur_img[:,[3],:,:] * dist_weights

            img_out[i,:,:,:] = cur_img

    return img_out


@input_check(2)
def emboss(img_in, height_map, intensity=0.5, max_intensity=10.0, light_angle=0.0, 
    highlight_color=[1.0,1.0,1.0], shadow_color=[0.0,0.0,0.0]):
    """Atomic function: Emboss (https://docs.substance3d.com/sddoc/emboss-172825208.html)

    Args:
        img_in (tensor): Input image.
        height_map (tensor): Height map (G only).
        intensity (float, optional): Normalized height_map multiplier. Defaults to 0.5.
        max_intensity (float, optional): Maximum propagation distance. Defaults to 10.0.
        light_angle (float, optional): Light angle. Defaults to 0.0.
        highlight_color (list, optional): Highlight color. Defaults to [1.0,1.0,1.0].
        shadow_color (list, optional): Shadow color. Defaults to [0.0,0.0,0.0].

    Returns:
        Tensor: Embossed image.
    """
    grayscale_input_check(height_map, 'height map')
    if img_in.shape[1] == 4:
        img_in_alpha = img_in[:,3,:,:].unsqueeze(1)
        img_in = img_in[:,:3,:,:]
        use_alpha = True
    else:
        use_alpha = False

    light_angle = to_tensor(light_angle) * np.pi * 2.0
    num_channels = img_in.shape[1]
    highlight_color = to_tensor(highlight_color[:num_channels]).view(1,num_channels,1,1).expand(*img_in.shape)
    shadow_color = to_tensor(shadow_color[:num_channels]).view(1,num_channels,1,1).expand(*img_in.shape)

    num_rows = img_in.shape[2]
    intensity = intensity * num_rows / 512
    N = normal(height_map, mode='object_space', intensity=to_zero_one(intensity), max_intensity=max_intensity)
    weight = (N[:,0,:,:] * th.cos(np.pi-light_angle) - N[:,1,:,:] * th.sin(np.pi-light_angle)).unsqueeze(0).expand(*img_in.shape)

    img_out = th.zeros_like(img_in)
    highlight_color = 2.0 * highlight_color - 1.0
    shadow_color = 2.0 * shadow_color - 1.0
    img_out[weight >= 0.0] = img_in[weight >= 0.0] + highlight_color[weight >= 0.0] * weight[weight >= 0.0]
    img_out[weight < 0.0]  = img_in[weight < 0.0]  + shadow_color[weight < 0.0] * (-weight[weight < 0.0])
    img_out = th.clamp(img_out, 0.0, 1.0)

    if use_alpha:
        img_out = th.cat([img_out, img_in_alpha], dim=1)
        
    return img_out


@input_check(1)
def gradient_map(img_in, interpolate=True, mode='color', use_alpha=False, num_anchors=2, anchors=None):
    """Atomic function: Gradient Map (https://docs.substance3d.com/sddoc/gradient-map-172825246.html)

    Args:
        img_in (tensor): Input image.
        interpolate (bool, optional): Use a bezier curve when set to False. Defaults to True.
        mode (str, optional): 'color' or 'gray'. Defaults to 'color'.
        use_alpha (bool, optional): Enable the alpha channel. Defaults to False.
        num_anchors (int, optional): Number of anchors. Defaults to 2.
        anchors (list, optional): Anchors. Defaults to None.

    Returns:
        Tensor: Gradient map image.
    """
    grayscale_input_check(img_in, "input image")

    if num_anchors == 0:
        return img_in

    num_col = 2 if mode == 'gray' else 4 + use_alpha
    if anchors is None:
        anchors = th.linspace(0.0, 1.0, num_anchors).view(num_anchors, 1).repeat(1, num_col)
    else:
        anchors = to_tensor(anchors)
        assert anchors.shape[1] == num_col, "shape of anchors doesn't match color mode"
        shuffle_idx = th.argsort(anchors[:,0])
        anchors = anchors[shuffle_idx, :]

    # compute mapping
    img_out = th.zeros(img_in.shape[0], img_in.shape[2], img_in.shape[3], num_col-1)
    img_in = img_in.squeeze(1)

    img_out[img_in < anchors[0,0], :] = anchors[0,1:]
    img_out[img_in >= anchors[num_anchors-1,0], :] = anchors[num_anchors-1,1:]
    for j in range(num_anchors-1):
        a = (img_in.unsqueeze(3) - anchors[j,0]) / (anchors[j+1,0] - anchors[j,0] + 1e-8)
        if not interpolate:  # this should be a Bezier curve
            a = a ** 2 * 3 - a ** 3 * 2
        img_map = (1 - a) * anchors[j,1:] + a * anchors[j+1,1:]
        cond = (img_in >= anchors[j,0]) & (img_in < anchors[j+1,0])
        img_out[cond,:] = img_map[cond,:]

    # revert the order of dimensions
    img_out = img_out.permute(0,3,1,2)
    return img_out

@input_check(1)
def gradient_map_dyn(img_in, img_gradient, orientation='horizontal', use_alpha = False, position=0.0):
    """Atomic function: Gradient Map (https://docs.substance3d.com/sddoc/gradient-map-172825246.html)

    Args:
        img_in (tensor): Input image.
        img_gradient (tensor): Gradient image.
        orientation (str, optional): 'vertical' or 'horizontal', sampling direction. 
            Defaults to 'horizontal'.
        use_alpha (bool, optional): Enable the alpha channel. Defaults to False.
        position (float, optional): Normalized position to sample. Defaults to 0.0.

    Returns:
        Tensor: Gradient map image.
    """
    grayscale_input_check(img_in, "input image")
    if img_gradient.shape[1] == 3 or img_gradient.shape[1] == 4:
        img_gradient = img_gradient[:,:3,:,:]
        mode = 'color'
    else:
        mode = 'gray'

    assert img_gradient.shape[0] == 1, "please input a single gradient image"

    h_res, w_res = img_in.shape[2], img_in.shape[3]
    grad_h_res, grad_w_res = img_gradient.shape[2], img_gradient.shape[3]
    gs_interp_mode = 'bilinear'
    gs_padding_mode ='zeros'

    img_in_perm = img_in.permute(0, 2, 3, 1)
    if orientation == 'vertical':
        row_grid = img_in_perm * 2.0 - 1.0
        col_grid = to_tensor((position * 2.0 - 1.0) * (grad_w_res - 1) / grad_w_res).expand_as(img_in_perm)
    else:
        row_grid = to_tensor((position * 2.0 - 1.0) * (grad_h_res - 1) / grad_h_res).expand_as(img_in_perm)
        col_grid = img_in_perm * 2.0 - 1.0

    sample_grid = th.cat([col_grid, row_grid], dim=3)
    img_out = th.nn.functional.grid_sample(img_gradient, sample_grid, mode=gs_interp_mode, padding_mode=gs_padding_mode, align_corners=False)

    return img_out

@input_check(1)
def c2g(img_in, flatten_alpha=False, rgba_weights=[1.0/3.0, 1.0/3.0, 1.0/3.0, 0.0], bg=1.0):
    """Atomic function: Grayscale Conversion (https://docs.substance3d.com/sddoc/grayscale-conversion-172825250.html)

    Args:
        img_in (tensor): Input image
        flatten_alpha (bool, optional): Set the behaviour of alpha on the final grayscale image. Defaults to False.
        rgba_weights (list, optional): Rgba combination weights. Defaults to [1.0/3.0, 1.0/3.0, 1.0/3.0, 0.0].
        bg (float, optional): Uniform background color. Defaults to 1.0.

    Returns:
        Tensor: Grayscale converted image.
    """
    color_input_check(img_in, 'input image')

    rgba_weights = to_tensor(rgba_weights)
    img_out = (img_in * rgba_weights[:img_in.shape[1]].view(1,img_in.shape[1],1,1)).sum(dim=1, keepdim=True)
    if flatten_alpha and img_in.shape[1] == 4:
        img_out = img_out * img_in[:,3,:,:] + bg * (1.0 - img_in[:,3,:,:])

    return img_out
  

@input_check(1)
def hsl(img_in, hue=0.5, saturation=0.5, lightness=0.5):
    """Atomic function: HSL (https://docs.substance3d.com/sddoc/hsl-172825254.html)

    Args:
        img_in (tensor): Input image
        hue (float, optional): Hue. Defaults to 0.5.
        saturation (float, optional): Saturation. Defaults to 0.5.
        lightness (float, optional): Lightness. Defaults to 0.5.

    Returns:
        Tensor: HSL adjusted image.
    """
    color_input_check(img_in, 'input image')

    if img_in.shape[1] == 4:
        img_in_alpha = img_in[:,3,:,:].unsqueeze(1)
        img_in = img_in[:,:3,:,:]
        use_alpha = True
    else:
        use_alpha = False

    r = img_in[:,0,:,:]
    g = img_in[:,1,:,:]
    b = img_in[:,2,:,:]

    # compute s,v
    max_vals, _ = th.max(img_in, 1, False)
    min_vals, _ = th.min(img_in, 1, False)
    delta = max_vals - min_vals
    delta_mask = delta > 0.0
    l = (max_vals + min_vals) / 2.0
    s = th.zeros_like(delta)
    s_mask = (l > 0.0) * (l < 1.0) 
    # still need a small constant...
    s[s_mask] = delta[s_mask] / (1.0 - th.abs(2*l[s_mask] - 1.0) + 1e-8)
    h = th.zeros_like(s)
    
    # compute h
    red_mask = (img_in[:,0,:,:] == max_vals) * delta_mask
    green_mask = (img_in[:,1,:,:] == max_vals) * delta_mask
    blue_mask = (img_in[:,2,:,:] == max_vals) * delta_mask
    h[red_mask] = th.remainder((g[red_mask]-b[red_mask])/delta[red_mask], 6.0) / 6.0
    h[green_mask] = ((b[green_mask]-r[green_mask])/delta[green_mask] + 2.0) / 6.0
    h[blue_mask] = ((r[blue_mask]-g[blue_mask])/delta[blue_mask] + 4.0) / 6.0

    # modify hsv
    h = th.remainder(h + (hue-0.5) * 2.0 + 2.0, 1.0)
    l = th.clamp(l + 2.0*lightness - 1.0, 0.0, 1.0)
    s = th.clamp(s + 2.0*saturation - 1.0, 0.0, 1.0)

    # convert back to rgb
    c = (1.0 - th.abs(2.0 * l - 1.0)) * s
    x = c * (1.0 - th.abs( th.remainder(h/(1.0/6.0), 2.0) - 1.0))
    m = l - c/2.0

    r_out = th.zeros_like(r)
    g_out = th.zeros_like(g)
    b_out = th.zeros_like(b)
    h_1_mask = (h >= 0.0) * (h < 1.0/6.0)
    h_2_mask = (h >= 1.0/6.0) * (h < 2.0/6.0)
    h_3_mask = (h >= 2.0/6.0) * (h < 3.0/6.0)
    h_4_mask = (h >= 3.0/6.0) * (h < 4.0/6.0)
    h_5_mask = (h >= 4.0/6.0) * (h < 5.0/6.0)
    h_6_mask = (h >= 5.0/6.0) * (h <= 6.0/6.0)
    r_out[h_1_mask + h_6_mask] = c[h_1_mask + h_6_mask]
    r_out[h_2_mask + h_5_mask] = x[h_2_mask + h_5_mask]
    g_out[h_1_mask + h_4_mask] = x[h_1_mask + h_4_mask]
    g_out[h_2_mask + h_3_mask] = c[h_2_mask + h_3_mask] 
    b_out[h_3_mask + h_6_mask] = x[h_3_mask + h_6_mask]
    b_out[h_4_mask + h_5_mask] = c[h_4_mask + h_5_mask]

    rgb_out = th.stack([r_out, g_out, b_out], dim=1) + m

    if use_alpha:
        rgb_out = th.cat([rgb_out, img_in_alpha], dim=1)

    return rgb_out


@input_check(1)
def levels(img_in, in_low=[0.0], in_mid=[0.5], in_high=[1.0], out_low=[0.0], out_high=[1.0]):
    """Atomic function: Levels (https://docs.substance3d.com/sddoc/levels-172825279.html)

    Args:
        img_in (tensor): Input image
        in_low (list, optional): Low cutoff for input. Defaults to [0.0].
        in_mid (list, optional): Middle point for calculating gamma correction. Defaults to [0.5].
        in_high (list, optional): High cutoff for input. Defaults to [1.0].
        out_low (list, optional): Low cutoff for output. Defaults to [0.0].
        out_high (list, optional): High cutoff for output. Defaults to [1.0].

    Returns:
        Tensor: Level adjusted image.
    """
    def param_process(param_in, default_val):
        param_in = to_tensor(param_in)
        if len(param_in.shape) == 0 or param_in.shape[0] == 1:
            param_in = param_in.view(1).expand(3)
        if param_in.shape[0] == 3 and img_in.shape[1] == 4:
            param_in = th.cat([param_in, to_tensor([default_val])])
        return param_in

    num_channels = img_in.shape[1] 
    limit = to_tensor(9.0)
    in_low = param_process(in_low, 0.0)
    in_mid = param_process(in_mid, 0.5)
    in_high = param_process(in_high, 1.0)
    out_low = param_process(out_low, 0.0)
    out_high = param_process(out_high, 1.0)
    
    img_out = th.zeros_like(img_in)
    for i in range(num_channels):
        if in_low[i] > in_high[i]:
            img_in_slice = 1.0 - img_in[:,i,:,:].clone()
            left, right = in_high[i], in_low[i]
        else:
            img_in_slice = img_in[:,i,:,:].clone()
            left, right = in_low[i], in_high[i]
        if left == right:
            right = right + 0.0001

        gamma_corr = 1.0 + (8.0 * th.abs(2.0 * in_mid[i] - 1.0))
        gamma_corr = th.min(gamma_corr, limit)
        if in_mid[i] < 0.5:
            gamma_corr = 1.0 / gamma_corr

        img_in_slice = th.min(th.max(img_in_slice, left), right)
        # magic number 1e-15
        img_slice = th.pow((img_in_slice - left + 1e-15) / (right - left + 1e-15), gamma_corr)

        if out_low[i] > out_high[i]:
            img_slice = 1.0 - img_slice
            left, right = out_high[i], out_low[i]
        else:
            left, right = out_low[i], out_high[i]
        img_out_slice = img_slice * (right - left) + left
        img_out_slice = th.min(th.max(img_out_slice, left), right)
        img_out[:,i,:,:] = img_out_slice

    return img_out
    
@input_check(1)
def normal(img_in, mode='tangent_space', normal_format='dx', use_input_alpha=False, use_alpha=False, intensity=1.0/3.0, max_intensity=3.0):
    """Atomic function: Normal (https://docs.substance3d.com/sddoc/normal-172825289.html)

    Args:
        img_in (tensor): Input image.
        mode (str, optional): 'tangent space' or 'object space'. Defaults to 'tangent_space'.
        normal_format (str, optional): 'gl' or 'dx'. Defaults to 'dx'.
        use_input_alpha (bool, optional): Use input alpha. Defaults to False.
        use_alpha (bool, optional): Enable the alpha channel. Defaults to False.
        intensity (float, optional): Normalized height map multiplier on dx, dy. Defaults to 1.0/3.0.
        max_intensity (float, optional): Maximum height map multiplier. Defaults to 3.0.

    Returns:
        Tensor: Normal image.
    """
    grayscale_input_check(img_in, "input height field")

    img_size = img_in.shape[2]
    # intensity = intensity * max_intensity * img_size / 256.0 # magic number to match sbs, check it later
    intensity = (intensity * 2.0 - 1.0) * max_intensity * img_size / 256.0 # magic number to match sbs, check it later
    dx = roll_col(img_in, -1) - img_in
    dy = roll_row(img_in, -1) - img_in
    if normal_format == 'gl':
        img_out = th.cat((intensity*dx, -intensity*dy, th.ones_like(dx)), 1)
    elif normal_format == 'dx':
        img_out = th.cat((intensity*dx, intensity*dy, th.ones_like(dx)), 1)
    else:
        img_out = th.cat((-intensity*dx, intensity*dy, th.ones_like(dx)), 1)
    img_out = normalize(img_out)
    if mode == 'tangent_space':
        img_out = img_out / 2.0 + 0.5
    
    if use_alpha == True:
        if use_input_alpha:
            img_out = th.cat([img_out, img_in], dim=1)
        else:
            img_out = th.cat([img_out, th.ones(img_out.shape[0], 1, img_out.shape[2], img_out.shape[3])], dim=1)

    return img_out


@input_check(1)
def sharpen(img_in, intensity=1.0 / 3.0, max_intensity=3.0):
    """Atomic function: Sharpen (https://docs.substance3d.com/sddoc/sharpen-172825322.html)

    Args:
        img_in (tensor): Input image.
        intensity (float, optional): Normalized unsharp mask multiplier. Defaults to 1.0/3.0.
        max_intensity (float, optional): Maximum unsharp mask multiplier. Defaults to 3.0.

    Returns:
        Tensor: Sharpened image.
    """
    num_group = img_in.shape[1]
    intensity = to_tensor(intensity) * max_intensity
    kernel = to_tensor([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
    kernel = kernel.view(1,1,3,3).expand(num_group, 1, 3, 3)
    # manually pad input image
    p2d = [1, 1, 1, 1]
    in_pad  = th.nn.functional.pad(img_in, p2d, mode='circular')
    # perform depth-wise convolution without implicit padding
    unsharp_mask = th.nn.functional.conv2d(in_pad, kernel, groups=num_group, padding = 0)
    img_out = th.clamp(img_in + unsharp_mask * intensity, to_tensor(0.0), to_tensor(1.0))
    return img_out

@input_check(1)
def transform_2d(img_in, tile_mode=3, sample_mode='bilinear', mipmap_mode='auto', mipmap_level=0, x1=1.0, x1_max=1.0, x2=0.5, x2_max=1.0,
                 x_offset=0.5, x_offset_max=1.0, y1=0.5, y1_max=1.0, y2=1.0, y2_max=1.0, y_offset=0.5, y_offset_max=1.0,
                 matte_color=[0.0, 0.0, 0.0, 1.0]):
    """Atomic function: Transform 2D (https://docs.substance3d.com/sddoc/transformation-2d-172825332.html)

    Args:
        img_in (tensor): input image
        tile_mode (int, optional): 0=no tile, 
                                   1=horizontal tile, 
                                   2=vertical tile, 
                                   3=horizontal and vertical tile. Defaults to 3.
        sample_mode (str, optional): 'bilinear' or 'nearest'. Defaults to 'bilinear'.
        mipmap_mode (str, optional): 'auto' or 'manual'. Defaults to 'auto'.
        mipmap_level (int, optional): Mipmap level. Defaults to 0.
        x1 (float, optional): Entry in the affine transformation matrix, same for the below. Defaults to 1.0.
        x1_max (float, optional): . Defaults to 1.0.
        x2 (float, optional): . Defaults to 0.5.
        x2_max (float, optional): . Defaults to 1.0.
        x_offset (float, optional): . Defaults to 0.5.
        x_offset_max (float, optional): . Defaults to 1.0.
        y1 (float, optional): . Defaults to 0.5.
        y1_max (float, optional): . Defaults to 1.0.
        y2 (float, optional): . Defaults to 1.0.
        y2_max (float, optional): . Defaults to 1.0.
        y_offset (float, optional): . Defaults to 0.5.
        y_offset_max (float, optional): . Defaults to 1.0.
        matte_color (list, optional): background color. Defaults to [0.0, 0.0, 0.0, 1.0].

    Returns:
        Tensor: Transformed image.
    """
    assert sample_mode in ('bilinear', 'nearest')
    assert mipmap_mode in ('auto', 'manual')

    gs_padding_mode = 'zeros'
    gs_interp_mode = sample_mode

    x1 = to_tensor((x1 * 2.0 - 1.0) * x1_max).squeeze()
    x2 = to_tensor((x2 * 2.0 - 1.0) * x2_max).squeeze()
    x_offset = to_tensor((x_offset * 2.0 - 1.0) * x_offset_max).squeeze()
    y1 = to_tensor((y1 * 2.0 - 1.0) * y1_max).squeeze()
    y2 = to_tensor((y2 * 2.0 - 1.0) * y2_max).squeeze()
    y_offset = to_tensor((y_offset * 2.0 - 1.0) * y_offset_max).squeeze()
    matte_color = to_tensor(matte_color).view(-1)
    matte_color = resize_color(matte_color, img_in.shape[1])

    # compute mipmap level
    mm_level = mipmap_level
    det = th.abs(x1 * y2 - x2 * y1)
    if det < 1e-6:
        print('Warning: singular transformation matrix may lead to unexpected results.')
        mm_level = 0
    elif mipmap_mode == 'auto':
        inv_h1 = th.sqrt(x2 * x2 + y2 * y2)
        inv_h2 = th.sqrt(x1 * x1 + y1 * y1)
        max_compress_ratio = th.max(inv_h1, inv_h2)
        # !! this is a hack !!
        upper_limit = 2895.329
        thresholds = to_tensor([upper_limit / (1 << i) for i in reversed(range(12))])
        mm_level = th.sum(max_compress_ratio > thresholds).item()
        # Special cases
        is_pow2 = lambda x: th.remainder(th.log2(x), 1.0) == 0
        if th.abs(x1) == th.abs(y2) and x2 == 0 and y1 == 0 and is_pow2(th.abs(x1)) or \
           th.abs(x2) == th.abs(y1) and x1 == 0 and y2 == 0 and is_pow2(th.abs(x2)):
            scale = th.max(th.abs(x1), th.abs(x2))
            if th.remainder(x_offset * scale, 1.0) == 0 and th.remainder(y_offset * scale, 1.0) == 0:
                mm_level = max(0, mm_level - 1)

    # mipmapping (optional)
    if mm_level > 0:
        mm_level = min(mm_level, int(np.floor(np.log2(img_in.shape[2]))))
        img_mm = automatic_resize(img_in, -mm_level)
        img_mm = manual_resize(img_mm, mm_level)
        assert img_mm.shape == img_in.shape
    else:
        img_mm = img_in

    # compute sampling tensor
    res_x, res_y = img_in.shape[3], img_in.shape[2]
    theta_first_row = th.stack([x1, y1, x_offset * 2.0])
    theta_second_row = th.stack([x2, y2, y_offset * 2.0])
    theta = th.stack([theta_first_row, theta_second_row]).unsqueeze(0).expand(img_in.shape[0],2,3)
    sample_grid = th.nn.functional.affine_grid(theta, img_in.shape, align_corners=False)

    if tile_mode in (1, 3):
        sample_grid[:,:,:,0] = (th.remainder(sample_grid[:,:,:,0] + 1.0, 2.0) - 1.0) * res_x / (res_x + 2)
    if tile_mode in (2, 3):
        sample_grid[:,:,:,1] = (th.remainder(sample_grid[:,:,:,1] + 1.0, 2.0) - 1.0) * res_y / (res_y + 2)

    # Deduce background color from the image if tiling is not fully applied
    if tile_mode < 3:
        img_mm = img_mm - matte_color[:, None, None]

    # Pad input image
    if tile_mode == 0:
        img_pad = img_mm
    else:
        pad_arr = [[0, 0, 0, 0], [1, 1, 0, 0], [0, 0, 1, 1], [1, 1, 1, 1]]
        img_pad = th.nn.functional.pad(img_mm, pad_arr[tile_mode], mode='circular')

    # compute output
    img_out = th.nn.functional.grid_sample(img_pad, sample_grid, mode=gs_interp_mode, padding_mode=gs_padding_mode, align_corners=False)

    # Add the background color back after sampling
    if tile_mode < 3:
        img_out = img_out + matte_color[:, None, None]

    return img_out

@input_check(1)
def special_transform(img_in, tile_mode=3, sample_mode='bilinear', scale=0.25, scale_max=4.0, x_offset=0.5, x_offset_max=1.0, 
    y_offset=0.5, y_offset_max=1.0):
    """Special transform for only changing the scale and offset of the image

    Args:
        img_in (tensor): Input image.
        tile_mode (int, optional): 0=no tile, 
                                   1=horizontal tile, 
                                   2=vertical tile, 
                                   3=horizontal and vertical tile. Defaults to 3.
        sample_mode (str, optional): 'bilinear' or 'nearest'. Defaults to 'bilinear'.
        scale (float, optional): Normalized scale. Defaults to 0.25.
        scale_max (float, optional): Maximum scale. Defaults to 4.0.
        x_offset (float, optional): Normalized x-axis translation. Defaults to 0.5.
        x_offset_max (float, optional): Maximum x-axis translation. Defaults to 1.0.
        y_offset (float, optional): Normalized y-axis translation. Defaults to 0.5.
        y_offset_max (float, optional): Maximum y-axis translation. Defaults to 1.0.

    Returns:
        Tensor: Special transformed image.
    """    
    assert sample_mode in ('bilinear', 'nearest')
    scale = scale / 2.0 + 0.5
    img_out = transform_2d(img_in, tile_mode, sample_mode, 'manual', 0, scale, scale_max, 0.5, 1.0, x_offset, x_offset_max, 0.5, 1.0, scale, scale_max, y_offset, y_offset_max)

    return img_out

@input_check(0)
def uniform_color(mode='color', num_imgs=1, res_h=512, res_w=512, use_alpha=False, rgba=[0.0, 0.0, 0.0, 1.0]):
    """Atomic function: Uniform Color (https://docs.substance3d.com/sddoc/uniform-color-172825339.html)

    Args:
        mode (str, optional): 'gray' or 'color'. Defaults to 'color'.
        num_imgs (int, optional): Number of images. Defaults to 1.
        res_h (int, optional): Height. Defaults to 512.
        res_w (int, optional): Width. Defaults to 512.
        use_alpha (bool, optional): Enable the alpha channel. Defaults to False.
        rgba (list, optional): RGBA color. Defaults to [0.0, 0.0, 0.0, 1.0].

    Returns:
        Tensor: Uniform image.
    """
    def param_process(param_in):
        param_in = to_tensor(param_in)
        if len(param_in.shape) == 0 or param_in.shape[0] == 1:
            param_in = param_in.view(1).expand(3)
        if param_in.shape[0] == 3 and use_alpha:
            param_in = th.cat([param_in, to_tensor([1.0])])
        return param_in

    rgba = param_process(rgba)
    if mode == 'gray':
        img_out = th.ones(num_imgs,1,res_h,res_w) * rgba[0]
    else:
        if use_alpha:
            img_out = th.ones(num_imgs,4,res_h,res_w) * rgba.view(1,4,1,1)
        else:
            img_out = th.ones(num_imgs,3,res_h,res_w) * rgba[:3].view(1,3,1,1)
    return img_out


@input_check(2)
def warp(img_in, intensity_mask, intensity=0.5, max_intensity=2.0):
    """Atomic function: Warp (https://docs.substance3d.com/sddoc/warp-172825344.html)

    Args:
        img_in (tensor): Input image
        intensity_mask (tensor): Intensity mask for computing displacement
        intensity (float, optional): Normalized intensity mask multiplier. Defaults to 0.5.
        max_intensity (float, optional): Maximum intensity mask multiplier. Defaults to 2.0.

    Returns:
        Tensor: Warped image.
    """
    grayscale_input_check(intensity_mask, "intensity mask")

    num_row = img_in.shape[2]
    num_col = img_in.shape[3]
    intensity = to_tensor(intensity) * max_intensity
    gs_interp_mode = 'bilinear'
    gs_padding_mode = 'zeros'
    row_scale = num_row / 256.0 # magic number, similar to d_warp
    col_scale = num_col / 256.0 # magic number
    shift = th.cat([intensity_mask - roll_row(intensity_mask, -1), intensity_mask - roll_col(intensity_mask, -1)], 1)
    row_grid, col_grid = th.meshgrid(th.linspace(0, num_row-1, num_row), th.linspace(0, num_col-1, num_col))
    row_shift = shift[:,[0],:,:] * intensity * num_row * row_scale
    col_shift = shift[:,[1],:,:] * intensity * num_col * col_scale
    # mod the index to behavior as tiling 
    row_grid = th.remainder((row_grid + row_shift + 0.5) / num_row * 2.0, 2.0) - 1.0
    col_grid = th.remainder((col_grid + col_shift + 0.5) / num_col * 2.0, 2.0) - 1.0
    row_grid = row_grid * num_row / (num_row + 2)
    col_grid = col_grid * num_col / (num_col + 2)
    sample_grid = th.cat([col_grid, row_grid], 1).permute(0,2,3,1).expand(intensity_mask.shape[0], num_row, num_col, 2)
    in_pad = th.nn.functional.pad(img_in, [1, 1, 1, 1], mode='circular')
    img_out = th.nn.functional.grid_sample(in_pad, sample_grid, mode=gs_interp_mode, padding_mode=gs_padding_mode, align_corners=False)
    return img_out

@input_check(1)
def passthrough(img_in):
    """Helper function: Dot Node (https://docs.substance3d.com/sddoc/graph-items-186056822.html)

    Args:
        img_in (tensor): Input image

    Returns:
        Tensor: The same image.
    """
    return img_in


# ---------------------------------------------
# Non-atomic functions

# add sbsnode support
@input_check(1)
def linear_to_srgb(img_in):
    """Non-atomic function: Convert to sRGB (https://docs.substance3d.com/sddoc/convert-to-srgb-159449240.html)

    Args:
        img_in (tensor): Input image.

    Returns:
        Tensor: Gamma mapped image.
    """
    if img_in.shape[1] == 3 or img_in.shape[1]==4:
        img_out = levels(img_in, in_mid=[0.425,0.425,0.425,0.5])
    else:
        img_out = levels(img_in, in_mid=[0.425])

    return img_out

# add sbsnode support
@input_check(1)
def srgb_to_linear(img_in):
    """Non-atomic function: Convert to Linear (https://docs.substance3d.com/sddoc/convert-to-linear-159449235.html)

    Args:
        img_in (tensor): Input image.

    Returns:
        tensor: Gamma corrected image.
    """
    if img_in.shape[1] == 3 or img_in.shape[1]==4:
        img_out = levels(img_in, in_mid=[0.575,0.575,0.575,0.5])
    else:
        img_out = levels(img_in, in_mid=[0.575])
    return img_out

@input_check(1)
def curvature(normal, normal_format='dx', emboss_intensity=0.1, emboss_max_intensity=10.0):
    """Non-atomic function: Curvature (https://docs.substance3d.com/sddoc/curvature-filter-node-159450514.html)

    Args:
        normal (tensor): Input normal image.
        normal_format (str, optional): Normal format (DirectX 'dx' | OpenGL 'gl'). Defaults to 'dx'.
        emboss_intensity (float, optional): Normalized intensity multiplier. Defaults to 0.1.
        emboss_max_intensity (float, optional): Maximum intensity multiplier. Defaults to 10.0.

    Returns:
        tensor: Curvature image.
    """
    color_input_check(normal, 'input normal')

    normal_col_shifted = roll_col(normal, 1)[:,[0],:,:]
    normal_row_shifted = roll_row(normal, 1)[:,[1],:,:]
    gray = uniform_color('gray', res_h=normal_col_shifted.shape[2], res_w=normal_col_shifted.shape[3], rgba=0.5)
    pixel_size = 2048 / normal_col_shifted.shape[2] * 0.1
    embossed_col = emboss(gray, normal_col_shifted, emboss_intensity * pixel_size, emboss_max_intensity)
    embossed_row = emboss(gray, normal_row_shifted, emboss_intensity * pixel_size, emboss_max_intensity,
                          light_angle=0.25 if normal_format == 'dx' else 0.75)
    img_out = blend(embossed_col, embossed_row, opacity=0.5, blending_mode='add_sub')
        
    return img_out


@input_check(1)
def invert(img_in, invert_switch=True):
    """Non-atomic function: Invert (https://docs.substance3d.com/sddoc/invert-159449221.html)

    Args:
        img_in (tensor): Input image.
        invert_switch (bool, optional): Invert switch. Defaults to True.

    Returns:
        Tensor: Inverted image.
    """
    if invert_switch == True:
        img_out = img_in.clone()
        if img_out.shape[1] == 1:
            img_out = th.clamp(1.0 - img_out, 0.0, 1.0)
        else: 
            img_out[:,:3,:,:] = th.clamp(1.0 - img_in[:,:3,:,:].clone(), 0.0, 1.0)
        return img_out
    else:
        return img_in

@input_check(1)
def histogram_scan(img_in, invert_position=False, position=0.0, contrast=0.0):
    """Non-atomic function: Histogram Scan (https://docs.substance3d.com/sddoc/histogram-scan-159449213.html)

    Args:
        img_in (tensor): Input image
        invert_position (bool, optional): Invert position. Defaults to False.
        position (float, optional): Used to shift the middle point. Defaults to 0.0.
        contrast (float, optional): Used to adjust the contrast of the input. Defaults to 0.0.

    Returns:
        Tensor: Histogram scan image.
    """
    grayscale_input_check(img_in, 'input image')

    position = to_tensor(position)
    contrast = to_tensor(contrast)
    position_ = position if invert_position else 1.0 - position
    start_low = (th.max(to_tensor(0.5), position_) - 0.5) * 2.0
    end_low = th.min(position_ * 2.0, to_tensor(1.0))
    weight_low = th.clamp(contrast * 0.5, 0.0, 1.0)
    in_low = th.clamp(lerp(start_low, end_low, weight_low), 0.0, 1.0)
    in_high = th.clamp(lerp(end_low, start_low, weight_low), 0.0, 1.0)
    img_out = levels(img_in, in_low, 0.5, in_high, 0.0, 1.0)
    return img_out

@input_check(1)
def histogram_range(img_in, ranges=0.5, position=0.5):
    """Non-atomic function: Histogram Range (https://docs.substance3d.com/sddoc/histogram-range-159449207.html)

    Args:
        img_in (tensor): Input image.
        ranges (float, optional): How much to reduce the range down from. 
        This is similar to moving both Levels min and Max sliders inwards. Defaults to 0.5.
        position (float, optional): Offset for the range reduction, setting a different midpoint 
        for the range reduction. Defaults to 0.5.

    Returns:
        Tensor: Histogram range image.
    """
    grayscale_input_check(img_in, 'input image')

    ranges = to_tensor(ranges)
    position = to_tensor(position)
    out_low  = th.clamp(1.0 - th.min(ranges * 0.5 + (1.0 - position), (1.0 - position) * 2.0), 0.0, 1.0)
    out_high = th.clamp(th.min(ranges * 0.5 + position, position * 2.0), 0.0, 1.0)
    img_out = levels(img_in, 0.0, 0.5, 1.0, out_low, out_high)
    return img_out

@input_check(1)
def histogram_select(img_in, position=0.5, ranges=0.25, contrast=0.0):
    """Non-atomic function: Histogram Select (https://docs.substance3d.com/sddoc/histogram-select-166363409.html)

    Args:
        img_in (tensor): Input image.
        position (float, optional): Sets the middle position where the range selection happens.. Defaults to 0.5.
        ranges (float, optional): Sets width of the selection range. Defaults to 0.25.
        contrast (float, optional): Adjusts the contrast/falloff of the result. Defaults to 0.0.

    Returns:
        Tensor: Histogram select image.
    """
    grayscale_input_check(img_in, 'input image')
    position = to_tensor(position)
    ranges = to_tensor(ranges)
    contrast = to_tensor(contrast)

    if ranges == 0.0:
        return th.ones_like(img_in)
    img = th.clamp(1.0 - th.abs(img_in - position) / ranges, 0.0, 1.0)
    img_out = levels(img, contrast * 0.5, 0.5, 1.0 - contrast * 0.5, 0.0, 1.0)
    return img_out

@input_check(1)
def edge_detect(img_in, invert_flag=False, edge_width=2.0/16.0, max_edge_width=16.0, edge_roundness=4.0/16.0, max_edge_roundness=16.0,
                tolerance=0.0):
    """Non-atomic function: Edge Detect (https://docs.substance3d.com/sddoc/edge-detect-159450524.html)

    Args:
        img_in (tensor): Input image
        invert_flag (bool, optional): Invert the result. Defaults to False.
        edge_width (float, optional): Normalized width of the detected areas around the edges. Defaults to 2.0/16.0.
        max_edge_width (float, optional): Maximum width of the detected areas around the edges. Defaults to 16.0.
        edge_roundness (float, optional): Normalized rounds, blurs and smooths together the generated mask. Defaults to 4.0/16.0.
        max_edge_roundness (float, optional): Maximum rounds, blurs and smooths together the generated mask. Defaults to 16.0.
        tolerance (float, optional): Tolerance threshold factor for where edges should appear. Defaults to 0.0.

    Returns:
        Tensor: Detected edge image.
    """
    grayscale_input_check(img_in, 'input image')

    # Process input image
    edge_width = to_tensor(edge_width) * max_edge_width
    edge_roundness = to_tensor(edge_roundness) * max_edge_roundness
    tolerance = to_tensor(tolerance)

    # Edge detect
    img_scale = 256.0 / min(img_in.shape[2], img_in.shape[3])
    in_blur = blur(img_in, img_scale, 1.0)
    blend_sub_1 = blend(in_blur, img_in, blending_mode='subtract')
    blend_sub_2 = blend(img_in, in_blur, blending_mode='subtract')
    img_out = blend(blend_sub_1, blend_sub_2, blending_mode='add')
    img_out = levels(img_out, 0.0, 0.5, 0.05, 0.0, 1.0)
    levels_out_high = lerp(to_tensor(0.2), to_tensor(1.2), tolerance) / 100.0
    img_out = levels(img_out, 0.002, 0.5, levels_out_high, 0.0, 1.0)

    # Add edge width
    max_dist = th.max(edge_width - 1.0, to_tensor(0.0))
    if max_dist > 0:
        img_out = distance(img_out, img_out, combine=False, dist=max_dist, max_dist=1.0)

    # Edge roundness
    max_dist = th.max(th.ceil(edge_roundness), to_tensor(0.0))
    if max_dist > 0:
        img_out = distance(img_out, img_out, combine=False, dist=max_dist, max_dist=1.0)
    img_out = 1.0 - img_out
    if max_dist > 0:
        img_out = distance(img_out, img_out, combine=False, dist=max_dist, max_dist=1.0)
    img_out = 1.0 - img_out if invert_flag else img_out
    return img_out

@input_check(1)
def safe_transform(img_in, tile=1, tile_safe_rot=True, symmetry='none', tile_mode=3, mipmap_mode='auto', mipmap_level=0, offset_x=0.5, offset_y=0.5, angle=0.0):
    """Non-atomic function: Safe Transform (https://docs.substance3d.com/sddoc/safe-transform-159450643.html)

    Args:
        img_in (tensor): Input image.
        tile (int, optional): Scales the input down by tiling it. Defaults to 1.
        tile_safe_rot (bool, optional): Determines the behaviors of the rotation, whether it should snap to 
            safe values that don't blur any pixels. Defaults to True.
        symmetry (str, optional): 'X'|'Y'|'X+Y'|'none', performs symmetric transformation on the input. Defaults to 'none'.
        tile_mode (int, optional): 0=no tile, 
                                   1=horizontal tile, 
                                   2=vertical tile, 
                                   3=horizontal and vertical tile. Defaults to 3.Defaults to 3.
        mipmap_mode (str, optional): 'auto' or 'manual'. Defaults to 'auto'.
        mipmap_level (int, optional): Mipmap level. Defaults to 0.
        offset_x (float, optional): x-axis offset. Defaults to 0.5.
        offset_y (float, optional): y-axis offset. Defaults to 0.5.
        angle (float, optional): Rotates input along angle. Defaults to 0.0.

    Returns:
        Tensor: Safe transformed image.
    """
    num_row = img_in.shape[2]
    num_col = img_in.shape[3]
    # initial transform
    if symmetry == 'X':
        img_out = th.flip(img_in, dims=[2])
    elif symmetry == 'Y':
        img_out = th.flip(img_in, dims=[3])
    elif symmetry == 'X+Y':
        img_out = th.flip(th.flip(img_in, dims=[3]), dims=[2])
    elif symmetry=='none':
        img_out = img_in
    else:
        raise ValueError('unknown symmetry mode')
    # main transform
    angle = to_tensor(angle)
    tile = to_tensor(tile)
    offset_tile = th.remainder(tile + 1.0, 2.0) * to_tensor(0.5)
    if tile_safe_rot:
        angle = th.floor(angle * 8.0) / 8.0
        angle_res = th.remainder(th.abs(angle), 0.25) * (np.pi * 2.0)
        tile = tile * (th.cos(angle_res) + th.sin(angle_res))
    offset_x = th.floor((to_tensor(offset_x) * 2.0 - 1.0) * num_col) / num_col + offset_tile
    offset_y = th.floor((to_tensor(offset_y) * 2.0 - 1.0) * num_row) / num_row + offset_tile
    # compute affine transformation matrix
    angle = angle * np.pi * 2.0
    scale_matrix = to_tensor([[th.cos(angle), -th.sin(angle)],[th.sin(angle), th.cos(angle)]])
    rotation_matrix = to_tensor([[tile, 0.0],[0.0, tile]])
    scale_rotation_matrix = th.mm(rotation_matrix, scale_matrix)
    img_out = transform_2d(img_out, tile_mode=tile_mode, mipmap_mode=mipmap_mode, mipmap_level=mipmap_level,
                           x1=to_zero_one(scale_rotation_matrix[0,0]), x2=to_zero_one(scale_rotation_matrix[0,1]), x_offset=to_zero_one(offset_x), 
                           y1=to_zero_one(scale_rotation_matrix[1,0]), y2=to_zero_one(scale_rotation_matrix[1,1]), y_offset=to_zero_one(offset_y))
    return img_out

@input_check(1)
def blur_hq(img_in, high_quality=False, intensity=10.0 / 16.0, max_intensity=16.0):
    """Non-atomic function: Blur HQ (https://docs.substance3d.com/sddoc/blur-hq-159450455.html)

    Args:
        img_in (tensor): Input image.
        high_quality (bool, optional): Increases internal sampling amount for even higher quality, 
            at reduced computation speed. Defaults to False.
        intensity (tensor, optional): Normalized strength (Radius) of the blur. The higher this value, 
            the further the blur will reach. Defaults to 10.0/16.0.
        max_intensity (float, optional): Maximum strength (Radius) of the blur. Defaults to 16.0.

    Returns:
        Tensor: High quality blurred image.
    """
    intensity = to_tensor(intensity) * max_intensity
    # blur path 1s
    blur_intensity = intensity * 0.66
    blur_1 = d_blur(img_in, blur_intensity, 1.0, 0.0)
    blur_1 = d_blur(blur_1, blur_intensity, 1.0, 0.125)
    blur_1 = d_blur(blur_1, blur_intensity, 1.0, 0.25)
    blur_1 = d_blur(blur_1, blur_intensity, 1.0, 0.875)
    if high_quality:
        # blur path 2
        blur_2 = d_blur(img_in, blur_intensity, 1.0, 0.0625)
        blur_2 = d_blur(blur_2, blur_intensity, 1.0, 0.4375)
        blur_2 = d_blur(blur_2, blur_intensity, 1.0, 0.1875)
        blur_2 = d_blur(blur_2, blur_intensity, 1.0, 0.3125)
        # blending
        img_out = blend(blur_1, blur_2, opacity=0.5)
    else:
        img_out = blur_1
    return img_out

@input_check(2)
def non_uniform_blur(img_in, img_mask, samples=4, blades=5, intensity=0.2, max_intensity=50.0, anisotropy=0.0, asymmetry=0.0, angle=0.0):
    """Non-atomic function: Non-uniform Blur (https://docs.substance3d.com/sddoc/non-uniform-blur-159450461.html)

    Args:
        img_in (tensor): Input image.
        img_mask (tensor): Blur map.
        samples (int, optional): Amount of samples, determines quality. Multiplied by amount of Blades. Defaults to 4.
        blades (int, optional): Amount of sampling sectors, determines quality. Multiplied by amount of Samples. Defaults to 5.
        intensity (float, optional): Normalized intensity of blur. Defaults to 0.2.
        max_intensity (float, optional): Maximum intensity of blur. Defaults to 50.0.
        anisotropy (float, optional): Optionally adds directionality to the blur effect. 
            Driven by the Angle parameter. Defaults to 0.0.
        asymmetry (float, optional): Optionally adds a bias to the sampling. Driven by the Angle parameter. Defaults to 0.0.
        angle (float, optional): Angle to set directionality and sampling bias. Defaults to 0.0.

    Returns:
        Tensor: Non-uniform blurred image.
    """
    intensity = to_tensor(intensity) * max_intensity
    anisotropy = to_tensor(anisotropy)
    asymmetry = to_tensor(asymmetry)
    angle = to_tensor(angle)
    assert isinstance(samples, int) and samples >= 1 and samples <= 16
    assert isinstance(blades, int) and blades >= 1 and blades <= 9

    # compute progressive warping results based on 'samples'
    def non_uniform_blur_sample(img_in, img_mask, intensity=10.0, inner_rotation=0.0):
        img_out = img_in
        for i in range(1, blades + 1):
            e_vec = ellipse(blades, i, intensity, anisotropy, angle, inner_rotation, asymmetry)
            warp_intensity = e_vec.norm()
            warp_angle = th.atan2(e_vec[1] + 1e-15, e_vec[0] + 1e-15) / (np.pi * 2.0)
            img_warp = d_warp(img_in, img_mask, warp_intensity, 1.0, warp_angle)
            img_out = blend(img_warp, img_out, None, 'switch', opacity=1.0 / (i + 1))
        return img_out

    # compute progressive blurring based on 'samples' and 'intensity'
    samples_level = th.min(th.tensor(samples), th.ceil(intensity * np.pi).long())
    img_out = non_uniform_blur_sample(img_in, img_mask, intensity, 1 / samples)
    for i in range(1, samples_level):
        blur_intensity = intensity * th.exp(-i * np.sqrt(np.log(1e3) / np.e) / samples_level.float()) ** 2
        img_out = non_uniform_blur_sample(img_out, img_mask, blur_intensity, 1 / (samples * (i + 1)))
    return img_out

@input_check(1)
def bevel(img_in, non_uniform_blur_flag=True, use_alpha = False, dist=0.75, max_dist=1.0, smoothing=0.0, max_smoothing=5.0, \
        normal_intensity=0.2, max_normal_intensity=50.0):
    """Non-atomic function: Bevel (https://docs.substance3d.com/sddoc/bevel-filter-node-159450511.html)

    Args:
        img_in (tensor): Input image.
        non_uniform_blur_flag (bool, optional): Whether smoothing should be done non-uniformly. Defaults to True.
        use_alpha (bool, optional): Enable the alpha channel. Defaults to False.
        dist (float, optional): How far the bevel effect should reach. Defaults to 0.75.
        max_dist (float, optional): Maximum value of 'dist' parameter. Defaults to 1.0.
        smoothing (float, optional): How much additional smoothing (blurring) to perform after the bevel. Defaults to 0.0.
        max_smoothing (float, optional): Maximum value of 'smoothing' parameter. Defaults to 5.0.
        normal_intensity (float, optional): Normalized intensity of the generated Normalmap. Defaults to 0.2.
        max_normal_intensity (float, optional): Maximum intensity of the generated Normalmap. Defaults to 50.0.

    Returns:
        Tensor: Bevel image.
    """        
    grayscale_input_check(img_in, 'input image')

    dist = to_tensor(dist * 2.0 - 1.0) * max_dist
    smoothing = to_tensor(smoothing) * max_smoothing

    # height
    height = img_in
    if dist > 0:
        height = distance(height, None, combine=True, dist=dist * 128, max_dist=1.0)
    elif dist < 0:
        height = invert(height)
        height = distance(height, None, combine=True, dist=-dist * 128, max_dist=1.0)
        height = invert(height)
    if smoothing > 0:
        if non_uniform_blur_flag:
            img_blur = blur(height, 0.5, 1.0)
            img_blur = levels(img_blur, 0.0, 0.5, 0.0214, 0.0, 1.0)
            height = non_uniform_blur(height, img_blur, 6, 5, smoothing, 1.0, 0.0, 0.0, 0.0)
        else:
            height = blur_hq(height, False, smoothing, 1.0)
    
    # normal
    normal_intensity = to_tensor(normal_intensity) * max_normal_intensity
    normal_one = transform_2d(height, mipmap_mode='manual', x1=to_zero_one(-1.0), y2=to_zero_one(-1.0))
    normal_one = normal(normal_one, use_alpha=use_alpha, intensity=to_zero_one(1.0), max_intensity=normal_intensity)
    normal_one = transform_2d(normal_one, mipmap_mode='manual', x1=to_zero_one(-1.0), y2=to_zero_one(-1.0))
    normal_one = levels(normal_one, [1.0,1.0,0.0,0.0], [0.5]*4, [0.0,0.0,1.0,1.0], [0.0,0.0,0.0,1.0], [1.0]*4)

    normal_two = normal(height, use_alpha=use_alpha, intensity=to_zero_one(1.0), max_intensity=normal_intensity)
    normal_two = levels(normal_two, [0.0]*4, [0.5]*4, [1.0]*4, [0.0,0.0,0.0,1.0], [1.0]*4)

    normal_out = blend(normal_one, normal_two, None, 'copy', opacity=0.5)

    return height, normal_out

@input_check(2)
def slope_blur(img_in, img_mask, samples=1, mode='blur', intensity=10.0 / 16.0, max_intensity=16.0):
    """Non-atomic function: Slope Blur (https://docs.substance3d.com/sddoc/slope-blur-159450467.html)

    Args:
        img_in (tensor): Input image.
        img_mask (tensor): Mask image.
        samples (int, optional): Amount of samples, affects the quality at the expense of speed. Defaults to 1.
        mode (str, optional): Blending mode for consequent blur passes. "Blur" behaves more like a standard 
            Anisotropic Blur, while Min will "eat away" existing areas and Max will "smear out" white areas. 
            Defaults to 'blur'.
        intensity (tensor, optional): Normalized blur amount or strength. Defaults to 10.0/16.0.
        max_intensity (float, optional): Maximum blur amount or strength. Defaults to 16.0.

    Returns:
        Tensor: Slope blurred image.
    """
    grayscale_input_check(img_mask, 'slope map')

    assert isinstance(samples, int) and samples >= 1 and samples <= 32
    assert mode in ['blur', 'min', 'max']
    intensity = to_tensor(intensity) * max_intensity
    if intensity == 0 or th.min(img_in) == th.max(img_in):
        return img_in
    # progressive warping and blending
    warp_intensity = intensity / samples
    img_warp = warp(img_in, img_mask, warp_intensity, 1.0)
    img_out = img_warp
    blending_mode = 'copy' if mode == 'blur' else mode
    for i in range(2, samples + 1):
        img_warp_next = warp(img_warp, img_mask, warp_intensity, 1.0)
        img_out = blend(img_warp_next, img_out, None, blending_mode, opacity=1 / i)
        img_warp = img_warp_next
    return img_out

@input_check(2)
def mosaic(img_in, img_mask, samples=1, intensity=0.5, max_intensity=1.0):
    """Non-atomic function: Mosaic (https://docs.substance3d.com/sddoc/mosaic-159450535.html)

    Args:
        img_in (tensor): Input image.
        img_mask (tensor): Mask image.
        samples (int, optional): Determines multi-sample quality. Defaults to 1.
        intensity (float, optional): Normalized strength of the effect. Defaults to 0.5.
        max_intensity (float, optional): Maximum strength of the effect. Defaults to 1.0.

    Returns:
        Tensor: Mosaic image.
    """
    grayscale_input_check(img_mask, 'warp map')

    assert isinstance(samples, int) and samples >= 1 and samples <= 16
    intensity = to_tensor(intensity) * max_intensity
    if intensity == 0 or th.min(img_in) == th.max(img_in):
        return img_in
    # progressive warping
    warp_intensity = intensity / samples
    img_out = img_in
    for i in range(samples):
        img_out = warp(img_out, img_mask, warp_intensity, 1.0)
    return img_out

@input_check(1)
def auto_levels(img_in):
    """Non-atomic function: Auto Levels (https://docs.substance3d.com/sddoc/auto-levels-159449154.html)

    Args:
        img_in (tensor): Input image.

    Returns:
        Tensor: Auto leveled image.
    """
    grayscale_input_check(img_in, 'input image')

    max_val, min_val = th.max(img_in), th.min(img_in)
    # when input is a uniform image, and pixel value smaller (greater) than 0.5
    # output a white (black) image
    if max_val == min_val and max_val <= 0.5:
        img_out = (img_in - min_val + 1e-15) / (max_val - min_val + 1e-15)
    else:
        img_out = (img_in - min_val) / (max_val - min_val + 1e-15)
    return img_out

@input_check(1)
def ambient_occlusion(img_in, spreading=0.15, max_spreading=1.0, equalizer=[0.0, 0.0, 0.0], levels_param=[0.0, 0.5, 1.0]):
    """Non-atomic function: Ambient Occlusion (deprecated)

    Args:
        img_in (tensor): Input image.
        spreading (float, optional): . Defaults to 0.15.
        max_spreading (float, optional): . Defaults to 1.0.
        equalizer (list, optional): . Defaults to [0.0, 0.0, 0.0].
        levels_param (list, optional): . Defaults to [0.0, 0.5, 1.0].

    Returns:
        Tensor: ambient occlusion image
    """
    grayscale_input_check(img_in, 'input image')

    # Process parameters
    spreading = to_tensor(spreading) * max_spreading
    equalizer = to_tensor(equalizer)
    levels_param = to_tensor(levels_param)

    # Initial processing
    img_blur = blur_hq(1.0 - img_in, intensity=spreading, max_intensity=128.0)
    img_ao = blend(img_blur, img_in, blending_mode='add')
    img_ao = levels(img_ao, in_low=0.5)
    img_gs = c2g(normal(img_in, intensity=to_zero_one(1.0), max_intensity=16.0), rgba_weights=[0.0, 0.0, 1.0])
    img_ao_2 = blend(img_ao, 1.0 - img_gs, blending_mode='add')
    img_ao = blend(img_ao, img_ao_2, blending_mode='multiply')

    # Further processing
    img_ao_blur = blur_hq(manual_resize(img_ao, -1), intensity=1.0, max_intensity=2.2)
    img_ao_blur_2 = blur_hq(manual_resize(img_ao_blur, -1), intensity=1.0, max_intensity=3.3)
    img_blend = blend(manual_resize(1.0 - img_ao_blur, 1), img_ao, blending_mode='add_sub', opacity=0.5)
    img_blend_1 = blend(manual_resize(1.0 - img_ao_blur_2, 1), img_ao_blur, blending_mode='add_sub', opacity=0.5)

    img_ao_blur_2 = levels(img_ao_blur_2, in_mid=(equalizer[0] + 1) * 0.5)
    img_blend_1 = blend(img_blend_1, manual_resize(img_ao_blur_2, 1), blending_mode='add_sub',
                        opacity=th.clamp(equalizer[1] + 0.5, 0.0, 1.0))
    img_blend = blend(img_blend, manual_resize(img_blend_1, 1), blending_mode='add_sub',
                      opacity=th.clamp(equalizer[2] + 0.5, 0.0, 1.0))
    img_ao = levels(img_blend, in_low=levels_param[0], in_mid=levels_param[1], in_high=levels_param[2])

    return img_ao

@input_check(1)
def hbao(img_in, quality=4, depth=0.1, radius=1.0):
    """Non-atomic function: Ambient Occlusion (HBAO) (https://docs.substance3d.com/sddoc/ambient-occlusion-hbao-filter-node-159450550.html)

    Args:
        img_in (tensor): Input image.
        quality (int, optional): Amount of samples used for calculation. Defaults to 4.
        depth (float, optional): Height depth. Defaults to 0.1.
        radius (float, optional): The spread of the AO. Defaults to 1.0.

    Returns:
        Tensor: HBAO image.
    """
    grayscale_input_check(img_in, 'input image')
    assert quality in [4, 8, 16], 'quality must be 4, 8, or 16'
    num_row = img_in.shape[2]
    num_col = img_in.shape[3]
    pixel_size = 1.0 / max(num_row, num_col)
    min_size_log2 = int(np.log2(min(num_row, num_col)))

    # Performance triggers
    full_upsampling = True      # Enable full-sized mipmap sampling (identical results to sbs)
    batch_processing = False    # Enable batched HBAO cone sampling (20% faster; higher GPU memory cost)

    # Process input parameters
    depth = to_tensor(depth) * min(num_row, num_col)
    radius = to_tensor(radius)

    # Create mipmap stack
    in_low = levels(img_in, 0.0, 0.5, 1.0, 0.0, 0.5)
    in_high = levels(img_in, 0.0, 0.5, 1.0, 0.5, 1.0)
    mipmaps_level = 11
    mipmaps = create_mipmaps(in_high, mipmaps_level, keep_size=full_upsampling)

    # Precompute weights
    weights = [hbao_radius(min_size_log2, i + 1, radius) for i in range(mipmaps_level)]

    # HBAO cone sampling
    img_out = th.zeros_like(img_in)
    row_grid_init, col_grid_init = th.meshgrid(th.linspace(0, num_row - 1, num_row), th.linspace(0, num_col - 1, num_col))
    row_grid_init = (row_grid_init + 0.5) / num_row
    col_grid_init = (col_grid_init + 0.5) / num_col

    # Sampling all cones together
    if batch_processing:
        sample_grid_init = th.stack([col_grid_init, row_grid_init], 2)
        angle_vec = lambda i: to_tensor([np.cos(i * np.pi * 2.0 / quality), np.sin(i * np.pi * 2.0 / quality)])
        img_sample = th.zeros_like(img_in)

        # perform sampling on each mipmap level
        for mm_idx, img_mm in enumerate(mipmaps):
            mm_scale = 2.0 ** (mm_idx + 1)
            mm_row, mm_col = img_mm.shape[2], img_mm.shape[3]
            sample_grid = th.stack([th.remainder(sample_grid_init + mm_scale * pixel_size * angle_vec(i), 1.0) * 2.0 - 1.0 \
                                    for i in range(quality)])
            sample_grid = sample_grid * to_tensor([mm_col / (mm_col + 2), mm_row / (mm_row + 2)])
            img_mm = img_mm.view(1, img_in.shape[0], mm_row, mm_col).expand(quality, img_in.shape[0], mm_row, mm_col)
            img_mm_pad = th.nn.functional.pad(img_mm, [1, 1, 1, 1], mode='circular')
            img_mm_gs = th.nn.functional.grid_sample(img_mm_pad, sample_grid, 'bilinear', 'zeros', align_corners=False)

            img_diff = (img_mm_gs - in_low - 0.5) / mm_scale
            img_max = th.max(img_max, img_diff) if mm_idx else img_diff
            img_sample = lerp(img_sample, img_max, weights[mm_idx])

        # integrate into sampled image
        img_sample = img_sample * depth * 2.0
        img_sample = img_sample / th.sqrt(img_sample * img_sample + 1.0)
        img_out = th.sum(img_sample, 0, keepdim=True).view_as(img_in)

    # Sampling each cone individually
    else:
        for i in range(quality):
            cone_angle = i * np.pi * 2.0 / quality
            sin_angle = np.sin(cone_angle)
            cos_angle = np.cos(cone_angle)
            img_sample = th.zeros_like(img_in)

            # perform sampling on each mipmap level
            for mm_idx, img_mm in enumerate(mipmaps):
                mm_scale = 2.0 ** (mm_idx + 1)
                mm_row, mm_col = img_mm.shape[2], img_mm.shape[3]
                row_grid = th.remainder(row_grid_init + mm_scale * pixel_size * sin_angle, 1.0) * 2.0 - 1.0
                col_grid = th.remainder(col_grid_init + mm_scale * pixel_size * cos_angle, 1.0) * 2.0 - 1.0
                row_grid = row_grid * mm_row / (mm_row + 2)
                col_grid = col_grid * mm_col / (mm_col + 2)
                sample_grid = th.stack([col_grid, row_grid], 2).expand(img_in.shape[0], num_row, num_col, 2)
                img_mm_pad = th.nn.functional.pad(img_mm, [1, 1, 1, 1], mode='circular')
                img_mm_gs = th.nn.functional.grid_sample(img_mm_pad, sample_grid, 'bilinear', 'zeros', align_corners=False)

                img_diff = (img_mm_gs - in_low - 0.5) / mm_scale
                img_max = img_diff if mm_idx == 0 else th.max(img_max, img_diff)
                img_sample = lerp(img_sample, img_max, weights[mm_idx])

            # integrate into sampled image
            img_sample = img_sample * depth * 2.0
            img_sample = img_sample / th.sqrt(img_sample * img_sample + 1.0)
            img_out = img_out + img_sample

    # final output
    img_out = th.clamp(1.0 - img_out / quality, 0.0, 1.0)

    return img_out

@input_check(1)
def highpass(img_in, radius=6.0/64.0, max_radius=64.0):
    """Non-atomic function: Highpass (https://docs.substance3d.com/sddoc/highpass-159449203.html)

    Args:
        img_in (tensor): Input image.
        radius (float, optional): A small radius removes small differences, 
            a bigger radius removes large areas. Defaults to 6.0/64.0.
        max_radius (float, optional): Maximum value of 'radius'. Defaults to 64.0.

    Returns:
        Tensor: Highpass filtered image.
    """
    radius = to_tensor(radius) * max_radius
    img_out = blur(img_in, radius, 1.0)
    img_out = invert(img_out)
    img_out = blend(img_out, img_in, None, 'add_sub', opacity=0.5)
    return img_out

@input_check(1)
def normal_normalize(normal):
    """Non-atomic function: Normal Normalize (https://docs.substance3d.com/sddoc/normal-normalize-159450586.html)

    Args:
        normal (tensor): Normal image.

    Returns:
        tensor: Normal normalized image.
    """
    color_input_check(normal, 'input image')

    # reimplementation of the internal pixel processor (latest implementation)
    normal_rgb = normal[:,:3,:,:] if normal.shape[1] == 4 else normal
    normal_rgb = normal_rgb * 2.0 - 1.0
    normal_length = th.norm(normal_rgb, p=2, dim=1, keepdim=True)
    normal_rgb = normal_rgb / normal_length
    normal_rgb = normal_rgb / 2.0 + 0.5
    normal = th.cat([normal_rgb, normal[:,3,:,:].unsqueeze(1)], dim=1) if normal.shape[1] == 4 else normal_rgb
    return normal

@input_check(1)
def channel_mixer(img_in, monochrome=False, red=[0.75,0.5,0.5,0.5], green=[0.5,0.75,0.5,0.5], blue=[0.5,0.5,0.75,0.5]):
    """Non-atomic function: Channel Mixer (https://docs.substance3d.com/sddoc/channel-mixer-159449157.html)

    Args:
        img_in (tensor): Input image.
        monochrome (bool, optional): Output monochrome image. Defaults to False.
        red (list, optional): Mixing weights for output red channel. Defaults to [0.75,0.5,0.5,0.5].
        green (list, optional): Mixing weights for output green channel. Defaults to [0.5,0.75,0.5,0.5].
        blue (list, optional): Mixing weights for output blue channel. Defaults to [0.5,0.5,0.75,0.5].

    Returns:
        Tensor: Channel mixed image.
    """
    color_input_check(img_in, 'input image')

    # scale to range [-2,2]
    red = (to_tensor(red) - 0.5) * 4
    green = (to_tensor(green) - 0.5) *4
    blue = (to_tensor(blue) - 0.5)* 4

    weight = [red, green, blue]
    img_out = th.zeros_like(img_in)
    active_channels = 1 if monochrome else 3
    for i in range(active_channels):
        for j in range(3):
            img_out[:,i,:,:] = img_out[:,i,:,:] + img_in[:,j,:,:] * weight[i][j]
        img_out[:,i,:,:] = img_out[:,i,:,:] + weight[i][3]
    img_out = th.clamp(img_out, 0.0, 1.0)
    
    if monochrome:
        img_out = img_out[:,0,:,:].unsqueeze(1)

    return img_out

@input_check(2)
def normal_combine(normal_one, normal_two, mode='whiteout'):
    """Non-atomic function: Normal Combine (https://docs.substance3d.com/sddoc/normal-combine-159450580.html)

    Args:
        normal_one (tensor): First normal image.
        normal_two (tensor): Second normal image.
        mode (str, optional): 'whiteout'|'channel_mixer'|'detail_oriented'. Defaults to 'whiteout'.

    Returns:
        Tensor: Normal combined image.
    """
    color_input_check(normal_one, 'normal one')
    color_input_check(normal_two, 'normal two')
    assert normal_one.shape == normal_two.shape, "two input normals don't have same shape"

    # top branch
    normal_one_r = normal_one[:,0,:,:].unsqueeze(1)
    normal_one_g = normal_one[:,1,:,:].unsqueeze(1)
    normal_one_b = normal_one[:,2,:,:].unsqueeze(1)

    normal_two_r = normal_two[:,0,:,:].unsqueeze(1)
    normal_two_g = normal_two[:,1,:,:].unsqueeze(1)
    normal_two_b = normal_two[:,2,:,:].unsqueeze(1)

    if mode == 'whiteout':
        r_blend = blend(normal_one_r, normal_two_r, None, 'add_sub', opacity=0.5)
        g_blend = blend(normal_one_g, normal_two_g, None, 'add_sub', opacity=0.5)
        b_blend = blend(normal_one_b, normal_two_b, None, 'multiply', opacity=1.0)

        rgb_blend = th.cat([r_blend, g_blend, b_blend], dim=1)
        normal_out = normal_normalize(rgb_blend)

        if normal_one.shape[1] == 4:
            normal_out = th.cat([normal_out, th.ones(normal_out.shape[0], 1,\
                    normal_out.shape[2], normal_out.shape[3])], dim=1)

    elif mode == 'channel_mixer':
        # middle left branch 
        normal_two_levels_one = levels(normal_two, [0.5,0.5,0.0,0.0], [0.5]*4, [1.0] * 4, [0.5,0.5,0.0,1.0], [1.0,1.0,0.0,1.0])
        normal_two_levels_one[:,:2,:,:] = normal_two_levels_one[:,:2,:,:] - 0.5

        normal_two_levels_two = levels(normal_two, [0.0]*4, [0.5]*4, [0.5,0.5,1.0,1.0], [0.0,0.0,0.0,1.0], [0.5,0.5,1.0,1.0])
        normal_two_levels_two[:,:2,:,:] = -normal_two_levels_two[:,:2,:,:] + 0.5
        normal_two_levels_two[:,2,:,:] = -normal_two_levels_two[:,2,:,:] + 1.0

        # bottom left branch
        grayscale_blend = blend(normal_two_b, normal_one_b, None, 'min', opacity=1.0)

        # bottom middle branch
        cm_normal_one_blend = blend(normal_two_levels_two, normal_one, None, 'subtract', opacity=1.0)
        normal_out = blend(normal_two_levels_one, cm_normal_one_blend, None, 'add', opacity=1.0)
        normal_out[:,2,:,:] = grayscale_blend

    elif mode == 'detail_oriented':
        # implement pixel processorggb_rgb_temp
        r_one_temp = normal_one_r * 2.0 - 1.0
        g_one_temp = normal_one_g * 2.0 - 1.0
        b_one_temp = normal_one_b * 2.0 - 1.0
        b_invert_one_temp = 1.0 / (normal_one_b + 1.0)
        rg_one_temp = -r_one_temp * g_one_temp  
        rgb_one_temp = b_invert_one_temp * rg_one_temp
        rrb_one_temp = 1.0 - r_one_temp * r_one_temp * b_invert_one_temp 
        ggb_one_temp = 1.0 - g_one_temp * g_one_temp * b_invert_one_temp

        rrb_rgb_temp = th.cat([rrb_one_temp, rgb_one_temp, -r_one_temp], dim=1)
        rrb_rgb_if   = th.zeros_like(rrb_rgb_temp)
        rrb_rgb_if[:,1,:,:] = -1.0
        rrb_rgb_temp[normal_one_b.expand(-1,3,-1,-1) < -0.9999] = rrb_rgb_if[normal_one_b.expand(-1,3,-1,-1) < -0.9999]
        ggb_rgb_temp = th.cat([rgb_one_temp, ggb_one_temp, -g_one_temp], dim=1)
        ggb_rgb_if   = th.zeros_like(ggb_rgb_temp)
        ggb_rgb_if[:,0,:,:] = -1.0
        ggb_rgb_temp[normal_one_b.expand(-1,3,-1,-1) < -0.9999] = ggb_rgb_if[normal_one_b.expand(-1,3,-1,-1) < -0.9999]

        rrb_rgb_temp = rrb_rgb_temp * (normal_two_r * 2.0 - 1.0)
        ggb_rgb_temp = ggb_rgb_temp * (normal_two_g * 2.0 - 1.0)
        b_rgb_temp = (normal_one[:,:3,:,:] * 2.0 - 1.0) * (normal_two_b * 2.0 - 1.0)
        normal_out = (rrb_rgb_temp + ggb_rgb_temp + b_rgb_temp) * 0.5 + 0.5

        if normal_one.shape[1] == 4:
            normal_out = th.cat([normal_out, th.ones(normal_out.shape[0], 1,\
                    normal_out.shape[2], normal_out.shape[3])], dim=1)

    else:
        raise ValueError("Can't recognized the mode")

    return normal_out

@input_check(1)
def height_to_normal_world_units(img_in, normal_format='gl', sampling_mode='standard', use_alpha=False, surface_size=0.3, max_surface_size=1000.0,
                                height_depth=0.16, max_height_depth=100.0):
    """Non-atomic function: Height to Normal World Units (https://docs.substance3d.com/sddoc/height-to-normal-world-units-159450573.html)

    Args:
        img_in (tensor): Input image.
        normal_format (str, optional): 'gl' or 'dx'. Defaults to 'gl'.
        sampling_mode (str, optional): 'standard' or 'sobel', switches between two sampling modes determining accuracy. Defaults to 'standard'.
        use_alpha (bool, optional): Enable the alpha channel. Defaults to False.
        surface_size (float, optional): Normalized dimensions of the input Heightmap. Defaults to 0.3.
        max_surface_size (float, optional): Maximum dimensions of the input Heightmap (cm). Defaults to 1000.0.
        height_depth (float, optional): Normalized depth of heightmap details. Defaults to 0.16.
        max_height_depth (float, optional): Maximum depth of heightmap details. Defaults to 100.0.

    Returns:
        Tensor: Normal image.
    """
    # Check input validity
    grayscale_input_check(img_in, 'input image')
    assert normal_format in ('dx', 'gl'), "normal format must be 'dx' or 'gl'"
    assert sampling_mode in ('standard', 'sobel'), "sampling mode must be 'standard' or 'sobel'"

    surface_size = to_tensor(surface_size) * max_surface_size
    height_depth = to_tensor(height_depth) * max_height_depth
    res_x, inv_res_x = img_in.shape[2], 1.0 / img_in.shape[2]
    res_y, inv_res_y = img_in.shape[3], 1.0 / img_in.shape[3]

    # Standard normal conversion
    if sampling_mode == 'standard':
        img_out = normal(img_in, normal_format=normal_format, use_alpha=use_alpha, intensity=to_zero_one(height_depth / surface_size), max_intensity=256.0)
    # Sobel sampling
    else:
        # Convolution
        db_x = d_blur(img_in, inv_res_x, 256.0)
        db_y = d_blur(img_in, inv_res_y, 256.0, angle=0.25)
        db_x = th.nn.functional.pad(db_x, (0, 0, 1, 1), mode='circular')
        db_y = th.nn.functional.pad(db_y, (1, 1, 0, 0), mode='circular')
        sample_x = th.nn.functional.conv2d(db_y, th.linspace(1.0, -1.0, 3).view((1, 1, 1, 3)))
        sample_y = th.nn.functional.conv2d(db_x, th.linspace(-1.0, 1.0, 3).view((1, 1, 3, 1)))

        # Multiplier
        mult_x = res_x * height_depth * 0.5 / surface_size
        mult_y = (-1.0 if normal_format == 'dx' else 1.0) * res_y * height_depth * 0.5 / surface_size
        sample_x = sample_x * mult_x * (1.0 if res_x < res_y else res_y / res_x)
        sample_y = sample_y * mult_y * (res_x / res_y if res_x < res_y else 1.0)

        # Output
        scale = 0.5 / th.sqrt(sample_x ** 2 + sample_y ** 2 + 1)
        img_out = th.cat([sample_x, sample_y, th.ones_like(img_in)], dim=1) * scale + 0.5

        # Add opaque alpha channel
        if use_alpha:
            img_out = th.cat([img_out, th.ones_like(img_in)], dim=1)

    return img_out

@input_check(1)
def normal_to_height(img_in, normal_format='dx', relief_balance=[0.5, 0.5, 0.5], opacity=0.36, max_opacity=1.0):
    """Non-atomic function: Normal to Height (https://docs.substance3d.com/sddoc/normal-to-height-159450591.html)

    Args:
        img_in (tensor): Input image.
        normal_format (str, optional): 'gl' or 'dx'. Defaults to 'dx'.
        relief_balance (list, optional): Adjust the extent to which the different frequencies influence the final result. 
        This is largely dependent on the input map and requires a fair bit of tweaking. Defaults to [0.5, 0.5, 0.5].
        opacity (float, optional): Normalized global opacity of the effect. Defaults to 0.36.
        max_opacity (float, optional): Maximum global opacity of the effect. Defaults to 1.0.

    Returns:
        Tensor: Height image.
    """
    color_input_check(img_in, 'input image')
    assert img_in.shape[2] == img_in.shape[3], 'input image must be in square shape'
    in_size = img_in.shape[2]
    in_size_log2 = int(np.log2(in_size))
    assert in_size_log2 >= 7, 'input size must be at least 128'

    # Construct variables
    low_freq = to_tensor(relief_balance[0])
    mid_freq = to_tensor(relief_balance[1])
    high_freq = to_tensor(relief_balance[2])
    opacity = to_tensor(opacity) * max_opacity

    # Frequency transform for R and G channels
    img_freqs = frequency_transform(img_in[:,:2,:,:], normal_format)
    img_blend = [None, None]

    # Low frequencies (for 16x16 images only)
    for i in range(4):
        for c in (0, 1):
            img_i_c = img_freqs[c][i]
            blend_opacity = th.clamp(0.0625 * 2 * (8 >> i) * low_freq * 100 * opacity, 0.0, 1.0)
            img_blend[c] = img_i_c if img_blend[c] is None else blend(img_i_c, img_blend[c], blending_mode='add_sub', opacity=blend_opacity)

    # Mid frequencies
    for i in range(min(2, len(img_freqs[0]) - 4)):
        for c in (0, 1):
            img_i_c = img_freqs[c][i + 4]
            blend_opacity = th.clamp(0.0156 * 2 * (2 >> i) * mid_freq * 100 * opacity, 0.0, 1.0)
            img_blend[c] = blend(img_i_c, manual_resize(img_blend[c], 1), blending_mode='add_sub', opacity=blend_opacity)

    # High frequencies
    for i in range(min(6, len(img_freqs[0]) - 6)):
        for c in (0, 1):
            img_i_c = img_freqs[c][i + 6]
            blend_opacity = th.clamp(0.0078 * 0.0625 * (32 >> i) * high_freq * 100 * opacity, 0.0, 1.0) if i < 5 else \
                            th.clamp(0.0078 * 0.0612 * high_freq * 100 * opacity)
            img_blend[c] = blend(img_i_c, manual_resize(img_blend[c], 1), blending_mode='add_sub', opacity=blend_opacity)

    # Combine both channels
    img_out = blend(img_blend[0], img_blend[1], blending_mode='add_sub', opacity=0.5)
    return img_out

@input_check(1)
def curvature_smooth(img_in, normal_format='dx'):
    """Non-atomic function: Curvature Smooth (https://docs.substance3d.com/sddoc/curvature-smooth-159450517.html)

    Args:
        img_in (tensor): Input normal image.
        normal_format (str, optional): 'gl' or 'dx'. Defaults to 'dx'.

    Returns:
        Tensor: Curvature smooth image.
    """
    # Check input validity
    color_input_check(img_in, 'input image')
    assert img_in.shape[2] == img_in.shape[3], 'input image must be in square shape'
    assert img_in.shape[2] >= 16, 'input size must be at least 16'
    assert normal_format in ('dx', 'gl')

    # Frequency transform for R and G channels
    img_freqs = frequency_transform(img_in[:,:2,:,:], normal_format)
    img_blend = [img_freqs[0][0], img_freqs[1][0]]

    # Low frequencies (for 16x16 images only)
    for i in range(1, 4):
        for c in (0, 1):
            img_i_c = img_freqs[c][i]
            img_blend[c] = blend(img_i_c, img_blend[c], blending_mode='add_sub', opacity=0.25)

    # Other frequencies
    for i in range(len(img_freqs[0]) - 4):
        for c in (0, 1):
            img_i_c = img_freqs[c][i + 4]
            img_blend[c] = blend(img_i_c, manual_resize(img_blend[c], 1), blending_mode='add_sub', opacity=1.0 / (i + 5))

    # Combine both channels
    img_out = blend(img_blend[0], img_blend[1], blending_mode='add_sub', opacity=0.5)
    return img_out

@input_check(0)
def multi_switch(img_list, input_number, input_selection):
    """Non-atomic function: Multi Switch (https://docs.substance3d.com/sddoc/multi-switch-159450377.html)

    Args:
        img_list (list): A list of input images.
        input_number (int): Amount of inputs to expose.
        input_selection (int): Which input to return as the result.

    Returns:
        Tensor: The selected input image.
    """
    assert input_number == len(img_list), "input number does not match the length of image list"
    assert input_selection <= input_number, "input selection should be less equal then the input number"
    if img_list[0].shape[1] == 1:
        for i, img in enumerate(img_list):
            grayscale_input_check(img, 'input image %d' % i)
    else:
        for i, img in enumerate(img_list):
            color_input_check(img, 'input image %d' % i)
    return img_list[input_selection-1]

@input_check(1)
def rgba_split(rgba):
    """Non-atomic function: RGBA Split (https://docs.substance3d.com/sddoc/rgba-split-159450492.html)

    Args:
        rgba (tensor): RGBA input image.

    Returns:
        Tensor: 4 single-channel images.
    """
    color_input_check(rgba, 'input image')
    assert rgba.shape[1] == 4, 'input image must contain alpha channel'
    r = rgba[:,[0],:,:].clone()
    g = rgba[:,[1],:,:].clone()
    b = rgba[:,[2],:,:].clone()
    a = rgba[:,[3],:,:].clone()
    return r, g, b, a

@input_check(0)
def rgba_merge(r, g, b, a=None, use_alpha=False):
    """Non-atomic function: RGBA Merge (https://docs.substance3d.com/sddoc/rgba-merge-159450486.html)

    Args:
        r (tensor): Red channel.
        g (tensor): Green channel.
        b (tensor): Blue channel.
        a (tensor, optional): Alpha channel. Defaults to None.
        use_alpha (bool, optional): Enable the alpha channel. Defaults to False.

    Returns:
        Tensor: RGBA image merged from the input 4 single-channel images.
    """
    channels = [r,g,b,a]
    active_channels = 4 if use_alpha else 3
    for i in range(3):
        grayscale_input_check(channels[i], 'channel image')
    assert channels[0].shape == channels[1].shape == channels[2].shape, "rgb channels doesn't have same shape"

    if use_alpha and channels[3] is None:
        channels[3] = th.ones_like(channels[0])
    img_out = th.cat(channels[:active_channels], dim=1)
    return img_out

@input_check(3)
def pbr_converter(base_color, roughness, metallic, use_alpha=False):
    """Non-atomic function: BaseColor / Metallic / Roughness converter 
    (https://docs.substance3d.com/sddoc/basecolor-metallic-roughness-converter-159451054.html)

    Args:
        base_color (tensor): Base color map.
        roughness (tensor): Roughness map.
        metallic (tensor): Metallic map.
        use_alpha (bool, optional): Enable the alpha channel. Defaults to False.

    Returns:
        Tensor: Diffuse, specular and glossiness maps.
    """
    grayscale_input_check(roughness, 'roughness')
    grayscale_input_check(metallic, 'metallic')
    
    # compute diffuse
    invert_metallic = levels(metallic, out_low=1.0, out_high=0.0)
    invert_metallic_sRGB = linear_to_srgb(invert_metallic)
    invert_metallic_sRGB = levels(invert_metallic_sRGB, out_low=1.0, out_high=0.0)
    black = th.zeros_like(base_color)
    if use_alpha and base_color.shape[1] == 4:
        black[:,3,:,:] = 1.0
    diffuse = blend(black, base_color, invert_metallic_sRGB)
    
    # compute specular
    base_color_linear = srgb_to_linear(base_color)
    specular_blend = blend(black, base_color_linear, invert_metallic)
    specular_levels = th.clamp(levels(invert_metallic, out_high = 0.04), 0.0, 1.0)
    if use_alpha:
        specular_levels = th.cat([specular_levels,specular_levels,specular_levels,th.ones_like(specular_levels)], dim=1)
    else:
        specular_levels = th.cat([specular_levels,specular_levels,specular_levels], dim=1)

    specular_blend_2 = blend(specular_levels, specular_blend)
    specular = linear_to_srgb(specular_blend_2)

    # compute glossiness
    glossiness = levels(roughness, out_low=1.0, out_high=0.0)
    return diffuse, specular, glossiness

@input_check(1)
def alpha_split(rgba):
    """Non-atomic function: Alpha Split (https://docs.substance3d.com/sddoc/alpha-split-159450495.html)

    Args:
        rgba (tensor): Rgba input image.

    Returns:
        Tensor: Rgb and a images.
    """
    color_input_check(rgba, 'rgba image')
    assert rgba.shape[1] == 4, 'image input must contain alpha channel'
    rgb = th.cat([rgba[:,:3,:,:], th.ones(rgba.shape[0], 1, rgba.shape[2], rgba.shape[3])], 1)
    a = rgba[:,[3],:,:].clone()
    return rgb, a

@input_check(2)
def alpha_merge(rgb, a):
    """Non-atomic function: Alpha Merge (https://docs.substance3d.com/sddoc/alpha-merge-159450489.html)

    Args:
        rgb (tensor): Rgb input image.
        a (tensor): Alpha input image.

    Returns:
        Tensor: Rgba input image.
    """
    color_input_check(rgb, 'rgb image')
    grayscale_input_check(a, 'alpha image')

    channels = [rgb[:,:3,:,:], a]
    res_h = max(rgb.shape[2], a.shape[2])
    res_w = max(rgb.shape[3], a.shape[3])

    for i, channel in enumerate(channels):
        assert res_h//channel.shape[2] == res_w//channel.shape[3], "rectangular input is not supported"
        if res_h//channel.shape[2] != 1:
            channels[i] = manual_resize(channel, res_h//channel.shape[2])

    img_out = th.cat(channels, dim=1)
    return img_out

@input_check(2)
def switch(img_1, img_2, flag=True):
    """Non-atomic function: Switch (https://docs.substance3d.com/sddoc/switch-159450385.html)

    Args:
        img_1 (tensor): First input image.
        img_2 (tensor): Second input image.
        flag (bool, optional): Output first image if True. Defaults to True.

    Returns:
        Tensor : First or second input image.
    """
    if flag:
        return img_1
    else:
        return img_2

@input_check(2)
def normal_blend(normal_fg, normal_bg, mask=None, use_mask=True, opacity=1.0):
    """Non-atomic function: Normal Blend (https://docs.substance3d.com/sddoc/normal-blend-159450576.html)

    Args:
        normal_fg (tensor): Foreground normal.
        normal_bg (tensor): Background normal.
        mask (tensor, optional): Mask slot used for masking the node's effects. Defaults to None.
        use_mask (bool, optional): Use mask if True. Defaults to True.
        opacity (float, optional): Blending opacity between foreground and background. Defaults to 1.0.

    Returns:
        Tensor: Blended normal image.
    """
    color_input_check(normal_fg, 'normal foreground')
    color_input_check(normal_bg, 'normal background')
    assert normal_fg.shape == normal_bg.shape, 'the shape of normal fg and bg does not match'
    if mask is not None:
        grayscale_input_check(mask, 'mask')
        assert normal_fg.shape[2] == mask.shape[2] and normal_fg.shape[3] == mask.shape[3], 'the shape of normal fg and bg does not match'

    opacity = to_tensor(opacity)
    if use_mask and mask is not None:
        mask_blend = blend(mask, th.zeros_like(mask), opacity=opacity)
    else:
        dummy_mask = th.ones(normal_fg.shape[0], 1, normal_fg.shape[2], normal_fg.shape[3])
        dummy_mask_2 = th.zeros(normal_fg.shape[0], 1, normal_fg.shape[2], normal_fg.shape[3])
        mask_blend = blend(dummy_mask, dummy_mask_2, opacity=opacity)
    
    out_normal = blend(normal_fg[:,:3,:,:], normal_bg[:,:3,:,:], mask_blend)

    # only when both normal inputs have alpha, process and append alpha
    if normal_fg.shape[1] == 4 and normal_bg.shape[1] == 4:    
        out_normal_alpha = blend(normal_fg[:,3,:,:], normal_bg[:,3,:,:], mask_blend)
        out_normal = th.cat([out_normal, out_normal_alpha], dim=1)

    out_normal = normal_normalize(out_normal)
    return out_normal    

@input_check(1)
def mirror(img_in, mirror_axis=0, invert_axis=False, corner_type=0, offset=0.5):
    """Non-atomic function: Mirror (https://docs.substance3d.com/sddoc/mirror-filter-node-159450617.html)

    Args:
        img_in (tensor): Input image
        mirror_axis (int, optional): 'x'|'y'|'corner'. Defaults to 0.
        invert_axis (bool, optional): Whether flip direction. Defaults to False.
        corner_type (int, optional): 'tl'|'tr'|'bl'|'br'. Defaults to 0.
        offset (float, optional): Where the axis locates. Defaults to 0.5.

    Returns:
        Tensor: Mirrored image.
    """
    res_h = img_in.shape[2]
    res_w = img_in.shape[3]
    mirror_axis_list = ['x', 'y', 'corner']
    corner_type_list = ['tl', 'tr', 'bl', 'br']
    mirror_axis = mirror_axis_list[mirror_axis]
    corner_type = corner_type_list[corner_type]

    if mirror_axis == 'x':
        axis_w = res_w * offset
        if (axis_w==0 and invert_axis==True) or (axis_w==res_w and invert_axis==False):
            return img_in

        if invert_axis:
            # invert image first
            img_in = th.flip(img_in, dims=[3])
            axis_w = res_w - axis_w

        # compute img_out_two
        double_offset = int(np.ceil(axis_w*2))
        axis_w_floor = int(np.floor(axis_w))
        axis_w_ceil = int(np.ceil(axis_w))
        if double_offset % 2 == 1:
            img_out = th.zeros_like(img_in)
            img_out[:,:,:,:axis_w_floor] = img_in[:,:,:,:axis_w_floor]
            if double_offset <= res_w:
                img_out[:,:,:,axis_w_floor:double_offset] = th.flip(img_in[:,:,:,:axis_w_floor+1], dims=[3])
                img_out[:,:,:,axis_w_floor:double_offset] = \
                    img_out[:,:,:,axis_w_floor:double_offset] * (1 - (double_offset - axis_w*2)) + \
                    img_out[:,:,:,axis_w_floor+1:double_offset+1] * (double_offset - axis_w*2)
            else:
                img_out[:,:,:,axis_w_floor:res_w] = \
                    th.flip(img_in[:,:,:,axis_w_floor-(res_w-axis_w_floor)+1:axis_w_floor+1], dims=[3]) * (1 - (double_offset - axis_w*2)) + \
                    th.flip(img_in[:,:,:,axis_w_floor-(res_w-axis_w_floor):axis_w_floor], dims=[3]) * (double_offset - axis_w*2)
        else:
            img_out = th.zeros_like(img_in)
            img_out[:,:,:,:axis_w_ceil] = img_in[:,:,:,:axis_w_ceil]
            if double_offset <= res_w:
                img_out[:,:,:,axis_w_ceil:double_offset] = th.flip(img_in[:,:,:,:axis_w_ceil], dims=[3])
                img_out[:,:,:,axis_w_ceil:double_offset] = \
                    img_out[:,:,:,axis_w_ceil:double_offset] * (1 - (double_offset - axis_w*2)) + \
                    img_out[:,:,:,axis_w_ceil+1:double_offset+1] * (double_offset - axis_w*2)
            else:
                img_out[:,:,:,axis_w_ceil:res_w] = \
                    th.flip(img_in[:,:,:,axis_w_ceil-(res_w-axis_w_ceil)+1:axis_w_ceil+1], dims=[3]) * (1 - (double_offset - axis_w*2)) + \
                    th.flip(img_in[:,:,:,axis_w_ceil-(res_w-axis_w_ceil):axis_w_ceil], dims=[3]) * (double_offset - axis_w*2)

        if invert_axis:
            img_out = th.flip(img_out, dims=[3])

    elif mirror_axis == 'y':
        axis_h = res_h * (1 - offset)
        if (axis_h==0 and invert_axis==True) or (axis_h==res_h and invert_axis==False):
            return img_in

        if invert_axis:
            # invert image first
            img_in = th.flip(img_in, dims=[2])
            axis_h = res_h - axis_h

        # compute img_out_two
        double_offset = int(np.ceil(axis_h*2))
        axis_h_floor = int(np.floor(axis_h))
        axis_h_ceil = int(np.ceil(axis_h))
        if double_offset % 2 == 1:
            img_out = th.zeros_like(img_in)
            img_out[:,:,:axis_h_floor,:] = img_in[:,:,:axis_h_floor,:]
            if double_offset <= res_h:
                img_out[:,:,axis_h_floor:double_offset,:] = th.flip(img_in[:,:,:axis_h_floor+1,:], dims=[2])
                img_out[:,:,axis_h_floor:double_offset,:] = \
                    img_out[:,:,axis_h_floor:double_offset,:] * (1 - (double_offset - axis_h*2)) + \
                    img_out[:,:,axis_h_floor+1:double_offset+1,:] * (double_offset - axis_h*2)
            else:
                img_out[:,:,axis_h_floor:res_h,:] = \
                    th.flip(img_in[:,:,axis_h_floor-(res_h-axis_h_floor)+1:axis_h_floor+1,:], dims=[2]) * (1 - (double_offset - axis_h*2)) + \
                    th.flip(img_in[:,:,axis_h_floor-(res_h-axis_h_floor):axis_h_floor,:], dims=[2]) * (double_offset - axis_h*2)
        else:
            img_out = th.zeros_like(img_in)
            img_out[:,:,:axis_h_ceil,:] = img_in[:,:,:axis_h_ceil,:]
            if double_offset <= res_h:
                img_out[:,:,axis_h_ceil:double_offset,:] = th.flip(img_in[:,:,:axis_h_ceil,:], dims=[2])
                img_out[:,:,axis_h_ceil:double_offset,:] = \
                    img_out[:,:,axis_h_ceil:double_offset,:] * (1 - (double_offset - axis_h*2)) + \
                    img_out[:,:,axis_h_ceil+1:double_offset+1,:] * (double_offset - axis_h*2)
            else:
                img_out[:,:,axis_h_ceil:res_h,:] = \
                    th.flip(img_in[:,:,axis_h_ceil-(res_h-axis_h_ceil)+1:axis_h_ceil+1,:], dims=[2]) * (1 - (double_offset - axis_h*2)) + \
                    th.flip(img_in[:,:,axis_h_ceil-(res_h-axis_h_ceil):axis_h_ceil,:], dims=[2]) * (double_offset - axis_h*2)

        if invert_axis:
            img_out = th.flip(img_out, dims=[2])

    elif mirror_axis == 'corner':
        img_out = img_in
        if corner_type == 'tl':
            # top right
            img_out[:,:, :res_h//2, res_w//2:] = th.flip(img_out[:,:, :res_h//2, :res_w//2], dims=[3])
            # bottom 
            img_out[:,:, res_h//2:, :] = th.flip(img_out[:,:, :res_h//2, :], dims=[2])
        elif corner_type == 'tr':
            # top left
            img_out[:,:, :res_h//2, :res_w//2] = th.flip(img_out[:,:, :res_h//2, res_w//2:], dims=[3])
            # bottom
            img_out[:,:, res_h//2:, :] = th.flip(img_out[:,:, :res_h//2, :], dims=[2])
        elif corner_type == 'bl':
            # bottom right
            img_out[:,:, res_h//2:, res_w//2:] = th.flip(img_out[:,:, res_h//2:, :res_w//2], dims=[3])
            # top
            img_out[:,:, :res_h//2, :] = th.flip(img_out[:,:, res_h//2:, :], dims=[2])
        elif corner_type == 'br':
            # bottom left
            img_out[:,:, res_h//2:, :res_w//2] = th.flip(img_out[:,:, res_h//2:, res_w//2:], dims=[3])
            # top
            img_out[:,:, :res_h//2, :] = th.flip(img_out[:,:, res_h//2:, :], dims=[2])
    else:
        raise ValueError("unknown mirror options")

    return img_out

@input_check(1)
def make_it_tile_patch(img_in, octave=3, seed=0, use_alpha=False, mask_size=1.0, max_mask_size=1.0, mask_precision=0.5, max_mask_precision=1.0,
                       mask_warping=0.5, max_mask_warping=100.0, pattern_width=0.2, max_pattern_width=1000.0, pattern_height=0.2, max_pattern_height=1000.0,
                       disorder=0.0, max_disorder=1.0, size_variation=0.0, max_size_variation=100.0, rotation=0.5, rotation_variation=0.0,
                       background_color=[0.0, 0.0, 0.0, 1.0], color_variation=0.0):
    """Non-atomic function: Make it Tile Patch (https://docs.substance3d.com/sddoc/make-it-tile-patch-159450499.html)

    Args:
        img_in (tensor): Input image.
        octave (int, optional): Logarithm of the tiling factor (by 2). Defaults to 3.
        seed (int, optional): Random seed. Defaults to 0.
        use_alpha (bool, optional): Enable the alpha channel. Defaults to False.
        mask_size (float, optional): Normalized size of the round mask used when stamping the patch. Defaults to 1.0.
        max_mask_size (float, optional): Maximum mask_size multiplier. Defaults to 1.0.
        mask_precision (float, optional): Falloff/smoothness precision of the mask. Defaults to 0.5.
        max_mask_precision (float, optional): Maximum mask_precision multiplier. Defaults to 1.0.
        mask_warping (float, optional): Normalized warping intensity at mask edges. Defaults to 0.5.
        max_mask_warping (float, optional): Maximum mask_warping multiplier. Defaults to 100.0.
        pattern_width (float, optional): Normalized width of the patch. Defaults to 0.2.
        max_pattern_width (float, optional): Maximum pattern_width multiplier. Defaults to 1000.0.
        pattern_height (float, optional): Normalized height of the patch. Defaults to 0.2.
        max_pattern_height (float, optional): Maximum pattern_height multiplier. Defaults to 1000.0.
        disorder (float, optional): Normalized translational randomness. Defaults to 0.0.
        max_disorder (float, optional): Maximum disorder multiplier. Defaults to 1.0.
        size_variation (float, optional): Normalized size variation for the mask. Defaults to 0.0.
        max_size_variation (float, optional): Maximum size_variation multiplier. Defaults to 100.0.
        rotation (float, optional): Rotation angle of the patch (in turning number). Defaults to 0.5.
        rotation_variation (float, optional): Randomness in rotation for every patch stamp. Defaults to 0.0.
        background_color (list, optional): Background color for areas where no patch appears. Defaults to [0.0, 0.0, 0.0, 1.0].
        color_variation (float, optional): Color (or luminosity) variation per patch. Defaults to 0.0.

    Returns:
        Tensor: Image with stamped patches of the input.
    """
    def param_process(param_in):
        param_in = to_tensor(param_in)
        if len(param_in.shape) == 0 or param_in.shape[0] == 1:
            param_in = param_in.view(1).expand(3)
        if param_in.shape[0] == 3 and use_alpha:
            param_in = th.cat([param_in, to_tensor([1.0])])
        return param_in

    # Process input parameters
    mask_size = to_tensor(mask_size) * max_mask_size
    mask_precision = to_tensor(mask_precision) * max_mask_precision
    mask_warping = to_tensor(mask_warping * 2.0 - 1.0) * max_mask_warping
    pattern_width = to_tensor(pattern_width) * max_pattern_width
    pattern_height = to_tensor(pattern_height) * max_pattern_height
    disorder = to_tensor(disorder) * max_disorder
    size_variation = to_tensor(size_variation) * max_size_variation
    rotation = to_tensor(rotation * 2.0 - 1.0)
    rotation_variation = to_tensor(rotation_variation)
    background_color = param_process(background_color)
    color_variation = to_tensor(color_variation)
    grid_size = 1 << octave

    # Mode switch
    mode_color = img_in.shape[1] > 1

    # Set random seed
    th.manual_seed(seed)

    # Gaussian pattern (sigma is approximated)
    x = th.linspace(-31 / 32, 31 / 32, 32).expand(32, 32)
    x = x ** 2 + x.transpose(1, 0) ** 2
    img_gs = th.exp(-0.5 * x / 0.089).expand(1, 1, 32, 32)
    img_gs = automatic_resize(img_gs, int(np.log2(img_in.shape[2] >> 5)))
    img_gs = levels(img_gs, [1.0 - mask_size], [0.5], [1 - mask_precision * mask_size])

    # Add alpha channel
    if mask_warping > 0.0:
        img_in_gc = c2g(img_in) if mode_color else img_in
        img_a = d_blur(img_in_gc, 1.6, 1.0)
        img_a = d_blur(img_a, 1.6, 1.0, angle=0.125)
        img_a = d_blur(img_a, 1.6, 1.0, angle=0.25)
        img_a = d_blur(img_a, 1.6, 1.0, angle=0.875)
        img_a = warp(img_gs, img_a, mask_warping * 0.05, 1.0)
    else:
        img_a = img_gs

    img_patch = img_in[:, :3, :, :] if mode_color else img_in.expand(img_in.shape[0], 3, img_in.shape[2], img_in.shape[3])
    img_patch = th.cat([img_patch, img_a], dim=1)

    # 'blend' operation with alpha processing
    def alpha_blend(img_fg, img_bg):
        fg_alpha = img_fg[:, [3], :, :]
        return th.cat([img_fg[:, :3, :, :] * fg_alpha, fg_alpha], dim=1) + img_bg * (1.0 - fg_alpha)

    # 'transform_2d' operation using only scaling and without tiling
    def scale_without_tiling(img_patch, height_scale, width_scale):
        if height_scale > 1.0 or width_scale > 1.0:
            print('Warning: the result might not be tiling correctly.')
        if width_scale != 1.0 or height_scale != 1.0:
            img_patch_rgb = transform_2d(img_patch[:, :3, :, :], mipmap_mode='manual', x1=to_zero_one(1.0 / width_scale), y2=to_zero_one(1.0 / height_scale))
            img_patch_a = transform_2d(img_patch[:, [3], :, :], tile_mode=0, mipmap_mode='manual', x1=to_zero_one(1.0 / width_scale), y2=to_zero_one(1.0 / height_scale))
        return th.cat([img_patch_rgb, img_patch_a], dim=1)

    # Pre-computation for transformed pattern (for non-random cases)
    if pattern_height == grid_size * 100 and pattern_width == grid_size * 100:
        img_patch_sc = img_patch
    else:
        img_patch_sc = scale_without_tiling(img_patch, pattern_height / (100 * grid_size), pattern_width / (100 * grid_size))
    if rotation == 0.0:
        img_patch_rot = img_patch_sc
    else:
        angle = rotation * np.pi * 2.0
        sin_angle, cos_angle = th.sin(angle), th.cos(angle)
        img_patch_rot = transform_2d(img_patch_sc, mipmap_mode='manual', x1=to_zero_one(cos_angle), x2=to_zero_one(-sin_angle),
                                     y1=to_zero_one(sin_angle), y2=to_zero_one(cos_angle))
    img_patch_double = alpha_blend(img_patch_rot, img_patch_rot)

    # Randomly transform the input patch (scaling, rotation, translation, color adjustment)
    def random_transform(img_patch, pos):
        size_delta = th.rand(1) * size_variation - th.rand(1) * size_variation
        h_size = th.clamp((pattern_height + size_delta) * 0.01, 0.0, 10.0)
        w_size = th.clamp((pattern_width + size_delta) * 0.01, 0.0, 10.0)
        rot_angle = (rotation + th.rand(1) * rotation_variation) * np.pi * 2.0
        off_angle = th.rand(1) * np.pi * 2.0
        pos_x = pos[0] + disorder * th.cos(off_angle)
        pos_y = pos[1] + disorder * th.sin(off_angle)
        col_scale = th.cat([1.0 - th.rand(3) * color_variation, th.ones(1)])

        # Scaling
        if size_variation == 0.0:
            img_patch = img_patch_sc
        else:
            img_patch = scale_without_tiling(img_patch, h_size / grid_size, w_size / grid_size)

        # Rotation and translation
        sin_angle, cos_angle = th.sin(rot_angle), th.cos(rot_angle)
        img_patch = transform_2d(img_patch, mipmap_mode='manual', x1=to_zero_one(cos_angle), x2=to_zero_one(-sin_angle), y1=to_zero_one(sin_angle), y2=to_zero_one(cos_angle))
        img_patch = transform_2d(img_patch, mipmap_mode='manual', x_offset=to_zero_one(pos_x), y_offset=to_zero_one(pos_y))
        return img_patch * col_scale.view(1, 4, 1, 1)

    # Create two layers of randomly transformed patterns (element for FX-Map)
    def gen_double_pattern(img_patch, pos):
        if size_variation == 0.0 and rotation_variation == 0.0 and disorder == 0.0 and \
           color_variation == 0.0:
            return transform_2d(img_patch_double, mipmap_mode='manual', x_offset=to_zero_one(pos[0]), y_offset=to_zero_one(pos[1]))
        else:
            return alpha_blend(random_transform(img_patch, pos), random_transform(img_patch, pos))

    # Calculate FX-Map
    fx_map = uniform_color(res_h=img_in.shape[2], res_w=img_in.shape[3], use_alpha=True,
                           rgba=background_color if mode_color else [color_variation] * 3 + [1.0])
    for i in range(grid_size):
        for j in range(grid_size):
            pos = [(i + 0.5) / grid_size - 0.5, (j + 0.5) / grid_size - 0.5]
            fx_map = alpha_blend(gen_double_pattern(img_patch, pos), fx_map)

    # Output channel conversion (if needed)
    img_out = fx_map if mode_color else c2g(fx_map)
    img_out = img_out[:, :3, :, :] if mode_color and not use_alpha else img_out

    return img_out

@input_check(1)
def make_it_tile_photo(img_in, mask_warping_x=0.5, max_mask_warping_x=100.0, mask_warping_y=0.5, max_mask_warping_y=100.0,
                       mask_size_x=0.1, max_mask_size_x=1.0, mask_size_y=0.1, max_mask_size_y=1.0,
                       mask_precision_x=0.5, max_mask_precision_x=1.0, mask_precision_y=0.5, max_mask_precision_y=1.0):
    """Non-atomic function: Make it Tile Photo (https://docs.substance3d.com/sddoc/make-it-tile-photo-159450503.html)

    Args:
        img_in (tensor): Input image.
        mask_warping_x (float, optional): Normalized warping intensity on the X-axis. Defaults to 0.5.
        max_mask_warping_x (float, optional): Maximum mask_warping_x multiplier. Defaults to 100.0.
        mask_warping_y (float, optional): Normalized warping intensity on the Y-axis. Defaults to 0.5.
        max_mask_warping_y (float, optional): Maximum mask_warping_y multiplier. Defaults to 100.0.
        mask_size_x (float, optional): Normalized width of the transition edge. Defaults to 0.1.
        max_mask_size_x (float, optional): Maximum mask_size_x multiplier. Defaults to 1.0.
        mask_size_y (float, optional): Normalized height of the transition edge. Defaults to 0.1.
        max_mask_size_y (float, optional): Maximum mask_size_y multiplier. Defaults to 1.0.
        mask_precision_x (float, optional): Normalized smoothness of the horizontal transition. Defaults to 0.5.
        max_mask_precision_x (float, optional): Maximum mask_precision_x multiplier. Defaults to 1.0.
        mask_precision_y (float, optional): Normalized smoothness of the vertical transition. Defaults to 0.5.
        max_mask_precision_y (float, optional): Maximum mask_precision_y multiplier. Defaults to 1.0.

    Returns:
        Tensor: Image with fixed tiling behavior.
    """
    # Process input parameters
    mask_warping_x = to_tensor(mask_warping_x * 2.0 - 1.0)
    mask_warping_y = to_tensor(mask_warping_y * 2.0 - 1.0)
    mask_size_x = to_tensor(mask_size_x) * max_mask_size_x
    mask_size_y = to_tensor(mask_size_y) * max_mask_size_y
    mask_precision_x = to_tensor(mask_precision_x) * max_mask_precision_x
    mask_precision_y = to_tensor(mask_precision_y) * max_mask_precision_y

    # Check input shape
    assert img_in.shape[2] == img_in.shape[3], "Input image is required to be in square shape"
    res = img_in.shape[2]

    # Split channels
    if img_in.shape[1] >= 3:
        img_rgb = img_in[:, :3, :, :]
        img_gs = c2g(img_rgb)
        if img_in.shape[1] == 4:
            img_a = img_in[:, [3], :, :]
    else:
        img_gs = img_in

    # Create pyramid pattern
    vec_grad = th.linspace((-res + 1) / res, (res - 1) / res, res)
    img_grad_x = th.abs(vec_grad.unsqueeze(0).expand(res, res))
    img_grad_y = th.abs(vec_grad.view(res, 1).expand(res, res))
    img_pyramid = th.clamp((1.0 - th.max(img_grad_x, img_grad_y)) * 7.39, 0.0, 1.0).view(1, 1, res, res)

    # Create cross mask
    img_grad = 1.0 - vec_grad ** 2
    img_grad_x = img_grad.unsqueeze(0).expand(1, 1, res, res)
    img_grad_y = img_grad.view(res, 1).expand(1, 1, res, res)
    img_grad_x = levels(img_grad_x, [1.0 - mask_size_x], [0.5], [1.0 - mask_size_x * mask_precision_x])
    img_grad_y = levels(img_grad_y, [1.0 - mask_size_y], [0.5], [1.0 - mask_size_y * mask_precision_y])

    img_gs = blur_hq(img_gs.view(1, 1, res, res), intensity=1.0, max_intensity=2.75)
    if mask_warping_x != 0:
        img_grad_x = d_warp(img_grad_x, img_gs, mask_warping_x, max_mask_warping_x)
        img_grad_x = d_warp(img_grad_x, img_gs, mask_warping_x, max_mask_warping_x, 0.5)
    if mask_warping_y != 0:
        img_grad_y = d_warp(img_grad_y, img_gs, mask_warping_y, max_mask_warping_y, 0.25)
        img_grad_y = d_warp(img_grad_y, img_gs, mask_warping_y, max_mask_warping_y, 0.75)
    img_cross = blend(img_grad_x, img_grad_y, None, 'max', opacity=1.0)
    img_cross = blend(img_pyramid, img_cross, blending_mode='multiply')

    # Create sphere mask
    img_grad = vec_grad ** 2 * 16
    img_grad_x = img_grad.unsqueeze(0).expand(res, res)
    img_grad_y = img_grad.view(res, 1).expand(res, res)
    img_grad = th.clamp(1.0 - (img_grad_x + img_grad_y), 0.0, 1.0).view(1, 1, res, res)
    img_sphere = blend(th.cat((img_grad[:, :, res >> 1:, :], img_grad[:, :, :res >> 1, :]), dim=2),
                       th.cat((img_grad[:, :, :, res >> 1:], img_grad[:, :, :, :res >> 1]), dim=3),
                       blending_mode='add')
    img_sphere = warp(img_sphere, img_gs, 0.24, 1.0)

    # Fix tiling for an image
    def fix_tiling(img_in):
        img = blend(img_in, transform_2d(img_in, sample_mode='nearest', mipmap_mode='manual', x_offset=to_zero_one(0.5), y_offset=to_zero_one(0.5)), img_cross)
        img = blend(transform_2d(img, sample_mode='nearest', mipmap_mode='manual', x_offset=to_zero_one(0.25), y_offset=to_zero_one(0.25)),
                    transform_2d(img, sample_mode='nearest', mipmap_mode='manual', x_offset=to_zero_one(0.5), y_offset=to_zero_one(0.5)), img_sphere)
        return img

    # Process separate channels
    if img_in.shape[1] == 3:
        return fix_tiling(img_rgb)
    elif img_in.shape[1] == 4:
        return th.cat((fix_tiling(img_rgb), fix_tiling(img_a)), dim=1)
    else:
        return fix_tiling(img_in)

@input_check(1)
def replace_color(img_in, source_color, target_color):
    """Non-atomic function: Replace Color (https://docs.substance3d.com/sddoc/replace-color-159449260.html)

    Args:
        img_in (tensor): Input image.
        source_color (list): Color to start hue shifting from
        target_color (list): Color where hue shifting ends

    Returns:
        Tensor: Replaced color image.
    """
    color_input_check(img_in, 'input image')
    target_hsl = rgb2hsl(target_color)
    source_hsl = rgb2hsl(source_color)
    diff_hsl = (target_hsl - source_hsl) * 0.5 + 0.5
    img_out = hsl(img_in, diff_hsl[0], diff_hsl[1], diff_hsl[2])
    return img_out

@input_check(0)
def normal_color(normal_format='dx', num_imgs=1, res_h=512, res_w=512, use_alpha=False, direction=0.0, slope_angle=0.0):
    """a non-atomic function of uniform_color

    Args:
        normal_format (str, optional): Normal format ('dx'|'gl'). Defaults to 'dx'.
        num_imgs (int, optional): Batch size. Defaults to 1.
        res_h (int, optional): Resolution in height. Defaults to 512.
        res_w (int, optional): Resolution in width. Defaults to 512.
        use_alpha (bool, optional): Enable the alpha channel. Defaults to False.
        direction (float, optional): Normal direction (in turning number). Defaults to 0.0.
        slope_angle (float, optional): Normal slope angle (in turning number). Defaults to 0.0.

    Returns:
        Tensor: Uniform normal color image.
    """
    assert normal_format in ('dx', 'gl')
    direction = to_tensor(direction) * (np.pi * 2)
    slope_angle = to_tensor(slope_angle) * (np.pi * 2)
    vec = th.stack([-th.cos(direction), th.sin(direction) * (1.0 if normal_format == 'gl' else -1.0)])
    vec = vec * th.sin(slope_angle) * 0.5 + 0.5
    rgba = th.cat([vec, th.ones(2)])
    img_out = uniform_color('color', num_imgs, res_h, res_w, use_alpha, rgba)
    return img_out

@input_check(1)
def vector_morph(img_in, vector_field=None, amount=1.0, max_amount=1.0):
    """Non-atomic function: Vector Morph (https://docs.substance3d.com/sddoc/vector-morph-166363405.html)

    Args:
        img_in (tensor): Input image.
        vector_field (tensor, optional): Vector map that drives warping. Defaults to None.
        amount (float, optional): Normalized warping intensity as a multiplier for the vector map. Defaults to 1.0.
        max_amount (float, optional): Maximum warping intensity multiplier. Defaults to 1.0.

    Returns:
        Tensor: Warped image.
    """
    # Check input
    res_h, res_w = img_in.shape[2], img_in.shape[3]
    if vector_field is None:
        vector_field = img_in.expand(img_in.shape[0], 2, res_h, res_w) if img_in.shape[1] == 1 else img_in[:, :2, :, :]
    else:
        color_input_check(vector_field, 'vector field')
        vector_field = vector_field[:, :2, :, :]

    # Process parameter
    amount = to_tensor(amount)
    if amount == 0.0:
        return img_in

    # Progressive vector field sampling
    row_grid, col_grid = th.meshgrid(th.linspace(1, res_h * 2 - 1, res_h) / (res_h * 2), \
                                     th.linspace(1, res_w * 2 - 1, res_w) / (res_w * 2))
    sample_grid = th.stack((col_grid, row_grid), dim=2).expand(img_in.shape[0], res_h, res_w, 2)

    gs_interp_mode = 'bilinear'
    gs_padding_mode = 'zeros'
    vector_field_pad = th.nn.functional.pad(vector_field, [1, 1, 1, 1], 'circular')
    for i in range(16):
        if i == 0:
            vec = vector_field
        else:
            sample_grid_sp = (sample_grid * 2.0 - 1.0) * to_tensor([res_w / (res_w + 2), res_h / (res_h + 2)])
            vec = th.nn.functional.grid_sample(vector_field_pad, sample_grid_sp, gs_interp_mode, gs_padding_mode, align_corners=False)
        sample_grid = th.remainder(sample_grid + (vec.permute(0, 2, 3, 1) - 0.5) * amount * 0.0625, 1.0)

    # Final image sampling
    sample_grid = (sample_grid * 2.0 - 1.0) * to_tensor([res_w / (res_w + 2), res_h / (res_h + 2)])
    img_in_pad = th.nn.functional.pad(img_in, [1, 1, 1, 1], 'circular')
    img_out = th.nn.functional.grid_sample(img_in_pad, sample_grid, gs_interp_mode, gs_padding_mode, align_corners=False)

    return img_out

@input_check(1)
def vector_warp(img_in, vector_map=None, vector_format='dx', intensity=1.0, max_intensity=1.0):
    """Non-atomic function: Vector Warp (https://docs.substance3d.com/sddoc/vector-warp-159450546.html)

    Args:
        img_in (tensor): Input image.
        vector_map (tensor, optional): Distortion driver map. Defaults to None.
        vector_format (str, optional): Normal format of the vector map ('dx'|'gl'). Defaults to 'dx'.
        intensity (float, optional): Normalized intensity multiplier of the vector map. Defaults to 1.0.
        max_intensity (float, optional): Maximum intensity multiplier of the vector map. Defaults to 1.0.

    Returns:
        Tensor: Distorted image.
    """
    # Check input
    assert vector_format in ('dx', 'gl')
    res_h, res_w = img_in.shape[2], img_in.shape[3]
    if vector_map is None:
        vector_map = th.zeros(img_in.shape[0], 2, res_h, res_w)
    else:
        color_input_check(vector_map, 'vector map')
        vector_map = vector_map[:, :2, :, :]

    # Process input parameters
    intensity = to_tensor(intensity)
    if intensity == 0.0:
        return img_in

    # Calculate displacement field
    vector_map = vector_map * 2.0 - 1.0
    if vector_format == 'gl':
        vector_map[:, [1], :, :] = -vector_map[:, [1], :, :]
    vector_map = vector_map * th.sqrt(th.sum(vector_map ** 2, 1, keepdim=True)) * intensity

    # Sample input image
    row_grid, col_grid = th.meshgrid(th.linspace(1, res_h * 2 - 1, res_h) / (res_h * 2), \
                                     th.linspace(1, res_w * 2 - 1, res_w) / (res_w * 2))
    sample_grid = th.stack((col_grid, row_grid), dim=2).expand(img_in.shape[0], res_h, res_w, 2)
    sample_grid = th.remainder(sample_grid + vector_map.permute(0, 2, 3, 1), 1.0)
    sample_grid = (sample_grid * 2.0 - 1.0) * to_tensor([res_w / (res_w + 2), res_h / (res_h + 2)])

    gs_interp_mode = 'bilinear'
    gs_padding_mode = 'zeros'
    img_in_pad = th.nn.functional.pad(img_in, [1, 1, 1, 1], 'circular')
    img_out = th.nn.functional.grid_sample(img_in_pad, sample_grid, gs_interp_mode, gs_padding_mode, align_corners=False)

    return img_out

@input_check(1)
def contrast_luminosity(img_in, contrast=0.5, luminosity=0.5):
    """Non-atomic function: Contrast/Luminosity (https://docs.substance3d.com/sddoc/contrast-luminosity-159449189.html)

    Args:
        img_in (tensor): Input image
        contrast (float, optional): Contrast of the result. Defaults to 0.5.
        luminosity (float, optional): Brightness of the result. Defaults to 0.5.

    Returns:
        Tensor: Adjusted image.
    """
    # Process input parameters
    contrast = to_tensor(contrast) * 2.0 - 1.0
    luminosity = to_tensor(luminosity) * 2.0 - 1.0

    in_low = th.clamp(contrast * 0.5, 0.0, 0.5)
    in_high = th.clamp(1.0 - contrast * 0.5, 0.5, 1.0)
    temp = th.abs(th.min(contrast, to_tensor(0.0))) * 0.5
    out_low = th.clamp(temp + luminosity, 0.0, 1.0)
    out_high = th.clamp(luminosity + 1.0 - temp, 0.0, 1.0)
    img_out = levels(img_in, in_low, 0.5, in_high, out_low, out_high)

    return img_out

@input_check(1)
def p2s(img_in):
    """Non-atomic function: Pre-Multiplied to Straight (https://docs.substance3d.com/sddoc/pre-multiplied-to-straight-159450478.html)

    Args:
        img_in (tensor): Image with pre-multiplied color.

    Returns:
        Tensor: Image with straight color.
    """
    color_input_check(img_in, 'input image')
    assert img_in.shape[1] == 4, 'input image must contain alpha channel'

    rgb = img_in[:,:3,:,:]
    a = img_in[:,[3],:,:]
    img_out = th.cat([(rgb / (a + 1e-15)).clamp(0.0, 1.0), a], 1)
    return img_out

@input_check(1)
def s2p(img_in):
    """Non-atomic function: Straight to Pre-Multiplied (https://docs.substance3d.com/sddoc/straight-to-pre-multiplied-159450483.html)

    Args:
        img_in (tensor): Image with straight color.

    Returns:
        Tensor: Image with pre-multiplied color.
    """
    color_input_check(img_in, 'input image')
    assert img_in.shape[1] == 4, 'input image must contain alpha channel'

    rgb = img_in[:,:3,:,:]
    a = img_in[:,[3],:,:]
    img_out = th.cat([rgb * a, a], 1)
    return img_out

@input_check(1)
def clamp(img_in, clamp_alpha=True, low=0.0, high=1.0):
    """Non-atomic function: Clamp (https://docs.substance3d.com/sddoc/clamp-159449164.html)

    Args:
        img_in (tensor): Input image
        clamp_alpha (bool, optional): Clamp the alpha channel. Defaults to True.
        low (float, optional): Lower clamp limit. Defaults to 0.0.
        high (float, optional): Upper clamp limit. Defaults to 1.0.

    Returns:
        Tensor: Clamped image.
    """
    low = to_tensor(low)
    high = to_tensor(high)
    if img_in.shape[1] == 4 and not clamp_alpha:
        img_out = th.cat([img_in[:,:3,:,:].clamp(low, high), img_in[:,[3],:,:]], 1)
    else:
        img_out = img_in.clamp(low, high)
    return img_out

@input_check(1)
def pow(img_in, exponent=0.4, max_exponent=10.0):
    """Non-atomic function: Pow (https://docs.substance3d.com/sddoc/pow-159449251.html)

    Args:
        img_in (tensor): Input image
        exponent (float, optional): Normalized exponent of the power function. Defaults to 0.4.
        max_exponent (float, optional): Maximum exponent multiplier. Defaults to 10.0.

    Returns:
        Tensor: Powered image.
    """
    exponent = to_tensor(exponent) * max_exponent
    use_alpha = False
    if img_in.shape[1] == 4:
        use_alpha = True
        img_in_alpha = img_in[:,[3],:,:]
        img_in = img_in[:,:3,:,:]

    # Levels
    in_mid = (exponent - 1.0) / 16.0 + 0.5 if exponent >= 1.0 else \
             0.5625 if exponent == 0 else (1.0 / exponent - 9.0) / -16.0
    img_out = levels(img_in, in_mid=in_mid)

    if use_alpha:
        img_out = th.cat([img_out, img_in_alpha], 1)
    return img_out

@input_check(1)
def quantize(img_in, quantize_number=3):
    """Non-atomic function: Quantize (https://docs.substance3d.com/sddoc/quantize-159449255.html)

    Args:
        img_in (tensor): Input image.
        quantize_number (int or list, optional): Number of quantization steps (a list input controls each channel separately). Defaults to 3.

    Returns:
        Tensor: Quantized image.
    """
    qn = (to_tensor(quantize_number) - 1) / 255.0
    qt_shift = 1.0 - 286.0 / 512.0
    img_in = levels(img_in, out_high=qn)
    img_qt = th.floor(img_in * 255.0 + qt_shift) / 255.0
    img_out = levels(img_qt, in_high=qn)
    return img_out

@input_check(1)
def anisotropic_blur(img_in, high_quality=False, intensity=10.0/16.0, max_intensity=16.0, anisotropy=0.5, angle=0.0):
    """Non-atomic function: Anisotropic Blur (https://docs.substance3d.com/sddoc/anisotropic-blur-159450450.html)

    Args:
        img_in (tensor): Input image.
        high_quality (bool, optional): Switch between a box blur (False) and an HQ blur (True) internally. Defaults to False.
        intensity (float, optional): Normalized directional blur intensity. Defaults to 10.0/16.0.
        max_intensity (float, optional): Maximum directional blur intensity multiplier. Defaults to 16.0.
        anisotropy (float, optional): Directionality of the blur. Defaults to 0.5.
        angle (float, optional): Angle of the blur direction (in turning number). Defaults to 0.0.

    Returns:
        Tensor: Anisotropically blurred image.
    """
    intensity = to_tensor(intensity) * max_intensity
    anisotropy = to_tensor(anisotropy)
    angle = to_tensor(angle)
    quality_factor = 0.6 if high_quality else 1.0

    # Two-pass directional blur
    img_out = d_blur(img_in, intensity * quality_factor, 1.0, angle)
    img_out = d_blur(img_out, intensity * (1.0 - anisotropy) * quality_factor, 1.0, angle + 0.25)
    if high_quality:
        img_out = d_blur(img_out, intensity * quality_factor, 1.0, angle)
        img_out = d_blur(img_out, intensity * (1.0 - anisotropy) * quality_factor, 1.0, angle + 0.25)

    return img_out

@input_check(1)
def glow(img_in, glow_amount=0.5, clear_amount=0.5, size=0.5, max_size=20.0, color=[1.0, 1.0, 1.0, 1.0]):
    """Non-atomic function: Glow (https://docs.substance3d.com/sddoc/glow-159450531.html)

    Args:
        img_in (tensor): Input image.
        glow_amount (float, optional): Global opacity of the glow effect. Defaults to 0.5.
        clear_amount (float, optional): Cut-off threshold of the glow effect. Defaults to 0.5.
        size (float, optional): Normalized scale of the glow effect. Defaults to 0.5.
        max_size (float, optional): Maximum scale multiplier. Defaults to 20.0.
        color (list, optional): Color of the glow effect. Defaults to [1.0, 1.0, 1.0, 1.0].

    Returns:
        Tensor: Image with the glow effect.
    """
    glow_amount = to_tensor(glow_amount)
    clear_amount = to_tensor(clear_amount)
    size = to_tensor(size) * max_size
    color = to_tensor(color)

    # Calculate glow mask
    num_channels = img_in.shape[1]
    img_mask = img_in[:,:3,:,:].sum(dim=1, keepdim=True) / 3 if num_channels > 1 else img_in
    img_mask = levels(img_mask, in_low=clear_amount - 0.01, in_high=clear_amount + 0.01)
    img_mask = blur_hq(img_mask, intensity=size, max_intensity=1.0)

    # Blending in glow effect
    if num_channels > 1:
        img_out = blend(color[:num_channels].view(1,num_channels,1,1).expand_as(img_in), img_in, img_mask * glow_amount, 'add')
    else:
        img_out = blend(img_mask, img_in, None, 'add', opacity=glow_amount)
    return img_out

@input_check(1)
def car2pol(img_in):
    """Non-atomic function: Cartesian to Polar (https://docs.substance3d.com/sddoc/cartesian-to-polar-159450598.html)

    Args:
        img_in (tensor): Input image in Cartesian coordinates.

    Returns:
        Tensor: Image in polar coordinates.
    """
    res_h = img_in.shape[2]
    res_w = img_in.shape[3]
    row_grid, col_grid = th.meshgrid(th.linspace(0.5, res_h - 0.5, res_h) / res_h - 0.5,
                                     th.linspace(0.5, res_w - 0.5, res_w) / res_w - 0.5)
    rad_grid = th.remainder(th.sqrt(row_grid ** 2 + col_grid ** 2) * 2.0, 1.0) * 2.0 - 1.0
    ang_grid = th.remainder(-th.atan2(row_grid, col_grid) / (np.pi * 2), 1.0) * 2.0 - 1.0
    rad_grid = rad_grid * res_h / (res_h + 2)
    ang_grid = ang_grid * res_w / (res_w + 2)
    sample_grid = th.stack([ang_grid, rad_grid], 2).expand(img_in.shape[0], res_h, res_w, 2)
    in_pad = th.nn.functional.pad(img_in, [1, 1, 1, 1], 'circular')
    img_out = th.nn.functional.grid_sample(in_pad, sample_grid, 'bilinear', 'zeros', align_corners=False)
    return img_out

@input_check(1)
def pol2car(img_in):
    """Non-atomic function: Polar to Cartesian (https://docs.substance3d.com/sddoc/polar-to-cartesian-159450602.html)

    Args:
        img_in (tensor): Image in polar coordinates.

    Returns:
        Tensor: Image in Cartesian coordinates.
    """
    res_h = img_in.shape[2]
    res_w = img_in.shape[3]
    row_grid, col_grid = th.meshgrid(th.linspace(0.5, res_h - 0.5, res_h) / res_h,
                                     th.linspace(0.5, res_w - 0.5, res_w) / res_w)
    ang_grid = -col_grid * (np.pi * 2.0)
    rad_grid = row_grid * 0.5
    x_grid = th.remainder(rad_grid * th.cos(ang_grid) + 0.5, 1.0) * 2.0 - 1.0
    y_grid = th.remainder(rad_grid * th.sin(ang_grid) + 0.5, 1.0) * 2.0 - 1.0
    x_grid = x_grid * res_w / (res_w + 2)
    y_grid = y_grid * res_h / (res_h + 2)
    sample_grid = th.stack([x_grid, y_grid], 2).expand(img_in.shape[0], res_h, res_w, 2)
    in_pad = th.nn.functional.pad(img_in, [1, 1, 1, 1], 'circular')
    img_out = th.nn.functional.grid_sample(in_pad, sample_grid, 'bilinear', 'zeros', align_corners=False)
    return img_out

@input_check(1)
def normal_sobel(img_in, normal_format='dx', use_alpha=False, intensity=2.0/3.0, max_intensity=3.0):
    """Non-atomic node: Normal Sobel (https://docs.substance3d.com/sddoc/normal-sobel-159450588.html)

    Args:
        img_in (tensor): Input height map.
        normal_format (str, optional): Input normal format. Defaults to 'dx'.
        use_alpha (bool, optional): Output alpha channel. Defaults to False.
        intensity (float, optional): Normal intensity (normalized). Defaults to 2.0/3.0.
        max_intensity (float, optional): Max normal intensity multiplier. Defaults to 3.0.

    Returns:
        Tensor: Output normal map.
    """
    grayscale_input_check(img_in, 'input height map')
    assert normal_format in ('dx', 'gl')
    intensity = to_tensor((intensity * 2.0 - 1.0) * max_intensity).squeeze()

    # Pre-compute scale multipliers
    res_h, res_w = img_in.shape[2], img_in.shape[3]
    mult_x = intensity * res_w / 512.0
    mult_y = intensity * res_h / 512.0 * (-1 if normal_format == 'dx' else 1)

    # Pre-blur the input image
    img_blur_x = d_blur(img_in, intensity=256 / res_w, max_intensity=1.0)
    img_blur_y = d_blur(img_in, intensity=256 / res_h, max_intensity=1.0, angle=0.25)

    # Compute normal
    normal_x = (roll_col(img_blur_y, -1) - roll_col(img_blur_y, 1)) * mult_x
    normal_y = (roll_row(img_blur_x, 1) - roll_row(img_blur_x, -1)) * mult_y
    normal = th.cat((normal_x, normal_y, th.ones_like(normal_x)), dim=1)
    img_normal = (normal * 0.5 / normal.norm(dim=1, keepdim=True)) + 0.5

    # Add output alpha channel
    if use_alpha:
        img_normal = th.cat((img_normal, th.ones_like(normal_x)), dim=1)
    return img_normal

@input_check(1)
def normal_vector_rotation(img_in, img_map=None, normal_format='dx', rotation=0.5, rotation_max=1.0):
    """Non-atomic function: Normal Vector Rotation (https://docs.substance3d.com/sddoc/normal-vector-rotation-172819817.html)

    Args:
        img_in (tensor): Input normal map.
        img_map (tensor, optional): Rotation map. Defaults to 'None'.
        normal_format (str, optional): Input normal format ('dx' or 'gl'). Defaults to 'dx'.
        rotation (float, optional): Normal vector rotation angle (normalized). Defaults to 0.5.
        rotation_max (float, optional): Max rotation angle multiplier. Defaults to 1.0.

    Returns:
        Tensor: Output normal map.
    """
    color_input_check(img_in, 'input image')
    if img_map is not None:
        grayscale_input_check(img_map, 'rotation map')
    assert normal_format in ('dx', 'gl')
    rotation = to_tensor((rotation * 2.0 - 1.0) * rotation_max).squeeze()
    if img_map is None:
        img_map = th.zeros(1, 1, img_in.shape[2], img_in.shape[3])

    # Rotate normal vector map
    nx, ny, nzw = img_in[:,[0],:,:], img_in[:,[1],:,:], img_in[:,2:,:,:]
    nx = nx * 2 - 1
    ny = 1 - ny * 2 if normal_format == 'dx' else ny * 2 - 1
    angle_rad_map = (img_map + rotation) * 2.0 * np.pi
    cos_angle, sin_angle = th.cos(angle_rad_map), th.sin(angle_rad_map)
    nx_rot = nx * cos_angle + ny * sin_angle
    ny_rot = ny * cos_angle - nx * sin_angle
    nx_rot = nx_rot * 0.5 + 0.5
    ny_rot = 0.5 - ny_rot * 0.5 if normal_format == 'dx' else ny_rot * 0.5 + 0.5

    img_out = th.cat((nx_rot, ny_rot, nzw), dim=1)
    return img_out

@input_check(1)
def non_square_transform(img_in, tiling=3, tile_mode='automatic', x_tile=1, y_tile=1, tile_safe_rotation=True, x_offset=0.5, x_offset_max=1.0, y_offset=0.5, y_offset_max=1.0,
                         rotation=0.5, rotation_max=1.0, background_color=[0.0, 0.0, 0.0, 1.0]):
    """Non-atomic function: Non-Square Transform (https://docs.substance3d.com/sddoc/non-square-transform-159450647.html)

    Args:
        img_in (tensor): Input image.
        tiling (int, optional): Output tiling (see function 'transform_2d'). Defaults to 3.
        tile_mode (str, optional): Tiling mode control ('automatic' or 'manual'). Defaults to 'automatic'.
        x_tile (int, optional): Tiling in X direction (if tile_mode is 'manual'). Defaults to 1.
        y_tile (int, optional): Tiling in Y direction (if tile_mode is 'manual'). Defaults to 1.
        tile_safe_rotation (bool, optional): Snaps to safe values to maintain sharpness of pixels. Defaults to True.
        x_offset (float, optional): X translation offset. Defaults to 0.5.
        x_offset_max (float, optional): Maximum X offset multiplier. Defaults to 1.0.
        y_offset (float, optional): Y translation offset. Defaults to 0.5.
        y_offset_max (float, optional): Maximum Y offset multiplier. Defaults to 1.0.
        rotation (float, optional): Image rotation angle. Defaults to 0.5.
        rotation_max (float, optional): Maximum rotation angle multiplier. Defaults to 1.0.
        background_color (list, optional): Background color when tiling is disabled. Defaults to [0.0, 0.0, 0.0, 1.0].

    Returns:
        Tensor: Transformed image.
    """
    assert tile_mode in ('automatic', 'manual')
    res_h, res_w = img_in.shape[2], img_in.shape[3]

    # Process parameters
    x_offset = to_tensor((x_offset * 2.0 - 1.0) * x_offset_max).squeeze()
    y_offset = to_tensor((y_offset * 2.0 - 1.0) * y_offset_max).squeeze()
    rotation = to_tensor((rotation * 2.0 - 1.0) * rotation_max).squeeze()
    background_color = to_tensor(background_color).view(-1)
    background_color = resize_color(background_color, img_in.shape[1])

    # Compute rotation angle
    angle_trunc = th.floor(rotation * 4) * 0.25
    angle = angle_trunc if tile_safe_rotation else rotation
    angle_rad = angle * 2.0 * np.pi

    # Compute scale
    angle_remainder_rad = th.remainder(angle_trunc.abs(), 0.25) * 2.0 * np.pi
    rotation_scale = th.cos(angle_remainder_rad) + th.sin(angle_remainder_rad) if tile_safe_rotation else 1.0
    x_scale = x_tile * min(res_w / res_h, 1.0) * rotation_scale if tile_mode == 'manual' else max(res_w / res_h, 1.0)
    y_scale = y_tile * min(res_h / res_w, 1.0) * rotation_scale if tile_mode == 'manual' else max(res_h / res_w, 1.0)

    # Compute transform matrix
    x1, x2 = x_scale * th.cos(angle_rad), x_scale * -th.sin(angle_rad)
    y1, y2 = y_scale * th.sin(angle_rad), y_scale * th.cos(angle_rad)

    # Compute offset
    x_offset = th.floor((x_offset - 0.5) * res_w) / res_w + 0.5
    y_offset = th.floor((y_offset - 0.5) * res_h) / res_h + 0.5

    # Call transformation 2D operator
    img_out = transform_2d(img_in, tile_mode=tiling, mipmap_mode='manual', x1=to_zero_one(x1), x2=to_zero_one(x2), y1=to_zero_one(y1), y2=to_zero_one(y2),
                           x_offset=to_zero_one(x_offset), y_offset=to_zero_one(y_offset), matte_color=background_color)
    return img_out

@input_check(1)
def quad_transform(img_in, culling='f-b', enable_tiling=False, sampling='bilinear', p00=[0.0, 0.0], p01=[0.0, 1.0], p10=[1.0, 0.0], p11=[1.0, 1.0],
                   background_color=[0.0, 0.0, 0.0, 1.0]):
    """Non-atomic function: Quad Transform (https://docs.substance3d.com/sddoc/quad-transform-172819829.html)

    Args:
        img_in (tensor): Input image.
        culling (str, optional): Set culling/hiding of shape when points cross over each other. Defaults to 'f-b'.
            [Options]
                - 'f': front only.
                - 'b': back only.
                - 'f-b': front over back.
                - 'b-f': back over front.
        enable_tiling (bool, optional): Enable tiling. Defaults to False.
        sampling (str, optional): Set sampling quality ('bilinear' or 'nearest'). Defaults to 'bilinear'.
        p00 (list, optional): Top left point. Defaults to [0.0, 0.0].
        p01 (list, optional): Bottom left point. Defaults to [0.0, 1.0].
        p10 (list, optional): Top right point. Defaults to [1.0, 0.0].
        p11 (list, optional): Bottom right point. Defaults to [1.0, 1.0].
        background_color (list, optional): Solid background color if tiling is off. Defaults to [0.0, 0.0, 0.0, 1.0].

    Returns:
        Tensor: Transformed image.
    """
    assert culling in ('f', 'b', 'f-b', 'b-f')
    assert sampling in ('bilinear', 'nearest')
    p00, p01, p10, p11 = to_tensor(p00), to_tensor(p01), to_tensor(p10), to_tensor(p11)
    background_color = to_tensor(background_color).view(-1)
    background_color = resize_color(background_color, img_in.shape[1])

    # Compute a few derived (or renamed) values
    b, c, a = p01 - p00, p10 - p00, p11 - p01
    d = a - c
    x2_1, x2_2 = cross_2d(c, d), cross_2d(b, d)
    x1_a, x1_b = cross_2d(b, c), d
    p0, x0_1, x0_2 = p00, b, c
    enable_2sided = len(culling) > 1 and not enable_tiling
    front_first = culling.startswith('f')

    # Solve quadratic equations
    res_h, res_w = img_in.shape[2], img_in.shape[3]
    pos_offset = p0 - get_pos(res_h, res_w)
    x1_b = cross_2d(pos_offset, x1_b)
    x0_1 = cross_2d(pos_offset, x0_1)
    x0_2 = cross_2d(pos_offset, x0_2)
    qx1, qy1, error1 = solve_poly_2d(x2_1, x1_b - x1_a, x0_1)
    qx2, qy2, error2 = solve_poly_2d(x2_2, x1_b + x1_a, x0_2)

    # Compute sampling positions
    sample_pos_ff = th.stack((qx1, qy2), dim=2)
    sample_pos_bf = th.stack((qy1, qx2), dim=2)
    in_01_ff = th.all((sample_pos_ff >= 0) & (sample_pos_ff <= 1), dim=2)
    in_01_bf = th.all((sample_pos_bf >= 0) & (sample_pos_bf <= 1), dim=2)

    # Determine which face is being considered
    cond_face = th.as_tensor((in_01_ff if front_first else ~in_01_bf) if enable_2sided else front_first)
    in_01 = th.where(cond_face, in_01_ff, in_01_bf)
    sample_pos = th.where(cond_face.expand(res_h, res_w, 1), sample_pos_ff, sample_pos_bf)

    # Perform sampling
    sample_pos = (th.remainder(sample_pos, 1.0) * 2.0 - 1.0) * to_tensor([res_h / (res_h + 2), res_w / (res_w + 2)])
    sample_pos = sample_pos.expand(img_in.shape[0], res_h, res_w, 2)
    in_pad = th.nn.functional.pad(img_in, [1, 1, 1, 1], 'circular')
    img_sample = th.nn.functional.grid_sample(in_pad, sample_pos, sampling, 'zeros', align_corners=False)

    # Choose between background color or sampling result
    img_bg = background_color[:, None, None]
    cond = error1 | error2 | ~(in_01 | enable_tiling)
    img_out = th.where(cond, img_bg, img_sample)
    return img_out

# Finalize
@input_check(1)
def chrominance_extract(img_in):
    """Non-atomic function: Chrominance Extract (https://docs.substance3d.com/sddoc/chrominance-extract-159449160.html)

    Args:
        img_in (tensor): Input image (RGB(A) only).

    Returns:
        Tensor: Chrominance of the image.
    """
    color_input_check(img_in, 'input image')
    
    blend_fg = 1 - c2g(img_in, False, [0.3,0.59,0.11,0.0]).repeat(1,3,1,1)
    if img_in.shape[1] == 4:
        blend_fg = append_alpha(blend_fg)
    img_out = blend(blend_fg, img_in, blending_mode="add_sub", opacity=0.5)
    return img_out 
    
# Finalized
@input_check(1)
def histogram_shift(img_in, position=0.5):
    """Non-atomic function: Histogram Shift (https://docs.substance3d.com/sddoc/chrominance-extract-159449160.html)

    Args:
        img_in (tensor): Input image (G only)
        position (float, optional): How much to shift the input by (normalized). Defaults to 0.5.

    Returns:
        Tensor: Histogram shifted image.
    """    

    grayscale_input_check(img_in, 'input image')
    position = to_tensor(position)

    levels_1 = levels(img_in, in_low=position, in_mid=0.5, in_high=1.0, out_low=0.0, out_high=1.0-position)
    levels_2 = levels(img_in, in_low=0, in_mid=0.5, in_high=position, out_low=1.0-position, out_high=1.0)
    levels_3 = levels(img_in, in_low=position, in_mid=0.5, in_high=position, out_low=0.0, out_high=1.0)
    img_out = blend(levels_1, levels_2, levels_3, blending_mode='copy', opacity=1.0)
    return img_out

# Finalized
@input_check(1)
def height_map_frequencies_mapper(img_in, relief=0.5, max_relief=32.0):
    """Non-atomic function: Height Map Frequencies Mapper
 (https://docs.substance3d.com/sddoc/height-map-frequencies-mapper-159449198.html)

    Args:
        img_in (tensor): Input image (G only).
        relief (float, optional): Controls the displacement output's detail size (normalized). Defaults to 0.5.
        max_relief (float, optional): Maximum displacement. Defaults to 32.0.

    Returns:
        Tensor: Height map frequency mapped image.
    """
    grayscale_input_check(img_in, 'input image')
    relief = to_tensor(relief * max_relief)

    blend_fg = uniform_color('gray', num_imgs = img_in.shape[0], res_h=img_in.shape[2], res_w=img_in.shape[3], use_alpha=False, rgba=[0.498])
    blend_bg = blur_hq(img_in, False, 1.0, relief)
    displacement = blend(blend_fg, blend_bg, blending_mode='copy', opacity=relief / 32.0)
    levels_out = levels(displacement, out_low=1.0, out_high=0.0)
    relief_parallax = blend(levels_out, img_in, blending_mode='add_sub', opacity = 0.5)
    return displacement, relief_parallax

# Finalized
@input_check(1)
def luminance_highpass(img_in, radius=6.0/64.0, max_radius=64.0):
    """Non-atomic function: Luminance Highpass (https://docs.substance3d.com/sddoc/luminance-highpass-159449246.html)

    Args:
        img_in (tensor): Input image (RGB(A) only).
        radius (float, optional): Normalized radius of the highpass effect. Defaults to 6.0
        max_radius (float, optional): Maximum radius. Defaults to 64.0.

    Returns:
        Tensor: Luminance highpassed image.
    """
    color_input_check(img_in, 'input image')
    radius = to_tensor(radius * max_radius)

    if img_in.shape[1] == 4:
        use_alpha=True
        alpha = img_in[:,3:4,:,:]
        img_in = img_in[:,:3,:,:]
    else:
        use_alpha=False

    grayscale = c2g(img_in, False, [0.3,0.59,0.11,0.0])
    highpassed = highpass(grayscale, 1.0, radius)
    transformed = transform_2d(grayscale, mipmap_level=12, mipmap_mode='manual')
    blended_fg = blend(highpassed, transformed, blending_mode='add_sub', opacity=0.5)
    # add alpha, expand with alpha
    blended_fg_color = append_alpha(blended_fg.repeat(1,3,1,1)) if img_in.shape[1]==4 else blended_fg.repeat(1,3,1,1)
    grayscale_to_color = append_alpha(1 - grayscale.repeat(1,3,1,1)) if img_in.shape[1]==4 else 1 - grayscale.repeat(1,3,1,1)
    blended_bg = blend(grayscale_to_color, img_in, blending_mode='add_sub', opacity=0.5)
    img_out = blend(blended_fg_color, blended_bg, blending_mode='add_sub', opacity=0.5)
    if use_alpha:
        img_out = th.cat((img_out, alpha), axis=1)
    return img_out

# Finalized
@input_check(1)
def replace_color_range(img_in, source_color=[0.501961]*3, target_color=[0.501961]*3, source_range=0.5, threshold=1.0):
    """Non-atomic function: Replace Color Range (https://docs.substance3d.com/sddoc/replace-color-range-159449321.html)

    Args:
        img_in (tensor): Input image (RGB(A) only).
        source_color (list, optional): Color to replace. Defaults to [0.501961]*3.
        target_color (list, optional): Color to replace with. Defaults to [0.501961]*3.
        source_range (float, optional): Range or tolerance of the picked Source. Can be increased so further neighbouring colours are also hue-shifted. Defaults to 0.5.
        threshold (float, optional): Falloff/contrast for range. Set low to replace only Source color, set higher to replace colors blending into Source as well. Defaults to 1.0.

    Returns:
        Tensor: Color replaced image.
    """    
    color_input_check(img_in, 'input image')

    img_in = th.clamp(img_in, 0.0, 1.0)

    if img_in.shape[1] == 4:
        use_alpha=True
        alpha = img_in[:,3:4,:,:]
        img_in = img_in[:,:3,:,:]
    else:
        use_alpha=False

    source_color_img = uniform_color('color', img_in.shape[0], img_in.shape[2], img_in.shape[3], img_in.shape[1]==4, source_color)
    blend_1 = blend(source_color_img, img_in, blending_mode="subtract", opacity=1.0)
    blend_2 = blend(img_in, source_color_img, blending_mode="subtract", opacity=1.0)
    blend_3 = blend(blend_1, blend_2, blending_mode="max", opacity=1.0)
    blend_4 = blend(blend_3, blend_3, blending_mode="multiply", opacity=1.0)
    grayscale = c2g(blend_4, False, [1.0,1.0,1.0,0.0])
    grayscale = th.clamp(grayscale, 0.0, 1.0)
    grayscale = 1.0 - grayscale
    final_blend_mask = levels(grayscale, 1-threshold, 0.5, 1.0, th.max(to_tensor(source_range-0.5), th.tensor(0.0)) * 2, th.min(to_tensor(source_range), th.tensor(0.5))*2)
    final_blend_fg = replace_color(img_in, source_color, target_color)
    img_out = blend(final_blend_fg, img_in, final_blend_mask, blending_mode='copy', opacity=1.0)
    if use_alpha:
        img_out = th.cat((img_out, alpha), axis=1)
    return img_out

# Finalized
@input_check(0)
def dissolve(img_fg=None, img_bg=None, mask=None, alpha_blending=True, opacity=1.0):
    """Non-atomic function: Dissolve (https://docs.substance3d.com/sddoc/dissolve-159450368.html)

    Args:
        img_fg (tensor): Foreground (RGB(A) only).
        img_bg (tensor): Fackground (RGB(A) only).
        alpha_blending (bool, optional): Toggles blending of the Foreground and Background alpha channels. If set to False, the alpha channel of the foreground is ignored. Defaults to True.
        mask (tensor, optional): Mask slot used for masking the node's effects (G only). Defaults to None.
        opacity (float, optional): Blending Opacity between Foreground and Background. Defaults to 1.0.

    Returns:
        Tensor: Dissolved image.
    """
    assert (img_fg != None or img_bg !=None)
    use_alpha=False
    if img_fg == None:
        return img_bg
    else:
        color_input_check(img_fg, 'img_fg')
        use_alpha = use_alpha or img_fg.shape[1]==4
    if img_bg == None:
        pass
    else:
        color_input_check(img_fg, 'img_fg')
        use_alpha = use_alpha or img_bg.shape[1]==4
    if mask != None:
        grayscale_input_check(mask, 'mask')

    # generate white noise
    white_noise = th.rand([img_fg.shape[0],1,img_fg.shape[2],img_fg.shape[3]])
    blend_1 = blend(th.ones(img_fg.shape[0],1,img_fg.shape[2],img_fg.shape[3]), th.zeros(img_fg.shape[0],1,img_fg.shape[2],img_fg.shape[3]), mask, blending_mode='copy', opacity=1.0)
    blend_2 = blend(blend_1, white_noise, blending_mode='multiply', opacity=1.0)
    blend_3_mask = levels(blend_2, in_low=1.0-opacity, in_high=1.0-opacity)
    img_out = blend(img_fg, img_bg, blend_3_mask, blending_mode='copy', opacity=1.0)
    if alpha_blending==False and use_alpha:
        if img_bg==None:
            img_out[:,3,:,:] = 0
        else:
            img_out[:,3,:,:] = img_bg[:,3,:,:]

    return img_out

# Finalized
@input_check(0)
def color_blend(img_fg=None, img_bg=None, mask=None, alpha_blending=True, opacity=1.0):
    """Non-atomic function: Color Blend (https://docs.substance3d.com/sddoc/color-blend-node-159450232.html)

    Args:
        img_fg (tensor): Foreground (RGB(A) only).
        img_bg (tensor): Fackground (RGB(A) only).
        alpha_blending (bool, optional): Toggles blending of the Foreground and Background alpha channels. If set to False, the alpha channel of the foreground is ignored. Defaults to True.
        mask (tensor, optional): Mask slot used for masking the node's effects (G only). Defaults to None.
        opacity (float, optional): Blending Opacity between Foreground and Background. Defaults to 1.0.

    Returns:
        Tensor: Color blended image.
    """

    assert (img_fg != None or img_bg !=None)
    use_alpha=False
    if img_fg == None:
        pass
    else:
        color_input_check(img_fg, 'img_fg')
        use_alpha = use_alpha or img_fg.shape[1]==4
    if img_bg == None:
        pass
    else:
        color_input_check(img_bg, 'img_bg')
        use_alpha = use_alpha or img_bg.shape[1]==4
    if mask != None:
        grayscale_input_check(mask, 'mask')

    grayscale_1 = c2g(img_fg, flatten_alpha=False, rgba_weights=[0.3, 0.59, 0.11, 0.0]) if img_fg !=None else th.zeros(img_bg.shape[0],1,img_bg.shape[2],img_bg.shape[3])
    grayscale_2 = c2g(img_bg, flatten_alpha=False, rgba_weights=[0.3, 0.59, 0.11, 0.0]) if img_bg !=None else th.zeros(img_fg.shape[0],1,img_fg.shape[2],img_fg.shape[3])

    gm_1 = (append_alpha(grayscale_1.expand(-1,3,-1,-1)) if use_alpha else grayscale_1.expand(-1,3,-1,-1))
    gm_2 = (append_alpha(grayscale_2.expand(-1,3,-1,-1)) if use_alpha else grayscale_2.expand(-1,3,-1,-1))

    blend_1 = blend(gm_1, img_fg, blending_mode='subtract', opacity=1.0)
    if img_fg == None:
        blend_1[:,3,:,:] = 0.0
    elif use_alpha:
        blend_1[:,3,:,:] = img_fg[:,3,:,:]

    if img_bg == None:
        gm_2[:,3,:,:] = 0.0
    elif use_alpha:
        gm_2[:,3,:,:] = img_bg[:,3,:,:]

    gm_3 = uniform_color(mode='color', num_imgs=img_fg.shape[0], res_h=img_fg.shape[2], res_w=img_fg.shape[3], use_alpha=use_alpha, rgba=[0.0,0.0,0.0,1.0]) if img_bg == None else (append_alpha((th.clamp(grayscale_2, th.tensor(0.0), th.tensor(0.101)) / 0.101 * 249.0 / 255.0).expand(-1,3,-1,-1)) if use_alpha else (th.clamp(grayscale_2, th.tensor(0.0), th.tensor(0.101)) / 0.101 * 249.0 / 255.0).expand(-1,3,-1,-1))

    blend_2 = blend(blend_1, gm_2, blending_mode='add', opacity=1.0)
    if alpha_blending==False and use_alpha:
        blend_2[:,3,:,:] = gm_2[:,3,:,:]
    
    blend_3 = blend(gm_3, blend_2, blending_mode='multiply', opacity=1.0)
    if use_alpha:
        blend_3[:,3,:,:] = blend_2[:,3,:,:]

    blend_4 = blend(blend_3, img_bg, mask, blending_mode='copy', opacity=1.0)
    img_out = blend(blend_4, img_bg, blending_mode='copy', opacity=opacity)

    return img_out

# Finalized
@input_check(0)
def color_burn(img_fg=None, img_bg=None, mask=None, alpha_blending=True, opacity=1.0):
    """Non-atomic function: Color Burn (https://docs.substance3d.com/sddoc/color-burn-159450235.html)

    Args:
        img_fg (tensor): Foreground (RGB(A) only).
        img_bg (tensor): Fackground (RGB(A) only).
        alpha_blending (bool, optional): Toggles blending of the Foreground and Background alpha channels. If set to False, the alpha channel of the foreground is ignored. Defaults to True.
        mask (tensor, optional): Mask slot used for masking the node's effects (G only). Defaults to None.
        opacity (float, optional): Blending Opacity between Foreground and Background. Defaults to 1.0.

    Returns:
        Tensor: Color burn image.
    """
    assert (img_fg != None or img_bg !=None)
    use_alpha=False
    if img_fg == None:
        pass
    else:
        color_input_check(img_fg, 'img_fg')
        use_alpha = use_alpha or img_fg.shape[1]==4
    if img_bg == None:
        pass
    else:
        color_input_check(img_bg, 'img_bg')
        use_alpha = use_alpha or img_bg.shape[1]==4
    if mask != None:
        grayscale_input_check(mask, 'mask')

    levels_1 = img_fg if img_fg != None else uniform_color(mode='color', num_imgs=img_bg.shape[0], res_h=img_bg.shape[2], res_w=img_bg.shape[3], use_alpha=use_alpha, rgba=[0.0,0.0,0.0,1.0])
    levels_2 = (1 - img_bg) if img_bg != None else th.ones(*img_fg.shape)
    blend_1 = blend(levels_1, levels_2, blending_mode='divide', opacity=1)
    levels_3 = 1 - blend_1
    blend_2 = blend(img_fg, img_bg, blending_mode='copy',opacity=opacity)
    if use_alpha:
        levels_3[:,3,:,:] = blend_2[:,3,:,:]
    blend_3 = blend(levels_3, img_bg, mask, blending_mode='copy', opacity=1)
    img_out = blend(blend_3, img_bg, blending_mode='copy', opacity=opacity)
    if alpha_blending==False and use_alpha:
        if img_bg==None:
            img_out[:,3,:,:] = 0
        else:
            img_out[:,3,:,:] = img_bg[:,3,:,:]
    return img_out

# Finalized
@input_check(0)
def color_dodge(img_fg=None, img_bg=None, mask=None, alpha_blending=True, opacity=1.0):
    """Non-atomic function: Color Dodge (https://docs.substance3d.com/sddoc/color-dodge-159450239.html)

    Args:
        img_fg (tensor): Foreground (RGB(A) only).
        img_bg (tensor): Fackground (RGB(A) only).
        alpha_blending (bool, optional): Toggles blending of the Foreground and Background alpha channels. If set to False, the alpha channel of the foreground is ignored. Defaults to True.
        mask (tensor, optional): Mask slot used for masking the node's effects (G only). Defaults to None.
        opacity (float, optional): Blending Opacity between Foreground and Background. Defaults to 1.0.

    Returns:
        Tensor: Color dodge image.
    """   
    assert (img_fg != None or img_bg !=None)
    use_alpha=False
    if img_fg == None:
        pass
    else:
        color_input_check(img_fg, 'img_fg')
        use_alpha = use_alpha or img_fg.shape[1]==4
    if img_bg == None:
        return uniform_color(mode='color', num_imgs=img_fg.shape[0], res_h=img_fg.shape[2], res_w=img_fg.shape[3], use_alpha=(img_fg.shape[1]==4), rgba=[0.0,0.0,0.0,1.0]) * opacity
    else:
        color_input_check(img_bg, 'img_bg')
        use_alpha = use_alpha or img_bg.shape[1]==4
    if mask != None:
        grayscale_input_check(mask, 'mask')

    levels_1 = 1 - img_fg if img_fg != None else th.ones(*img_bg.shape)
    if use_alpha:
        levels_1[:,3,:,:] = 1.0
    
    blend_1 = blend(levels_1, img_bg, blending_mode='divide', opacity=1.0)
    blend_2 = blend(blend_1, img_bg, mask, blending_mode='switch', opacity=1.0)
    img_out = blend(blend_2, img_bg, blending_mode='switch', opacity=opacity)
    return img_out


# Finalized
@input_check(0)
def difference(img_bg=None, img_fg=None, mask=None, alpha_blending=True, opacity=1.0):
    """Non-atomic function: Difference (https://docs.substance3d.com/sddoc/difference-159450228.html)

    Args:
        img_fg (tensor): Foreground (RGB(A) only).
        img_bg (tensor): Fackground (RGB(A) only).
        alpha_blending (bool, optional): Toggles blending of the Foreground and Background alpha channels. If set to False, the alpha channel of the foreground is ignored. Defaults to True.
        mask (tensor, optional): Mask slot used for masking the node's effects (G only). Defaults to None.
        opacity (float, optional): Blending Opacity between Foreground and Background. Defaults to 1.0.

    Returns:
        Tensor: Difference image.
    """    
    assert (img_fg != None or img_bg !=None)
    use_alpha=False
    if img_fg == None:
        return img_bg
    else:
        color_input_check(img_fg, 'img_fg')
        use_alpha = use_alpha or img_fg.shape[1]==4
    if img_bg == None:
        return img_fg * opacity
    else:
        color_input_check(img_bg, 'img_bg')
        use_alpha = use_alpha or img_bg.shape[1]==4
    if mask != None:
        grayscale_input_check(mask, 'mask')

    # set alpha channel to 1.0
    channel_shuffle_1 = img_bg.clone()
    if use_alpha:
        channel_shuffle_1[:,3,:,:] = 1.0

    channel_shuffle_2 = img_fg.clone()
    if use_alpha:
        channel_shuffle_2[:,3,:,:] = 1.0;

    blend_1 = blend(channel_shuffle_2, channel_shuffle_1, blending_mode='max', opacity=1.0)
    blend_2 = blend(channel_shuffle_2, channel_shuffle_1, blending_mode='min', opacity=1.0)
    blend_3 = blend(img_bg, img_fg, blending_mode='copy', opacity=1.0)
    if alpha_blending==False and use_alpha:
        if img_fg==None:
            blend_3[:,3,:,:] = 0
        else:
            blend_3[:,3,:,:] = img_fg[:,3,:,:]

    blend_4 = blend(blend_2, blend_1, blending_mode='subtract', opacity=1.0)
    if use_alpha:
        blend_4[:,3,:,:] = blend_3[:,3,:,:]
    blend_5 = blend(blend_4, img_fg, mask, blending_mode='switch', opacity=1.0)
    img_out = blend(blend_5, img_fg, blending_mode='switch', opacity=opacity)
    return img_out

# Finalized
@input_check(0)
def linear_burn(img_fg=None, img_bg=None, mask=None, alpha_blending=True, opacity=1.0):
    """Non-atomic function: Linear Burn (https://docs.substance3d.com/sddoc/linear-burn-159450364.html)

    Args:
        img_fg (tensor): Foreground (RGB(A) only).
        img_bg (tensor): Fackground (RGB(A) only).
        alpha_blending (bool, optional): Toggles blending of the Foreground and Background alpha channels. If set to False, the alpha channel of the foreground is ignored. Defaults to True.
        mask (tensor, optional): Mask slot used for masking the node's effects (G only). Defaults to None.
        opacity (float, optional): Blending Opacity between Foreground and Background. Defaults to 1.0.

    Returns:
        Tensor: Linear burn image.
    """
    assert (img_fg != None or img_bg !=None)
    use_alpha=False
    if img_fg == None:
        return img_bg
    else:
        color_input_check(img_fg, 'img_fg')
        use_alpha = use_alpha or img_fg.shape[1]==4
    if img_bg == None:
        pass
    else:
        color_input_check(img_bg, 'img_bg')
        use_alpha = use_alpha or img_bg.shape[1]==4
    if mask != None:
        grayscale_input_check(mask, 'mask')

    levels_1 = levels(img_fg,0.0,0.5,1.0,0.0,0.5)
    levels_2 = levels(img_bg,0.0,0.5,1.0,0.0,0.5) if img_bg != None else uniform_color(mode='color', num_imgs=img_fg.shape[0], res_h=img_fg.shape[2], res_w=img_fg.shape[3], use_alpha=use_alpha, rgba=[0.0,0.0,0.0,1.0])

    blend_1 = blend(levels_1, levels_2, blending_mode='add', opacity=1.0)
    blend_2 = blend(uniform_color(mode='color', num_imgs=1, res_h=blend_1.shape[2], res_w=blend_1.shape[3], use_alpha=use_alpha, rgba=[0.5,0.5,0.5,1.0]), blend_1, blending_mode='subtract', opacity=1.0)
    blend_3 = blend(blend_2, blend_2, blending_mode='add', opacity=1.0)
    blend_4 = blend(img_fg, img_bg, blending_mode='copy', opacity=opacity)
    if use_alpha:
        blend_3[:,3,:,:] = blend_4[:,3,:,:]

    blend_5 = blend(blend_3, img_bg, blending_mode='copy', opacity=1.0)
    if alpha_blending==False and use_alpha:
        if img_bg==None:
            blend_5[:,3,:,:] = 0
        else:
            blend_5[:,3,:,:] = img_bg[:,3,:,:]

    blend_6 = blend(blend_5, img_bg, mask, blending_mode='switch', opacity=1.0)
    img_out = blend(blend_6, img_bg, blending_mode='switch', opacity=opacity)
    return img_out

# Finalized
@input_check(0)
def luminosity(img_fg=None, img_bg=None, mask=None, alpha_blending=True, opacity=1.0):
    """Non-atomic function: Luminosity (https://docs.substance3d.com/sddoc/luminosity-blend-node-159450373.html)

    Args:
        img_fg (tensor): Foreground (RGB(A) only).
        img_bg (tensor): Fackground (RGB(A) only).
        alpha_blending (bool, optional): Toggles blending of the Foreground and Background alpha channels. If set to False, the alpha channel of the foreground is ignored. Defaults to True.
        mask (tensor, optional): Mask slot used for masking the node's effects (G only). Defaults to None.
        opacity (float, optional): Blending Opacity between Foreground and Background. Defaults to 1.0.

    Returns:
        Tensor: Luminosity image.
    """    
    assert (img_fg != None or img_bg !=None)
    use_alpha=False

    if img_fg == None:
        pass
    else:
        color_input_check(img_fg, 'img_fg')
        use_alpha = use_alpha or img_fg.shape[1]==4
    if img_bg == None:
        pass
    else:
        color_input_check(img_bg, 'img_bg')
        use_alpha = use_alpha or img_bg.shape[1]==4
    if mask != None:
        grayscale_input_check(mask, 'mask')

    grayscale_fg = c2g(img_fg, flatten_alpha=False, rgba_weights=[1.0 /3.0,1.0 /3.0,1.0 /3.0,0.0], bg=1.0) if img_fg != None else th.zeros(img_bg.shape[0], 1, img_bg.shape[2], img_bg.shape[3])
    grayscale_bg = c2g(img_bg, flatten_alpha=False, rgba_weights=[1.0 /3.0,1.0 /3.0,1.0 /3.0,0.0], bg=1.0) if img_bg != None else th.zeros(img_fg.shape[0], 1, img_fg.shape[2], img_fg.shape[3])
    gm_fg = append_alpha(grayscale_fg.repeat(1,3,1,1)) if use_alpha else grayscale_fg.repeat(1,3,1,1)
    gm_bg = append_alpha((1-grayscale_bg).repeat(1,3,1,1)) if use_alpha else (1-grayscale_bg).repeat(1,3,1,1)
    blend_1 = blend(gm_bg, img_bg, blending_mode="add_sub", opacity=0.5)
    blend_2 = blend(gm_fg, blend_1, blending_mode="add_sub", opacity=0.5)
    blend_3 = blend(blend_2, img_bg, mask, blending_mode="copy", opacity=1.0)
    img_out = blend(blend_3, img_bg, blending_mode="copy", opacity=opacity)

    img_out = blend(blend_3, img_bg, blending_mode="copy", opacity=opacity)
    if alpha_blending==False and use_alpha:
        if img_bg==None:
            img_out[:,3,:,:] = 0
        else:
            img_out[:,3,:,:] = img_bg[:,3,:,:]

    return img_out

# Finalized
@input_check(2)
def multi_dir_warp(img_in, intensity_mask, mode='average', directions=4, intensity=0.5, max_intensity=20.0, angle=0.0):
    """Non-atomic function: Multi Directional Warp (https://docs.substance3d.com/sddoc/multi-directional-warp-180192289.html)

    Args:
        img_in (tensor): Base map to which the warping will be applied. Can be color or grayscale.
        intensity_mask (tensor): Mandatory mask map that drives the intensity of the warping effect, must be grayscale.
        mode (str, optional): Sets the Blend mode for consecutive passes. Only has effect if Direcions is 2 or 4! Defaults to 'average'.
        directions (int, optional): Sets in how many Axes the warp works. 1 means it moves in the direction of the Angle, and the opposite of that direction, 2 means the axis of the angle, plus the perpendicular axis, 4 means the previous axes, plus 45 degree inclements. Defaults to 4.
        intensity (float, optional): Sets the intensity of the warp effect, how far to push pixels out (normalized). Defaults to 0.5.
        max_intensity (float, optional): Maximum warp intensity. Defaults to 20.0.
        angle (float, optional): Sets the Angle or direction in which to apply the Warp effect. Defaults to 0.0.

    Returns:
        Tensor: Multi-directional warped image.
    """

    grayscale_input_check(intensity_mask, 'intensity_mask')

    if mode=='average':
        blending_mode='copy'
    elif mode=='min' or mode=='max':
        blending_mode=mode
    else:
        blending_mode='screen'

    d_warp_1 = d_warp(img_in, intensity_mask, intensity, max_intensity, (angle+0.875) % 1)
    d_warp_2 = d_warp(d_warp_1, intensity_mask, intensity, max_intensity, (angle+0.375) % 1)
    d_warp_3 = d_warp(d_warp_2, intensity_mask, intensity, max_intensity, (angle+0.625) % 1)
    d_warp_4 = d_warp(d_warp_3, intensity_mask, intensity, max_intensity, (angle+0.125) % 1)

    blend_1 = blend(d_warp_4, img_in, blending_mode='switch', opacity=1 if directions > 2 else 0)

    d_warp_5 = d_warp(blend_1, intensity_mask, intensity, max_intensity, (angle+0.75) % 1)
    d_warp_6 = d_warp(d_warp_5, intensity_mask, intensity, max_intensity, (angle+0.25) % 1)

    blend_2 = blend(d_warp_6, img_in, blending_mode='switch', opacity=1 if directions > 1 else 0)

    d_warp_7 = d_warp(blend_2, intensity_mask, intensity, max_intensity, (angle+0.5) % 1)
    d_warp_8 = d_warp(d_warp_7, intensity_mask, intensity, max_intensity, angle % 1)

    d_warp_9 = d_warp(img_in, intensity_mask, intensity, max_intensity, (angle+0.625) % 1)
    d_warp_10 = d_warp(d_warp_9, intensity_mask, intensity, max_intensity, (angle+0.125) % 1)

    blend_3 = blend(d_warp_10, d_warp_2, blending_mode=blending_mode, opacity=0.5)

    d_warp_11 = d_warp(img_in, intensity_mask, intensity, max_intensity, (angle+0.75) % 1)
    d_warp_12 = d_warp(d_warp_11, intensity_mask, intensity, max_intensity, (angle+0.25) % 1)

    d_warp_13 = d_warp(img_in, intensity_mask, intensity, max_intensity, (angle+0.5) % 1)
    d_warp_14 = d_warp(d_warp_13, intensity_mask, intensity, max_intensity, angle % 1)

    blend_4 = blend(d_warp_14, d_warp_12, blending_mode=blending_mode, opacity=0.5)
    blend_5 = blend(blend_4, blend_3, blending_mode=blending_mode, opacity=0.5)
    blend_6 = blend(blend_4, d_warp_14, blending_mode='switch', opacity=1 if directions > 2 else 0)
    blend_7 = blend(blend_5, blend_6, blending_mode='switch', opacity=1 if directions == 4 else 0)
    img_out = blend(d_warp_8, blend_7, blending_mode='switch', opacity=1 if blending_mode=='screen' else 0)
    return img_out

# changed
@input_check(1)
def shape_drop_shadow(img_in, input_is_pre_multiplied=True, pre_multiplied_output=False, angle=0.25, dist=0.52, 
    max_dist = 0.5, size=0.15, max_size=1.0, spread=0.0, opacity=0.5, mask_color=[1.0,1.0,1.0], shadow_color=[0.0,0.0,0.0]):
    """Non-atomic function: Shape Drop Shadow (https://docs.substance3d.com/sddoc/shape-drop-shadow-159450554.html)

    Args:
        img_in (tensor): Input image.
        input_is_pre_multiplied (bool, optional): Whether the input should be assumed as pre-multiplied (color version only). Defaults to True.
        pre_multiplied_output (bool, optional): Whether the output should be pre-multiplied. Defaults to False.
        angle (float, optional): Incidence Angle of the (fake) light. Defaults to 0.25.
        dist (float, optional): Distance the shadow drop down to/moves away from the shape (normalized). Defaults to 0.52.
        max_dist (float, optional): Maximum distance. Defaults to 0.5.
        size (float, optional): Controls blurring/fuzzines of the shadow (normalized). Defaults to 0.15.
        max_size (float, optional): Maximum blurring/fuzzines size. Defaults to 1.0.
        spread (float, optional): Cutoff/treshold for the blurring effect, makes the shadow spread away further. Defaults to 0.0.
        opacity (float, optional): Blending Opacity for the shadow effect. Defaults to 0.5.
        mask_color (list, optional): Solid color to be used for the transparency mapped output. Defaults to [1.0,1.0,1.0].
        shadow_color (list, optional): Color tint to be applied to the shadow. Defaults to [0.0,0.0,0.0].

    Returns:
        Tensor: Shape drop shadow image.
    """

    angle = to_tensor(angle)
    dist =to_tensor((dist*2.0-1.0)*max_dist)
    size = to_tensor(size*max_size)
    use_alpha = img_in.shape[1]==4

    if img_in.shape[1] == 1:
        alpha = img_in.clone()
        img_in_new = alpha_merge(uniform_color('color', img_in.shape[0], img_in.shape[2], img_in.shape[3], False, mask_color), alpha)
    elif img_in.shape[1] == 4:
        alpha = img_in[:,3:4,:,:]
        img_in_new = img_in
    else:
        alpha = th.ones(img_in.shape[0], 1, img_in.shape[2], img_in.shape[3])
        img_in_new = append_alpha(img_in)
    
    alpha_gm = append_alpha(alpha.expand(-1,3,-1,-1))
    blend_1 = blend(alpha_gm, img_in_new, blending_mode='divide', opacity=1.0 if input_is_pre_multiplied else 0)
    alpha_merge_1 = alpha_merge(blend_1, alpha)
    invert_alpha = 1 - alpha

    tmp_x = dist * th.cos((angle-0.5)*np.pi*2.0) / 2.0 + 0.5 + 0.5
    tmp_y = dist * th.sin((angle-0.5)*np.pi*2.0) / 2.0 + 0.5 + 0.5

    transform_2d_1 = transform_2d(alpha, x_offset = (tmp_x+1.0)/2.0, y_offset = (tmp_y+1.0)/2.0)
    blur_hq_1 = blur_hq(transform_2d_1, high_quality=False, intensity=1.0, max_intensity=size*size*64)
    levels_1 = levels(blur_hq_1, 0.0, 0.5, 1.0-spread, 0.0, opacity)
    blend_2 = blend(invert_alpha, levels_1, blending_mode='multiply', opacity=1.0)

    uniform_color_1 = uniform_color(res_h=img_in.shape[2], res_w=img_in.shape[3], use_alpha=False, rgba=shadow_color)
    alpha_merge_2 = th.cat([uniform_color_1, levels_1], axis=1)
    blend_3 = blend(alpha_merge_1, alpha_merge_2, blending_mode='copy', opacity=1.0)
    
    blend_3_rgb = blend_3[:,:3,:,:]
    blend_3_alpha = blend_3[:,3:4,:,:]
    blend_4 = blend(blend_3_rgb, blend_3_alpha.expand(-1,3,-1,-1), blending_mode='multiply', opacity=1.0)
    channel_shuffle_1 = th.cat([blend_4, blend_3_alpha], axis=1)
    if pre_multiplied_output:
        return channel_shuffle_1[:,:4 if use_alpha else 3,:,:], blend_2
    else:
        return blend_3[:,:4 if use_alpha else 3,:,:], blend_2

# changed
@input_check(1)
def shape_glow(img_in, input_is_pre_multiplied=True, pre_multiplied_output=False, mode='soft', 
    width=0.625, spread=0.0, opacity=1.0, mask_color=[1.0,1.0,1.0],glow_color=[1.0,1.0,1.0]):
    """Non-atomic function: Shape Glow (https://docs.substance3d.com/sddoc/shape-glow-159450558.html)

    Args:
        img_in (tensor): Input image.
        input_is_pre_multiplied (bool, optional): Whether the input should be assumed as pre-multiplied. Defaults to True.
        pre_multiplied_output (bool, optional): Whether the output should be pre-multiplied. Defaults to False.
        mode (str, optional): Switches between two accuracy modes. Defaults to 'soft'.
        width (float, optional): Controls how far the glow reaches (normalized). Defaults to 0.625.
        spread (float, optional): Cut-off / treshold for the blurring effect, makes the glow appear solid close to the shape. Defaults to 0.0.
        opacity (float, optional): Blending Opacity for the glow effect. Defaults to 1.0.
        mask_color (list, optional): Solid color to be used for the transparency mapped output. Defaults to [1.0,1.0,1.0].
        glow_color (list, optional): Color tint to be applied to the glow. Defaults to [1.0,1.0,1.0].

    Returns:
        Tensor: Shape glow image.
    """

    width = to_tensor(width*2.0-1.0)
    use_alpha = img_in.shape[1]==4

    if img_in.shape[1] == 1:
        alpha = img_in.clone()
        img_in_new = alpha_merge(uniform_color('color', img_in.shape[0], img_in.shape[2], img_in.shape[3], False, mask_color), alpha)
    elif img_in.shape[1] == 4:
        alpha = img_in[:,3:4,:,:]
        img_in_new = img_in
    else:
        alpha = th.ones(img_in.shape[0], 1, img_in.shape[2], img_in.shape[3])
        img_in_new = append_alpha(img_in)
    
    # bottom path
    alpha_gm = append_alpha(alpha.expand(-1,3,-1,-1))
    blend_1 = blend(alpha_gm, img_in_new, blending_mode='divide', opacity=1.0)
    if input_is_pre_multiplied:
        alpha_merge_1 = alpha_merge(blend_1, alpha)
    else:
        alpha_merge_1 = alpha_merge(img_in_new, alpha)
    
    # top path
    invert_alpha = 1 - alpha if width < 0 else alpha
    levels_1 = levels(invert_alpha,0,0.5,0.0314,0.0,1.0)
    distance_1 = distance(levels_1, mode='gray', combine=True, use_alpha=False, dist=1.0, max_dist=8*th.abs(width))
    distance_2 = distance(levels_1, mode='gray', combine=True, use_alpha=False, dist=1.0, max_dist=128*width*width)
    blur_hq_1 = blur_hq(distance_1, high_quality=False, intensity=1.0, max_intensity=64*width*width)
    blur_hq_2 = blur_hq(distance_2, high_quality=False, intensity=1.0, max_intensity=2*(1-th.abs(width)))
    srgb_1 = linear_to_srgb(blur_hq_1)
    levels_2 = levels(srgb_1 if mode=='soft' else blur_hq_2, 0.0, 0.5, 1.0-spread, 0.0, 1.0)

    invert_invert_alpha = 1 - invert_alpha
    # output
    blend_2 = blend(invert_invert_alpha, levels_2, blending_mode='multiply', opacity=1.0)
    blend_3 = blend(levels_2, alpha, blending_mode='max', opacity=opacity)
    uniform_color_1 = append_alpha(uniform_color('color', img_in.shape[0], img_in.shape[2], img_in.shape[3], False, glow_color))
    blend_4 = blend(alpha_merge_1, uniform_color_1, blending_mode='copy', opacity=1.0)
    gm_1 = append_alpha(blend_3.expand(-1,3,-1,-1))
    blend_5 = blend(gm_1, blend_4, blending_mode='multiply', opacity=1.0)
    alpha_merge_2 = alpha_merge(blend_5 if pre_multiplied_output else blend_4, blend_3)

    levels_3 = levels(levels_2, 0.0, 0.5, 1.0, 0.0, opacity)
    levels_3_gm = append_alpha(levels_3.expand(-1,3,-1,-1))
    blend_6 = blend(uniform_color_1, blend_1 if input_is_pre_multiplied else img_in_new, blending_mode='copy', opacity=1.0)
    blend_7 = blend(levels_3_gm, blend_6, blending_mode='multiply', opacity=1.0)
    
    alpha_merge_3 = alpha_merge(blend_7 if pre_multiplied_output else blend_6, alpha)

    if width >= 0:
        return alpha_merge_2[:,:4 if use_alpha else 3,:,:], blend_2
    else:
        return alpha_merge_3[:,:4 if use_alpha else 3,:,:], blend_2

# Finalized
@input_check(1)
def swirl(img_in, tile_mode=3, amount=0.75, max_amount=16.0, x1=1.0, x1_max=2.0, x2=0.5, x2_max=1.0, x_offset=0.5, x_offset_max=1.0, y1=0.5, y1_max=1.0, y2=1.0, y2_max=2.0, y_offset=0.5, y_offset_max=1.0):
    """Non-atomic function: Swirl (https://docs.substance3d.com/sddoc/swirl-166363401.html)

    Args:
        img_in (tensor): Input image.
        tile_mode (int, optional): Tile mode. Defaults to 3 (horizontal and vertical).
        amount (float, optional): Strength of the swirling effect (normalized). Defaults to 0.75.
        max_amount (float, optional): Maximum strength of swirling effect. Defaults to 16.0.
        x1 (float, optional): Entry in the affine transformation matrix, same for the below. Defaults to 1.0.
        x1_max (float, optional): . Defaults to 2.0.
        x2 (float, optional): . Defaults to 0.5.
        x2_max (float, optional): . Defaults to 1.0.
        x_offset (float, optional): . Defaults to 0.5.
        x_offset_max (float, optional): . Defaults to 1.0.
        y1 (float, optional): . Defaults to 0.5.
        y1_max (float, optional): . Defaults to 1.0.
        y2 (float, optional): . Defaults to 2.0.
        y2_max (float, optional): . Defaults to 1.0.
        y_offset (float, optional): . Defaults to 0.5.
        y_offset_max (float, optional): . Defaults to 1.0.

    Returns:
        Tensor: Swirl image.
    """

    amount = th.tensor((amount*2.0-1.0)*max_amount)
    # aliased variables
    # x->x1
    # z->x2
    # y->y1
    # w->y2
    # xx->x_offset
    # yy->y_offset

    x = to_tensor((x1 * 2.0 - 1.0) * x1_max).squeeze()
    z = to_tensor((x2 * 2.0 - 1.0) * x2_max).squeeze()
    xx = to_tensor((x_offset * 2.0 - 1.0) * x_offset_max).squeeze()
    y = to_tensor((y1 * 2.0 - 1.0) * y1_max).squeeze()
    w = to_tensor((y2 * 2.0 - 1.0) * y2_max).squeeze()
    yy = to_tensor((y_offset * 2.0 - 1.0) * y_offset_max).squeeze()

    def transform_position(position_x, position_y, x, z, xx, y, w, yy):
        position_x_ = (position_x - 0.5)*x + (position_y - 0.5)*z + 0.5 + xx
        position_y_ = (position_x - 0.5)*y + (position_y - 0.5)*w + 0.5 + yy
        return position_x_, position_y_

    def inverse_transform_position(position_x, position_y, x, z, xx, y, w, yy):
        mtx = th.tensor([[x,z],[y,w]])
        inv_mtx = th.inverse(mtx)
        position_x_, position_y_ = transform_position(position_x-xx, position_y-yy, inv_mtx[0][0], inv_mtx[0][1],0,inv_mtx[1][0], inv_mtx[1][1],0)
        return position_x_, position_y_
    
    num_row = img_in.shape[2]
    num_col = img_in.shape[3]
    row_grid, col_grid = th.meshgrid(th.linspace(0, num_row-1, num_row), th.linspace(0, num_col-1, num_col))
    row_grid = (row_grid+0.5)/num_row
    col_grid = (col_grid+0.5)/num_col

    inv_center_col, inv_center_row = inverse_transform_position(0.5, 0.5, x, z, xx, y, w, yy)
    minus_opt_1_row = row_grid - (inv_center_row + 0.5)
    minus_opt_1_col = col_grid - (inv_center_col + 0.5)
    floor_opt_1_row = th.floor(minus_opt_1_row)
    floor_opt_1_col = th.floor(minus_opt_1_col)
    plus_opt_1_row = floor_opt_1_row + (inv_center_row + 0.5) + 0.5
    plus_opt_1_col = floor_opt_1_col + (inv_center_col + 0.5) + 0.5

    # bottom part
    or_opt_1_row = (minus_opt_1_row > 0) | (minus_opt_1_row < -1)
    or_opt_1_col = (minus_opt_1_col > 0) | (minus_opt_1_col < -1)
    or_opt_2_row = (tile_mode == 0) or (tile_mode == 1)
    or_opt_2_col = (tile_mode == 0) or (tile_mode == 2)
    and_opt_1_row = th.tensor(or_opt_2_row) & or_opt_1_row
    and_opt_1_col = th.tensor(or_opt_2_col) & or_opt_1_col

    # middle part
    active_row = plus_opt_1_row.clone()
    active_row[and_opt_1_row] = inv_center_row
    active_col = plus_opt_1_col.clone()
    active_col[and_opt_1_col] = inv_center_col

    # right part
    trans_1_col, trans_1_row = transform_position(active_col, active_row,x,z,-0.5,y,w,-0.5)
    neg_trans_1_col = trans_1_col * -1.0
    neg_trans_1_row = trans_1_row * -1.0
    trans_2_col, trans_2_row = transform_position(col_grid, row_grid, x,z,neg_trans_1_col, y,w,neg_trans_1_row)
    dist = th.sqrt((trans_2_col-0.5)**2 + (trans_2_row-0.5)**2)
    minus_opt_2_row = trans_2_row - 0.5
    minus_opt_2_col = trans_2_col - 0.5
    before_cos = (th.max(0.5-dist, th.tensor(0.0))**2)*2*np.pi*amount
    after_cos = th.cos(before_cos) 
    after_sin = th.sin(before_cos)
    before_inv_col = after_cos * minus_opt_2_col + after_sin * minus_opt_2_row + 0.5
    before_inv_row = after_cos * minus_opt_2_row - after_sin * minus_opt_2_col + 0.5
    final_col,final_row = inverse_transform_position(before_inv_col, before_inv_row, x, z,neg_trans_1_col, y, w, neg_trans_1_row)

    # sample and return
    final_col = (final_col % 1.0) * 2.0 - 1.0
    final_row = (final_row % 1.0) * 2.0 - 1.0
    sample_grid = th.stack([final_col, final_row], 2).expand(img_in.shape[0], num_row, num_col, 2)
    sample_grid = sample_grid * num_row / (num_row + 2)
    img_in_pad = th.nn.functional.pad(img_in, (1, 1, 1, 1), mode='circular')
    img_out = th.nn.functional.grid_sample(img_in_pad, sample_grid, "bilinear", 'zeros', align_corners=False)
    return img_out

# Finalized
@input_check(1)
def curvature_sobel(img_in, normal_format='dx', intensity=0.75, max_intensity=1.0):
    """Non-atomic function: Curvature Sobel (https://docs.substance3d.com/sddoc/curvature-sobel-159450520.html)

    Args:
        img_in (tensor): Input image (RGB(A) only).
        normal_format (str, optional): Normal format. Defaults to 'dx'.
        intensity (float, optional): Intensity of the effect, adjusts contrast (normalized). Defaults to 0.5.
        max_intensity (float, optional): Maximum intensity. Defaults to 1.0.

    Returns:
        Tensor: Curvature sobel image.
    """

    color_input_check(img_in, "img_in")
    
    intensity = to_tensor((intensity * 2.0 - 1.0) * max_intensity)

    inverted_normal = normal_invert(img_in)
    blend_1 = blend(inverted_normal, img_in, blending_mode='switch', opacity=0.0 if normal_format=='dx' else 1.0)
    blend_1 = th.nn.functional.pad(blend_1, [1,1,1,1], mode='circular')
    blend_1_r = blend_1[:,0:1,:,:]
    blend_1_g = blend_1[:,1:2,:,:]

    x_kernel = th.tensor([[-1.0,0.0,1.0], [-2.0,0.0,2.0], [-1.0,0.0,1.0]]).view(1,1,3,3)
    y_kernel = th.tensor([[-1.0,-2.0,-1.0], [0.0,0.0,0.0], [1.0,2.0,1.0]]).view(1,1,3,3)

    sobel_x = th.nn.functional.conv2d(blend_1_r, x_kernel, groups=1, padding = 0)
    sobel_y = th.nn.functional.conv2d(blend_1_g, y_kernel, groups=1, padding = 0)

    img_out = (sobel_x + sobel_y) * intensity * 0.5 + 0.5

    return img_out

# Double check
@input_check(2)
def emboss_with_gloss(img_in, height, intensity=0.5, max_intensity=10.0, light_angle=0.0, gloss=0.625, max_gloss=1.0, highlight_color=[1.0,1.0,1.0], shadow_color=[0.0,0.0,0.0]):
    """Non-atomic function: Emboss with Gloss (https://docs.substance3d.com/sddoc/emboss-with-gloss-159450527.html)

    Args:
        img_in (tensor): Input image (RGB(A) only).
        height (tensor): Height image (G only).
        intensity (float, optional): Normalized intensity of the highlight. Defaults to 0.5.
        max_intensity (float, optional): Maximum intensity. Defaults to 10.0.
        light_angle (float, optional): Light angle. Defaults to 0.0.
        gloss (float, optional): Glossiness highlight size. Defaults to 0.25.
        max_gloss (float, optional): Max glossiness. Defaults to 1.0.
        highlight_color (list, optional): Highlight color. Defaults to [1.0,1.0,1.0].
        shadow_color (list, optional): Shadow color. Defaults to [0.0,0.0,0.0].

    Returns:
        Tensor: Emboss with gloss image.
    """    
    
    color_input_check(img_in, 'img_in')    
    grayscale_input_check(height, 'height')
    gloss = to_tensor((gloss *2.0 - 1.0) * max_gloss)

    if img_in.shape[1] == 4:
        use_alpha=True
        alpha = img_in[:,3:4,:,:]
        img_in = img_in[:,:3,:,:]
    else:
        use_alpha=False
        
    emboss_1 = emboss(uniform_color(mode='color', num_imgs=1, res_h=img_in.shape[2], res_w=img_in.shape[3], use_alpha=False, rgba=[0.4980, 0.4980, 0.4980, 1.0]),
        height, intensity=intensity, max_intensity=max_intensity, light_angle=light_angle, highlight_color=highlight_color, shadow_color=shadow_color)
    levels_1 = levels(emboss_1, 0.503831, 0.5, 1.0, 0.0, 1.0)
    levels_2 = levels(emboss_1, 0.0, 0.5, 0.484674, 1.0, 0.0)
    grayscale_1 = c2g(levels_1, flatten_alpha=False, rgba_weights=[1.0/3.0, 1.0/3.0, 1.0/3.0, 0.0], bg=1.0)
    blur_hq_1 = blur_hq(grayscale_1, high_quality=False, intensity=1.5/16.0, max_intensity=16.0)
    warp_1 = warp(levels_1, blur_hq_1, intensity=-1.0*gloss, max_intensity=max_gloss)
    blend_1 = blend(warp_1, img_in, blending_mode="add", opacity=1.0)
    blend_2 = blend(levels_2, blend_1, blending_mode="subtract", opacity=1.0)
    img_out = blend(img_in, blend_2, blending_mode='switch', opacity=1 if intensity==0 else 0)
    if use_alpha:
        img_out = th.cat((img_out, alpha), axis=1)
    return img_out

# Finalized
@input_check(1)
def facing_normal(img_in):
    """Non-atomic function: Facing Normal (https://docs.substance3d.com/sddoc/facing-normal-159450567.html)

    Args:
        img_in (tensor): Input image (RGB(A) only).

    Returns:
        Tensor: Facing normal image.
    """    
    color_input_check(img_in, 'img_in')    

    grayscale_1 = img_in[:,0:1,:,:]
    grayscale_2 = img_in[:,1:2,:,:]
    
    blend_1 = blend(levels(grayscale_1, 0.5,0.5,1.0,0.0,1.0), levels(grayscale_1, 0.0,0.5,0.5,0.0,1.0), blending_mode="subtract", opacity=1.0)
    blend_2 = blend(levels(grayscale_2, 0.5,0.5,1.0,0.0,1.0), levels(grayscale_2, 0.0,0.5,0.5,0.0,1.0), blending_mode="subtract", opacity=1.0)
    img_out = blend(blend_1, blend_2, blending_mode="multiply", opacity=1.0)
    return img_out

# Retest this node
@input_check(2)
def height_normal_blend(img_height, img_normal, normal_format='dx', normal_intensity=0.5, max_normal_intensity=1.0):
    """Non-atomic function: Height Normal Blender (https://docs.substance3d.com/sddoc/height-normal-blender-159450570.html)

    Args:
        img_height (tensor): Grayscale Heightmap to blend with (G only).
        img_normal (tensor): Base Normalmap to blend onto (RGB(A) only).
        normal_format (str, optional): Normal format. Defaults to 'dx'.
        normal_intensity (float, optional): Normalized normal intensity. Defaults to 0.0.
        max_normal_intensity (float, optional): Max normal intensity. Defaults to 1.0.

    Returns:
        Tensor: Height normal blender image.
    """
    grayscale_input_check(img_height, 'img_height')
    color_input_check(img_normal, 'img_normal')
    use_alpha = img_normal.shape[1]==4

    normal_1 = normal(img_height, mode='tangent_space', normal_format=normal_format, use_input_alpha=False, use_alpha=False, intensity=normal_intensity, max_intensity=max_normal_intensity)
    blend_1 = blend(normal_1[:,0:1,:,:], img_normal[:,0:1,:,:], blending_mode="add_sub", opacity=0.5)
    blend_2 = blend(normal_1[:,1:2,:,:], img_normal[:,1:2,:,:], blending_mode="add_sub", opacity=0.5)
    blend_3 = blend(normal_1[:,2:3,:,:], img_normal[:,2:3,:,:], blending_mode="add_sub", opacity=0.5)
    rgba_merge_1 = rgba_merge(blend_1, blend_2, blend_3, a=uniform_color(mode='gray', num_imgs=1, res_h=img_normal.shape[2], res_w=img_normal.shape[3], use_alpha=False, rgba=[1.0]), use_alpha=use_alpha)
    img_out = normal_normalize(rgba_merge_1)
    return img_out

# changed
@input_check(1)
def normal_invert(img_in, invert_red=False, invert_green=True, invert_blue=False, invert_alpha=False):
    """Non-atomic function: Normal Invert (https://docs.substance3d.com/sddoc/normal-invert-159450583.html)

    Args:
        img_in (tensor): Normal image (RGB(A) only)/
        invert_red (bool, optional): invert red channel flag. Defaults to False.
        invert_green (bool, optional): invert green channel flag. Defaults to True.
        invert_blue (bool, optional): invert blue channel flag. Defaults to False.
        invert_alpha (bool, optional): invert alpha channel flag. Defaults to False.

    Returns:
        Tensor: Normal inverted image.
    """    
    color_input_check(img_in, 'img_in')
    img_out = img_in.clone()
    if invert_red:
        img_out[:,0,:,:] = 1.0 - img_out[:,0,:,:]
    if invert_green:
        img_out[:,1,:,:] = 1.0 - img_out[:,1,:,:]
    if invert_blue:
        img_out[:,2,:,:] = 1.0 - img_out[:,2,:,:]
    if img_out.shape[1]==4 and invert_alpha:
        img_out[:,3,:,:] = 1.0 - img_out[:,3,:,:]
    return img_out

# Finalized
@input_check(1)
def skew(img_in, axis="horizontal", align="top_left", amount=0.5, max_amount=1.0):
    """Non-atomic function: Skew (https://docs.substance3d.com/sddoc/skew-159450652.html)

    Args:
        img_in (tensor): Input image.
        axis (str, optional): Choose to skew vertically or horizontally. Defaults to "horizontal".
        align (str, optional): Sets the origin point of the Skew transformation. Defaults to "top_left".
        amount (int, optional): Normalized amount of skew. Defaults to 0.
        max_amount (float, optional): Maximum skew amount. Defaults to 1.0.

    Returns:
        Tensor: Skewed image.
    """    
    amount = to_tensor((amount * 2.0 - 1.0) * max_amount)

    if axis == "horizontal":
        x1 = 1.0
        x2 = 0.0
        y1 = amount
        y2 = 1.0
    else:
        x1 = 1.0
        x2 = amount
        y1 = 0.0
        y2 = 1.0
    
    if align == "center":
        x_offset = 0.0
        y_offset = 0.0
    elif align == "top_left":
        if axis == "horizontal":
            x_offset = 0.5*amount
            y_offset = 0.0
        else:
            x_offset = 0.0
            y_offset = 0.5*amount
    else: # "center"
        if axis == "horizontal":
            x_offset = -0.5*amount
            y_offset = 0.0
        else:
            x_offset = 0.0
            y_offset = -0.5*amount

    img_out = transform_2d(img_in, tile_mode=3, sample_mode='bilinear', mipmap_mode='auto', mipmap_level=0, x1=(x1+1.0)/2.0, x1_max=1.0, x2=(x2+1.0)/2.0, x2_max=1.0,
                 x_offset=(x_offset+1.0)/2.0, x_offset_max=1.0, y1=(y1+1.0)/2.0, y1_max=1.0, y2=(y2+1.0)/2.0, y2_max=1.0, y_offset=(y_offset+1.0)/2.0, y_offset_max=1.0)
    return img_out

# changed
@input_check(1)
def trapezoid_transform(img_in, sampling='bilinear', tile_mode=3, top_stretch=0.5, max_top_stretch=1.0, bottom_stretch=0.5, max_bottom_stretch=1.0, bg_color=[0.0,0.0,0.0,1.0]):
    """Non-atomic function: Trapezoid Transform (https://docs.substance3d.com/sddoc/trapezoid-transform-172819821.html)

    Args:
        img_in (tensor): Input image.
        sampling (str, optional): Set sampling quality. Defaults to 'bilinear', other option 'nearest'.
        tile_mode (int, optional): Tiling mode. Defaults to 3.
        top_stretch (float, optional): Set the amount of stretch or squash at the top. Defaults to 0.0.
        bottom_stretch (float, optional): Set the amount of stretch or squash at the botton. Defaults to 0.0.
        bg_color (list, optional): Set solid background color in case tiling is turned off. Defaults to [0.0,0.0,0.0,1.0].

    Returns:
        Tensor: Trapezoid transformed image.
    """
    top_stretch = to_tensor((top_stretch * 2.0 - 1.0) * max_top_stretch)
    bottom_stretch = to_tensor((bottom_stretch * 2.0 - 1.0) * max_bottom_stretch)

    lerp_a = (1-top_stretch)
    lerp_b = (1-bottom_stretch)

    num_row = img_in.shape[2]
    num_col = img_in.shape[3]
    bg_color = to_tensor(bg_color[:img_in.shape[1]]).view(1,img_in.shape[1],1,1).expand(*img_in.shape)
    row_grid, col_grid = th.meshgrid(th.linspace(0, num_row-1, num_row), th.linspace(0, num_col-1, num_col))
    row_grid = (row_grid+0.5)/num_row
    col_grid = (col_grid+0.5)/num_col

    denominator = (1-row_grid)*lerp_a + row_grid*lerp_b
    nominator = col_grid - 0.5
    division = nominator / denominator
    division_abs = th.abs(division)
    division_add = ((division + 0.5) % 1.0) * 2.0 - 1.0
    row_grid = row_grid * 2.0 - 1.0
    sample_grid = th.stack([division_add, row_grid], 2).expand(img_in.shape[0], num_row, num_col, 2)
    sample_grid = sample_grid * num_row / (num_row + 2)
    img_in_pad = th.nn.functional.pad(img_in, (1, 1, 1, 1), mode='circular')
    img_out = th.nn.functional.grid_sample(img_in_pad, sample_grid, sampling, 'zeros', align_corners=False)
    mask_1 = ((division_abs > 0.5) & (tile_mode % 2 == 0)).expand(img_in.shape[0], img_in.shape[1], num_row, num_col)
    img_out[mask_1] = bg_color[mask_1]
    return img_out

# Changed
@input_check(1)
def color_to_mask(img_in, flatten_alpha=False, keying_type='rgb', rgb=[0.0,1.0,0.0], mask_range=0.0, mask_softness=0.0):
    """Non-atomic function: Color to Mask (https://docs.substance3d.com/sddoc/color-to-mask-159449185.html)

    Args:
        img_in (tensor): Input image (RGB(A) only).
        flatten_alpha (bool, optional): Whether the alpha should be flattened for the result. Defaults to False.
        keying_type (str, optional): Keying type for isolating the color. Defaults to 'rgb', other options chrominance|luminance.
        rgb (list, optional): Which color to base the mask on. Defaults to [0.0,1.0,0.0].
        mask_range (float, optional): Width of the range that should be selected. Defaults to 0.0.
        mask_softness (float, optional): How hard the contrast/falloff of the mask should be. Defaults to 0.0.

    Returns:
        Tensor: Color to mask image.
    """
    color_input_check(img_in, "img_in")
    use_alpha = img_in.shape[1]==4

    # check if use alpha is necessary
    grayscale_1 = c2g(img_in, flatten_alpha=flatten_alpha, rgba_weights = [0.299, 0.587, 0.114, 0.0], bg=0.0)
    gm_1 = append_alpha((1.0-grayscale_1).expand(-1,3,-1,-1)) if use_alpha else (1.0-grayscale_1).expand(-1,3,-1,-1)
    blend_1 = blend(gm_1, img_in, blending_mode='add_sub', opacity=0.5)
    blend_2 = blend(img_in, blend_1, blending_mode='switch', opacity=1.0 if keying_type=='rgb' else 0.0)

    uniform_color_1 = uniform_color(mode='color', num_imgs=img_in.shape[0], res_h=img_in.shape[2], res_w=img_in.shape[3], use_alpha=use_alpha, rgba=rgb+[1.0])
    grayscale_2 = c2g(uniform_color_1, flatten_alpha=False, rgba_weights = [0.299, 0.587, 0.114, 0.0], bg=1.0)
    gm_2 = append_alpha((1.0-grayscale_2).expand(-1,3,-1,-1)) if use_alpha else (1.0-grayscale_2).expand(-1,3,-1,-1)
    blend_3 = blend(gm_2, uniform_color_1, blending_mode='add_sub', opacity=0.5)
    blend_4 = blend(uniform_color_1, blend_3, blending_mode='switch', opacity=1.0 if keying_type=='rgb' else 0.0)
    blend_5 = blend(blend_4, blend_2, blending_mode='subtract', opacity=1.0)
    blend_6 = blend(blend_2, blend_4, blending_mode='subtract', opacity=1.0)
    blend_7 = blend(blend_5, blend_6, blending_mode='max', opacity=1.0)
    grayscale_3 = c2g(blend_7, flatten_alpha=False, rgba_weights=[1.0,1.0,1.0,0.0], bg=1.0)

    blend_8 = blend(grayscale_2, grayscale_1, blending_mode='subtract', opacity=1.0)
    blend_9 = blend(grayscale_1, grayscale_2, blending_mode='subtract', opacity=1.0)
    blend_10 = blend(blend_8, blend_9, blending_mode='max', opacity=1.0)

    blend_11 = blend(grayscale_3, blend_10, blending_mode='switch', opacity=0.0 if keying_type=='luminance' else 1.0)
    img_out = levels(blend_11, in_low=(1-mask_softness)*0.25*th.max(to_tensor(mask_range), to_tensor(0.0005)), in_mid=0.5, in_high=0.25*th.max(to_tensor(mask_range), to_tensor(0.0005)), out_low=1.0, out_high=0.0)
    return img_out

# Finalized
@input_check(1)
def c2g_advanced(img_in, grayscale_type="desaturation"):
    """Non-atomic function: Grayscale Conversion Advanced (https://docs.substance3d.com/sddoc/grayscale-conversion-advanced-159449191.html)

    Args:
        img_in (tensor): Input image (RGB(A) only).
        grayscale_type (str, optional): Grayscale conversion type. Defaults to "desaturation", other options luma|average|max|min

    Returns:
        Tensor: Grayscale converted image.
    """

    color_input_check(img_in, "img_in")
    
    hsl_1 = hsl(img_in, 0.5, 0.0, 0.5)
    grayscale_1 = c2g(hsl_1, flatten_alpha=False, rgba_weights=[1.0/3.0, 1.0/3.0, 1.0/3.0, 0.0], bg=1.0)
    grayscale_2 = c2g(img_in, flatten_alpha=False, rgba_weights=[0.299, 0.587, 0.114, 0.0], bg=1.0)
    grayscale_3 = c2g(img_in, flatten_alpha=False, rgba_weights=[1.0/3.0, 1.0/3.0, 1.0/3.0, 0.0], bg=1.0)
    grayscale_4 = img_in[:,0,:,:]
    grayscale_5 = img_in[:,1,:,:]
    grayscale_6 = img_in[:,2,:,:]

    blend_1 = blend(grayscale_4, grayscale_5, blending_mode='max', opacity=1.0)
    blend_2 = blend(grayscale_4, grayscale_5, blending_mode='min', opacity=1.0)
    blend_3 = blend(blend_1, grayscale_6, blending_mode='max', opacity=1.0)
    blend_4 = blend(blend_2, grayscale_6, blending_mode='min', opacity=1.0)

    if grayscale_type == "desaturation":
        return grayscale_1
    elif grayscale_type == "luma":
        return grayscale_2
    elif grayscale_type == "average":
        return grayscale_3
    elif grayscale_type == "max":
        return blend_3
    elif grayscale_type == "min":
        return blend_4
    
# ---------------------------------------------------------------------------- #
#          Mathematical functions used in the implementation of nodes.         #
# ---------------------------------------------------------------------------- #

def lerp(start, end, weight):
    """Linear interpolation function.

    Args:
        start (float): start value
        end (float): end value
        weight (float): combination weight

    Returns:
        Float: interpolated value
    """             
    assert weight >= 0.0 and weight <= 1.0, 'weight should be in [0,1]'
    return start + (end-start) * weight

def cross_2d(v1, v2):
    """2D cross product function.

    Args:
        v1 (tensor): the first vector (or an array of vectors)
        v2 (tensor): the second vector

    Returns:
        Float: cross product of v1 and v2
    """
    assert v1.shape[-1] == 2 and v2.shape == (2,), 'v1 and v2 should both be 2D vectors'
    v1r = v1.reshape(-1, 2)
    ret = v1r[:, 0] * v2[1] - v1r[:, 1] * v2[0]
    ret = ret.reshape(v1.shape[:-1])
    return ret

def solve_poly_2d(a, b, c):
    """Solve quadratic equations (ax^2 + bx + c = 0).

    Args:
        a (tensor): 2D array of value a's (M x N)
        b (tensor): 2D array of value b's (M x N)
        c (tensor): 2D array of value c's (M x N)

    Returns:
        Tensor: the first solutions of the equations
        Tensor: the second solutions of the equations
        Tensor: error flag when equations are invalid
    """
    delta = b * b - 4 * a * c
    error = delta < 0

    # Return solutions
    sqrt_delta = th.sqrt(delta)
    x_quad_1, x_quad_2 = (sqrt_delta - b) * 0.5 / a, (-sqrt_delta - b) * 0.5 / a
    x_linear = -c / b
    cond = (a == 0) | error
    x1 = th.where(cond, x_linear, x_quad_1)
    x2 = th.where(cond, x_linear, x_quad_2)
    return x1, x2, error

def rgb2hsl(rgb):
    """RGB to HSL.

    Args:
        rgb (tensor): rgb value

    Returns:
        Tensor: hsl value
    """
    rgb = to_tensor(rgb)
    r = rgb[0]
    g = rgb[1]
    b = rgb[2]

    # compute s,v
    max_vals = th.max(rgb)
    min_vals = th.min(rgb)
    delta = max_vals - min_vals
    l = (max_vals + min_vals) / 2.0
    if delta == 0:
        s = to_tensor(0.0)
    else:
        s = delta / (1.0 - th.abs(2*l - 1.0))
    h = th.zeros_like(s)
    
    # compute h
    if delta == 0:
        h = to_tensor(0.0)
    elif rgb[0] == max_vals:
        h = th.remainder((g-b)/delta, 6.0) / 6.0
    elif rgb[1] == max_vals:
        h = ((b-r)/delta + 2.0) / 6.0
    elif rgb[2] == max_vals:
        h = ((r-g)/delta + 4.0) / 6.0

    return th.stack([h,s,l])

def ellipse(samples, sample_number, radius, ellipse_factor, rotation, inner_rotation, center_x):
    """Ellipse sampling function. (No detailed explanation in SBS. Parameter descriptions are speculated.)

    Args:
        samples (int): total number of sampling cones
        sample_number (int): index of the current sampling cone
        radius (float): sampling radius
        ellipse_factor (float): controls the aspect ratio of the ellipse. A zero value indicates a full circle.
        rotation (float): global rotation angle (in turning number)
        inner_rotation (float): inner rotation angle in the sampling cone
        center_x (float): X coordinate of the ellipse center

    Returns:
        tensor: orientation of the ellipse
    """
    radius = to_tensor(radius)
    ###
    angle_1 = (to_tensor(sample_number) / to_tensor(samples) + to_tensor(inner_rotation)) * np.pi * 2.0
    angle_2 = -to_tensor(rotation) * np.pi * 2.0
    sin_1, cos_1 = th.sin(angle_1), th.cos(angle_1)
    sin_2, cos_2 = th.sin(angle_2), th.cos(angle_2)
    factor_1 = (1.0 - to_tensor(ellipse_factor)) * sin_1
    factor_2 = lerp(cos_1, th.abs(cos_1), th.max(to_tensor(center_x) * 0.5, to_tensor(0.0)))
    # assemble results
    res_x = radius * (factor_1 * sin_2 + factor_2 * cos_2)
    res_y = radius * (factor_1 * cos_2 - factor_2 * sin_2)
    return th.cat((res_x.unsqueeze(0), res_y.unsqueeze(0)))

def hbao_radius(min_size_log2, mip_level, radius):
    """'hbao_radius_function' in sbs, used in HBAO. (Parameter descriptions are speculated.)

    Args:
        min_size_log2 (int): minimum size of the sampled image
        mip_level (int): mipmap level
        radius (float): sampling radius

    Returns:
        Float: scaled sampling radius
    """
    min_size_log2 = to_tensor(min_size_log2)
    mip_level = to_tensor(mip_level)
    radius = to_tensor(radius) * 2.0 ** (min_size_log2 - mip_level) - 1
    return th.clamp(radius, 0.0, 1.0)

# ------------------------------------------------------------------------------------ #
#          Parameter adjustment functions used in the implementation of nodes.         #
# ------------------------------------------------------------------------------------ #

def resize_color(color, num_channels):
    """Resize color to a specified number of channels.

    Args:
        color (tensor): input color
        num_channels (int): target number of channels

    Raises:
        ValueError: Resizing failed due to channel mismatch.

    Returns:
        Tensor: resized color
    """
    assert color.ndim == 1
    assert num_channels >= 1 and num_channels <= 4

    # Match background color with image channels
    if len(color) > num_channels:
        color = color[:num_channels]
    elif num_channels == 4 and len(color) == 3:
        color = th.cat((color, th.ones(1)))
    elif len(color) == 1:
        color = color.repeat(num_channels)
    elif len(color) != num_channels:
        raise ValueError('Channel mismatch between input image and background color')

    return color

# ---------------------------------------------------------------------------------- #
#          Image manipulation functions used in the implementation of nodes.         #
# ---------------------------------------------------------------------------------- #

def get_pos(res_h, res_w):
    """Get the $pos matrix of an input image (the center coordinates of each pixel).

    Args:
        res_h (int): input image height
        res_w (int): input image width

    Returns:
        Tensor: $pos matrix (size: (H, W, 2))
    """
    row_grid, col_grid = th.meshgrid(th.linspace(0.5, res_h - 0.5, res_h) / res_h,
                                     th.linspace(0.5, res_w - 0.5, res_w) / res_w)
    return th.stack((col_grid, row_grid), dim=2)

def create_mipmaps(img_in, mipmaps_level, keep_size=False):
    """Create mipmap levels for an input image using box filtering.

    Args:
        img_in (tensor): input image
        mipmaps_level (int): number of mipmap levels
        keep_size (bool, optional): switch for restoring the original image size after downsampling. Defaults to False.

    Returns:
        List[Tensor]: mipmap stack
    """
    mipmaps = []
    img_mm = img_in
    last_shape = img_in.shape[2]
    for i in range(mipmaps_level):
        img_mm = manual_resize(img_mm, -1) if img_mm.shape[2] > 1 else img_mm
        mipmaps.append(img_mm if not keep_size else \
                       mipmaps[-1] if last_shape == 1 else \
                       img_mm.expand_as(img_in) if last_shape == 2 else \
                       automatic_resize(img_mm, i + 1))
        last_shape = img_mm.shape[2]
    return mipmaps

def frequency_transform(img_in, normal_format='dx'):
    """Calculate convolution at multiple frequency levels.

    Args:
        img_in (tensor): input image
        normal_format (str, optional): switch for inverting the vertical 1-D convolution direction ('dx'|'gl'). Defaults to 'dx'.

    Returns:
        List[List[Tensor]]: list of convoluted images (in X and Y direction respectively)
    """
    in_size = img_in.shape[2]
    in_size_log2 = int(np.log2(in_size))

    # Create mipmap levels for R and G channels
    img_in = img_in[:, :2, :, :]
    mm_list = [img_in]
    if in_size_log2 > 4:
        mm_list.extend(create_mipmaps(img_in, in_size_log2 - 4))

    # Define convolution operators
    def conv_x(img):
        img_bw = th.clamp(img - roll_col(img, -1), -0.5, 0.5)
        img_fw = th.clamp(roll_col(img, 1) - img, -0.5, 0.5)
        return (img_fw + img_bw) * 0.5 + 0.5

    def conv_y(img):
        dr = -1 if normal_format == 'dx' else 1
        img_bw = th.clamp(img - roll_row(img, dr), -0.5, 0.5)
        img_fw = th.clamp(roll_row(img, -dr) - img, -0.5, 0.5)
        return (img_fw + img_bw) * 0.5 + 0.5

    conv_ops = [conv_x, conv_y]

    # Init blended images
    img_freqs = [[], []]

    # Low frequencies (for 16x16 images only)
    img_4 = mm_list[-1]
    img_4_scale = [None, None, None, img_4]
    for i in range(3):
        img_4_scale[i] = transform_2d(img_4, x1=to_zero_one(2.0 ** (3 - i)), y2=to_zero_one(2.0 ** (3 - i)))
    for i, scale in enumerate([8.0, 4.0, 2.0, 1.0]):
        for c in (0, 1):
            img_4_c = conv_ops[c](img_4_scale[i][:, [c], :, :])
            if scale > 1.0:
                img_4_c = transform_2d(img_4_c, mipmap_mode='manual', x1=to_zero_one(1.0 / scale), y2=to_zero_one(1.0 / scale))
            img_freqs[c].append(img_4_c)

    # Other frequencies
    for i in range(len(mm_list) - 1):
        for c in (0, 1):
            img_i_c = conv_ops[c](mm_list[-2 - i][:, [c], :, :])
            img_freqs[c].append(img_i_c)

    return img_freqs

def automatic_resize(img_in, scale_log2, filtering='bilinear'):
    """Progressively resize an input image.

    Args:
        img_in (tensor): input image
        scale_log2 (int): size change relative to the input resolution (after log2)
        filtering (str, optional): 'bilinear' or 'nearest'. Defaults to 'bilinear'.

    Returns:
        Tensor: resized image
    """
    # Check input validity
    assert filtering in ('bilinear', 'nearest')
    in_size_log2 = int(np.log2(img_in.shape[2]))
    out_size_log2 = max(in_size_log2 + scale_log2, 0)

    # Equal size
    if out_size_log2 == in_size_log2:
        img_out = img_in
    # Down-sampling (regardless of filtering)
    elif out_size_log2 < in_size_log2:
        img_out = img_in
        for _ in range(in_size_log2 - out_size_log2):
            img_out = manual_resize(img_out, -1)
    # Up-sampling (progressive bilinear filtering)
    elif filtering == 'bilinear':
        img_out = img_in
        for _ in range(scale_log2):
            img_out = manual_resize(img_out, 1)
    # Up-sampling (nearest sampling)
    else:
        img_out = manual_resize(img_in, scale_log2, filtering)

    return img_out

def manual_resize(img_in, scale_log2, filtering='bilinear'):
    """Manually resize an input image (all-in-one sampling).

    Args:
        img_in (tensor): input image
        scale_log2 (int): size change relative to input (after log2).
        filtering (str, optional): 'bilinear' or 'nearest'. Defaults to 'bilinear'.

    Returns:
        Tensor: resized image
    """
    # Check input validity
    assert filtering in ('bilinear', 'nearest')
    in_size = img_in.shape[2]
    in_size_log2 = int(np.log2(in_size))
    out_size_log2 = max(in_size_log2 + scale_log2, 0)
    out_size = 1 << out_size_log2

    # Equal size
    if out_size_log2 == in_size_log2:
        img_out = img_in
    else:
        row_grid, col_grid = th.meshgrid(th.linspace(1, out_size * 2 - 1, out_size), th.linspace(1, out_size * 2 - 1, out_size))
        sample_grid = th.stack([col_grid, row_grid], 2).expand(img_in.shape[0], out_size, out_size, 2)
        sample_grid = sample_grid / (out_size * 2) * 2.0 - 1.0
        # Down-sampling
        if out_size_log2 < in_size_log2:
            img_out = th.nn.functional.grid_sample(img_in, sample_grid, filtering, 'zeros', align_corners=False)
        # Up-sampling
        else:
            sample_grid = sample_grid * in_size / (in_size + 2)
            img_in_pad = th.nn.functional.pad(img_in, (1, 1, 1, 1), mode='circular')
            img_out = th.nn.functional.grid_sample(img_in_pad, sample_grid, filtering, 'zeros', align_corners=False)

    return img_out
        
def append_alpha(img_in, uniform_alpha_val=1.0):
    """append alpha channel with given alpha value

    Args:
        img_in (tensor): input rgb image
        uniform_alpha_val (float, optional): alpha value. Defaults to 1.0.

    Returns:
        Tensor: alpha channel appended image
    """    
    color_input_check(img_in, 'img_in')
    alpha_channel = th.ones(img_in.shape[0],1,img_in.shape[2],img_in.shape[3]) * uniform_alpha_val
    img_out = th.cat([img_in, alpha_channel], axis=1)
    return img_out

def display_img(img_in):
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

    
