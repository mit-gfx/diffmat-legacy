import os
import sys
import time
import math
import numpy as np
import torch as th
import scipy.ndimage as spi
import matplotlib.pyplot as plt

from diffmat.sbs_core.util import input_check, roll_row, roll_col, normalize, color_input_check, grayscale_input_check, to_zero_one, to_tensor

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