#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh, SH2RGB
from utils.graphics_utils import fov2focal
import random


def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color, scaling_modifier = 1.0, black_video = False,
           override_color = None, sh_deg_aug_ratio = 0.1, bg_aug_ratio = 0.3, shs_aug_ratio=1.0, scale_aug_ratio=1.0, test = False, act_SH = 0,iteration = 0 ):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    bg_color = bg_color
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    try:
        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform,
            sh_degree=act_SH,
            campos=viewpoint_camera.camera_center,
            prefiltered=False
        )
    except TypeError as e:
        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform,
            sh_degree=act_SH,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            debug=False
        )


    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        # print(scales)
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if colors_precomp is None:
        if pipe.convert_SHs_python:
            raw_rgb = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2).squeeze()[:,:3]
            rgb = torch.sigmoid(raw_rgb)
            colors_precomp = rgb
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    if random.random() < shs_aug_ratio and not test:
        variance = (0.2 ** 0.5) * shs
        shs = shs + (torch.randn_like(shs) * variance)
    # add noise to scales
    if random.random() < scale_aug_ratio and not test:
        variance = (0.2 ** 0.5) * scales / 4
        scales = torch.clamp(scales + (torch.randn_like(scales) * variance), 0.0)
        # scales = scales

    rendered_image, radii, depth_alpha = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)
    depth, alpha = torch.chunk(depth_alpha, 2)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    if iteration>700 and iteration<0:
        focal = 1 / (2 * math.tan(viewpoint_camera.FoVx / 2))
        disp = focal / (depth + (alpha * 10) + 1e-5)
        try:
            min_d = disp[alpha <= 0.1].min()
        except Exception:
            min_d = disp.min()
        disp1 = disp.clone()
        disp1[disp1 > 0.1] += 0.05

        disp = torch.clamp((disp1 - min_d) / (disp.max() - min_d), 0.0, 1.0)
    else:
        focal = 1 / (2 * math.tan(viewpoint_camera.FoVx / 2))
        disp = focal / (depth + (alpha * 10) + 1e-5)

        try:
            min_d = disp[alpha <= 0.1].min()
        except Exception:
            min_d = disp.min()

        disp = torch.clamp((disp - min_d) / (disp.max() - min_d), 0.0, 1.0)

    return {"render": rendered_image,
            "depth": disp,
            "alpha": alpha,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "scales": scales}

