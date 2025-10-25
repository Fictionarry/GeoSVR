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

# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import sys
import json
import time
import uuid
import imageio
import datetime
import numpy as np
from tqdm import tqdm
from random import randint
import random
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
import torchvision

from src.config import cfg, update_argparser, update_config

from src.utils.system_utils import seed_everything
from src.utils.image_utils import im_tensor2np, viz_tensordepth
from src.utils.bounding_utils import decide_main_bounding
from src.utils import mono_utils
from src.utils import loss_utils
from src.utils import depth_utils
from src.utils.graphics_utils import patch_offsets, patch_warp
from src.utils.graphics_utils import normal_from_depth_image, render_normal_func

from src.dataloader.data_pack import DataPack, compute_iter_idx
from src.sparse_voxel_model import SparseVoxelModel

import svraster_cuda

def training(args):
    # Init and load data pack
    data_pack = DataPack(cfg.data, cfg.model.white_background)

    # Instantiate data loader
    tr_cams = data_pack.get_train_cameras()
    tr_cam_indices = compute_iter_idx(len(tr_cams), cfg.procedure.n_iter)

    if cfg.auto_exposure.enable:
        for cam in tr_cams:
            cam.auto_exposure_init()

    # Prepare monocular depth priors if instructed
    if cfg.regularizer.lambda_depthanythingv2:
        mono_utils.prepare_depthanythingv2(
            cameras=tr_cams,
            source_path=cfg.data.source_path,
            force_rerun=False)
        mono_utils.prepare_depthanythingv2(
            cameras=data_pack.get_test_cameras(),
            source_path=cfg.data.source_path,
            force_rerun=False)

    if cfg.regularizer.lambda_mast3r_metric_depth:
        mono_utils.prepare_mast3r_metric_depth(
            cameras=tr_cams,
            source_path=cfg.data.source_path,
            mast3r_repo_path=cfg.regularizer.mast3r_repo_path)

    # Decide main (inside) region bounding box
    bounding = decide_main_bounding(
        cfg_bounding=cfg.bounding,
        tr_cams=tr_cams,
        pcd=data_pack.point_cloud,  # Not used
        suggested_bounding=data_pack.suggested_bounding,  # Can be None
    )

    # Init voxel model
    voxel_model = SparseVoxelModel(cfg.model)

    if args.load_iteration:
        loaded_iter = voxel_model.load_iteration(args.load_iteration)
    else:
        loaded_iter = None
        voxel_model.model_init(
            bounding=bounding,
            cfg_init=cfg.init,
            cameras=tr_cams)

    first_iter = loaded_iter if loaded_iter else 1
    print(f"Start optmization from iters={first_iter}.")

    # Init optimizer
    voxel_model.optimizer_init(cfg.optimizer)
    if loaded_iter and args.load_optimizer:
        voxel_model.optimizer_load_iteration(loaded_iter)

    # Init lr warmup scheduler
    if first_iter <= cfg.optimizer.n_warmup:
        rate = max(first_iter - 1, 0) / cfg.optimizer.n_warmup
        for param_group in voxel_model.optimizer.param_groups:
            param_group["base_lr"] = param_group["lr"]
            param_group["lr"] = rate * param_group["base_lr"]

    # Init subdiv
    remain_subdiv_times = sum(
        (i >= first_iter)
        for i in range(
            cfg.procedure.subdivide_from, cfg.procedure.subdivide_until+1,
            cfg.procedure.subdivide_every
        )
    )
    subdivide_scale = cfg.procedure.subdivide_target_scale ** (1 / remain_subdiv_times)
    subdivide_prop = max(0, (subdivide_scale - 1) / 7)
    print(f"Subdiv: times={remain_subdiv_times:2d} scale-each-time={subdivide_scale*100:.1f}% prop={subdivide_prop*100:.1f}%")

    # Some other initialization
    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)
    elapsed = 0

    tr_render_opt = {
        'track_max_w': True,
        'lambda_R_concen': cfg.regularizer.lambda_R_concen,
        'output_T': False,
        'output_depth': False,
        'ss': 1.0,  # disable supersampling at first
        'rand_bg': cfg.regularizer.rand_bg,
        'use_auto_exposure': cfg.auto_exposure.enable,
    }

    sparse_depth_loss = loss_utils.SparseDepthLoss(
        iter_end=cfg.regularizer.sparse_depth_until)
    
    if cfg.regularizer.enable_da2_uncertainty:
        depthanythingv2_loss = loss_utils.DepthAnythingv2UncertaintyLoss(
            iter_from=cfg.regularizer.depthanythingv2_from,
            iter_end=cfg.regularizer.depthanythingv2_end,
            end_mult=cfg.regularizer.depthanythingv2_end_mult,
            overall=cfg.regularizer.depthanythingv2_overall,
            power_level_uncertainty=cfg.regularizer.power_level_uncertainty,
            alpha_adjust=cfg.regularizer.depthanythingv2_alpha_adjust)
    else:
        depthanythingv2_loss = loss_utils.DepthAnythingv2Loss_old(
            iter_from=cfg.regularizer.depthanythingv2_from,
            iter_end=cfg.regularizer.depthanythingv2_end,
            end_mult=cfg.regularizer.depthanythingv2_end_mult,
            alpha_adjust=cfg.regularizer.depthanythingv2_alpha_adjust)
    
    mast3r_metric_depth_loss = loss_utils.Mast3rMetricDepthLoss(
        iter_from=cfg.regularizer.mast3r_metric_depth_from,
        iter_end=cfg.regularizer.mast3r_metric_depth_end,
        end_mult=cfg.regularizer.mast3r_metric_depth_end_mult)
    nd_loss = loss_utils.NormalDepthConsistencyLoss(
        iter_from=cfg.regularizer.n_dmean_from,
        iter_end=cfg.regularizer.n_dmean_end,
        ks=cfg.regularizer.n_dmean_ks,
        tol_deg=cfg.regularizer.n_dmean_tol_deg)
    nmed_loss = loss_utils.NormalMedianConsistencyLoss(
        iter_from=cfg.regularizer.n_dmed_from,
        iter_end=cfg.regularizer.n_dmed_end)
    
    nmono_loss = loss_utils.NormalMonoConsistencyLoss(
        iter_from=cfg.regularizer.n_dmono_from,
        iter_end=cfg.regularizer.n_dmono_end)

    ema_loss_for_log = 0.0
    ema_psnr_for_log = 0.0
    ema_ncc_for_log = 0.0
    ema_pixel_for_log = 0.0
    ema_photo_for_log = 0.0
    ema_depth_for_log = 0.0
    iter_rng = range(first_iter, cfg.procedure.n_iter+1)
    progress_bar = tqdm(iter_rng, desc="Training", ascii=True)
    for iteration in iter_rng:

        # Start processing time tracking of this iteration
        iter_start.record()

        # Increase the degree of SH by one up to a maximum degree
        if iteration % 1000 == 0:
            voxel_model.sh_degree_add1()

        # Recompute sh from cameras
        if iteration in cfg.procedure.reset_sh_ckpt:
            print("Reset sh0 from cameras.")
            print("Reset shs to zero.")
            voxel_model.reset_sh_from_cameras(tr_cams)
            torch.cuda.empty_cache()

        # Use default super-sampling option
        if iteration > 1000:
            if cfg.regularizer.ss_aug_max > 1:
                tr_render_opt['ss'] = np.random.uniform(1, cfg.regularizer.ss_aug_max)
            elif 'ss' in tr_render_opt:
                tr_render_opt.pop('ss')  # Use default ss

        need_sparse_depth = cfg.regularizer.lambda_sparse_depth > 0 and sparse_depth_loss.is_active(iteration)
        need_depthanythingv2 = cfg.regularizer.lambda_depthanythingv2 > 0 and depthanythingv2_loss.is_active(iteration)
        need_mast3r_metric_depth = cfg.regularizer.lambda_mast3r_metric_depth > 0 and mast3r_metric_depth_loss.is_active(iteration)
        need_nd_loss = cfg.regularizer.lambda_normal_dmean > 0 and nd_loss.is_active(iteration)
        need_nmed_loss = cfg.regularizer.lambda_normal_dmed > 0 and nmed_loss.is_active(iteration)
        need_nmono_loss = cfg.regularizer.lambda_normal_dmono > 0 and nmono_loss.is_active(iteration)
        need_level_uncertainty = cfg.regularizer.power_level_uncertainty > 0 and iteration > cfg.regularizer.level_uncertainty_from
        tr_render_opt['output_T'] = cfg.regularizer.lambda_T_concen > 0 or cfg.regularizer.lambda_T_inside > 0 or cfg.regularizer.lambda_mask > 0 or need_sparse_depth or need_nd_loss or need_depthanythingv2 or need_mast3r_metric_depth or need_level_uncertainty
        tr_render_opt['output_normal'] = need_nd_loss or need_nmed_loss or iteration > cfg.regularizer.multi_view_weight_from_iter
        tr_render_opt['output_depth'] = need_sparse_depth or need_nd_loss or need_nmed_loss or need_depthanythingv2 or need_mast3r_metric_depth or iteration > cfg.regularizer.multi_view_weight_from_iter
        # tr_render_opt['output_depth'] = True

        if iteration >= cfg.regularizer.dist_from and cfg.regularizer.lambda_dist:
            tr_render_opt['lambda_dist'] = cfg.regularizer.lambda_dist

        if iteration >= cfg.regularizer.ascending_from and cfg.regularizer.lambda_ascending:
            tr_render_opt['lambda_ascending'] = cfg.regularizer.lambda_ascending
        if iteration >= cfg.regularizer.rectifiy_from and cfg.regularizer.lambda_rectify:
            # Lazy implementation. Change to rectify mode.
            tr_render_opt['lambda_ascending'] = 0
            
        if iteration > cfg.regularizer.scaling_penalty_from and cfg.regularizer.lambda_scaling_penalty:
            tr_render_opt['lambda_scaling_penalty'] = cfg.regularizer.lambda_scaling_penalty
            tr_render_opt['min_voxel_size'] = voxel_model.vox_size.min()
        if iteration > cfg.regularizer.scaling_penalty_end:
            tr_render_opt['lambda_scaling_penalty'] = 0

        # Update auto exposure
        if cfg.auto_exposure.enable and iteration in cfg.auto_exposure.auto_exposure_upd_ckpt:
            for cam in tr_cams:
                with torch.no_grad():
                    ref = voxel_model.render(cam, ss=1.0)['color']
                cam.auto_exposure_update(ref, cam.image.cuda())

        # Pick a Camera
        cam = tr_cams[tr_cam_indices[iteration-1]]

        # Get gt image
        gt_image = cam.image.cuda()
        
        if cfg.regularizer.lambda_R_concen > 0:
            tr_render_opt['gt_color'] = gt_image
        
        if 'vox_feats' in tr_render_opt:
            tr_render_opt.pop("vox_feats")

        # Render
        render_pkg = voxel_model.render(cam, **tr_render_opt)
        render_image = render_pkg['color']
        
        # Loss
        mse = loss_utils.l2_loss(render_image, gt_image)

        if cfg.regularizer.use_l1:
            photo_loss = loss_utils.l1_loss(render_image, gt_image)
        elif cfg.regularizer.use_huber:
            photo_loss = loss_utils.huber_loss(render_image, gt_image, cfg.regularizer.huber_thres)
        else:
            photo_loss = mse
        loss = cfg.regularizer.lambda_photo * photo_loss # * (1 if iteration < cfg.regularizer.multi_view_weight_from_iter else 0.1)
        

        if need_sparse_depth:
            loss += cfg.regularizer.lambda_sparse_depth * sparse_depth_loss(cam, render_pkg)

        if cfg.regularizer.lambda_mask:
            gt_T = 1 - cam.mask.cuda()
            loss += cfg.regularizer.lambda_mask * loss_utils.l2_loss(render_pkg['T'], gt_T)

        if need_mast3r_metric_depth:
            loss += cfg.regularizer.lambda_mast3r_metric_depth * mast3r_metric_depth_loss(cam, render_pkg, iteration)

        if cfg.regularizer.lambda_ssim:
            loss += cfg.regularizer.lambda_ssim * loss_utils.fast_ssim_loss(render_image, gt_image)
        if cfg.regularizer.lambda_T_concen:
            loss += cfg.regularizer.lambda_T_concen * loss_utils.prob_concen_loss(render_pkg[f'raw_T'])
        if cfg.regularizer.lambda_T_inside:
            loss += cfg.regularizer.lambda_T_inside * render_pkg[f'raw_T'].square().mean()
        if need_nd_loss:
            loss += cfg.regularizer.lambda_normal_dmean * nd_loss(cam, render_pkg, iteration)
        if need_nmed_loss:
            loss += cfg.regularizer.lambda_normal_dmed * nmed_loss(cam, render_pkg, iteration)
            
            
        if iteration > cfg.regularizer.W_enlarge_from:
            max_w = render_pkg["max_w"]
            if max_w.max() > 0.5:
                # loss += 0.1 * (1 - max_w[max_w > 0.5]).mean()
                loss += 0.2 * torch.log((1 - max_w[max_w > 0.5]) + 1).mean()
        

        # Backward to get gradient of current iteration
        voxel_model.optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # Grid-level regularization
        grid_reg_interval = iteration >= cfg.regularizer.tv_from and iteration <= cfg.regularizer.tv_until
        if cfg.regularizer.lambda_tv_density and grid_reg_interval:
            lambda_tv_mult = cfg.regularizer.tv_decay_mult ** (iteration // cfg.regularizer.tv_decay_every)
            svraster_cuda.grid_loss_bw.total_variation(
                grid_pts=voxel_model._geo_grid_pts,
                vox_key=voxel_model.vox_key,
                weight=cfg.regularizer.lambda_tv_density * lambda_tv_mult,
                vox_size_inv=voxel_model.vox_size_inv,
                no_tv_s=True,
                tv_sparse=cfg.regularizer.tv_sparse,
                grid_pts_grad=voxel_model._geo_grid_pts.grad)

        # Optimizer step
        voxel_model.optimizer.step()

        # Learning rate warmup scheduler step
        if iteration <= cfg.optimizer.n_warmup:
            rate = iteration / cfg.optimizer.n_warmup
            for param_group in voxel_model.optimizer.param_groups:
                param_group["lr"] = rate * param_group["base_lr"]

        if iteration in cfg.optimizer.lr_decay_ckpt:
            for param_group in voxel_model.optimizer.param_groups:
                ori_lr = param_group["lr"]
                param_group["lr"] *= cfg.optimizer.lr_decay_mult
                print(f'LR decay of {param_group["name"]}: {ori_lr} => {param_group["lr"]}')

        ######################################################
        # Gradient statistic should happen before adaptive op
        ######################################################

        need_stat = (
            iteration >= 500 and \
            iteration <= cfg.procedure.subdivide_until)
        if need_stat:
            voxel_model.subdiv_meta += voxel_model._subdiv_p.grad

        ######################################################
        # Start adaptive voxels pruning and subdividing
        ######################################################

        need_pruning = (
            iteration % cfg.procedure.prune_every == 0 and \
            iteration >= cfg.procedure.prune_from and \
            iteration <= cfg.procedure.prune_until)
        need_subdividing = (
            iteration % cfg.procedure.subdivide_every == 0 and \
            iteration >= cfg.procedure.subdivide_from and \
            iteration <= cfg.procedure.subdivide_until and \
            voxel_model.num_voxels < cfg.procedure.subdivide_max_num)

        # Do nothing in last 500 iteration
        need_pruning &= (iteration <= cfg.procedure.n_iter-1500)
        need_subdividing &= (iteration <= cfg.procedure.n_iter-1500)

        if need_pruning or need_subdividing:
            stat_pkg = voxel_model.compute_training_stat(camera_lst=tr_cams)
            torch.cuda.empty_cache()
        
        if need_pruning:
            ori_n = voxel_model.num_voxels

            # Compute pruning threshold
            prune_all_iter = max(1, cfg.procedure.prune_until - cfg.procedure.prune_every)
            prune_now_iter = max(0, iteration - cfg.procedure.prune_every)
            prune_iter_rate = max(0, min(1, prune_now_iter / prune_all_iter))
            thres_inc = max(0, cfg.procedure.prune_thres_final - cfg.procedure.prune_thres_init)
            prune_thres = cfg.procedure.prune_thres_init + thres_inc * prune_iter_rate

            # Prune voxels
            prune_mask = (stat_pkg['max_w'] < prune_thres).squeeze(1)

            voxel_model.pruning(prune_mask)

            # Prune statistic (for the following subdivision)
            kept_idx = (~prune_mask).argwhere().squeeze(1)
            for k, v in stat_pkg.items():
                stat_pkg[k] = v[kept_idx]

            new_n = voxel_model.num_voxels
            print(f'[PRUNING]     {ori_n:7d} => {new_n:7d} (x{new_n/ori_n:.2f};  thres={prune_thres:.4f})')
            torch.cuda.empty_cache()

        if need_subdividing:
            # Exclude some voxels
            size_thres = stat_pkg['min_samp_interval'] * cfg.procedure.subdivide_samp_thres
            large_enough_mask = (voxel_model.vox_size * 0.5 > size_thres).squeeze(1)
            non_finest_mask = voxel_model.octlevel.squeeze(1) < svraster_cuda.meta.MAX_NUM_LEVELS
            valid_mask = large_enough_mask & non_finest_mask

            # Get some statistic for subdivision priority
            priority = voxel_model.subdiv_meta.squeeze(1) * valid_mask

            # Compute priority rank (larger value has higher priority)
            rank = torch.zeros_like(priority)
            rank[priority.argsort()] = torch.arange(len(priority), dtype=torch.float32, device="cuda")

            # Determine the number of voxels to subdivided
            if iteration <= cfg.procedure.subdivide_all_until:
                thres = -1
            else:
                thres = rank.quantile(1 - subdivide_prop)

            # Compute subdivision mask
            subdivide_mask = (rank > thres) & valid_mask

            # In case the number of voxels over the threshold
            max_n_subdiv = round((cfg.procedure.subdivide_max_num - voxel_model.num_voxels) / 7)
            if subdivide_mask.sum() > max_n_subdiv:
                n_removed = subdivide_mask.sum() - max_n_subdiv
                subdivide_mask &= (rank > rank[subdivide_mask].sort().values[n_removed-1])

            # Subdivision
            ori_n = voxel_model.num_voxels
            if subdivide_mask.sum() > 0:
                voxel_model.subdividing(subdivide_mask, cfg.procedure.subdivide_save_gpu)
            new_n = voxel_model.num_voxels
            in_p = voxel_model.inside_mask.float().mean().item()
            print(f'[SUBDIVIDING] {ori_n:7d} => {new_n:7d} (x{new_n/ori_n:.2f}; inside={in_p*100:.1f}%)')

            voxel_model.subdiv_meta.zero_()  # reset subdiv meta
            remain_subdiv_times -= 1
            torch.cuda.empty_cache()

        ######################################################
        # End of adaptive voxels procedure
        ######################################################
        
                
        ######################################################
        # Multi-view Regularize
        ######################################################
        
        # Render
        torch.cuda.empty_cache()
        
        if iteration >= cfg.regularizer.level_uncertainty_from and cfg.regularizer.power_level_uncertainty >= 0:
            tr_render_opt["vox_feats"] = voxel_model.octlevel.detach() * 1.0
        elif 'vox_feats' in tr_render_opt:
            tr_render_opt.pop("vox_feats")
        if iteration >= cfg.regularizer.rectifiy_from and cfg.regularizer.lambda_rectify:
            # Lazy implementation. Change to rectify mode.
            tr_render_opt['lambda_ascending'] = -1 * cfg.regularizer.lambda_rectify
            
        if iteration > 1000:
            if cfg.regularizer.voxel_dropout_min < 1:
                tr_render_opt['ss'] = np.random.uniform(cfg.regularizer.voxel_dropout_min, tr_render_opt['ss'])
                
        render_pkg = voxel_model.render(cam, **tr_render_opt)
        
        reg_loss = torch.tensor(0.0, requires_grad=True).cuda()
        
        depth_loss = torch.tensor(0)
        if need_depthanythingv2:
            depth_loss = depthanythingv2_loss(cam, render_pkg, iteration)
            reg_loss += cfg.regularizer.lambda_depthanythingv2 * depth_loss


        if need_nmono_loss:
            reg_loss += cfg.regularizer.lambda_normal_dmono * nmono_loss(cam, render_pkg, iteration)

        if iteration > cfg.regularizer.depth_smooth_from:
            reg_loss += 0.01 * loss_utils.loss_depth_smoothness(render_pkg['depth'][0][None, None, ...], render_pkg["color"][None, ...])
        
        
        ncc_loss = geo_loss = torch.tensor(0)
        if iteration > cfg.regularizer.multi_view_weight_from_iter and iteration % cfg.regularizer.multi_view_interval == 0:
        
            nearest_cam = None if len(cam.nearest_id) == 0 else tr_cams[random.sample(cam.nearest_id,1)[0]]
            
            if nearest_cam is not None:
                patch_size = cfg.regularizer.multi_view_patch_size
                patch_size *= 1 + cfg.regularizer.multi_view_anneal_scale * (1 - (iteration / cfg.procedure.n_iter))
                patch_size = int(patch_size)
                sample_num = cfg.regularizer.multi_view_sample_num
                pixel_noise_th = cfg.regularizer.multi_view_pixel_noise_th
                total_patch_size = (patch_size * 2 + 1) ** 2
                ncc_weight = cfg.regularizer.multi_view_ncc_weight
                geo_weight = cfg.regularizer.multi_view_geo_weight
                ## compute geometry consistency mask and loss
                # render_depth = render_pkg['depth'][0] / (1 - render_pkg['T'][0]).clamp(min=1e-3)
                render_depth = render_pkg['depth'][0]
                
                H, W = render_pkg['depth'][0].squeeze().shape                
                ix, iy = torch.meshgrid(
                    torch.arange(W), torch.arange(H), indexing='xy')
                pixels = torch.stack([ix, iy], dim=-1).float().to(render_depth.device)

                nearest_render_pkg = voxel_model.render(nearest_cam, **tr_render_opt)
                # nearest_render_depth = nearest_render_pkg['depth'][0] / (1 - nearest_render_pkg['T'][0]).clamp(min=1e-3)
                nearest_render_depth = nearest_render_pkg['depth'][0]
                

                pts = depth_utils.get_points_from_depth(cam, render_depth[None])
                pts_in_nearest_cam = pts @ nearest_cam.world_view_transform[:3,:3] + nearest_cam.world_view_transform[3,:3]
                pts_in_nearest_cam[:, 2] = torch.where(pts_in_nearest_cam[:, 2] <= 1e-3, 1e-3, pts_in_nearest_cam[:, 2])
                
                map_z, d_mask = depth_utils.get_points_depth_in_depth_map(nearest_cam, nearest_render_depth[None], pts_in_nearest_cam)
                
                pts_in_nearest_cam = pts_in_nearest_cam / (pts_in_nearest_cam[:,2:3])
                pts_in_nearest_cam = pts_in_nearest_cam * map_z.squeeze()[...,None]
                R = torch.tensor(nearest_cam.R).float().cuda()
                T = torch.tensor(nearest_cam.T).float().cuda()
                                
                pts_ = (pts_in_nearest_cam-T)@R.transpose(-1,-2)
                pts_in_view_cam = pts_ @ cam.world_view_transform[:3,:3] + cam.world_view_transform[3,:3]
                pts_in_view_cam[:, 2] = torch.where(pts_in_view_cam[:, 2] <= 1e-3, 1e-3, pts_in_view_cam[:, 2])
                
                pts_projections = torch.stack(
                            [pts_in_view_cam[:,0] * cam.Fx / pts_in_view_cam[:,2] + cam.Cx,
                            pts_in_view_cam[:,1] * cam.Fy / pts_in_view_cam[:,2] + cam.Cy], -1).float()
                pixel_noise = torch.norm(pts_projections - pixels.reshape(*pts_projections.shape), dim=-1)

                d_mask = d_mask & (pixel_noise < pixel_noise_th)
                weights = (1.0 / torch.exp(pixel_noise)).detach()
                weights[~d_mask] = 0

                if d_mask.sum() > 0:
                    geo_loss = ((weights * pixel_noise)[d_mask]).mean()
                    reg_loss += geo_weight * geo_loss

                    if ncc_weight > 0:
                        with torch.no_grad():
                            '''debug'''
                            # if iteration % 200 == 0:
                            #     sample_num = 1e9
                            #     d_mask_real = d_mask.clone()
                            #     d_mask = torch.ones_like(d_mask)
                            
                            ## sample mask
                            d_mask = d_mask.reshape(-1)
                            valid_indices = torch.arange(d_mask.shape[0], device=d_mask.device)[d_mask]
                            
                            if d_mask.sum() > sample_num:
                                index = np.random.choice(d_mask.sum().cpu().numpy(), sample_num, replace = False)
                                valid_indices = valid_indices[index]
                                
                            weights = weights.reshape(-1)[valid_indices]
                            ## sample ref frame patch
                            pixels = pixels.reshape(-1,2)[valid_indices]
                            offsets = patch_offsets(patch_size, pixels.device)
                            ori_pixels_patch = pixels.reshape(-1, 1, 2) / cam.ncc_scale + offsets.float()
                            
                            gt_image_gray = cam.get_ncc_gt().cuda()

                            H, W = gt_image_gray.squeeze().shape
                            pixels_patch = ori_pixels_patch.clone()
                            pixels_patch[:, :, 0] = 2 * pixels_patch[:, :, 0] / (W - 1) - 1.0
                            pixels_patch[:, :, 1] = 2 * pixels_patch[:, :, 1] / (H - 1) - 1.0
                            ref_gray_val = F.grid_sample(gt_image_gray.unsqueeze(1).contiguous(), pixels_patch.view(1, -1, 1, 2).contiguous(), align_corners=True)
                            ref_gray_val = ref_gray_val.reshape(-1, total_patch_size)
                            
                            ref_to_neareast_r = nearest_cam.world_view_transform[:3,:3].transpose(-1,-2) @ cam.world_view_transform[:3,:3]
                            ref_to_neareast_t = -ref_to_neareast_r @ cam.world_view_transform[3,:3] + nearest_cam.world_view_transform[3,:3]

                        ## compute Homography
                        '''norm w2c'''
                        render_normal = render_pkg["normal"].clone()
                        render_normal[0:3] = render_normal[0:3] * -1
                        cam_normal = (render_normal.reshape(3,-1).permute(1,0) @ cam.world_view_transform[:3,:3]).permute(1,0).reshape(render_normal.shape)
                        
                        
                        ref_local_n_temp = cam_normal.reshape(3,-1).permute(1,0)
                        ref_local_n_temp = ref_local_n_temp / ref_local_n_temp.norm(dim=-1, keepdim=True).clamp(min=1e-3).detach()
                        ref_local_n = ref_local_n_temp.reshape(-1,3)[valid_indices]

                        rays_d = cam.get_rays()
                        ref_local_d = render_depth.view(-1) * ((ref_local_n_temp * rays_d.reshape(-1,3)).sum(-1).abs())
                        ref_local_d = ref_local_d.reshape(*render_depth.shape).clamp(min=1e-6)
                        
                        ref_local_d = ref_local_d.reshape(-1)[valid_indices]

                        H_ref_to_neareast = ref_to_neareast_r[None] - \
                            torch.matmul(ref_to_neareast_t[None,:,None].expand(ref_local_d.shape[0],3,1), 
                                        ref_local_n[:,:,None].expand(ref_local_d.shape[0],3,1).permute(0, 2, 1))/ref_local_d[...,None,None]
                        H_ref_to_neareast = torch.matmul(nearest_cam.get_k(nearest_cam.ncc_scale)[None].expand(ref_local_d.shape[0], 3, 3), H_ref_to_neareast)
                        H_ref_to_neareast = H_ref_to_neareast @ cam.get_inv_k(cam.ncc_scale)
                        
                        
                        ## compute neareast frame patch
                        grid = patch_warp(H_ref_to_neareast.reshape(-1,3,3), ori_pixels_patch)
                        grid[:, :, 0] = 2 * grid[:, :, 0] / (W - 1) - 1.0
                        grid[:, :, 1] = 2 * grid[:, :, 1] / (H - 1) - 1.0
                        nearest_image_gray = nearest_cam.get_ncc_gt().cuda()
                        
                        sampled_gray_val = F.grid_sample(nearest_image_gray[None].contiguous(), grid.reshape(1, -1, 1, 2).contiguous(), align_corners=True)
                        sampled_gray_val = sampled_gray_val.reshape(-1, total_patch_size)
                        
                        
                        '''debug'''
                        # if iteration % 200 == 0:
                        #     outdir = os.path.join(voxel_model.model_path, "pg_view")
                        #     os.makedirs(outdir, exist_ok=True)
                            
                        #     ref_gray_val[~d_mask_real[..., None].repeat(1, total_patch_size)] = 0
                        #     sampled_gray_val[~d_mask_real[..., None].repeat(1, total_patch_size)] = 0
                            
                        #     torchvision.utils.save_image(ref_gray_val.reshape(cam.image_height, cam.image_width, total_patch_size)[..., -1], fp = os.path.join(voxel_model.model_path, "pg_view", f"iter{iteration:06d}_ref.jpg"))
                        #     torchvision.utils.save_image(sampled_gray_val.reshape(cam.image_height, cam.image_width, total_patch_size)[..., -1], fp = os.path.join(voxel_model.model_path, "pg_view", f"iter{iteration:06d}_sample.jpg"))
                        #     # torchvision.utils.save_image(nearest_image_gray, fp = os.path.join(voxel_model.model_path, "pg_view", f"iter{iteration:06d}_image_nearest.jpg"))
                        #     # torchvision.utils.save_image(gt_image_gray, fp = os.path.join(voxel_model.model_path, "pg_view", f"iter{iteration:06d}_image_gt.jpg"))
                            
                        #     ref_gray_val = ref_gray_val[d_mask_real]
                        #     sampled_gray_val = sampled_gray_val[d_mask_real]
                        #     weights = weights[d_mask_real]
                            
                        #     with torch.no_grad():
                        #         im = np.concatenate([
                        #             np.concatenate([
                        #                 im_tensor2np(render_pkg['color']),
                        #                 viz_tensordepth(render_pkg['depth'][0], 1 - render_pkg['T'][0]),
                        #                 # im_tensor2np(render_normal_func(cam, render_pkg['depth'][0].squeeze()) * 0.5 + 0.5),
                        #                 im_tensor2np(cam_normal * 0.5 + 0.5),
                        #             ], axis=1),
                        #             np.concatenate([
                        #                 im_tensor2np(nearest_render_pkg['color']),
                        #                 viz_tensordepth(nearest_render_pkg['depth'][0], 1 - nearest_render_pkg['T'][0]),
                        #                 im_tensor2np(render_normal_func(nearest_cam, nearest_render_pkg['depth'][0].squeeze()) * 0.5 + 0.5),
                        #             ], axis=1),
                        #         ], axis=0)
                        #         torch.cuda.empty_cache()
                        #         imageio.imwrite(os.path.join(outdir, f"iter{iteration:06d}_nearest_geo_{tr_render_opt['ss']:.2f}.jpg"), im)

                        
                        
                        ## compute loss
                        ncc, ncc_mask = loss_utils.lncc(ref_gray_val, sampled_gray_val)
                        mask = ncc_mask.reshape(-1)
                        ncc = ncc.reshape(-1) * weights
                        ncc = ncc[mask].squeeze()

                        if mask.sum() > 0:
                            ncc_loss = ncc.mean()
                            reg_loss += ncc_weight * ncc_loss         
        
        # Backward to get gradient of current iteration
        voxel_model.optimizer.zero_grad(set_to_none=True)
        # if iteration > cfg.regularizer.multi_view_weight_from_iter:
        #     torch.autograd.set_detect_anomaly(True)
        reg_loss.backward()

        # Optimizer step
        voxel_model.optimizer.step()        
        
        

        # End processing time tracking of this iteration
        iter_end.record()
        torch.cuda.synchronize()
        elapsed += iter_start.elapsed_time(iter_end)

        # Logging
        with torch.no_grad():
            # Metric
            loss = loss.item()
            psnr = -10 * np.log10(mse.item())

            # Progress bar
            # ema_p = max(0.01, 1 / (iteration - first_iter + 1))
            ema_loss_for_log = 0.6 * ema_loss_for_log + 0.4 * loss
            ema_psnr_for_log = 0.6 * ema_psnr_for_log + 0.4 * psnr
            ema_ncc_for_log = 0.6 * ema_ncc_for_log + 0.4 * ncc_loss.item()
            ema_pixel_for_log = 0.6 * ema_pixel_for_log + 0.4 * geo_loss.item()
            ema_photo_for_log = 0.6 * ema_photo_for_log + 0.4 * photo_loss.item()
            ema_depth_for_log = 0.6 * ema_depth_for_log + 0.4 * depth_loss.item()
            if iteration % 10 == 0:
                pb_text = {
                    "Loss": f"{ema_loss_for_log:.5f}",
                    "photo": f"{ema_photo_for_log:.4f}",
                    "pixel": f"{ema_pixel_for_log:.4f}",
                    "ncc": f"{ema_ncc_for_log:.4f}",
                    # "depth": f"{ema_depth_for_log:.5f}",
                    "psnr": f"{ema_psnr_for_log:.2f}",
                }
                progress_bar.set_postfix(pb_text)
                progress_bar.update(10)
            if iteration == cfg.procedure.n_iter:
                progress_bar.close()

            # Log and save
            training_report(
                data_pack=data_pack,
                voxel_model=voxel_model,
                iteration=iteration,
                loss=loss,
                psnr=psnr,
                elapsed=elapsed,
                ema_psnr=ema_psnr_for_log,
                pg_view_every=args.pg_view_every,
                test_iterations=args.test_iterations)

            if iteration in args.checkpoint_iterations or iteration == cfg.procedure.n_iter:
                voxel_model.save_iteration(iteration, quantize=args.save_quantized)
                if args.save_optimizer:
                    voxel_model.optimizer_save_iteration(iteration)
                print(f"[SAVE] path={voxel_model.latest_save_path}")


def training_report(data_pack, voxel_model, iteration, loss, psnr, elapsed, ema_psnr, pg_view_every, test_iterations):

    voxel_model.freeze_vox_geo()

    # Progress view
    if pg_view_every > 0 and (iteration % pg_view_every == 0 or iteration == 1):
        torch.cuda.empty_cache()
        test_cameras = data_pack.get_test_cameras()
        if len(test_cameras) == 0:
            test_cameras = data_pack.get_train_cameras()
        pg_idx = iteration // pg_view_every
        # pg_idx = 0
        # pg_idx = 131
        view = test_cameras[pg_idx % len(test_cameras)]
        
        tr_render_opt = {"vox_feats": voxel_model.octlevel * 1.0}
        
        render_pkg = voxel_model.render(view, output_depth=True, output_normal=True, output_T=True, **tr_render_opt)
        render_image = render_pkg['color']
        render_depth = render_pkg['depth'][0]
        render_depth_med = render_pkg['depth'][2]
        render_normal = render_pkg['normal']
        render_alpha = 1 - render_pkg['T'][0]
        
        render_normal_from_points = render_normal_func(view, render_depth.squeeze())
        render_normal_med_from_points = render_normal_func(view, render_depth_med.squeeze())
        mono = 1 / view.depthanythingv2.clamp(min=1e-4).cuda()
        if render_depth.shape[-2:] != mono.shape[-2:]:
            mono = torch.nn.functional.interpolate(
                mono[None, None], size=render_depth.shape[-2:], mode='bilinear').squeeze()
        render_normal_mono_from_points = render_normal_func(view, mono)
        
        render_vox_level = render_pkg['feat']/(1-render_pkg['T'][0]).clamp(min=0.1).squeeze().detach()
        level_weight = (render_vox_level.max()-render_vox_level.min())/(render_vox_level-render_vox_level.min()).clamp(min=1.0)
                
        im = np.concatenate([
            np.concatenate([
                im_tensor2np(render_image),
                im_tensor2np(render_alpha)[...,None].repeat(3, axis=-1),
                im_tensor2np(render_normal_mono_from_points * 0.5 + 0.5),
                im_tensor2np(level_weight / level_weight.max())[...,None].repeat(3, axis=-1),
            ], axis=1),
            np.concatenate([
                viz_tensordepth(render_depth, render_alpha),
                im_tensor2np(render_normal * 0.5 + 0.5),
                im_tensor2np(render_normal_from_points * 0.5 + 0.5),
                im_tensor2np(render_vox_level / render_vox_level.max())[...,None].repeat(3, axis=-1),
            ], axis=1),
            np.concatenate([
                im_tensor2np(view.depth2normal(render_depth) * 0.5 + 0.5),
                im_tensor2np(view.depth2normal(render_depth_med) * 0.5 + 0.5),
                viz_tensordepth(render_depth_med, render_alpha),
                im_tensor2np(render_vox_level / render_vox_level.max())[...,None].repeat(3, axis=-1),
            ], axis=1),
        ], axis=0)
        torch.cuda.empty_cache()

        outdir = os.path.join(voxel_model.model_path, "pg_view")
        outpath = os.path.join(outdir, f"iter{iteration:06d}.jpg")
        os.makedirs(outdir, exist_ok=True)

        imageio.imwrite(outpath, im)

        eps_file = os.path.join(voxel_model.model_path, "pg_view", "eps.txt")
        with open(eps_file, 'a') as f:
            f.write(f"{iteration},{elapsed/1000:.1f}\n")

    # Report test and samples of training set
    if iteration in test_iterations:
        print(f"[EVAL] running...")
        torch.cuda.empty_cache()
        test_cameras = data_pack.get_test_cameras()
        save_every = max(1, len(test_cameras) // 8)
        outdir = os.path.join(voxel_model.model_path, "test_view")
        os.makedirs(outdir, exist_ok=True)
        psnr_lst = []
        video = []
        max_w = torch.zeros([voxel_model.num_voxels, 1], dtype=torch.float32, device="cuda")
        for idx, camera in enumerate(test_cameras):
            render_pkg = voxel_model.render(camera, output_normal=True, track_max_w=True)
            render_image = render_pkg['color']
            im = im_tensor2np(render_image)
            gt = im_tensor2np(camera.image)
            video.append(im)
            if idx % save_every == 0:
                outpath = os.path.join(outdir, f"idx{idx:04d}_iter{iteration:06d}.jpg")
                cat = np.concatenate([gt, im], axis=1)
                imageio.imwrite(outpath, cat)

                outpath = os.path.join(outdir, f"idx{idx:04d}_iter{iteration:06d}_normal.jpg")
                render_normal = render_pkg['normal']
                render_normal = im_tensor2np(render_normal * 0.5 + 0.5)
                imageio.imwrite(outpath, render_normal)
            mse = np.square(im/255 - gt/255).mean()
            psnr_lst.append(-10 * np.log10(mse))
            max_w = torch.maximum(max_w, render_pkg['max_w'])
        avg_psnr = np.mean(psnr_lst)
        imageio.mimwrite(
            os.path.join(outdir, f"video_iter{iteration:06d}.mp4"),
            video, fps=30)
        torch.cuda.empty_cache()

        fps = time.time()
        for idx, camera in enumerate(test_cameras):
            voxel_model.render(camera, track_max_w=False)
        torch.cuda.synchronize()
        fps = len(test_cameras) / (time.time() - fps)
        torch.cuda.empty_cache()

        # Sample training views to render
        train_cameras = data_pack.get_train_cameras()
        for idx in range(0, len(train_cameras), max(1, len(train_cameras)//8)):
            camera = train_cameras[idx]
            render_pkg = voxel_model.render(
                camera, output_normal=True, track_max_w=True,
                use_auto_exposure=cfg.auto_exposure.enable)
            render_image = render_pkg['color']
            im = im_tensor2np(render_image)
            gt = im_tensor2np(camera.image)
            outpath = os.path.join(outdir, f"train_idx{idx:04d}_iter{iteration:06d}.jpg")
            cat = np.concatenate([gt, im], axis=1)
            imageio.imwrite(outpath, cat)

            outpath = os.path.join(outdir, f"train_idx{idx:04d}_iter{iteration:06d}_normal.jpg")
            render_normal = render_pkg['normal']
            render_normal = im_tensor2np(render_normal * 0.5 + 0.5)
            imageio.imwrite(outpath, render_normal)

        print(f"[EVAL] iter={iteration:6d}  psnr={avg_psnr:.2f}  fps={fps:.0f}")

        outdir = os.path.join(voxel_model.model_path, "test_stat")
        outpath = os.path.join(outdir, f"iter{iteration:06d}.json")
        os.makedirs(outdir, exist_ok=True)
        with open(outpath, 'w') as f:
            q = torch.linspace(0,1,5, device="cuda")
            max_w_q = max_w.quantile(q).tolist()
            peak_mem = torch.cuda.memory_stats()["allocated_bytes.all.peak"] / 1024 ** 3
            stat = {
                'psnr': avg_psnr,
                'ema_psnr': ema_psnr,
                'elapsed': elapsed,
                'fps': fps,
                'n_voxels': voxel_model.num_voxels,
                'max_w_q': max_w_q,
                'peak_mem': peak_mem,
            }
            json.dump(stat, f, indent=4)

    voxel_model.unfreeze_vox_geo()



if __name__ == "__main__":

    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser(
        description="Sparse voxels raster optimization."
        "You can specify a list of config files to overwrite the default setups."
        "All config fields can also be overwritten by command line.")
    parser.add_argument('--cfg_files', default=[], nargs='*')
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="*", type=int, default=[-1])
    parser.add_argument("--pg_view_every", type=int, default=200)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--load_iteration", type=int, default=None)
    parser.add_argument("--load_optimizer", action='store_true')
    parser.add_argument("--save_optimizer", action='store_true')
    parser.add_argument("--save_quantized", action='store_true')
    args, cmd_lst = parser.parse_known_args()

    # Update config from files and command line
    update_config(args.cfg_files, cmd_lst)

    # Global init
    seed_everything(cfg.procedure.seed)
    torch.cuda.set_device(torch.device("cuda:0"))
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    # Setup output folder and dump config
    if not cfg.model.model_path:
        datetime_str = datetime.datetime.now().strftime("%Y-%m%d-%H%M")
        unique_str = str(uuid.uuid4())[:6]
        folder_name = f"{datetime_str}-{unique_str}"
        cfg.model.model_path = os.path.join(f"./output", folder_name)

    os.makedirs(cfg.model.model_path, exist_ok=True)
    with open(os.path.join(cfg.model.model_path, "config.yaml"), "w") as f:
            f.write(cfg.dump())
    print(f"Output folder: {cfg.model.model_path}")

    # Apply scheduler scaling
    if cfg.procedure.sche_mult != 1:
        sche_mult = cfg.procedure.sche_mult

        # for key in ['geo_lr', 'sh0_lr', 'shs_lr']:
        #     cfg.optimizer[key] /= sche_mult
        cfg.optimizer.n_warmup = round(cfg.optimizer.n_warmup * sche_mult)
        cfg.optimizer.lr_decay_ckpt = [
            round(v * sche_mult) if v > 0 else v
            for v in cfg.optimizer.lr_decay_ckpt]

        for key in [
                'dist_from', 'tv_from', 'tv_until',
                'n_dmean_from', 'n_dmean_end',
                'n_dmed_from', 'n_dmed_end',
                'n_dmono_from', 'n_dmono_end',
                'multi_view_weight_from_iter',
                'depthanythingv2_from', 'depthanythingv2_end',
                'mast3r_metric_depth_from', 'mast3r_metric_depth_end']:
            cfg.regularizer[key] = round(cfg.regularizer[key] * sche_mult)

        for key in [
                'n_iter',
                'prune_from', 'prune_every', 'prune_until',
                'subdivide_from', 'subdivide_every', 'subdivide_until']:
            cfg.procedure[key] = round(cfg.procedure[key] * sche_mult)
        cfg.procedure.reset_sh_ckpt = [
            round(v * sche_mult) if v > 0 else v
            for v in cfg.procedure.reset_sh_ckpt]

    # Update negative iterations
    for i in range(len(args.test_iterations)):
        if args.test_iterations[i] < 0:
            args.test_iterations[i] += cfg.procedure.n_iter + 1
    for i in range(len(args.checkpoint_iterations)):
        if args.checkpoint_iterations[i] < 0:
            args.checkpoint_iterations[i] += cfg.procedure.n_iter + 1

    # Launch training loop
    training(args)
    print("Everything done.")
