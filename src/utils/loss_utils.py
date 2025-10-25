# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import torch
import numpy as np

from lpips import LPIPS
from pytorch_msssim import SSIM
from fused_ssim import fused_ssim
import torch.nn.functional as F
from math import exp
from torch.autograd import Variable


metric_networks = {}


def l1_loss(x, y):
    return torch.nn.functional.l1_loss(x, y)

def l2_loss(x, y):
    return torch.nn.functional.mse_loss(x, y)

def huber_loss(x, y, thres=0.01):
    l1 = (x - y).abs().mean(0)
    l2 = (x - y).pow(2).mean(0)
    loss = torch.where(
        l1 < thres,
        l2,
        2 * thres * l1 - thres ** 2)
    return loss.mean()

def cauchy_loss(x, y, reduction='mean'):
    loss_map = torch.log1p(torch.square(x - y))
    if reduction == 'sum':
        return loss_map.sum()
    if reduction == 'mean':
        return loss_map.mean()
    raise NotImplementedError

def psnr_score(x, y):
    return -10 * torch.log10(l2_loss(x, y))

def psnr(img1, img2):
    mse = (((img1 - img2)) ** 2).reshape(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def ssim_score(x, y):
    if 'SSIM' not in metric_networks:
        metric_networks['SSIM'] = SSIM(data_range=1, win_size=11, win_sigma=1.5, channel=3).cuda()
    if len(x.shape) == 3:
        x = x.unsqueeze(0)
    if len(y.shape) == 3:
        y = y.unsqueeze(0)
    return metric_networks['SSIM'](x, y)

def ssim_loss(x, y):
    return 1 - ssim_score(x, y)

def fast_ssim_loss(x, y):
    # Note! Only x get gradient in backward.
    is_train = x.requires_grad or y.requires_grad
    return 1 - fused_ssim(x.unsqueeze(0), y.unsqueeze(0), padding="valid", train=is_train)

def lpips_loss(x, y, net='vgg'):
    key = f'LPIPS_{net}'
    if key not in metric_networks:
        metric_networks[key] = LPIPS(net=net, version='0.1').cuda()
    if len(x.shape) == 3:
        x = x.unsqueeze(0)
    if len(y.shape) == 3:
        y = y.unsqueeze(0)
    return metric_networks[key](x, y)

def correct_lpips_loss(x, y, net='vgg'):
    key = f'LPIPS_{net}'
    if key not in metric_networks:
        metric_networks[key] = LPIPS(net=net, version='0.1').cuda()
    if len(x.shape) == 3:
        x = x.unsqueeze(0)
    if len(y.shape) == 3:
        y = y.unsqueeze(0)
    return metric_networks[key](x*2-1, y*2-1)

def entropy_loss(prob):
    pos_prob = prob.clamp(1e-6, 1-1e-6)
    neg_prob = 1 - pos_prob
    return -(pos_prob * pos_prob.log() + neg_prob * neg_prob.log()).mean()

def prob_concen_loss(prob):
    return (prob.square() * (1 - prob).square()).mean()


def exp_anneal(end_mul, iter_now, iter_from, iter_end):
    if end_mul == 1 or iter_now >= iter_end:
        return 1
    total_len = iter_end - iter_from + 1
    now_len = max(0, iter_now - iter_from + 1)
    now_p = min(1.0, now_len / total_len)
    return end_mul ** now_p


class SparseDepthLoss:
    def __init__(self, iter_end):
        self.iter_end = iter_end

    def is_active(self, iteration):
        return iteration <= self.iter_end

    def __call__(self, cam, render_pkg):
        assert "raw_T" in render_pkg, "Forgot to set `output_depth=True` when calling render?"
        assert "raw_depth" in render_pkg, "Forgot to set `output_depth=True` when calling render?"
        assert hasattr(cam, "sparse_pt") and cam.sparse_pt is not None, "No sparse points depth?"
        depth = render_pkg['raw_depth'][0] / (1 - render_pkg['raw_T']).clamp_min_(1e-4)
        sparse_pt = cam.sparse_pt.cuda()
        sparse_uv, sparse_depth = cam.project(sparse_pt, return_depth=True)
        rend_sparse_depth = torch.nn.functional.grid_sample(
            depth[None],
            sparse_uv[None,None],
            mode='bilinear', align_corners=False).squeeze()
        sparse_depth = sparse_depth.squeeze(1)
        return torch.nn.functional.smooth_l1_loss(rend_sparse_depth, sparse_depth)


class DepthAnythingv2Loss_old:
    def __init__(self, iter_from, iter_end, end_mult, alpha_adjust):
        self.iter_from = iter_from
        self.iter_end = iter_end
        self.end_mult = end_mult
        self.alpha_adjust = alpha_adjust

    def is_active(self, iteration):
        return iteration >= self.iter_from and iteration <= self.iter_end

    def __call__(self, cam, render_pkg, iteration):
        assert hasattr(cam, "depthanythingv2"), "Estimated depth not loaded"
        assert "raw_T" in render_pkg, "Forgot to set `output_depth=True` when calling render?"
        assert "raw_depth" in render_pkg, "Forgot to set `output_depth=True` when calling render?"

        if not self.is_active(iteration):
            return 0
        
        depth = render_pkg['raw_depth'].clamp_min(cam.near)
        alpha = (1 - render_pkg['raw_T'])
        if self.alpha_adjust:
            depth /= alpha.clamp_min(1e-3).detach()

        invdepth = 1 / depth.unsqueeze(1).clamp_min(cam.near)
        alpha = (1 - render_pkg['raw_T'][None])
        mono = cam.depthanythingv2.cuda()
        mono = mono[None,None]

        if invdepth.shape[-2:] != mono.shape[-2:]:
            mono = torch.nn.functional.interpolate(
                mono, size=invdepth.shape[-2:], mode='bilinear')

        X, _, Xref = invdepth.split(1)
        X = X * alpha
        Y = mono

        with torch.no_grad():
            Ymed = Y.median()
            Ys = (Y - Ymed).abs().mean()
            Xmed = Xref.median()
            Xs = (Xref - Xmed).abs().mean()
            target = (Y - Ymed) * (Xs/Ys) + Xmed

        mask = (target > 0.01) & (alpha > 0.5)
        X = X * mask
        target = target * mask
        loss = l2_loss(X, target)

        ratio = (iteration - self.iter_from) / (self.iter_end - self.iter_from)
        mult = self.end_mult ** ratio
        return mult * loss


class Mast3rMetricDepthLoss:
    def __init__(self, iter_from, iter_end, end_mult):
        self.iter_from = iter_from
        self.iter_end = iter_end
        self.end_mult = end_mult

    def is_active(self, iteration):
        return iteration >= self.iter_from and iteration <= self.iter_end

    def __call__(self, cam, render_pkg, iteration):
        assert hasattr(cam, "mast3r_metric_depth"), "Estimated depth not loaded"
        assert "raw_T" in render_pkg, "Forgot to set `output_depth=True` when calling render?"
        assert "raw_depth" in render_pkg, "Forgot to set `output_depth=True` when calling render?"

        if not self.is_active(iteration):
            return 0

        alpha = (1 - render_pkg['raw_T'][None])
        depth = render_pkg['raw_depth'][[0]][None]
        ref = cam.mast3r_metric_depth[None,None].cuda()

        if depth.shape[-2:] != ref.shape[-2:]:
            alpha = torch.nn.functional.interpolate(
                alpha, size=ref.shape[-2:], mode='bilinear', antialias=True)
            depth = torch.nn.functional.interpolate(
                depth, size=ref.shape[-2:], mode='bilinear', antialias=True)
            # ref = torch.nn.functional.interpolate(
            #     ref, size=depth.shape[-2:], mode='bilinear')

        # Compute cauchy loss
        active_idx = torch.where(alpha > 0.5)
        depth = depth / alpha
        loss = cauchy_loss(depth[active_idx], ref[active_idx], reduction='sum')
        loss = loss * (1 / depth.numel())

        ratio = (iteration - self.iter_from) / (self.iter_end - self.iter_from)
        mult = self.end_mult ** ratio
        return mult * loss


class NormalDepthConsistencyLoss:
    def __init__(self, iter_from, iter_end, ks, tol_deg):
        self.iter_from = iter_from
        self.iter_end = iter_end
        self.ks = ks
        self.tol_cos = np.cos(np.deg2rad(tol_deg))

    def is_active(self, iteration):
        return iteration >= self.iter_from and iteration <= self.iter_end

    def __call__(self, cam, render_pkg, iteration):
        assert "raw_T" in render_pkg, "Forgot to set `output_T=True` when calling render?"
        assert "raw_depth" in render_pkg, "Forgot to set `output_depth=True` when calling render?"
        assert "raw_normal" in render_pkg, "Forgot to set `output_normal=True` when calling render?"

        if not self.is_active(iteration):
            return 0

        # Read rendering results
        render_alpha = 1 - render_pkg['raw_T'].detach().squeeze(0)
        render_depth = render_pkg['raw_depth'][0]
        render_normal = render_pkg['raw_normal']

        # Compute depth to normal
        N_mean = cam.depth2normal(render_depth, ks=self.ks, tol_cos=self.tol_cos)

        # Blend with alpha and compute target
        target = render_alpha.square()
        N_mean = N_mean * render_alpha

        # Compute loss
        mask = (N_mean != 0).any(0)
        loss_map = (target - (render_normal * N_mean).sum(dim=0)) * mask
        loss = loss_map.mean()
        return loss


class NormalMedianConsistencyLoss:
    def __init__(self, iter_from, iter_end):
        self.iter_from = iter_from
        self.iter_end = iter_end

    def is_active(self, iteration):
        return iteration >= self.iter_from and iteration <= self.iter_end

    def __call__(self, cam, render_pkg, iteration):
        assert "raw_depth" in render_pkg, "Forgot to set `output_depth=True` when calling render?"
        assert "raw_normal" in render_pkg, "Forgot to set `output_normal=True` when calling render?"

        if not self.is_active(iteration):
            return 0

        # TODO: median depth is not differentiable
        render_median = render_pkg['raw_depth'][2]
        render_normal = render_pkg['raw_normal']

        # Compute depth to normal
        N_med = cam.depth2normal(render_median, ks=3)

        # Compute loss
        mask = (N_med != 0).any(0)
        loss_map = (1 - (render_normal * N_med).sum(dim=0)) * mask
        loss = loss_map.mean()
        return loss
    
    
class NormalMonoConsistencyLoss:
    def __init__(self, iter_from, iter_end):
        self.iter_from = iter_from
        self.iter_end = iter_end

    def is_active(self, iteration):
        return iteration >= self.iter_from and iteration <= self.iter_end

    def __call__(self, cam, render_pkg, iteration):
        assert "raw_depth" in render_pkg, "Forgot to set `output_depth=True` when calling render?"
        assert "raw_normal" in render_pkg, "Forgot to set `output_normal=True` when calling render?"

        if not self.is_active(iteration):
            return 0

        # TODO: median depth is not differentiable
        render_normal = render_pkg['raw_normal']
        alpha = (1 - render_pkg['raw_T'])

        mono = cam.depthanythingv2.cuda()
        mono = 1 / mono.clamp(min=1e-3)
        if render_normal.shape[-2:] != mono.shape[-2:]:
            mono = torch.nn.functional.interpolate(
                mono[None, None], size=render_normal.shape[-2:], mode='bilinear')[0]

        # Compute depth to normal
        N_mono = cam.depth2normal(mono, ks=3)

        # Compute loss
        mask = (N_mono != 0).any(0) * (alpha > 0.8).repeat(3, 1, 1)
        loss_map = (1 - (render_normal * N_mono).sum(dim=0)) * mask
        loss_map_abs = (render_normal - N_mono).abs().sum(dim=0) * mask
        loss = loss_map.mean() + loss_map_abs.mean()
        return loss



def lncc(ref, nea):
    # ref_gray: [batch_size, total_patch_size]
    # nea_grays: [batch_size, total_patch_size]
    bs, tps = nea.shape
    patch_size = int(np.sqrt(tps))

    ref_nea = ref * nea
    ref_nea = ref_nea.view(bs, 1, patch_size, patch_size)
    ref = ref.view(bs, 1, patch_size, patch_size)
    nea = nea.view(bs, 1, patch_size, patch_size)
    ref2 = ref.pow(2)
    nea2 = nea.pow(2)

    # sum over kernel
    filters = torch.ones(1, 1, patch_size, patch_size, device=ref.device)
    padding = patch_size // 2
    ref_sum = F.conv2d(ref, filters, stride=1, padding=padding)[:, :, padding, padding]
    nea_sum = F.conv2d(nea, filters, stride=1, padding=padding)[:, :, padding, padding]
    ref2_sum = F.conv2d(ref2, filters, stride=1, padding=padding)[:, :, padding, padding]
    nea2_sum = F.conv2d(nea2, filters, stride=1, padding=padding)[:, :, padding, padding]
    ref_nea_sum = F.conv2d(ref_nea, filters, stride=1, padding=padding)[:, :, padding, padding]

    # average over kernel
    ref_avg = ref_sum / tps
    nea_avg = nea_sum / tps

    cross = ref_nea_sum - nea_avg * ref_sum
    ref_var = ref2_sum - ref_avg * ref_sum
    nea_var = nea2_sum - nea_avg * nea_sum

    cc = cross * cross / (ref_var * nea_var + 1e-10)
    ncc = 1 - cc
    ncc = torch.clamp(ncc, 0.0, 2.0)
    ncc = torch.mean(ncc, dim=1, keepdim=True)
    mask = (ncc < 0.9)
    return ncc, mask



from random import randint
class DepthAnythingv2UncertaintyLoss:
    def __init__(self, iter_from, iter_end, end_mult, overall, power_level_uncertainty, alpha_adjust):
        self.iter_from = iter_from
        self.iter_end = iter_end
        self.end_mult = end_mult
        self.overall = overall
        self.power_level_uncertainty = power_level_uncertainty
        self.alpha_adjust = alpha_adjust

    def is_active(self, iteration):
        return iteration >= self.iter_from and iteration <= self.iter_end

    def __call__(self, cam, render_pkg, iteration):
        assert hasattr(cam, "depthanythingv2"), "Estimated depth not loaded"
        assert "raw_T" in render_pkg, "Forgot to set `output_depth=True` when calling render?"
        assert "raw_depth" in render_pkg, "Forgot to set `output_depth=True` when calling render?"

        if not self.is_active(iteration):
            return 0

        depth = render_pkg['raw_depth'].clamp_min(cam.near)
        alpha = (1 - render_pkg['raw_T'])
        if self.alpha_adjust:
            depth /= alpha.clamp_min(1e-3).detach()
        mono = cam.depthanythingv2.cuda()
        mono = -mono[None]
        if depth.shape[-2:] != mono.shape[-2:]:
            mono = torch.nn.functional.interpolate(
                mono[None], size=depth.shape[-2:], mode='bilinear')[0]

        if iteration > 2000 and not torch.isnan(render_pkg['feat']).any():
            level_map = render_pkg['feat']/(1-render_pkg['T'][0]).clamp(min=0.1).squeeze().detach()
            if level_map.shape[-2:] != depth.shape[-2:]:
                level_map = F.interpolate(level_map[None], size=depth.shape[-2:], mode='nearest')[0]
            level_weight = (level_map.max() - level_map.min()) / (level_map - level_map.min()).clamp(min=1.0)
            level_weight = level_weight[None] ** self.power_level_uncertainty
        else:
            level_weight = None

        depth = depth[0][None]
        mask = alpha < 0.8
                
        mono[mask] = mono[~mask].mean()
        depth[mask] = depth[~mask].mean()
        loss = 0
        loss += patch_norm_mse_loss_global(depth[None,...], mono[None,...], randint(17, 31), 0.001, weight=level_weight)
        loss += 0.1 * patch_norm_mse_loss(depth[None,...], mono[None,...], randint(17, 31), 0.001, weight=level_weight)

        if self.overall:
            loss += 0.1 * patch_norm_mse_loss_global(depth[None,...], (1 / -mono)[None,...], margin=0.001, weight=level_weight)
        
        ratio = (iteration - self.iter_from) / (self.iter_end - self.iter_from)
        mult = self.end_mult ** ratio
        return loss * mult



def normalize(input, mean=None, std=None):
    input_mean = torch.mean(input, dim=1, keepdim=True) if mean is None else mean
    input_std = torch.std(input, dim=1, keepdim=True) if std is None else std
    return (input - input_mean) / (input_std + 1e-2*torch.std(input.reshape(-1) + 1e-5))



def margin_l2_loss(network_output, gt, margin, return_mask=False, weight=None):
    mask = (network_output - gt).abs() > margin
    if mask.sum() == 0:
        if not return_mask: return 0
        else: return 0, mask
    if not return_mask:
        return (((network_output - gt)**2) * (weight if weight is not None else 1.0))[mask].mean()
    else:
        return (((network_output - gt)**2) * (weight if weight is not None else 1.0))[mask].mean(), mask
    
def margin_l1_loss(network_output, gt, margin, return_mask=False):
    mask = (network_output - gt).abs() > margin
    if not return_mask:
        return ((network_output - gt)[mask].abs()).mean()
    else:
        return ((network_output - gt)[mask].abs()).mean(), mask
    

def patchify(input, patch_size):
    patches = F.unfold(input, kernel_size=patch_size, stride=patch_size).permute(0,2,1).view(-1, 1*patch_size*patch_size)
    return patches

def patch_norm_mse_loss(input, target, patch_size=None, margin=0, return_mask=False, weight=None):
    '''
    input: [N, 1, H, W]
    weight: [1, H, W] or [N, 1, H, W]
    '''
    if patch_size is not None:
        input_patches = normalize(patchify(input, patch_size))
        target_patches = normalize(patchify(target, patch_size))
        
        if weight is not None:
            weight_patches = patchify(weight, patch_size)
        else:
            weight_patches = None
    else:
        input_patches = normalize(input.view(input.shape[0], -1))
        target_patches = normalize(target.view(input.shape[0], -1))
        weight_patches = weight.view(input.shape[0], -1) if weight is not None else None

    return margin_l2_loss(input_patches, target_patches, margin, return_mask, weight_patches)

def patch_norm_mse_loss_global(input, target, patch_size=None, margin=0, return_mask=False, weight=None):
    if patch_size is not None:
        input_patches = normalize(patchify(input, patch_size), std = input.std().detach())
        target_patches = normalize(patchify(target, patch_size), std = target.std().detach())
        
        if weight is not None:
            weight_patches = patchify(weight, patch_size)
        else:
            weight_patches = None
            
    else:
        input_patches = normalize(input.view(input.shape[0], -1), std = input.std().detach())
        target_patches = normalize(target.view(input.shape[0], -1), std = target.std().detach())
        weight_patches = weight.view(input.shape[0], -1) if weight is not None else None

    return margin_l2_loss(input_patches, target_patches, margin, return_mask, weight_patches)


def loss_depth_smoothness(depth, img):
    img_grad_x = img[:, :, :, :-1] - img[:, :, :, 1:]
    img_grad_y = img[:, :, :-1, :] - img[:, :, 1:, :]
    weight_x = torch.exp(-torch.abs(img_grad_x).mean(1).unsqueeze(1))
    weight_y = torch.exp(-torch.abs(img_grad_y).mean(1).unsqueeze(1))

    loss = (((depth[:, :, :, :-1] - depth[:, :, :, 1:]).abs() * weight_x).sum() +
            ((depth[:, :, :-1, :] - depth[:, :, 1:, :]).abs() * weight_y).sum()) / \
           (weight_x.sum() + weight_y.sum())
    return loss


# Gaussian Splatting SSIM

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

def ssim2(img1, img2, window_size=11):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return ssim_map.mean(0)

