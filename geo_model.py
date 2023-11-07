import torch
import torch.nn as nn
from hrnet import HRNet_modified
import ray_utils
from pointnet import LocalPoolPointNet
from tqdm import tqdm
from decoder import Decoder
from functools import partial
import polyscope as ps
from icecream import ic
from config import Config


class GeoModel(nn.Module):

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        encoder_kwargs = self.cfg.model.encoder_kwargs
        self.c_dim = self.cfg.model.c_dim
        self.grid_dim = self.cfg.model.grid_dim
        self.nums = 2000
        self.hrnet = HRNet_modified(inputMode=self.cfg.model.input_mode,
                                    numOutput=self.c_dim,
                                    normLayer=nn.BatchNorm2d)

        self.resolution_grid = encoder_kwargs.grid_resolution
        self.decoder = Decoder(cfg)

        self.pointnet = LocalPoolPointNet(
            c_dim=self.grid_dim,
            dim=6,
            grid_resolution=self.resolution_grid,
            unet3d_kwargs=encoder_kwargs.unet3d_kwargs)

        self.normal_conv = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=self.cfg.normal_downsample),
            nn.Conv2d(self.c_dim, 32, 3, 1, 1), nn.ReLU(True),
            nn.Conv2d(32, 32, 3, 1, 1), nn.ReLU(True), nn.Conv2d(32, 3, 1, 1),
            nn.Tanh())

    def forward(
            self,
            p,
            p_query,
            depth=None,
            mask=None,
            color=None,
            Rs=None,
            ts=None,
            intri=None,
            c_depth=None,    # 2d
            center=None,
            scale=None,
            sample_pix=None,
            sample_intri=None,
            sample_depth=None,
            sample_R=None,
            sample_t=None,
            its=None):
        feat, normal_out = self.encode(p, depth, mask, color)

        decode = partial(self.decoder,
                         fea_grid=feat['grid'],
                         fea_hrnet=feat['hrnet'],
                         depth_img=depth,
                         mask_img=mask,
                         color_img=color,
                         Rs=Rs,
                         ts=ts,
                         intri=intri,
                         c_depth=c_depth,
                         center=center,
                         scale=scale)
        outs = decode(p_query=p_query)

        if normal_out is not None:
            outs['normal_2d'] = normal_out

        if sample_pix is not None:
            rays_dirs = ray_utils.get_ray_directions_pts(pts=sample_pix,
                                                         img_h=512,
                                                         img_w=512,
                                                         focal=[550, 550])
            rays_o = torch.einsum('bij,bi->bj', -sample_R, sample_t)
            rays_d = torch.einsum('bni,bij->bnj', rays_dirs, sample_R)
            rays_o = rays_o[:, None, :].repeat_interleave(rays_d.shape[1],
                                                          dim=1)

            with torch.no_grad():
                p_hat, mask_hat = self.ray_marching(rays_o, rays_d,
                                                    sample_depth, decode)

            batch_size, sample_pt_num, _ = sample_pix.shape
            sampled_albedo = torch.zeros((batch_size, sample_pt_num, 3),
                                         device=sample_pix.device)
            patch_albedo_out = []
            for b in range(batch_size):
                if mask_hat[b].sum() > 0:
                    p_valid = p_hat[b]
                    out_valid = decode(p_query=p_valid, batch=b)
                    patch_albedo_out.append(out_valid['color'][0].T)
            patch_albedo_out = torch.concat(patch_albedo_out)
            sampled_albedo[mask_hat] = patch_albedo_out

            outs['sampled_albedo'] = sampled_albedo
            outs['sampled_mask'] = mask_hat

        return outs

    def encode(self, p, depth, mask, color):
        fea = {}
        if hasattr(self, 'normalnet'):
            fea['normal'] = self.normalnet(color)
            normal_2d = self.normal_conv(fea['normal'])
            normal_2d = normal_2d / (
                torch.norm(normal_2d, dim=1, keepdim=True) + 1e-6)

        # 1 x n x c
        fea['grid'] = self.pointnet(p[..., :3], p[..., 3:])

        if not self.cfg.fea_3d_only:
            hrout = self.hrnet(rgb=color, mask=mask, depth=depth)    # norm
            fea['hrnet'] = hrout[-1]
            feat_normal = hrout[-1]
            normal_2d = self.normal_conv(feat_normal)
            normal_2d = normal_2d / (
                torch.norm(normal_2d, dim=1, keepdim=True) + 1e-6)
        else:
            fea['hrnet'] = None
            normal_2d = None

        return fea, normal_2d

    def fea_encode(
        self,
        pc,
        color,
        depth,
        mask,
        Rs,
        ts,
        intri,
        c_depth,
        center=None,
        scale=None,
    ):
        feat, _ = self.encode(pc, depth, mask, color)
        decode = partial(self.decoder,
                         fea_grid=feat['grid'],
                         fea_hrnet=feat['hrnet'],
                         depth_img=depth,
                         mask_img=mask,
                         color_img=color,
                         Rs=Rs,
                         ts=ts,
                         intri=intri,
                         c_depth=c_depth,
                         center=center,
                         scale=scale)

        return partial(self.infer_geo, decode=decode)

    def infer_geo(self, decode, p_q, infer_occ=True, infer_color=True):
        group_size = self.cfg.test.group_size

        pt_group_sdf_list = []
        pt_group_vis_list = []
        pt_group_color_list = []
        for pt_group in tqdm(torch.split(p_q, group_size)):
            fout = decode(p_query=pt_group.unsqueeze(0),
                          infer_occ=infer_occ,
                          infer_color=infer_color)

            if 'occ' in fout:
                pt_group_sdf_list.append(fout['occ'].detach().cpu())
            if 'vis' in fout:
                pt_group_vis_list.append(fout['vis'].detach().cpu())
            if 'color' in fout:
                pt_group_color_list.append(fout['color'].detach().cpu())

        if len(pt_group_sdf_list) != 0:
            occ = torch.cat(pt_group_sdf_list, -1).squeeze(0).T.reshape(-1,)
        else:
            occ = None

        if len(pt_group_vis_list) != 0:
            vis = torch.cat(pt_group_vis_list, -1).squeeze(0).T
        else:
            vis = None

        if len(pt_group_color_list) != 0:
            color = torch.cat(pt_group_color_list, -1).squeeze(0).T
        else:
            color = None

        return {'occ': occ, 'vis': vis, 'color': color}

    # Modified from: https://github.com/autonomousvision/differentiable_volumetric_rendering/blob/master/im2mesh/dvr/models/depth_function.py
    # We only use it to supervise albedo. Back-propagate loss to occ/sdf is not implemented
    def ray_marching(self,
                     rays_o,
                     rays_dir,
                     rays_depth,
                     decode,
                     depth_std=1e-2,
                     step_size=6,
                     secant_step=3):
        # rays_o: b x n x 3
        # rays_dir: b x n x 3
        # rays_depth: b x n x 1

        batch_size, _, _ = rays_o.shape

        intervals = torch.linspace(-depth_std, depth_std, step_size).to(rays_o)
        # b x n x 1
        proposals_mask = (rays_depth > 0).squeeze(-1)

        results = []
        # trade query calls with indexing complexity
        for b in range(batch_size):
            mask_batch = proposals_mask[b]
            # n' x 3
            rays_o_batch = rays_o[b][mask_batch]
            # n' x 3
            rays_dir_batch = rays_dir[b][mask_batch]
            # n' x 1
            rays_depth_batch = rays_depth[b][mask_batch]
            # step_size x n' x 3
            proposals_batch = torch.stack([
                rays_o_batch + (rays_depth_batch + interval) * rays_dir_batch
                for interval in intervals
            ])
            with torch.no_grad():
                # step_size x n'
                # here we treat repeated queries as single batch
                sdf_batch = decode(p_query=proposals_batch.reshape(-1, 3),
                                   infer_occ=True,
                                   infer_color=False,
                                   batch=b)["occ"].reshape(step_size, -1)

            # invalid if first query has sdf > 0
            invalid_first_sdf = sdf_batch[0, :] > 0
            # invalid if sign does not change
            sign_test = sdf_batch[:-1, :] * sdf_batch[1:, :]
            no_sign_change = (sign_test < 0).sum(dim=0) == 0

            invalid_mask = torch.logical_or(invalid_first_sdf, no_sign_change)
            # update mask
            mask_batch[mask_batch.nonzero()[invalid_mask]] = False

            # prepare for raymarching
            valid_mask = torch.logical_not(invalid_mask)
            valid_indices = valid_mask.nonzero().reshape(-1)
            # choose the earliest sign change
            sign_cost = torch.sign(
                sign_test.index_select(
                    dim=1, index=valid_indices)) * torch.arange(
                        step_size, 1, step=-1).to(sign_test)[:, None]
            min_indices = sign_cost.min(dim=0).indices
            sdf_batch_valid = sdf_batch.index_select(dim=1, index=valid_indices)

            # Run Secant method (faster than bisection, more efficient than Newton)
            rays_o_batch_valid = rays_o_batch[valid_mask]
            rays_dir_batch_valid = rays_dir_batch[valid_mask]
            rays_depth_batch_valid = rays_depth_batch[valid_mask]

            y_0 = torch.gather(sdf_batch_valid,
                               dim=0,
                               index=min_indices[None, :]).reshape(-1, 1)
            y_1 = torch.gather(sdf_batch_valid,
                               dim=0,
                               index=min_indices[None, :] + 1).reshape(-1, 1)

            z_0 = rays_depth_batch_valid + intervals[min_indices][:, None]
            z_1 = rays_depth_batch_valid + intervals[min_indices + 1][:, None]

            def f_sdf(z):
                with torch.no_grad():
                    sdf_eval = decode(p_query=rays_o_batch_valid +
                                      z * rays_dir_batch_valid,
                                      infer_occ=True,
                                      infer_color=False,
                                      batch=b)["occ"].reshape(-1, 1)
                return sdf_eval

            z_opt = self.run_secant(z_0,
                                    z_1,
                                    y_0,
                                    y_1,
                                    f_sdf,
                                    secant_step=secant_step)
            query_opt = rays_o_batch_valid + z_opt * rays_dir_batch_valid

            results.append(query_opt)

        return results, proposals_mask

    def run_secant(self, x_0, x_1, y_0, y_1, f_sdf, secant_step=8, eps=1e-5):
        x_2 = x_1 - y_1 * (x_1 - x_0) / (y_1 - y_0 + eps)
        y_2 = f_sdf(x_2)

        split_mask = y_2 > 0
        x_0 = torch.where(split_mask, x_0, x_2)
        x_1 = torch.where(split_mask, x_2, x_1)
        y_0 = torch.where(split_mask, y_0, y_2)
        y_1 = torch.where(split_mask, y_2, y_1)

        # set them equal, avoid numerical error
        # x_1 = torch.where(y_2.abs() < eps, x_0, x_1)

        if secant_step <= 0:
            return torch.where(y_0.abs() < y_1.abs(), x_0, x_1)
        else:
            return self.run_secant(x_0, x_1, y_0, y_1, f_sdf, secant_step - 1)
