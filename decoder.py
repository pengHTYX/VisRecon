import torch
import torch.nn as nn
import torch.nn.functional as F
from icecream import ic
from implicit.implicit_prt_gen import fibonacci_sphere
from attention import sdf_fusion, Encoder
from config import Config


class PositionalEncoding(object):

    def __init__(self, L=10):
        super().__init__()
        freq_bands = 2.**(torch.linspace(0, L - 1, L))
        self.freq_bands = freq_bands * torch.pi
        self.out_dim = 3 + 2 * L

    def __call__(self, p):
        out = []
        for freq in self.freq_bands:
            out.append(torch.sin(freq * p))
            out.append(torch.cos(freq * p))
        # nv x (3 x freq x 2)
        p = torch.cat(out, dim=1)
        return p


class ResConvBlock(nn.Module):

    def __init__(self, input_dim):
        super().__init__()
        self.conv1 = nn.Conv1d(input_dim, input_dim, 1)
        self.conv2 = nn.Conv1d(input_dim, input_dim, 1)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        return x + F.relu(self.conv2(out))


class ResConvNet(nn.Module):

    def __init__(self,
                 input_dim,
                 output_dim,
                 feature_dim,
                 block_size,
                 ignore_last=False):
        super().__init__()

        layers = nn.ModuleList(
            [nn.Conv1d(input_dim, feature_dim, 1)] +
            [ResConvBlock(feature_dim) for _ in range(block_size)] +
            ([] if ignore_last else [nn.Conv1d(feature_dim, output_dim, 1)]))
        self.res_module = nn.Sequential(*layers)

    def forward(self, x):
        return self.res_module(x)


def trilinear_interpolation(fea_grid,
                            p_query,
                            offset=0.5,
                            use_grid_sample=True):
    if use_grid_sample:
        fea_3d = F.grid_sample(fea_grid,
                               p_query.unsqueeze(2).unsqueeze(2), 'bilinear',
                               'border',
                               True).squeeze(-1).squeeze(-1)    # [B, ch, N]
    else:
        # Implement 5D grid_sample using 4D version, because it was not supported by TensorRT at the time
        batch_size, grid_fea_dim, grid_dim, _, _ = fea_grid.shape
        fea_3d = fea_grid.reshape(batch_size, -1, grid_dim, grid_dim)
        fea_3d = F.grid_sample(fea_3d,
                               p_query[..., :2].unsqueeze(-2) * 2,
                               padding_mode='border',
                               align_corners=True).reshape(
                                   batch_size, grid_fea_dim, grid_dim, -1)

        # align_corners=True
        z_scale = (grid_dim - 1) * (p_query[..., 2] + offset)
        z_floor = torch.floor(z_scale)
        z_delta = (z_scale - z_floor)[:, None, :]

        def query_z_value(z_index):
            z_index = torch.clamp(z_index, 0, grid_dim - 1).long()
            z_index = z_index[:, None, :].repeat_interleave(grid_fea_dim, dim=1)
            return fea_3d.gather(2, z_index[..., None, :]).squeeze(2)

        fea_3d = query_z_value(z_floor) * (
            1 - z_delta) + query_z_value(z_floor + 1) * z_delta

    return fea_3d


class Decoder(nn.Module):

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.grid_res = self.cfg.model.encoder_kwargs.grid_resolution

        hidden_dim = self.cfg.fea_dim
        hidden_size = self.cfg.hidden_size

        c_dim = self.cfg.model.c_dim
        grid_dim = self.cfg.model.grid_dim

        point_dim = 3
        if self.cfg.embedding:
            freq = self.cfg.embedding_freq
            self.encoding = PositionalEncoding(L=freq)
            point_dim += freq * 3 * 2

        fea_3d_dim = grid_dim + 3
        fea_2d_dim = c_dim + 3 + 1

        self.fea_encode_coarse = ResConvNet(
            point_dim + fea_3d_dim if self.cfg.fea_3d_only else point_dim +
            fea_3d_dim + fea_2d_dim,
            hidden_dim,
            hidden_dim,
            hidden_size,
            ignore_last=True)

        self.vis_infer = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim // 2, 1), nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim // 2, self.cfg.vis_sample_size, 1))

        self.occ_infer = nn.Sequential(
            ResConvNet(
                point_dim + fea_3d_dim if self.cfg.fea_3d_only else point_dim +
                fea_3d_dim + fea_2d_dim, hidden_dim // 2, hidden_dim,
                hidden_size), nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim // 2, 1, 1))

        self.color_infer = nn.Sequential(
            ResConvNet(
                point_dim + fea_3d_dim if self.cfg.fea_3d_only else point_dim +
                fea_3d_dim + fea_2d_dim, hidden_dim // 2, hidden_dim,
                hidden_size), nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim // 2, 3, 1), nn.Sigmoid())

        dirs, _, _ = fibonacci_sphere(self.cfg.vis_sample_size)

        self.register_buffer('sample_dirs', torch.from_numpy(dirs).float())
        # For backward compatibility, unused
        self.register_buffer(
            'volume_pos', torch.tensor([[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]]))

        if self.cfg.attention:
            self.fea_fusion = Encoder(n_head=8,
                                      n_layers=2,
                                      d_k=32,
                                      d_v=32,
                                      d_model=fea_2d_dim,
                                      d_inner=128)

    def forward(self,
                fea_grid,
                fea_hrnet,
                p_query,
                depth_img=None,
                mask_img=None,
                color_img=None,
                Rs=None,
                ts=None,
                intri=None,
                c_depth=None,
                center=None,
                scale=None,
                infer_occ=True,
                infer_color=True,
                batch=None,
                p_query_2d=None,
                p_2d_vis=None):
        '''
        feas:
        p_query: B, N, 3
        p_cam_vis: BV, N
        p_query_2d: BV, N, 3
        depth_img: BV, 1, H, W
        mask_img: BV, 1, H, W
        '''
        if len(p_query.shape) < 3:
            p_query = p_query.unsqueeze(0)
        _, point_num, _ = p_query.shape
        batch_size = fea_grid.shape[0]
        view_num = fea_hrnet.shape[0] // batch_size

        if batch is not None:
            batch_size = 1
            fea_grid = fea_grid[batch:batch + 1]
            fea_hrnet = fea_hrnet[batch * view_num:(batch + 1) * view_num]
            mask_img = mask_img[batch * view_num:(batch + 1) * view_num]
            Rs = Rs[batch:batch + 1]
            ts = ts[batch:batch + 1]
            intri = intri[batch:batch + 1]
            c_depth = c_depth[batch:batch + 1]
            center = center[batch:batch + 1]
            scale = scale[batch:batch + 1]

        p_query_normalized = p_query.clone()
        p_query_normalized = p_query_normalized - center
        p_query_normalized = p_query_normalized / scale[:, None, None]

        xyz_local = self.get_local_pos(p_query_normalized.clone()).squeeze(-1)
        p_query_normalized = p_query_normalized * 2    # normalize to [-1, 1], [B, N, 3]
        fea_3d = trilinear_interpolation(fea_grid, p_query_normalized)
        fea_3d = torch.cat([fea_3d, xyz_local], 1)

        if not self.cfg.fea_3d_only:
            cam_pos = torch.einsum('bvij,bvi->bvj', -Rs, ts)
            p_query_2d = torch.einsum(
                'bvni,bvji->bvnj',
                torch.einsum('bni,bvji->bvnj', p_query, Rs) + ts[..., None, :],
                intri)
            p_query_2d[
                ...,
                0] = (p_query_2d[..., 0] / 512 / p_query_2d[..., 2]) * 2 - 1
            p_query_2d[
                ...,
                1] = (p_query_2d[..., 1] / 512 / p_query_2d[..., 2]) * 2 - 1
            p_query_2d[..., 2] -= c_depth[..., None]

            pts_out_fov = (((p_query_2d[..., :2] >= 1).float() +
                            (p_query_2d[..., :2] <= -1).float()) >= 1).float()
            pts_out_fov = ((pts_out_fov[..., 0] + pts_out_fov[..., 1]) >=
                           1).float()
            p_2d_vis = 1. - pts_out_fov

            # 2d grid
            h_grid = p_query_2d[:, :, :, 0].view(batch_size * view_num,
                                                 point_num, 1, 1)
            v_grid = p_query_2d[:, :, :, 1].view(batch_size * view_num,
                                                 point_num, 1, 1)
            grid = torch.cat([h_grid, v_grid], dim=-1)

            # b x c x n
            p_2d_feat = F.grid_sample(fea_hrnet, grid,
                                      align_corners=True).squeeze(-1)
            pt_group_mask = F.grid_sample(mask_img,
                                          grid,
                                          mode='nearest',
                                          align_corners=True).squeeze(-1)
            # pt_color = F.grid_sample(color_img, grid,
            #                          align_corners=True).squeeze(-1)
            p_cam_xyz = p_query_2d.view(batch_size * view_num, point_num,
                                        3).permute(0, 2, 1)
            p_2d_feat = torch.cat([p_2d_feat, p_cam_xyz, pt_group_mask], dim=1)

            p_2d_feat = p_2d_feat.reshape(batch_size, view_num, -1, point_num)
            # pt_color = pt_color.reshape(batch_size, view_num, -1, point_num)

            p_2d_vis = p_2d_vis.unsqueeze(-2)

            weight = p_2d_vis
            weight_sum = weight.sum(dim=1, keepdim=True)
            weight_sum[weight_sum < 1e-6] = 1.
            weight = weight / weight_sum
            fea_2d = (p_2d_feat * weight).sum(dim=1)

        p_query_normalized = p_query_normalized.permute(0, 2, 1)
        if self.cfg.embedding:
            p_query_normalized = torch.cat(
                [p_query_normalized,
                 self.encoding(p_query_normalized)], dim=1)

        if not self.cfg.fea_3d_only:
            fea_in = torch.cat([p_query_normalized, fea_2d, fea_3d], dim=1)
        else:
            fea_in = torch.cat([p_query_normalized, fea_3d], dim=1)

        fea_coarse = self.fea_encode_coarse(fea_in)
        vis = self.vis_infer(fea_coarse)

        out = {"vis": vis}
        if not infer_occ and not infer_color:
            return out

        if self.cfg.fea_3d_only:
            if infer_occ:
                occ = self.occ_infer(fea_in)
                out["occ"] = occ
            if infer_color:
                color = self.color_infer(fea_in)
                out["color"] = color
            return out

        if self.cfg.vis_fuse:
            cam_pos = cam_pos.reshape(batch_size, view_num, 3, -1)
            p_query = p_query.permute(0, 2, 1)
            ray_dir = cam_pos - p_query[:, None, ...]

            if not self.sample_dirs.is_cuda:
                self.sample_dirs = self.sample_dirs.to(ray_dir.device)

            # TODO: need spatial index and a table, input ray dir, output 3 cloest dirs' indices
            geo_term = torch.einsum("bvcn,sc->bvsn", ray_dir, self.sample_dirs)
            geo_top_k = geo_term.topk(3, -2)

            vis_rep = torch.repeat_interleave(vis.detach()[:, None, ...],
                                              view_num,
                                              dim=1)
            ray_vis = torch.gather(vis_rep, dim=2, index=geo_top_k.indices)
            # cosine distance weighted interpolation
            ray_vis = ray_vis * (geo_top_k.values /
                                 geo_top_k.values.sum(dim=-2, keepdim=True))
            ray_vis = ray_vis.sum(dim=-2, keepdim=True)
            ray_vis = torch.sigmoid(ray_vis)
            ray_vis = ray_vis * p_2d_vis
            weight = -torch.log(1 - ray_vis)
            weight = torch.clamp(weight, 0, 100)
            weight_sum = weight.sum(dim=1, keepdim=True)
            weight_sum[weight_sum < 1e-6] = 1.
            weight = weight / weight_sum
            fea_2d = (p_2d_feat * weight).sum(dim=1)
            # fea_color = (pt_color * weight).sum(dim=1)
        elif self.cfg.attention:
            fea_2d = sdf_fusion(p_2d_feat.reshape(batch_size * view_num, -1,
                                                  point_num, 1),
                                self.fea_fusion,
                                view_num=view_num)
            fea_2d = fea_2d.reshape(batch_size, view_num, -1, point_num)
            fea_2d = (fea_2d * weight).sum(dim=1)

        fea_in = torch.cat([p_query_normalized, fea_3d, fea_2d], dim=1)

        if infer_occ:
            occ = self.occ_infer(fea_in)
            out["occ"] = occ

        if infer_color:
            color = self.color_infer(fea_in)
            out["color"] = color

        return out

    def get_local_pos(self, p):
        p = p + 0.5    # 0-1
        unit_size = 1 / self.grid_res
        p = torch.remainder(p, unit_size)
        p = p / unit_size
        p = p.unsqueeze(-1).permute(0, 2, 1, 3)
        return p


if __name__ == '__main__':
    from icecream import ic

    device = 'cpu'
    sample_size = 500000
    decoder = Decoder(Config()).to(device)
    fea_grid = torch.randn((1, 32, 64, 64, 64),
                           device=device,
                           dtype=torch.float32)
    fea_2d = torch.randn((4, 32, 256, 256), device=device, dtype=torch.float32)
    p_query = torch.randn((1, sample_size, 3),
                          device=device,
                          dtype=torch.float32)
    mask_img = torch.randn((4, 1, 512, 512), device=device, dtype=torch.float32)
    color_img = torch.randn((4, 3, 512, 512),
                            device=device,
                            dtype=torch.float32)
    Rs = torch.randn((1, 4, 3, 3), device=device, dtype=torch.float32)
    ts = torch.randn((1, 4, 3), device=device, dtype=torch.float32)
    intri = torch.randn((1, 4, 3, 3), device=device, dtype=torch.float32)
    c_depth = torch.randn((1, 4), device=device, dtype=torch.float32)

    torch.onnx.export(decoder, {
        'fea_grid': fea_grid,
        'fea_2d': fea_2d,
        'p_query': p_query,
        'mask_img': mask_img,
        'color_img': color_img,
        'Rs': Rs,
        'ts': ts,
        'intri': intri,
        'c_depth': c_depth,
    },
                      "onnx/decoder.onnx",
                      input_names=[
                          'fea_grid', 'fea_2d', 'p_query', 'mask_img',
                          'color_img', 'Rs', 'ts', 'intri', 'c_depth'
                      ],
                      output_names=['vis', 'occ', 'color'],
                      opset_version=16)
