# Modified from: https://github.com/yanx27/Pointnet_Pointnet2_pytorch/blob/master/models/pointnet2_utils.py
import torch
import torch.nn as nn
from unet3d import UNet3D
from icecream import ic
import dataclasses


# Resnet Blocks
class ResnetBlockFC(nn.Module):
    """ Fully connected ResNet Block class.
    Args:
        size_in (int): input dimension
        size_out (int): output dimension
        size_h (int): hidden dimension
    """

    def __init__(self, size_in, size_out=None, size_h=None):
        super().__init__()
        # Attributes
        if size_out is None:
            size_out = size_in

        if size_h is None:
            size_h = min(size_in, size_out)

        self.size_in = size_in
        self.size_h = size_h
        self.size_out = size_out
        # Submodules
        self.fc_0 = nn.Linear(size_in, size_h)
        self.fc_1 = nn.Linear(size_h, size_out)
        self.act_fn = nn.ReLU()

        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = nn.Linear(size_in, size_out, bias=False)
        # Initialization
        nn.init.zeros_(self.fc_1.weight)

    def forward(self, x):
        net = self.fc_0(self.act_fn(x))
        dx = self.fc_1(self.act_fn(net))

        if self.shortcut is not None:
            x_s = self.shortcut(x)
        else:
            x_s = x

        return x_s + dx


class LocalPoolPointNet(nn.Module):
    """ PointNet-based encoder network with ResNet blocks for each point.
        Number of input points are fixed.
    Args:
        c_dim (int): dimension of latent code c
        dim (int): input points dimension
        hidden_dim (int): hidden dimension of the network
        scatter_type (str): feature aggregation when doing local pooling
        use_unet3d (bool): weather to use 3D U-Net
        unet3d_kwargs (dict): 3D U-Net parameters
        plane_resolution (int): defined resolution for plane feature
        grid_resolution (int): defined resolution for grid feature
        plane_type (str, list): feature type, 'xz' - 1-plane, ['xz', 'xy', 'yz'] - 3-plane, ['grid'] - 3D grid volume
        padding (float): conventional padding parameter of ONet for unit cube, so [-0.5, 0.5] -> [-0.55, 0.55]
        n_blocks (int): number of blocks ResNetBlockFC layers
    """

    def __init__(self,
                 c_dim=128,
                 dim=3,
                 hidden_dim=128,
                 unet3d_kwargs=None,
                 grid_resolution=None,
                 n_blocks=5):
        super().__init__()
        self.c_dim = c_dim

        self.fc_pos = nn.Linear(dim, 2 * hidden_dim)
        self.blocks = nn.ModuleList([
            ResnetBlockFC(2 * hidden_dim, hidden_dim) for _ in range(n_blocks)
        ])
        self.fc_c = nn.Linear(hidden_dim, c_dim)
        self.act_fn = nn.ReLU()
        self.hidden_dim = hidden_dim

        self.unet3d = UNet3D(**dataclasses.asdict(unet3d_kwargs))

        self.resolution_grid = grid_resolution

    def generate_grid_features(self, index, c):
        # scatter grid features from points
        batch_dim, fea_dim = c.size(0), c.size(2)
        c = c.permute(0, 2, 1)
        dim = int(self.resolution_grid**3)
        fea_grid = torch.zeros((batch_dim, fea_dim, dim),
                               dtype=c.dtype,
                               device=c.device)
        index = torch.repeat_interleave(index, fea_dim, dim=1)
        fea_grid = fea_grid.scatter_reduce(-1,
                                           index,
                                           c,
                                           reduce='mean',
                                           include_self=False)

        fea_grid = fea_grid.reshape(
            index.size(0), self.c_dim, self.resolution_grid,
            self.resolution_grid,
            self.resolution_grid)    # sparse matrix (B _x 512 _x res _x res)

        fea_grid = self.unet3d(fea_grid)
        return fea_grid

    def pool_local(self, index, c: torch.Tensor):
        batch_dim, fea_dim = c.size(0), c.size(2)
        # scatter plane features from points
        c = c.permute(0, 2, 1)
        dim = int(self.resolution_grid**3)
        fea = torch.zeros((batch_dim, fea_dim, dim),
                          dtype=c.dtype,
                          device=c.device)
        index = torch.repeat_interleave(index, fea_dim, dim=1)
        fea = fea.scatter_reduce(-1,
                                 index,
                                 c,
                                 reduce='amax',
                                 include_self=False)
        # gather feature back to points
        fea = fea.gather(dim=2, index=index.expand(-1, fea_dim, -1))
        return fea.permute(0, 2, 1)

    def coordinate2index(self, xyz, offset=0.5):
        """ Normalize coordinate to [0, 1] for unit cube experiments.
            Corresponds to our 3D model
        Args:
            xyz (tensor): coordinate
        """
        xyz = torch.clamp((xyz + offset) * self.resolution_grid, 0,
                          self.resolution_grid - 1).long()
        index = xyz[:, :, 0] + self.resolution_grid * (
            xyz[:, :, 1] + self.resolution_grid * xyz[:, :, 2])
        index = index[:, None, :]
        return index

    def forward(self, xyz: torch.Tensor, fea: torch.Tensor):
        # acquire the index for each point
        index = self.coordinate2index(xyz)

        p_avg = torch.mean(xyz, dim=1, keepdim=True)    # [B, 1, 3]
        p_input = xyz - p_avg
        deno = torch.max(torch.std(p_input.permute(1, 0, 2), dim=0))
        p_input = p_input.div(deno)
        p_input = torch.cat([p_input, fea], -1)
        net = self.fc_pos(p_input)

        for i, block in enumerate(self.blocks):
            if i == 0:
                net = block(net)
            else:
                pooled = self.pool_local(index, net)
                net = torch.cat([net, pooled], dim=2)
                net = block(net)

        c = self.fc_c(net)
        grid_fea = self.generate_grid_features(index, c)
        return grid_fea


if __name__ == '__main__':
    """
    For PointNet testing
    """
    u_net_3d_param = {
        "num_levels": 3,
        "f_maps": 32,
        "in_channels": 32,
        "out_channels": 32,
    }
    occ_enc = LocalPoolPointNet(use_unet3d=True,
                                dim=6,
                                unet3d_kwargs=u_net_3d_param,
                                grid_resolution=64,
                                hidden_dim=32,
                                c_dim=32,
                                plane_type=['grid']).eval().to("cuda")
    xyz = torch.randn((3, 5000, 3), device="cuda", dtype=torch.float32)
    fea = torch.randn((3, 5000, 3), device="cuda", dtype=torch.float32)
    ic(occ_enc(xyz, fea).shape)

    # import torch_tensorrt
    # with torch_tensorrt.logging.debug():
    #     trt_module = torch_tensorrt.compile(
    #         occ_enc,
    #         inputs=[
    #             torch_tensorrt.Input(xyz.shape, dtype=torch.float32),
    #             torch_tensorrt.Input(fea.shape, dtype=torch.float32)
    #         ],
    #         min_block_size=1,
    #         truncate_long_and_double=True,
    #         debug=True)

    # print(trt_module(xyz, fea).shape)
