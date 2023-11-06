import os

from torch.utils import data as torch_data
import numpy as np
import cv2
from rgbd_streamer import Calibrations
from rgbd_reader import get_persp_matrix, generate_mesh_from_depth_map
from glob import glob
from config import Config
from thuman import sample_color_bilinear
import json
import argparse

import polyscope as ps
from icecream import ic


def unproj_depth_map(depth, mask, intr):
    height, width = np.shape(depth)
    N = height * width
    xx, yy = np.meshgrid(np.arange(width), np.arange(height))
    X = (xx - intr[0, 2]) * depth / intr[0, 0]
    Y = (yy - intr[1, 2]) * depth / intr[1, 1]
    points = np.stack((X, Y, depth), -1)
    points = np.reshape(points, (N, 3))
    mask = np.reshape(mask, (N))
    points = points[mask > 0]

    return points


class RGBDCalibration:

    def __init__(self):
        self.d_height = 0
        self.d_width = 0
        self.c_height = 0
        self.c_width = 0
        self.Kd = np.float32(np.identity(3, dtype=float))
        self.Kc = np.float32(np.identity(3, dtype=float))
        self.Tcalib = np.float32(np.identity(
            4, dtype=float))    # depth_0 -> depth_i
        self.Tglobal = np.float32(np.identity(
            4, dtype=float))    # depth_i -> world
        self.Td2c = np.float32(np.identity(4,
                                           dtype=float))    # depth_i -> color_i
        self.center_depth = 0.0

    def print(self):
        print('d_height: {}, d_width: {}'.format(self.d_height, self.d_width))
        print('c_height: {}, c_width: {}'.format(self.c_height, self.c_width))
        print('Kd: {}'.format(self.Kd))
        print('Kc: {}'.format(self.Kc))
        print('Tcalib:\n{}'.format(self.Tcalib))
        print('Tglobal:\n{}'.format(self.Tglobal))
        print("center_depth: ".format())


class THumanDataset(torch_data.Dataset):

    def __init__(self, dataset_folder, cfg: Config = None):
        self.cfg = cfg
        self.dataset_folder = os.path.expandvars(dataset_folder)

        mc = Calibrations(self.dataset_folder, camera_num=4)
        # 640 * 576 -> 512 * 512 -> 512 * 512
        mc.cut_d_calibs(512, 512, 512)
        # 720 * 1280 -> 720 * 720 -> 512 * 512
        mc.cut_c_calibs(512, 512, 720)

        self.calibs = mc.calibs
        self.Mwarp = np.array([[1., 0., 0., 0.], [0., -1., 0., 0.],
                               [0., 0., 1., 0.], [0., 0., 0., 1.]])

        temp_list = glob(
            os.path.join(self.dataset_folder,
                         'RGB_DEPTH/frame_*_cmask_view_0.png'))
        self.frame_id = []
        for item in temp_list:
            id = int(item.split('_cmask')[0].split('_')[-1])
            self.frame_id.append(id)
        self.frame_id = sorted(self.frame_id)

    def __len__(self):
        return len(self.frame_id)

    def __getitem__(self, idx):
        frame = self.frame_id[idx]
        data = {}
        ## visible point cloud
        view_ids = [0, 1, 2, 3][:self.cfg.data.view_num]

        depth_img_list = []
        mask_img_list = []
        mask_img_dilate_list = []
        color_img_list = []
        center_depth = []
        pc = []
        pc_normal = []
        pc_erode = []
        pc_erode_normal = []
        normals_lr = []
        Rs = []
        ts = []
        intris = []

        for id in view_ids:
            pose = self.calibs[id].Tcalib @ self.Mwarp
            intri = self.calibs[id].Kd
            pose_inv = np.linalg.inv(pose)

            # cache the color ones
            color_K = self.calibs[id].Kc
            color_T = self.calibs[id].Td2c @ pose
            intris.append(color_K)
            R = color_T[:3, :3]
            t = color_T[:3, 3]
            Rs.append(R)
            ts.append(t)

            depth_path = os.path.join(
                self.dataset_folder,
                'RGB_DEPTH/frame_%d_depth_view_%d.png' % (frame, id))
            color_path = os.path.join(
                self.dataset_folder,
                'RGB_DEPTH/frame_%d_color_view_%d.png' % (frame, id))

            depth_img = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            depth_img = np.float32(depth_img) / 1000
            depth_mask = np.ones_like(depth_img)
            depth_mask[depth_img > 20] = 0

            c_depth = (depth_img[depth_mask > 1e-6]).mean()
            center_depth.append(c_depth)

            depth_erode = cv2.erode(depth_img,
                                    np.ones((3, 3), np.uint8),
                                    iterations=7)
            depth_erode_mask = np.ones_like(depth_erode)
            depth_erode_mask[depth_erode > 20] = 0
            depth_erode_raw = depth_erode.copy()
            depth_erode[depth_mask > 0] = depth_img[depth_mask > 0]

            color_path = os.path.join(
                self.dataset_folder,
                'RGB_DEPTH/frame_%d_color_view_%d.jpg' % (frame, id))
            color_img = np.float32(
                cv2.cvtColor(cv2.imread(color_path), cv2.COLOR_BGR2RGB)) / 255
            color_img = 2.0 * color_img - 1.0    # transform pixel value to [-1,1]

            mask_path = os.path.join(
                self.dataset_folder,
                'RGB_DEPTH/frame_%d_cmask_view_%d.png' % (frame, id))
            mask_img = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED) / 255
            mask_img_dilate = cv2.dilate(mask_img,
                                         np.ones((3, 3), np.uint8),
                                         iterations=5)
            mask_img_dilate = np.float32(mask_img_dilate > 0)

            mask_img_list.append(mask_img)
            mask_img_dilate_list.append(mask_img_dilate)

            # world view
            normal_map = np.zeros_like(color_img)
            normals_lr.append(normal_map)

            def unproj(depth_img, depth_mask):
                pc = unproj_depth_map(depth_img, depth_mask, intri)    # N,3
                pc_normal = np.array([0, 0, 0]).reshape(1, -1) - pc
                pc_normal /= np.linalg.norm(pc_normal, axis=1, keepdims=True)

                pc = np.matmul(pc, np.transpose(
                    pose_inv[:3, :3])) + pose_inv[:3, 3]
                # inv(pose_inv).T
                M = pose.T
                pc_normal = np.matmul(pc_normal, np.transpose(
                    M[:3, :3])) + M[:3, 3]

                return pc, pc_normal

            depth_pc, depth_pc_normal = unproj(depth_erode, depth_mask)
            depth_dilate_pc, depth_dilate_pc_normal = unproj(
                depth_erode_raw, depth_erode_mask)

            # IMPORTANT! color and depth are NOT aligned by default
            V = depth_pc @ R.T + t
            V_im = V @ color_K.T
            V_im = V_im[:, :2] / V_im[:, -1][:, None]
            V_uv = V_im / np.array(color_img.shape[:2])[None, :]
            V_uv[:, 1] = 1 - V_uv[:, 1]
            color_cat = sample_color_bilinear(color_img, V_uv)

            depth_pc = np.concatenate([depth_pc, color_cat], 1)
            pc.append(depth_pc)
            pc_normal.append(depth_pc_normal)
            pc_erode.append(depth_dilate_pc)
            pc_erode_normal.append(depth_dilate_pc_normal)
            depth_erode -= c_depth
            depth_erode[depth_erode > 20] = 1 + 1e-6
            depth_img_list.append(depth_erode)
            color_img_list.append(color_img)
        center_depth = np.array(center_depth)

        pc = np.concatenate(pc, 0, dtype=np.float32)
        pc_normal = np.concatenate(pc_normal, 0, dtype=np.float32)
        pc_erode = np.concatenate(pc_erode, 0, dtype=np.float32)
        pc_erode_normal = np.concatenate(pc_erode_normal, 0, dtype=np.float32)

        # n x 3
        center = np.array([0, 0, self.cfg.test.zshift]).reshape(1, 3)
        scale = self.cfg.test.scale

        pc[:, :3] -= center
        pc[:, :3] /= scale

        pc_erode[:, :3] -= center
        pc_erode[:, :3] /= scale

        Rs = np.stack(Rs)
        ts = np.stack(ts)

        data = {
            'frame': frame,
            'pc': pc,
            'pc_erode': pc_erode,
            'depth_img': np.stack(depth_img_list, 0),
            'mask_img': np.stack(mask_img_list, 0),
            'mask_img_dilate': np.stack(mask_img_dilate_list, 0),
            'color_img': np.stack(color_img_list, 0),
            'normal_img': np.stack(normals_lr, 0),
            'Rs': Rs,
            'ts': ts,
            'intri': np.stack(intris, 0),
            'c_depth': center_depth,
            'center': center,
            'scale': scale
        }

        return data


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_folder', type=str, help='Path to data folder.')
    args = parser.parse_args()

    cfg = Config()
    if args.data_folder is not None:
        cfg.data.realworld_folder = args.data_folder

    val_dataset = THumanDataset(cfg.data.realworld_folder, cfg=cfg)
    val_data = next(iter(val_dataset))

    ic(val_data['frame'])

    # n x 3
    pc = val_data['pc'][:, :3]
    center = val_data['center']
    scale = val_data['scale']
    pc = (pc * scale) + center

    Rs = val_data['Rs']
    ts = val_data['ts']

    cam_pos = np.einsum('bji,bj->bi', Rs, -ts)

    ps.init()
    pc_vis = ps.register_point_cloud("depth pc", pc)
    pc_vis.add_color_quantity("depth color",
                              0.5 * (val_data['pc'][:, 3:] + 1.0),
                              enabled=True)
    ps.register_point_cloud(f"camera centers",
                            cam_pos,
                            radius=0.01,
                            enabled=True)
    ps.show()

    # exit()

    color_imgs = val_data['color_img']
    depth_imgs = val_data['depth_img']
    Ks = val_data['intri']
    c_depth = val_data['c_depth']

    pts_cam_view = np.einsum('bni,bji->bnj',
                             np.einsum('ni,bji->bnj', pc, Rs) + ts[:, None, :],
                             Ks)
    pts_cam_view[..., 0] /= pts_cam_view[..., 2]
    pts_cam_view[..., 1] /= pts_cam_view[..., 2]

    for i in range(cfg.data.view_num):
        img_pts = pts_cam_view[i]
        img = cv2.cvtColor(color_imgs[i], cv2.COLOR_RGB2BGR)
        for pt in img_pts:
            cv2.circle(img, (int(pt[0]), int(pt[1])),
                       radius=1,
                       color=(255, 0, 0),
                       thickness=-1)
        cv2.imshow(f"color_{i}", img)
    cv2.waitKey(0)
