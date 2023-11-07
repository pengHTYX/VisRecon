import os

import igl
import torch
from torch.utils import data as torch_data
from scipy.io import loadmat
import numpy as np
import cv2
import torch.nn.functional as F
import trimesh
from implicit.implicit_prt_gen import compute_sample_occlusion
from glob import glob
import skimage.measure
from icecream import ic
from config import Config

H, W = 512, 512
k = np.identity(3, dtype=np.float32)
k[0, 0], k[1, 1] = 550, 550
# k[0, 0], k[1, 1] = 1100, 1100
k[0, 2], k[1, 2] = W / 2, H / 2


def filter_visual_hull(pts,
                       Rs,
                       ts,
                       intri,
                       mask_img,
                       device=None,
                       is_numpy=True,
                       is_batch=False):
    if is_numpy:
        pts = torch.from_numpy(pts)
        Rs = torch.from_numpy(Rs)
        ts = torch.from_numpy(ts)
        intri = torch.from_numpy(intri)
        mask_img = torch.from_numpy(mask_img).float()

    if device is not None:
        pts = pts.to(device)

    pts_cam_view = torch.einsum(
        'bvni,bvji->bvnj' if is_batch else 'vni,vji->vnj',
        torch.einsum('ni,bvji->bvnj' if is_batch else 'ni,vji->vnj', pts, Rs) +
        ts[..., None, :], intri)
    pts_cam_view[...,
                 0] = (pts_cam_view[..., 0] / W / pts_cam_view[..., 2]) * 2 - 1
    pts_cam_view[...,
                 1] = (pts_cam_view[..., 1] / H / pts_cam_view[..., 2]) * 2 - 1

    if is_batch:
        pts_cam_view = pts_cam_view.reshape(1 * Rs.shape[1], -1, 3)

    pts_in_hull = ((pts_cam_view[..., :2] >= 1) +
                   (pts_cam_view[..., :2] <= -1)) >= 1
    pts_in_hull = (pts_in_hull[..., 0] + pts_in_hull[..., 1]) >= 1
    pts_in_hull = torch.logical_not(pts_in_hull)

    grid = torch.stack([pts_cam_view[..., None, 0], pts_cam_view[..., None, 1]],
                       dim=-1)
    pt_not_in_mask = torch.logical_not(
        F.grid_sample(mask_img, grid, 'bilinear', 'border', True).squeeze(
            1).squeeze(-1) > 0)
    valid_mask = torch.logical_and(pts_in_hull, pt_not_in_mask).sum(0) == 0
    pts = pts[valid_mask]

    if is_numpy:
        pts = pts.numpy()
        valid_mask = valid_mask.numpy()

    return pts, valid_mask


def sample_points_in_specific_area(sample_number,
                                   mask,
                                   inner_prob=0.9,
                                   patch_size=32,
                                   stride=1,
                                   normalized=False):
    """
    Sample points
    :param sample_number: number of points required
    :param mask: mask used to determine inner and outer area, [1, H ,W] or [H, W]
    :param inner_prob: probability of sampling point outside the mask
    :param patch_size: sample patches instead of individual points
    :param normalized: whether output normalized to [0, 1]
    :return: coordinates of sampled points
    """
    inner_prob = max(min(inner_prob, 1.0), 0.0)
    mask = np.squeeze(mask)
    h, w = mask.shape
    if patch_size != 1:
        mask_resized = skimage.measure.block_reduce(mask,
                                                    (patch_size, patch_size),
                                                    np.max)
    else:
        mask_resized = mask
    mask_resized = torch.from_numpy(mask_resized)
    prob_blocks = torch.zeros_like(mask_resized)
    prob_blocks[mask_resized < 0.5] = 1 - inner_prob
    prob_blocks[mask_resized > 0.5] = inner_prob

    if patch_size != 1:
        border = 1
        prob_blocks[:border, :] = 0
        prob_blocks[-border:, :] = 0
        prob_blocks[:, :border] = 0
        prob_blocks[:, -border:] = 0

    prob_blocks = torch.flatten(prob_blocks)
    # prob_list_test = prob_blocks.cpu().numpy()
    # prob_list_test = np.reshape(prob_list_test, (32, 32))

    num_points = sample_number // (patch_size**2)
    sample_blocks = torch.tensor(
        list(
            torch_data.WeightedRandomSampler(prob_blocks,
                                             num_points,
                                             replacement=False)))

    if patch_size == 1:
        sample_x = (torch.div(sample_blocks, w,
                              rounding_mode='floor')).unsqueeze(-1)
        sample_y = (sample_blocks % w).unsqueeze(-1)
        sample_point = torch.cat([sample_y, sample_x], dim=1)
    else:
        sample_point = torch.zeros((num_points, patch_size, patch_size, 2))
        sample_block_x = (torch.div(
            sample_blocks, (h // patch_size),
            rounding_mode='floor')).unsqueeze(-1) * patch_size
        sample_block_y = (sample_blocks %
                          (w // patch_size)).unsqueeze(-1) * patch_size
        for i in range(patch_size):
            for j in range(patch_size):
                sample_point[:, i, j, 0] = sample_block_y[:, 0] + j
                sample_point[:, i, j, 1] = sample_block_x[:, 0] + i
    sample_point = sample_point.view(-1, 2).float()

    # test_img = np.zeros((512, 512))
    # pts_camview = sample_point.numpy()
    # pts_camview[:, 0] = np.clip(pts_camview[:, 0], 0, 512)
    # pts_camview[:, 1] = np.clip(pts_camview[:, 1], 0, 512).astype(np.int32)
    # for j in range(patch_size*patch_size*2):
    #     test_img[int(pts_camview[j, 1]), int(pts_camview[j, 0])] =  255
    # exit()

    if normalized:
        sample_point[:, 0] = (sample_point[:, 0] / (w - 1)) * 2.0 - 1.0
        sample_point[:, 1] = (sample_point[:, 1] / (h - 1)) * 2.0 - 1.0
    return sample_point


def sample_target_view_rgbd(num=10000,
                            depth=None,
                            color=None,
                            albedo=None,
                            mask=None,
                            normal=None,
                            sample_outer_rate=0.,
                            patch=32,
                            stride=1):
    """
    :param color: rgb image
    :param mask: mask image
    :param sample_num: number of sample points
    :param sample_depth: whether we should sample depth, used in training color with gt depth
    :param depth: depth image
    :param sample_outer_rate: inner_prob: probability of sampling point outside the mask
    :param sample_patch: sample patches instead of individual points
    :return: sampled rgb, d and coordinates
    """
    device = 'cpu'
    result = {}
    # get sample points, [N, 2]
    sampled_xy = sample_points_in_specific_area(num,
                                                mask,
                                                1.0 - sample_outer_rate,
                                                patch,
                                                stride,
                                                normalized=True)

    # [1, 1, N, 2]
    sampled_xy = sampled_xy.unsqueeze(0).unsqueeze(0)

    if color is not None:
        assert (color.shape[2] == mask.shape[0])
        assert (color.shape[3] == mask.shape[1])

        sampled_rgb = F.grid_sample(color, sampled_xy, align_corners=True)
        sampled_rgb = sampled_rgb.squeeze(0).permute(1, 2, 0)    # [1, N, 3]
        result['sampled_rgb'] = sampled_rgb.to(device)

    if albedo is not None:
        sampled_alb = F.grid_sample(albedo,
                                    sampled_xy,
                                    mode='bilinear',
                                    align_corners=True)
        sampled_alb = sampled_alb.squeeze(0).permute(1, 2, 0)
        result['sampled_alb'] = sampled_alb.to(device)

    # sample depth
    if depth is not None:
        assert (depth.shape[2] == mask.shape[0])
        assert (depth.shape[3] == mask.shape[1])
        sampled_depth = F.grid_sample(depth,
                                      sampled_xy,
                                      mode='nearest',
                                      align_corners=True)

        sampled_depth = sampled_depth.squeeze(0).permute(1, 2, 0)    # [1, N, 1]
        result['sampled_depth'] = sampled_depth.to(device)

        # mask = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0)
        # sampled_mask = F.grid_sample(mask, sampled_xy, mode='bilinear', align_corners=True)
        # print(torch.sum(sampled_mask))
        # result['sampled_mask'] = sampled_mask.squeeze(0).permute(1,2,0).to(device)

    if normal is not None:
        sampled_normal = F.grid_sample(normal,
                                       sampled_xy,
                                       mode='bilinear',
                                       align_corners=True)
        result['sampled_normal'] = sampled_normal.squeeze(0).permute(
            1, 2, 0).to(device)    # 1,n,3

    result['sampled_xy'] = sampled_xy.squeeze(0).to(device)    # 1,n,2
    return result


def unproj_depth_map(depth, intr, mask=None):
    height, width = np.shape(depth)
    N = height * width
    xx, yy = np.meshgrid(np.arange(width), np.arange(height))
    X = (xx - intr[0, 2]) * depth / intr[0, 0]
    Y = (yy - intr[1, 2]) * depth / intr[1, 1]
    pts_view = np.stack((X, Y, depth), -1).reshape(-1, 3)
    if mask is not None:
        mask = np.reshape(mask, (N))
        pts_view = pts_view[mask > 0]

    return pts_view


def calculate_pts_visibility(pts_visibility, pts_cam_view):
    """
    :param pts_visibility: [B, N, 1]
    :param pts_cam_view: [B, N ,3]
    :return: no return
    """
    pts_visibility[:] = 0
    pts_out_fov = np.logical_or(pts_cam_view[:, 0:2] >= 1,
                                pts_cam_view[:, 0:2] <= -1)
    pts_out_fov = np.logical_or(pts_out_fov[:, 0], pts_out_fov[:, 1])
    pts_visibility[np.logical_not(pts_out_fov)] = 1


def generate_random_views(input_view):
    """
    get views
    :return: view_ids: input view, render_view: view to be rendered
    """
    view_ids = []
    total_view = 60
    interval = total_view // input_view
    # stratified sampling
    for i in range(input_view):
        random_view = i * interval + \
            np.random.randint(total_view // input_view)
        view_ids.append(random_view)
    return view_ids


def load_cam(cams, id):
    cam_rs = np.float32(cams['cam_rs'][id])
    cam_ts = np.float32(cams['cam_ts'][id])
    cam_r, cam_t = cv2.Rodrigues(cam_rs)[0], cam_ts.squeeze()
    center_depth = cams['center_depth'][0, id]
    return cam_r, cam_t, center_depth


def gradient_based_resampling(samples, samples_clr, samples_grad, count):
    # generate multi-frequency boolean masks
    high_grad_indexes = samples_grad >= 20    # [pnum,]
    medium_grad_indexes = np.all(np.concatenate(
        ((samples_grad < 20)[:, None], (samples_grad >= 5)[:, None]), axis=1),
                                 axis=1)    # [pnum,]
    low_grad_indexes = samples_grad < 5    # [pnum,]

    # select high frequency points
    samples_high = samples[high_grad_indexes]
    samples_clr_high = samples_clr[high_grad_indexes]

    samples_medium = samples[medium_grad_indexes]
    samples_clr_medium = samples_clr[medium_grad_indexes]

    samples_low = samples[low_grad_indexes]
    samples_clr_low = samples_clr[low_grad_indexes]

    sample_idx = np.arange(len(samples))
    sample_idx = np.concatenate([
        sample_idx[high_grad_indexes], sample_idx[medium_grad_indexes],
        sample_idx[low_grad_indexes]
    ])[:count]

    if samples_high.shape[0] >= count:
        samples = samples_high[:count, ...]
        samples_clr = samples_clr_high[:count, ...]
    elif (samples_high.shape[0] + samples_medium.shape[0]) >= count:
        samples = np.concatenate((samples_high, samples_medium), 0)[:count, ...]
        samples_clr = np.concatenate((samples_clr_high, samples_clr_medium),
                                     0)[:count, ...]
    else:
        samples = np.concatenate((samples_high, samples_medium, samples_low),
                                 0)[:count, ...]
        samples_clr = np.concatenate(
            (samples_clr_high, samples_clr_medium, samples_clr_low), 0)[:count,
                                                                        ...]
    return samples, samples_clr, sample_idx


def sample_points_with_color(mesh,
                             albedo_texture,
                             color_texture,
                             normal_texture,
                             count=5000,
                             sigma=0.05,
                             hair_centroid=None,
                             hair_scale=0.15):
    face_index = np.random.choice(np.arange(len(mesh.faces)), 5 * count)

    if hair_centroid is not None:
        voi = np.linalg.norm(mesh.vertices - hair_centroid, axis=1) < hair_scale
        v_sel = np.zeros(len(mesh.vertices))
        v_sel[voi] = 1
        f_sel = np.argwhere(np.sum(v_sel[mesh.faces], axis=1) > 0).reshape(-1,)

        n_face_sel = len(face_index)
        sel = int(max(0.5 * n_face_sel, min(len(f_sel), 0.75 * n_face_sel)))
        non_sel = n_face_sel - sel

        face_index = np.concatenate([
            f_sel[np.random.choice(np.arange(len(f_sel)), sel)],
            face_index[np.random.choice(np.arange(n_face_sel),
                                        non_sel,
                                        replace=False)]
        ])

    # # debug
    # face_index = np.arange(fnum)
    # pull triangles and uv-coordinates into the form of an origin + 2 vectors
    triangles = mesh.triangles[face_index, ...]    # [pnum, 3, 3]
    tri_origins = triangles[:, 0, ...]    # [pnum, 1, 3]
    tri_vectors = triangles[:, 1:, ...].copy()    # [pnum, 2, 3]
    tri_vectors -= np.tile(tri_origins, (1, 2)).reshape((-1, 2, 3))

    tri_uvs = mesh.visual.uv[mesh.faces][face_index, ...]    # [pnum,3,2]
    tri_origins_uv = tri_uvs[:, 0, ...]    # [pnum,1,2]
    tri_vectors_uv = tri_uvs[:, 1:, ...].copy()    # [pnum,2,2]
    tri_vectors_uv -= np.tile(tri_origins_uv, (1, 2)).reshape(
        (-1, 2, 2))    # [pnum,2,2]

    # calculate triangle normals for random shift later
    try:
        tri_normals = np.cross(tri_vectors[:, 0, ...], tri_vectors[:, 1, ...])
    except Exception as e:
        None
    tri_normals = tri_normals / np.linalg.norm(tri_normals, axis=1)[:, None]

    # randomly generate two 0-1 scalar components to multiply edge vectors by
    # points will be distributed on a quadrilateral if we use 2 0-1 samples
    # if the two scalar components sum less than 1.0 the point will be
    # inside the triangle, so we find vectors longer than 1.0 and
    # transform them to be inside the triangle
    random_lengths = np.random.random((len(tri_vectors), 2, 1))
    random_test = random_lengths.sum(axis=1).reshape(-1) > 1.0
    random_lengths[random_test] -= 1.0
    random_lengths = np.abs(random_lengths)

    # finally, offset by the origin to generate
    # (n,3) points in space on the triangle
    sample_vector = (tri_vectors * random_lengths).sum(axis=1)
    samples_surf = tri_origins + sample_vector    # random sample

    # adaptive random shifts
    curv_radius = 0.002
    curv_thresh = 0.004
    curvs = trimesh.curvature.discrete_gaussian_curvature_measure(
        mesh, samples_surf, curv_radius)
    curvs = abs(curvs)
    curvs = curvs / max(curvs)    # normalize curvature

    sigmas = np.zeros(curvs.shape)
    sigmas[curvs <= curv_thresh] = sigma
    sigmas[curvs > curv_thresh] = sigma / 5
    shift = (np.random.randn(samples_surf.shape[0]) *
             sigmas)[:, None] * tri_normals
    samples = np.float32(samples_surf + shift)    # normal shift

    sample_vector_uv = (tri_vectors_uv * random_lengths).sum(axis=1)
    samples_uv = tri_origins_uv + sample_vector_uv
    samples_alb = sample_color_bilinear(albedo_texture, samples_uv) / 255

    albedo_gradient = cv2.Sobel(cv2.Laplacian(albedo_texture, cv2.CV_64F),
                                cv2.CV_8U,
                                1,
                                1,
                                ksize=5)
    samples_grad = sample_color_bilinear(albedo_gradient, samples_uv)
    samples_grad = np.mean(samples_grad, -1)

    gradient_sample_count = 3 * count
    samples, samples_alb, gradient_sample_idx = gradient_based_resampling(
        samples, samples_alb, samples_grad, gradient_sample_count)

    shift = shift[gradient_sample_idx]

    count_alb, count_sigma, count_free = int(0.4 * count), int(
        0.4 * count), int(0.2 * count)

    shift = np.linalg.norm(shift, ord=2, axis=-1)
    shift_bool = shift < 0.01
    debug_num = np.sum(shift_bool)
    id_alb = np.random.choice(range(debug_num), count_alb)
    id_alb = np.arange(gradient_sample_count)[shift_bool][id_alb]
    p_alb = samples[id_alb, ...]
    samples_color = samples_alb[id_alb, ...]

    id_sigma = np.random.choice(range(gradient_sample_count - debug_num),
                                count_sigma)
    id_sigma = np.arange(gradient_sample_count)[~shift_bool][id_sigma]
    p_sigma = samples[id_sigma, ...]

    p_free = (np.random.random((count_free, 3)) * (np.ones(
        (3,)) * 1.0)) + (np.ones((3,)) * (-0.5))
    samples = np.concatenate([p_alb, p_sigma, p_free], 0)

    samples_sdf, _, _ = igl.signed_distance(samples, mesh.vertices, mesh.faces)

    return np.float32(samples), np.float32(
        samples_sdf < 0), np.float32(samples_color)


def sample_color_bilinear(img, uv):
    img_height = img.shape[0]
    img_width = img.shape[1]

    # UV coordinates should be (n, 2) float
    uv = np.asanyarray(uv, dtype=np.float64)

    # get texture image pixel positions of UV coordinates
    x = (uv[:, 0] * (img_width - 1))
    y = ((1 - uv[:, 1]) * (img_height - 1))

    # convert to int and wrap to image
    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)
    x1 = x0 + 1
    y1 = y0 + 1

    # clip
    x0 = np.clip(x0, 0, img_width - 1)
    x1 = np.clip(x1, 0, img_width - 1)
    y0 = np.clip(y0, 0, img_height - 1)
    y1 = np.clip(y1, 0, img_height - 1)

    # bilinear interpolation
    img = np.asanyarray(img)
    c0 = img[y0, x0]
    c1 = img[y1, x0]
    c2 = img[y0, x1]
    c3 = img[y1, x1]

    if c0.ndim == 1:
        c0 = c0[:, None]
        c1 = c1[:, None]
        c2 = c2[:, None]
        c3 = c3[:, None]

    w0 = ((x1 - x) * (y1 - y))[:, None]
    w1 = ((x1 - x) * (y - y0))[:, None]
    w2 = ((x - x0) * (y1 - y))[:, None]
    w3 = ((x - x0) * (y - y0))[:, None]

    colors = c0 * w0 + c1 * w1 + c2 * w2 + c3 * w3
    # wsum = w0 + w1 + w2 + w3
    # colors = colors / wsum
    # print('colors.shape: ', colors.shape)
    colors = np.array(np.squeeze(colors))
    return colors


class THumanDataset(torch_data.Dataset):

    def __init__(self, dataset_folder, cfg: Config = None, mode='train'):
        # Attributes
        self.dataset_folder = dataset_folder
        self.cfg = cfg
        self.mode = mode

        if self.cfg.overfit:
            self.dataset_folder = os.path.expandvars(self.cfg.data.val_folder)
            self.models = self.cfg.data.overfit_data
        else:
            self.dataset_folder = os.path.expandvars(dataset_folder)
            self.models = sorted(os.listdir(self.dataset_folder))

        if self.mode == 'test':
            self.cfg.hair = False

    def __len__(self):
        return len(self.models)

    def __getitem__(self, idx):
        model = self.models[idx]
        model_name = model
        model_path = os.path.join(self.dataset_folder, model)

        # It worth mentioning that for pixel alignment representation, the reconstruction quality is view dependent
        # Hence we fix the view ids for evaluation (i.e. computing metrics, etc.)
        if self.mode == 'test' and self.cfg.data.view_num == 4:
            view_ids = [0, 15, 30, 45]
            # For reproducibility
            np.random.seed(0)
            torch.manual_seed(0)
            torch.cuda.manual_seed_all(0)
        else:
            view_ids = generate_random_views(self.cfg.data.view_num)

        cams_mat_path = 'cams.mat'
        cams = loadmat(os.path.join(model_path, cams_mat_path))
        depth_img_list = []
        mask_img_list = []
        mask_img_dilate_list = []
        color_img_list = []
        center_depth = []
        pc = []
        hair_pc = []
        normal_img_list = []
        Rs = []
        ts = []
        for id in view_ids:
            R, t, c_depth = load_cam(cams, id)
            center_depth.append(c_depth)
            Rs.append(R)
            ts.append(t)

            depth_path_gt = os.path.join(model_path, f'depth_view_{id}.png')
            depth_img_gt = cv2.imread(depth_path_gt, cv2.IMREAD_UNCHANGED)
            depth_img_gt = np.float32(depth_img_gt) / 1000
            depth_mask_gt = np.ones_like(depth_img_gt)
            depth_mask_gt[depth_img_gt > 20] = 0

            mask_img_dilate = cv2.blur(depth_mask_gt, (9, 9))

            mask_img_list.append(depth_mask_gt)
            mask_img_dilate_list.append(mask_img_dilate)

            # Remove hair pointcloud during supervision.
            # Useful because hair depth cannot be (at least can not be accurately) capture by Kinect depth sensor
            if self.cfg.hair:
                # Generated using `hair_seg.py`
                depth_path = os.path.join(model_path,
                                          f'hair_masked_depth_view_{id}.png')
                depth_img = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
                depth_img = np.float32(depth_img) / 1000
                depth_mask = np.ones_like(depth_img)
                depth_mask[depth_img > 20] = 0

                hair_mask = (depth_mask_gt != depth_mask).reshape(-1,)
                if np.sum(hair_mask) > 0:
                    hair_pts = unproj_depth_map(depth_img_gt, k)
                    hair_pts = hair_pts[hair_mask]
                    hair_pts = (hair_pts - t) @ R
                    hair_pc.append(hair_pts)
            else:
                depth_img = depth_img_gt
                depth_mask = depth_mask_gt

            depth_erode = cv2.erode(depth_img,
                                    np.ones((3, 3), np.uint8),
                                    iterations=5)
            depth_erode[depth_mask > 0] = depth_img[depth_mask > 0]

            color_path = os.path.join(model_path,
                                      'color_view_' + str(id) + '.jpg')
            color_img = np.float32(
                cv2.cvtColor(cv2.imread(color_path), cv2.COLOR_BGR2RGB)) / 255
            # transform pixel value to [-1,1]
            color_img = 2.0 * color_img - 1.0

            normal_map = np.float32(
                cv2.cvtColor(
                    cv2.imread(
                        os.path.join(model_path, 'normal_view_' + str(id) +
                                     '.jpg')), cv2.COLOR_BGR2RGB)) / 255
            normal_map = 2. * normal_map - 1.
            # cam view
            normal_view = np.reshape(normal_map, (-1, 3))
            normal_view = np.matmul(normal_view, np.transpose(R))
            normal_view[:, -1] = -normal_view[:, -1]
            normal_map = np.reshape(normal_view, (H, W, 3))
            normal_img_list.append(normal_map)

            def unproj_depth(depth_img, depth_mask):
                pc = unproj_depth_map(depth_img, k, depth_mask)    # N,3
                pc = (pc - t) @ R
                return pc

            depth_pc = unproj_depth(depth_erode, depth_mask)
            color_cat = np.reshape(color_img[depth_mask > 0, :], (-1, 3))
            depth_pc = np.concatenate([depth_pc, color_cat], 1)
            pc.append(depth_pc)
            depth_erode -= c_depth
            depth_erode[depth_erode > 20] = 1 + 1e-6
            depth_img_list.append(depth_erode)
            color_img_list.append(color_img)

        Rs = np.stack(Rs)
        ts = np.stack(ts)
        intri = np.stack([k] * len(view_ids), 0)
        mask_img = np.stack(mask_img_list, 0)

        center_depth = np.array(center_depth)
        pc = np.concatenate(pc, 0, dtype=np.float32)

        target_sample_size = 10000
        pc_indices = np.random.choice(np.arange(len(pc)), target_sample_size)
        pc = pc[pc_indices]

        hair_centroid = np.average(np.vstack(hair_pc), axis=0,
                                   keepdims=True) if len(hair_pc) > 0 else None

        # scale point cloud to unit cube (IMPORTANT otherwise missing limbs...)
        xyz = pc[..., :3]
        aabb_min = np.min(xyz, axis=0, keepdims=True)
        aabb_max = np.max(xyz, axis=0, keepdims=True)
        center = 0.5 * (aabb_max + aabb_min)
        scale = 1.05 * 2.0 * np.max(aabb_max - center)

        pc[..., :3] -= center
        pc[..., :3] /= scale

        data = {
            'pc': pc,
            'depth_img': np.stack(depth_img_list, 0),
            'mask_img': mask_img,
            'mask_img_dilate': np.stack(mask_img_dilate_list, 0),
            'color_img': np.stack(color_img_list, 0),
            'normal_img': np.stack(normal_img_list, 0),
            'Rs': Rs,
            'ts': ts,
            'intri': intri,
            'c_depth': center_depth,
            'center': center,
            'scale': scale,
            'model_name': model_name
        }

        # query points
        if self.mode == 'train':
            point_num = self.cfg.point_num

            if self.cfg.live_sample or self.cfg.data_gen:
                file_path = os.path.join(model_path, model_name + '.obj')
                mesh: trimesh.Trimesh = trimesh.load(file_path,
                                                     process=False,
                                                     maintain_order=True,
                                                     use_embree=True)
                material_file = glob(os.path.join(model_path, "*.mtl"))[0]
                with open(material_file) as f:
                    lines = f.readlines()
                    for l in lines:
                        if l.startswith("map_Kd"):
                            texture_file = l.split(' ')[-1]
                            texture_file = texture_file.strip('\n')
                albedo_texture = cv2.cvtColor(
                    cv2.imread(os.path.join(model_path, texture_file),
                               cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)
                color_texture = None
                normal_texture = None

                vertices = np.array(mesh.vertices)
                vertex_normals = np.array(mesh.vertex_normals)
                faces = mesh.faces

                mesh_samples = vertices + 3e-4 * vertex_normals
                mesh_sample_normals = vertex_normals
                mesh_sample_vis = compute_sample_occlusion(
                    vertices, faces, mesh_samples)
                mesh_sample_color = sample_color_bilinear(
                    albedo_texture, mesh.visual.uv) / 255

                if not self.cfg.data_gen:
                    mesh_sample_idx = np.random.choice(
                        np.arange(len(vertices)),
                        point_num,
                        replace=len(vertices) < point_num)
                    mesh_samples = mesh_samples[mesh_sample_idx]
                    mesh_sample_normals = mesh_sample_normals[mesh_sample_idx]
                    mesh_sample_vis = mesh_sample_vis[mesh_sample_idx]
                    mesh_sample_color = mesh_sample_color[mesh_sample_idx]

                samples, sample_occ, sample_color = sample_points_with_color(
                    mesh, albedo_texture, color_texture, normal_texture,
                    point_num, 0.05, hair_centroid)

                sample_visibility = compute_sample_occlusion(
                    vertices, faces, samples)

                samples = np.concatenate([samples, mesh_samples], axis=0)
                sample_visibility = np.concatenate(
                    [sample_visibility, mesh_sample_vis], axis=0)
                sample_color = np.concatenate([sample_color, mesh_sample_color],
                                              axis=0)

            else:
                querys = np.load(os.path.join(model_path, 'querys.npy'))
                occ = np.load(os.path.join(model_path, 'occ.npy'))
                color = np.load(os.path.join(model_path, 'color.npy'))
                visibility = np.load(os.path.join(model_path, 'visibility.npy'))
                normals = np.load(os.path.join(model_path, 'normals.npy'))

                # near
                point_num_near = int(0.4 * point_num)
                point_num_near_all = int(0.4 * len(occ))
                near_idx = np.random.choice(np.arange(point_num_near_all),
                                            point_num_near)

                sample_near = querys[:point_num_near_all][near_idx]
                sample_near_occ = occ[:point_num_near_all][near_idx]
                sample_near_color = color[:point_num_near_all][near_idx]
                sample_near_vis = visibility[:point_num_near_all][near_idx]

                # far
                point_num_far = point_num - point_num_near
                point_num_far_all = len(occ) - point_num_near_all
                far_idx = np.random.choice(np.arange(point_num_far_all),
                                           point_num_far)
                non_surface_split = point_num_near_all + point_num_far_all

                sample_far = querys[point_num_near_all:non_surface_split][
                    far_idx]
                sample_far_occ = occ[point_num_near_all:non_surface_split][
                    far_idx]
                sample_far_vis = visibility[
                    point_num_near_all:non_surface_split][far_idx]

                # surface
                point_num_surface = point_num
                point_num_surface_all = len(normals)
                surface_idx = np.random.choice(np.arange(point_num_surface_all),
                                               point_num_surface)

                sample_surface = querys[non_surface_split:][surface_idx]
                sample_surface_color = color[point_num_near_all:][surface_idx]
                sample_surface_vis = visibility[non_surface_split:][surface_idx]
                sample_surface_normal = normals[surface_idx]

                samples = np.concatenate(
                    [sample_near, sample_far, sample_surface], axis=0)
                sample_occ = np.concatenate([sample_near_occ, sample_far_occ],
                                            axis=0)
                sample_color = np.concatenate(
                    [sample_near_color, sample_surface_color], axis=0)
                sample_visibility = np.concatenate(
                    [sample_near_vis, sample_far_vis, sample_surface_vis],
                    axis=0)
                mesh_sample_normals = sample_surface_normal

            pts_cam_view = np.einsum(
                'vni,vji->vnj',
                np.einsum('ni,vji->vnj', samples, Rs) + ts[..., None, :], intri)
            pts_cam_view[
                ...,
                0] = (pts_cam_view[..., 0] / W / pts_cam_view[..., 2]) * 2 - 1
            pts_cam_view[
                ...,
                1] = (pts_cam_view[..., 1] / H / pts_cam_view[..., 2]) * 2 - 1
            pts_cam_view[..., 2] -= c_depth[..., None]

            pts_cam_view = pts_cam_view.reshape(1 * self.cfg.data.view_num, -1,
                                                3)

            pts_in_hull = ((pts_cam_view[..., :2] >= 1) +
                           (pts_cam_view[..., :2] <= -1)) >= 1
            pts_in_hull = (pts_in_hull[..., 0] + pts_in_hull[..., 1]) >= 1
            points_camview_vis = np.logical_not(pts_in_hull)

            data.update({
                'querys': samples,
                'occ': sample_occ,
                'color': sample_color,
                'visibility': sample_visibility,
                'normals': mesh_sample_normals,
                'pts_cam_vis': points_camview_vis
            })

            if self.cfg.differential:
                first_view = view_ids[0]
                interval = self.cfg.data.total_view // 6
                target_id = np.random.randint(
                    first_view, first_view + interval *
                    (self.cfg.data.view_num - 1)) % self.cfg.data.total_view
                target_R, target_t, target_depth = load_cam(cams, target_id)

                depth_path = os.path.join(
                    model_path, 'depth_view_' + str(target_id) + '.png')
                depth_img = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
                target_depth = np.float32(depth_img) / 1000

                depth_mask = np.float32(np.ones_like(depth_img))
                depth_mask[target_depth > 20] = 0
                target_depth[target_depth > 20] = 0

                normal_map = cv2.cvtColor(
                    cv2.imread(
                        os.path.join(model_path,
                                     'normal_view_' + str(target_id) + '.jpg'),
                        cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)
                normal_map = np.float32(normal_map / 255. * 2 - 1.)

                color = cv2.cvtColor(
                    cv2.imread(
                        os.path.join(model_path,
                                     'color_view_' + str(target_id) + '.jpg')),
                    cv2.COLOR_BGR2RGB)
                color = np.float32(color / 255.)

                albedo = cv2.cvtColor(
                    cv2.imread(
                        os.path.join(model_path,
                                     'albedo_view_' + str(target_id) + '.jpg')),
                    cv2.COLOR_BGR2RGB)
                albedo = np.float32(albedo / 255.)

                normal = torch.from_numpy(normal_map).permute(
                    2, 0, 1).unsqueeze(0)    # 1,3,h,w
                depth = torch.from_numpy(target_depth).unsqueeze(0).unsqueeze(
                    0)    # 1,h,w
                color = torch.from_numpy(color).permute(2, 0, 1).unsqueeze(
                    0)    # 1,3,h,w
                albedo = torch.from_numpy(albedo).permute(2, 0, 1).unsqueeze(
                    0)    # 1,3,h,w

                stride = np.random.randint(1, 3)
                sample_info = sample_target_view_rgbd(
                    num=self.cfg.diff_samples,
                    mask=depth_mask,
                    depth=depth,
                    color=color,
                    albedo=albedo,
                    normal=normal,
                    sample_outer_rate=0.,
                    patch=self.cfg.data.patch_dim,
                    stride=stride)

                sample_info['sample_R'] = torch.from_numpy(target_R)
                sample_info['sample_t'] = torch.from_numpy(target_t)
                sample_info['sample_intri'] = torch.from_numpy(k)
                data.update({'sample_info': sample_info})
        return data

    def get_model_dict(self, idx):
        return self.models[idx]


if __name__ == '__main__':
    import argparse
    import ray_utils
    import polyscope as ps

    torch.manual_seed(0)
    np.random.seed(0)

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_folder', type=str, help='Path to data folder.')
    args = parser.parse_args()

    cfg = Config()
    cfg.data.pointcloud_n = 20000
    cfg.differential = True
    cfg.data_gen = False
    cfg.live_sample = False
    cfg.hair = False
    cfg.point_num = 35000

    if args.data_folder is not None:
        cfg.data.train_folder = args.data_folder

    val_dataset = THumanDataset(cfg.data.train_folder, cfg=cfg, mode='train')

    # ---------------------------------------------------
    # Visualize point cloud sample

    val_data = val_dataset[0]
    # for val_data in val_dataset:
    pc = val_data['pc'][:, :3]
    center = val_data['center']
    scale = val_data['scale']
    pc = (pc * scale) + center

    pc_color = 0.5 * (val_data['pc'][:, 3:] + 1.)
    Rs = val_data['Rs']
    ts = val_data['ts']

    ps.init()
    pc_vis = ps.register_point_cloud("depth pc",
                                     pc,
                                     radius=0.001,
                                     point_render_mode='quad')
    pc_vis.add_color_quantity('color', pc_color, enabled=True)
    ps.show()

    # ---------------------------------------------------
    # Visualize patch sample

    val_data = val_dataset[0]
    depth_pc = val_data['pc'][:, :3]
    sample_info = val_data['sample_info']

    sampled_normal = sample_info['sampled_normal']
    sampled_rgb = sample_info['sampled_rgb']
    sampled_alb = sample_info['sampled_alb']
    sampled_xy = sample_info['sampled_xy']
    sampled_depth = sample_info['sampled_depth']
    sample_R = sample_info['sample_R'].unsqueeze(0)
    sample_t = sample_info['sample_t'].unsqueeze(0)

    rays_dirs = ray_utils.get_ray_directions_pts(pts=sampled_xy,
                                                 img_h=512,
                                                 img_w=512,
                                                 focal=[550, 550])
    rays_o = torch.einsum('bij,bi->bj', -sample_R, sample_t)
    rays_d = torch.einsum('bni,bij->bnj', rays_dirs, sample_R)
    rays_o = rays_o[:, None, :].repeat_interleave(rays_d.shape[1], dim=1)

    rays_o = rays_o.squeeze(0)
    rays_d = rays_d.squeeze(0)
    sampled_depth = sampled_depth.squeeze(0)

    sample_point_cloud = rays_o + sampled_depth * rays_d
    sampled_depth_mask = (sampled_depth > 0).reshape(-1,)

    sampled_normal = sampled_normal.squeeze(0)[sampled_depth_mask]
    sample_point_cloud = sample_point_cloud[sampled_depth_mask]

    depth_offset = 3e-2
    sample_point_cloud_plus = rays_o + (sampled_depth + depth_offset) * rays_d
    sample_point_cloud_plus = sample_point_cloud_plus[sampled_depth_mask]
    sample_point_cloud_minus = rays_o + (sampled_depth - depth_offset) * rays_d
    sample_point_cloud_minus = sample_point_cloud_minus[sampled_depth_mask]

    ps.init()
    ps.register_point_cloud("depth_pc", depth_pc)
    pc = ps.register_point_cloud("sample_point_cloud",
                                 sample_point_cloud.numpy())
    ps.register_point_cloud("sample_point_cloud_plus",
                            sample_point_cloud_plus.numpy())
    ps.register_point_cloud("sample_point_cloud_minus",
                            sample_point_cloud_minus.numpy())
    pc.add_vector_quantity("sampled_normal", sampled_normal.numpy())
    ps.show()

    # ---------------------------------------------------
    # Visualize reprojection

    val_data = val_dataset[0]
    pc = val_data['pc'][:, :3]

    Rs = val_data['Rs']
    ts = val_data['ts']

    center = val_data['center']
    scale = val_data['scale']

    # Recover center and scale
    pc = pc * scale
    pc += center

    color_imgs = val_data['color_img']
    Ks = val_data['intri']

    pts_cam_view = np.einsum('bni,bji->bnj',
                             np.einsum('ni,bji->bnj', pc, Rs) + ts[:, None, :],
                             Ks)
    pts_cam_view[..., 0] /= pts_cam_view[..., 2]
    pts_cam_view[..., 1] /= pts_cam_view[..., 2]

    for i in range(4):
        img_pts = pts_cam_view[i]
        img = cv2.cvtColor(color_imgs[i], cv2.COLOR_RGB2BGR)
        for pt in img_pts:
            cv2.circle(img, (int(pt[0]), int(pt[1])),
                       radius=1,
                       color=(255, 0, 0),
                       thickness=-1)
        cv2.imshow(f"color_{i}", img)
        cv2.waitKey(0)
