import os
import torch
import torch.nn as nn
import numpy as np
from skimage.measure import marching_cubes
from geo_model import GeoModel
from torch.optim import Adam
from torch.optim.lr_scheduler import CyclicLR
import cv2
from implicit.implicit_prt_gen import fibonacci_sphere, getSHCoeffs
import math
import open3d as o3d
import moderngl
from thuman_renderer import PrtRenderTargetVC, NormalRenderTarget
from pyrr import Matrix44
import vis_fuse_utils
from thuman import filter_visual_hull
from config import Config
import torchvision

from icecream import ic


# https://gist.github.com/alper111/8233cdb0414b4cb5853f2f730ab95a49
class VGGPerceptualLoss(torch.nn.Module):

    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        blocks.append(
            torchvision.models.vgg16(pretrained=True).features[:4].eval())
        blocks.append(
            torchvision.models.vgg16(pretrained=True).features[4:9].eval())
        blocks.append(
            torchvision.models.vgg16(pretrained=True).features[9:16].eval())
        blocks.append(
            torchvision.models.vgg16(pretrained=True).features[16:23].eval())
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.resize = resize
        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self,
                input,
                target,
                feature_layers=[0, 1, 2, 3],
                style_layers=[]):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        input = (input - self.mean) / self.std
        target = (target - self.mean) / self.std
        if self.resize:
            input = self.transform(input,
                                   mode='bilinear',
                                   size=(224, 224),
                                   align_corners=False)
            target = self.transform(target,
                                    mode='bilinear',
                                    size=(224, 224),
                                    align_corners=False)
        loss = 0.0
        x = input
        y = target
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            if i in feature_layers:
                loss += torch.nn.functional.l1_loss(x, y)
            if i in style_layers:
                act_x = x.reshape(x.shape[0], x.shape[1], -1)
                act_y = y.reshape(y.shape[0], y.shape[1], -1)
                gram_x = act_x @ act_x.permute(0, 2, 1)
                gram_y = act_y @ act_y.permute(0, 2, 1)
                loss += torch.nn.functional.l1_loss(gram_x, gram_y)
        return loss


class Trainer:

    def __init__(self, cfg: Config, out_dir=None):
        self.cfg = cfg
        gpus = [eval(id) for id in self.cfg.gpus.split(',')]
        self.gpus = gpus
        self.device = torch.device(
            "cuda:%d" % gpus[0] if torch.cuda.is_available() else "cpu")
        model = GeoModel(cfg)
        if len(gpus) > 1:
            model = torch.nn.DataParallel(model, device_ids=gpus)
        self.model = model.to(self.device)
        # opti
        self.opti = Adam(self.model.parameters(), lr=1e-4)
        self.lr_scheduler = CyclicLR(self.opti,
                                     base_lr=5e-5,
                                     max_lr=5e-4,
                                     step_size_up=200 * 5,
                                     scale_fn=lambda x: 1 /
                                     (2.**((x - 22) // 20)),
                                     cycle_momentum=False)
        self.out_dir = out_dir
        self.bce_no = nn.BCEWithLogitsLoss(reduction='none').to(self.device)
        self.bce_logits = nn.BCEWithLogitsLoss().to(self.device)
        self.l1 = nn.SmoothL1Loss().to(self.device)
        self.l1_no = nn.SmoothL1Loss(reduction='none').to(self.device)
        self.patch_loss = VGGPerceptualLoss().to(self.device)
        self.diff_it = 200 if self.cfg.overfit else 2000

        env_shs = np.load('implicit/env_sh.npy')
        self.default_light_coeff = env_shs[0]
        # Account for MC error: sqrt(3) / 512
        self.query_epsilon = 3e-3

    def train_step(self, data, iters):
        self.model.train()
        loss = self.compute_loss(data, iters)
        loss['total'].backward()

        self.opti.step()
        self.lr_scheduler.step()
        self.opti.zero_grad()

        return loss

    def compute_loss(self, data, it=None):
        device = self.device
        point_num = self.cfg.point_num
        bs, _, h, w = data['depth_img'].shape

        query_pts = data['querys'].float().to(device)
        occ = data['occ'].float().to(device)    # b,N
        color = data['color'].float().permute(0, 2, 1).to(device)
        visibility = data['visibility'].float().permute(0, 2, 1).to(device)
        mesh_sample_normals = data['normals'].float().to(device)

        p_camview_vis = data['pts_cam_vis'].float().to(device)    # b,4,n
        pc = data['pc'].to(device)
        depth_img = data['depth_img'].unsqueeze(2).to(device)    # b,4,1,h,w
        mask_img = data['mask_img_dilate' if self.
                        cfg.mask_dilate else 'mask_img'].unsqueeze(2).to(
                            device)    # b,4,1,h,w
        color_img = data['color_img'].to(device)    # b,4,3,h,w
        color_img = color_img.permute(0, 1, 4, 2, 3)
        Rs = data['Rs'].float().to(device)    # b,v,3,3
        ts = data['ts'].float().to(device)    # b,v,3
        intri = data['intri'].float().to(device)    # B,4,3,3
        c_depth = data['c_depth'].float().to(device)    #  B,4,
        center = data['center'].float().to(device)
        scale = data['scale'].float().to(device)

        if 'normal_img' in data:
            normal_lr = data['normal_img'].to(device)
            normal_lr = normal_lr.permute(0, 1, 4, 2, 3)
            normal_lr = normal_lr.view([bs * self.cfg.data.view_num] +
                                       list(normal_lr.shape[2:]))
        else:
            normal_lr = None
            # surf_normal = None

        depth_img = depth_img.view([bs * self.cfg.data.view_num] +
                                   list(depth_img.shape[2:]))    # BV,1,H,W
        mask_img = mask_img.view([bs * self.cfg.data.view_num] +
                                 list(mask_img.shape[2:]))    # BV,1,H,W
        color_img = color_img.view([bs * self.cfg.data.view_num] +
                                   list(color_img.shape[2:]))
        # p_camview = p_camview.view([bs*view_num]+list(p_camview.shape[2:])) # BV,N,3
        p_camview_vis = p_camview_vis.view(
            [bs * self.cfg.data.view_num] +
            list(p_camview_vis.shape[2:]))    # BV,N

        loss_weight = p_camview_vis.view([bs, self.cfg.data.view_num] +
                                         list(data['pts_cam_vis'].shape[2:]))
        loss_weight = loss_weight.sum(dim=1)
        loss_weight[torch.logical_not(
            loss_weight.isclose(torch.tensor([0.], device=self.device)))] = 1
        loss_total = {'total': 0}

        differential_condition = self.cfg.differential and it > self.diff_it
        if differential_condition:
            target_info = data['sample_info']
            target_R = target_info['sample_R'].to(device).float()
            target_t = target_info['sample_t'].to(device).float()
            target_intri = target_info['sample_intri'].to(device)    #  b,3,3
            target_xy = target_info['sampled_xy'].squeeze(1).to(
                device)    # b,n,2
            target_alb = target_info['sampled_alb'].squeeze(1).to(
                device)    # b,n,3
            # target_clr = target_info['sampled_rgb'].squeeze(1).to(
            #     device)    # b,n,3
            # target_normal = target_info['sampled_normal'].squeeze(1).to(
            #     device)    # b,n,3
            target_depth = target_info['sampled_depth'].squeeze(1).to(
                device)    # b,n,1

        if not differential_condition:
            outputs = self.model(pc,
                                 query_pts,
                                 color=color_img,
                                 depth=depth_img,
                                 mask=mask_img,
                                 Rs=Rs,
                                 ts=ts,
                                 intri=intri,
                                 c_depth=c_depth,
                                 center=center,
                                 scale=scale)
        else:
            outputs = self.model(pc,
                                 query_pts,
                                 color=color_img,
                                 depth=depth_img,
                                 mask=mask_img,
                                 Rs=Rs,
                                 ts=ts,
                                 intri=intri,
                                 c_depth=c_depth,
                                 center=center,
                                 scale=scale,
                                 sample_pix=target_xy,
                                 sample_intri=target_intri,
                                 sample_depth=target_depth,
                                 sample_R=target_R,
                                 sample_t=target_t,
                                 its=it)

        if 'occ' in outputs:
            out_around = outputs['occ'][:, 0, :point_num]
            loss_around = self.bce_no(out_around, occ)
            loss_total['occ'] = (loss_weight[:, :point_num] * loss_around
                                ).sum() / loss_weight[:, :point_num].sum()
            loss_total['total'] += loss_total['occ']

        color_point_num = int(0.4 * point_num)
        if 'color' in outputs:
            out_color = outputs['color']
            loss_total['color'] = self.l1(out_color[..., :color_point_num],
                                          color[..., :color_point_num])
            loss_total['total'] += loss_total['color']

            loss_total['sur_color'] = self.l1(out_color[..., point_num:],
                                              color[..., color_point_num:])
            loss_total['total'] += loss_total['sur_color']

        if 'vis' in outputs:
            out_visibility = outputs['vis']
            vis_query = out_visibility[..., :point_num]
            vis_query_gt = visibility[..., :point_num]

            loss_visibility = self.bce_logits(vis_query, vis_query_gt)
            loss_total['vis'] = loss_visibility
            loss_total['total'] += loss_total['vis']

            vis_mesh = out_visibility[..., point_num:]
            vis_mesh_gt = visibility[..., point_num:]

            vis_mesh = vis_mesh.permute(0, 2, 1)
            vis_mesh_gt = vis_mesh_gt.permute(0, 2, 1)

            loss_total['sur_vis'] = self.bce_logits(vis_mesh, vis_mesh_gt)
            loss_total['total'] += loss_total['sur_vis']

            dirs, phi, theta = fibonacci_sphere(self.cfg.vis_sample_size)
            SH = getSHCoeffs(2, phi, theta)

            dirs = torch.from_numpy(dirs).float().to(device)
            SH = torch.from_numpy(SH).float().to(device)

            geo_term = torch.clip(
                torch.einsum("bik,jk->bij", mesh_sample_normals, dirs), 0, 1)
            PRT_pred = torch.einsum(
                "bij,bij,jk->bik", torch.sigmoid(vis_mesh), geo_term,
                SH) * 4.0 * math.pi / self.cfg.vis_sample_size
            PRT_gt = torch.einsum("bij,bij,jk->bik", vis_mesh_gt, geo_term,
                                  SH) * 4.0 * math.pi / self.cfg.vis_sample_size

            loss_total['sur_prt'] = self.l1(PRT_pred, PRT_gt)
            loss_total['total'] += loss_total['sur_prt']

        if 'normal_2d' in outputs:
            normals_out = outputs['normal_2d']
            if it % 2000 == 0:
                with torch.no_grad():
                    save_out = normals_out.detach() * mask_img
                    save_out = save_out[1].permute(1, 2, 0).cpu().numpy()
                    save_out = (save_out + 1) * 0.5
                    save_gt = normal_lr * mask_img
                    save_gt = save_gt[1].permute(1, 2, 0).cpu().numpy()
                    save_gt = (save_gt + 1) * 0.5
                    save_normal = np.concatenate([save_gt, save_out], axis=1)
                    cv2.imwrite(
                        '%s/normal_%03d.jpg' % (self.out_dir, it // 2000),
                        save_normal[..., ::-1] * 255)

            normal_loss = self.l1_no(normal_lr, normals_out)
            normal_loss = torch.sum(normal_loss, 1, keepdim=True)
            normal_loss = (normal_loss * mask_img).sum() / mask_img.sum()
            loss_total['normal_2d'] = normal_loss
            loss_total['total'] += loss_total['normal_2d']

        if 'sampled_albedo' in outputs:
            mask_valid = outputs['sampled_mask']

            patch_vis_iter = self.cfg.training.visualize_every

            def vis_patch(predict_patch, target_patch, tag):
                vis_out_patch = np.ones((512, 512, 3))
                vis_gt_patch = np.ones((512, 512, 3))
                with torch.no_grad():
                    pix_xy = target_xy[0].detach().cpu().numpy()
                    pix_xy = (pix_xy + 1) * 0.5 * 512
                    pix_xy = np.clip(pix_xy, 0, 512)
                    out_alb_debug = predict_patch[0].detach().cpu().numpy()
                    target_alb_debug = target_patch[0].detach().cpu().numpy()
                    for j in range(pix_xy.shape[0]):
                        vis_out_patch[int(pix_xy[j, 1]),
                                      int(pix_xy[j,
                                                 0]), :] = out_alb_debug[j, :]
                        vis_gt_patch[int(pix_xy[j, 1]),
                                     int(pix_xy[j,
                                                0]), :] = target_alb_debug[j, :]
                    save_patch = np.concatenate([vis_out_patch, vis_gt_patch],
                                                axis=1)

                    patch_save_path = os.path.join(
                        self.out_dir,
                        f"patch_{tag}_iters_%03d.jpg" % (it // patch_vis_iter))
                    cv2.imwrite(patch_save_path, save_patch[:, :, ::-1] * 255)

            def compute_patch_loss(predict_patch, target_patch, tag):
                target_patch[torch.logical_not(mask_valid)] = 0.

                if tag is not None:
                    vis_patch(predict_patch, target_patch, tag)

                patch_dim = self.cfg.data.patch_dim
                predict_patch = predict_patch.reshape(-1, patch_dim, patch_dim,
                                                      3)
                target_patch = target_patch.reshape(-1, patch_dim, patch_dim, 3)

                predict_patch = predict_patch.permute(0, 3, 1, 2)
                target_patch = target_patch.permute(0, 3, 1, 2)

                loss_l1 = self.l1_no(predict_patch,
                                     target_patch).sum() / mask_valid.sum()
                loss_patch = self.patch_loss(predict_patch, target_patch)
                return loss_l1 + loss_patch

            sampled_albedo = outputs['sampled_albedo']
            loss_total['patch_alb'] = compute_patch_loss(
                sampled_albedo, target_alb,
                "alb" if it % patch_vis_iter == 0 else None)
            loss_total['total'] += loss_total['patch_alb']

        return loss_total

    def eval_step(self, data, out_path, mesh_name, nx=256, save=False):
        self.model.eval()

        ctx = moderngl.create_context(standalone=True, backend='egl')
        ctx.enable(moderngl.DEPTH_TEST)
        ctx.gc_mode = "auto"
        prt_vc_target = PrtRenderTargetVC(ctx, 512, 512)
        normal_target = NormalRenderTarget(ctx, 512, 512)

        device = self.device
        depth_img = data['depth_img'].unsqueeze(2).squeeze(0).to(
            device)    # 4,1,h,w
        mask_img = data['mask_img_dilate' if self.cfg.
                        mask_dilate else 'mask_img'].unsqueeze(2).squeeze(
                            0).float().to(device)    # 4,1,h,w
        mask_img_dilate = data['mask_img_dilate'].unsqueeze(2).squeeze(
            0).float().to(device)
        color_img = data['color_img'].squeeze(0).permute(0, 3, 1, 2).to(
            device)    # 4,3,h,w
        Rs = data['Rs'].float().to(device)    # b,v,3,3
        ts = data['ts'].float().to(device)    # b,v,3
        intri = data['intri'].float().to(device)    # b,v,3,3
        c_depth = data['c_depth'].float().to(device)    #  b,v

        pc = data['pc'].to(device)    # 1,n,3
        center = data['center'].float()
        scale = data['scale'].float()

        _, _, H, W = depth_img.shape

        indices = torch.arange(nx)
        grid_indices = torch.stack(torch.meshgrid(indices, indices,
                                                  indices)).reshape(3, -1).T
        pointsf = grid_indices / (nx - 1) - 0.5

        pointsf = pointsf * scale.squeeze(0)
        pointsf += center.squeeze(0)

        filter_results = [
            filter_visual_hull(pt_group, Rs, ts, intri, mask_img_dilate, device,
                               False, True)
            for pt_group in torch.split(pointsf, 10 * self.cfg.test.group_size)
        ]
        pointsf = torch.cat([res[0] for res in filter_results])
        valid_mask = torch.cat([res[1] for res in filter_results])

        def save_batched_img(batched_img,
                             tag,
                             to_uint8=True,
                             is_numpy=False,
                             with_mask=True,
                             is_normal=False):
            batch_size = batched_img.shape[0]
            if batch_size > 1:
                if with_mask:
                    batched_mask = torch.hstack([
                        mask.permute(1, 2, 0) for mask in mask_img
                    ]).detach().cpu().numpy()
                else:
                    batched_mask = None
                if is_numpy:
                    batched_img = np.hstack([image for image in batched_img])
                else:
                    batched_img = torch.hstack([
                        image.permute(1, 2, 0) for image in batched_img
                    ]).detach().cpu().numpy()
            else:
                if with_mask:
                    batched_mask = mask_img[0].permute(
                        1, 2, 0).detach().cpu().numpy()
                else:
                    batched_mask = None
                if is_numpy:
                    batched_img = batched_img[0]
                else:
                    batched_img = batched_img[0].permute(
                        1, 2, 0).detach().cpu().numpy()

            if to_uint8:
                if is_normal:
                    batched_img_mask = batched_img == 0
                    batched_img = np.uint8(
                        255 * np.clip(0.5 * (batched_img + 1), 0, 1))
                    batched_img[batched_img_mask] = np.uint8(0)
                else:
                    batched_img = np.uint8(255 * np.clip(batched_img, 0, 1))

            if with_mask:
                batched_img = batched_mask * batched_img

            cv2.imwrite(os.path.join(out_path, f'{mesh_name}_{tag}.png'),
                        cv2.cvtColor(batched_img, cv2.COLOR_RGB2BGR))

        model = self.model.module if isinstance(
            self.model, torch.nn.DataParallel) else self.model

        decode = model.fea_encode(pc,
                                  color_img,
                                  depth_img,
                                  mask_img,
                                  Rs,
                                  ts,
                                  intri.to(device),
                                  c_depth.to(device),
                                  center=center.to(device),
                                  scale=scale.to(device))

        occ = decode(p_q=pointsf, infer_occ=True, infer_color=False)['occ']

        print(f"Occupancy count: {(occ > 0).sum()}")

        value_grid = torch.zeros(nx * nx * nx)
        value_grid[valid_mask] = occ
        value_grid = torch.reshape(value_grid, (nx, nx, nx)).numpy()

        voxel_size = 1. / nx
        try:
            vertices, faces, normals, _ = marching_cubes(value_grid,
                                                         0.,
                                                         spacing=(voxel_size,
                                                                  voxel_size,
                                                                  voxel_size))
        except Exception as e:
            print("No surface detected")
            return

        vertices = vertices - np.ones((1, 3)) * (1. / 2 - voxel_size / 2)
        vertices = vertices * scale.item()
        vertices += center.squeeze(0).numpy()
        vertices = np.float32(vertices)
        faces = faces[:, (1, 0, 2)]    # anti-clockwise triangle face indices
        vertex_color = None

        aabb_min = np.min(vertices, axis=0)
        aabb_max = np.max(vertices, axis=0)
        delta = self.query_epsilon * np.min(aabb_max - aabb_min)
        vertex_query_point = vertices + delta * normals

        vertices_vis_infer = torch.from_numpy(vertex_query_point).float().to(
            device)

        vis = decode(p_q=vertices_vis_infer, infer_occ=False,
                     infer_color=False)['vis']
        vis = torch.sigmoid(vis).numpy() > 0.5

        vertices_color_infer = torch.from_numpy(vertices).float().to(device)

        vertex_color = decode(p_q=vertices_color_infer,
                              infer_occ=False,
                              infer_color=True)['color']
        if vertex_color is not None:
            vertex_color = vertex_color.numpy()

        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        mesh.triangles = o3d.utility.Vector3iVector(faces)
        mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_color)

        if not os.path.exists(out_path):
            os.makedirs(out_path)
        mesh_path = os.path.join(out_path, f"{mesh_name}.obj")
        o3d.io.write_triangle_mesh(mesh_path, mesh)

        order = 2
        dirs, phi, theta = fibonacci_sphere(self.cfg.vis_sample_size)
        SH = getSHCoeffs(order, phi, theta)

        geo_term = np.clip(np.einsum("ik,jk->ij", normals, dirs), 0, 1)
        PRT = np.einsum("ij,ij,jk->ik", vis, geo_term,
                        SH) * 4.0 * math.pi / self.cfg.vis_sample_size

        if save:
            viz_gt = np.logical_not(
                vis_fuse_utils.sample_occlusion_embree(
                    vertices, faces, vertices + self.query_epsilon * normals,
                    dirs))
            PRT_gt = np.einsum("ij,ij,jk->ik", viz_gt, geo_term,
                               SH) * 4.0 * math.pi / self.cfg.vis_sample_size

            np.save(os.path.join(out_path, "vis_gen.npy"), vis)
            np.save(os.path.join(out_path, "prt_gen.npy"), PRT)
            np.save(os.path.join(out_path, "prt_gt.npy"), PRT_gt)
            np.save(os.path.join(out_path, "vertex_color.npy"), vertex_color)

        width = 512

        per_face_vertices = vertices[faces].reshape(-1, 3)
        per_face_vc = vertex_color[faces].reshape(-1, 3)
        per_face_prt = PRT[faces].reshape(-1, 9)
        per_face_normal = normals[faces].reshape(-1, 3)

        prt_vc_target.build_vao(per_face_vertices, per_face_vc, per_face_prt)
        prt_target = prt_vc_target
        normal_target.build_vao(per_face_vertices, per_face_normal)

        T_gl_cv = np.eye(4)
        T_gl_cv[1, 1] = -1
        T_gl_cv[2, 2] = -1

        light_coeff = self.default_light_coeff
        sh = light_coeff.T.reshape(-1,).astype('f4')

        light_images = []
        color_images = []
        albedo_images = []
        normal_images = []

        for i in range(self.cfg.data.view_num):
            R = Rs[0, i].detach().cpu().numpy()
            t = ts[0, i].detach().cpu().numpy()

            fx = intri[0, i, 0, 0]
            fov = math.degrees(2 * math.atan(0.5 * width / fx))
            proj = Matrix44.perspective_projection(fov, 1.0, 0.01, 100.0)

            view = np.eye(4)
            view[:3, :3] = R
            view[:3, 3] = t
            view = T_gl_cv @ view

            mv = np.ascontiguousarray(view.T).astype('f4')
            mvp = (proj * view.T).astype('f4')
            image_light, image_color, image_albedo = prt_target.render(mvp, sh)
            image_normal, _ = normal_target.render(mv, mvp)

            light_images.append(image_light)
            color_images.append(image_color)
            albedo_images.append(image_albedo)
            normal_images.append(image_normal)

        light_images = np.stack(light_images)
        color_images = np.stack(color_images)
        albedo_images = np.stack(albedo_images)
        normal_images = np.stack(normal_images)

        save_batched_img(light_images,
                         'render_light',
                         is_numpy=True,
                         to_uint8=False,
                         with_mask=False)
        save_batched_img(color_images,
                         'render_color',
                         is_numpy=True,
                         to_uint8=False,
                         with_mask=False)
        save_batched_img(albedo_images,
                         'render_albedo',
                         is_numpy=True,
                         to_uint8=False,
                         with_mask=False)
        save_batched_img(normal_images,
                         'render_normal',
                         is_numpy=True,
                         to_uint8=True,
                         with_mask=False,
                         is_normal=True)

    def print_params(self):
        nparameters = sum(p.numel() for p in self.model.parameters())
        print('Total number of parameters: %d' % nparameters)

    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad
