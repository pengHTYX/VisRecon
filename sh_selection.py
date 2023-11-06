import argparse
import moderngl
from thuman_renderer import PrtRenderTargetVC
import numpy as np
import math
from implicit.implicit_prt_gen import fibonacci_sphere, getSHCoeffs, compute_sample_occlusion
from implicit.implicit_render_prt import get_view_matrix, rotateSH
from pyrr import Matrix44, matrix33
from icecream import ic
import trimesh
import cv2
from itertools import combinations, permutations
import os
from tqdm import tqdm

np.random.seed(0)


class ColorSphereRender:

    def __init__(self,
                 width=512,
                 height=512,
                 ortho_ratio=0.7,
                 vis_sample_size=64):
        self.ctx = moderngl.create_context(standalone=True, backend='egl')
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.gc_mode = "auto"

        self.env_shs = np.load('implicit/env_sh.npy')
        ic(self.env_shs.shape)

        self.prt_vc_target = PrtRenderTargetVC(self.ctx, width, height)

        self.proj = Matrix44.orthogonal_projection(-ortho_ratio, ortho_ratio,
                                                   -ortho_ratio, ortho_ratio,
                                                   0.01, 100.0)
        order = 2
        dirs, phi, theta = fibonacci_sphere(vis_sample_size)
        SH = getSHCoeffs(order, phi, theta)

        sphere_mesh: trimesh.Trimesh = trimesh.Trimesh(
            0.5 * fibonacci_sphere(256)[0], np.load('implicit/face_256.npy'))
        # assume white color
        sphere_vertex_color = np.ones_like(sphere_mesh.vertices)

        mesh_samples = sphere_mesh.vertices + 3e-4 * sphere_mesh.vertex_normals
        mesh_sample_vis = compute_sample_occlusion(sphere_mesh.vertices,
                                                   sphere_mesh.faces,
                                                   mesh_samples)
        geo_term = np.clip(
            np.einsum("ik,jk->ij", sphere_mesh.vertex_normals, dirs), 0, 1)
        PRT = np.einsum("ij,ij,jk->ik", np.float64(mesh_sample_vis), geo_term,
                        SH) * 4.0 * math.pi / vis_sample_size

        per_face_vertices = sphere_mesh.vertices[sphere_mesh.faces].reshape(
            -1, 3)
        per_face_vc = sphere_vertex_color[sphere_mesh.faces].reshape(-1, 3)
        per_face_prt = PRT[sphere_mesh.faces].reshape(-1, 9)

        self.prt_vc_target.build_vao(per_face_vertices, per_face_vc,
                                     per_face_prt)

    def render_color_orbit(self, sh_idx, image_render_count):
        sh_rot_factor = np.random.randint(0, 30)
        R_sh = matrix33.create_from_eulers(
            [-0.5 * sh_rot_factor, 0.1 * sh_rot_factor, -sh_rot_factor])
        sh = rotateSH(np.copy(self.env_shs[sh_idx]), R_sh)

        # https://en.wikipedia.org/wiki/Grayscale
        gray_scale = 0.2126 * sh[:, 0] + 0.7152 * sh[:, 1] + 0.0722 * sh[:, 2]
        var = 1 + 0.025 * np.random.randn(3)
        sh = gray_scale[:, None].repeat(3, -1) * var[None, :]
        # sh = gray_scale[:, None].repeat(3, -1)
        sh_render = sh.T.reshape(-1,).astype('f4')
        delta_radians = 2 * np.pi / image_render_count

        image_color_list = []
        for i in range(image_render_count):
            view_angle = i * delta_radians
            eye = [math.cos(-view_angle), 0.0, math.sin(-view_angle)]
            view = get_view_matrix(eye)
            mvp = (self.proj * view).astype('f4')

            _, image_color, _ = self.prt_vc_target.render(mvp,
                                                          sh_render,
                                                          alpha=True)
            image_color_list.append(image_color)

        return sh, np.stack(image_color_list)


if __name__ == '__main__':
    # This script is used to filter out environment lightings that are too dramatic
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_folder',
                        type=str,
                        default="$HOME/dataset/sh_selection",
                        help='Path to data folder.')
    args = parser.parse_args()

    save_folder = args.save_folder
    save_folder = os.path.expandvars(save_folder)

    image_render_count = 12

    perms = np.array(list(permutations(np.arange(image_render_count),
                                       2))).reshape(image_render_count,
                                                    image_render_count - 1, 2)

    comb_list = list(combinations(np.arange(image_render_count), 2))

    sphere_renderer = ColorSphereRender()

    sh_list = []
    loop_size = 200
    for loop_idx in tqdm(range(loop_size)):
        sh_idx = loop_idx % len(sphere_renderer.env_shs)
        sh, color_imgs = sphere_renderer.render_color_orbit(
            sh_idx, image_render_count)

        deviation_list = []
        for perm_list in perms:
            psnr_list = []
            for perm in perm_list:
                im_1 = color_imgs[perm[0]]
                im_2 = color_imgs[perm[1]]

                mask = (im_1[..., -1] > 0)
                mse = ((im_1 / 255. - im_2 / 255.)**2 *
                       mask[..., None]).sum() / mask.sum() / 3.
                psnr = 10. * np.log10(1 / mse)
                psnr_list.append(psnr)
            deviation = np.max(psnr_list) - np.min(psnr_list)
            deviation_list.append(deviation)

        devs = np.array(deviation_list)
        selection_criterial = np.max(devs)

        if selection_criterial < 8:
            cv2.imwrite(
                os.path.join(save_folder, f"{str(len(sh_list)).zfill(3)}.png"),
                np.hstack(color_imgs))
            sh_list.append(sh)

    grayish_env_shs = np.array(sh_list)
    ic(grayish_env_shs.shape)
    np.save('implicit/grayish_env_shs.npy', grayish_env_shs)
