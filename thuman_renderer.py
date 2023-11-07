import os
import numpy as np
import math
import moderngl
from pyrr import Matrix44, matrix33
from PIL import Image
import argparse
from view_renderer import import_mesh
from scipy.io import loadmat, savemat
from thuman import load_cam
import random
from implicit.implicit_prt_gen import fibonacci_sphere, getSHCoeffs
from implicit.implicit_render_prt import rotateSH
import vis_fuse_utils
import cv2
import hashlib
from joblib import Parallel, delayed
from icecream import ic

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"


class PrtRenderTargetUV:

    def __init__(self,
                 ctx: moderngl.Context,
                 width,
                 height,
                 program_path='implicit/shaders/prt_uv'):
        vert = open(f"{program_path}.vs").read()
        frag = open(f"{program_path}.fs").read()
        self.ctx = ctx
        self.prog = ctx.program(vertex_shader=vert, fragment_shader=frag)

        color_attachments = [
            ctx.renderbuffer((width, height), 4, samples=8),
            ctx.renderbuffer((width, height), 4, samples=8),
            ctx.renderbuffer((width, height), 4, samples=8)
        ]
        self.fbo = ctx.framebuffer(color_attachments=color_attachments,
                                   depth_attachment=ctx.depth_renderbuffer(
                                       (width, height), samples=8))
        color_attachments2 = [
            ctx.texture((width, height), 4),
            ctx.texture((width, height), 4),
            ctx.texture((width, height), 4)
        ]
        self.fbo2 = ctx.framebuffer(color_attachments=color_attachments2,
                                    depth_attachment=ctx.depth_renderbuffer(
                                        (width, height)))
        self.vao = None

    def build_vao(self, per_face_vertices, per_face_uv, per_face_prt,
                  texture_image):
        vbo_vert = self.ctx.buffer(per_face_vertices.astype('f4'))
        vbo_uv = self.ctx.buffer(per_face_uv.astype('f4'))
        vbo_prt = self.ctx.buffer(per_face_prt.astype('f4'))

        self.vao = self.ctx.vertex_array(self.prog, [(vbo_vert, '3f', 'in_pos'),
                                                     (vbo_uv, '2f', 'in_uv'),
                                                     (vbo_prt, '9f', 'in_sh')])
        self.texture = self.ctx.texture(texture_image.size, 3,
                                        texture_image.tobytes())
        self.texture.build_mipmaps()

    def render(self, mvp, sh):
        self.prog['mvp'].write(mvp)
        self.prog['env_sh'].write(sh)

        self.fbo.use()
        self.fbo.clear(red=1., green=1., blue=1.)
        self.texture.use()
        self.vao.render()

        self.ctx.copy_framebuffer(self.fbo2, self.fbo)

        light_color_data = self.fbo2.read(components=3, attachment=0)
        image_light_color = Image.frombytes('RGB', self.fbo2.size,
                                            light_color_data).transpose(
                                                Image.Transpose.FLIP_TOP_BOTTOM)

        color_data = self.fbo2.read(components=3, attachment=1)
        image_color = Image.frombytes('RGB', self.fbo2.size,
                                      color_data).transpose(
                                          Image.Transpose.FLIP_TOP_BOTTOM)

        albedo_data = self.fbo2.read(components=3, attachment=2)
        image_albedo = Image.frombytes('RGB', self.fbo2.size,
                                       albedo_data).transpose(
                                           Image.Transpose.FLIP_TOP_BOTTOM)

        return np.array(image_light_color), np.array(image_color), np.array(
            image_albedo)


class PrtRenderTargetVC:

    def __init__(self,
                 ctx: moderngl.Context,
                 width,
                 height,
                 program_path='implicit/shaders/prt_vc_simple'):
        vert = open(f"{program_path}.vs").read()
        frag = open(f"{program_path}.fs").read()
        self.ctx = ctx
        self.prog = ctx.program(vertex_shader=vert, fragment_shader=frag)

        color_attachments = [
            ctx.renderbuffer((width, height), 4, samples=8),
            ctx.renderbuffer((width, height), 4, samples=8),
            ctx.renderbuffer((width, height), 4, samples=8)
        ]
        self.fbo = ctx.framebuffer(color_attachments=color_attachments,
                                   depth_attachment=ctx.depth_renderbuffer(
                                       (width, height), samples=8))
        color_attachments2 = [
            ctx.texture((width, height), 4),
            ctx.texture((width, height), 4),
            ctx.texture((width, height), 4)
        ]
        self.fbo2 = ctx.framebuffer(color_attachments=color_attachments2,
                                    depth_attachment=ctx.depth_renderbuffer(
                                        (width, height)))
        self.vao = None

    def build_vao(self, per_face_vertices, per_face_vc, per_face_prt):
        vbo_vert = self.ctx.buffer(per_face_vertices.astype('f4'))
        vbo_vc = self.ctx.buffer(per_face_vc.astype('f4'))
        vbo_prt = self.ctx.buffer(per_face_prt.astype('f4'))

        self.vao = self.ctx.vertex_array(self.prog,
                                         [(vbo_vert, '3f', 'in_pos'),
                                          (vbo_vc, '3f', 'in_vert_color'),
                                          (vbo_prt, '9f', 'in_sh')])

    def render(self, mvp, sh, alpha=False):
        self.prog['mvp'].write(mvp)
        self.prog['env_sh'].write(sh)

        self.fbo.use()
        self.fbo.clear(red=1., green=1., blue=1.)
        self.vao.render()

        self.ctx.copy_framebuffer(self.fbo2, self.fbo)

        if alpha:
            mode = 'RGBA'
            components = 4
        else:
            mode = 'RGB'
            components = 3

        light_color_data = self.fbo2.read(components=components, attachment=0)
        image_light_color = Image.frombytes(mode, self.fbo2.size,
                                            light_color_data).transpose(
                                                Image.Transpose.FLIP_TOP_BOTTOM)

        color_data = self.fbo2.read(components=components, attachment=1)
        image_color = Image.frombytes(mode, self.fbo2.size,
                                      color_data).transpose(
                                          Image.Transpose.FLIP_TOP_BOTTOM)

        albedo_data = self.fbo2.read(components=components, attachment=2)
        image_albedo = Image.frombytes(mode, self.fbo2.size,
                                       albedo_data).transpose(
                                           Image.Transpose.FLIP_TOP_BOTTOM)

        return np.array(image_light_color), np.array(image_color), np.array(
            image_albedo)


class DepthRenderTarget:

    def __init__(self,
                 ctx: moderngl.Context,
                 width,
                 height,
                 program_path='implicit/shaders/depth'):
        vert = open(f"{program_path}.vs").read()
        frag = open(f"{program_path}.fs").read()
        self.ctx = ctx
        self.prog = ctx.program(vertex_shader=vert, fragment_shader=frag)

        color_attachments = [ctx.texture((width, height), 1, dtype='f4')]
        self.fbo = ctx.framebuffer(color_attachments=color_attachments,
                                   depth_attachment=ctx.depth_renderbuffer(
                                       (width, height)))
        self.vao = None

    def build_vao(self, per_face_vertices):
        vbo_vert = self.ctx.buffer(per_face_vertices.astype('f4'))

        self.vao = self.ctx.vertex_array(self.prog,
                                         [(vbo_vert, '3f', 'in_pos')])

    def render(self, mv, mvp):
        self.prog['mv'].write(mv)
        self.prog['mvp'].write(mvp)

        self.fbo.use()
        self.fbo.clear()
        self.vao.render()

        depth_data = self.fbo.read(components=1, attachment=0, dtype='f4')
        image_depth = Image.frombytes('F', self.fbo.size, depth_data).transpose(
            Image.Transpose.FLIP_TOP_BOTTOM)

        return np.array(image_depth)


class NormalRenderTarget:

    def __init__(self,
                 ctx: moderngl.Context,
                 width,
                 height,
                 program_path='implicit/shaders/normal'):
        vert = open(f"{program_path}.vs").read()
        frag = open(f"{program_path}.fs").read()
        self.ctx = ctx
        self.prog = ctx.program(vertex_shader=vert, fragment_shader=frag)

        color_attachments = [
            ctx.texture((width, height), 3, dtype='f4'),
            ctx.texture((width, height), 3, dtype='f4')
        ]
        self.fbo = ctx.framebuffer(color_attachments=color_attachments,
                                   depth_attachment=ctx.depth_renderbuffer(
                                       (width, height)))
        self.vao = None

    def build_vao(self, per_face_vertices, per_face_vertex_normals):
        vbo_vert = self.ctx.buffer(per_face_vertices.astype('f4'))
        vbo_vert_normal = self.ctx.buffer(per_face_vertex_normals.astype('f4'))

        self.vao = self.ctx.vertex_array(
            self.prog, [(vbo_vert, '3f', 'in_pos'),
                        (vbo_vert_normal, '3f', 'in_vert_normal')])

    def render(self, mv, mvp):
        self.prog['mv'].write(mv)
        self.prog['mvp'].write(mvp)

        self.fbo.use()
        self.fbo.clear()
        self.vao.render()

        local_normal_data = self.fbo.read(components=3,
                                          attachment=0,
                                          dtype='f4')
        image_local_normal = np.frombuffer(local_normal_data,
                                           dtype='f4').reshape(
                                               self.fbo.height, self.fbo.width,
                                               3)
        image_local_normal = np.flip(image_local_normal, axis=0)

        normal_data = self.fbo.read(components=3, attachment=1, dtype='f4')
        image_normal = np.frombuffer(normal_data, dtype='f4').reshape(
            self.fbo.height, self.fbo.width, 3)
        image_normal = np.flip(image_normal, axis=0)

        return image_local_normal, image_normal


class VisTarget:

    def __init__(self,
                 ctx: moderngl.Context,
                 width,
                 height,
                 program_path='implicit/shaders/vis'):
        vert = open(f"{program_path}.vs").read()
        frag = open(f"{program_path}.fs").read()
        self.ctx = ctx
        self.prog = ctx.program(vertex_shader=vert, fragment_shader=frag)

        color_attachments = [
            ctx.texture((width, height), 3, dtype='f4'),
            ctx.texture((width, height), 3, dtype='f4'),
            ctx.texture((width, height), 3, dtype='f4')
        ]
        self.fbo = ctx.framebuffer(color_attachments=color_attachments,
                                   depth_attachment=ctx.depth_renderbuffer(
                                       (width, height)))
        self.vao = None

    def build_vao(self, per_face_vertices, per_face_prt):
        vbo_vert = self.ctx.buffer(per_face_vertices.astype('f4'))
        vbo_prt = self.ctx.buffer(per_face_prt.astype('f4'))

        self.vao = self.ctx.vertex_array(self.prog, [(vbo_vert, '3f', 'in_pos'),
                                                     (vbo_prt, '9f', 'in_sh')])

    def render(self, mvp):
        self.prog['mvp'].write(mvp)

        self.fbo.use()
        self.fbo.clear()
        self.vao.render()

        coeff_0_data = self.fbo.read(components=3, attachment=0, dtype='f4')
        coeff_0 = np.frombuffer(coeff_0_data,
                                dtype='f4').reshape(self.fbo.height,
                                                    self.fbo.width, 3)
        coeff_1_data = self.fbo.read(components=3, attachment=1, dtype='f4')
        coeff_1 = np.frombuffer(coeff_1_data,
                                dtype='f4').reshape(self.fbo.height,
                                                    self.fbo.width, 3)
        coeff_2_data = self.fbo.read(components=3, attachment=2, dtype='f4')
        coeff_2 = np.frombuffer(coeff_2_data,
                                dtype='f4').reshape(self.fbo.height,
                                                    self.fbo.width, 3)

        image_vis = np.concatenate([coeff_0, coeff_1, coeff_2], axis=-1)
        image_vis = np.flip(image_vis, axis=0)
        return image_vis


class HdriTarget:

    def __init__(self,
                 ctx: moderngl.Context,
                 height,
                 program_path='implicit/shaders/hdri'):
        vert = open(f"{program_path}.vs").read()
        frag = open(f"{program_path}.fs").read()
        self.ctx = ctx
        self.prog = ctx.program(vertex_shader=vert, fragment_shader=frag)

        color_attachments = [ctx.texture((2 * height, height), 4, dtype='f4')]
        self.fbo = ctx.framebuffer(color_attachments=color_attachments)

        vbo_vert_screen_space = ctx.buffer(
            np.array([[-1.0, -1.0], [3.0, -1.0], [-1.0, 3.0]]).astype('f4'))
        vbo_uv_screen_space = ctx.buffer(
            np.array([[0.0, 1.0], [2.0, 1.0], [0.0, -1.0]]).astype('f4'))
        ibo_screen_space = ctx.buffer(np.array([0, 1, 2]).astype('i4'))

        self.vao = ctx.vertex_array(self.prog,
                                    [(vbo_vert_screen_space, '2f', 'in_pos'),
                                     (vbo_uv_screen_space, '2f', 'in_uv')],
                                    ibo_screen_space)

    def render(self, sh, R):
        self.prog['env_sh'].write(sh)
        self.prog['R'].write(R)

        self.fbo.use()
        self.fbo.clear()
        self.vao.render()

        hdri_data = self.fbo.read(components=4, attachment=0, dtype='f4')
        hdri = np.frombuffer(hdri_data,
                             dtype='f4').reshape(self.fbo.height,
                                                 self.fbo.width, 4)
        hdri = np.flip(hdri, axis=0)
        return hdri


# Kinect v2 for Mobile Robot Navigation:Evaluation and Modeling
def add_depth_noise(image_normal, image_depth, depth_mask):
    max_noise_value_front = 10    # mm
    max_noise_value_back = 100    # mm
    model_scale = 0.5

    # w x h x 3
    theta = np.arccos(image_normal[depth_mask] @ np.array([0., 0., 1.]))
    z = -image_depth[depth_mask] / model_scale
    axial_sigma = 1.5 - 0.5 * z + 0.3 * z * z + 0.1 * np.sqrt(
        z * z * z) * (theta * theta) / np.power(np.pi / 2. - theta, 2)
    axial = np.random.randn(len(axial_sigma)) * axial_sigma
    axial[axial < -max_noise_value_front] = -max_noise_value_front
    axial[axial > max_noise_value_back] = max_noise_value_back

    lateral_sigma = 1.6
    lateral = np.random.randn(len(axial_sigma)) * lateral_sigma

    depth_with_noise = -(z + axial / 1000. + lateral / 1000.)
    scale_factor = -depth_with_noise / z

    image_depth[depth_mask] *= scale_factor


def render_views(model_name, dataset_folder, env_shs, width=512, height=512):
    # moderngl context cannot be pickled
    ctx = moderngl.create_context(standalone=True, backend='egl')
    ctx.enable(moderngl.DEPTH_TEST)
    ctx.gc_mode = "auto"

    prt_uv_target = PrtRenderTargetUV(ctx, width, height)
    prt_uv_cos_target = PrtRenderTargetUV(ctx, width, height)
    depth_target = DepthRenderTarget(ctx, width, height)
    normal_target = NormalRenderTarget(ctx, width, height)
    vis_target = VisTarget(ctx, width, height)
    hdri_target = HdriTarget(ctx, 256)

    T_gl_cv = np.eye(4)
    T_gl_cv[1, 1] = -1
    T_gl_cv[2, 2] = -1

    fov = math.degrees(2 * math.atan(0.5 * width / 550))
    proj = Matrix44.perspective_projection(fov, 1.0, 0.01, 100.0)

    vis_sample_size = 1024
    order = 2
    dirs, phi, theta = fibonacci_sphere(vis_sample_size)
    SH = getSHCoeffs(order, phi, theta)

    model_path = os.path.join(dataset_folder, model_name, f"{model_name}.obj")

    mesh = import_mesh(model_path)
    aabb_min = np.min(mesh.vertices, axis=0).reshape(1, -1)
    aabb_max = np.max(mesh.vertices, axis=0).reshape(1, -1)
    delta = 1e-3 * np.min(aabb_max - aabb_min)

    vis = np.logical_not(
        vis_fuse_utils.sample_occlusion_embree(
            mesh.vertices, mesh.faces,
            mesh.vertices + delta * mesh.vertex_normals, dirs))
    geo_term = np.clip(np.einsum("ik,jk->ij", mesh.vertex_normals, dirs), 0, 1)
    prt = np.einsum("ij,ij,jk->ik", vis, geo_term,
                    SH) * 4.0 * np.pi / vis_sample_size
    prt_cos = np.einsum("ij,jk->ik", geo_term,
                        SH) * 4.0 * np.pi / vis_sample_size
    prt_vis = np.einsum("ij,jk->ik", vis, SH) * 4.0 * np.pi / vis_sample_size

    per_face_vertices = mesh.vertices[mesh.faces].reshape(-1, 3)
    per_face_uv = mesh.uvs[mesh.face_uvs_idx].reshape(-1, 2)
    per_face_prt = prt[mesh.faces].reshape(-1, 9)
    per_face_prt_cos = prt_cos[mesh.faces].reshape(-1, 9)
    per_face_prt_vis = prt_vis[mesh.faces].reshape(-1, 9)
    per_face_vertex_normal = mesh.vertex_normals[mesh.faces].reshape(-1, 3)

    texture_image = mesh.materials[0].transpose(Image.Transpose.FLIP_TOP_BOTTOM)

    prt_uv_target.build_vao(per_face_vertices, per_face_uv, per_face_prt,
                            texture_image)
    prt_uv_cos_target.build_vao(per_face_vertices, per_face_uv,
                                per_face_prt_cos, texture_image)
    depth_target.build_vao(per_face_vertices)
    normal_target.build_vao(per_face_vertices, per_face_vertex_normal)
    vis_target.build_vao(per_face_vertices, per_face_prt_vis)

    sh_rot_factor = np.random.randint(0, 30)
    sh_id = np.random.randint(0, len(env_shs))
    R_sh = matrix33.create_from_eulers(
        [-0.5 * sh_rot_factor, 0.1 * sh_rot_factor, -sh_rot_factor])
    sh = rotateSH(np.copy(env_shs[sh_id]), R_sh)
    sh_path = os.path.join(dataset_folder, model_name, "sh.npy")
    np.save(sh_path, sh)

    sh = sh.T.reshape(-1,).astype('f4')

    # `cams.mat` stores pre randomized camera poses. See `load_cam` in `thuman.py`
    # You could replace it with poses of your choice
    cams = loadmat(os.path.join(dataset_folder, model_name, 'cams.mat'))

    for i in range(60):
        R, t, _ = load_cam(cams, i)
        view = np.eye(4)
        view[:3, :3] = R
        view[:3, 3] = t
        view = T_gl_cv @ view

        mv = np.ascontiguousarray(view.T).astype('f4')
        mvp = (proj * view.T).astype('f4')

        hdri_img = hdri_target.render(sh,
                                      np.ascontiguousarray(R.T).astype('f4'))
        image_light, image_color, image_albedo = prt_uv_target.render(mvp, sh)
        image_light_cos, image_color_cos, _ = prt_uv_cos_target.render(mvp, sh)
        image_depth = depth_target.render(mv, mvp)
        depth_mask = image_depth < -1e-6
        image_local_normal, image_normal = normal_target.render(mv, mvp)
        image_vis = vis_target.render(mvp)

        image_depth_gt = np.uint16(-1000 * image_depth)
        image_depth_gt[np.logical_not(depth_mask)] = 30000

        add_depth_noise(image_local_normal, image_depth, depth_mask)

        image_depth = np.uint16(-1000 * image_depth)
        image_depth[np.logical_not(depth_mask)] = 30000

        image_normal = np.uint8((image_normal + 1.) / 2. * 255)
        image_normal[np.logical_not(depth_mask)] = 0
        image_normal = np.flip(image_normal, axis=2)

        image_local_normal = np.uint8((image_local_normal + 1.) / 2. * 255)
        image_local_normal[np.logical_not(depth_mask)] = 0
        image_local_normal = np.flip(image_local_normal, axis=2)

        # np.save(
        #     os.path.join(dataset_folder, model_name,
        #                  f'vis_view_{i}.npy'), image_vis)
        # cv2.imwrite(
        #     os.path.join(dataset_folder, model_name,
        #                  f'depth_view_gt_{i}.png'), image_depth_gt)
        cv2.imwrite(
            os.path.join(dataset_folder, model_name, f'depth_view_{i}.png'),
            image_depth)
        cv2.imwrite(
            os.path.join(dataset_folder, model_name, f'normal_view_{i}.jpg'),
            image_normal)
        # cv2.imwrite(
        #     os.path.join(dataset_folder, model_name,
        #                  f'local_normal_view_{i}.jpg'),
        #     image_local_normal)
        cv2.imwrite(
            os.path.join(dataset_folder, model_name, f'color_view_{i}.jpg'),
            cv2.cvtColor(image_color, cv2.COLOR_BGR2RGB))
        # cv2.imwrite(
        #     os.path.join(dataset_folder, model_name,
        #                  f'color_view_cos_{i}.jpg'),
        #     cv2.cvtColor(image_color_cos, cv2.COLOR_BGR2RGB))
        # cv2.imwrite(
        #     os.path.join(dataset_folder, model_name,
        #                  f'light_view_{i}.jpg'),
        #     cv2.cvtColor(image_light, cv2.COLOR_BGR2RGB))
        # cv2.imwrite(
        #     os.path.join(dataset_folder, model_name,
        #                  f'light_view_cos_{i}.jpg'),
        #     cv2.cvtColor(image_light_cos, cv2.COLOR_BGR2RGB))
        cv2.imwrite(
            os.path.join(dataset_folder, model_name, f'albedo_view_{i}.jpg'),
            cv2.cvtColor(image_albedo, cv2.COLOR_BGR2RGB))
        # cv2.imwrite(
        #     os.path.join(dataset_folder, model_name,
        #                  f'hdri_view_{i}.exr'),
        #     cv2.cvtColor(hdri_img[..., :3], cv2.COLOR_RGB2BGR))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_folder', type=str, help='Path to data folder.')
    args = parser.parse_args()

    data_folder = args.data_folder
    if data_folder.endswith('/'):
        results_folder = data_folder[:-1]
    tag = data_folder.split('/')[-1]

    # https://stackoverflow.com/questions/16008670/how-to-hash-a-string-into-8-digits
    seed = int(hashlib.sha256(tag.encode('utf-8')).hexdigest(), 16) % 10**8
    random.seed(seed)
    np.random.seed(seed)

    width = 512
    height = 512

    # Or `grayish_env_shs` generated from `sh_selection.py`, more suitable for indoor scene
    env_shs = np.load('implicit/env_sh.npy')[:10, ...]

    model_list = sorted(os.listdir(data_folder))
    Parallel(n_jobs=4)(
        delayed(render_views)(model_name, data_folder, env_shs, width, height)
        for model_name in model_list)
