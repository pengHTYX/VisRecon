from view_renderer import import_mesh
from implicit.implicit_render_prt import get_view_matrix, rotateSH
import numpy as np
import moderngl
import math
from pyrr import Matrix44, matrix33
from PIL import Image
from tqdm import tqdm
import os
import random
from icecream import ic

random.seed(0)
np.random.seed(0)

datasets = ['thuman', 'thuman']
render_count = 12
view_angles = np.linspace(0, 360, render_count + 1)[:-1]
env_ids = [0, 2, 6, 9, 10, 11, 12, 14][:render_count]
Rs = [
    matrix33.create_from_eulers([-0.05 * i, 0.01 * i, -0.1 * i])
    for i in range(render_count)
]


def load_program(ctx: moderngl.Context, program_name):
    vert = open(f"implicit/shaders/{program_name}.vs").read()
    frag = open(f"implicit/shaders/{program_name}.fs").read()
    return ctx.program(vertex_shader=vert, fragment_shader=frag)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_folder',
                        type=str,
                        default='$HOME/dataset/test_full',
                        help='Path to data folder.')
    parser.add_argument('--result', type=str, help='Path to result folder.')
    parser.add_argument('--out',
                        type=str,
                        default='$HOME/dataset/paper_validation_metrics',
                        help='Path to output folder.')
    args = parser.parse_args()

    data_folder = args.data_folder
    results_folder = args.result
    render_save_folder = args.out

    if results_folder.endswith('/'):
        results_folder = results_folder[:-1]
    configs = os.listdir(results_folder)

    prt_file_name = 'prt_gen.npy'
    env_shs = np.load('implicit/env_sh.npy')

    ctx = moderngl.create_context(standalone=True, backend='egl')
    ctx.enable(moderngl.DEPTH_TEST)
    ctx.gc_mode = "auto"

    width = 512
    height = 512
    color_attachments = [
        ctx.texture((width, height), 4, samples=8),
        ctx.texture((width, height), 4, samples=8),
        ctx.texture((width, height), 4, samples=8)
    ]
    fbo = ctx.framebuffer(color_attachments=color_attachments,
                          depth_attachment=ctx.depth_renderbuffer(
                              (width, height), samples=8))

    color_attachments2 = [
        ctx.texture((width, height), 4),
        ctx.texture((width, height), 4),
        ctx.texture((width, height), 4)
    ]
    fbo2 = ctx.framebuffer(color_attachments=color_attachments2,
                           depth_attachment=ctx.depth_renderbuffer(
                               (width, height)))

    prog_uv = load_program(ctx, "prt_uv")
    prog_vc = load_program(ctx, "prt_vc_simple")

    ortho_ratio = 0.5
    proj = Matrix44.orthogonal_projection(-ortho_ratio, ortho_ratio,
                                          -ortho_ratio, ortho_ratio, 0.01,
                                          100.0)

    def build_vao(model_path, prt_path):
        mesh = import_mesh(model_path)
        prt = np.load(prt_path)

        # normalize
        # aabb_min = np.min(mesh.vertices, axis=0).reshape(1, -1)
        # aabb_max = np.max(mesh.vertices, axis=0).reshape(1, -1)
        # center = 0.5 * (aabb_max + aabb_min)
        # scale = 2.0 * np.max(aabb_max - center)
        # vertices = mesh.vertices / scale
        vertices = mesh.vertices

        per_face_vertices = vertices[mesh.faces].reshape(-1, 3)
        per_face_prt = prt[mesh.faces].reshape(-1, 9)

        vbo_vert = ctx.buffer(per_face_vertices.astype('f4'))
        vbo_sh = ctx.buffer(per_face_prt.astype('f4'))

        texture = None
        if mesh.vertex_colors is not None:
            per_face_vc = mesh.vertex_colors[mesh.faces].reshape(-1, 3)
            vbo_vc = ctx.buffer(per_face_vc.astype('f4'))
            vao = ctx.vertex_array(prog_vc, [(vbo_vert, '3f', 'in_pos'),
                                             (vbo_vc, '3f', 'in_vert_color'),
                                             (vbo_sh, '9f', 'in_sh')])
        else:
            per_face_uv = mesh.uvs[mesh.face_uvs_idx].reshape(-1, 2)
            vbo_uv = ctx.buffer(per_face_uv.astype('f4'))
            vao = ctx.vertex_array(prog_uv, [(vbo_vert, '3f', 'in_pos'),
                                             (vbo_uv, '2f', 'in_uv'),
                                             (vbo_sh, '9f', 'in_sh')])

            texture_image = mesh.materials[0].transpose(
                Image.Transpose.FLIP_TOP_BOTTOM)
            texture = ctx.texture(texture_image.size, 3,
                                  texture_image.tobytes())
            texture.build_mipmaps()
        return {'vao': vao, 'texture': texture}

    for dataset in datasets:
        print(f"Generate dataset {dataset}")

        save_folder = os.path.join(render_save_folder, 'render')
        if not os.path.exists(save_folder):
            os.mkdir(save_folder)

        for item in tqdm(os.listdir(os.path.join(data_folder, dataset))):
            model_name = f'{item}.obj'
            render_targets = {}

            model_folder_path = os.path.join(data_folder, dataset, item)
            model_path = os.path.join(model_folder_path, model_name)
            prt_path = os.path.join(model_folder_path, prt_file_name)

            render_targets['gt'] = build_vao(model_path, prt_path)

            if not os.path.exists(os.path.join(save_folder, 'gt')):
                os.mkdir(os.path.join(save_folder, 'gt'))

            for cfg in configs:
                model_folder_path = os.path.join(results_folder, cfg, item)
                model_path = os.path.join(model_folder_path, model_name)
                prt_path = os.path.join(model_folder_path, prt_file_name)

                render_targets[cfg] = build_vao(model_path, prt_path)

                if not os.path.exists(os.path.join(save_folder, cfg)):
                    os.mkdir(os.path.join(save_folder, cfg))

            def render_pass(view_angle, sh_id, R):
                eye = [
                    math.cos(math.radians(-view_angle)), 0.0,
                    math.sin(math.radians(-view_angle))
                ]
                lookat = get_view_matrix(eye)
                sh = rotateSH(np.copy(env_shs[sh_id]), R)
                sh = sh.T.reshape(-1,)

                fbo.use()

                prog_uv['mvp'].write((proj * lookat).astype('f4'))
                prog_uv['env_sh'].write(sh.astype('f4'))
                prog_vc['mvp'].write((proj * lookat).astype('f4'))
                prog_vc['env_sh'].write(sh.astype('f4'))

                for cfg, data in render_targets.items():
                    fbo.clear()
                    texture = data['texture']
                    if texture is not None:
                        texture.use()
                    data['vao'].render()
                    ctx.copy_framebuffer(fbo2, fbo)
                    light_image_data = fbo2.read(components=4, attachment=0)
                    light_image = Image.frombytes(
                        'RGBA', fbo2.size, light_image_data).transpose(
                            Image.Transpose.FLIP_TOP_BOTTOM)
                    color_image_data = fbo2.read(components=4, attachment=1)
                    color_image = Image.frombytes(
                        'RGBA', fbo2.size, color_image_data).transpose(
                            Image.Transpose.FLIP_TOP_BOTTOM)
                    albedo_image_data = fbo2.read(components=4, attachment=2)
                    albedo_image = Image.frombytes(
                        'RGBA', fbo2.size, albedo_image_data).transpose(
                            Image.Transpose.FLIP_TOP_BOTTOM)

                    save_folder_tag = os.path.join(save_folder, cfg)
                    view_angle = int(view_angle)
                    light_image.save(
                        os.path.join(save_folder_tag,
                                     f'{item}_{view_angle}_light_color.png'))
                    color_image.save(
                        os.path.join(save_folder_tag,
                                     f'{item}_{view_angle}_color.png'))
                    albedo_image.save(
                        os.path.join(save_folder_tag,
                                     f'{item}_{view_angle}_albedo.png'))

            for (view_angle, env_id, R) in zip(view_angles, env_ids, Rs):
                render_pass(view_angle, env_id, R)
