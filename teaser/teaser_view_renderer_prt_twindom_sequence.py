import os
import numpy as np
from implicit.implicit_render_prt import get_view_matrix
import math
import moderngl
from pyrr import Matrix44
from PIL import Image
from tqdm import tqdm
import igl
import argparse
from icecream import ic

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_folder', type=str, help='Path to data folder.')
    parser.add_argument('--out',
                        type=str,
                        default='$HOME/dataset/paper_teaser/realworld',
                        help='Path to output folder.')
    args = parser.parse_args()

    save_folder = os.path.join(args.out, args.data_folder.split('/')[-1])

    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    ctx = moderngl.create_context(standalone=True, backend='egl')
    ctx.enable(moderngl.DEPTH_TEST)
    ctx.gc_mode = "auto"

    vert = open("implicit/shaders/prt_vc_simple.vs").read()
    frag = open("implicit/shaders/prt_vc_simple.fs").read()
    prog = ctx.program(vertex_shader=vert, fragment_shader=frag)

    env_shs = np.load('implicit/env_sh.npy')

    sh = np.copy(env_shs[0]).T.reshape(-1,).astype('f4')

    proj = Matrix44.perspective_projection(55, 1.0, 0.01, 100.0)
    width = 512
    height = 512
    color_attachments = [
        ctx.renderbuffer((width, height), 4, samples=8),
        ctx.renderbuffer((width, height), 4, samples=8),
        ctx.renderbuffer((width, height), 4, samples=8)
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
    image_tags = ['light', 'color']

    def render(dataset_folder, save_folder):
        model_list = sorted(os.listdir(dataset_folder))
        for model_name in tqdm(model_list):
            model_path = os.path.join(dataset_folder, model_name,
                                      f"{model_name}.obj")
            vertex_color_path = os.path.join(dataset_folder, model_name,
                                             "vertex_color.npy")
            prt_path = os.path.join(dataset_folder, model_name, "prt_gen.npy")

            v, f = igl.read_triangle_mesh(model_path)

            vertex_color = np.load(vertex_color_path)
            prt = np.load(prt_path)

            vertex_color = np.load(vertex_color_path)

            per_face_vertices = v[f].reshape(-1, 3)
            vbo_vert = ctx.buffer(per_face_vertices.astype('f4'))

            per_face_vertex_color = vertex_color[f].reshape(-1, 3)
            vbo_vc = ctx.buffer(per_face_vertex_color.astype('f4'))

            per_face_prt = prt[f].reshape(-1, 9)
            vbo_prt = ctx.buffer(per_face_prt.astype('f4'))

            vao = ctx.vertex_array(prog, [(vbo_vert, '3f', 'in_pos'),
                                          (vbo_vc, '3f', 'in_vert_color'),
                                          (vbo_prt, '9f', 'in_sh')])

            model = Matrix44.from_y_rotation(-90)

            def render_pass(view_angle):
                eye = [
                    math.cos(math.radians(-view_angle)), 0.0,
                    math.sin(math.radians(-view_angle))
                ]
                view = get_view_matrix(eye)

                mvp = (proj * view * model).astype('f4')

                prog['mvp'].write(mvp)
                prog['env_sh'].write(sh)

                fbo.use()
                fbo.clear()
                vao.render()

                ctx.copy_framebuffer(fbo2, fbo)

                light_color_data = fbo2.read(components=3, attachment=0)
                image_light_color = Image.frombytes(
                    'RGB', fbo2.size,
                    light_color_data).transpose(Image.Transpose.FLIP_TOP_BOTTOM)

                color_data = fbo2.read(components=3, attachment=1)
                image_color = Image.frombytes(
                    'RGB', fbo2.size,
                    color_data).transpose(Image.Transpose.FLIP_TOP_BOTTOM)

                return image_light_color, image_color

            render_results = render_pass(30)

            for (image_tag, image) in zip(image_tags, render_results):
                save_path = os.path.join(save_folder,
                                         f"{model_name}_{image_tag}.png")
                image.save(save_path)

        for image_tag in image_tags:
            cmd = f"gifski -H {height} -Q 100 -o {save_folder}/{image_tag}.gif {save_folder}/*_{image_tag}.png"
            os.system(cmd)
            cmd = f'rm %s/*_{image_tag}.png' % save_folder
            os.system(cmd)

    render(args.data_folder, save_folder)
