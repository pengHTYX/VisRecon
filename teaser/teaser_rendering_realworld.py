from implicit.implicit_render_prt import get_view_matrix
from implicit.implicit_prt_gen import fibonacci_sphere, getSHCoeffs
import numpy as np
import moderngl
import math
from pyrr import matrix44, matrix33, Matrix44
from PIL import Image
from tqdm import tqdm
import os
import random
import argparse
import igl
from icecream import ic

random.seed(0)
np.random.seed(0)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_folder', type=str, help='Path to data folder.')
    parser.add_argument('--out',
                        type=str,
                        default='$HOME/dataset/paper_teaser/realworld_relit',
                        help='Path to output folder.')
    args = parser.parse_args()

    data_tag = args.data_folder.split('/')[-1]
    dataset_folder = args.data_folder
    save_folder = args.out

    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    cubemap_folder = "implicit/cubemaps"
    cubemap_list = os.listdir(cubemap_folder)
    prt_file_name = 'prt_gen.npy'

    ctx = moderngl.create_context(standalone=True, backend='egl')
    ctx.gc_mode = "auto"

    light_dirs, phi, theta = fibonacci_sphere(256)
    light_sh_basis = getSHCoeffs(2, phi, theta)

    def load_cubemap(cubemap_name):
        faces = [
            Image.open(os.path.join(cubemap_folder, cubemap_name, f'{tag}.png'))
            for tag in ['px', 'nx', 'py', 'ny', 'pz', 'nz']
        ]
        face_bytes = [face_img.tobytes() for face_img in faces]
        cubemap_texture = ctx.texture_cube(faces[0].size, 4,
                                           b''.join(face_bytes))

        sh_path = os.path.join(cubemap_folder, cubemap_name, 'sh.txt')
        with open(sh_path) as f:
            lines = f.readlines()
            data = [l.split('(')[1].split(')')[0].split(',') for l in lines]
            cubemap_sh = np.array(data).astype('f4')

        return [cubemap_texture, cubemap_sh]

    cubemaps = [[cubemap_name] + load_cubemap(cubemap_name)
                for cubemap_name in cubemap_list]

    cubemap_prog = ctx.program(vertex_shader='''
            #version 330

            uniform mat4 mvp;
            layout (location = 0) in vec3 in_pos;
            out vec3 pos;
            void main() {
                gl_Position =  mvp * vec4(in_pos, 1.0);
                pos = in_pos;
            }
        ''',
                               fragment_shader='''
            #version 330
            uniform samplerCube texture0;
            in vec3 pos;
            layout (location = 0) out vec4 fragColor;
            void main() {
                fragColor = texture(texture0, normalize(pos));
            }
        ''')

    def make_cube(
            size=(1.0, 1.0, 1.0), center=(0.0, 0.0, 0.0), attr_names='in_pos'):
        width, height, depth = size
        width, height, depth = width / 2.0, height / 2.0, depth / 2.0

        pos = np.array(
            [[center[0] + width, center[1] - height, center[2] + depth],
             [center[0] + width, center[1] + height, center[2] + depth],
             [center[0] - width, center[1] - height, center[2] + depth],
             [center[0] + width, center[1] + height, center[2] + depth],
             [center[0] - width, center[1] + height, center[2] + depth],
             [center[0] - width, center[1] - height, center[2] + depth],
             [center[0] + width, center[1] - height, center[2] - depth],
             [center[0] + width, center[1] + height, center[2] - depth],
             [center[0] + width, center[1] - height, center[2] + depth],
             [center[0] + width, center[1] + height, center[2] - depth],
             [center[0] + width, center[1] + height, center[2] + depth],
             [center[0] + width, center[1] - height, center[2] + depth],
             [center[0] + width, center[1] - height, center[2] - depth],
             [center[0] + width, center[1] - height, center[2] + depth],
             [center[0] - width, center[1] - height, center[2] + depth],
             [center[0] + width, center[1] - height, center[2] - depth],
             [center[0] - width, center[1] - height, center[2] + depth],
             [center[0] - width, center[1] - height, center[2] - depth],
             [center[0] - width, center[1] - height, center[2] + depth],
             [center[0] - width, center[1] + height, center[2] + depth],
             [center[0] - width, center[1] + height, center[2] - depth],
             [center[0] - width, center[1] - height, center[2] + depth],
             [center[0] - width, center[1] + height, center[2] - depth],
             [center[0] - width, center[1] - height, center[2] - depth],
             [center[0] + width, center[1] + height, center[2] - depth],
             [center[0] + width, center[1] - height, center[2] - depth],
             [center[0] - width, center[1] - height, center[2] - depth],
             [center[0] + width, center[1] + height, center[2] - depth],
             [center[0] - width, center[1] - height, center[2] - depth],
             [center[0] - width, center[1] + height, center[2] - depth],
             [center[0] + width, center[1] + height, center[2] - depth],
             [center[0] - width, center[1] + height, center[2] - depth],
             [center[0] + width, center[1] + height, center[2] + depth],
             [center[0] - width, center[1] + height, center[2] - depth],
             [center[0] - width, center[1] + height, center[2] + depth],
             [center[0] + width, center[1] + height, center[2] + depth]],
            dtype=np.float32)
        vbo_vert = ctx.buffer(pos.astype('f4'))
        vao = ctx.vertex_array(cubemap_prog, [(vbo_vert, '3f', attr_names)])
        return vao

    cubemap_vao = make_cube(size=(20, 20, 20))

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

    def load_program(program_name):
        vert = open(f"implicit/shaders/{program_name}.vs").read()
        frag = open(f"implicit/shaders/{program_name}.fs").read()
        return ctx.program(vertex_shader=vert, fragment_shader=frag)

    prog = load_program("prt_vc")
    proj = Matrix44.perspective_projection(55, 1.0, 0.01, 100.0)

    cube_proj = Matrix44.perspective_projection(45.0, 1.0, 0.01, 100.0)

    scale = 2
    y_offset = -0.1
    z_offset = -1.9

    model_list = sorted(os.listdir(dataset_folder))

    def build_vao(model_name):
        model_path = os.path.join(dataset_folder, model_name,
                                  f"{model_name}.obj")
        vertex_color_path = os.path.join(dataset_folder, model_name,
                                         "vertex_color.npy")
        prt_path = os.path.join(dataset_folder, model_name, "prt_gen.npy")

        v, f = igl.read_triangle_mesh(model_path)

        vertex_color = np.load(vertex_color_path)
        prt = np.load(prt_path)

        vertex_color = np.load(vertex_color_path)

        v[:, 1] += y_offset
        v[:, 2] += z_offset
        v /= scale

        v = v @ matrix33.create_from_x_rotation(
            math.radians(-10 if data_tag == 'wentao' else -20))

        per_face_vertices = v[f].reshape(-1, 3)
        vbo_vert = ctx.buffer(per_face_vertices.astype('f4'))

        per_face_vertex_color = vertex_color[f].reshape(-1, 3)
        vbo_vc = ctx.buffer(per_face_vertex_color.astype('f4'))

        per_face_prt = prt[f].reshape(-1, 9)
        vbo_prt = ctx.buffer(per_face_prt.astype('f4'))

        vao = ctx.vertex_array(prog, [(vbo_vert, '3f', 'in_pos'),
                                      (vbo_vc, '3f', 'in_vert_color'),
                                      (vbo_prt, '9f', 'in_sh')])
        return vao

    view_angle = 30
    eye = [
        math.cos(math.radians(-view_angle)), 0.0,
        math.sin(math.radians(-view_angle))
    ]
    lookat = get_view_matrix(eye)
    cube_transform = np.copy(lookat)
    cube_transform[3][0] = 0
    cube_transform[3][1] = 0
    cube_transform[3][2] = 0

    interval = 1.
    for cubemap in cubemaps:
        cubemap_name, cubemap_texture, cubemap_sh = cubemap
        sh = cubemap_sh.T.reshape(-1,)

        start_frame = view_angle

        frame_count = len(model_list)

        def render_pass(angle, counter):
            fbo.use()

            model = matrix44.create_from_eulers([0, 0, angle]).astype('f4')

            cube_mvp = (cube_proj * cube_transform * model).astype('f4')
            cubemap_prog['mvp'].write(cube_mvp)

            model_mvp = (proj * lookat).astype('f4')
            prog['vp'].write(model_mvp)
            prog['model'].write(model.astype('f4'))
            prog['env_sh'].write(sh.astype('f4'))

            model_name = model_list[counter]
            vao = build_vao(model_name)

            fbo.clear()
            ctx.front_face = 'ccw'
            cubemap_texture.use(location=0)
            cubemap_vao.render()

            ctx.front_face = 'cw'
            ctx.enable(moderngl.DEPTH_TEST)

            vao.render()

            ctx.copy_framebuffer(fbo2, fbo)

            light_image_data = fbo2.read(components=3, attachment=0)
            light_image = Image.frombytes('RGB', fbo2.size,
                                          light_image_data).transpose(
                                              Image.Transpose.FLIP_TOP_BOTTOM)
            color_image_data = fbo2.read(components=3, attachment=1)
            color_image = Image.frombytes('RGB', fbo2.size,
                                          color_image_data).transpose(
                                              Image.Transpose.FLIP_TOP_BOTTOM)

            light_image.save(
                os.path.join(save_folder,
                             f"{data_tag}_{model_name}_light_color.png"))
            color_image.save(
                os.path.join(save_folder, f"{data_tag}_{model_name}_color.png"))

        for i in tqdm(range(frame_count)):
            render_pass(-math.radians(start_frame + i * interval), i)

        for image_tag in ['light_color', 'color']:
            cmd = f"gifski -H {height} -Q 100 -o {save_folder}/{data_tag}_{image_tag}_{cubemap_name}.gif {save_folder}/{data_tag}_realworld_*_{image_tag}.png"
            os.system(cmd)
            cmd = f'rm %s/{data_tag}_realworld_*_{image_tag}.png' % save_folder
            os.system(cmd)
