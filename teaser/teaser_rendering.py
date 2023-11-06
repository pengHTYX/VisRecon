from view_renderer import import_mesh
from implicit.implicit_render_prt import get_view_matrix
from implicit.implicit_prt_gen import fibonacci_sphere, getSHCoeffs
import numpy as np
import moderngl
import math
from pyrr import matrix44, Matrix44
from PIL import Image
from tqdm import tqdm
import os
import random
import argparse
from icecream import ic

random.seed(0)
np.random.seed(0)

datasets = ['thuman', 'twindom']
configs = ['vis_fuse']
render_count = 10

view_angle_map = {
    '0478': -150,
    '0537': -200,
    '0549': 130,
    '126111536790861': -90,
    '126211540489441': -120,
    '0261': -110,
    '0282': -60,
    '0298': 145,
    '126111539897023': -90,
    '0289': -15,
    # '126111551391968': -100,
    '0307': 90,
    '0309': -120,
    '0277': -120,
    '140711542935158': -120,
    '141311548775566': -120,
    '141511557173943': -60,
    '141511558561737': -110
}

cubemap_list = [
    'kiara_1_dawn', 'cape_hill', 'blaubeuren_night', 'signal_hill_sunrise',
    'belfast_sunset', 'qwantani_puresky', 'ehingen_hillside'
]

cubemap_start_map = {
    'kiara_1_dawn': -220,
    'ehingen_hillside': -200,
    'signal_hill_sunrise': -240,
    'cape_hill': -220,
    'blaubeuren_night': -240,
    'belfast_sunset': -220,
    'qwantani_puresky': -220
}

cubemap_range_map = {
    'kiara_1_dawn': 220,
    'ehingen_hillside': 200,
    'signal_hill_sunrise': 240,
    'cape_hill': 240,
    'blaubeuren_night': 240,
    'belfast_sunset': 180,
    'qwantani_puresky': 220
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_folder', type=str, help='Path to data folder.')
    parser.add_argument('--out',
                        type=str,
                        default='$HOME/dataset/paper_teaser/realworld_relit',
                        help='Path to output folder.')
    args = parser.parse_args()

    render_data_folder = args.data_folder
    render_save_folder = args.out

    cubemap_folder = "implicit/cubemaps"

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

    prog_uv = load_program("prt_uv")
    prog_vc = load_program("prt_vc")

    ortho_ratio = 0.6
    proj = Matrix44.orthogonal_projection(-ortho_ratio, ortho_ratio,
                                          -ortho_ratio, ortho_ratio, 0.01,
                                          100.0)

    cube_proj = Matrix44.perspective_projection(45.0, 1.0, 0.01, 100.0)

    def build_vao(model_path, prt_path):
        mesh = import_mesh(model_path)
        prt = np.load(prt_path)

        # normalize
        aabb_min = np.min(mesh.vertices, axis=0).reshape(1, -1)
        aabb_max = np.max(mesh.vertices, axis=0).reshape(1, -1)
        center = 0.5 * (aabb_max + aabb_min)
        scale = (aabb_max[0, 1] - aabb_min[0, 1])
        vertices = (mesh.vertices - center) / scale
        # vertices = mesh.vertices

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
        return {'vao': vao, 'texture': texture}

    for item in os.listdir(render_data_folder):
        save_folder = render_save_folder
        if not os.path.exists(render_save_folder):
            os.mkdir(render_save_folder)

        model_name = f'{item}.obj'
        render_targets = {}

        for cfg in configs:
            model_folder_path = os.path.join(render_data_folder, item)
            model_path = os.path.join(model_folder_path, model_name)
            prt_path = os.path.join(model_folder_path, prt_file_name)

            render_targets[cfg] = build_vao(model_path, prt_path)

            if not os.path.exists(os.path.join(save_folder, cfg)):
                os.mkdir(os.path.join(save_folder, cfg))

        view_angle = view_angle_map[item] if item in view_angle_map else 0
        eye = [
            math.cos(math.radians(-view_angle)), 0.0,
            math.sin(math.radians(-view_angle))
        ]
        lookat = get_view_matrix(eye)
        cube_transform = np.copy(lookat)
        cube_transform[3][0] = 0
        cube_transform[3][1] = 0
        cube_transform[3][2] = 0

        interval = 2.
        counter = 0
        for cubemap in cubemaps:
            cubemap_name, cubemap_texture, cubemap_sh = cubemap
            sh = cubemap_sh.T.reshape(-1,)

            start_frame = cubemap_start_map[cubemap_name] + view_angle
            frame_count = int(cubemap_range_map[cubemap_name] // interval)

            def render_pass(angle, frame_idx):
                fbo.use()

                model = matrix44.create_from_eulers([0, 0, angle]).astype('f4')

                cube_mvp = (cube_proj * cube_transform * model).astype('f4')
                cubemap_prog['mvp'].write(cube_mvp)

                model_mvp = (proj * lookat).astype('f4')
                prog_uv['mvp'].write(model_mvp)
                prog_uv['env_sh'].write(sh.astype('f4'))

                prog_vc['vp'].write(model_mvp)
                prog_vc['model'].write(model.astype('f4'))
                prog_vc['env_sh'].write(sh.astype('f4'))

                for tag, data in render_targets.items():
                    fbo.clear(red=1., green=1., blue=1.)
                    ctx.front_face = 'ccw'
                    cubemap_texture.use(location=0)
                    cubemap_vao.render()

                    ctx.front_face = 'cw'
                    ctx.enable(moderngl.DEPTH_TEST)

                    texture = data['texture']
                    if texture is not None:
                        texture.use()
                    data['vao'].render()

                    ctx.copy_framebuffer(fbo2, fbo)

                    light_image_data = fbo2.read(components=3, attachment=0)
                    light_image = Image.frombytes(
                        'RGB', fbo2.size, light_image_data).transpose(
                            Image.Transpose.FLIP_TOP_BOTTOM)
                    color_image_data = fbo2.read(components=3, attachment=1)
                    color_image = Image.frombytes(
                        'RGB', fbo2.size, color_image_data).transpose(
                            Image.Transpose.FLIP_TOP_BOTTOM)

                    save_folder_tag = os.path.join(save_folder, tag)

                    light_image = light_image.crop((128, 0, 128 + 256, 512))
                    light_image.save(
                        os.path.join(save_folder_tag,
                                     f'{item}_light_color_%04d.png' %
                                     frame_idx))

                    color_image = color_image.crop((128, 0, 128 + 256, 512))
                    color_image.save(
                        os.path.join(save_folder_tag,
                                     f'{item}_color_%04d.png' % frame_idx))

            for i in tqdm(range(frame_count)):
                render_pass(-math.radians(start_frame + i * interval), counter)
                counter += 1

        for tag, data in render_targets.items():
            save_folder_tag = os.path.join(save_folder, tag)

            for im_tag in ['light_color', 'color']:
                ims_in = os.path.join(save_folder_tag, f'{item}_{im_tag}_*.png')
                gif_out = os.path.join(save_folder_tag, f'{item}_{im_tag}.gif')

                cmd = f"gifski -H {height} -Q 90 -r 30 -o {gif_out} {ims_in}"
                os.system(cmd)
                cmd = f'rm {ims_in}'
                os.system(cmd)
