from pyrr import Matrix44
import moderngl
import numpy as np
from implicit.implicit_prt_gen import fibonacci_sphere, getSHCoeffs
from implicit.implicit_render_prt import get_view_matrix
from PIL import Image
import math
import os

render_count = 120
view_angles = np.linspace(0, 360, render_count + 1)[:-1]
env_ids = [0, 2, 6, 9, 10, 11, 12, 14]

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--out',
                        type=str,
                        default='$HOME/dataset/paper_validation_metrics',
                        help='Path to output folder.')
    args = parser.parse_args()

    light_save_folder = args.out

    ctx = moderngl.create_context(standalone=True, backend='egl')
    ctx.enable(moderngl.DEPTH_TEST)
    ctx.gc_mode = "auto"

    ortho_ratio = 0.5
    proj = Matrix44.orthogonal_projection(-ortho_ratio, ortho_ratio,
                                          -ortho_ratio, ortho_ratio, 0.01,
                                          100.0)

    env_shs = np.load('implicit/env_sh.npy')

    sphere_prog = ctx.program(vertex_shader='''
                    #version 330

                    uniform mat4 mvp;

                    layout (location = 0) in vec3 in_pos;
                    layout (location = 1) in vec3 in_vert_color;

                    out vec3 vert_color;
                    void main() {
                        gl_Position = mvp * vec4(in_pos, 1.0);
                        vert_color = in_vert_color;
                    }
                ''',
                              fragment_shader='''
                    #version 330
                    in vec3 vert_color;
                    layout (location = 0) out vec4 frag_color;
                    void main() {
                        frag_color = vec4(vert_color, 1.0);
                    }
                ''')

    light_dirs, phi, theta = fibonacci_sphere(256)
    light_sh_basis = getSHCoeffs(2, phi, theta)
    sphere_vertices = light_dirs * 0.5
    sphere_indices = np.load('implicit/face_256.npy')
    sphere_vbo = ctx.buffer(sphere_vertices.astype('f4'))
    sphere_ibo = ctx.buffer(sphere_indices.astype('i4'))

    light_width = 128
    light_height = 128
    sphere_fbo = ctx.framebuffer(color_attachments=ctx.texture(
        (light_width, light_height), 4),
                                 depth_attachment=ctx.depth_renderbuffer(
                                     (light_width, light_height)))

    for view_angle in view_angles:
        for sh_id in env_ids:
            eye = [
                math.cos(math.radians(-view_angle)), 0.0,
                math.sin(math.radians(-view_angle))
            ]
            lookat = get_view_matrix(eye)

            env_sh = np.copy(env_shs[sh_id])

            sphere_prog['mvp'].write((proj * lookat).astype('f4'))
            light_vert_color = light_sh_basis @ env_sh
            vbo_vc = ctx.buffer(light_vert_color.astype('f4'))
            sphere_vao = ctx.vertex_array(sphere_prog,
                                          [(sphere_vbo, '3f', 'in_pos'),
                                           (vbo_vc, '3f', 'in_vert_color')],
                                          sphere_ibo)
            sphere_fbo.use()
            sphere_fbo.clear()
            sphere_vao.render()
            light_color_data = sphere_fbo.read(components=4, attachment=0)
            light_color = Image.frombytes('RGBA', sphere_fbo.size,
                                          light_color_data).transpose(
                                              Image.Transpose.FLIP_TOP_BOTTOM)
            view_angle = int(view_angle)
            light_color.save(
                os.path.join(light_save_folder,
                             f'{view_angle}_{sh_id}_light.png'))
