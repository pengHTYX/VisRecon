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

# This is an experiment script to render outline using two passes
# It was not used for figure generate and merely kept for future reference
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

    width = 512
    height = 512
    ctx = moderngl.create_context(standalone=True, backend='egl')
    ctx.enable(moderngl.DEPTH_TEST)
    ctx.gc_mode = "auto"

    prog_prt = ctx.program(vertex_shader='''
        #version 330

        uniform mat4 mvp;

        layout (location = 0) in vec3 in_pos;
        layout (location = 1) in vec3 in_vert_color;
        layout (location = 2) in float in_sh[9];

        out vec3 vert_color;
        out float sh[9];

        void main() {
            gl_Position = mvp * vec4(in_pos, 1.0);
            vert_color = in_vert_color;
            sh = in_sh;
        }
    ''',
                           fragment_shader='''
        #version 330

        uniform float env_sh[27];

        in vec3 vert_color;
        in float sh[9];

        layout (location = 0) out vec4 light_color;
        layout (location = 1) out vec4 frag_color;

        vec4 gammaCorrection(vec4 vec, float g) {
            return vec4(pow(vec.x, 1.0/g), pow(vec.y, 1.0/g), pow(vec.z, 1.0/g), vec.w);
        }

        void main() {
            light_color = vec4(0.0);
            for (int i = 0; i < 9; i++) {
                light_color.x += sh[i] * env_sh[i];
                light_color.y += sh[i] * env_sh[9 + i];
                light_color.z += sh[i] * env_sh[18 + i];
            }
            light_color.w = 1.0;
            light_color = gammaCorrection(light_color, 2.2);
            frag_color = light_color * vec4(vert_color, 1.0);
        }
    ''')

    env_shs = np.load('implicit/env_sh.npy')
    sh = np.copy(env_shs[0]).T.reshape(-1,).astype('f4')

    proj = Matrix44.perspective_projection(55, 1.0, 0.01, 100.0)

    fbo_prt = ctx.framebuffer(color_attachments=[
        ctx.texture((width, height), 4),
        ctx.texture((width, height), 4)
    ],
                              depth_attachment=ctx.depth_texture(
                                  (width, height)))

    fbo_screen_space = ctx.framebuffer(color_attachments=[
        ctx.texture((width, height), 4),
        ctx.texture((width, height), 4)
    ],
                                       depth_attachment=ctx.depth_renderbuffer(
                                           (width, height)))

    image_tags = ['light_color', 'color']

    scale = 2
    y_offset = -0.2
    z_offset = -1.9

    screen_space_prog = ctx.program(vertex_shader='''
        #version 330 core

        layout(location = 0) in vec2 in_pos;
        layout(location = 1) in vec2 in_uv;

        out vec2 uv;

        void main() {
            uv = in_uv;
            gl_Position = vec4(in_pos, 0.0, 1.0);
        }
    ''',
                                    fragment_shader='''
        #version 330 core
        layout (location = 0) out vec4 light_color;
        layout (location = 1) out vec4 frag_color;

        in vec2 uv;

        uniform sampler2D textures[3];

        void main() {
            float theDepth = texture(textures[0], uv).r;
            light_color = texture(textures[1], uv);
            frag_color = texture(textures[2], uv);

            if (theDepth != 1.0) {
                float outlineWidth = 20.0;
                bool isNear = false;
                float nearDepth;
                int radSq = int(round(sqrt(outlineWidth)));
                int rad = int(ceil(outlineWidth));

                for (int i = -rad; (!isNear) && (i <= rad); ++i) {
                    for (int j = -rad; (!isNear) && (j <= rad); ++j) {
                        int sqSum = i * i + j * j;
                        if ((sqSum != 0) && (sqSum <= radSq)) {
                            vec2 testPt = uv + vec2(float(i) / 512.0, float(j) / 512.0);
                            theDepth = texture(textures[0], testPt).r;
                            if (theDepth == 1.0) {
                                isNear = true;
                                nearDepth = theDepth;
                            }
                        }
                    }
                }

                if (isNear) {
                    float scale = 2.0;
                    float alpha = texture(textures[1], uv + vec2(scale, 0.0) / 512.0).a
                            + texture(textures[1], uv - vec2(scale, 0.0) / 512.0).a
                            + texture(textures[1], uv + vec2(0.0, scale) / 512.0).a
                            + texture(textures[1], uv - vec2(0.0, scale) / 512.0).a;

                    light_color.a = 0.20 * alpha;
                    frag_color.a = 0.20 * alpha;

                    // light_color = vec4(1.0);
                    // frag_color = vec4(1.0);
                    // gl_FragDepth = nearDepth;
                }
            }
        }
    ''')

    vbo_vert_screen_space = ctx.buffer(
        np.array([[-1.0, -1.0], [3.0, -1.0], [-1.0, 3.0]]).astype('f4'))
    vbo_uv_screen_space = ctx.buffer(
        np.array([[0.0, -1.0], [2.0, -1.0], [0.0, 1.0]]).astype('f4'))
    ibo_screen_space = ctx.buffer(np.array([0, 1, 2]).astype('i4'))

    # fullscreen triangle
    vao_screen_space = ctx.vertex_array(
        screen_space_prog, [(vbo_vert_screen_space, '2f', 'in_pos'),
                            (vbo_uv_screen_space, '2f', 'in_uv')],
        ibo_screen_space)

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

            v[:, 1] += y_offset
            v[:, 2] += z_offset
            v /= scale

            per_face_vertices = v[f].reshape(-1, 3)
            vbo_vert = ctx.buffer(per_face_vertices.astype('f4'))

            per_face_vertex_color = vertex_color[f].reshape(-1, 3)
            vbo_vc = ctx.buffer(per_face_vertex_color.astype('f4'))

            per_face_prt = prt[f].reshape(-1, 9)
            vbo_prt = ctx.buffer(per_face_prt.astype('f4'))

            vao_prt = ctx.vertex_array(prog_prt,
                                       [(vbo_vert, '3f', 'in_pos'),
                                        (vbo_vc, '3f', 'in_vert_color'),
                                        (vbo_prt, '9f', 'in_sh')])

            model = Matrix44.from_y_rotation(0)

            def render_pass(view_angle):
                eye = [
                    math.cos(math.radians(-view_angle)), 0.0,
                    math.sin(math.radians(-view_angle))
                ]
                view = get_view_matrix(eye)

                mvp = (proj * view * model).astype('f4')

                prog_prt['mvp'].write(mvp)
                prog_prt['env_sh'].write(sh)

                fbo_prt.use()
                fbo_prt.clear()
                vao_prt.render()

                screen_space_prog['textures'].value = [0, 1, 2]

                fbo_prt.depth_attachment.use(0)
                fbo_prt.color_attachments[0].use(location=1)
                fbo_prt.color_attachments[1].use(location=2)

                fbo_screen_space.use()
                fbo_screen_space.clear()
                vao_screen_space.render()

                out_fbo = fbo_screen_space
                # out_fbo = fbo_prt
                light_color_data = out_fbo.read(components=4, attachment=0)
                image_light_color = Image.frombytes(
                    'RGBA', out_fbo.size,
                    light_color_data).transpose(Image.Transpose.FLIP_TOP_BOTTOM)

                color_data = out_fbo.read(components=4, attachment=1)
                image_color = Image.frombytes(
                    'RGBA', out_fbo.size,
                    color_data).transpose(Image.Transpose.FLIP_TOP_BOTTOM)

                return image_light_color, image_color

            render_results = render_pass(30)

            for (image_tag, image) in zip(image_tags, render_results):
                save_path = os.path.join(save_folder,
                                         f"{model_name}_{image_tag}.png")
                image.save(save_path)

        for image_tag in image_tags:
            cmd = f"gifski -H {height} -Q 100 -o {save_folder}/{image_tag}.gif {save_folder}/realworld_*_{image_tag}.png"
            os.system(cmd)
            cmd = f'rm %s/realworld_*_{image_tag}.png' % save_folder
            os.system(cmd)

    render(args.data_folder, save_folder)
