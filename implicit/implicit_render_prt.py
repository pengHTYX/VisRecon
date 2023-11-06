from pyrr import Matrix44, matrix33
import moderngl
import numpy as np
from PIL import Image
import os
import math
from tqdm import tqdm
from implicit.implicit_prt_gen import fibonacci_sphere, getSHCoeffs


# Copied from: https://github.com/shunsukesaito/PIFu/blob/master/apps/render_data.py
def rotateSH(SH, R):
    SHn = SH

    # 1st order
    SHn[1] = R[1, 1] * SH[1] - R[1, 2] * SH[2] + R[1, 0] * SH[3]
    SHn[2] = -R[2, 1] * SH[1] + R[2, 2] * SH[2] - R[2, 0] * SH[3]
    SHn[3] = R[0, 1] * SH[1] - R[0, 2] * SH[2] + R[0, 0] * SH[3]

    # 2nd order
    SHn[4:, 0] = rotateBand2(SH[4:, 0], R)
    SHn[4:, 1] = rotateBand2(SH[4:, 1], R)
    SHn[4:, 2] = rotateBand2(SH[4:, 2], R)

    return SHn


def rotateBand2(x, R):
    s_c3 = 0.94617469575
    s_c4 = -0.31539156525
    s_c5 = 0.54627421529

    s_c_scale = 1.0 / 0.91529123286551084
    s_c_scale_inv = 0.91529123286551084

    s_rc2 = 1.5853309190550713 * s_c_scale
    s_c4_div_c3 = s_c4 / s_c3
    s_c4_div_c3_x2 = (s_c4 / s_c3) * 2.0

    s_scale_dst2 = s_c3 * s_c_scale_inv
    s_scale_dst4 = s_c5 * s_c_scale_inv

    sh0 = x[3] + x[4] + x[4] - x[1]
    sh1 = x[0] + s_rc2 * x[2] + x[3] + x[4]
    sh2 = x[0]
    sh3 = -x[3]
    sh4 = -x[1]

    r2x = R[0][0] + R[0][1]
    r2y = R[1][0] + R[1][1]
    r2z = R[2][0] + R[2][1]

    r3x = R[0][0] + R[0][2]
    r3y = R[1][0] + R[1][2]
    r3z = R[2][0] + R[2][2]

    r4x = R[0][1] + R[0][2]
    r4y = R[1][1] + R[1][2]
    r4z = R[2][1] + R[2][2]

    sh0_x = sh0 * R[0][0]
    sh0_y = sh0 * R[1][0]
    d0 = sh0_x * R[1][0]
    d1 = sh0_y * R[2][0]
    d2 = sh0 * (R[2][0] * R[2][0] + s_c4_div_c3)
    d3 = sh0_x * R[2][0]
    d4 = sh0_x * R[0][0] - sh0_y * R[1][0]

    sh1_x = sh1 * R[0][2]
    sh1_y = sh1 * R[1][2]
    d0 += sh1_x * R[1][2]
    d1 += sh1_y * R[2][2]
    d2 += sh1 * (R[2][2] * R[2][2] + s_c4_div_c3)
    d3 += sh1_x * R[2][2]
    d4 += sh1_x * R[0][2] - sh1_y * R[1][2]

    sh2_x = sh2 * r2x
    sh2_y = sh2 * r2y
    d0 += sh2_x * r2y
    d1 += sh2_y * r2z
    d2 += sh2 * (r2z * r2z + s_c4_div_c3_x2)
    d3 += sh2_x * r2z
    d4 += sh2_x * r2x - sh2_y * r2y

    sh3_x = sh3 * r3x
    sh3_y = sh3 * r3y
    d0 += sh3_x * r3y
    d1 += sh3_y * r3z
    d2 += sh3 * (r3z * r3z + s_c4_div_c3_x2)
    d3 += sh3_x * r3z
    d4 += sh3_x * r3x - sh3_y * r3y

    sh4_x = sh4 * r4x
    sh4_y = sh4 * r4y
    d0 += sh4_x * r4y
    d1 += sh4_y * r4z
    d2 += sh4 * (r4z * r4z + s_c4_div_c3_x2)
    d3 += sh4_x * r4z
    d4 += sh4_x * r4x - sh4_y * r4y

    dst = x
    dst[0] = d0
    dst[1] = -d1
    dst[2] = d2 * s_scale_dst2
    dst[3] = -d3
    dst[4] = d4 * s_scale_dst4

    return dst


def get_view_matrix(eye, offset=0.0):
    try:
        lookat = Matrix44.look_at(
            eye,
            [0.0, offset, 0.0],
            [0.0, 1.0, 0.0],
        )
    except Exception as e:
        None
    return lookat


def concat_image(images):
    if len(images) == 1:
        return images[0]

    img_cat = Image.new(
        'RGBA', (sum([image.width for image in images]), images[0].height))
    for i in range(len(images)):
        image = images[i]
        img_cat.paste(image, (i * image.width, 0))
    return img_cat


# Modified from: https://github.com/shunsukesaito/PIFu/tree/master/lib/renderer/gl
def render_prt(model_name,
               vertices,
               faces,
               gen_prt,
               vertex_color,
               out_folder,
               gt_prt=None,
               unshaded_prt=None,
               render_image_sample=False,
               image_render_count=10,
               render_rot_gif=False,
               render_still_gif=False,
               render_lighting=False,
               gif_frames=180,
               res=1024,
               fix_lighting=True,
               gif_init_rot=0.,
               image_concate=False):
    if not render_image_sample and not render_rot_gif and not render_still_gif and not render_lighting:
        return

    aabb_min = np.min(vertices, axis=0).reshape(1, -1)
    aabb_max = np.max(vertices, axis=0).reshape(1, -1)
    center = 0.5 * (aabb_max + aabb_min)
    scale = 2.0 * np.max(aabb_max - center)
    vertices = (vertices - center) / scale

    ctx = moderngl.create_context(standalone=True, backend='egl')
    ctx.enable(moderngl.DEPTH_TEST)
    ctx.gc_mode = "auto"

    per_face_vertices = vertices[faces].reshape(-1, 3)
    vbo_vert = ctx.buffer(per_face_vertices.astype('f4'))

    per_face_gen_prt = gen_prt[faces].reshape(-1, 9)
    vbo_gen_sh = ctx.buffer(per_face_gen_prt.astype('f4'))

    if gt_prt is not None:
        per_face_gt_prt = gt_prt[faces].reshape(-1, 9)
        vbo_gt_sh = ctx.buffer(per_face_gt_prt.astype('f4'))

    if unshaded_prt is not None:
        per_face_unshaded_prt = unshaded_prt[faces].reshape(-1, 9)
        vbo_unshaded_sh = ctx.buffer(per_face_unshaded_prt.astype('f4'))

    ortho_ratio = 0.7
    proj = Matrix44.orthogonal_projection(-ortho_ratio, ortho_ratio,
                                          -ortho_ratio, ortho_ratio, 0.01,
                                          100.0)
    width = res
    height = res

    color_attachments = [
        ctx.texture((width, height), 4),
        ctx.texture((width, height), 4),
        ctx.texture((width, height), 4)
    ]
    fbo = ctx.framebuffer(color_attachments=color_attachments,
                          depth_attachment=ctx.depth_renderbuffer(
                              (width, height)))

    def get_images_from_fbo():
        light_data = fbo.read(components=4, attachment=0)
        image_light = Image.frombytes('RGBA', fbo.size, light_data).transpose(
            Image.Transpose.FLIP_TOP_BOTTOM)

        texture_data = fbo.read(components=4, attachment=1)
        image_color = Image.frombytes('RGBA', fbo.size, texture_data).transpose(
            Image.Transpose.FLIP_TOP_BOTTOM)

        albedo_data = fbo.read(components=4, attachment=2)
        image_albedo = Image.frombytes('RGBA', fbo.size, albedo_data).transpose(
            Image.Transpose.FLIP_TOP_BOTTOM)
        return image_light, image_color, image_albedo

    env_shs = np.load('implicit/env_sh.npy')

    if render_lighting:
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

    if vertex_color is not None:
        vert = open("implicit/shaders/prt_vc_simple.vs").read()
        frag = open("implicit/shaders/prt_vc_simple.fs").read()
        prog = ctx.program(vertex_shader=vert, fragment_shader=frag)

        per_face_vc = vertex_color[faces].reshape(-1, 3)
        vbo_vc = ctx.buffer(per_face_vc.astype('f4'))
        vao_gen = ctx.vertex_array(prog, [(vbo_vert, '3f', 'in_pos'),
                                          (vbo_vc, '3f', 'in_vert_color'),
                                          (vbo_gen_sh, '9f', 'in_sh')])

        if gt_prt is not None:
            vao_gt = ctx.vertex_array(prog, [(vbo_vert, '3f', 'in_pos'),
                                             (vbo_vc, '3f', 'in_vert_color'),
                                             (vbo_gt_sh, '9f', 'in_sh')])

        if unshaded_prt is not None:
            vao_unshaded = ctx.vertex_array(prog,
                                            [(vbo_vert, '3f', 'in_pos'),
                                             (vbo_vc, '3f', 'in_vert_color'),
                                             (vbo_unshaded_sh, '9f', 'in_sh')])
    else:
        vert = open("implicit/shaders/prt.vs").read()
        frag = open("implicit/shaders/prt.fs").read()
        prog = ctx.program(vertex_shader=vert, fragment_shader=frag)
        vao_gen = ctx.vertex_array(prog, [(vbo_vert, '3f', 'in_pos'),
                                          (vbo_gen_sh, '9f', 'in_sh')])

        if gt_prt is not None:
            vao_gt = ctx.vertex_array(prog, [(vbo_vert, '3f', 'in_pos'),
                                             (vbo_gt_sh, '9f', 'in_sh')])

        if unshaded_prt is not None:
            vao_unshaded = ctx.vertex_array(prog,
                                            [(vbo_vert, '3f', 'in_pos'),
                                             (vbo_unshaded_sh, '9f', 'in_sh')])

    def render_pass(view_angle, env_sh):
        eye = [
            math.cos(math.radians(-view_angle)), 0.0,
            math.sin(math.radians(-view_angle))
        ]
        view = get_view_matrix(eye)
        mvp = (proj * view).astype('f4')

        render_result = {}

        if render_lighting:
            sphere_prog['mvp'].write(mvp)
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
            render_result['light_color'] = light_color

        fbo.use()
        fbo.clear()
        prog['mvp'].write(mvp)
        prog['env_sh'].write(env_sh.T.reshape(-1,).astype('f4'))

        vao_gen.render()
        light_gen, color_gen, albedo_gen = get_images_from_fbo()
        render_result['gen_light'] = light_gen
        render_result['gen_color'] = color_gen
        render_result['gen_albedo'] = albedo_gen

        if gt_prt is not None:
            fbo.clear()
            vao_gt.render()
            light_gt, color_gt, _ = get_images_from_fbo()
            if image_concate:
                light_gen_gt = concat_image([light_gen, light_gt])
                color_gen_gt = concat_image([color_gen, color_gt])
                render_result['gen_gt_light'] = light_gen_gt
                render_result['gen_gt_color'] = color_gen_gt
            else:
                render_result['gt_light'] = light_gt
                render_result['gt_color'] = color_gt

        if unshaded_prt is not None:
            fbo.clear()
            vao_unshaded.render()
            light_unshaded, color_unshaded, _ = get_images_from_fbo()
            if image_concate:
                light_gen_unshaded = concat_image([light_gen, light_unshaded])
                color_gen_unshaded = concat_image([color_gen, color_unshaded])
                render_result['gen_unshaded_light'] = light_gen_unshaded
                render_result['gen_unshaded_color'] = color_gen_unshaded
            else:
                render_result['unshaded_light'] = light_unshaded
                render_result['unshaded_color'] = color_unshaded

        return render_result

    if render_image_sample:
        delta_angle = 360 / image_render_count
        for i in tqdm(range(image_render_count)):
            if fix_lighting:
                sh = np.copy(env_shs[0])
            else:
                sh = np.copy(env_shs[i])

            render_result = render_pass(i * delta_angle, sh)

            for image_tag, image in render_result.items():
                image_save_path = os.path.join(
                    out_folder, f"{model_name}_{i}_{image_tag}.png")
                image.save(image_save_path)

    def gen_gif(image_tag, suffix):
        cmd = f"gifski -H {height} -Q 100 -o {out_folder}/{model_name}_{image_tag}_{suffix}.gif {out_folder}/{image_tag}_*.png"
        os.system(cmd)
        cmd = f'rm %s/{image_tag}_*.png' % out_folder
        os.system(cmd)

    if render_rot_gif:
        delta_angle = 360 / 180
        sh = env_shs[0]
        image_tag_set = set()
        for i in tqdm(range(180)):
            render_result = render_pass(gif_init_rot + i * delta_angle, sh)

            for image_tag, image in render_result.items():
                image_save_path = os.path.join(out_folder,
                                               f"{image_tag}_%04d.png" % i)
                image.save(image_save_path)
                image_tag_set.add(image_tag)

        for image_tag in image_tag_set:
            gen_gif(image_tag, "rot")

    if render_still_gif:
        for sh_idx in [2, 6, 14, 11]:
            image_tag_set = set()
            for i in tqdm(range(gif_frames)):
                sh = env_shs[sh_idx]
                R = matrix33.create_from_eulers([-0.05 * i, 0.01 * i, -0.1 * i])
                sh = rotateSH(np.copy(sh), R)
                render_result = render_pass(gif_init_rot, sh)

                for image_tag, image in render_result.items():
                    image_save_path = os.path.join(out_folder,
                                                   f"{image_tag}_%04d.png" % i)
                    image.save(image_save_path)
                    image_tag_set.add(image_tag)

            for image_tag in image_tag_set:
                gen_gif(image_tag, f"still_{sh_idx}")
