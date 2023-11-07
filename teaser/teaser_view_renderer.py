import os
import numpy as np
from implicit.implicit_render_prt import get_view_matrix
import math
import trimesh
import moderngl
from pyrr import Matrix44, Matrix33
from PIL import Image
from tqdm import tqdm
import argparse
from icecream import ic

datasets = ['thuman', 'twindom']

view_angle_map = {
    '0478': -150,
    '0537': -200,
    '0549': 130,
    '126111536790861': -90,
    '126211540489441': -120,
    '0261': -110,
    '0282': -60,
    '0298': 110,
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_folder', type=str, help='Path to data folder.')
    parser.add_argument('--out',
                        type=str,
                        default='$HOME/dataset/paper_teaser/ablation_fuse',
                        help='Path to output folder.')
    parser.add_argument('--free', action='store_true', help='Free data sample.')
    args = parser.parse_args()
    data_tag = "" if args.free else "_" + args.data_folder.split('/')[-1]

    ctx = moderngl.create_context(standalone=True, backend='egl')
    ctx.enable(moderngl.DEPTH_TEST)
    ctx.gc_mode = "auto"

    prog = ctx.program(vertex_shader='''
        #version 330

        uniform mat4 mvp;
        uniform mat3 rot;

        layout (location = 0) in vec3 in_pos;
        layout (location = 1) in vec3 in_vert_color;
        layout (location = 2) in vec3 in_vert_normal;

        out vec3 vert_color;
        out vec3 vert_normal;
        void main() {
            gl_Position = mvp * vec4(in_pos, 1.0);
            vert_color = in_vert_color;
            vert_normal = 0.5 * (rot * in_vert_normal + 1.0);
        }
    ''',
                       fragment_shader='''
        #version 330
        in vec3 vert_color;
        in vec3 vert_normal;
        layout (location = 0) out vec4 albedo;
        layout (location = 1) out vec4 normal;
        void main() {
            albedo = vec4(vert_color, 1.0);
            normal = vec4(vert_normal, 1.0);
        }
    ''')

    ortho_ratio = 0.6
    proj = Matrix44.orthogonal_projection(-ortho_ratio, ortho_ratio,
                                          -ortho_ratio, ortho_ratio, 0.01,
                                          100.0)
    width = 512
    height = 512
    color_attachments = [
        ctx.renderbuffer((width, height), 4, samples=8),
        ctx.renderbuffer((width, height), 4, samples=8)
    ]
    fbo = ctx.framebuffer(color_attachments=color_attachments,
                          depth_attachment=ctx.depth_renderbuffer(
                              (width, height), samples=8))

    color_attachments2 = [
        ctx.renderbuffer((width, height), 4),
        ctx.renderbuffer((width, height), 4)
    ]
    fbo2 = ctx.framebuffer(color_attachments=color_attachments2,
                           depth_attachment=ctx.depth_renderbuffer(
                               (width, height)))

    def render(dataset_folder, save_folder):
        for model_name in tqdm(os.listdir(dataset_folder)):
            model_path = os.path.join(dataset_folder, model_name,
                                      f"{model_name}.obj")
            vertex_color_path = os.path.join(dataset_folder, model_name,
                                             "vertex_color.npy")

            mesh: trimesh.Trimesh = trimesh.load(
                model_path,
                process=False,
                maintain_order=True,
            )
            vertex_color = np.load(vertex_color_path)
            vertices = mesh.vertices
            aabb_min = np.min(vertices, axis=0).reshape(1, -1)
            aabb_max = np.max(vertices, axis=0).reshape(1, -1)
            center = 0.5 * (aabb_max + aabb_min)
            scale = 2.0 * np.max(aabb_max - center)
            vertices = (vertices - center) / scale

            per_face_vertices = vertices[mesh.faces].reshape(-1, 3)
            vbo_vert = ctx.buffer(per_face_vertices.astype('f4'))

            per_face_vertex_color = vertex_color[mesh.faces].reshape(-1, 3)
            vbo_vc = ctx.buffer(per_face_vertex_color.astype('f4'))

            per_face_vertex_normal = mesh.vertex_normals[mesh.faces].reshape(
                -1, 3)
            vbo_vn = ctx.buffer(per_face_vertex_normal.astype('f4'))

            vao = ctx.vertex_array(prog, [(vbo_vert, '3f', 'in_pos'),
                                          (vbo_vc, '3f', 'in_vert_color'),
                                          (vbo_vn, '3f', 'in_vert_normal')])

            def render_pass(init_angle, view_angle):
                eye = [
                    math.cos(math.radians(-view_angle)), 0.0,
                    math.sin(math.radians(-view_angle))
                ]
                view = get_view_matrix(eye)
                model = Matrix44.from_y_rotation(math.radians(init_angle))
                mvp = (proj * view * model).astype('f4')
                rot = Matrix33.from_matrix44(view * model).astype('f4')

                prog['mvp'].write(mvp)
                prog['rot'].write(rot)

                fbo.use()
                fbo.clear(red=1., green=1., blue=1.)
                vao.render()

                ctx.copy_framebuffer(fbo2, fbo)

                albedo_data = fbo2.read(components=3, attachment=0)
                image_albedo = Image.frombytes(
                    'RGB', fbo2.size,
                    albedo_data).transpose(Image.Transpose.FLIP_TOP_BOTTOM)

                normal_data = fbo2.read(components=3, attachment=1)
                image_normal = Image.frombytes(
                    'RGB', fbo2.size,
                    normal_data).transpose(Image.Transpose.FLIP_TOP_BOTTOM)

                return image_albedo, image_normal

            image_tags = ['albedo', 'normal']
            for i in range(180):
                render_results = render_pass(view_angle_map[model_name], i * 2)

                for (image_tag, image) in zip(image_tags, render_results):
                    save_path = os.path.join(
                        save_folder, f"{image_tag}{data_tag}_%04d.png" % i)
                    image = image.crop((128, 0, 128 + 256, 512))
                    image.save(save_path)

            for image_tag in image_tags:
                cmd = f"gifski -H {height} -Q 100 -o {save_folder}/{model_name}_{image_tag}{data_tag}.gif {save_folder}/{image_tag}{data_tag}_*.png"
                os.system(cmd)
                cmd = f'rm %s/{image_tag}{data_tag}_*.png' % save_folder
                os.system(cmd)

    if args.free:
        dataset_folder = args.data_folder
        save_folder = args.out
        if not os.path.exists(save_folder):
            os.mkdir(save_folder)
        render(dataset_folder, save_folder)
    else:
        for dataset in datasets:
            dataset_folder = os.path.join(args.data_folder, dataset)
            save_folder = os.path.join(args.out, dataset)
            if not os.path.exists(save_folder):
                os.mkdir(save_folder)
            render(dataset_folder, save_folder)
