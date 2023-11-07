import os
import numpy as np
from implicit.implicit_render_prt import get_view_matrix
import math
import trimesh
import moderngl
from pyrr import Matrix44
from PIL import Image
from tqdm import tqdm
import argparse
from turbo_colormap import interpolate, turbo_colormap_data
from icecream import ic

datasets = ['thuman', 'twindom']

view_angle_map = {
    '0261': -110,
    '0282': -60,
    '0298': 110,
    '0289': -15,
    '0307': 90,
    '0309': -120,
    '0277': -120,
    '0295': 200,
    '0269': 160,
    '0319': 100,
    '0273': 0,
    '0288': -100,
    '0553': 80,
    '126111539897023': -90,
    '126111551391968': -100,
    '140711542935158': -120,
    '141311548775566': -120,
    '141511557173943': -60,
    '141511558561737': -110,
    '126111539903174': -80,
    '126111555700499': -90,
    '126111539905058': -90,
    '126111539908752': -90,
    '126111535847796': -90,
    '126111539902286': -90,
    '126111539896120': -90,
    '126111539903577': -90,
    '126111539897215': -90,
    '126111539896832': -90,
    '126111541531875': -90
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_folder', type=str, help='Path to data folder.')
    parser.add_argument('--out',
                        type=str,
                        default='$HOME/dataset/paper_teaser/ablation_3d',
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
        in vec3 vert_normal;
        layout (location = 0) out vec4 albedo;
        void main() {
            albedo = vec4(vert_color, 1.0);
        }
    ''')

    ortho_ratio = 0.6
    proj = Matrix44.orthogonal_projection(-ortho_ratio, ortho_ratio,
                                          -ortho_ratio, ortho_ratio, 0.01,
                                          100.0)
    width = 512
    height = 512
    fbo = ctx.framebuffer(
        color_attachments=[ctx.renderbuffer((width, height), 4, samples=8)],
        depth_attachment=ctx.depth_renderbuffer((width, height), samples=8))

    fbo2 = ctx.framebuffer(color_attachments=[ctx.texture((width, height), 4)],
                           depth_attachment=ctx.depth_renderbuffer(
                               (width, height)))

    def render(dataset_folder, save_folder):
        for model_name in tqdm(os.listdir(dataset_folder)):
            model_path = os.path.join(dataset_folder, model_name,
                                      f"{model_name}.obj")
            vis_gen = np.load(
                os.path.join(dataset_folder, model_name, "vis_gen.npy")) > 0.5
            vis_gt = np.load(
                os.path.join(dataset_folder, model_name, "viz_gt.npy"))

            mesh: trimesh.Trimesh = trimesh.load(
                model_path,
                process=False,
                maintain_order=True,
            )
            vertices = mesh.vertices
            aabb_min = np.min(vertices, axis=0).reshape(1, -1)
            aabb_max = np.max(vertices, axis=0).reshape(1, -1)
            center = 0.5 * (aabb_max + aabb_min)
            scale = 2.0 * np.max(aabb_max - center)
            vertices = (vertices - center) / scale

            vis_error = np.sum(vis_gen != vis_gt, axis=1) / 64

            vertex_color = np.array(
                [interpolate(turbo_colormap_data, err) for err in vis_error])

            per_face_vertices = vertices[mesh.faces].reshape(-1, 3)
            vbo_vert = ctx.buffer(per_face_vertices.astype('f4'))

            per_face_vertex_color = vertex_color[mesh.faces].reshape(-1, 3)
            vbo_vc = ctx.buffer(per_face_vertex_color.astype('f4'))

            vao = ctx.vertex_array(prog, [(vbo_vert, '3f', 'in_pos'),
                                          (vbo_vc, '3f', 'in_vert_color')])

            def render_pass(init_angle, view_angle):
                eye = [
                    math.cos(math.radians(-view_angle)), 0.0,
                    math.sin(math.radians(-view_angle))
                ]
                view = get_view_matrix(eye)
                model = Matrix44.from_y_rotation(math.radians(init_angle))
                mvp = (proj * view * model).astype('f4')

                prog['mvp'].write(mvp)

                fbo.use()
                fbo.clear(red=1., green=1., blue=1.)
                vao.render()

                ctx.copy_framebuffer(fbo2, fbo)

                vis_error_data = fbo2.read(components=3, attachment=0)
                image_vis_error = Image.frombytes(
                    'RGB', fbo2.size,
                    vis_error_data).transpose(Image.Transpose.FLIP_TOP_BOTTOM)

                return image_vis_error

            image_tag = 'vis_error'
            for i in range(180):
                image_vis_error = render_pass(view_angle_map[model_name], i * 2)
                save_path = os.path.join(save_folder,
                                         f"{image_tag}{data_tag}_%04d.png" % i)
                image_vis_error.save(save_path)

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
