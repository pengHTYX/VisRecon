import os
from implicit.implicit_render_prt import get_view_matrix
import math
import trimesh
import moderngl
from pyrr import Matrix44, Matrix33
from PIL import Image
from tqdm import tqdm
import igl

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

datasets = ['vis_fuse', 'pifuhd', 'function4D', 'ipnet_res', 'multipifu_rgbd']

if __name__ == '__main__':

    render_data_folder = '$HOME/dataset/paper_validation_results/comparison/'
    render_save_folder = '$HOME/dataset/paper_teaser/comparison'

    if not os.path.exists(render_save_folder):
        os.mkdir(render_save_folder)

    ctx = moderngl.create_context(standalone=True, backend='egl')
    ctx.enable(moderngl.DEPTH_TEST)
    ctx.gc_mode = "auto"

    prog = ctx.program(vertex_shader='''
        #version 330

        uniform mat4 mvp;
        uniform mat3 rot;

        layout (location = 0) in vec3 in_pos;
        layout (location = 1) in vec3 in_vert_normal;

        out vec3 vert_color;
        out vec3 vert_normal;
        void main() {
            gl_Position = mvp * vec4(in_pos, 1.0);
            vert_normal = 0.5 * (rot * in_vert_normal + 1.0);
        }
    ''',
                       fragment_shader='''
        #version 330
        in vec3 vert_color;
        in vec3 vert_normal;
        layout (location = 0) out vec4 normal;
        void main() {
            normal = vec4(vert_normal, 1.0);
        }
    ''')

    ortho_ratio = 0.55
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

    for dataset in datasets:
        dataset_folder = os.path.join(render_data_folder, dataset)
        for model in tqdm(os.listdir(dataset_folder)):
            model_name = model.split('.')[0]
            model_path = os.path.join(dataset_folder, model)

            v, f = igl.read_triangle_mesh(model_path)
            mesh: trimesh.Trimesh = trimesh.Trimesh(vertices=v, faces=f)

            per_face_vertices = mesh.vertices[mesh.faces].reshape(-1, 3)
            vbo_vert = ctx.buffer(per_face_vertices.astype('f4'))

            per_face_vertex_normal = mesh.vertex_normals[mesh.faces].reshape(
                -1, 3)
            vbo_vn = ctx.buffer(per_face_vertex_normal.astype('f4'))

            vao = ctx.vertex_array(prog, [(vbo_vert, '3f', 'in_pos'),
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

                normal_data = fbo2.read(components=3, attachment=0)
                image_normal = Image.frombytes(
                    'RGB', fbo2.size,
                    normal_data).transpose(Image.Transpose.FLIP_TOP_BOTTOM)

                return image_normal

            for i in range(180):
                image = render_pass(view_angle_map[model_name], i * 2)
                image = image.crop((128, 0, 128 + 256, 512))
                save_path = os.path.join(render_save_folder,
                                         f"{model_name}_%04d.png" % i)
                image.save(save_path)

            cmd = f"gifski -H {height} -Q 100 -o {render_save_folder}/{model_name}_{dataset}.gif {render_save_folder}/{model_name}_*.png"
            os.system(cmd)
            cmd = f'rm %s/{model_name}_*.png' % render_save_folder
            os.system(cmd)
