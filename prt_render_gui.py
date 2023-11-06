import trimesh
import moderngl
import moderngl_window as mglw
from moderngl_window import geometry
from moderngl_window.integrations.imgui import ModernglWindowRenderer
from moderngl_window.scene.camera import OrbitCamera
from pyrr import Matrix44, Matrix33
import imgui
from implicit.implicit_prt_gen import fibonacci_sphere, getSHCoeffs
import numpy as np
import math
from pyrr import matrix44, matrix33
from thuman import sample_color_bilinear
import os
import cv2
import argparse
from icecream import ic

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
# Modify this to folder of output with saved PRT
model_root_folder = "out/vis_fuse/test/4"


class WindowEvents(mglw.WindowConfig):
    gl_version = (3, 3)
    title = "PRT renderer"
    aspect_ratio = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        imgui.create_context()
        self.wnd.ctx.error
        self.imgui = ModernglWindowRenderer(self.wnd)

        io = imgui.get_io()
        io.config_windows_move_from_title_bar_only = True

        self.camera = OrbitCamera(near=0.01, aspect_ratio=1.0)
        self.camera.mouse_sensitivity = 2.5
        self.camera.zoom_sensitivity = 0.25

        self.ortho_ratio = 1.0
        self.ortho_proj = Matrix44.orthogonal_projection(
            -self.ortho_ratio, self.ortho_ratio, -self.ortho_ratio,
            self.ortho_ratio, 0.01, 100.0)
        self.use_ortho = False

        self.width = 1024
        self.height = 1024
        self.width_gui = 1024
        self.height_gui = 1024
        self.ctx.gc_mode = "auto"

        self.model_root_folder = os.path.expandvars(model_root_folder)
        self.model_list = os.listdir(self.model_root_folder)
        self.current_model = 0
        self.model_table = {}

        self.light_dirs, phi, theta = fibonacci_sphere(256)
        self.light_sh_basis = getSHCoeffs(2, phi, theta)

        cubemap_folder = "implicit/cubemaps"
        self.cubemap_folder = os.path.expandvars(cubemap_folder)
        self.cubemap_list = os.listdir(self.cubemap_folder)
        self.cubemap_list = [
            cubemap_folder for cubemap_folder in self.cubemap_list
            if os.path.isdir(os.path.join(self.cubemap_folder, cubemap_folder))
        ]
        self.current_cubemap = 0
        self.cubemap_texture = None
        self.cubemap_sh = None
        self.load_cubemap()
        self.cube = geometry.cube(size=(20, 20, 20))
        self.cubemap_prog = self.ctx.program(vertex_shader='''
                #version 330

                uniform mat4 mvp;
                layout (location = 0) in vec3 in_position;
                out vec3 pos;
                void main() {
                    gl_Position =  mvp * vec4(in_position, 1.0);
                    pos = in_position.xyz;
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

        vert = open("implicit/shaders/prt_vc.vs").read()
        frag = open("implicit/shaders/prt_vc.fs").read()
        self.prog = self.ctx.program(vertex_shader=vert, fragment_shader=frag)

        self.T = np.eye(4, dtype=np.float32)
        self.T[1, 1] = -1

        # n x 9 x 3
        self.env_shs = np.load('implicit/env_sh.npy')

        self.row = 0.0
        self.yaw = 0.0
        self.pitch = 0.0
        self.R_sh = matrix33.create_from_eulers(
            [self.row, self.yaw, self.pitch])
        self.R_cube = matrix33.create_from_eulers(
            [self.row, self.yaw, self.pitch])

        color_attachments = [
            self.ctx.texture((self.width, self.height), 4),
            self.ctx.texture((self.width, self.height), 4),
            self.ctx.texture((self.width, self.height), 4)
        ]

        self.fbo = self.ctx.framebuffer(
            color_attachments=color_attachments,
            depth_attachment=self.ctx.depth_renderbuffer(
                (self.width, self.height)))
        # Ensure imgui knows about this texture
        # This is the color layer in the framebuffer
        self.imgui.register_texture(self.fbo.color_attachments[0])
        self.imgui.register_texture(self.fbo.color_attachments[1])
        self.imgui.register_texture(self.fbo.color_attachments[2])
        self.image_view_hovered = False
        self.render_mode = ['Light', 'Color', 'Albedo']
        self.current_render_mode = 0
        self.shading_mode = ['Gen', 'Gt', 'No vis']
        self.current_shading_mode = 0
        self.lighting_mode = ['SH', 'Analytic', 'Cubemap']
        self.current_lighting_mode = 2
        self.light_sh = self.cubemap_sh
        self.sh_id = 0

        self.sample_size = 64
        self.dirs, phi, theta = fibonacci_sphere(self.sample_size)
        self.sh_basis = getSHCoeffs(2, phi, theta)

        self.analytical_sh = np.array([
            8.657540000000000235e-01, 8.801959999999999784e-01,
            9.471540000000000514e-01, -2.056330000000000102e-01,
            -2.112150000000000138e-01, -2.505040000000000044e-01,
            3.495840000000000058e-01, 3.650840000000000196e-01,
            4.632530000000000259e-01, -7.896219999999999628e-02,
            -7.507339999999999847e-02, -9.140500000000000014e-02,
            7.769099999999999617e-02, 8.107709999999999917e-02,
            9.222840000000000205e-02, -4.026850000000000152e-01,
            -4.083399999999999808e-01, -4.627629999999999799e-01,
            3.289070000000000049e-01, 3.286560000000000037e-01,
            3.707250000000000267e-01, -1.318149999999999877e-01,
            -1.409920000000000062e-01, -1.582979999999999943e-01,
            -9.922930000000000639e-02, -9.836860000000000048e-02,
            -1.079750000000000015e-01
        ]).reshape(9, 3)

        vert_hdri = open("implicit/shaders/hdri.vs").read()
        frag_hdri = open("implicit/shaders/hdri.fs").read()
        self.hdri_prog = self.ctx.program(vertex_shader=vert_hdri,
                                          fragment_shader=frag_hdri)
        self.hdri_fbo = self.ctx.framebuffer(
            color_attachments=[self.ctx.texture((512, 256), 4)])
        vbo_vert_screen_space = self.ctx.buffer(
            np.array([[-1.0, -1.0], [3.0, -1.0], [-1.0, 3.0]]).astype('f4'))
        vbo_uv_screen_space = self.ctx.buffer(
            np.array([[0.0, 1.0], [2.0, 1.0], [0.0, -1.0]]).astype('f4'))
        ibo_screen_space = self.ctx.buffer(np.array([0, 1, 2]).astype('i4'))
        self.hdri_vao = self.ctx.vertex_array(
            self.hdri_prog, [(vbo_vert_screen_space, '2f', 'in_pos'),
                             (vbo_uv_screen_space, '2f', 'in_uv')],
            ibo_screen_space)
        self.imgui.register_texture(self.hdri_fbo.color_attachments[0])

        self.sphere_prog = self.ctx.program(vertex_shader='''
                #version 330

                uniform mat4 mvp;

                layout (location = 0) in vec3 in_pos;

                out vec3 vert_color;
                out vec3 pos;
                void main() {
                    gl_Position = mvp * vec4(in_pos, 1.0);
                    pos = in_pos;
                }
            ''',
                                            fragment_shader='''
                #version 330

                #define PI 3.1415926538

                in vec3 vert_color;
                in vec3 pos;

                uniform sampler2D textures[1];

                layout (location = 0) out vec4 frag_hdri;

                void main() {
                    vec3 dir = pos.xzy;
                    float theta = atan(dir.y, dir.x);
                    float phi = atan(sqrt(dir.x * dir.x + dir.y * dir.y) / dir.z);
                    vec2 uv = vec2(theta / 2.0 / PI, phi / PI);
                    frag_hdri = texture(textures[0], uv);
                }
            ''')

        fibonacci_sphere_index = np.load('implicit/face_256.npy')
        self.sphere_vbo = self.ctx.buffer(self.light_dirs.astype('f4'))
        self.sphere_ibo = self.ctx.buffer(fibonacci_sphere_index.astype('i4'))
        self.sphere_vao = self.ctx.vertex_array(
            self.sphere_prog, [(self.sphere_vbo, '3f', 'in_pos')],
            self.sphere_ibo)

        self.sphere_fbo = self.ctx.framebuffer(
            color_attachments=[
                self.ctx.texture((self.width, self.height), 4),
                self.ctx.texture((self.width, self.height), 4)
            ],
            depth_attachment=self.ctx.depth_renderbuffer(
                (self.width, self.height)))
        self.imgui.register_texture(self.sphere_fbo.color_attachments[0])

        self.load_model(0)

    def load_cubemap(self):
        cubemap_name = self.cubemap_list[self.current_cubemap]
        cubemap_path = os.path.abspath(
            os.path.join(self.cubemap_folder, cubemap_name))
        self.cubemap_texture = self.load_texture_cube(
            neg_x=os.path.join(cubemap_path, 'nx.png'),
            neg_y=os.path.join(cubemap_path, 'ny.png'),
            neg_z=os.path.join(cubemap_path, 'nz.png'),
            pos_x=os.path.join(cubemap_path, 'px.png'),
            pos_y=os.path.join(cubemap_path, 'py.png'),
            pos_z=os.path.join(cubemap_path, 'pz.png'))

        # self.sample_light_color()
        sh_path = os.path.join(self.cubemap_folder, cubemap_name, 'sh.txt')
        with open(sh_path) as f:
            lines = f.readlines()
            data = [l.split('(')[1].split(')')[0].split(',') for l in lines]
            self.cubemap_sh = np.array(data).astype('f4')

    def sample_light_color(self):
        cubemap_name = self.cubemap_list[self.current_cubemap]
        hdr_path = os.path.join(self.cubemap_folder, cubemap_name,
                                f'{cubemap_name}.hdr')
        hdr_img = cv2.imread(hdr_path, cv2.IMREAD_UNCHANGED)
        hdr_img = 1.0 - np.exp(-0.1 * hdr_img)
        xx, yy, zz = self.light_dirs[:,
                                     0], self.light_dirs[:,
                                                         1], self.light_dirs[:,
                                                                             2]
        d = np.sqrt(xx * xx + yy * yy)
        r = (1.0 / np.pi / 2.0) * np.arccos(zz) / d
        coord_x = (0.5 + xx * r)
        coord_y = (0.5 + yy * r)

        sample_coord = np.stack([coord_x, coord_y]).T
        sample_colors = sample_color_bilinear(hdr_img, sample_coord)

        self.cubemap_sh = 4 * np.pi * (
            self.light_sh_basis.T @ sample_colors).astype('f4') / 256

    def update_ortho(self):
        self.ortho_proj = Matrix44.orthogonal_projection(
            -self.ortho_ratio, self.ortho_ratio, -self.ortho_ratio,
            self.ortho_ratio, 0.01, 100.0)

    def load_model(self, idx):
        model_name = self.model_list[idx]
        model_folder = os.path.join(self.model_root_folder, model_name)
        model_path = os.path.join(model_folder, f"{model_name}.obj")
        prt_gen_path = os.path.join(model_folder, "prt_gen.npy")
        prt_gt_path = os.path.join(model_folder, "prt_gt.npy")
        vertex_color_path = os.path.join(model_folder, "vertex_color.npy")

        if not os.path.exists(model_path) or not os.path.exists(
                prt_gen_path) or not os.path.exists(
                    prt_gen_path) or not os.path.exists(vertex_color_path):
            return

        mesh = trimesh.load(model_path,
                            process=False,
                            maintain_order=True,
                            use_embree=True)
        prt_gen = np.load(prt_gen_path)
        prt_gt = np.load(prt_gt_path)
        vertex_color = np.load(vertex_color_path)

        vertices = mesh.vertices
        faces = mesh.faces

        aabb_min = np.min(vertices, axis=0).reshape(1, -1)
        aabb_max = np.max(vertices, axis=0).reshape(1, -1)
        center = 0.5 * (aabb_max + aabb_min)
        scale = 2.0 * np.max(aabb_max - center)
        # move it up so it stands on the ground plane
        vertices = (vertices - center) / scale

        per_face_vertices = vertices[faces].reshape(-1, 3)
        vbo_vert = self.ctx.buffer(per_face_vertices.astype('f4'))

        geo_term = np.clip(
            np.einsum("ik,jk->ij", mesh.vertex_normals, self.dirs), 0, 1)
        prt_no_vis = np.einsum("ij,jk->ik", geo_term,
                               self.sh_basis) * 4.0 * math.pi / self.sample_size

        per_face_gen_prt = prt_gen[faces].reshape(-1, 9)
        vbo_gen_sh = self.ctx.buffer(per_face_gen_prt.astype('f4'))

        per_face_gt_prt = prt_gt[faces].reshape(-1, 9)
        vbo_gt_sh = self.ctx.buffer(per_face_gt_prt.astype('f4'))

        per_face_no_vis_prt = prt_no_vis[faces].reshape(-1, 9)
        vbo_no_vis_sh = self.ctx.buffer(per_face_no_vis_prt.astype('f4'))

        per_face_vc = vertex_color[faces].reshape(-1, 3)
        vbo_vc = self.ctx.buffer(per_face_vc.astype('f4'))
        vao_gen = self.ctx.vertex_array(self.prog,
                                        [(vbo_vert, '3f', 'in_pos'),
                                         (vbo_vc, '3f', 'in_vert_color'),
                                         (vbo_gen_sh, '9f', 'in_sh')])
        vao_gt = self.ctx.vertex_array(self.prog,
                                       [(vbo_vert, '3f', 'in_pos'),
                                        (vbo_vc, '3f', 'in_vert_color'),
                                        (vbo_gt_sh, '9f', 'in_sh')])
        vao_no_vis = self.ctx.vertex_array(self.prog,
                                           [(vbo_vert, '3f', 'in_pos'),
                                            (vbo_vc, '3f', 'in_vert_color'),
                                            (vbo_no_vis_sh, '9f', 'in_sh')])
        render_data = {
            "model_name": model_name,
            "vao": [vao_gen, vao_gt, vao_no_vis]
        }

        self.model_table[idx] = render_data
        self.model_angle = 0.

    def render(self, time: float, frametime: float):
        env_sh = self.light_sh.T.reshape(-1,)
        self.hdri_fbo.use()
        self.hdri_fbo.clear()
        self.hdri_prog['env_sh'].write(env_sh.astype('f4'))
        self.hdri_prog['R'].write(
            Matrix33.from_matrix44(self.camera.matrix).astype('f4'))
        self.hdri_vao.render()

        proj = self.ortho_proj.astype(
            'f4') if self.use_ortho else self.camera.projection.matrix

        self.fbo.use()
        self.fbo.clear()

        self.ctx.enable_only(moderngl.CULL_FACE)
        self.ctx.front_face = 'ccw'

        self.cubemap_texture.use(location=0)
        cube_transform = np.copy(self.camera.matrix)
        cube_transform[3][0] = 0
        cube_transform[3][1] = 0
        cube_transform[3][2] = 0

        model = matrix44.create_from_eulers([0, 0,
                                             self.model_angle]).astype('f4')

        mvp = proj * cube_transform * self.T * model
        self.cubemap_prog['mvp'].write(mvp)
        self.cube.render(self.cubemap_prog)

        self.ctx.front_face = 'cw'
        mvp = proj * self.camera.matrix * self.T

        self.ctx.enable(moderngl.DEPTH_TEST)
        self.prog['vp'].write(mvp)
        self.prog['model'].write(model.astype('f4'))

        self.prog['env_sh'].write(env_sh.astype('f4'))

        if self.current_model in self.model_table:
            self.model_table[self.current_model]['vao'][
                self.current_shading_mode].render()

        self.sphere_prog['textures'].value = 0
        self.hdri_fbo.color_attachments[0].use(location=0)

        self.sphere_fbo.use()
        self.sphere_fbo.clear()
        sphere_transform = np.copy(self.camera.matrix)
        sphere_transform[3][2] = -2
        sphere_transform[:3, :3] = np.eye(3, dtype='f4')
        mvp = proj * sphere_transform * self.T
        self.sphere_prog['mvp'].write(mvp)
        self.sphere_vao.render()

        # Render UI to screen
        self.wnd.use()
        self.render_ui()

    def render_ui(self):
        """Render the UI"""
        imgui.new_frame()
        if imgui.begin_main_menu_bar():
            if imgui.begin_menu("File", True):

                clicked_quit, selected_quit = imgui.menu_item(
                    "Quit", 'Cmd+Q', False, True)

                if clicked_quit:
                    exit(1)

                imgui.end_menu()
            imgui.end_main_menu_bar()

        imgui.begin("Control panel", True)
        show, _ = imgui.collapsing_header("Camera config")
        if show:
            _, self.use_ortho = imgui.checkbox(
                label="Use orthogonal projection",
                state=self.use_ortho,
            )

        show, _ = imgui.collapsing_header("Model selection")
        if show:
            _, self.current_model = imgui.listbox(
                label="Model",
                current=self.current_model,
                items=self.model_list,
                height_in_items=6,
            )
            if self.current_model not in self.model_table:
                self.load_model(self.current_model)
        show, _ = imgui.collapsing_header("Render config")
        if show:
            _, self.current_render_mode = imgui.listbox(
                label="Color",
                current=self.current_render_mode,
                items=self.render_mode,
                height_in_items=3,
            )
            _, self.current_shading_mode = imgui.listbox(
                label="Shading",
                current=self.current_shading_mode,
                items=self.shading_mode,
                height_in_items=3,
            )

        show, _ = imgui.collapsing_header("Light config")
        if show:
            light_changed, self.sh_id = imgui.input_int("Light SH",
                                                        self.sh_id,
                                                        step_fast=5)
            self.sh_id = max(0, min(self.sh_id, len(self.env_shs) - 1))

            cubemap_changed, self.current_cubemap = imgui.listbox(
                label="CubeMap",
                current=self.current_cubemap,
                items=self.cubemap_list,
                height_in_items=6,
            )
            if cubemap_changed:
                self.load_cubemap()

            lighting_mode_changed, self.current_lighting_mode = imgui.listbox(
                label="Lighting",
                current=self.current_lighting_mode,
                items=self.lighting_mode,
                height_in_items=3,
            )

            sh_slider_range = 2 * np.pi
            _, self.model_angle = imgui.slider_float(label="model",
                                                     value=self.model_angle,
                                                     min_value=-sh_slider_range,
                                                     max_value=sh_slider_range)
            if light_changed or cubemap_changed or lighting_mode_changed:
                sh = None
                if self.current_lighting_mode == 0:
                    sh = np.copy(self.env_shs[self.sh_id])
                elif self.current_lighting_mode == 1:
                    sh = np.copy(self.analytical_sh)
                else:
                    sh = np.copy(self.cubemap_sh)
                self.light_sh = sh

        imgui.end()

        imgui.begin("Render view", True)
        imgui.image(self.fbo.color_attachments[self.current_render_mode].glo,
                    self.width_gui, self.height_gui)
        self.image_view_hovered = imgui.is_item_hovered()
        imgui.end()

        imgui.begin("HDRI view", True)
        imgui.image(self.hdri_fbo.color_attachments[0].glo, 512, 256)
        imgui.end()

        imgui.begin("Sphere view", True)
        imgui.image(self.sphere_fbo.color_attachments[0].glo, 128, 128)
        imgui.end()

        imgui.render()
        self.imgui.render(imgui.get_draw_data())

    def resize(self, width: int, height: int):
        self.imgui.resize(width, height)

    def key_event(self, key, action, modifiers):
        self.imgui.key_event(key, action, modifiers)

    def mouse_position_event(self, x, y, dx, dy):
        self.imgui.mouse_position_event(x, y, dx, dy)

    def mouse_drag_event(self, x, y, dx, dy):
        if self.image_view_hovered:
            self.camera.rot_state(dx, dy)
        self.imgui.mouse_drag_event(x, y, dx, dy)

    def mouse_scroll_event(self, x_offset, y_offset):
        if self.image_view_hovered:
            delta_ratio = 0.9 if y_offset > 0 else 1.1
            self.ortho_ratio *= delta_ratio
            self.update_ortho()
            self.camera.zoom_state(y_offset)
        self.imgui.mouse_scroll_event(x_offset, y_offset)

    def mouse_press_event(self, x, y, button):
        self.imgui.mouse_press_event(x, y, button)

    def mouse_release_event(self, x: int, y: int, button: int):
        self.imgui.mouse_release_event(x, y, button)

    def unicode_char_entered(self, char):
        self.imgui.unicode_char_entered(char)


if __name__ == '__main__':
    mglw.run_window_config(WindowEvents)
