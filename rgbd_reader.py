from rgbd_streamer import RGBDFrame, MultiStreamer, Calibrations
import cv2
import numpy as np
import os
import math
import moderngl
from PIL import Image


# Adapted from https://github.com/shunsukesaito/PIFu/blob/master/lib/renderer/glm.py
def get_ortho_matrix(left, right, bottom, top, zNear, zFar):
    res = np.identity(4, dtype=np.float32)
    res[0][0] = 2 / (right - left)
    res[1][1] = 2 / (top - bottom)
    res[2][2] = -2 / (zFar - zNear)
    res[3][0] = -(right + left) / (right - left)
    res[3][1] = -(top + bottom) / (top - bottom)
    res[3][2] = -(zFar + zNear) / (zFar - zNear)
    return res.T


def get_persp_matrix(fx, fy, cx, cy, zNear, zFar):
    return np.array([[fx, 0, -cx, 0], [0, fy, -cy, 0],
                     [0, 0, zNear + zFar, zNear * zFar], [0, 0, -1, 0]])


def generate_mesh_from_depth_map(depth, mask, intr, need_face=False):
    height, width = np.shape(depth)
    N = height * width
    xx, yy = np.meshgrid(np.arange(width), np.arange(height))
    idx = xx + yy * width

    X = (xx - intr[0, 2]) * depth / intr[0, 0]
    Y = (yy - intr[1, 2]) * depth / intr[1, 1]
    points = np.stack((X, Y, depth), -1)
    points = np.reshape(points, (N, 3))
    mask = np.reshape(mask, (N))
    vertices = points[mask > 0]

    if need_face:
        idx_valid = np.reshape(idx, (N))[mask > 0]
        idx_map = np.zeros((N), dtype=np.compat.long)
        idx_map[idx_valid] = np.arange(idx_valid.shape[0])
        # idx_map = torch.zeros((N)).scatter_(dim=-1, index=torch.from_numpy(idx_valid), src=torch.arange(idx_valid.shape[0]))
        idx_map = idx_map.reshape((height, width))
        faces = []
        for r in range(height - 1):
            for c in range(width - 1):
                id00 = idx_map[r, c]
                id01 = idx_map[r, c + 1]
                id10 = idx_map[r + 1, c]
                id11 = idx_map[r + 1, c + 1]
                if id00 > 0 and id01 > 0 and id10 > 0 and id11 > 0:
                    faces.append(np.array([id00, id11, id01]))
                    faces.append(np.array([id00, id10, id11]))
                elif id00 > 0 and id01 > 0 and id11 > 0:
                    faces.append(np.array([id00, id11, id01]))
                elif id00 > 0 and id10 > 0 and id11 > 0:
                    faces.append(np.array([id00, id10, id11]))
                elif id00 > 0 and id01 > 0 and id10 > 0:
                    faces.append(np.array([id00, id10, id01]))
                elif id01 > 0 and id10 > 0 and id11 > 0:
                    faces.append(np.array([id01, id10, id11]))
        faces = np.stack(faces, axis=0)
        return vertices, faces
    else:
        return vertices


class RVM:

    def __init__(self, model_path="onnx_infer/rvm32.onnx", H=720, W=1280):
        from onnx_infer.rvm_infer import RVMInferEngine
        print("Initializing RobustVideoMatting Inference Engine...")
        infer_shape = (1, 3, H, W)
        self.engine = RVMInferEngine(model_path=model_path)
        self.engine.initialize(infer_shape)
        print(f"Initialized from {model_path}")

    def infer(self, rgb_imgs):
        color_mask = self.engine.infer(rgb_imgs)[0][0]
        out = np.zeros_like(color_mask)
        out[color_mask > 0.5] = 1
        return out


class RGBDReader:

    def __init__(self,
                 folder_name,
                 depth_scale=1.0,
                 tar_height=512,
                 tar_width=512,
                 color_cut_size=720,
                 depth_cut_size=512,
                 mode='c2d'):
        self.depth_scale = depth_scale
        self.tar_height = tar_height
        self.tar_width = tar_width
        self.depth_cut_size = depth_cut_size
        self.color_cut_size = color_cut_size

        self.dmin, self.dmax = 200, 3000
        self.depth_bg = 30 * 1000
        self.ms = MultiStreamer(folder_name)
        self.mc = Calibrations(folder_name, self.ms.camera_num)
        self.mc.scale_calibs(self.depth_scale)
        assert mode == 'c2d' or mode == 'd2c', "mode error"

        self.mc.cut_c_calibs(self.tar_height, self.tar_width,
                             self.color_cut_size)
        self.mc.cut_d_calibs(self.tar_height, self.tar_width,
                             self.depth_cut_size)
        self.calibs = self.mc.calibs
        self.rvm = RVM(H=tar_height, W=tar_height)    # H, W = 720, 1280

        self.ctx = moderngl.create_context(standalone=True, backend='egl')
        self.ctx.gc_mode = "auto"
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.prog = self.ctx.program(vertex_shader="""
            #version 330
            uniform mat4 mvp;
            layout (location = 0) in vec3 in_pos;
            void main() {
                gl_Position = mvp * vec4(in_pos, 1.0);
            }
            """,
                                     fragment_shader="""
            #version 330
            layout (location = 0) out vec4 frag_color;
            void main() {
                frag_color = vec4(1.0);
            }
            """)
        self.T_GL_CV = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
        self.zNear = 0.01
        self.zFar = 100.
        self.width = tar_width
        self.height = tar_height
        self.ortho = get_ortho_matrix(0, self.width, self.height, 0, self.zNear,
                                      self.zFar)
        color_attachments = [self.ctx.texture((self.width, self.height), 4)]
        self.fbo = self.ctx.framebuffer(
            color_attachments=color_attachments,
            depth_attachment=self.ctx.depth_renderbuffer(
                (self.width, self.height)))

    def map_color_to_depth(self, color, depth, mask, Kc, Kd, Td2c, view):
        colidx = np.arange(depth.shape[1])
        rowidx = np.arange(depth.shape[0])
        colidx_map, rowidx_map = np.meshgrid(colidx, rowidx)
        col_indices = colidx_map[mask > 0]
        row_indices = rowidx_map[mask > 0]
        homo_padding = np.ones((col_indices.shape[0], 1), dtype=np.float32)
        homo_indices = np.concatenate(
            (col_indices[..., None], row_indices[..., None], homo_padding),
            axis=1)

        Kd_inv = np.linalg.inv(Kd)
        normalized_points = Kd_inv[None, ...] @ homo_indices[..., None]

        z_values = (depth / 1000)[mask > 0]
        valid_points = normalized_points.squeeze() * z_values[..., None]

        # debug: save multi-view point clouds in world coordinate
        # T = self.calibs[view].Tcalib
        # Tinv = np.linalg.inv(T)
        # R = Tinv[:3,:3]
        # t = Tinv[:3,3]
        # points = R[None,...] @ valid_points[...,None] + t[None,...,None]
        # points = points.squeeze()
        # with open('./debug/point_cloud_{}.obj'.format(view), 'w') as fp:
        #     for i in range(points.shape[0]):
        #         fp.write('v {} {} {}\n'.format(points[i][0], points[i][1], points[i][2]))
        # fp.close()
        # exit()

        R = Td2c[:3, :3]
        t = Td2c[:3, 3]
        valid_points = R[None, ...] @ valid_points[..., None] + t[None, ...,
                                                                  None]
        valid_uvs = Kc[None, ...] @ valid_points / valid_points[:, 2][..., None]
        valid_uvs = np.int32(valid_uvs.squeeze()[..., :2] + 0.5)
        valid_uvs[:, 0] = np.clip(valid_uvs[:, 0], 0, color.shape[1] - 1)
        valid_uvs[:, 1] = np.clip(valid_uvs[:, 1], 0, color.shape[0] - 1)
        mapped_color = np.ones(
            (depth.shape[0], depth.shape[1], 3), dtype=np.uint8) * 255
        mapped_color[mask > 0] = color[valid_uvs[:, 1], valid_uvs[:, 0]]

        # print(mapped_color.shape, mapped_color.dtype)
        # cv2.imshow('mapped color', mapped_color)
        # cv2.waitKey(0)
        return mapped_color

    def cut_center_image(self, img, cut_size):
        height_offset = (img.shape[0] - cut_size) // 2
        width_offset = (img.shape[1] - cut_size) // 2
        img = img[height_offset:(cut_size + height_offset),
                  width_offset:(cut_size + width_offset)]
        return img

    def get_multiview_frames(self, frame_idx):
        rgbd_frames = self.ms.get_multiview_frames(frame_idx)
        output = []
        for i in range(self.ms.camera_num):
            # mask and scale depth
            depth = rgbd_frames[i].depth
            depth = self.cut_center_image(depth, self.depth_cut_size)
            mask = rgbd_frames[i].mask
            mask = self.cut_center_image(mask, self.depth_cut_size)
            mask[depth < self.dmin] = 0
            mask[depth > self.dmax] = 0
            depth = depth * self.depth_scale
            depth[mask == 0] = self.depth_bg
            depth = np.uint16(depth)

            color = rgbd_frames[i].color
            color = self.cut_center_image(color, self.color_cut_size)
            color = cv2.resize(color, (self.tar_width, self.tar_height))
            color_nn = np.float32(
                np.expand_dims(np.transpose(color / 255., (2, 0, 1)), 0))
            color_mask = self.rvm.infer(color_nn) > 0.5
            color_mask = np.uint8(color_mask * 255)

            depth_img = depth / 1000
            depth_mask = depth_img > 20
            depth_erode = cv2.erode(depth_img,
                                    np.ones((3, 3), np.uint8),
                                    iterations=5)
            depth_erode[depth_mask > 0] = depth_img[depth_mask > 0]
            depth_erode_mask = depth_erode < 20

            if np.sum(depth_erode_mask) > 3:
                calib = rgbd_reader.calibs[i]
                v, f = generate_mesh_from_depth_map(depth_img,
                                                    depth_erode_mask,
                                                    calib.Kd,
                                                    need_face=True)
                R, t = calib.Td2c[:3, :3], calib.Td2c[:3, 3]
                v = (v @ R.T + t) @ self.T_GL_CV.T

                fx = calib.Kc[0, 0]
                fy = calib.Kc[1, 1]
                cx = calib.Kc[0, 2]
                cy = calib.Kc[1, 2]

                proj = self.ortho @ get_persp_matrix(
                    fx, fy, cx, self.height - cy, self.zNear, self.zFar)
                per_face_vertices = v[f].reshape(-1, 3)
                vbo_vert = self.ctx.buffer(per_face_vertices.astype('f4'))
                vao = self.ctx.vertex_array(self.prog,
                                            [(vbo_vert, '3f', 'in_pos')])

                self.fbo.use()
                self.ctx.clear()

                self.prog['mvp'].write(
                    np.ascontiguousarray(proj.T).astype('f4'))
                vao.render()

                depth_reproj_data = self.fbo.read(components=3, attachment=0)
                depth_reproj_image = Image.frombytes('RGB', self.fbo.size,
                                                     depth_reproj_data)
                depth_reproj_mask = np.array(depth_reproj_image)[:, :,
                                                                 0].astype(
                                                                     np.uint8)

                color_mask = color_mask + depth_reproj_mask

            color[color_mask == 0] = 255

            rgbd_frame = RGBDFrame(True, color, depth, color_mask)
            output.append(rgbd_frame)
        return output


if __name__ == "__main__":
    # The `rgbd` is a specific format to store multi-synced Kinect capturing data in our lab
    #   We could not provide the code to generate `rgbd` files, as it is linked to the in-house capture devices
    # However, we provide the parsing file `rgbd_streamer.py` & `rgbd_reader.py` for a glimpse of how data are captured
    #   They also contain useful reference code for camera calibration and depth re-projection

    import argparse
    from tqdm import tqdm
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_folder',
                        type=str,
                        default='$HOME/dataset/realworld/peng_li',
                        help='Path to rgbd file.')
    parser.add_argument('--start_frame',
                        type=int,
                        default=250,
                        help='Start frame')
    parser.add_argument('--duration',
                        type=int,
                        default=400,
                        help='Duration (frames)')
    args = parser.parse_args()

    data_folder = os.path.expandvars(args.data_folder)
    hh, ww = 512, 512
    rgbd_reader = RGBDReader(data_folder, tar_width=ww, tar_height=hh)

    # write as .jpg and .png
    save_path = os.path.join(data_folder, 'RGB_DEPTH')
    os.makedirs(save_path, exist_ok=True)
    end_frame = min(args.start_frame + args.duration, rgbd_reader.ms.frame_num)
    for i in tqdm(range(args.start_frame, end_frame)):
        rgbd_frames = rgbd_reader.get_multiview_frames(i)
        for view in [0, 1, 2, 3]:
            cv2.imwrite(
                os.path.join(save_path,
                             "frame_%d_color_view_%d.jpg" % (i, view)),
                rgbd_frames[view].color[:, :, ::-1])
            cv2.imwrite(
                os.path.join(save_path,
                             "frame_%d_depth_view_%d.png" % (i, view)),
                rgbd_frames[view].depth)
            cv2.imwrite(
                os.path.join(save_path,
                             "frame_%d_cmask_view_%d.png" % (i, view)),
                rgbd_frames[view].mask)
