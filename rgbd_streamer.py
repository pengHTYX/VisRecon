import struct
import numpy as np
import cv2 as cv
import glob
import json


class RGBDFrame:

    def __init__(self, flag, color, depth, mask=None, cmask=None):
        self.flag = flag
        self.color = color
        self.depth = depth
        self.mask = mask
        self.cmask = cmask


class RGBDStreamer:

    def __init__(self, filename):
        super(RGBDStreamer, self).__init__()

        self.int_bytes = 4
        self.flt_bytes = 4
        self.szt_bytes = 8

        self.fp = open(filename, 'rb')
        self.parse_file(self.fp)

    def parse_file(self, fp):
        self.frame_num, = struct.unpack('i', self.fp.read(self.int_bytes))
        # print(self.frame_num)
        if (self.frame_num == 0):
            self.frame_num = 9999

        self.cwidth, = struct.unpack('i', self.fp.read(self.int_bytes))
        self.cheight, = struct.unpack('i', self.fp.read(self.int_bytes))
        self.cchannels, = struct.unpack('i', self.fp.read(self.int_bytes))

        self.dwidth, = struct.unpack('i', self.fp.read(self.int_bytes))
        self.dheight, = struct.unpack('i', self.fp.read(self.int_bytes))
        self.dchannels, = struct.unpack('i', self.fp.read(self.int_bytes))

        self.depth_size = self.dwidth * self.dheight * self.dchannels
        self.mask_size = self.dwidth * self.dheight

        # skip reading intrinsics and extrinsics
        intrinsic_size = (3 * 3 + 10) * self.flt_bytes
        color_intr = self.fp.read(intrinsic_size)
        depth_intr = self.fp.read(intrinsic_size)
        extrinsic_size = 4 * 4 * self.flt_bytes
        depth2color = self.fp.read(extrinsic_size)

        header_size = 7 * self.int_bytes + 2 * (
            9 + 10) * self.flt_bytes + 16 * self.flt_bytes
        self.frame_offsets = []
        self.frame_offsets.append(header_size)
        for idx in range(1, self.frame_num):
            prev_offset = self.frame_offsets[idx - 1]
            if (prev_offset >= self.fp.seek(0, 2)):
                break
            self.fp.seek(prev_offset)
            prev_color_time, = struct.unpack('Q', self.fp.read(self.szt_bytes))
            prev_color_size, = struct.unpack('Q', self.fp.read(self.szt_bytes))
            prev_frame_size = 3 * self.szt_bytes + prev_color_size + self.depth_size + self.mask_size
            curr_offset = prev_offset + prev_frame_size
            self.frame_offsets.append(curr_offset)
        self.frame_num = len(self.frame_offsets)
        # print(self.frame_offsets)

    def get_frame(self, frame_idx):
        if (frame_idx < len(self.frame_offsets)):
            self.fp.seek(self.frame_offsets[frame_idx])

            # t0 = time.time()
            color_time, = struct.unpack('Q', self.fp.read(self.szt_bytes))
            color_size, = struct.unpack('Q', self.fp.read(self.szt_bytes))
            jpg_string = self.fp.read(color_size)
            jpg_buffer = np.frombuffer(jpg_string, dtype='uint8')
            color_image = cv.imdecode(jpg_buffer, cv.IMREAD_UNCHANGED)
            color_image = cv.cvtColor(color_image, cv.COLOR_BGR2RGB)
            # print('load color time: ', time.time() - t0)

            # t1 = time.time()
            depth_time, = struct.unpack('Q', self.fp.read(self.szt_bytes))
            depth_string = self.fp.read(self.depth_size)
            depth_image = np.fromstring(depth_string, dtype='uint16').reshape(
                self.dheight, self.dwidth)
            # print('load depth time: ', time.time() - t1)

            # t2 = time.time()
            mask_string = self.fp.read(self.mask_size)
            mask_image = np.fromstring(mask_string, dtype='uint8').reshape(
                self.dheight, self.dwidth)
            # print('load mask time: ', time.time() - t2)

            # cv.imshow('color', color_image)
            # cv.imshow('depth', depth_buffer)
            # cv.imshow('mask', mask_image)
            # cv.waitKey(1)
            return RGBDFrame(True, color_image, depth_image, mask_image)
        else:
            return RGBDFrame(False, None, None, None)


class MultiStreamer:

    def __init__(self, folder_name):
        super(MultiStreamer, self).__init__()
        self.multi_streamer = []
        self.frame_num = 9999
        file_list = sorted(glob.glob(folder_name + '/*.rgbd'))
        print(file_list)
        self.camera_num = len(file_list)
        for i in range(len(file_list)):
            streamer = RGBDStreamer(file_list[i])
            self.multi_streamer.append(streamer)
            if (streamer.frame_num < self.frame_num):
                self.frame_num = streamer.frame_num

    def get_multiview_frames(self, frame_idx):
        rgbd_frames = []
        for i in range(len(self.multi_streamer)):
            rgbd_frames.append(self.multi_streamer[i].get_frame(frame_idx))
        return rgbd_frames


class RGBDCalibration:

    def __init__(self):
        self.d_height = 0
        self.d_width = 0
        self.c_height = 0
        self.c_width = 0
        self.Kd = np.float32(np.identity(3, dtype=float))
        self.Kc = np.float32(np.identity(3, dtype=float))
        self.Tcalib = np.float32(np.identity(
            4, dtype=float))    # depth_0 -> depth_i
        self.Tglobal = np.float32(np.identity(
            4, dtype=float))    # depth_i -> world
        self.Td2c = np.float32(np.identity(4,
                                           dtype=float))    # depth_i -> color_i
        self.center_depth = 0.0

    def print(self):
        print('d_height: {}, d_width: {}'.format(self.d_height, self.d_width))
        print('c_height: {}, c_width: {}'.format(self.c_height, self.c_width))
        print('Kd: {}'.format(self.Kd))
        print('Kc: {}'.format(self.Kc))
        print('Tcalib:\n{}'.format(self.Tcalib))
        print('Tglobal:\n{}'.format(self.Tglobal))
        print('center_depth: '.format(self.center_depth))


class Calibrations:

    def __init__(self, folder_name, camera_num):
        self.calibs = []
        # load calibs
        jr = json.loads(open(folder_name + '/calibration.json').read())
        assert len(
            jr
        ) >= camera_num, "Not enough cameras fround in calibration.json file!"
        depth_cams = []
        color_cams = []
        for i in range(camera_num):
            color_cam = jr[str(2 * i)]
            color_cams.append([
                np.float32(np.array(color_cam['K']).reshape(3, 3)),
                np.float32(np.array(color_cam['R']).reshape(3, 3)),
                np.float32(np.array(color_cam['T'])),
                np.float32(np.array(color_cam['imgSize']))
            ])
            depth_cam = jr[str(2 * i + 1)]
            depth_cams.append([
                np.float32(np.array(depth_cam['K']).reshape(3, 3)),
                np.float32(np.array(depth_cam['R']).reshape(3, 3)),
                np.float32(np.array(depth_cam['T'])),
                np.float32(np.array(depth_cam['imgSize']))
            ])
        # print(depth_cams)

        # init calibrations
        Td02w = np.identity(4, dtype=float)
        Td02w[:3, :3] = depth_cams[0][1]
        Td02w[:3, 3] = depth_cams[0][2]

        for i in range(camera_num):
            Tdi2w = np.identity(4, dtype=float)
            Tdi2w[:3, :3] = depth_cams[i][1]
            Tdi2w[:3, 3] = depth_cams[i][2]
            Td02di = np.linalg.inv(Tdi2w) @ Td02w    # d0->di

            Tci2w = np.identity(4, dtype=float)
            Tci2w[:3, :3] = color_cams[i][1]
            Tci2w[:3, 3] = color_cams[i][2]
            Td2c = np.linalg.inv(Tci2w) @ Tdi2w

            c = RGBDCalibration()
            c.Tcalib = np.float32(Td02di)
            c.Td2c = Td2c
            c.Kd = depth_cams[i][0]
            c.Kc = color_cams[i][0]
            c.d_height = depth_cams[i][3][1]
            c.d_width = depth_cams[i][3][0]
            c.c_height = color_cams[i][3][1]
            c.c_width = color_cams[i][3][0]
            #c.print()
            self.calibs.append(c)

    def update_center_depth(self, camera_id, center_depth):
        self.calibs[camera_id].center_depth = center_depth
        if (camera_id == 0):
            self.update_calibs()

    def update_calibs(self):
        self.calibs[0].Tglobal = np.float32(np.identity(4, dtype=float))
        self.calibs[0].Tglobal[2, 3] = self.calibs[0].center_depth

        for i in range(len(self.calibs) - 1):
            camid = i + 1
            self.calibs[camid].Tglobal = self.calibs[camid].Tcalib
            self.calibs[camid].Tglobal = np.matmul(self.calibs[camid].Tglobal,
                                                   self.calibs[0].Tglobal)

    def scale_calibs(self, scale):
        for i in range(len(self.calibs)):
            self.calibs[i].Tcalib[:3, 3] = self.calibs[i].Tcalib[:3, 3] * scale
            self.calibs[i].Tglobal[:3,
                                   3] = self.calibs[i].Tglobal[:3, 3] * scale
            self.calibs[i].Td2c[:3, 3] = self.calibs[i].Td2c[:3, 3] * scale

    def cut_d_calibs(self, tar_d_height, tar_d_width, cut_size):
        for i in range(len(self.calibs)):
            self.calibs[i].Kd[0, 2] -= (self.calibs[i].d_width - cut_size) / 2
            self.calibs[i].Kd[1, 2] -= (self.calibs[i].d_height - cut_size) / 2

            self.calibs[i].Kd = self.calibs[i].Kd * tar_d_height / cut_size
            self.calibs[i].Kd[2, 2] = 1.0

    def cut_c_calibs(self, tar_c_height, tar_c_width, cut_size):
        for i in range(len(self.calibs)):
            self.calibs[i].Kc[0, 2] -= (self.calibs[i].c_width - cut_size) / 2
            self.calibs[i].Kc[1, 2] -= (self.calibs[i].c_height - cut_size) / 2

            self.calibs[i].Kc = self.calibs[i].Kc * tar_c_height / cut_size
            self.calibs[i].Kc[2, 2] = 1.0


if __name__ == '__main__':
    import argparse
    from tqdm import tqdm
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_folder',
                        type=str,
                        default='$HOME/dataset/realworld/peng_li',
                        help='Path to rgbd file.')
    args = parser.parse_args()

    ms = MultiStreamer(args.data_folder)
    mc = Calibrations(args.data_folder, ms.camera_num)

    for i in tqdm(range(ms.frame_num)):
        rgbd_frames = ms.get_multiview_frames(i)
        flags = []
        for j in range(len(rgbd_frames)):
            flags.append(rgbd_frames[j].flag)
        if (all(flags)):
            for i in range(len(rgbd_frames)):
                cv.imshow('color', rgbd_frames[i].color)
            cv.waitKey(1)
