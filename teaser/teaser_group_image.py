import os
import cv2
import numpy as np
from icecream import ic

views = [0, 15, 30, 45]

if __name__ == '__main__':

    render_data_folder = '$HOME/dataset/render_data'
    input_pc_folder = '$HOME/hrnet/out/vis_fuse/test/pc'
    input_save_folder = '$HOME/hrnet/out/vis_fuse/test/input'

    if not os.path.exists(input_save_folder):
        os.mkdir(input_save_folder)

    dim = 512

    for model_name in os.listdir(render_data_folder):
        model_folder = os.path.join(render_data_folder, model_name)

        color_concat = np.zeros((dim, dim * 4, 3), dtype=np.uint8)
        mask_concat = np.zeros((dim, dim * 4), dtype=np.uint8)

        for i in range(4):
            view = views[i]
            color_path = os.path.join(model_folder, f"color_view_{view}.jpg")
            depth_path = os.path.join(model_folder, f"depth_view_{view}.png")
            depth_img = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            target_depth = np.float32(depth_img) / 1000
            depth_mask = np.float32(np.ones_like(depth_img))
            depth_mask[target_depth > 20] = 0

            color_img = cv2.imread(color_path)
            depth_mask = (255 * depth_mask).astype(np.uint8)

            color_concat[:, i * dim:(i + 1) * dim] = color_img
            mask_concat[:, i * dim:(i + 1) * dim] = depth_mask

        mask_concat = mask_concat[..., None].repeat(3, -1)
        input_img = np.vstack([color_concat, mask_concat])
        input_img = cv2.resize(input_img, (512, 256))

        pc_path = os.path.join(input_pc_folder, f"{model_name}.png")
        pc_image = cv2.imread(pc_path, cv2.IMREAD_UNCHANGED)
        pc_image[pc_image[:, :, 3] == 0] = [255, 255, 255, 255]
        pc_image = cv2.resize(pc_image, (512, 512))
        pc_image = cv2.cvtColor(pc_image, cv2.COLOR_BGRA2BGR)

        input_img = np.vstack([input_img, pc_image])
        cv2.imwrite(os.path.join(input_save_folder, f'{model_name}.png'),
                    input_img)
