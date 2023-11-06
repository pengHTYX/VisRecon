import cv2
import numpy as np
import torch
import os
import sys

from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from networks import get_network
import torchvision.transforms as std_trnsf


class Config:

    def __init__(self):
        self.data_root = '$HOME/dataset/train_full'
        self.mode = 'imwrite'    # 'imwrite'
        self.crop_size = 256
        self.crop_h, self.crop_w = 0, 128
        self.device = 'cuda:0'
        self.views = 60
        self.depth_bg = 30000
        self.ckpt_dir = 'pspnet_resnet101_sgd_lr_0.002_epoch_100_test_iou_0.918.pth'
        self.networks = 'pspnet_resnet101'


# This script is used to remove hairs from segmentation mask
# To use, clone https://github.com/YBIGTA/pytorch-hair-segmentation
# Download pspnet_resnet101 then copy this file
if __name__ == '__main__':
    args = Config()
    view_num = args.views
    ckpt_dir = args.ckpt_dir
    network = args.networks.lower()
    data_root = args.data_root
    device = args.device
    crop_size = args.crop_size
    h_start, w_start = args.crop_h, args.crop_w
    depth_bg = args.depth_bg
    mode = args.mode

    data_root = os.path.expandvars(data_root)

    assert os.path.exists(ckpt_dir), 'ckpt_dir dosen' 't exist...'
    assert os.path.exists(data_root), 'data root dir dosen' 't exist...'

    # prepare network with trained parameters
    net = get_network(network).to(device)
    state = torch.load(ckpt_dir)
    net.load_state_dict(state['weight'])
    net.eval()

    test_image_transforms = std_trnsf.Compose([
        std_trnsf.ToTensor(),
        std_trnsf.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # prepare images
    model_paths = [
        os.path.join(data_root, k) for k in sorted(os.listdir(data_root))
    ]
    # model_paths = ['126111539901317-h']
    with torch.no_grad():
        for i, model_path in enumerate(model_paths):
            print('[{:3d}/{:3d}] processing model {:s}... '.format(
                i, len(model_paths),
                model_path.split('/')[-1]))
            for view in range(view_num):
                img_path = os.path.join(model_path, 'color_view_%d.jpg' % view)
                img = Image.open(img_path)
                img_show = img
                img = img.crop((w_start, h_start, w_start + crop_size,
                                h_start + crop_size))
                img = img.resize((512, 512))
                data = test_image_transforms(img)
                data = torch.unsqueeze(data, dim=0)
                data = data.to(device)

                # inference
                logit = net(data)

                # prepare mask
                pred = torch.sigmoid(logit.cpu())[0][0].data.numpy()
                mask = pred >= 0.5
                mask_float = np.zeros((512, 512))
                mask_float[mask] = 1.
                mask_float = cv2.resize(mask_float, (crop_size, crop_size),
                                        cv2.INTER_LINEAR)

                mask_hair = np.zeros((512, 512))
                mask_hair[h_start:h_start + crop_size,
                          w_start:w_start + crop_size] = mask_float
                mask_hair = cv2.dilate(mask_hair, np.ones((3, 3)), iterations=2)
                img = np.array(img_show) / 255

                # highlight
                mask_color = np.ones_like(mask_hair) * 0.6 * mask_hair
                img[:, :, -1] = mask_color + img[:, :, -1] * (1 - mask_hair)

                if mode == 'imshow':
                    ## imshow
                    cv2.imshow('img', img[:, :, ::-1])
                    cv2.imshow('mask', mask_hair)
                    cv2.waitKey(1)
                elif mode == 'imwrite':
                    ## mask depth and write
                    depth = np.float32(
                        cv2.imread(model_path + '/depth_view_%d.png' % view,
                                   cv2.IMREAD_UNCHANGED))
                    depth[mask_hair > 0] = depth_bg
                    save_path = os.path.join(
                        model_path, 'hair_masked_depth_view_%d.png' % view)
                    cv2.imwrite(save_path, np.uint16(depth))
                else:
                    raise NotImplementedError()
