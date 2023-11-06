import os
import torch
from PIL import Image
from tqdm import tqdm
import numpy as np
import random
import lpips
import pandas as pd
from skimage.metrics import structural_similarity as loss_ssim
from glob import glob
import argparse
from icecream import ic

datasets = ['thuman', 'twindom']

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_folder',
                        type=str,
                        default='$HOME/dataset/test_full',
                        help='Path to data folder.')
    parser.add_argument('--result', type=str, help='Path to result folder.')
    parser.add_argument('--out',
                        type=str,
                        default='$HOME/dataset/paper_validation_metrics',
                        help='Path to output folder.')
    args = parser.parse_args()

    random.seed(0)
    torch.manual_seed(0)
    np.random.seed(0)

    loss_lpips = lpips.LPIPS(net='vgg', spatial=True).cuda()

    data_folder = args.data_folder
    results_folder = args.result
    save_folder = args.out

    if results_folder.endswith('/'):
        results_folder = results_folder[:-1]
    cfg_tag = results_folder.split('/')[-1]
    configs = os.listdir(results_folder)

    if "gt" in configs:
        configs.remove("gt")

    results = {}
    metrics_column = ['item', 'tag', 'mse', 'psnr', 'lpips', 'ssim']

    for dataset in datasets:
        print(f"Evaluating dataset {dataset}")
        for item in tqdm(os.listdir(os.path.join(data_folder, dataset))):
            gt_img_folder = os.path.join(results_folder, 'gt')

            view_angles = [
                file_name.split('/')[-1].split('_')[1] for file_name in glob(
                    os.path.join(gt_img_folder, f"{item}_*_albedo.png"))
            ]

            for view_angle in view_angles:
                view_angle = int(view_angle)
                gt_img_path = os.path.join(gt_img_folder,
                                           f'{item}_{view_angle}_color.png')

                gt_img = Image.open(gt_img_path)
                gt_img = np.array(gt_img)
                gt_mask = gt_img[..., -1] > 0
                gt_mask_torch = torch.from_numpy(gt_mask).float().unsqueeze(
                    0).unsqueeze(0).cuda()

                gt_img = gt_img[..., :3]
                gt_img_torch = torch.from_numpy(gt_img).cuda()
                gt_img_torch = gt_img_torch.unsqueeze(0).permute(0, 3, 1, 2).to(
                    torch.uint8)
                # https://arxiv.org/pdf/2104.14868.pdf
                gt_img_torch_normalized = gt_img_torch.float() / 255.

                for cfg in configs:

                    def evaluate_metrics(tag):
                        val_img_path = os.path.join(
                            results_folder, cfg,
                            f'{item}_{view_angle}_{tag}.png')
                        val_img = Image.open(val_img_path)
                        val_img = np.array(val_img)[..., :3]
                        val_img_torch = torch.from_numpy(
                            np.array(val_img)).cuda()
                        val_img_torch = val_img_torch.unsqueeze(0).permute(
                            0, 3, 1, 2).to(torch.uint8)
                        val_img_torch_normalized = val_img_torch.float() / 255.

                        metrics_mse = (
                            (gt_img_torch_normalized - val_img_torch_normalized)
                            **2 *
                            gt_mask_torch).sum() / gt_mask_torch.sum() / 3.

                        metrics_psnr = 10. * torch.log10(1 / metrics_mse)

                        metrics_lpips = (
                            loss_lpips(2 * val_img_torch_normalized - 1,
                                       2 * gt_img_torch_normalized - 1) *
                            gt_mask_torch).sum() / gt_mask_torch.sum()

                        val_img = np.array(val_img)
                        _, ssim_img = loss_ssim(val_img,
                                                gt_img,
                                                data_range=255,
                                                channel_axis=2,
                                                full=True)
                        metrics_ssim = (ssim_img * gt_mask[..., None]
                                       ).sum() / gt_mask.sum() / 3.

                        metrics_mse = metrics_mse.item()
                        metrics_psnr = metrics_psnr.item()
                        metrics_lpips = metrics_lpips.item()

                        metrics_frame = pd.DataFrame([[
                            item, tag, metrics_mse, metrics_psnr, metrics_lpips,
                            metrics_ssim
                        ]],
                                                     columns=metrics_column)

                        return metrics_frame

                    for tag in ['color', 'albedo']:
                        if cfg not in results:
                            results[cfg] = evaluate_metrics(tag)
                        else:
                            results[cfg] = pd.concat(
                                [results[cfg],
                                 evaluate_metrics(tag)])

    for cfg, cfg_frames in results.items():
        cfg_frames.to_csv(
            os.path.join(save_folder, f"{cfg_tag}_{cfg}_render.csv"))
