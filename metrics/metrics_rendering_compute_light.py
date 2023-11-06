import os
import torch
from PIL import Image
from tqdm import tqdm
import numpy as np
import random
import lpips
import pandas as pd
from skimage.metrics import structural_similarity as loss_ssim
from icecream import ic

datasets = ['thuman', 'twindom']
configs = ['f4d', 'vis_fuse']

render_count = 5
view_angles = np.linspace(0, 360, render_count + 1)[:-1]
use_foreground_mask = False

if __name__ == '__main__':
    random.seed(0)
    torch.manual_seed(0)
    np.random.seed(0)

    loss_lpips = lpips.LPIPS(net='vgg', spatial=True)
    loss_mse = torch.nn.MSELoss(reduction='none')

    render_folder = '$HOME/dataset/paper_validation_metrics/render2'
    validation_dataset_folder = '$HOME/dataset/paper_validation_data'
    validation_save_folder = '$HOME/dataset/paper_validation_metrics'

    results = {}
    metrics_column = ['item', 'mse', 'psnr', 'lpips', 'ssim']
    for dataset in datasets:
        print(f"Evaluating dataset {dataset}")
        for item in tqdm(
                os.listdir(os.path.join(validation_dataset_folder, dataset))):

            for view_angle in view_angles:
                view_angle = int(view_angle)
                gt_img = Image.open(
                    os.path.join(render_folder, dataset, 'gt',
                                 f'{item}_{view_angle}_color.png'))
                gt_img = gt_img.convert('RGB')
                gt_img = np.array(gt_img)
                gt_img_torch = torch.from_numpy(gt_img)
                gt_img_torch = gt_img_torch.unsqueeze(0).permute(0, 3, 1, 2).to(
                    torch.uint8)

                gt_img_torch_f = 2 * (gt_img_torch.float() / 255) - 1.

                for cfg in configs:

                    def evaluate_metrics(tag):
                        val_img = Image.open(
                            os.path.join(render_folder, dataset, cfg,
                                         f'{item}_{view_angle}_{tag}.png'))

                        if use_foreground_mask:
                            mask = np.array(val_img)[:, :, -1] > 0
                            mask_torch = torch.from_numpy(
                                mask).float().unsqueeze(0).unsqueeze(0)
                            mask_sum = mask.sum()
                        else:
                            foreground = np.array(val_img)[:, :, -1]
                            x_idx, y_idx = np.where(foreground > 0)
                            mask = np.zeros_like(foreground)
                            mask[x_idx.min():x_idx.max(),
                                 y_idx.min():y_idx.max()] = 1.
                            mask_torch = torch.from_numpy(
                                mask).float().unsqueeze(0).unsqueeze(0)
                            mask_sum = mask.sum()

                        val_img = val_img.convert('RGB')
                        val_img_torch = torch.from_numpy(np.array(val_img))
                        val_img_torch = val_img_torch.unsqueeze(0).permute(
                            0, 3, 1, 2).to(torch.uint8)

                        val_img_torch_f = 2 * (val_img_torch.float() / 255) - 1.

                        metrics_mse = (loss_mse(val_img_torch_f, gt_img_torch_f)
                                       * mask_torch).sum() / mask_sum / 3.

                        metrics_psnr = 20.0 * torch.log10(
                            2.0 / torch.sqrt(metrics_mse))

                        metrics_lpips = (
                            loss_lpips(val_img_torch_f, gt_img_torch_f) *
                            mask_torch).sum() / mask_sum

                        val_img = np.array(val_img)
                        _, ssim_img = loss_ssim(val_img,
                                                gt_img,
                                                data_range=255,
                                                channel_axis=2,
                                                full=True)
                        metrics_ssim = (ssim_img *
                                        mask[..., None]).sum() / mask_sum / 3.

                        metrics_mse = metrics_mse.item()
                        metrics_psnr = metrics_psnr.item()
                        metrics_lpips = metrics_lpips.item()

                        metrics_frame = pd.DataFrame([[
                            item, metrics_mse, metrics_psnr, metrics_lpips,
                            metrics_ssim
                        ]],
                                                     columns=metrics_column)

                        return metrics_frame

                    tag = 'color'
                    if cfg not in results:
                        results[cfg] = evaluate_metrics(tag)
                    else:
                        results[cfg] = pd.concat(
                            [results[cfg], evaluate_metrics(tag)])

    for cfg, cfg_frames in results.items():
        cfg_frames.to_csv(
            os.path.join(validation_save_folder, f"{cfg}_render_light.csv"))
