import os
import pandas as pd
from glob import glob
import argparse
import numpy as np
from icecream import ic

geo_metrics_column = [
    'normal_consistency', 'chamfer_distance_l1', 'chamfer_distance_l2',
    'f_score_d', 'f_score_2d'
]

render_metrics_column = ['mse', 'psnr', 'lpips', 'ssim']

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--metrics_folder',
                        type=str,
                        default='$HOME/dataset/paper_validation_metrics',
                        help='Path to output folder.')
    args = parser.parse_args()

    save_path = os.path.join(args.metrics_folder, "collect")
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    geo_metrics_list = glob(os.path.join(args.metrics_folder, f"*_geo.csv"))

    def collect_average_geo(df, tag):
        return pd.DataFrame(
            [[tag] + [np.average(df[col]) for col in geo_metrics_column]],
            columns=['tag'] + geo_metrics_column)

    for item in geo_metrics_list:
        geo_metrics = pd.read_csv(item)

        dataset_div = len(geo_metrics) // 2

        df = pd.concat([
            collect_average_geo(geo_metrics[:dataset_div], "thuman"),
            collect_average_geo(geo_metrics[dataset_div:], "thuman")
        ])

        df.to_csv(os.path.join(save_path, item.split('/')[-1]))

    render_metrics_list = glob(
        os.path.join(args.metrics_folder, f"*_render.csv"))

    def collect_average_render(df, tag):
        return pd.DataFrame(
            [[tag] + [np.average(df[col]) for col in render_metrics_column]],
            columns=['tag'] + render_metrics_column)

    for item in render_metrics_list:
        render_metrics = pd.read_csv(item)

        albedo_metrics = render_metrics[render_metrics['tag'] == "albedo"]
        color_metrics = render_metrics[render_metrics['tag'] == "color"]

        dataset_div = len(albedo_metrics) // 2

        df = pd.concat([
            collect_average_render(albedo_metrics[:dataset_div],
                                   "albedo_thuman"),
            collect_average_render(albedo_metrics[dataset_div:],
                                   "albedo_twindom"),
            collect_average_render(color_metrics[:dataset_div], "color_thuman"),
            collect_average_render(color_metrics[dataset_div:], "color_twindom")
        ])

        df.to_csv(os.path.join(save_path, item.split('/')[-1]))
