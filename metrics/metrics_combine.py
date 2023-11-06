import argparse
import pandas as pd
import os
import numpy as np
from icecream import ic

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--metrics_folder',
                        type=str,
                        default='$HOME/dataset/paper_validation_metrics',
                        help='Path to output folder.')
    args = parser.parse_args()

    collect_folder = os.path.join(args.metrics_folder, 'collect')
    combine_folder = os.path.join(args.metrics_folder, 'combine')

    metrics_list = sorted(os.listdir(collect_folder))
    geo_metrics_list = [
        item for item in metrics_list if item.endswith('geo.csv')
    ]
    render_metrics_list = [
        item for item in metrics_list if item.endswith('render.csv')
    ]

    geo_metrics_column = [
        'normal_consistency', 'chamfer_distance_l1', 'f_score_d'
    ]
    geo_metrics_frame = pd.DataFrame(columns=['tag'] + geo_metrics_column)
    for tag in geo_metrics_list:
        geo_metrics = pd.read_csv(os.path.join(collect_folder, tag))

        geo_metrics_frame = pd.concat([
            geo_metrics_frame,
            pd.DataFrame([[tag.replace('_geo.csv', '')] + [
                np.average(geo_metrics[metrics_tag])
                for metrics_tag in geo_metrics_column
            ]],
                         columns=['tag'] + geo_metrics_column)
        ])

    render_metrics_column = ['psnr', 'lpips', 'ssim']
    albedo_metrics_frame = pd.DataFrame(columns=['tag'] + render_metrics_column)
    color_metrics_frame = pd.DataFrame(columns=['tag'] + render_metrics_column)

    for tag in render_metrics_list:
        render_metrics = pd.read_csv(os.path.join(collect_folder, tag))

        albedo_metrics_frame = pd.concat([
            albedo_metrics_frame,
            pd.DataFrame([[tag.replace('_render.csv', '')] + [
                np.average(render_metrics[metrics_tag][:2])
                for metrics_tag in render_metrics_column
            ]],
                         columns=['tag'] + render_metrics_column)
        ])

        color_metrics_frame = pd.concat([
            color_metrics_frame,
            pd.DataFrame([[tag.replace('_render.csv', '')] + [
                np.average(render_metrics[metrics_tag][2:])
                for metrics_tag in render_metrics_column
            ]],
                         columns=['tag'] + render_metrics_column)
        ])

    geo_metrics_frame.to_csv(os.path.join(combine_folder, 'geo.csv'))
    albedo_metrics_frame.to_csv(os.path.join(combine_folder, 'albedo.csv'))
    color_metrics_frame.to_csv(os.path.join(combine_folder, 'color.csv'))
