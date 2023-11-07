import os
import numpy as np
from tqdm import tqdm
import trimesh
import pandas as pd
import vis_fuse_utils

import polyscope as ps
from icecream import ic

datasets = ['thuman', 'twindom']
point_sample_size = 10000

if __name__ == '__main__':
    import random
    import argparse

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
    np.random.seed(0)

    data_folder = args.data_folder
    results_folder = args.result
    save_folder = args.out

    if results_folder.endswith('/'):
        results_folder = results_folder[:-1]
    tag = results_folder.split('/')[-1]
    configs = os.listdir(results_folder)

    if 'raw' in configs:
        configs.remove('raw')

    results = {}
    metrics_column = [
        'item', 'normal_consistency', 'chamfer_distance_l1',
        'chamfer_distance_l2', 'f_score_d', 'f_score_2d'
    ]
    for dataset in datasets:
        print(f"Validating {dataset}")
        dataset_path = os.path.join(data_folder, dataset)

        if not os.path.exists(dataset_path):
            continue

        for item in tqdm(os.listdir(dataset_path)):
            gt_model_path = os.path.join(dataset_path, item, f'{item}.obj')
            gt_mesh: trimesh.Trimesh = trimesh.load(gt_model_path,
                                                    process=False,
                                                    maintain_order=True)

            aabb = np.max(gt_mesh.vertices, axis=0) - np.min(gt_mesh.vertices,
                                                             axis=0)
            chamfer_scale = 1 / (0.1 * np.max(aabb))
            f_score_threshold = 0.005 * np.max(aabb)

            # Chamfer distance is insensitive to density distribution
            gt_mesh_samples, gt_mesh_sample_indices = trimesh.sample.sample_surface_even(
                gt_mesh, point_sample_size)
            gt_mesh_sample_normals = gt_mesh.face_normals[
                gt_mesh_sample_indices]

            for cfg in configs:
                val_model_path = os.path.join(results_folder, cfg, item,
                                              f'{item}.obj')
                if not os.path.isfile(val_model_path):
                    print(f"val data not exist {val_model_path}")
                    continue

                val_mesh: trimesh.Trimesh = trimesh.load(val_model_path,
                                                         process=False,
                                                         maintain_order=True)
                val_mesh_samples, val_mesh_sample_indices = trimesh.sample.sample_surface_even(
                    val_mesh, point_sample_size)
                val_mesh_sample_normals = val_mesh.face_normals[
                    val_mesh_sample_indices]

                gt_to_val_closest_dist, closest_val_indices, val_to_gt_closest_dist, closest_gt_indices = vis_fuse_utils.compute_closest_neighbor(
                    gt_mesh_samples, val_mesh_samples)

                closest_gt_normals = gt_mesh_sample_normals[closest_gt_indices]

                metrics_normal_consistency = np.average(
                    np.abs((val_mesh_sample_normals *
                            closest_gt_normals).sum(axis=1)))

                metrics_chamfer_l1 = chamfer_scale * (np.average(
                    np.linalg.norm(
                        gt_mesh_samples - val_mesh_samples[closest_val_indices],
                        ord=1,
                        axis=1)) + np.average(
                            np.linalg.norm(val_mesh_samples -
                                           gt_mesh_samples[closest_gt_indices],
                                           ord=1,
                                           axis=1)))

                metrics_chamfer_l2 = chamfer_scale * (
                    np.average(val_to_gt_closest_dist) +
                    np.average(gt_to_val_closest_dist))

                def f_score(d):
                    precision = np.sum(
                        val_to_gt_closest_dist < d) / point_sample_size
                    recall = np.sum(
                        gt_to_val_closest_dist < d) / point_sample_size
                    return 2 * precision * recall / (precision + recall)

                metrics_f_score_d = f_score(f_score_threshold)
                metrics_f_score_2d = f_score(2 * f_score_threshold)

                metrics = [
                    item, metrics_normal_consistency, metrics_chamfer_l1,
                    metrics_chamfer_l2, metrics_f_score_d, metrics_f_score_2d
                ]
                metrics_frame = pd.DataFrame([metrics], columns=metrics_column)

                if cfg not in results:
                    results[cfg] = metrics_frame
                else:
                    results[cfg] = pd.concat([results[cfg], metrics_frame])

    for cfg, cfg_frames in results.items():
        cfg_frames.to_csv(os.path.join(save_folder, f"{tag}_{cfg}_geo.csv"))
