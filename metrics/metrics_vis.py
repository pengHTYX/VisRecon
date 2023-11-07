import os
from view_renderer import import_mesh
import numpy as np
from tqdm import tqdm
from implicit.implicit_prt_gen import fibonacci_sphere
import pandas as pd
import polyscope as ps
from icecream import ic

datasets = ['thuman', 'twindom', 'peng_li']

configs = ['vis_fuse']
vis_sample_count = 64
vis_sample_dirs, _, _ = fibonacci_sphere(vis_sample_count)

if __name__ == '__main__':
    import random

    random.seed(0)
    np.random.seed(0)

    validation_dataset_folder = '$HOME/dataset/paper_validation_data'
    validation_results_folder = '$HOME/dataset/paper_validation_results'
    validation_save_folder = '$HOME/dataset/paper_validation_metrics'

    results = {}
    metrics_column = ['item', 'acc'] + [i for i in range(65)]
    for dataset in datasets:
        print("Validating dataset ", dataset)

        for cfg in configs:
            config_results_dataset_path = os.path.join(
                validation_results_folder, cfg, dataset)

            for item in tqdm(os.listdir(config_results_dataset_path)):
                config_results_item_path = os.path.join(
                    config_results_dataset_path, item)
                mesh = import_mesh(
                    os.path.join(config_results_item_path, f'{item}.obj'))
                vis_gen = np.load(
                    os.path.join(config_results_item_path, 'vis_gen.npy')) > 0.5
                viz_gt = np.load(
                    os.path.join(config_results_item_path, 'viz_gt.npy'))

                assert len(mesh.vertex_normals) == len(vis_gen)
                assert len(mesh.vertex_normals) == len(viz_gt)

                cos_term = np.clip(
                    np.einsum("ik,jk->ij", mesh.vertex_normals,
                              vis_sample_dirs), 0, 1)
                vis_match = vis_gen != viz_gt
                # only care error in outward dirs
                vis_match[cos_term < 0] = False

                vis_error = np.sum(vis_match, axis=1)

                vis_bin_count = np.zeros(vis_sample_count + 1, dtype=np.int64)
                unique, count = np.unique(vis_error, return_counts=True)
                vis_bin_count[unique] = count
                error_bin = vis_bin_count / np.sum(count)

                total_outward_sample = np.sum(cos_term > 0)
                acc = (total_outward_sample -
                       np.sum(vis_error)) / total_outward_sample

                metrics_frame = pd.DataFrame([[item, acc] + error_bin.tolist()],
                                             columns=metrics_column)

                if cfg not in results:
                    results[cfg] = metrics_frame
                else:
                    results[cfg] = pd.concat([results[cfg], metrics_frame])

                # ps.init()
                # mesh_plot = ps.register_surface_mesh(item, mesh.vertices, mesh.faces)
                # mesh_plot.add_scalar_quantity("vis_error", vis_error)
                # ps.show()

    for cfg, cfg_frames in results.items():
        cfg_frames.to_csv(os.path.join(validation_save_folder,
                                       f"{cfg}_vis.csv"))
