import os
from view_renderer import import_mesh
import numpy as np
from implicit.implicit_prt_gen import fibonacci_sphere
import polyscope as ps
from icecream import ic

vis_sample_count = 64
vis_sample_dirs, _, _ = fibonacci_sphere(vis_sample_count)

if __name__ == '__main__':
    validation_results_folder = '$HOME/dataset/paper_validation_results'
    dataset = 'thuman'
    item = '0277'
    configs = ['vis_fuse']

    ps.init()

    for cfg in configs:
        config_results_item_path = os.path.join(validation_results_folder, cfg,
                                                dataset, item)

        mesh = import_mesh(os.path.join(config_results_item_path,
                                        f'{item}.obj'))
        vis_gen = np.load(os.path.join(config_results_item_path,
                                       'vis_gen.npy')) > 0.5
        viz_gt = np.load(os.path.join(config_results_item_path, 'viz_gt.npy'))

        assert len(mesh.vertex_normals) == len(vis_gen)
        assert len(mesh.vertex_normals) == len(viz_gt)

        cos_term = np.clip(
            np.einsum("ik,jk->ij", mesh.vertex_normals, vis_sample_dirs), 0, 1)
        vis_match = vis_gen != viz_gt
        # only care error in outward dirs
        vis_match[cos_term < 0] = False

        vis_error = np.sum(vis_match, axis=1)

        mesh_plot = ps.register_surface_mesh(f"{cfg}", mesh.vertices,
                                             mesh.faces)
        mesh_plot.add_scalar_quantity("vis_error_map", vis_error, enabled=True)

    ps.show()
