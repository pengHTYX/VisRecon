import numpy as np
import argparse
from thuman import THumanDataset
from tqdm import tqdm
import os
from config import Config

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d',
                        '--data_folder',
                        type=str,
                        default='$HOME/dataset/val_full/')
    args = parser.parse_args()

    cfg = Config()
    cfg.point_num = 800000
    cfg.data_gen = True
    cfg.hair = False
    dataset_folder = args.data_folder
    twindom_dataset = THumanDataset(dataset_folder, cfg=cfg)
    for data_twindom in tqdm(twindom_dataset):
        folder = os.path.join(dataset_folder, data_twindom['model_name'])
        querys = data_twindom['querys']
        occ = data_twindom['occ']
        color = data_twindom['color']
        visibility = data_twindom['visibility']
        normals = data_twindom['normals']

        assert len(occ) + len(normals) == len(querys)
        assert int(0.4 * len(occ)) + len(normals) == len(color)

        # import polyscope as ps
        # from implicit.implicit_prt_gen import fibonacci_sphere

        # dirs, _, _ = fibonacci_sphere(64)

        # origins_rep = np.repeat(querys, 64, axis=0)
        # dirs_rep = np.tile(dirs, (len(querys), 1))
        # sample_pts = origins_rep + 0.1 * dirs_rep
        # ps.init()
        # pc = ps.register_point_cloud("querys_rep",
        #                              sample_pts,
        #                              radius=0.001,
        #                              point_render_mode='quad')
        # pc.add_scalar_quantity("vis_igl", visibility.reshape(-1, ))
        # ps.show()

        # exit()

        np.save(os.path.join(folder, "querys.npy"), np.array(querys))
        np.save(os.path.join(folder, "occ.npy"), np.array(occ))
        np.save(os.path.join(folder, "color.npy"), np.array(color))
        np.save(os.path.join(folder, "visibility.npy"), np.array(visibility))
        np.save(os.path.join(folder, "normals.npy"), np.array(normals))
