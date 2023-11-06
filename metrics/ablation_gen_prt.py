import vis_fuse_utils
from view_renderer import import_mesh
from implicit.implicit_prt_gen import fibonacci_sphere, getSHCoeffs
import os
from tqdm import tqdm
import numpy as np
import argparse
from icecream import ic

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_folder', type=str, help='Path to data folder.')
    parser.add_argument('--vis_sample_size',
                        type=int,
                        default=64,
                        help='Visibility dir sample size.')
    parser.add_argument('--quiet', action='store_true', help="disable tqdm")
    args = parser.parse_args()

    data_folder = args.data_folder

    prt_file_name = 'prt_gen.npy'
    vis_sample_size = args.vis_sample_size
    order = 2
    dirs, phi, theta = fibonacci_sphere(vis_sample_size)
    SH = getSHCoeffs(2, phi, theta)

    def gen_prt(model_path):
        mesh = import_mesh(model_path)
        vertices = mesh.vertices
        aabb_min = np.min(vertices, axis=0).reshape(1, -1)
        aabb_max = np.max(vertices, axis=0).reshape(1, -1)
        delta = 1e-3 * np.min(aabb_max - aabb_min)

        vis = np.logical_not(
            vis_fuse_utils.sample_occlusion_embree(
                mesh.vertices, mesh.faces,
                mesh.vertices + delta * mesh.vertex_normals, dirs))
        geo_term = np.clip(np.einsum("ik,jk->ij", mesh.vertex_normals, dirs), 0,
                           1)
        prt = np.einsum("ij,ij,jk->ik", vis, geo_term,
                        SH) * 4.0 * np.pi / vis_sample_size
        return prt

    mesh_list = os.listdir(data_folder)
    for item in mesh_list if args.quiet else tqdm(mesh_list):
        model_name = f'{item}.obj'
        model_folder_path = os.path.join(data_folder, item)
        model_path = os.path.join(model_folder_path, model_name)
        prt_path = os.path.join(model_folder_path, prt_file_name)

        prt_gen = gen_prt(model_path)
        np.save(prt_path, prt_gen)
