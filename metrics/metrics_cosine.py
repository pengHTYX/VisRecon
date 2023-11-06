import vis_fuse_utils
import numpy as np
from view_renderer import import_mesh
from implicit.implicit_prt_gen import fibonacci_sphere
from tqdm import tqdm
import torch
import random
from random import gauss
from icecream import ic

random.seed(0)
torch.manual_seed(0)
np.random.seed(0)


def make_rand_vector(dims):
    vec = [gauss(0, 1) for i in range(dims)]
    mag = sum(x**2 for x in vec)**.5
    return [x / mag for x in vec]


if __name__ == '__main__':
    test_model_path = '$HOME/dataset/render_data/0277/0277.obj'
    mesh = import_mesh(test_model_path)
    vertices = mesh.vertices
    aabb_min = np.min(vertices, axis=0).reshape(1, -1)
    aabb_max = np.max(vertices, axis=0).reshape(1, -1)
    delta = 1e-3 * np.min(aabb_max - aabb_min)
    num_vertex = len(vertices)

    def random_4():
        random_4_dirs = np.array([make_rand_vector(3) for _ in range(4)])
        random_4_dirs_vis_gt = np.logical_not(
            vis_fuse_utils.sample_occlusion_embree(
                mesh.vertices, mesh.faces,
                mesh.vertices + delta * mesh.vertex_normals, random_4_dirs))

        random_4_dirs = torch.from_numpy(random_4_dirs).float().unsqueeze(
            0).repeat_interleave(num_vertex, dim=0)

        random_4_dirs_vis_gt = torch.from_numpy(random_4_dirs_vis_gt).bool()

        return random_4_dirs, random_4_dirs_vis_gt

    verify_iter = 1000
    avg_acc_list = []
    for n in tqdm(range(2, 17)):
        print(f'Verifying {n}...')
        dirs, _, _ = fibonacci_sphere(n * n)
        dirs = torch.from_numpy(dirs).float()
        vis = torch.from_numpy(
            np.logical_not(
                vis_fuse_utils.sample_occlusion_embree(
                    mesh.vertices, mesh.faces,
                    mesh.vertices + delta * mesh.vertex_normals,
                    dirs))).float()
        vis_rep = vis.unsqueeze(1).repeat_interleave(4, dim=1)

        acc_list = []
        for _ in tqdm(range(verify_iter)):
            random_4_dirs, random_4_vis_gt = random_4()
            geo_term = torch.einsum('ndc,sc->nds', random_4_dirs, dirs)
            geo_top_k = geo_term.topk(3, -1)
            vis_pred = torch.gather(vis_rep, dim=-1, index=geo_top_k.indices)
            vis_pred = vis_pred * (geo_top_k.values /
                                   geo_top_k.values.sum(dim=-1, keepdim=True))
            vis_pred = vis_pred.sum(dim=-1) > 0.5

            acc = torch.sum((vis_pred == random_4_vis_gt)) / num_vertex / 4
            acc_list.append(acc.item())

        avg_acc = np.average(acc_list)
        print(f"Average accuracy {avg_acc} for {n} samples")
        avg_acc_list.append(avg_acc)
    print(avg_acc_list)
