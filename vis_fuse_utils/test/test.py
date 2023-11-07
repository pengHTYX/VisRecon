import numpy as np
import vis_fuse_utils
from icecream import ic
import time

from scipy.spatial.distance import cdist


def closest_neighbour_sp(ref_pts, query_pts):
    dist_mat = cdist(ref_pts, query_pts)
    ref_closest_index_sp = np.argmin(dist_mat, axis=1)
    ref_closest_dist_sp = np.min(dist_mat, axis=1)
    query_closest_index_sp = np.argmin(dist_mat, axis=0)
    query_closest_dist_sp = np.min(dist_mat, axis=0)

    return ref_closest_dist_sp, ref_closest_index_sp, query_closest_dist_sp, query_closest_index_sp


def test_main():

    np.random.seed(0)

    ref_nb = 10000
    query_nb = 10000
    dim = 3

    test_iter = 10

    print()
    start_time = time.time()
    for _ in range(test_iter):
        ref_pts = np.random.randn(ref_nb, dim)
        query_pts = np.random.randn(query_nb, dim)
        closest_neighbour_sp(ref_pts, query_pts)
    print(f"Scipy time: {(time.time() - start_time) / test_iter}")

    start_time = time.time()
    for _ in range(test_iter):
        ref_pts = np.random.randn(ref_nb, 3)
        query_pts = np.random.randn(query_nb, 3)
        vis_fuse_utils.compute_closest_neighbor(ref_pts, query_pts)
    print(f"CUDA time: {(time.time() - start_time) / test_iter}")

    ref_pts = np.random.randn(ref_nb, 3)
    query_pts = np.random.randn(query_nb, 3)

    ref_closest_dist_sp, ref_closest_index_sp, query_closest_dist_sp, query_closest_index_sp = closest_neighbour_sp(
        ref_pts, query_pts)

    ref_closest_dist_cuda, ref_closest_index_cuda, query_closest_dist_cuda, query_closest_index_cuda = vis_fuse_utils.compute_closest_neighbor(
        ref_pts, query_pts)

    closest_index_valid = (ref_closest_index_sp != ref_closest_index_cuda).sum(
    ) == 0 and (query_closest_index_sp != query_closest_index_cuda).sum() == 0

    closest_dist_valid = np.average(
        (ref_closest_dist_sp - ref_closest_dist_cuda)) < 1e-7 and np.average(
            (query_closest_dist_sp - query_closest_dist_cuda)) < 1e-7

    assert closest_index_valid
    assert closest_dist_valid
