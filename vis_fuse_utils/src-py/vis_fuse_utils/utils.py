import vis_fuse_utils_bind
import numpy as np


# We bind this because pytorch3d was difficult to setup in our environment at that time
# TODO: replace with pytorch3d.ops.knn_points & pytorch3d.ops.knn_gather
def compute_closest_neighbor(ref_pts, query_pts):
    """Compute mutual closest neighbour between two point set using CUDA. Modified from https://github.com/vincentfpgarcia/kNN-CUDA

    Args:
        ref_pts: n x d
        query_pts: m x d
    Returns:
        ref_to_query_closest_dist: distance from ref_pts to their closest neighbour in query_pts (n)
        closest_query_index: ref_pts' closest query index (n)
        query_to_ref_closest_dist: distance from query_pts to their closest neighbour in ref_pts (m)
        closest_ref_index: query_pts' closest ref index (m)
    """
    ref_to_query_closest_dist, closest_query_index, query_to_ref_closest_dist, closest_ref_index = vis_fuse_utils_bind.compute_closest_neighbor(
        np.asfortranarray(ref_pts), np.asfortranarray(query_pts))

    return ref_to_query_closest_dist.astype(
        np.float64), closest_query_index.astype(
            np.int64), query_to_ref_closest_dist.astype(
                np.float64), closest_ref_index.astype(np.int64)


def sample_occlusion(V, F, origins, dirs):
    return vis_fuse_utils_bind.sample_occlusion(V, F, origins,
                                                dirs).astype(bool)


def sample_occlusion_embree(V, F, origins, dirs):
    return vis_fuse_utils_bind.sample_occlusion_embree(V, F, origins,
                                                       dirs).astype(bool)
