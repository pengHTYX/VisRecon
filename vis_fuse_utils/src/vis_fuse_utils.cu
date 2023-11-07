#include <cuda.h>
#include <igl/Hit.h>
#include <igl/embree/EmbreeIntersector.h>
#include <igl/parallel_for.h>
#include <igl/ray_mesh_intersect.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <stdio.h>

#define BLOCK_DIM 16

/**
 * Computes the squared Euclidean distance matrix between the query points and
 * the reference points.
 *
 * @param ref          refence points stored in the global memory
 * @param ref_width    number of reference points
 * @param ref_pitch    pitch of the reference points array in number of column
 * @param query        query points stored in the global memory
 * @param query_width  number of query points
 * @param query_pitch  pitch of the query points array in number of columns
 * @param height       dimension of points = height of texture `ref` and of the
 * array `query`
 * @param dist         array containing the query_width x ref_width computed
 * distances
 */
__global__ void ComputeDistances(float *ref, int ref_width, int ref_pitch,
                                 float *query, int query_width, int query_pitch,
                                 int height, float *dist) {
  // Declaration of the shared memory arrays As and Bs used to store the
  // sub-matrix of A and B
  __shared__ float shared_A[BLOCK_DIM][BLOCK_DIM];
  __shared__ float shared_B[BLOCK_DIM][BLOCK_DIM];

  // Sub-matrix of A (begin, step, end) and Sub-matrix of B (begin, step)
  __shared__ int begin_A;
  __shared__ int begin_B;
  __shared__ int step_A;
  __shared__ int step_B;
  __shared__ int end_A;

  // Thread index
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  // Initializarion of the SSD for the current thread
  float ssd = 0.f;

  // Loop parameters
  begin_A = BLOCK_DIM * blockIdx.y;
  begin_B = BLOCK_DIM * blockIdx.x;
  step_A = BLOCK_DIM * ref_pitch;
  step_B = BLOCK_DIM * query_pitch;
  end_A = begin_A + (height - 1) * ref_pitch;

  // Conditions
  int cond0 = (begin_A + tx < ref_width);  // used to write in shared memory
  int cond1 = (begin_B + tx <
               query_width);  // used to write in shared memory & to
                              // computations and to write in output array
  int cond2 =
      (begin_A + ty <
       ref_width);  // used to computations and to write in output matrix

  // Loop over all the sub-matrices of A and B required to compute the block
  // sub-matrix
  for (int a = begin_A, b = begin_B; a <= end_A; a += step_A, b += step_B) {
    // Load the matrices from device memory to shared memory; each thread loads
    // one element of each matrix
    if (a / ref_pitch + ty < height) {
      shared_A[ty][tx] = (cond0) ? ref[a + ref_pitch * ty + tx] : 0;
      shared_B[ty][tx] = (cond1) ? query[b + query_pitch * ty + tx] : 0;
    } else {
      shared_A[ty][tx] = 0;
      shared_B[ty][tx] = 0;
    }

    // Synchronize to make sure the matrices are loaded
    __syncthreads();

    // Compute the difference between the two matrixes; each thread computes one
    // element of the block sub-matrix
    if (cond2 && cond1) {
      for (int k = 0; k < BLOCK_DIM; ++k) {
        float tmp = shared_A[k][ty] - shared_B[k][tx];
        ssd += tmp * tmp;
      }
    }

    // Synchronize to make sure that the preceeding computation is done before
    // loading two new sub-matrices of A and B in the next iteration
    __syncthreads();
  }

  // Write the block sub-matrix to device memory; each thread writes one element
  if (cond2 && cond1) {
    dist[(begin_A + ty) * query_pitch + begin_B + tx] = ssd;
  }
}

/**
 * For each query / reference point (i.e. each row / column) finds the smallest
 * distances of the distance matrix and their respective indexes and gathers
 * them at the top of the 2 arrays.
 *
 * @param dist         distance matrix
 * @param dist_pitch   pitch of the distance matrix given in number of columns
 * @param closest_indices        closest_indices matrix
 * @param closest_dists        closest_dists matrix
 * @param width        width of the distance matrix and of the index matrix
 * @param height       height of the distance matrix
 */
__global__ void ComputeClosestIndex(float *dist, int dist_pitch,
                                    int *closest_indices, float *closest_dists,
                                    int width, int height, bool is_col) {
  // Column position
  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

  // Do nothing if we are out of bounds
  if (index < (is_col ? width : height)) {
    // Pointer shift
    float *p_dist = dist + (is_col ? index : index * dist_pitch);
    int *closest_index = closest_indices + index;
    float *closest_dist = closest_dists + index;

    // initialize index
    *closest_index = -1;
    *closest_dist = INFINITY;

    // Go through all points
    for (int i = 0; i < (is_col ? height : width); ++i) {
      // Store current distance and associated index
      float curr_dist = sqrt(p_dist[(is_col ? i * dist_pitch : i)]);

      if (curr_dist < *closest_dist) {
        *closest_dist = curr_dist;
        *closest_index = i;
      }
    }
  }
}

// adapted from
// https://github.com/vincentfpgarcia/kNN-CUDA/blob/master/code/knncuda.h#L14
void ClosestNeighbourCuda(const float *ref, int ref_nb, const float *query,
                          int query_nb, int dim, float *knn_dist_ref,
                          int *knn_index_ref, float *knn_dist_query,
                          int *knn_index_query) {
  // Constants
  const unsigned int size_of_float = sizeof(float);
  const unsigned int size_of_int = sizeof(int);

  // Return variables
  cudaError_t err0, err1, err2, err3, err4, err5, err6;

  // Check that we have at least one CUDA device
  int nb_devices;
  err0 = cudaGetDeviceCount(&nb_devices);
  if (err0 != cudaSuccess || nb_devices == 0) {
    throw std::runtime_error("ERROR: No CUDA device found\n");
  }

  // Select the first CUDA device as default
  err0 = cudaSetDevice(0);
  if (err0 != cudaSuccess) {
    throw std::runtime_error("ERROR: Cannot set the chosen CUDA device\n");
  }

  // Allocate global memory
  float *ref_dev = nullptr;
  float *query_dev = nullptr;
  float *dist_dev = nullptr;
  size_t ref_pitch_in_bytes;
  size_t query_pitch_in_bytes;
  size_t dist_pitch_in_bytes;

  err0 = cudaMallocPitch((void **)&ref_dev, &ref_pitch_in_bytes,
                         ref_nb * size_of_float, dim);
  err1 = cudaMallocPitch((void **)&query_dev, &query_pitch_in_bytes,
                         query_nb * size_of_float, dim);
  err2 = cudaMallocPitch((void **)&dist_dev, &dist_pitch_in_bytes,
                         query_nb * size_of_float, ref_nb);

  int *closest_index_ref_dev = nullptr;
  float *closest_dist_ref_dev = nullptr;
  int *closest_index_query_dev = nullptr;
  float *closest_dist_query_dev = nullptr;

  err3 = cudaMalloc((void **)&closest_index_ref_dev, ref_nb * size_of_int);
  err4 = cudaMalloc((void **)&closest_dist_ref_dev, ref_nb * size_of_float);
  err5 = cudaMalloc((void **)&closest_index_query_dev, query_nb * size_of_int);
  err6 = cudaMalloc((void **)&closest_dist_query_dev, query_nb * size_of_float);

  auto free = [&]() {
    cudaFree(ref_dev);
    cudaFree(query_dev);
    cudaFree(dist_dev);
    cudaFree(closest_index_ref_dev);
    cudaFree(closest_dist_ref_dev);
    cudaFree(closest_index_query_dev);
    cudaFree(closest_dist_query_dev);
  };

  if (err0 != cudaSuccess || err1 != cudaSuccess || err2 != cudaSuccess ||
      err3 != cudaSuccess || err4 != cudaSuccess || err5 != cudaSuccess ||
      err6 != cudaSuccess) {
    free();
    throw std::runtime_error("ERROR: Memory allocation error\n");
  }

  // Deduce pitch values
  size_t ref_pitch = ref_pitch_in_bytes / size_of_float;
  size_t query_pitch = query_pitch_in_bytes / size_of_float;
  size_t dist_pitch = dist_pitch_in_bytes / size_of_float;

  // Check pitch values
  if (query_pitch != dist_pitch) {
    free();
    throw std::runtime_error("ERROR: Invalid pitch value\n");
  }

  // Copy reference and query data from the host to the device
  err0 = cudaMemcpy2D(ref_dev, ref_pitch_in_bytes, ref, ref_nb * size_of_float,
                      ref_nb * size_of_float, dim, cudaMemcpyHostToDevice);
  err1 = cudaMemcpy2D(query_dev, query_pitch_in_bytes, query,
                      query_nb * size_of_float, query_nb * size_of_float, dim,
                      cudaMemcpyHostToDevice);
  if (err0 != cudaSuccess || err1 != cudaSuccess) {
    free();
    throw std::runtime_error(
        "ERROR: Unable to copy data from host to device\n");
  }

  // Compute the squared Euclidean distances
  dim3 block0(BLOCK_DIM, BLOCK_DIM, 1);
  dim3 grid0(query_nb / BLOCK_DIM, ref_nb / BLOCK_DIM, 1);
  if (query_nb % BLOCK_DIM != 0) grid0.x += 1;
  if (ref_nb % BLOCK_DIM != 0) grid0.y += 1;
  ComputeDistances<<<grid0, block0>>>(ref_dev, ref_nb, ref_pitch, query_dev,
                                      query_nb, query_pitch, dim, dist_dev);
  if (cudaGetLastError() != cudaSuccess) {
    free();
    throw std::runtime_error(
        "ERROR: Unable to execute ComputeDistances kernel\n");
  }

  // Retrieve closest dist and index
  dim3 block_ref(256, 1, 1);
  dim3 grid_ref(ref_nb / 256, 1, 1);
  if (ref_nb % 256 != 0) grid_ref.x += 1;
  ComputeClosestIndex<<<grid_ref, block_ref>>>(
      dist_dev, dist_pitch, closest_index_ref_dev, closest_dist_ref_dev,
      query_nb, ref_nb, false);
  if (cudaGetLastError() != cudaSuccess) {
    free();
    throw std::runtime_error("ERROR: Unable to execute closest_index kernel\n");
  }

  dim3 block_query(256, 1, 1);
  dim3 grid_query(query_nb / 256, 1, 1);
  if (query_nb % 256 != 0) grid_query.x += 1;
  ComputeClosestIndex<<<grid_query, block_query>>>(
      dist_dev, dist_pitch, closest_index_query_dev, closest_dist_query_dev,
      query_nb, ref_nb, true);
  if (cudaGetLastError() != cudaSuccess) {
    free();
    throw std::runtime_error("ERROR: Unable to execute closest_index kernel\n");
  }

  // Copy the smallest distances / indexes from the device to the host
  err0 = cudaMemcpy(knn_dist_ref, closest_dist_ref_dev, ref_nb * size_of_float,
                    cudaMemcpyDeviceToHost);
  err1 = cudaMemcpy(knn_index_ref, closest_index_ref_dev, ref_nb * size_of_int,
                    cudaMemcpyDeviceToHost);
  err2 = cudaMemcpy(knn_dist_query, closest_dist_query_dev,
                    query_nb * size_of_float, cudaMemcpyDeviceToHost);
  err3 = cudaMemcpy(knn_index_query, closest_index_query_dev,
                    query_nb * size_of_int, cudaMemcpyDeviceToHost);

  if (err0 != cudaSuccess || err1 != cudaSuccess || err2 != cudaSuccess ||
      err3 != cudaSuccess) {
    free();
    throw std::runtime_error(
        "ERROR: Unable to copy data from device to host\n");
  }

  // Memory clean-up
  free();
}

namespace py = pybind11;

py::list ComputeClosestNeighbour(py::array_t<float> ref_pts,
                                 py::array_t<float> query_pts) {
  auto r_ref_pts = ref_pts.unchecked<2>();
  auto r_query_pts = query_pts.unchecked<2>();

  if (r_ref_pts.shape(1) != r_query_pts.shape(1)) {
    throw std::runtime_error(
        "Two point sets shoud have same feature dimension");
  }

  int ref_nb = r_ref_pts.shape(0);
  int query_nb = query_pts.shape(0);
  int dim = r_ref_pts.shape(1);

  float *closest_dist_ref = (float *)malloc(ref_nb * sizeof(float));
  int *closest_index_ref = (int *)malloc(ref_nb * sizeof(int));
  float *closest_dist_query = (float *)malloc(query_nb * sizeof(float));
  int *closest_index_query = (int *)malloc(query_nb * sizeof(int));

  ClosestNeighbourCuda(ref_pts.data(), ref_nb, query_pts.data(), query_nb, dim,
                       closest_dist_ref, closest_index_ref, closest_dist_query,
                       closest_index_query);

  py::list results_list;

  auto closest_dist_ref_py = py::array_t<float>(ref_nb, closest_dist_ref);
  auto closest_index_ref_py = py::array_t<int>(ref_nb, closest_index_ref);
  auto closest_dist_query_py = py::array_t<float>(query_nb, closest_dist_query);
  auto closest_index_query_py = py::array_t<int>(query_nb, closest_index_query);

  results_list.append(closest_dist_ref_py);
  results_list.append(closest_index_ref_py);
  results_list.append(closest_dist_query_py);
  results_list.append(closest_index_query_py);

  return results_list;
}

using RowMatrixXd =
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using RowMatrixXi =
    Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

RowMatrixXi SampleOcclusionEmbree(Eigen::Ref<const RowMatrixXd> V,
                                  Eigen::Ref<const RowMatrixXi> F,
                                  Eigen::Ref<const RowMatrixXd> origins,
                                  Eigen::Ref<const RowMatrixXd> dirs) {
  igl::embree::EmbreeIntersector intersector;
  intersector.init(V.cast<float>(), F.cast<int>(), true);

  RowMatrixXi occlusion(origins.rows(), dirs.rows());
  igl::parallel_for(
      origins.rows() * dirs.rows(),
      [&](int i) {
        int origin_idx = i / dirs.rows();
        int dir_idx = i % dirs.rows();
        igl::Hit hit;
        bool is_hit =
            intersector.intersectRay(origins.row(origin_idx).cast<float>(),
                                     dirs.row(dir_idx).cast<float>(), hit);
        occlusion.coeffRef(origin_idx, dir_idx) = static_cast<int>(is_hit);
      },
      1000);
  return occlusion;
}

RowMatrixXi SampleOcclusion(Eigen::Ref<const RowMatrixXd> V,
                            Eigen::Ref<const RowMatrixXi> F,
                            Eigen::Ref<const RowMatrixXd> origins,
                            Eigen::Ref<const RowMatrixXd> dirs) {
  RowMatrixXi occlusion(origins.rows(), dirs.rows());
  igl::parallel_for(
      origins.rows() * dirs.rows(),
      [&](int i) {
        int origin_idx = i / dirs.rows();
        int dir_idx = i % dirs.rows();
        igl::Hit hit;
        bool is_hit =
            igl::ray_mesh_intersect(origins.row(origin_idx).cast<float>(),
                                    dirs.row(dir_idx).cast<float>(), V, F, hit);
        occlusion.coeffRef(origin_idx, dir_idx) = static_cast<int>(is_hit);
      },
      1000);
  return occlusion;
}

PYBIND11_MODULE(vis_fuse_utils_bind, m) {
  m.def("compute_closest_neighbor", &ComputeClosestNeighbour,
        "Compute mutual closest point distance and index. Modified from "
        "https://github.com/vincentfpgarcia/kNN-CUDA");
  m.def("sample_occlusion", &SampleOcclusion,
        py::return_value_policy::reference_internal, "Sample occlusions");
  m.def("sample_occlusion_embree", &SampleOcclusionEmbree,
        py::return_value_policy::reference_internal,
        "Sample occlusions using embree");
}
