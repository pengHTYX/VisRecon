cmake_minimum_required(VERSION 3.16)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 75)
endif()

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "-O3")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

project(vis_fuse_utils LANGUAGES C CXX CUDA)

# Libigl
list(PREPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
include(libigl)
igl_include(embree)

set(SRC_DIR src)

# python lib
set(VIS_FUSE_UTILS_PYTHON_LIB vis_fuse_utils_bind)
add_subdirectory(pybind11)
pybind11_add_module(${VIS_FUSE_UTILS_PYTHON_LIB} ${SRC_DIR}/vis_fuse_utils.cu)
target_link_libraries(${VIS_FUSE_UTILS_PYTHON_LIB} PUBLIC igl::embree)
