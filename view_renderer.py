import os
from collections import namedtuple
import trimesh
import numpy as np
from PIL import Image
import polyscope as ps
from icecream import ic

# support vertex color
return_type = namedtuple('return_type', [
    'vertices', 'faces', 'uvs', 'face_uvs_idx', 'materials', 'vertex_colors',
    'vertex_normals'
])


# Modified from https://kaolin.readthedocs.io/en/latest/modules/kaolin.io.obj.html#module-kaolin.io.obj for vertex color support
def import_mesh(path, with_normals=True):
    r"""Load data from an obj file as a single mesh.
    With limited materials support to Kd, Ka, Ks, map_Kd, map_Ka and map_Ks.
    Followed format described in: http://paulbourke.net/dataformats/obj/
    Args:
        path (str): path to the obj file (with extension).
        with_normals (bool): if True, load vertex normals. Default: True.
    Returns:
        (obj.return_type):
            nametuple of:
            - **vertices** (np.array): of shape :math:`(\text{num_vertices}, 3)`.
            - **faces** (np.int64):
              of shape :math:`(\text{num_faces}, \text{face_size})`.
            - **vertex_colors** (np.array): of shape :math:`(\text{num_vertices}, 3)`.
            - **vertex_normals** (np.array): of shape :math:`(\text{num_vertices}, 3)`.
    """
    vertices = []
    faces = []
    uvs = []
    vertex_normals = []
    vertex_colors = []
    face_uvs_idx = []
    mtl_path = None
    materials = []

    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            data = line.split()
            if len(data) == 0:
                continue
            if data[0] == 'v':
                vertices.append(data[1:4])
                vertex_colors.append(data[4:])
            elif data[0] == 'vt':
                uvs.append(data[1:3])
            elif data[0] == 'f':
                data = [da.split('/') for da in data[1:]]
                faces.append([int(d[0]) for d in data])
                if len(data[1]) > 1 and data[1][1] != '':
                    face_uvs_idx.append([int(d[1]) for d in data])
                else:
                    face_uvs_idx.append([0] * len(data))
            elif data[0] == 'mtllib':
                mtl_path = os.path.join(os.path.dirname(path), data[1])
                with open(mtl_path) as f:
                    lines = f.readlines()
                    for l in lines:
                        if l.startswith("map_Kd"):
                            texture_file = l.split(' ')[-1]
                            texture_file = texture_file.strip('\n')
                            texture_file_path = os.path.join(
                                os.path.dirname(path), texture_file)
                            img = Image.open(texture_file_path)
                            if img.mode != 'RGB':
                                img = img.convert('RGB')
                            materials.append(img)

    uvs = np.float32([float(el) for sublist in uvs for el in sublist
                     ]).reshape(-1, 2)
    face_uvs_idx = np.int64(face_uvs_idx) - 1

    vertices = np.float32([float(el) for sublist in vertices for el in sublist
                          ]).reshape(-1, 3)
    vertex_colors = np.float32([
        float(el) for sublist in vertex_colors for el in sublist
    ]).reshape(-1, 3)

    if len(vertex_colors) == 0:
        vertex_colors = None

    faces = np.int64(faces) - 1

    if with_normals:
        vertex_normals = trimesh.Trimesh(
            vertices=vertices,
            faces=faces,
            process=False,
            maintain_order=True,
        ).vertex_normals
        assert not np.isnan(vertex_normals).any()
        assert len(vertex_normals) == len(vertices)
    else:
        vertex_normals = None

    return return_type(vertices, faces, uvs, face_uvs_idx, materials,
                       vertex_colors, vertex_normals)


if __name__ == '__main__':
    from implicit.implicit_render_prt import render_prt
    from implicit.implicit_prt_gen import fibonacci_sphere, getSHCoeffs
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--out", type=str)
    args = parser.parse_args()

    model_path = args.model
    model_name = model_path.split('/')[-2]
    prt_path = model_path.replace(model_path.split('/')[-1], "prt_gen.npy")

    mesh = import_mesh(model_path)
    prt = np.load(prt_path)

    assert len(mesh.vertices) == len(prt)

    # Adjust this for each model
    gif_init_rot = 90

    output_path = os.path.join(args.out, model_name, f"{gif_init_rot}")
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    render_prt(model_name,
               mesh.vertices,
               mesh.faces,
               prt,
               mesh.vertex_colors,
               output_path,
               unshaded_prt=None,
               gt_prt=None,
               render_image_sample=False,
               render_rot_gif=False,
               render_still_gif=True,
               render_lighting=True,
               res=1024,
               gif_init_rot=gif_init_rot)
