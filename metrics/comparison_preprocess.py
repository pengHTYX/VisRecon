import os
import argparse
import pymeshlab
from tqdm import tqdm
from icecream import ic

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_folder', type=str, help='Path to data folder.')
    parser.add_argument('--out', type=str, help='Path to output folder.')
    parser.add_argument('--cfg', type=str, help='data cfg.')
    parser.add_argument('--dataset', type=str, help='dataset name.')
    parser.add_argument('--out_folder', type=str, help='mesh out folder.')
    parser.add_argument('--out_folder_smooth',
                        type=str,
                        help='smooth mesh out folder.')
    parser.add_argument('--quiet', action='store_true', help="disable tqdm")
    args = parser.parse_args()

    data_folder = args.data_folder
    out = args.out
    cfg = args.cfg
    dataset = args.dataset
    out_folder = args.out_folder
    out_folder_smooth = args.out_folder_smooth

    mesh_folder = os.path.join(data_folder, f"{cfg}_{dataset}")
    mesh_list = os.listdir(mesh_folder)

    for mesh in mesh_list if args.quiet else tqdm(mesh_list):
        if cfg == "f4d" or cfg == "pifu":
            mesh_name = mesh.split("_")[0]
        elif cfg == "pifuhd":
            mesh_name = mesh.split("_")[1].split(".")[0]

        if dataset == "thuman":
            mesh_name = mesh_name.split("-")[0]

        new_mesh_folder = os.path.join(out_folder, mesh_name)
        if not os.path.exists(new_mesh_folder):
            os.makedirs(new_mesh_folder)

        mesh_file = os.path.join(mesh_folder, mesh)
        new_mesh_file = os.path.join(new_mesh_folder, f"{mesh_name}.obj")

        cmd = f"cp {mesh_file} {new_mesh_file}"
        os.system(cmd)

        if cfg == "f4d" or cfg == "pifuhd":
            new_mesh_folder_smooth = os.path.join(out_folder_smooth, mesh_name)
            if not os.path.exists(new_mesh_folder_smooth):
                os.makedirs(new_mesh_folder_smooth)

            new_mesh_file_smooth = os.path.join(new_mesh_folder_smooth,
                                                f"{mesh_name}.obj")

            ms = pymeshlab.MeshSet()
            ms.load_new_mesh(mesh_file)

            # remove isolated pieces
            if cfg == "pifuhd":
                ms.meshing_remove_connected_component_by_face_number(
                    mincomponentsize=10000)
                ms.save_current_mesh(new_mesh_file)

            ms.apply_coord_laplacian_smoothing()
            ms.save_current_mesh(new_mesh_file_smooth)
