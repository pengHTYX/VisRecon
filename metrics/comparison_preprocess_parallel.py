import argparse
from pqdm.processes import pqdm
import subprocess
import multiprocessing
import os


def preprocess_runner(data_folder, out, cfg, dataset, out_folder,
                      out_folder_smooth):
    subprocess.run([
        "python", "-m", "metrics.comparison_preprocess", "--data_folder",
        f"{data_folder}", "--out", f"{out}", "--cfg", f"{cfg}", "--dataset",
        f"{dataset}", "--out_folder", f"{out_folder}", "--out_folder_smooth",
        f"{out_folder_smooth}", "--quiet"
    ])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_folder',
                        type=str,
                        default='$HOME/dataset/comparison/raw',
                        help='Path to data folder.')
    parser.add_argument('--out',
                        type=str,
                        default='$HOME/dataset/comparison',
                        help='Path to output folder.')
    args = parser.parse_args()

    comparsion_cfgs = ["f4d", "pifu", "pifuhd"]
    datasets = ["thuman", "thuman"]

    for cfg in comparsion_cfgs:
        out_folder = os.path.join(args.out, cfg)

        if not os.path.exists(out_folder):
            os.makedirs(out_folder)

        if cfg == "f4d" or cfg == "pifuhd":
            out_folder_smooth = os.path.join(args.out, f"{cfg}_smooth")

            if not os.path.exists(out_folder_smooth):
                os.makedirs(out_folder_smooth)

    runner_cfg = [(args.data_folder, args.out, cfg, dataset,
                   os.path.join(args.out,
                                cfg), os.path.join(args.out, f"{cfg}_smooth"))
                  for cfg in comparsion_cfgs
                  for dataset in datasets]

    pqdm(runner_cfg,
         preprocess_runner,
         n_jobs=multiprocessing.cpu_count(),
         argument_type='args')
