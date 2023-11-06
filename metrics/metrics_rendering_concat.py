import os
from PIL import Image
from tqdm import tqdm
import argparse
from glob import glob
from icecream import ic

datasets = ['thuman', 'twindom']
height = 512

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_folder',
                        type=str,
                        default='$HOME/dataset/test_full',
                        help='Path to data folder.')
    parser.add_argument('--render_folder',
                        type=str,
                        default='$HOME/dataset/paper_validation_metrics/render',
                        help='Path to output folder.')
    args = parser.parse_args()

    render_folder = args.render_folder
    data_folder = args.data_folder

    configs = os.listdir(render_folder)

    concat_folder = os.path.join(
        render_folder.replace(render_folder.split('/')[-1], ""), "concate")

    if not os.path.exists(concat_folder):
        os.makedirs(concat_folder)

    for dataset in datasets:
        item_list = os.listdir(os.path.join(data_folder, dataset))
        for item in tqdm(item_list):

            view_angles = [
                file_name.split('/')[-1].split('_')[1] for file_name in glob(
                    os.path.join(os.path.join(render_folder, 'gt'),
                                 f"{item}_*_albedo.png"))
            ]

            for view_angle in view_angles:
                view_angle = int(view_angle)
                img_cat_light_color = Image.new('RGBA',
                                                (len(configs) * height, height))
                img_cat_color = Image.new('RGBA',
                                          (len(configs) * height, height))
                img_cat_albedo = Image.new('RGBA',
                                           (len(configs) * height, height))
                for i in range(len(configs)):
                    cfg = configs[i]
                    light_color_image = Image.open(
                        os.path.join(render_folder, cfg,
                                     f"{item}_{view_angle}_light_color.png"))
                    color_image = Image.open(
                        os.path.join(render_folder, cfg,
                                     f"{item}_{view_angle}_color.png"))
                    albedo_image = Image.open(
                        os.path.join(render_folder, cfg,
                                     f"{item}_{view_angle}_albedo.png"))

                    img_cat_light_color.paste(light_color_image,
                                              (i * height, 0))
                    img_cat_color.paste(color_image, (i * height, 0))
                    img_cat_albedo.paste(albedo_image, (i * height, 0))

                light_color_save_path = os.path.join(concat_folder,
                                                     'light_color')
                if not os.path.exists(light_color_save_path):
                    os.mkdir(light_color_save_path)
                img_cat_light_color.save(
                    os.path.join(light_color_save_path,
                                 f"{item}_{view_angle}_light_color.png"))

                color_save_path = os.path.join(concat_folder, 'color')
                if not os.path.exists(color_save_path):
                    os.mkdir(color_save_path)
                img_cat_color.save(
                    os.path.join(color_save_path,
                                 f"{item}_{view_angle}_color.png"))

                albedo_save_path = os.path.join(concat_folder, 'albedo')
                if not os.path.exists(albedo_save_path):
                    os.mkdir(albedo_save_path)
                img_cat_albedo.save(
                    os.path.join(albedo_save_path,
                                 f"{item}_{view_angle}_albedo.png"))
