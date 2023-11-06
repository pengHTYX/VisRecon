# Learning Visibility Field for Detailed 3D Human Reconstruction and Relighting

[Project Page](https://ytrock.com/vis-fuse-page/) | [Paper](https://arxiv.org/abs/2304.11900)

This repository is an implementation of paper:

> "Learning Visibility Field for Detailed 3D Human Reconstruction and Relighting" by Ruichen Zheng, Peng Li, [Haoqian Wang](https://www.sigs.tsinghua.edu.cn/whq_en/main.htm), [Tao Yu](http://ytrock.com)

**WARNING** This is a messy research repo with limited code quality> We have only tested it on Linux (Ubuntu 22.04).

However, we provide detailed environment setup, sample data (both synthesized and realworld captured), pretrained model weight, that should run out of the box.

## Setup minimal environment

1. Create conda environment

   ```
   conda create --name vis-fuse -y python=3.10
   conda activate vis-fuse
   ```

2. Install pytorch. Snippet below is an example, please following official [instructions](https://pytorch.org/get-started/locally/)
   ```
   conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
   ```
3. Install common python packages
   ```
   pip install -r requirements.txt
   ```
4. Install [Libigl](https://github.com/libigl/libigl-python-bindings)
   ```
   python -m pip install libigl
   ```
5. Install our custom CPP lib (requires `CUDA`)
   ```
   cd vis_fuse_utils
   python setup.py install
   ```
6. (Optional) Additional dependencies

   > Not needed to run the demo

   - [gifski](https://gif.ski): Generate gif from sequences of renderings
   - [cmgen](https://github.com/google/filament/tree/main/tools/cmgen): Generate Spherical Harmonics from HDRi image

## Run demo

1. Download data and pretrained model weight from release page, unzip. Move `sample_data_thuman` and `peng_li` to desired location (following instructions refer it as `/path/to/`). Move `out` to project root folder
2. Render views (RGB-D)
   ```
   python thuman_renderer.py --data_folder /path/to/sample_data_thuman
   ```
3. Reconstruct and save results to `out/vis_fuse/test/4/#`
   ```
   python train.py --config configs/vis_fuse.json --save --test --data_folder /path/to/sample_data_thuman
   ```
4. Visualize using interactive viewer
   ```
   python prt_render_gui.py  --window glfw
   ```
5. Reconstruct from realworld data and save results to `out/vis_fuse/test/4/realworld_#`
   ```
   python train.py --config configs/vis_fuse.json --test --realworld --save --data_folder /path/to/peng_li
   ```

## Training tips

1. Request [THuman2.0](https://github.com/ytrock/THuman2.0-Dataset), download and train-val-test split
2. Generate random poses for each model and save as `cams.mat` in each subfolder and render views (or use other format for your liking)
   ```
   python thuman_renderer.py --data_folder /path/to/training_dataset
   ```
   > We do not include code to generate random pose
3. Generate occlusion samples
   ```
   python thuman_gen.py --data_folder /path/to/training_dataset
   ```
4. Visualize and verify training data
   ```
   python thuman.py --data_folder ~/dataset/training_dataset
   ```
5. Refer to `config.py` to write config (i.e. `my_config.json`) and save it to `configs`
6. Train
   ```
   python train.py --config configs/my_config.json
   ```
