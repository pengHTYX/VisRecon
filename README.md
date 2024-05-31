# Learning Visibility Field for Detailed 3D Human Reconstruction and Relighting (CVPR2023)

[Project Page](https://ytrock.com/vis-fuse-page/) | [Paper](https://arxiv.org/abs/2304.11900)

This repository is an implementation of paper:

> "Learning Visibility Field for Detailed 3D Human Reconstruction and Relighting" by Ruichen Zheng, Peng Li, [Haoqian Wang](https://www.sigs.tsinghua.edu.cn/whq_en/main.htm), [Tao Yu](http://ytrock.com)

**WARNING** This is a messy research repo with limited code quality. We have only tested it on Linux (Ubuntu 22.04).

We provide synthesized data sample and pretrained model weight, that should run out of the box. Realworld results represented in the paper have code specific to our in-house capturing system, hence could not be included.

If you find our work useful, please consider cite our paper
```
@InProceedings{Zheng_2023_CVPR,
   author    = {Zheng, Ruichen and Li, Peng and Wang, Haoqian and Yu, Tao},
   title     = {Learning Visibility Field for Detailed 3D Human Reconstruction and Relighting},
   booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
   month     = {June},
   year      = {2023},
   pages     = {216-226}
}

```

## Clone this repo
```
git clone --recursive https://github.com/pengHTYX/VisRecon.git -b release --depth=1
```

If you forgot `--recursive`, call

```
git submodule update --init --recursive
```
to clone [pybind11](https://github.com/pybind/pybind11)

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

1. Download data and pretrained model weight from release page, unzip. Move `sample_data_thuman` to desired location (following instructions refer it as `/path/to/sample_data_thuman`). Move `out` to project root folder
2. Download [env_sh.npy](https://github.com/shunsukesaito/PIFu/blob/master/env_sh.npy) for PIFu repo and place it under `implicit`
3. Render views (RGB-D)
   ```
   python thuman_renderer.py --data_folder /path/to/sample_data_thuman
   ```
4. Reconstruct and save results to `out/vis_fuse/test/4/#`
   ```
   python train.py --config configs/vis_fuse.json --save --test --data_folder /path/to/sample_data_thuman
   ```
5. Visualize using interactive viewer
   ```
   python prt_render_gui.py  --window glfw
   ```

## Training tips

1. Request [THuman2.0](https://github.com/ytrock/THuman2.0-Dataset), download and train-val-test split
2. Generate random poses for each model and save as `cams.mat` in each subfolder and render views (or use other format for your liking)
   ```
   python thuman_renderer.py --data_folder /path/to/training_dataset
   ```
   > We do not include code to generate random poses
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
