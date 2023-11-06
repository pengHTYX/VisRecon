from dataclasses import dataclass, field


@dataclass
class DataConfig:
    train_folder: str = "$HOME/dataset/train_full/"
    val_folder: str = "$HOME/dataset/val_full/"
    realworld_folder: str = "$HOME/dataset/peng_li/"
    overfit_data: list[str] = field(default_factory=lambda: ["0553"])
    pointcloud_n: int = 20000
    view_num: int = 4
    total_view: int = 60
    patch_dim: int = 32


@dataclass
class UNet3dConfig:
    num_levels: int = 3
    f_maps: int = 32
    in_channels: int = 32
    out_channels: int = 32
    num_groups: int = 4


@dataclass
class EncoderConfig:
    hidden_dim: int = 64
    grid_resolution: int = 64
    unet3d_kwargs: UNet3dConfig = UNet3dConfig()


@dataclass
class ModelConfig:
    input_mode: str = "rgb_mask"
    c_dim: int = 32
    grid_dim: int = 32

    encoder_kwargs: EncoderConfig = EncoderConfig()


@dataclass
class TrainingConfig:
    batch_size: int = 2
    print_every: int = 20
    visualize_every: int = 4000
    checkpoint_every: int = 4000


@dataclass
class TestConfig:
    nx: int = 512
    realworld: bool = False
    scale: float = 2.0
    zshift: float = 1.9
    group_size: int = 400000


@dataclass
class RandomConfig:
    seed: int = 2012209151    # 1110111111011111101111111111111
    np_seed: int = 2113665023    # 1111101111110111111011111111111
    torch_seed: int = 2139028991    # 1111111011111101111110111111111
    train_seed: int = 2145369983    # 1111111110111111011111101111111
    val_seed: int = 2146955231    # 1111111111101111110111111011111
    cuda_seed: int = 2147450621    # 1111111111111110111111011111101


@dataclass
class Config:
    overfit: bool = False
    gpus: str = "0"
    differential: bool = False
    diff_samples: int = 10000
    mode: str = "train"
    fea_dim: int = 256
    hidden_size: int = 4
    hair: bool = False
    vis_sample_size: int = 64
    point_num: int = 5000
    live_sample: bool = False
    vis_fuse: bool = True
    embedding: bool = False
    embedding_freq: int = 12
    mask_dilate: bool = False
    fea_3d_only: bool = False
    attention: bool = False
    data_gen: bool = False
    live_sample: bool = False
    normal_downsample: int = 2

    data: DataConfig = DataConfig()
    model: ModelConfig = ModelConfig()
    training: TrainingConfig = TrainingConfig()
    test: TestConfig = TestConfig()
    random: RandomConfig = RandomConfig()
