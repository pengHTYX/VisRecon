import torch

from tensorboardX import SummaryWriter
import numpy as np
import os
import argparse
import time
import datetime

from checkpoints import CheckpointIO
from thuman import THumanDataset
from training import Trainer
import random
from config import Config
import dataclasses
import json

import warnings
from icecream import ic

if __name__ == '__main__':
    warnings.filterwarnings('ignore')

    # Arguments
    parser = argparse.ArgumentParser(
        description='Train body reconstruction model.')
    parser.add_argument('--config', type=str, help='Path to config file.')
    parser.add_argument('--out',
                        type=str,
                        default='out',
                        help='Path to output folder.')
    parser.add_argument('--gpus', type=str, default='0', help='GPU id.')
    parser.add_argument('--test',
                        action='store_true',
                        help='Whether in test mode.')
    parser.add_argument('--realworld',
                        action='store_true',
                        help='Path to config file.')
    parser.add_argument('--save',
                        action='store_true',
                        help='Save vertex color, visibility, prt to file.')
    parser.add_argument('--data_folder', type=str, help='Path to datafolder.')
    parser.add_argument('--view_num',
                        type=int,
                        default=4,
                        help='Path to datafolder.')
    parser.add_argument("--num_epochs", type=int, default=1000)
    args = parser.parse_args()

    if args.config is not None:
        cfg = Config(**json.load(open(args.config)))
    else:
        cfg = Config()

    random.seed(cfg.random.seed)
    np.random.seed(cfg.random.np_seed)
    torch.manual_seed(cfg.random.torch_seed)
    g_train = torch.Generator()
    g_train.manual_seed(cfg.random.train_seed)
    g_val = torch.Generator()
    g_val.manual_seed(cfg.random.val_seed)
    torch.cuda.manual_seed_all(cfg.random.cuda_seed)

    cfg.gpus = args.gpus
    cfg.data.view_num = args.view_num
    # Set t0
    t0 = time.time()

    config_path = args.config
    config_file_name = config_path.split('/')[-1].split('.')[0]
    args.out = os.path.join(args.out, config_file_name)

    if args.test:
        cfg.test.realworld = args.realworld
        cfg.hair = False
        if args.data_folder is not None:
            if args.realworld:
                raise Exception('Realworld implementation not included!')
            else:
                cfg.data.val_folder = args.data_folder

    # Shorthands
    if cfg.overfit:
        args.out = os.path.join(args.out, 'overfit')
        cfg.gpus = '0'
        cfg.training.batch_size = 1
        cfg.training.visualize_every = 500
        cfg.training.checkpoint_every = 1000

    out_dir = args.out
    batch_size = cfg.training.batch_size

    # Output directory
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    with open(os.path.join(out_dir, 'config.json'), 'w') as f:
        json.dump(dataclasses.asdict(cfg), f)

    # Dataset
    realworld = cfg.test.realworld
    if realworld:
        raise Exception('Realworld implementation not included!')
    else:
        val_dataset = THumanDataset(cfg.data.val_folder, cfg=cfg, mode='test')

    # Model
    trainer = Trainer(cfg, out_dir=out_dir)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=1,
                                             num_workers=1,
                                             shuffle=False,
                                             generator=g_val)

    # load pretrained
    checkpoint_io = CheckpointIO(trainer.gpus,
                                 out_dir,
                                 model=trainer.model,
                                 opti=trainer.opti)

    try:
        load_dict = checkpoint_io.load('model.pt', map_location='cuda:0')
    except FileExistsError:
        load_dict = dict()

    first_epoch = load_dict.get('epoch_it', 0)
    it = load_dict.get('it', 0)

    # log
    logger = SummaryWriter(os.path.join(out_dir, 'logs'))

    # Shorthands
    print_every = cfg.training.print_every
    checkpoint_every = cfg.training.checkpoint_every
    visualize_every = cfg.training.visualize_every

    # Print model
    trainer.print_params()

    if args.test:
        with torch.no_grad():
            for data_val in val_loader:
                if realworld:
                    raise Exception('Realworld implementation not included!')

                else:
                    model_name = data_val['model_name'][0]

                print(f'Visualizing {model_name}')

                out_model_dir = os.path.join(args.out, 'test',
                                             str(args.view_num), model_name)
                trainer.eval_step(data_val,
                                  out_model_dir,
                                  model_name,
                                  nx=cfg.test.nx,
                                  save=args.save)
                # break
        exit()

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    train_dataset = THumanDataset(cfg.data.train_folder, cfg=cfg, mode='train')
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size * len(trainer.gpus),
        num_workers=max(2 * batch_size * len(cfg.gpus.split(',')), 4),
        shuffle=True,
        drop_last=True,
        worker_init_fn=seed_worker,
        generator=g_train)

    epoch_save_every = 100
    next_epoch_save = (1 + first_epoch // epoch_save_every) * epoch_save_every
    for epoch_it in range(first_epoch, args.num_epochs):
        for batch in train_loader:
            it += 1
            loss = trainer.train_step(batch, it)
            if loss is not None:
                logger.add_scalar('train/loss_grid', loss['total'].item(), it)
                # Print output
                if print_every > 0 and (it % print_every) == 0:
                    t = datetime.datetime.now()
                    n_print = len(loss)
                    loss_str = ''
                    for key, val in loss.items():
                        loss_str = loss_str + '%s=%.3f, ' % (key, val.item())
                    print('[Epoch %02d] it=%03d, time:%.2fs, %02d:%02d, ' %
                          (epoch_it, it, time.time() - t0, t.hour, t.minute) +
                          loss_str)

            # Visualize output
            if (visualize_every > 0 and (it % visualize_every) == 0):
                with torch.no_grad():
                    print('Visualizing')
                    for data_val in val_loader:
                        trainer.eval_step(data_val, out_dir,
                                          'debug_%d' % epoch_it)
                        break

            # Save checkpoint
            if ((checkpoint_every > 0 and
                 (it % checkpoint_every) == 0)) or it == 200:
                print('Saving checkpoint')
                checkpoint_io.save('model.pt', epoch_it=epoch_it, it=it)

            if epoch_it != 0 and epoch_it == next_epoch_save:
                next_epoch_save += epoch_save_every
                print(f'Epoch {epoch_it} reached. Saving checkpoint')
                checkpoint_io.save(f'model_{epoch_it}.pt',
                                   epoch_it=epoch_it,
                                   it=it)

                with torch.no_grad():
                    print(f'Visualizing {epoch_it}')
                    for data_val in val_loader:
                        trainer.eval_step(data_val, out_dir,
                                          'checkpoint_%d' % epoch_it)
                        break
