import os
from argparse import ArgumentParser, Namespace

import torch
import wandb
from torch import nn

from src.constants import PROJECT_NAME, CHECKPOINT_DIR, CONFIG_FILE
from src.dataset import MUSDBDataset
from src.model import OpenUnmix
from src.optimization import get_lr_scheduler
from src.trainer import Trainer
from src.transforms import AudioEncoder
from src.utils import set_random_seeds, get_time, read_json, save_json, bandwidth_to_max_bin


def parse_arguments() -> Namespace:
    parser = ArgumentParser(description='Train source seperation model')

    # dataset setting
    parser.add_argument(
        '--data',
        type=str,
        default='musdb18',
        help='root path of dataset'
    )
    parser.add_argument(
        '--target',
        type=str,
        default='vocals',
        help='target source (will be passed to the dataset)',
    )
    parser.add_argument(
        '--samples_per_track',
        type=int,
        default=64,
    )
    parser.add_argument(
        '--source_augmentations',
        type=str,
        default=['gain', 'channelswap'],
        nargs='+'
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=4,
        help='Number of workers for dataloader.',
    )
    
    # model setting
    parser.add_argument(
        '--seq_dur',
        type=float,
        default=6.0,
        help='Sequence duration in seconds' 'value of <=0.0 will use full/variable length',
    )
    parser.add_argument(
        '--unidirectional',
        action='store_true',
        default=False,
        help='Use unidirectional LSTM',
    )
    parser.add_argument(
        '--nfft',
        type=int,
        default=4096,
        help='STFT fft size and window size'
    )
    parser.add_argument(
        '--nhop',
        type=int,
        default=1024,
        help='STFT hop size'
    )
    parser.add_argument(
        '--hidden-size',
        type=int,
        default=512,
        help='hidden size parameter of bottleneck layers',
    )
    parser.add_argument(
        '--bandwidth',
        type=int,
        default=16000,
        help='maximum model bandwidth in herz',
    )
    parser.add_argument(
        '--num_channels',
        type=int,
        default=2,
        help='set number of channels for model (1, 2)',
    )

    # training setting
    parser.add_argument(
        '--epochs',
        type=int,
        default=150,
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=16,
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=1e-3,
    )
    parser.add_argument(
        '--weight_decay',
        type=float,
        default=1e-5,
    )
    parser.add_argument(
        '--lr_scheduler',
        type=str,
        default='one_cycle',
    )

    return parser.parse_args()


if __name__ == '__main__':
    set_random_seeds()
    args = parse_arguments()
    checkpoint_dir = os.path.join(CHECKPOINT_DIR, get_time())
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Prepare dataset
    train_set = MUSDBDataset(
        root=args.data,
        subsets='train',
        split='train',
        samples_per_track=args.samples_per_track,
        seq_duration=args.seq_dur,
        source_augmentations=args.source_augmentations,
        random_track_mix=True,
        is_wav=True,
        target=args.target,
    )
    train_loader = train_set.get_loader(
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    valid_set = MUSDBDataset(
        root=args.data,
        subsets='train',
        split='valid',
        samples_per_track=1,
        seq_duration=None,
        is_wav=True,
        target=args.target,
    )
    valid_loader = valid_set.get_loader(
        batch_size=1,
        shuffle=False,
        num_workers=1,
    )

    # Prepare training
    device = torch.device(f'cuda:0'if torch.cuda.is_available() else 'cpu')
    encoder = AudioEncoder(
        n_fft=args.nfft,
        n_hop=args.nhop,
        sample_rate=train_set.sample_rate,
        num_channels=args.num_channels,
    )

    train_data_stats = train_set.get_stats(encoder)
    max_bin = bandwidth_to_max_bin(train_set.sample_rate, args.nfft, args.bandwidth)
    nb_bins = args.nfft // 2 + 1

    save_json(
        vars(args) | {
            'nb_bins': nb_bins,
            'max_bin': int(max_bin),
            'sample_rate': MUSDBDataset.sample_rate,
            'train_data_mean': train_data_stats['mean'].tolist(),
            'train_data_std': train_data_stats['std'].tolist(),
        },
        os.path.join(checkpoint_dir, CONFIG_FILE)
    )

    model = OpenUnmix(
        input_mean=train_data_stats['mean'],
        input_scale=train_data_stats['std'],
        nb_bins=nb_bins,
        nb_channels=args.num_channels,
        hidden_size=args.hidden_size,
        max_bin=max_bin,
        unidirectional=args.unidirectional,
    )

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    lr_scheduler = get_lr_scheduler(
        name=args.lr_scheduler,
        optimizer=optimizer,
        max_lr=args.lr,
        steps_for_one_epoch=len(train_loader),
        epochs=args.epochs,
    )

    # Prepare logger
    wandb.init(
        project=PROJECT_NAME,
        name=os.path.basename(checkpoint_dir),
        config=vars(args),
    )
    wandb.watch(model, log='all')

    # Start training
    trainer = Trainer(
        encoder=encoder,
        model=model,
        device=device,
        train_loader=train_loader,
        valid_loader=valid_loader,
        criterion=criterion,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        accum_grad_step=1,
        clip_grad_norm=1.0,
        logger=wandb,
        checkpoint_dir=checkpoint_dir,
    )
    trainer.fit(epochs=args.epochs)