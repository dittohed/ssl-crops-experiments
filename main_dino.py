# TODO:
# [X] args
# [X] loaders
# [X] DINO
# [X] training loop
# [X] quick review so far
# [ ] implement superpatch as transform, not as dataset
# [ ] quick check and review so far
# [ ] add kNN evaluation
# [ ] ...
# run MultiCrop as usual (no resizing before)


import random
import argparse
import wandb

from pathlib import Path

import numpy as np
import torch

from lightly.loss import DINOLoss
from lightly.transforms.dino_transform import DINOTransform

from src.modules import DINO
from src.loops import train
from src.data import FlatImageFolder


def get_args_parser():
    parser = argparse.ArgumentParser('Pretrain using DINO')

    # Training params
    parser.add_argument('--use_amp', action='store_true',  # TODO
        help='Whether to use Automatic Mixed Precision for training.')
    parser.add_argument('--batch_size', default=128, type=int,
        help='''Number of distinct images for which a single
        backward pass will be calculated.''')
    parser.add_argument('--n_epochs', default=100, type=int, 
        help='Number of epochs of training.')
    parser.add_argument('--base_lr', default=0.025, type=float,  # TODO
        help='''Learning rate at the end of linear warmup (highest used during 
        training).''')
    parser.add_argument('--warmup_epochs', default=0, type=int,  # TODO
        help='Number of epochs for the linear learning-rate warm up.')
    parser.add_argument('--end_lr', type=float, default=1e-6,  # TODO
        help='''Target lr at the end of optimization. We use a cosine lr 
        schedule with linear warmup.''')
    parser.add_argument('--wd', type=float, default=1e-5,  # TODO
        help='Weight decay throughout the training.')
    
    # Other params
    parser.add_argument('--run_name', default='test', type=str,
        help='Unique run/experiment name.')
    parser.add_argument('--data_dir', default='./data/val2017', type=str,
        help='Path to pretraining data directory.')
    parser.add_argument('--chkpt_dir', default='./local/chkpts', type=str, 
        help='Path to directory for storing trained model\'s last checkpoint.')
    parser.add_argument('--seed', default=4294967295, type=int, 
        help='Random seed.')
    parser.add_argument('--num_workers', default=0, type=int, 
        help='Number of data loading workers.')
    parser.add_argument('--use_wandb', action='store_true',
        help='Whether to log training config and results to W&B.')


    return parser


def main(args):
    # Basic setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Prepare data
    transform = DINOTransform()
    dataset = FlatImageFolder(
        args.data_dir,
        transform=transform
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available()
    )

    # Prepare model
    backbone = torch.hub.load(
        'facebookresearch/dino:main', 'dino_vits16', pretrained=False
    )
    input_dim = backbone.embed_dim
    model = DINO(backbone, input_dim).to(device)

    # Prepare other stuff
    criterion = DINOLoss(
        output_dim=2048,
        warmup_teacher_temp_epochs=5
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.base_lr)
    scaler = torch.cuda.amp.GradScaler() if args.use_amp else None
    Path(args.chkpt_dir).mkdir(parents=True, exist_ok=True)

    for epoch in range(args.n_epochs):
        log_dict = train(
            model, criterion, loader, optimizer, epoch, 
            scaler, device, args
        )
        torch.save(
            model.state_dict(), 
            Path(args.chkpt_dir)/Path(args.run_name+'.pt')
        )
        if args.use_wandb:
            wandb.log(log_dict)


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()

    args.batch_size = 2  # TODO: rm after debugging

    if args.use_wandb:
        wandb.init(
            project='superpatch-first-exps',
            name=args.run_name,
            config=vars(args)
        )
        wandb.define_metric('train/loss', summary='min')
    
    main(args)