import random
import argparse
import wandb

from pathlib import Path

import numpy as np
import torch

from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from lightly.loss import DINOLoss
from lightly.transforms.dino_transform import DINOTransform

from src.modules import DINO
from src.loops import pretrain, train_evaluate_knn
from src.data import FlatImageFolder, SuperpatchDataset, DINOViewTransform


def get_args_parser():
    parser = argparse.ArgumentParser('Pretrain using DINO')

    # Superpatch params
    parser.add_argument('--use_spatch', action='store_true',
        help='Whether to use superpatches instead of multi-crop for positive pairs.')
    parser.add_argument('--nn_json_path', default='./local/nns.json', type=str,
        help='Path to .json with nearest neighbors for each superpatch.')

    # Training params
    parser.add_argument('--use_amp', action='store_true',  # TODO
        help='Whether to use Automatic Mixed Precision for training.')
    parser.add_argument('--batch_size', default=128, type=int,
        help='''Number of distinct images for which a single
        backward pass will be calculated.''')
    parser.add_argument('--n_epochs', default=100, type=int, 
        help='Number of epochs of training.')
    parser.add_argument('--base_lr', default=0.0005, type=float,
        help='Final lr = base_lr * batch_size/256')
    parser.add_argument('--wd', type=float, default=0.4,
        help='Weight decay throughout the training.')
    
    # Other params
    parser.add_argument('--run_name', default='test', type=str,
        help='Unique run/experiment name.')
    parser.add_argument('--pretrain_data_dir', default='./data/val2017', type=str,
        help='Path to pretraining data directory.')
    parser.add_argument('--chkpt_dir', default='./local/chkpts', type=str, 
        help='Path to directory for storing trained model\'s last checkpoint.')
    parser.add_argument('--seed', default=4294967295, type=int, 
        help='Random seed.')
    parser.add_argument('--num_workers', default=0, type=int, 
        help='Number of data loading workers.')
    parser.add_argument('--use_wandb', action='store_true',
        help='Whether to log training config and results to W&B.')
    parser.add_argument('--evaluate', action='store_true',
        help='Whether to evaluate pretraining using kNN.')  # TODO: add actual if
    parser.add_argument('--eval_data_dir', default='./data/VOC2007/Cropped', type=str,
        help='Path to evaluation data directory.')

    return parser


def main(args):
    # Basic setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Prepare data for pretraining
    if args.use_spatch:
        # Use aug settings as for global crops in MultiCrop
        spatch_transform_1 = DINOViewTransform(solarization_prob=0)
        spatch_transform_2 = DINOViewTransform(gaussian_blur=0.1)
        pretrain_dataset = SuperpatchDataset(
            args.pretrain_data_dir,
            args.nn_json_path,
            spatch_transform_1,
            spatch_transform_2
        )
    else:
        pretrain_dataset = FlatImageFolder(
            args.pretrain_data_dir,
            transform=DINOTransform()
        )
    pretrain_loader = DataLoader(
        pretrain_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available()
    )

    # Prepare data for kNN evaluation
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])

    knn_train_dataset = ImageFolder(
        root=Path(args.eval_data_dir)/'train', 
        transform=transform
    )
    knn_train_loader = DataLoader(
        knn_train_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available()
    )

    knn_val_dataset = ImageFolder(
        root=Path(args.eval_data_dir)/'val', 
        transform=transform
    )
    knn_val_loader = DataLoader(
        knn_val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
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
        output_dim=1024,
        warmup_teacher_temp_epochs=5
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=args.base_lr * args.batch_size/256,
        weight_decay=args.wd
    )
    scaler = torch.cuda.amp.GradScaler() if args.use_amp else None
    Path(args.chkpt_dir).mkdir(parents=True, exist_ok=True)

    for epoch in range(args.n_epochs):
        loss = pretrain(
            model, criterion, pretrain_loader, optimizer, epoch, 
            scaler, device, args
        )
        torch.save(
            model.state_dict(), 
            Path(args.chkpt_dir)/Path(args.run_name+'.pt')
        )  # TODO: best chkpt
        acc = train_evaluate_knn(
            backbone, knn_train_loader, knn_val_loader, device 
        )

        log_dict = {'train/loss': loss, 'val/acc': acc}
        if args.use_wandb:
            wandb.log(log_dict)


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()

    if args.use_wandb:
        wandb.init(
            project='superpatch-first-exps',
            name=args.run_name,
            config=vars(args)
        )
        wandb.define_metric('train/loss', summary='min')
    
    main(args)