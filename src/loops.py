import argparse

import torch
import numpy as np

from tqdm import tqdm
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from lightly.models.utils import update_momentum
from lightly.utils.scheduler import cosine_schedule

from src.utils import AverageAggregator


# TODO: doublecheck
def pretrain(
    model: torch.nn.Module, criterion: torch.nn.Module, 
    loader: torch.utils.data.DataLoader, optimizer: torch.optim.Optimizer, 
    epoch: int, scaler: torch.cuda.amp.grad_scaler.GradScaler, device: torch.device, 
    args: argparse.Namespace
) -> dict:
    model.train()
    avg_loss = AverageAggregator()
    momentum_val = cosine_schedule(epoch, args.n_epochs, 0.996, 1)

    tqdm_it = tqdm(loader, leave=True)
    tqdm_it.set_description(f'Epoch: [{epoch+1}/{args.n_epochs}]')

    for batch in tqdm_it:
        update_momentum(model.student_backbone, model.teacher_backbone, m=momentum_val)
        update_momentum(model.student_head, model.teacher_head, m=momentum_val)

        views = [view.to(device) for view in batch]
        global_views = views[:2]

        teacher_out = [model.forward_teacher(view) for view in global_views]
        student_out = [model.forward(view) for view in views]

        loss = criterion(teacher_out, student_out, epoch=epoch)
        avg_loss.update(loss.item(), n=batch[0].shape[0])
        tqdm_it.set_postfix(  
            loss=str(loss.item())  # str() for no rounding
        ) 

        loss.backward()
        # We only cancel gradients of student head.
        model.student_head.cancel_last_layer_gradients(current_epoch=epoch)
        optimizer.step()
        optimizer.zero_grad()

    return avg_loss.item()


@torch.no_grad()
def train_evaluate_knn(
    model: torch.nn.Module, train_loader: torch.utils.data.DataLoader, 
    val_loader: torch.utils.data.DataLoader, device: torch.device
) -> dict:
    model.eval()

    loaders = {
        'train': train_loader,
        'val': val_loader
    }
    data = {
        'X_train': [],
        'y_train': [],
        'X_val': [],
        'y_val': []
    }

    for subset, loader in loaders.items():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            data[f'X_{subset}'].append(model(imgs).cpu().numpy())
            data[f'y_{subset}'].append(labels.cpu().numpy())

    data = {k: np.concatenate(l) for k, l in data.items()}

    estimator = KNeighborsClassifier()
    estimator.fit(data['X_train'], data['y_train'])

    y_val_pred = estimator.predict(data['X_val'])
    acc = accuracy_score(data['y_val'], y_val_pred)
    print(f'kNN val accuracy: {acc:.4f}')

    return acc