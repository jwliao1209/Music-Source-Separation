import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


def get_lr_scheduler(
    name: str,
    optimizer: Optimizer,
    max_lr: float,
    steps_for_one_epoch: int,
    epochs: int = 10,
    ) -> _LRScheduler:

    if name == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer=optimizer,
            step_size=10000,
            gamma=0.9,
        )

    elif name == "one_cycle":
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer=optimizer,
            max_lr=max_lr,
            steps_per_epoch=steps_for_one_epoch,
            epochs=epochs,
            pct_start=0.1,
            div_factor=1e3,
            final_div_factor=1e4,
            anneal_strategy='cos',
        )

    elif name == "cosine_annealing":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer,
            T_max=epochs * steps_for_one_epoch,
        )
    else:
        raise ValueError(f"Scheduler {name} not found")
