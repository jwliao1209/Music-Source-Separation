from torch import nn
from torch.optim import SGD, Adam, AdamW, Optimizer

from .lion import Lion


def get_optimizer(
    name: str,
    model: nn.Module,
    lr: float = 3e-5,
    weight_decay: float = 0,
) -> Optimizer:

    match name:
        case 'sgd':
            optimizer = SGD
        case 'adam':
            optimizer = Adam
        case 'adamw':
            optimizer = AdamW
        case 'lion':
            optimizer = Lion
        case _:
            raise ValueError(f'Optimizer {name} not found')

    return optimizer(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )
