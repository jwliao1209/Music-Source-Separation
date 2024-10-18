from torch import nn


def get_loss(name: str) -> nn.Module:
    match name:
        case 'mse':
            return nn.MSELoss()
        case 'mae':
            return nn.L1Loss()
        case 'smooth_mae':
            return nn.SmoothL1Loss()
        case _:
            raise ValueError(f'Scheduler {name} not found')
