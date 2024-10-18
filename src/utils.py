import json
import random
from datetime import datetime
from typing import Dict, List, Union

import numpy as np
import torch
from easydict import EasyDict


def read_json(path: str) -> Dict[str, Union[str, int, float]]:
    with open(path, 'r') as f:
        return json.load(f)


def save_json(data, path: str) -> None:
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)


def load_config(path: str) -> EasyDict:
    return EasyDict(read_json(path))
    

def get_time() -> str:
    return datetime.today().strftime('%m-%d-%H-%M-%S')


def set_random_seeds(seed: int = 0) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def dict_to_device(data: Dict[str, List[float]], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device) if not isinstance(v, list) else v for k, v in data.items()}


def bandwidth_to_max_bin(rate: float, n_fft: int, bandwidth: float) -> np.ndarray:
    """Convert bandwidth to maximum bin count

    Assuming lapped transforms such as STFT

    Args:
        rate (int): Sample rate
        n_fft (int): FFT length
        bandwidth (float): Target bandwidth in Hz

    Returns:
        np.ndarray: maximum frequency bin
    """
    freqs = np.linspace(0, rate / 2, n_fft // 2 + 1, endpoint=True)
    return np.max(np.where(freqs <= bandwidth)[0]) + 1
