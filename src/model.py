from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
import zeta
from torch import nn


class OpenUnmix(nn.Module):
    """OpenUnmix Core spectrogram based separation module.

    Args:
        nb_bins (int): Number of input time-frequency bins (Default: `4096`).
        nb_channels (int): Number of input audio channels (Default: `2`).
        hidden_size (int): Size for bottleneck layers (Default: `512`).
        nb_layers (int): Number of Bi-LSTM layers (Default: `3`).
        unidirectional (bool): Use causal model useful for realtime purpose.
            (Default `False`)
        input_mean (ndarray or None): global data mean of shape `(nb_bins, )`.
            Defaults to zeros(nb_bins)
        input_scale (ndarray or None): global data mean of shape `(nb_bins, )`.
            Defaults to ones(nb_bins)
        max_bin (int or None): Internal frequency bin threshold to
            reduce high frequency content. Defaults to `None` which results
            in `nb_bins`
    """

    def __init__(
        self,
        nb_bins: int = 4096,
        nb_channels: int = 2,
        hidden_size: int = 512,
        nb_layers: int = 3,
        unidirectional: bool = False,
        input_mean: Optional[np.ndarray] = None,
        input_scale: Optional[np.ndarray] = None,
        max_bin: Optional[int] = None,
    ):
        super(OpenUnmix, self).__init__()

        self.nb_output_bins = nb_bins
        self.nb_bins = max_bin if max_bin else self.nb_output_bins
        self.hidden_size = hidden_size
        self.fc1 = nn.Linear(self.nb_bins * nb_channels, hidden_size, bias=False)
        self.bn1 = nn.BatchNorm1d(hidden_size)

        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size if unidirectional else hidden_size // 2,
            num_layers=nb_layers,
            bidirectional=not unidirectional,
            batch_first=False,
            dropout=0.4 if nb_layers > 1 else 0,
        )

        self.fc2 = nn.Linear(
            in_features=hidden_size * 2,
            out_features=hidden_size,
            bias=False
        )
        self.bn2 = nn.BatchNorm1d(hidden_size)

        self.fc3 = nn.Linear(
            in_features=hidden_size,
            out_features=self.nb_output_bins * nb_channels,
            bias=False,
        )
        self.bn3 = nn.BatchNorm1d(self.nb_output_bins * nb_channels)

        self.input_mean = nn.Parameter(
            torch.from_numpy(-input_mean[: self.nb_bins]).float() if input_mean is not None
            else torch.zeros(self.nb_bins)
        )
        self.input_scale = nn.Parameter(
            torch.from_numpy(1.0 / input_scale[: self.nb_bins]).float() if input_scale is not None
            else torch.ones(self.nb_bins)
        )

        self.output_scale = nn.Parameter(torch.ones(self.nb_output_bins).float())
        self.output_mean = nn.Parameter(torch.ones(self.nb_output_bins).float())

    def freeze(self):
        # set all parameters as not requiring gradient, more RAM-efficient at test time
        for p in self.parameters():
            p.requires_grad = False
        self.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: input spectrogram of shape (nb_samples, nb_channels, nb_bins, nb_frames)

        Returns:
            Tensor: filtered spectrogram of shape (nb_samples, nb_channels, nb_bins, nb_frames)
        """
        # permute so that batch is last for lstm
        x = x.permute(3, 0, 1, 2)
        nb_frames, nb_samples, nb_channels, nb_bins = x.data.shape

        mix = x.detach().clone()

        # crop
        x = x[..., : self.nb_bins]
        # shift and scale input to mean=0 std=1 (across all bins)
        x = (x + self.input_mean) * self.input_scale

        # to (nb_frames * nb_samples, nb_channels * nb_bins) and encode to (nb_frames * nb_samples, hidden_size)
        x = self.fc1(x.reshape(-1, nb_channels * self.nb_bins))
        x = self.bn1(x)
        x = x.reshape(nb_frames, nb_samples, self.hidden_size)
        x = torch.tanh(x) # squash range ot [-1, 1]

        # apply 3-layers of stacked LSTM
        lstm_out, _ = self.lstm(x)
        x = torch.cat([x, lstm_out], -1) # lstm skip connection

        # first dense stage + batch norm
        x = self.fc2(x.reshape(-1, x.shape[-1]))
        x = self.bn2(x)
        x = F.relu(x)

        # second dense stage + layer norm
        x = self.fc3(x)
        x = self.bn3(x)

        # reshape back to original dim
        x = x.reshape(nb_frames, nb_samples, nb_channels, self.nb_output_bins)

        # apply output scaling
        x = self.output_scale * x + self.output_mean

        # since our output is non-negative, we can apply RELU
        x = F.relu(x) * mix
        # permute back to (nb_samples, nb_channels, nb_bins, nb_frames)
        return x.permute(1, 2, 3, 0)


class OpenUnmixCNN(nn.Module):
    def __init__(
        self,
        nb_bins: int = 4096,
        nb_channels: int = 2,
        hidden_size: int = 512,
        input_mean: Optional[np.ndarray] = None,
        input_scale: Optional[np.ndarray] = None,
        max_bin: Optional[int] = None,
    ):
        super(OpenUnmixCNN, self).__init__()

        self.nb_output_bins = nb_bins
        self.nb_bins = max_bin if max_bin else self.nb_output_bins
        self.hidden_size = hidden_size

        self.fc1 = nn.Linear(self.nb_bins * nb_channels, hidden_size, bias=False)
        self.bn1 = nn.BatchNorm1d(hidden_size)

        self.cnn_stack = nn.Sequential(
            nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
        )

        self.fc2 = nn.Linear(
            in_features=hidden_size * 2,
            out_features=hidden_size,
            bias=False
        )
        self.bn2 = nn.BatchNorm1d(hidden_size)

        self.fc3 = nn.Linear(
            in_features=hidden_size,
            out_features=self.nb_output_bins * nb_channels,
            bias=False,
        )
        self.bn3 = nn.BatchNorm1d(self.nb_output_bins * nb_channels)

        self.input_mean = nn.Parameter(
            torch.from_numpy(-input_mean[: self.nb_bins]).float() if input_mean is not None
            else torch.zeros(self.nb_bins)
        )
        self.input_scale = nn.Parameter(
            torch.from_numpy(1.0 / input_scale[: self.nb_bins]).float() if input_scale is not None
            else torch.ones(self.nb_bins)
        )

        self.output_scale = nn.Parameter(torch.ones(self.nb_output_bins).float())
        self.output_mean = nn.Parameter(torch.ones(self.nb_output_bins).float())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: input spectrogram of shape
                `(nb_samples, nb_channels, nb_bins, nb_frames)`
        Returns:
            Tensor: filtered spectrogram of shape
                `(nb_samples, nb_channels, nb_bins, nb_frames)`
        """
        # permute so that batch is last for lstm
        x = x.permute(3, 0, 1, 2)
        nb_frames, nb_samples, nb_channels, nb_bins = x.data.shape

        mix = x.detach().clone()

        # crop
        x = x[..., : self.nb_bins]
        # shift and scale input to mean=0 std=1 (across all bins)
        x = (x + self.input_mean) * self.input_scale

        # to (nb_frames * nb_samples, nb_channels * nb_bins) and encode to (nb_frames * nb_samples, hidden_size)
        x = self.fc1(x.reshape(-1, nb_channels * self.nb_bins))
        x = self.bn1(x)
        x = x.reshape(nb_frames, nb_samples, self.hidden_size)
        x = torch.tanh(x) # squash range ot [-1, 1]

        # Apply stacked CNN layers 
        cnn_out = self.cnn_stack(x.permute(1, 2, 0)).permute(2, 0, 1)
        x = torch.cat([x, cnn_out], -1) # cnn skip connection

        # first dense stage + batch norm
        x = self.fc2(x.reshape(-1, x.shape[-1]))
        x = self.bn2(x)
        x = F.relu(x)

        # second dense stage + layer norm
        x = self.fc3(x)
        x = self.bn3(x)

        # reshape back to original dim
        x = x.reshape(nb_frames, nb_samples, nb_channels, self.nb_output_bins)

        # apply output scaling
        x = self.output_scale * x + self.output_mean

        # since our output is non-negative, we can apply RELU
        x = F.relu(x) * mix
        # permute back to (nb_samples, nb_channels, nb_bins, nb_frames)
        return x.permute(1, 2, 3, 0)


class OpenUnmixAttention(nn.Module):
    def __init__(
        self,
        nb_bins: int = 4096,
        nb_channels: int = 2,
        hidden_size: int = 512,
        input_mean: Optional[np.ndarray] = None,
        input_scale: Optional[np.ndarray] = None,
        max_bin: Optional[int] = None,
    ):
        super(OpenUnmixAttention, self).__init__()

        self.nb_output_bins = nb_bins
        self.nb_bins = max_bin if max_bin else self.nb_output_bins
        self.hidden_size = hidden_size

        self.fc1 = nn.Linear(self.nb_bins * nb_channels, hidden_size, bias=False)
        self.bn1 = nn.BatchNorm1d(hidden_size)

        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=4,
            batch_first=False,
        )

        self.fc2 = nn.Linear(
            in_features=hidden_size * 2,
            out_features=hidden_size,
            bias=False,
        )
        self.bn2 = nn.BatchNorm1d(hidden_size)

        self.fc3 = nn.Linear(
            in_features=hidden_size,
            out_features=self.nb_output_bins * nb_channels,
            bias=False,
        )
        self.bn3 = nn.BatchNorm1d(self.nb_output_bins * nb_channels)

        self.input_mean = nn.Parameter(
            torch.from_numpy(-input_mean[: self.nb_bins]).float() if input_mean is not None
            else torch.zeros(self.nb_bins)
        )
        self.input_scale = nn.Parameter(
            torch.from_numpy(1.0 / input_scale[: self.nb_bins]).float() if input_scale is not None
            else torch.ones(self.nb_bins)
        )

        self.output_scale = nn.Parameter(torch.ones(self.nb_output_bins).float())
        self.output_mean = nn.Parameter(torch.ones(self.nb_output_bins).float())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: input spectrogram of shape (nb_samples, nb_channels, nb_bins, nb_frames)
        Returns:
            Tensor: filtered spectrogram of shape (nb_samples, nb_channels, nb_bins, nb_frames)
        """
        # permute so that batch is last for lstm
        x = x.permute(3, 0, 1, 2)
        nb_frames, nb_samples, nb_channels, nb_bins = x.data.shape

        mix = x.detach().clone()

        # crop
        x = x[..., : self.nb_bins]
        # shift and scale input to mean=0 std=1 (across all bins)
        x = (x + self.input_mean) * self.input_scale

        # to (nb_frames * nb_samples, nb_channels * nb_bins) and encode to (nb_frames * nb_samples, hidden_size)
        x = self.fc1(x.reshape(-1, nb_channels * self.nb_bins))
        x = self.bn1(x)
        x = x.reshape(nb_frames, nb_samples, self.hidden_size)
        x = torch.tanh(x) # squash range ot [-1, 1]

        # Apply self attention
        # import pdb; pdb.set_trace()
        attn_out, _ = self.attention(x, x, x)
        x = torch.cat([x, attn_out], -1) # cnn skip connection

        # first dense stage + batch norm
        x = self.fc2(x.reshape(-1, x.shape[-1]))
        x = self.bn2(x)
        x = F.relu(x)

        # second dense stage + layer norm
        x = self.fc3(x)
        x = self.bn3(x)

        # reshape back to original dim
        x = x.reshape(nb_frames, nb_samples, nb_channels, self.nb_output_bins)

        # apply output scaling
        x = self.output_scale * x + self.output_mean

        # since our output is non-negative, we can apply RELU
        x = F.relu(x) * mix
        # permute back to (nb_samples, nb_channels, nb_bins, nb_frames)
        return x.permute(1, 2, 3, 0)


class OpenUnmixSSM(nn.Module):
    def __init__(
        self,
        nb_bins: int = 4096,
        nb_channels: int = 2,
        hidden_size: int = 512,
        input_mean: Optional[np.ndarray] = None,
        input_scale: Optional[np.ndarray] = None,
        max_bin: Optional[int] = None,
    ):
        super(OpenUnmixSSM, self).__init__()

        self.nb_output_bins = nb_bins
        self.nb_bins = max_bin if max_bin else self.nb_output_bins
        self.hidden_size = hidden_size

        self.fc1 = nn.Linear(self.nb_bins * nb_channels, hidden_size, bias=False)
        self.bn1 = nn.BatchNorm1d(hidden_size)

        self.ssm = zeta.nn.SSM(in_features=hidden_size,dt_rank=8,dim_inner=64,d_state=256,)

        self.fc2 = nn.Linear(
            in_features=hidden_size * 2,
            out_features=hidden_size,
            bias=False
        )
        self.bn2 = nn.BatchNorm1d(hidden_size)

        self.fc3 = nn.Linear(
            in_features=hidden_size,
            out_features=self.nb_output_bins * nb_channels,
            bias=False,
        )
        self.bn3 = nn.BatchNorm1d(self.nb_output_bins * nb_channels)

        self.input_mean = nn.Parameter(
            torch.from_numpy(-input_mean[: self.nb_bins]).float() if input_mean is not None
            else torch.zeros(self.nb_bins)
        )
        self.input_scale = nn.Parameter(
            torch.from_numpy(1.0 / input_scale[: self.nb_bins]).float() if input_scale is not None
            else torch.ones(self.nb_bins)
        )

        self.output_scale = nn.Parameter(torch.ones(self.nb_output_bins).float())
        self.output_mean = nn.Parameter(torch.ones(self.nb_output_bins).float())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: input spectrogram of shape (nb_samples, nb_channels, nb_bins, nb_frames)
        Returns:
            Tensor: filtered spectrogram of shape (nb_samples, nb_channels, nb_bins, nb_frames)
        """
        # permute so that batch is last for lstm
        x = x.permute(3, 0, 1, 2)
        nb_frames, nb_samples, nb_channels, nb_bins = x.data.shape

        mix = x.detach().clone()

        # crop
        x = x[..., : self.nb_bins]
        # shift and scale input to mean=0 std=1 (across all bins)
        x = (x + self.input_mean) * self.input_scale

        # to (nb_frames * nb_samples, nb_channels * nb_bins) and encode to (nb_frames * nb_samples, hidden_size)
        x = self.fc1(x.reshape(-1, nb_channels * self.nb_bins))
        x = self.bn1(x)
        x = x.reshape(nb_frames, nb_samples, self.hidden_size)
        x = torch.tanh(x) # squash range ot [-1, 1]

        # Apply self attention
        ssm_out, _ = self.ssm(x)
        x = torch.cat([x, ssm_out], -1) # cnn skip connection

        # first dense stage + batch norm
        x = self.fc2(x.reshape(-1, x.shape[-1]))
        x = self.bn2(x)
        x = F.relu(x)

        # second dense stage + layer norm
        x = self.fc3(x)
        x = self.bn3(x)

        # reshape back to original dim
        x = x.reshape(nb_frames, nb_samples, nb_channels, self.nb_output_bins)

        # apply output scaling
        x = self.output_scale * x + self.output_mean

        # since our output is non-negative, we can apply RELU
        x = F.relu(x) * mix
        # permute back to (nb_samples, nb_channels, nb_bins, nb_frames)
        return x.permute(1, 2, 3, 0)


def get_model(
    name: str,
    data_mean: Optional[np.ndarray] = None,
    data_std: Optional[np.ndarray] = None,
    num_bins: int = 4096,
    num_channels: int = 2,
    hidden_size: int = 512,
    max_bin: Optional[int] = None,
    unidirectional: bool = False,
    *args,
    **kwargs,
) -> nn.Module:

    match name:
        case 'openunmix':
            model = OpenUnmix(
                input_mean=data_mean,
                input_scale=data_std,
                nb_bins=num_bins,
                nb_channels=num_channels,
                hidden_size=hidden_size,
                max_bin=max_bin,
                unidirectional=unidirectional,
            )
        case 'openunmix_cnn':
            model = OpenUnmixCNN(
                input_mean=data_mean,
                input_scale=data_std,
                nb_bins=num_bins,
                nb_channels=num_channels,
                hidden_size=hidden_size,
                max_bin=max_bin,
            )
        case 'openunmix_attention':
            model = OpenUnmixAttention(
                input_mean=data_mean,
                input_scale=data_std,
                nb_bins=num_bins,
                nb_channels=num_channels,
                hidden_size=hidden_size,
                max_bin=max_bin,
            )
        case 'openunmix_ssm':
            model = OpenUnmixSSM(
                input_mean=data_mean,
                input_scale=data_std,
                nb_bins=num_bins,
                nb_channels=num_channels,
                hidden_size=hidden_size,
                max_bin=max_bin,
            )
        case _:
            raise ValueError(f'Model {name} not found')

    return model
