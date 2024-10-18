import os
from typing import Optional, Mapping, Union

import librosa
import numpy as np
import torch
from torch import  nn

from src.constants import CONFIG_FILE, CKPT_FILE
from src.filters import wiener
from src.model import get_model
from src.transforms import make_filterbanks, ComplexNorm
from src.utils import load_config


class Separator(nn.Module):
    """
    Separator class to encapsulate all the stereo filtering
    as a torch Module, to enable end-to-end learning.

    Args:
        targets (dict of str: nn.Module): dictionary of target models
            the spectrogram models to be used by the Separator.
        niter (int): Number of EM steps for refining initial estimates in a
            post-processing stage. Zeroed if only one target is estimated.
            defaults to `1`.
        residual (bool): adds an additional residual target, obtained by
            subtracting the other estimated targets from the mixture,
            before any potential EM post-processing.
            Defaults to `False`.
        wiener_win_len (int or None): The size of the excerpts
            (number of frames) on which to apply filtering
            independently. This means assuming time varying stereo models and
            localization of sources.
            None means not batching but using the whole signal. It comes at the
            price of a much larger memory usage.
        filterbank (str): filterbank implementation method.
            Supported are `['torch', 'asteroid']`. `torch` is about 30% faster
            compared to `asteroid` on large FFT sizes such as 4096. However,
            asteroids stft can be exported to onnx, which makes is practical
            for deployment.
    """

    def __init__(
        self,
        target_models: Mapping[str, nn.Module],
        niter: int = 0,
        softmask: bool = False,
        residual: bool = False,
        sample_rate: float = 44100.0,
        n_fft: int = 4096,
        n_hop: int = 1024,
        nb_channels: int = 2,
        wiener_win_len: Optional[int] = 300,
        filterbank: str = "torch",
    ):
        super(Separator, self).__init__()

        # saving parameters
        self.niter = niter
        self.residual = residual
        self.softmask = softmask
        self.wiener_win_len = wiener_win_len

        self.stft, self.istft = make_filterbanks(
            n_fft=n_fft,
            n_hop=n_hop,
            center=True,
        )
        self.complexnorm = ComplexNorm(mono=nb_channels == 1)

        # registering the targets models
        self.target_models = nn.ModuleDict(target_models)
        # adding till https://github.com/pytorch/pytorch/issues/38963
        self.nb_targets = len(self.target_models)
        # get the sample_rate as the sample_rate of the first model
        # (tacitly assume it's the same for all targets)
        self.register_buffer("sample_rate", torch.as_tensor(sample_rate))

    def freeze(self):
        # set all parameters as not requiring gradient, more RAM-efficient
        # at test time
        for p in self.parameters():
            p.requires_grad = False
        self.eval()

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """Performing the separation on audio input

        Args:
            audio (Tensor): [shape=(nb_samples, nb_channels, nb_timesteps)]
                mixture audio waveform

        Returns:
            Tensor: stacked tensor of separated waveforms
                shape `(nb_samples, nb_targets, nb_channels, nb_timesteps)`
        """

        nb_sources = self.nb_targets
        nb_samples = audio.shape[0]

        # getting the STFT of mix:
        # (nb_samples, nb_channels, nb_bins, nb_frames, 2)
        mix_stft = self.stft(audio)
        X = self.complexnorm(mix_stft)

        # initializing spectrograms variable
        spectrograms = torch.zeros(X.shape + (nb_sources,), dtype=audio.dtype, device=X.device)

        for j, (target_name, target_module) in enumerate(self.target_models.items()):
            # apply current model to get the source spectrogram
            target_spectrogram = target_module(X.detach().clone())
            spectrograms[..., j] = target_spectrogram

        # transposing it as
        # (nb_samples, nb_frames, nb_bins,{1,nb_channels}, nb_sources)
        spectrograms = spectrograms.permute(0, 3, 2, 1, 4)

        # rearranging it into:
        # (nb_samples, nb_frames, nb_bins, nb_channels, 2) to feed
        # into filtering methods
        mix_stft = mix_stft.permute(0, 3, 2, 1, 4)

        # create an additional target if we need to build a residual
        if self.residual:
            # we add an additional target
            nb_sources += 1

        if nb_sources == 1 and self.niter > 0:
            raise Exception(
                "Cannot use EM if only one target is estimated."
                "Provide two targets or create an additional "
                "one with `--residual`"
            )

        nb_frames = spectrograms.shape[1]
        targets_stft = torch.zeros(mix_stft.shape + (nb_sources,), dtype=audio.dtype, device=mix_stft.device)
        for sample in range(nb_samples):
            pos = 0
            if self.wiener_win_len:
                wiener_win_len = self.wiener_win_len
            else:
                wiener_win_len = nb_frames
            while pos < nb_frames:
                cur_frame = torch.arange(pos, min(nb_frames, pos + wiener_win_len))
                pos = int(cur_frame[-1]) + 1

                targets_stft[sample, cur_frame] = wiener(
                    spectrograms[sample, cur_frame],
                    mix_stft[sample, cur_frame],
                    self.niter,
                    softmask=self.softmask,
                    residual=self.residual,
                )

        # getting to (nb_samples, nb_targets, channel, fft_size, n_frames, 2)
        targets_stft = targets_stft.permute(0, 5, 3, 2, 1, 4).contiguous()

        # inverse STFT
        estimates = self.istft(targets_stft, length=audio.shape[2])

        return estimates

    def to_dict(self, estimates: torch.Tensor, aggregate_dict: Optional[dict] = None) -> dict:
        """Convert estimates as stacked tensor to dictionary

        Args:
            estimates (Tensor): separated targets of shape
                (nb_samples, nb_targets, nb_channels, nb_timesteps)
            aggregate_dict (dict or None)

        Returns:
            (dict of str: Tensor):
        """
        estimates_dict = {}
        for k, target in enumerate(self.target_models):
            estimates_dict[target] = estimates[:, k, ...]

        # in the case of residual, we added another source
        if self.residual:
            estimates_dict["residual"] = estimates[:, -1, ...]

        if aggregate_dict is not None:
            new_estimates = {}
            for key in aggregate_dict:
                new_estimates[key] = torch.tensor(0.0)
                for target in aggregate_dict[key]:
                    new_estimates[key] = new_estimates[key] + estimates_dict[target]
            estimates_dict = new_estimates
        return estimates_dict

    def seperate(self, audio: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        nb_sources = self.nb_targets
        nb_samples = audio.shape[0]

        # getting the STFT of mix:
        # (nb_samples, nb_channels, nb_bins, nb_frames, 2)
        mix_stft = self.stft(audio)
        X = self.complexnorm(mix_stft)

        # initializing spectrograms variable
        spectrograms = torch.zeros(X.shape + (nb_sources,), dtype=audio.dtype, device=X.device)

        for j, (target_name, target_module) in enumerate(self.target_models.items()):
            # apply current model to get the source spectrogram
            target_spectrogram = target_module(X.detach().clone())
            spectrograms[..., j] = target_spectrogram

        # Get estimated vocals
        vocals_spectrogram = spectrograms[..., 0].cpu().detach().numpy()  # Assuming vocals is the first target

        # Calculate nonvocals by subtracting vocals from the original mix
        mix_magnitude = X.cpu().detach().numpy()
        nonvocals_spectrogram = mix_magnitude - vocals_spectrogram

        nb_channels = vocals_spectrogram.shape[1]
        
        
        estimated_nonvocals = []

        # Inverse transform for vocals
        estimated_vocals = []
        for c in range(nb_channels):
            estimated_vocals.append(
                librosa.griffinlim(
                    vocals_spectrogram[0, c, :, :],
                    n_iter=32,
                    hop_length=self.stft.n_hop,
                    n_fft=self.stft.n_fft,
                    length=audio.shape[2],
                    pad_mode='constant',
                    momentum=0.99,
                )
            )
        estimated_vocals = np.stack(estimated_vocals, axis=1)
            
        # Inverse transform for nonvocals
        for c in range(nb_channels):
            estimated_nonvocals.append(
                librosa.griffinlim(
                    nonvocals_spectrogram[0, c, :, :],
                    n_iter=32,
                    hop_length=self.stft.n_hop,
                    n_fft=self.stft.n_fft,
                    length=audio.shape[2],
                    pad_mode='constant',
                    momentum=0.99,
                )
            )
        estimated_nonvocals = np.stack(estimated_nonvocals, axis=1)

        return estimated_vocals, estimated_nonvocals


def load_separator(
    checkpoint_path,
    targets: Optional[list] = None,
    niter: int = 1,
    residual: bool = False,
    wiener_win_len: Optional[int] = 300,
    device: Union[str, torch.device] = "cpu",
    pretrained: bool = True,
    filterbank: str = "torch",
    freeze=True,
) -> Separator:

    checkpoint = torch.load(os.path.join(checkpoint_path, CKPT_FILE), weights_only=True)['model']
    device = torch.device(f'cuda:0'if torch.cuda.is_available() else 'cpu')
    config = load_config(os.path.join(checkpoint_path, CONFIG_FILE))
    model = get_model(
        name=config.model_type,
        data_mean=np.array(config.train_data_mean),
        data_std=np.array(config.train_data_std),
        num_bins=config.nb_bins,
        num_channels=config.num_channels,
        hidden_size=config.hidden_size,
        max_bin=config.max_bin,
        unidirectional=config.unidirectional,
    )
    model.load_state_dict(checkpoint)
    model.to(device)

    separator = Separator(
        target_models={targets[0]: model},
        niter=niter,
        residual=residual,
        wiener_win_len=wiener_win_len,
        sample_rate=config.sample_rate,
        n_fft=config.nfft,
        n_hop=config.nhop,
        nb_channels=config.num_channels,
        filterbank=filterbank,
    ).to(device)

    if freeze:
        separator.freeze()

    return separator
