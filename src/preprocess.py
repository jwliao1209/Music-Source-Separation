import warnings
from typing import Optional

import torch
from torchaudio.transforms import Resample


def preprocess(
    audio: torch.Tensor,
    rate: Optional[float] = None,
    model_rate: Optional[float] = None,
) -> torch.Tensor:
    """
    From an input tensor, convert it to a tensor of shape
    shape = (nb_samples, nb_channels, nb_timesteps). This includes:
    -  if input is 1D, adding the samples and channels dimensions.
    -  if input is 2D
        o and the smallest dimension is 1 or 2, adding the samples one.
        o and all dimensions are > 2, assuming the smallest is the samples
          one, and adding the channel one
    - at the end, if the number of channels is greater than the number
      of time steps, swap those two.
    - resampling to target rate if necessary

    Args:
        audio (Tensor): input waveform
        rate (float): sample rate for the audio
        model_rate (float): sample rate for the model

    Returns:
        Tensor: [shape=(nb_samples, nb_channels=2, nb_timesteps)]
    """
    shape = torch.as_tensor(audio.shape, device=audio.device)

    if len(shape) == 1:
        # assuming only time dimension is provided.
        audio = audio[None, None, ...]

    elif len(shape) == 2:
        if shape.min() <= 2:
            # assuming sample dimension is missing
            audio = audio[None, ...]
        else:
            # assuming channel dimension is missing
            audio = audio[:, None, ...]

    if audio.shape[1] > audio.shape[2]:
        # swapping channel and time
        audio = audio.transpose(1, 2)

    if audio.shape[1] > 2:
        warnings.warn("Channel count > 2!. Only the first two channels " "will be processed!")
        audio = audio[..., :2]

    if audio.shape[1] == 1:
        # if we have mono, we duplicate it to get stereo
        audio = torch.repeat_interleave(audio, 2, dim=1)

    if rate != model_rate:
        warnings.warn("resample to model sample rate")
        # we have to resample to model samplerate if needed this makes sure we resample input only once
        resampler = Resample(
            orig_freq=rate,
            new_freq=model_rate,
            resampling_method="sinc_interpolation"
        ).to(audio.device)
        audio = resampler(audio)

    return audio
