
import librosa
import musdb
import numpy as np
import soundfile as sf
import torch
import tqdm

from openunmix import utils


def load_model(model_str_or_path, targets, niter, residual, wiener_win_len, device, filterbank):
    separator = utils.load_separator(
        model_str_or_path=model_str_or_path,
        targets=targets,
        niter=niter,
        residual=residual,
        wiener_win_len=wiener_win_len,
        device=device,
        pretrained=True,
        filterbank=filterbank,
        output_spectrograms=True,
    )
    separator.freeze()
    separator.to(device)
    return separator


def main():
    device = torch.device("cuda")
    model = load_model(
        model_str_or_path='/home/jiawei/Desktop/github/Source-Separation/open-unmix-pytorch/checkpoint',
        targets=['vocals'],
        niter=1,
        residual=True,
        wiener_win_len=None,
        device=device,
        filterbank="torch"
    )
    mus = musdb.DB(
        root="/home/jiawei/Desktop/github/Source-Separation/musdb18",
        subsets='test',
        is_wav=True,
    )

    for track in tqdm.tqdm(mus.tracks):
        print(track)
        audio = torch.as_tensor(track.audio, dtype=torch.float32, device=device)
        audio = utils.preprocess(audio, track.rate, model.sample_rate)
        spectrograms = model(audio).cpu().detach().numpy()
        # (nb_samples, nb_channels, nb_bins, nb_frames, nb_sources)

        nb_channels = spectrograms.shape[1]
        estimated_audio = []
        for c in range(nb_channels):
            estimated_audio.append(
                librosa.griffinlim(
                    spectrograms[0, c, :, :, 0],
                    n_iter=32,
                    hop_length=model.stft.n_hop,
                    win_length=model.wiener_win_len,
                    n_fft=model.stft.n_fft,
                    length=audio.shape[2],
                    pad_mode='constant',
                    momentum=0.99, 
                )
            )

        estimated_audio = np.stack(estimated_audio, axis=1)
        sf.write(
            f"separated_sample_vocal.wav",
            estimated_audio,
            model.sample_rate
        )


if __name__ == "__main__":
    main()
