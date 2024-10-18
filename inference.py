import json
import os
from argparse import ArgumentParser, Namespace

import musdb
import museval
import soundfile as sf
import torch
from tqdm import tqdm

from src.preprocess import preprocess
from src.separator import load_separator


def parse_arguments() -> Namespace:
    parser = ArgumentParser(description='Inference')
    parser.add_argument(
        '--targets',
        default=['vocals'],
        type=str,
        nargs='+',
        help='provide targets to be processed. If none, all available targets will be computed',
    )
    parser.add_argument(
        '--task',
        default=1,
        type=int,
    )
    parser.add_argument(
        '--checkpoint_path',
        default='checkpoints/10-18-18-19-00',
        type=str,
        help='path to mode base directory of pretrained models',
    )
    parser.add_argument(
        '--outdir',
        type=str,
        default='checkpoints/10-18-18-19-00/results',
        help='Results path where audio evaluation results are stored',
    )
    parser.add_argument(
        '--evaldir',
        type=str,
        default='checkpoints/10-18-18-19-00/results',
        help='Results path for museval estimates',
    )
    parser.add_argument(
        '--root',
        type=str,
        default='musdb18',
        help='Path to MUSDB18',
    )
    parser.add_argument(
        '--niter',
        type=int,
        default=1,
        help='number of iterations for refining results.',
    )
    parser.add_argument(
        '--wiener-win-len',
        type=int,
        default=300,
        help='Number of frames on which to apply filtering independently',
    )
    parser.add_argument(
        '--residual',
        action='store_true',
        default=True,
        help='if provided, build a source with given name' 'for the mix minus all estimated targets',
    )
    parser.add_argument(
        '--aggregate',
        type=str,
        default='{"vocals":["vocals"], "accompaniment":["residual"]}',
        help='if provided, must be a string containing a valid expression for '
        'a dictionary, with keys as output target names, and values '
        'a list of targets that are used to build it. For instance: '
        "\'{'vocals':['vocals'], 'accompaniment':['drums', 'bass', 'other']}\'",
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()

    mus = musdb.DB(
        # root=args.root,
        subsets='test',
        download=True,
        is_wav=False,
    )

    aggregate_dict = None if args.aggregate is None else json.loads(args.aggregate)
    device = torch.device(f'cuda:0'if torch.cuda.is_available() else 'cpu')
    results = museval.EvalStore()
    separator = load_separator(
        checkpoint_path=args.checkpoint_path,
        targets=args.targets,
        niter=args.niter,
        residual=args.residual,
        wiener_win_len=args.wiener_win_len,
        device=device,
        pretrained=True,
    )

    for track in tqdm(mus.tracks):
        audio = torch.as_tensor(track.audio, dtype=torch.float32, device=device)
        audio = preprocess(audio, track.rate, separator.sample_rate)

        # Task 1
        if args.task == 1:
            estimates = separator(audio)
            estimates = separator.to_dict(estimates, aggregate_dict=aggregate_dict)

            for key in estimates:
                estimates[key] = estimates[key][0].cpu().detach().numpy().T

            if args.outdir:
                mus.save_estimates(estimates, track, args.outdir)
        
        elif args.task == 2:
            vocal_audio, nonvocal_audio = separator.seperate(audio)
            estimates = {
                'vocals': vocal_audio,
                'accompaniment': nonvocal_audio,
            }
            sf.write(f'separated_sample_vocal.wav', vocal_audio, separator.sample_rate)
            sf.write(f'separated_sample_nonvocal.wav', nonvocal_audio, separator.sample_rate)


        scores = museval.eval_mus_track(track, estimates, output_dir=args.evaldir)
        results.add_track(scores)
        print(track, '\n', scores)

    print(results)
    method = museval.MethodStore()
    method.add_evalstore(results, 'test')
    method.df.to_csv(os.path.join(args.checkpoint_path, 'test_results.csv'), index=False)
