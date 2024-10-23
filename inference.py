import json
import os
from argparse import ArgumentParser, Namespace

import musdb
import museval
import torch
from tqdm import tqdm

from src.preprocess import preprocess
from src.separator import load_separator
from src.utils import get_device


def parse_arguments() -> Namespace:
    parser = ArgumentParser(description='Inference')
    parser.add_argument(
        '--targets',
        type=str,
        default=['vocals'],
        nargs='+',
        help='provide targets to be processed. If none, all available targets will be computed',
    )
    parser.add_argument(
        '--task',
        type=int,
        default=1,
    )
    parser.add_argument(
        '--checkpoint_path',
        type=str,
        default='checkpoints/10-19-21-01-27',
        help='path to mode base directory of pretrained models',
    )
    parser.add_argument(
        '--weight',
        type=str,
        default='checkpoint.pth',
        help='path to model checkpoint',
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
    save_folder = os.path.join(args.checkpoint_path, args.weight.replace(".pth", ""), f'task{args.task}')

    mus = musdb.DB(
        root=args.root,
        subsets='test',
        # download=True,
        is_wav=True,
    )

    aggregate_dict = None if args.aggregate is None else json.loads(args.aggregate)
    device = get_device()
    results = museval.EvalStore()
    separator = load_separator(
        checkpoint_path=args.checkpoint_path,
        weight=args.weight,
        targets=args.targets,
        niter=args.niter,
        residual=args.residual,
        wiener_win_len=args.wiener_win_len,
        device=device,
    )

    for track in tqdm(mus.tracks):
        audio = torch.as_tensor(track.audio, dtype=torch.float32, device=device)
        audio = preprocess(audio, track.rate, separator.sample_rate)

        # Task 1
        match args.task:
            case 1:
                estimates = separator(audio)
                estimates = separator.to_dict(estimates, aggregate_dict=aggregate_dict)

                for key in estimates:
                    estimates[key] = estimates[key][0].cpu().detach().numpy().T

            case 2:
                estimates = separator.seperate(audio)

        mus.save_estimates(estimates, track, save_folder)
        scores = museval.eval_mus_track(
            track,
            estimates,
            output_dir=save_folder,
        )
        results.add_track(scores)
        print(track, '\n', scores)

    print(results)

    method = museval.MethodStore()
    method.add_evalstore(results, 'test')
    method.df.to_csv(os.path.join(save_folder,'test_results.csv'),index=False)
