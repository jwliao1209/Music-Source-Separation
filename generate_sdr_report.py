import os
from argparse import ArgumentParser, Namespace

import pandas as pd

from src.constants import CONFIG_FILE
from src.utils import load_config


def parse_arguments() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        '--checkpoint_path',
        type=str,
        default='checkpoints/10-17-22-55-15',
    )
    parser.add_argument(
        '--weight',
        type=str,
        default='checkpoint',
    )
    parser.add_argument(
        '--task',
        type=int,
        default=1,
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    config = load_config(os.path.join(args.checkpoint_path, CONFIG_FILE))
    df = pd.read_csv(
            os.path.join(args.checkpoint_path, args.weight, f'task{args.task}', 'test_results.csv')
        )

    vocal_sdr = df[
        (df['target'] == 'vocals') & (df['metric'] == 'SDR')
    ].groupby(by='track')['score'].median().median()

    nonvocal_sdr = df[
        (df['target'] == 'accompaniment') & (df['metric'] == 'SDR')
    ].groupby(by='track')['score'].median().median()

    print('Task:', args.task)
    print('Model:', config.model_type)
    print('Loss:', config.loss)
    print('Optimizer:', config.optimizer)
    print('Augmentations:', config.source_augmentations)
    print('Vocal median of median SDR:', f'{vocal_sdr:.4f}')
    print('Non-vocal median of median SDR:', f'{nonvocal_sdr:.4f}')
    print('===============================================')
