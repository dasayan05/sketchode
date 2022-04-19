'''
Unpack Quick Draw OR DiDi data to make Data loading more efficient.
Otherwise full loading of '.ndjson' takes a while.

For QD, "python unpack_ndjson.py --data_folder /path/to/QD/raw -c cat -o /path/to/empty/dir"
For DiDi, "python unpack_ndjson.py --data_folder /path/to/DiDi -c diagrams_wo_text_20200131 -o /path/to/empty/dir"
Author: Ayan Das
'''

import os
import pickle
import argparse
import numpy as np
import ndjson as nj
from tqdm import tqdm


def main(args):
    data_path = os.path.join(args.data_folder, args.category + '.ndjson')
    with open(data_path, 'r') as f:
        data = nj.load(f)

    train_path = os.path.join(args.out_folder, args.category, 'train')
    test_path = os.path.join(args.out_folder, args.category, 'test')

    if not os.path.exists(train_path):
        os.makedirs(train_path)

    if not os.path.exists(test_path):
        os.makedirs(test_path)

    for i, sample in enumerate(tqdm(data)):
        out_path = train_path if np.random.rand() < 0.8 else test_path
        out_path = os.path.join(out_path, f'sketch_{i}')
        with open(out_path, 'wb') as f:
            pickle.dump(sample, f)

        if i > args.max_sketches:
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_folder', type=str, required=True,
                        help='QD folder of raw data (.ndjson)')
    parser.add_argument('-c', '--category', type=str, required=True, help='name of a category')
    parser.add_argument('-o', '--out_folder', type=str, required=True, help='output folder (empty)')
    parser.add_argument('-m', '--max_sketches', type=int, required=False, default=10000)
    args = parser.parse_args()

    main(args)
