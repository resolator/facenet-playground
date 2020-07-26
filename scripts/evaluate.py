#!/usr/bin/env python3
# -*- coding: utf-8 -*
"""Evaluate two .tsv files."""
import argparse
import pandas as pd


def get_args():
    """Arguments parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gt-tsv', required=True,
                        help='Path to .tsv with ground truth.')
    parser.add_argument('--pd-tsv', required=True,
                        help='Path to .tsv with prediction.')

    return parser.parse_args()


def main():
    """Application entry point."""
    args = get_args()
    gt_df = pd.read_csv(args.gt_tsv, sep='\t', names=['gt_mark', 'test'])
    pd_df = pd.read_csv(args.pd_tsv, sep='\t', names=['pd_mark', 'test'])

    gt_df = gt_df.set_index('test')
    pd_df = pd_df.set_index('test')

    merged_df = gt_df.join(pd_df)
    hits_mask = merged_df['gt_mark'] == merged_df['pd_mark']
    hits = hits_mask.sum()

    print('Accuracy:', hits / len(hits_mask))


if __name__ == '__main__':
    main()
