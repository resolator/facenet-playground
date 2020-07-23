#!/usr/bin/env python3
# -*- coding: utf-8 -*
"""Demo application to use FaceNet."""
import cv2
import nmslib
import argparse

from pathlib import Path
from facenet import FaceNet


def get_args():
    """Arguments parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--database-dir', type=Path, required=True,
                        help='Path to dir with images for database only.')
    parser.add_argument('--testing-dir', type=Path,
                        help='Path to dir with images for search in database.')
    parser.add_argument('--img-path',
                        help='Path to image for search in database.')
    parser.add_argument('--model-path', default='../data/facenet.pb',
                        help='Path to saved model.')
    parser.add_argument('--save-to', type=Path,
                        help='Path to save dir.')

    return parser.parse_args()


def predict_once():
    pass


def predict():
    pass


def main():
    """Application entry point."""
    args = get_args()

    if args.img_path is None and args.testing_dir is None:
        print('Nothing for process. Pass --img-path or --testing-dir.')
        return

    # calculate embeddings and create a database
    net = FaceNet(args.model_path)
    db = nmslib.init(method='hnsw', space='l2')

    for img_path in args.database_dir.glob('*'):
        img = cv2.imread(img_path)
        emb = net.calc_embeddings([img])[0]
        db.addDataPoint(emb)

    db.createIndex(print_progress=True)

    # demo work
    if args.img_path is not None:
        predict_once()

    if args.img_path is not None:
        predict()


if __name__ == '__main__':
    main()
