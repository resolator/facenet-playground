#!/usr/bin/env python3
# -*- coding: utf-8 -*
"""Detect, cut and save faces."""
import cv2
import sys
import argparse

import tensorflow as tf

from tqdm import tqdm
from pathlib import Path
from face_detector import FaceDetector
from common import set_memory_growth


def get_args():
    """Arguments parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--images-dir', type=Path, required=True,
                        help='Path to dir with images only for search in '
                             'database.')
    parser.add_argument('--detector-model-dir',
                        default=sys.path[0] + '/../data/face_detector/',
                        help='Path to dir with tf saved model for detector.')
    parser.add_argument('--bs', type=int, default=1,
                        help='Batch size.')
    parser.add_argument('--save-to', type=Path, required=True,
                        help='Path to save dir to save cropped images.')

    return parser.parse_args()


def main():
    """Application entry point."""
    args = get_args()
    set_memory_growth()
    args.save_to.mkdir(exist_ok=True, parents=True)

    loaded = tf.saved_model.load(args.detector_model_dir)
    det = loaded.signatures['serving_default']

    detector = FaceDetector(det)

    imgs_paths = list(args.images_dir.glob('*'))
    for i in tqdm(range(0, len(imgs_paths), args.bs), desc='Cutting'):
        batch = imgs_paths[i:i + args.bs]
        imgs_raw = [cv2.imread(str(x)) for x in batch]
        faces = detector.detect_and_cut(imgs_raw)

        for face, old_img_path in zip(faces, batch):
            new_img_path = args.save_to.joinpath(old_img_path.name)
            cv2.imwrite(str(new_img_path), face)


if __name__ == '__main__':
    main()
