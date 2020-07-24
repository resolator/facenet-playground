#!/usr/bin/env python3
# -*- coding: utf-8 -*
"""Demo application to use FaceNet."""
import cv2
import sys
import nmslib
import argparse

import pandas as pd

from tqdm import tqdm
from pathlib import Path
from face_net import FaceNet
from face_detector import FaceDetector


def get_args():
    """Arguments parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--database-dir', type=Path, required=True,
                        help='Path to dir with images only for create a '
                             'database.')
    parser.add_argument('--testing-dir', type=Path,
                        help='Path to dir with images only for search in '
                             'database.')
    parser.add_argument('--img-path',
                        help='Path to image for search in database.')
    parser.add_argument('--model-path',
                        default=sys.path[0] + '/../data/facenet.pb',
                        help='Path to saved model.')
    parser.add_argument('--bs', type=int, default=1,
                        help='Batch size for testing dir. Increasing could '
                             'speed up testing.')
    parser.add_argument('--scale-factor', type=float, default=1.01,
                        help='Hyperparameter for face detection.')
    parser.add_argument('--min-neighbors', type=int, default=5,
                        help='Hyperparameter for face detection.')
    parser.add_argument('--detector-name',
                        default='haarcascade_frontalface_alt.xml',
                        choices=['haarcascade_frontalface_alt.xml',
                                 'haarcascade_frontalface_default.xml',
                                 'haarcascade_frontalface_alt_tree.xml',
                                 'haarcascade_frontalface_alt2.xml'],
                        help='Name of the cascade config (model name).')
    parser.add_argument('--save-to', type=Path,
                        help='Path to save dir.')

    return parser.parse_args()


def predict_once(img_path, db, net):
    """Search closest neighbor for image in the database.

    Parameters
    ----------
    img_path : str
        Path to image for search in the database.
    db : nmslib.Index
        nmslib database.
    net : FaceNet
        Instance of FaceNet.

    Returns
    -------
    int
        Index of the closest neighbor in the database.

    """
    emb = net.calc_embeddings([img_path])[0]
    db_idx, _ = db.knnQuery(emb, k=1)

    return db_idx


def predict(images_dir, db, net, detector, bs=1):
    """Search closest neighbor in the database for each image in images_dir.

    Parameters
    ----------
    images_dir : pathlib.Path
        Path to the dir with images for search in the database.
    db : nmslib.Index
        nmslib database.
    net : FaceNet
        Instance of the FaceNet.
    detector : FaceDetector
        Instance of the FaceDetector.
    bs : int
        Batch size (increasing could speed up calculation).

    Returns
    -------
    dict
        Dict with closest indexes in the database and image paths.

    """
    imgs_paths = list(images_dir.glob('*'))

    db_idxs = []
    for i in tqdm(range(0, len(imgs_paths), bs), desc='Predicting'):
        batch_paths = imgs_paths[i:i + bs]
        imgs = [cv2.imread(str(x)) for x in batch_paths]
        print(batch_paths[0])
        print(imgs[0])
        faces = [detector.detect_and_cut(x) for x in imgs]
        embds = net.calc_embeddings(faces)
        db_idxs.extend([x[0][0] for x in db.knnQueryBatch(embds, k=1)])

    return {'db_idx': db_idxs, 'test_name': [int(x.stem) for x in imgs_paths]}


def main():
    """Application entry point."""
    args = get_args()

    if args.img_path is None and args.testing_dir is None:
        raise RuntimeError(
            'nothing to process. Pass --img-path or --testing-dir.'
        )

    if args.testing_dir is not None and args.save_to is None:
        raise NotImplementedError('pass also --save-to with --testing-dir.')

    # calculate embeddings and create a database
    net = FaceNet(args.model_path)
    detector = FaceDetector(scale_factor=args.scale_factor,
                            min_neighbors=args.min_neighbors,
                            model_name=args.detector_name)
    db = nmslib.init(method='hnsw', space='l2')

    for img_path in args.database_dir.glob('*'):
        emb = net.calc_embeddings([str(img_path)])[0]
        db.addDataPoint(data=emb, id=int(img_path.stem))

    db.createIndex(print_progress=True)

    # demo work
    if args.img_path is not None:
        # TODO: do something with output
        predict_once(args.img_path, db, net)

    if args.testing_dir is not None:
        predicted = predict(args.testing_dir, db, net, detector, args.bs)
        args.save_to.mkdir(parents=True, exist_ok=True)
        save_file = args.save_to.joinpath('predicts.tsv')
        df = pd.DataFrame(predicted)
        df.to_csv(save_file, index=False, sep='\t', header=False)

    print('Done!\n')


if __name__ == '__main__':
    main()
