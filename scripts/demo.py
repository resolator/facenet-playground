#!/usr/bin/env python3
# -*- coding: utf-8 -*
"""Demo application to use FaceNet."""
import cv2
import sys
import nmslib
import argparse

import pandas as pd
import tensorflow as tf

from tqdm import tqdm
from pathlib import Path
from face_net import FaceNet
from common import set_memory_growth
from face_detector import FaceDetector


def get_args():
    """Arguments parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--database-dir', type=Path, required=True,
                        help='Path to dir with images only for create a '
                             'database.')
    parser.add_argument('--testing-dir',
                        help='Path to dir with images only for search in '
                             'database.')
    parser.add_argument('--already-faces', action='store_true',
                        help='Testing dir already contains faces only.')
    parser.add_argument('--img-path',
                        help='Path to image for search in database.')
    parser.add_argument('--model-path',
                        default=sys.path[0] + '/../data/facenet.pb',
                        help='Path to saved model.')
    parser.add_argument('--detector-model-dir',
                        default=sys.path[0] + '/../data/face_detector/',
                        help='Path to dir with tf saved model for detector.')
    parser.add_argument('--bs', type=int, default=1,
                        help='Batch size for testing dir. Increasing could '
                             'speed up testing.')
    parser.add_argument('--save-to', type=Path,
                        help='Path to save dir.')

    return parser.parse_args()


def calc_embeddings(imgs_paths, detector, net, bs=1):
    """Calculate embeddings (with faces extraction) for given list of paths.

    Parameters
    ----------
    imgs_paths : array-like
        Paths to images for calculate embeddings.
    detector : FaceDetector
        Instance of the FaceDetector.
    net : FaceNet
        Instance of the FaceNet.
    bs : int
        Batch size.

    Returns
    -------
    array-like
        Array on embeddings with shape [len(

    """
    for i in range(0, len(imgs_paths), bs):
        batch_paths = imgs_paths[i:i + bs]
        imgs_batch = [cv2.imread(str(x))[:, :, ::-1] for x in batch_paths]
        faces_batch = detector.detect_and_cut(imgs_batch)
        emb_batch = net.calc_embeddings(faces_batch)

        ids = [int(x.stem) for x in batch_paths]

        yield emb_batch, ids


def predict_once(img_path, db, net, detector):
    """Search closest neighbor for image in the database.

    Parameters
    ----------
    img_path : str
        Path to image for search in the database.
    db : nmslib.Index
        nmslib database.
    net : FaceNet
        Instance of FaceNet.
    detector : FaceDetector
        Instance of FaceDetector.

    Returns
    -------
    int
        Index of the closest neighbor in the database.

    """
    img = cv2.imread(img_path)[:, :, ::-1]  # BGR2RGB
    face = detector.detect_and_cut([img])[0]
    emb = net.calc_embeddings([face])[0]
    db_idx = db.knnQuery(emb, k=1)[0][0]

    return db_idx


def predict(images_dir, db, net, detector, bs):
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
    gt_idxs = []

    gen = calc_embeddings(imgs_paths, detector, net, bs)
    for emb_batch, ids in tqdm(gen, desc='Predicting'):
        db_out = db.knnQueryBatch(emb_batch, k=1)
        db_idxs.extend([x[0][0] for x in db_out])
        gt_idxs.extend(ids)

    return {'db_idx': db_idxs, 'test_name': [int(x.stem) for x in imgs_paths]}


def main():
    """Application entry point."""
    args = get_args()
    set_memory_growth()

    if args.img_path is None and args.testing_dir is None:
        raise RuntimeError(
            'nothing to process. Pass --img-path or --testing-dir.'
        )

    if args.testing_dir is not None and args.save_to is None:
        raise NotImplementedError('pass also --save-to with --testing-dir.')

    # initialize nets
    print('\nNets initialization')
    loaded = tf.saved_model.load(args.detector_model_dir)
    det = loaded.signatures['serving_default']
    detector = FaceDetector(det)

    net = FaceNet(args.model_path)
    db = nmslib.init(method='hnsw', space='l2')

    # calculate embeddings and create a database
    print('\nBulding database')
    imgs_paths = list(args.database_dir.glob('*'))
    gen = calc_embeddings(imgs_paths, detector, net, args.bs)
    for emb_batch, ids in tqdm(gen, 'Building database'):
        db.addDataPointBatch(data=emb_batch, ids=ids)

    db.createIndex()

    # demo work
    print('\nDemo')
    if args.img_path is not None:
        img_id = predict_once(args.img_path, db, net, detector)
        closets_img_path = args.database_dir.joinpath(str(img_id) + '.jpg')
        img = cv2.imread(str(closets_img_path))
        print('Closest img path:', closets_img_path)

        cv2.namedWindow('img', cv2.WINDOW_NORMAL)
        cv2.imshow('img', img)
        cv2.waitKey()

    if args.testing_dir is not None:
        predicted = predict(args.testing_dir, db, net, detector, args.bs)
        args.save_to.mkdir(parents=True, exist_ok=True)
        save_file = args.save_to.joinpath('predicts.tsv')
        df = pd.DataFrame(predicted)
        df.to_csv(save_file, index=False, sep='\t', header=False)

    print('Done!\n')


if __name__ == '__main__':
    main()
