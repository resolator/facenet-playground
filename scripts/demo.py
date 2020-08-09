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
from face_detector import FaceDetector
from common import set_memory_growth, get_ds


def get_args():
    """Arguments parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--database-dir', required=True,
                        help='Path to dir with images only for create a '
                             'database.')
    parser.add_argument('--testing-dir',
                        help='Path to dir with images only for search in '
                             'database.')
    parser.add_argument('--img-path', type=Path,
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
    parser.add_argument('--img-size', type=int, nargs=2, default=(150, 150),
                        help='Resize all images to this size before run.')
    parser.add_argument('--keep-ratio', action='store_true',
                        help='Keep ratio when resize image '
                             '(diff will be padded).')
    parser.add_argument('--expand-factors', type=float, nargs=4,
                        help='Expand resulted detections by factors '
                             '(top, bottom, right, left).')
    parser.add_argument('--space', default='l2',
                        choices=['l2', 'cosinesimil'],
                        help='Space for embeddings.')
    parser.add_argument('--save-to', type=Path,
                        help='Path to save dir.')

    return parser.parse_args()


def calc_embeddings(ds, detector, net, desc=''):
    """Calculate embeddings (with faces extraction) for given list of paths.

    Parameters
    ----------
    ds : tensorflow.data.Dataset
        Created TF dataset.
    detector : FaceDetector
        Instance of the FaceDetector.
    net : FaceNet
        Instance of the FaceNet.
    desc : str
        Optional description for tqdm loop.

    Returns
    -------
    tuple
        Embedding and images IDs.

    """
    for imgs_batch, paths_batch in tqdm(ds, desc=desc):
        faces_batch = detector.detect_and_cut(imgs_batch)
        emb_batch = net.calc_embeddings(faces_batch)

        ids = [int(Path(x.numpy().decode('utf-8')).stem) for x in paths_batch]

        yield emb_batch, ids


def predict(ds, db, net, detector):
    """Search closest neighbor in the database for each image in images_dir.

    Parameters
    ----------
    ds : tensorflow.data.Dataset
        Created TF dataset.
    db : nmslib.Index
        nmslib database.
    net : FaceNet
        Instance of the FaceNet.
    detector : FaceDetector
        Instance of the FaceDetector.

    Returns
    -------
    dict
        Dict with closest indexes in the database and image paths.

    """
    db_idxs = []
    gt_idxs = []

    gen = calc_embeddings(ds, detector, net, 'Predicting')
    for emb_batch, ids in gen:
        db_out = db.knnQueryBatch(emb_batch, k=1)
        db_idxs.extend([x[0][0] for x in db_out])
        gt_idxs.extend(ids)

    return {'db_idx': db_idxs, 'test_name': gt_idxs}


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
    det_tf = loaded.signatures['serving_default']
    detector = FaceDetector(det_tf, args.expand_factors)

    net = FaceNet(args.model_path)
    db = nmslib.init(method='hnsw', space=args.space)

    # calculate embeddings and create a database
    print('\nBulding database')
    db_ds = get_ds(args.database_dir, args.bs, args.img_size, args.keep_ratio)
    gen = calc_embeddings(db_ds, detector, net, 'Building database')

    for emb_batch, ids in gen:
        db.addDataPointBatch(data=emb_batch, ids=ids)

    db.createIndex()

    # demo work
    print('\nDemo')
    if args.img_path is not None:
        ds = get_ds(str(args.img_path.parent),
                    args.bs,
                    args.img_size,
                    args.keep_ratio,
                    exts=[args.img_path.name])
        img_id = predict(ds, db, net, detector)['db_idx'][0]
        closets_img_path = Path(args.database_dir).joinpath(
            str(img_id) + '.jpg')
        img = cv2.imread(str(closets_img_path))
        print('Closest img path:', closets_img_path)

        cv2.namedWindow('img', cv2.WINDOW_NORMAL)
        cv2.imshow('img', img)
        cv2.waitKey()

    if args.testing_dir is not None:
        ds = get_ds(args.testing_dir, args.bs, args.img_size, args.keep_ratio)
        predicted = predict(ds, db, net, detector)

        # save results
        args.save_to.mkdir(parents=True, exist_ok=True)
        save_file = args.save_to.joinpath('predicts.tsv')
        df = pd.DataFrame(predicted)
        df.to_csv(save_file, index=False, sep='\t', header=False)

    print('Done!\n')


if __name__ == '__main__':
    main()
