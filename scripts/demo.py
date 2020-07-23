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
                        help='Path to dir with images only for create a '
                             'database.')
    parser.add_argument('--testing-dir', type=Path,
                        help='Path to dir with images only for search in '
                             'database.')
    parser.add_argument('--img-path',
                        help='Path to image for search in database.')
    parser.add_argument('--model-path', default='../data/facenet.pb',
                        help='Path to saved model.')
    parser.add_argument('--bs', type=int, default=1,
                        help='Batch size for testing dir. Increasing could '
                             'speed up testing.')
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


def predict(images_dir, db, net, bs=1):
    """Search closest neighbor in the database for each image in images_dir.

    Parameters
    ----------
    images_dir : pathlib.Path
        Path to the dir with images for search in the database.
    db : nmslib.Index
        nmslib database.
    net : FaceNet
        Instance of FaceNet.
    bs : int
        Batch size (increasing could speed up calculation).

    Returns
    -------
    List
        List of pairs [dataset_idx, img_path].

    """
    imgs_paths = list(images_dir.glob('*'))

    db_idxs = []
    for i in range(0, len(imgs_paths), bs):
        batch_paths = imgs_paths[i:i + bs]
        imgs = [cv2.imread(x) for x in batch_paths]
        embds = net.calc_embeddings(imgs)
        db_out = db.knnQueryBatch(embds, k=1)
        db_idxs.extend([[x[0][0], y] for x, y in zip(db_out, batch_paths)])

    return db_idxs


def main():
    """Application entry point."""
    args = get_args()

    if args.img_path is None and args.testing_dir is None:
        print('Nothing to process. Pass --img-path or --testing-dir.')
        return

    # calculate embeddings and create a database
    net = FaceNet(args.model_path)
    db = nmslib.init(method='hnsw', space='l2')

    for img_path in args.database_dir.glob('*'):
        emb = net.calc_embeddings([str(img_path)])[0]
        db.addDataPoint(emb)

    db.createIndex(print_progress=True)

    # demo work
    if args.img_path is not None:
        # TODO: do something with output
        predict_once(args.img_path, db, net)

    if args.img_path is not None:
        # TODO: do something with output
        predict(args.testing_dir, db, net)

    print('Done!\n')


if __name__ == '__main__':
    main()
