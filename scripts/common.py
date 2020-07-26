# -*- coding: utf-8 -*
"""Commonn functions."""
import cv2

import numpy as np
import tensorflow as tf

from tqdm import tqdm
from functools import partial
from multiprocessing import Pool
from pathlib import Path


def set_memory_growth():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.experimental.list_logical_devices(
                    'GPU')
                print('Detect {} Physical GPUs, {} Logical GPUs.'.format(
                        len(gpus), len(logical_gpus)))
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)


def pad_img(img, max_steps=32):
    """Pad image to suitable shape.

    Parameters
    ----------
    img : numpy.ndarray
        Read image.
    max_steps : int
        The number from train config (default is 32).

    Returns
    -------
    tuple
        Padded image and padding params for recovering.

    """
    img_h, img_w, _ = img.shape

    img_pad_h = 0
    if img_h % max_steps > 0:
        img_pad_h = max_steps - img_h % max_steps

    img_pad_w = 0
    if img_w % max_steps > 0:
        img_pad_w = max_steps - img_w % max_steps

    padd_val = np.mean(img, axis=(0, 1)).astype(np.uint8)
    img = cv2.copyMakeBorder(img, 0, img_pad_h, 0, img_pad_w,
                             cv2.BORDER_CONSTANT, value=padd_val.tolist())
    pad_params = (img_h, img_w, img_pad_h, img_pad_w)

    return img, pad_params


def read_dir(imgs_dir, rs_size=(150, 150)):
    for img_path in Path(imgs_dir.decode('utf-8')).glob('*'):
        img_original = cv2.imread(str(img_path))
        img = img_original[:, :, ::-1]
        img = cv2.resize(img, rs_size)
        img, _ = pad_img(img)

        yield img.astype(np.float32), img_original, int(img_path.stem)


def get_ds(imgs_dir, bs=1, rs_size=(150, 150)):
    """

    Parameters
    ----------
    imgs_dir : str
    bs
    rs_size
    workers

    Returns
    -------
    tf.data.Dataset
        Created dataset.

    """
    test_img = np.zeros((*rs_size, 3))
    test_img, _ = pad_img(test_img)

    ds = tf.data.Dataset.from_generator(
        partial(read_dir, rs_size=rs_size),
        args=[imgs_dir],
        output_types=(tf.float32, tf.int32),
        output_shapes=(test_img.shape, ())
    )
    ds = ds.batch(bs)

    return ds
