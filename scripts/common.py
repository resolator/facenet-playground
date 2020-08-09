# -*- coding: utf-8 -*
"""Commonn functions."""
import tensorflow as tf
from functools import partial


def set_memory_growth():
    """Set memory allocation more economical."""
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


def ds_map_fn(img_path_tf, img_size=(224, 224), max_steps=32,
              keep_ratio=False):
    """Auxiliary function for TF dataset."""
    img = tf.io.read_file(img_path_tf)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, img_size, preserve_aspect_ratio=keep_ratio)

    # img padding according to RetinaFace restrictions
    img_pad_h = img_size[0]
    h_diff = img_pad_h % max_steps
    if h_diff > 0:
        img_pad_h += max_steps - h_diff

    img_pad_w = img_size[1]
    w_diff = img_pad_w % max_steps
    if w_diff > 0:
        img_pad_w += max_steps - w_diff

    img = tf.image.pad_to_bounding_box(img, 0, 0, img_pad_h, img_pad_w)

    return img, img_path_tf


def get_ds(root_dir, bs=1, img_size=(150, 150), keep_ratio=False,
           exts=('*.jpg', '*.png')):
    """Create and return a TF dataset.

    Parameters
    ----------
    root_dir : str
        Path to dir with images for the dataset.
    bs : int
        Batch size.
    img_size : array-like
        Target size of resulted images.
    keep_ratio : bool
        Keep ratio when resize image (diff will be padded).
    exts : array-like
        Array with images extensions for load (type with "*.").

    Returns
    -------
    tensorflow.data.Dataset
        Created TF dataset.

    """
    if root_dir[-1] != '/':
        root_dir = root_dir + '/'

    patterns = [root_dir + x for x in exts]
    ds = tf.data.Dataset.list_files(patterns, shuffle=False)

    autotune = tf.data.experimental.AUTOTUNE
    map_fn = partial(ds_map_fn, img_size=img_size, keep_ratio=keep_ratio)
    ds = ds.map(tf.function(map_fn), num_parallel_calls=autotune)

    ds = ds.batch(bs)
    ds = ds.prefetch(buffer_size=autotune)

    return ds
