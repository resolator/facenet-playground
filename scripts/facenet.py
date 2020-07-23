# -*- coding: utf-8 -*
"""FaceNet inference implementation."""
import cv2
import numpy as np
import tensorflow as tf


class FaceNet:
    """FaceNet inference implementation."""

    def __init__(self, model_path):
        """Class constructor.

        Parameters
        ----------
        model_path : str
            Path to a trained FaceNet model.

        """
        # model loading
        graph_def = tf.GraphDef()
        with open(model_path, 'rb') as f:
            graph_def.ParseFromString(f.read())

        with tf.Graph().as_default() as graph:
            tf.import_graph_def(graph_def, name='')

        # session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self._sess = tf.Session(graph=graph, config=config)

        # graph nodes
        self._img_ph = graph.get_tensor_by_name('input:0')
        self._is_training_ph = graph.get_tensor_by_name('phase_train:0')
        self._emb_op = graph.get_tensor_by_name('embeddings:0')

    @staticmethod
    def _preprocess(cv_rgb_img):
        """Preprocess image before forward pass.

        Parameters
        ----------
        cv_rgb_img : np.array
            3-dimensional array (image).

        Returns
        -------
        np.array
            Preprocessed image.

        """
        mean = np.mean(cv_rgb_img)
        std = np.std(cv_rgb_img)
        std_adj = np.maximum(std, 1.0 / np.sqrt(cv_rgb_img.size))
        res = np.multiply(np.subtract(cv_rgb_img, mean), 1 / std_adj)

        return res

    def calc_embeddings(self, cv_bgr_imgs):
        """Calculate embeddings for given images.

        Parameters
        ----------
        cv_bgr_imgs : array-like or str
            Array with BGR images (supposed to be read with cv2.imread).
            If array of str passed then try to read them with cv2.imread.

        Returns
        -------
        np.array
            An array with calculated embeddings for each image.

        """
        if isinstance(cv_bgr_imgs[0], str):
            cv_rgb_imgs = [cv2.imread(x)[:, :, ::-1] for x in cv_bgr_imgs]
        else:
            cv_rgb_imgs = [x[:, :, ::-1] for x in cv_bgr_imgs]

        # resize
        model_img_shape = (160, 160)
        resized_imgs = [cv2.resize(x, model_img_shape,
                                   interpolation=cv2.INTER_LINEAR)
                        for x in cv_rgb_imgs]

        # preprocess
        preprocessed_imgs = [self._preprocess(x) for x in resized_imgs]

        # calculate embeddings
        embeddings = self._sess.run(self._emb_op,
                                    feed_dict={self._img_ph: preprocessed_imgs,
                                               self._is_training_ph: False})

        return embeddings
