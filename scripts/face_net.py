# -*- coding: utf-8 -*
"""FaceNet inference implementation."""
import tensorflow.compat.v1 as tf


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

    def calc_embeddings(self, imgs_batch):
        """Calculate embeddings for given images.

        Parameters
        ----------
        imgs_batch : tf.EagerTensor
            TF EagerTensor with resized to (160, 160) faces.

        Returns
        -------
        list
            A list with calculated embeddings for each image.

        """
        np_imgs_batch = imgs_batch.numpy()
        embeddings = self._sess.run(self._emb_op,
                                    feed_dict={self._img_ph: np_imgs_batch,
                                               self._is_training_ph: False})

        return embeddings.tolist()
