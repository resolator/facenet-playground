# -*- coding: utf-8 -*
"""Wrapper for face detection."""
import tensorflow as tf


class FaceDetector:
    """Wrapper for face detection."""

    def __init__(self, saved_model_tf, expand_factors=None):
        """Class constructor.

        Parameters
        ----------
        saved_model_tf : tf.saved_model
            Loaded TF saved model.
        expand_factors : array-like
            Array with TOP, BOTTOM, RIGHT and LEFT expand factors for
            detections.

        """
        self._det = saved_model_tf
        self._expand_factors = expand_factors

    def detect_and_cut(self, imgs_batch):
        """Detect faces, cut first one and return resulted image.

        Parameters
        ----------
        imgs_batch : tf.Tensor
            Batch of images.

        Returns
        -------
        np.array
            Cropped and resized for FaceNet face
            or resized source image if no face was found.

        """
        outputs = self._det(imgs_batch)
        boxes_batch = tf.squeeze(
            outputs['tf_op_layer_CombinedNonMaxSuppression'], axis=1)

        cropped_imgs = self._tf_crop(imgs_batch, boxes_batch)

        return cropped_imgs

    @tf.function
    def _tf_crop(self, images, boxes, min_size=(20, 20)):
        face_shape = (160, 160)  # according to the FaceNet
        boxes = tf.abs(boxes)
        cropped_imgs = []

        for i, (img, box) in enumerate(zip(tf.unstack(images),
                                           tf.unstack(boxes))):
            # relative x1, y1, x2, y2
            y1 = img.shape[0] * box[0]
            x1 = img.shape[1] * box[1]
            y2 = img.shape[0] * box[2]
            x2 = img.shape[1] * box[3]

            # expand detection
            if self._expand_factors is not None:
                crop_h, crop_w = y2 - y1, x2 - x1
                y1 = tf.maximum(y1 - crop_h * self._expand_factors[0], 0)
                x1 = tf.maximum(x1 - crop_w * self._expand_factors[3], 0)

                y2 = tf.minimum(y2 + crop_h * self._expand_factors[1],
                                img.shape[0])
                x2 = tf.minimum(x2 + crop_w * self._expand_factors[2],
                                img.shape[1])

            # crop
            cropped = img
            if y2 > min_size[0] and x2 > min_size[1]:
                cropped = tf.image.crop_to_bounding_box(
                    img,
                    tf.cast(y1, tf.int32),
                    tf.cast(x1, tf.int32),
                    tf.cast(y2 - y1, tf.int32),
                    tf.cast(x2 - x1, tf.int32)
                )

            cropped = tf.image.resize(cropped, face_shape)

            # facenet preproc
            mean = tf.math.reduce_mean(cropped)
            std = tf.math.reduce_std(cropped)
            std_adj = tf.maximum(std, 1.0 / tf.sqrt(
                tf.cast(cropped.shape[0] * cropped.shape[1] * cropped.shape[2],
                        tf.float32)))
            res = tf.multiply(tf.subtract(cropped, mean), 1 / std_adj)

            cropped_imgs.append(res)

        return tf.stack(cropped_imgs)
