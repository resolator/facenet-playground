# -*- coding: utf-8 -*
"""Wrapper for face detection."""
import cv2
import numpy as np
import tensorflow as tf


class FaceDetector:
    """Wrapper for face detection using MTCNN."""

    def __init__(self, saved_model_tf):
        """Class constructor.

        Parameters
        ----------
        saved_model_tf : tf.saved_model
            Loaded TF saved model.

        """
        # loaded = tf.saved_model.load(model_dir)
        # self._det = loaded.signatures['serving_default']
        self._det = saved_model_tf

    @staticmethod
    def unpad_boxes(boxes, pad_params):
        """Recover the padded output effect."""
        img_h, img_w, img_pad_h, img_pad_w = pad_params
        reshaped_bb = np.reshape(boxes, [-1, 2, 2])
        recover_xy = reshaped_bb * [(img_pad_w + img_w) / img_w,
                                    (img_pad_h + img_h) / img_h]
        boxes = np.reshape(recover_xy, [4])

        return boxes

    @staticmethod
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

    def detect_and_cut(self, imgs_batch, rs_size=(150, 150)):
        """Detect faces, cut first one and return resulted image.

        Parameters
        ----------
        imgs_batch : array of numpy.ndarray
            Read BGR images.
        rs_size : tuple
            Resize images to this size before forward pass.

        Returns
        -------
        np.array
            Cropped face.

        """
        # prepare data
        rgb_rs_batch = [cv2.resize(x[:, :, ::-1], rs_size) for x in imgs_batch]
        pad_batch = [self.pad_img(x) for x in rgb_rs_batch]

        pad_params = [x[1] for x in pad_batch]
        imgs_pad = [x[0] for x in pad_batch]

        inp = tf.constant(imgs_pad, dtype=tf.float32)

        outputs = self._det(inp)

        boxes_batch = outputs['tf_op_layer_CombinedNonMaxSuppression'].numpy()
        # confs = outputs['tf_op_layer_CombinedNonMaxSuppression_1'].numpy()

        # postprocessing
        boxes_batch = np.squeeze(boxes_batch, axis=1)  # model returns 1 box
        boxes_batch = [self.unpad_boxes(x, y)
                       for x, y in zip(boxes_batch, pad_params)]

        cropped_imgs = []
        for img, box in zip(imgs_batch, boxes_batch):
            # relative x1, y1, x2, y2
            x1 = int(img.shape[0] * box[0])
            y1 = int(img.shape[1] * box[1])
            x2 = int(img.shape[0] * box[2])
            y2 = int(img.shape[1] * box[3])

            cropped_imgs.append(img[x1:x2, y1:y2])

        return cropped_imgs
