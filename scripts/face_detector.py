# -*- coding: utf-8 -*
"""Class for face detection using OpenCV."""
import cv2
import numpy as np
from pathlib import Path


class FaceDetector:
    """Class for face detection using OpenCV."""

    def __init__(self, scale_factor=1.1, min_neighbors=3,
                 model_name='haarcascade_frontalface_alt.xml'):
        """Class constructor."""
        self._scale_factor = scale_factor
        self._min_neighbors = min_neighbors

        model_path = str(Path(cv2.__file__).parent.joinpath(
            'data/' + model_name)
        )
        self._cascade = cv2.CascadeClassifier()
        if not self._cascade.load(model_path):
            raise RuntimeError(
                'can\'t load {} model. Try default model.'.format(model_name)
            )

    def detect_and_cut(self, img, equalize=True):
        """Detect faces, cut first one and return resulted image.

        Parameters
        ----------
        img : np.array
            Read BGR image.
        equalize : bool
            Do cv2.equalizeHist() before detection.

        Returns
        -------
        np.array
            Cropped face.

        """
        cv2.namedWindow('img', cv2.WINDOW_NORMAL)

        g_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if equalize:
            cv2.imshow('img', g_img)
            cv2.waitKey()
            g_img = cv2.equalizeHist(g_img)

        faces = self._cascade.detectMultiScale(
            image=g_img,
            scaleFactor=self._scale_factor,
            minNeighbors=self._min_neighbors
        )

        if len(faces) > 1:
            print('WARNING: multiple faces detected.')

        print(faces)
        face = self._postproc_faces(img, faces)

        print(img.shape)
        cv2.imshow('img', img)
        cv2.waitKey()
        cv2.imshow('img', g_img)
        cv2.waitKey()
        res = img[face[0]:face[2], face[1]:face[3]]
        cv2.imshow('img', res)
        cv2.waitKey()

        return res

    @staticmethod
    def _postproc_faces(img, faces):
        """Found faces postprocessing.

        Parameters
        ----------
        img : np.array
            Image where faces have been found.
        faces : array-like
            Array with found faces using cv2.CascadeClassifier.

        Returns
        -------
        List
            Biggest fitted to image bounding box (x_min, y_min, x_max, y_max)
            or empty list.

        """
        if len(faces) == 0:
            return faces

        spaces = []
        for face in faces:
            # convert bb format
            face[2], face[3] = face[0] + face[2], face[1] + face[3]

            # fit to image
            face = np.abs(face)
            face[2] = max([face[2], img.shape[0]])
            face[3] = max([face[3], img.shape[1]])

            # calc bb space
            spaces.append((face[2] - face[0]) * (face[3] - face[1]))

        # return biggest bb
        return faces[np.argmax(spaces)]
