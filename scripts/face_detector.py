# -*- coding: utf-8 -*
"""Class for face detection using OpenCV."""
import cv2
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

    def detect(self, img, equalize=True):
        """Detect faces on the image.

        Parameters
        ----------
        img : np.array
            Read BGR image.
        equalize : bool
            Do cv2.equalizeHist() before detection.

        Returns
        -------
        List
            Bounding boxes in format [x, y, h, w] for each found face.

        """
        g_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if equalize:
            g_img = cv2.equalizeHist(g_img)

        faces = self._cascade.detectMultiScale(
            image=g_img,
            scaleFactor=self._scale_factor,
            minNeighbors=self._min_neighbors
        )

        return faces
