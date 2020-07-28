# -*- coding: utf-8 -*
"""Unit tests."""
import sys
import numpy as np
from scripts.face_net import FaceNet


def test_face_net():
    """Test FaceNet model."""
    net = FaceNet(sys.path[0] + '/data/facenet.pb')
    img = np.ones((1, 160, 160, 3), dtype=np.uint8)
    emb = net.calc_embeddings(img)[0]
    assert len(emb) == 128, \
        'Wrong number of embedding features. Possibly wrong model. '
