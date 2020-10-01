import cv2
import numpy as np
import os

def loadimage(filename):
    if not os.path.exists(filename):
        raise FileNotFoundError()
    img = cv2.imread(filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # img = img.transpose((2, 0, 1))
    return img


def loadvideo(filename):
    if not os.path.exists(filename):
        raise FileNotFoundError()
    capture = cv2.VideoCapture(filename)

    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # empty numpy array of appropriate length, fill in when possible from front
    v = np.zeros((frame_count, frame_width, frame_height, 3), np.float32)

    for count in range(frame_count):
        ret, frame = capture.read()
        if not ret:
            raise ValueError("Failed to load frame #{} of {}.".format(count, filename))

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        v[count] = frame

    v = v.transpose((3, 0, 1, 2))

    return v

def preprocess_input(img, size, mean, std):
    img = cv2.resize(img,(size,size))/255.
    img = (img - mean) / std
    img = img.transpose((2, 0, 1))
    return img