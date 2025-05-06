"""Non-tensorflow models that I made along the way.

This is mostly play and hasn't been tested in a long time. These
models should work though.
"""
import cv2
import numpy as np
import sklearn.svm


CAMERA_URL = "http://localhost:8080/zm/cgi-bin/zms?monitor=1"


UPPER_LEFT = (900, 40)
LOWER_RIGHT = (1200, 380)
SLICE_X = slice(UPPER_LEFT[0], LOWER_RIGHT[0])
SLICE_Y = slice(UPPER_LEFT[1], LOWER_RIGHT[1])


class SVM:
    """SVM based classifier"""
    CLOSED_IMGS = [
        "data/closed/closed_001.jpg",
        "data/closed/closed_002.jpg",
        "data/closed/closed_003.jpg"
    ]
    OPEN_IMGS = [
        "data/open/open_001.jpg"
        "data/open/open_002.jpg"
    ]
    UPPER_LEFT = (900, 40)
    LOWER_RIGHT = (1200, 380)
    slice_x = SLICE_X
    slice_y = SLICE_Y

    def __init__(self):
        self.x_open = self.load_data(self.OPEN_IMGS)
        self.x_closed = self.load_data(self.CLOSED_IMGS)
        self.svm = sklearn.svm.SVC()
        self._fit()

    def _fit(self):
        X = np.concatenate((self.x_open, self.x_closed))
        y = np.zeros((X.shape[0]),)
        y[:self.x_open.shape[0]] = 1

        self.svm.fit(X, y)

    def predict(self, img):
        img = img[self.slice_y, self.slice_x]
        img = img.flatten()
        img = np.expand_dims(img, axis=0)
        return self.svm.predict(img)[0]

    @staticmethod
    def load_data(imgs):
        img_data = []
        for img_fn in imgs:
            img = cv2.imread(img_fn, cv2.IMREAD_GRAYSCALE)
            img = img[SLICE_Y, SLICE_X]
            img = img.flatten()
            img_data.append(img)

        return np.array(img_data)

class _Detector:
    """
    A OpenCV based detector

    May work but hasn't been tested in a long time.
    """
    CLOSED_IMG = "closed3.jpg"
    OPEN_IMG = "open.jpg"
    UPPER_LEFT = (900, 40)
    LOWER_RIGHT = (1200, 380)

    THRESH = .2

    def __init__(self, ref_image_fn=CLOSED_IMG):

        self.slice_x = slice(self.UPPER_LEFT[0], self.LOWER_RIGHT[0])
        self.slice_y = slice(self.UPPER_LEFT[1], self.LOWER_RIGHT[1])

        self.ref_image = None
        self._create_ref_image(ref_image_fn)

    def _create_ref_image(self, ref_image_fn):
        img = cv2.imread(ref_image_fn, 0)
        # self.ref_image = img
        self.ref_image = img[self.slice_y, self.slice_x]
        # self.ref_image = cv2.normalize(self.ref_image)

    def absdiff(self, img):

        img = img[self.slice_y, self.slice_x]
        return cv2.absdiff(img, self.ref_image)

    def is_closed(self, img):
        img = img[self.slice_y, self.slice_x]
        norm = cv2.norm(self.ref_image,
                        img,
                        cv2.NORM_L2SQR | cv2.NORM_RELATIVE)

        return norm < self.THRESH

    def matcher(self, img):

        img = img[self.slice_y, self.slice_x]
        # orb = cv2.SIFT()
        orb = cv2.ORB_create()
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        ref_kps, ref_desc = orb.detectAndCompute(self.ref_image, None)
        kps, desc = orb.detectAndCompute(img, None)

        matches = bf.match(ref_desc, desc)
        out = np.array([])
        matches = sorted(matches, key=lambda x: x.distance)
        print([m.distance for m in matches[:5]])
        img_comp = cv2.drawMatches(self.ref_image, ref_kps,
                                   img, kps, matches[:5], out)
        # import matplotlib.pyplot as plt
        # plt.imshow(img_comp)
        # plt.show()

        return matches
