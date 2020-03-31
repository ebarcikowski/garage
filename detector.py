import cv2
import numpy as np


CAMERA_URL = "http://localhost:8080/zm/cgi-bin/zms?monitor=1"


class Detector:
    CLOSED_IMG="closed.jpg"
    OPEN_IMG="open.jpg"
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
                        cv2.NORM_L2SQR | cv2.NORM_RELATIVE
        )
        return norm < self.THRESH


def noise_sample(img_fn):
    img = cv2.imread(img_fn, 0)
    detector = Detector()
    diff_img = detector.is_closed(img)
    print('{}'.format(np.sum(diff_img)))
    cv2.imshow("", diff_img)
    cv2.waitKey(0)


class Capture:

    def __init__(self, url=CAMERA_URL):
        self.url = CAMERA_URL
        self.cap = cv2.VideoCapture()
        assert self.cap.open(self.url) == True

    def get_image(self):
        rc, img_color = self.cap.read()
        img = cv2.cvtColor(img_color, cv2.COLOR_RGB2GRAY)
        return img


if __name__ == '__main__':
    cap = Capture()
    detector = Detector()

    img = cap.get_image()
    is_closed = detector.is_closed(img)
    print(is_closed)

    if not is_closed:
        raise Exception("Garage is open")
