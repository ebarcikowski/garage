import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import numpy as np
CAMERA_URL = "http://localhost:8080/zm/cgi-bin/zms?monitor=1"
from argparse import ArgumentParser


def train_input_fn(path='data'):
    """
    Create and return TF function for model training

    Note: The way the data is setup:

     - 0 is closed
     - 1 is open
    :param path: path to open and closed data
    :return: function returning Dataset.
    """
    def fn():
        ds = tf.keras.preprocessing.image_dataset_from_directory(
            path,
            label_mode='binary',
            batch_size=1
        )
        ds = ds.map(lambda x, y: ({"x": x}, y))
        return ds.repeat()
    return fn


def get_linear_model(model_dir=None, warm_start=False):
    """
    Return basic linear classifier
    """
    fc = tf.feature_column.numeric_column("x", shape=(256, 256, 3))

    if not model_dir:
        model_dir = 'tensorflow_gd'

    if warm_start:
        return tf.estimator.LinearClassifier(
            feature_columns=[fc],
            n_classes=2,
            model_dir=model_dir,
            warm_start_from=model_dir
        )
    return tf.estimator.LinearClassifier(
        feature_columns=[fc],
        n_classes=2,
        model_dir=model_dir,
    )


def export_estimator(estimator, saved_model_path):
    serving_input_fn = tf.estimator.export.build_raw_serving_input_receiver_fn()
    #    tf.feature_column.numeric_column("x", shape=(256, 256, 3))
    # )
    estimator.export_saved_model(saved_model_path, serving_input_fn)


class Capture:
    """
    Make link to the web camera API and return data for inference.
    """
    def __init__(self, url=CAMERA_URL):
        self.url = CAMERA_URL
        self.cap = cv2.VideoCapture()
        assert self.cap.open(self.url) is True

    def get_image(self):
        """Return color image from web api"""
        rc, img_color = self.cap.read()
        return img_color

    def make_pred_fn(self):
        """
        Create TF function for model inference
        """
        img = self.get_image()
        img = cv2.resize(img, (256, 256))
        img = np.expand_dims(img, axis=0)
        ds = tf.data.Dataset.from_tensor_slices(img)
        ds = ds.map(lambda x: {"x": x})
        return ds.batch(1)


if __name__ == '__main__':
    parser = ArgumentParser('perform garage door open training')
    parser.add_argument('--steps', type=int, default=100)
    args = parser.parse_args()

    model = get_linear_model(warm_start=True)
    # model.train(input_fn=train_input_fn('../data'), steps=args.steps)
    cap = Capture()
    print(list(model.predict(cap.make_pred_fn)))
    # export_estimator(model, 'saved_model')

