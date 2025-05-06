import tensorflow as tf
from tensorflow import keras
from keras import layers
import cv2
import numpy as np
CAMERA_URL = "http://zm:80/zm/cgi-bin/zms?monitor=1"
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


def get_training_data(path='data'):
    ds = tf.keras.preprocessing.image_dataset_from_directory(
            path,
            label_mode='binary',
            batch_size=1,
            image_size=(256, 256),
            color_mode="grayscale",
        )
    return ds

def get_model(model_dir=None, warm_start=False):
    """
    Return basic linear classifier
    """
    model = keras.models.Sequential([
        layers.Rescaling(1./255, input_shape=(256, 256, 1)),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(2)
    ])
    model.compile(
        optimizer='adam',
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    return model


class Capture:
    """
    Make link to the web camera API and return data for inference.
    """
    DEFAULT_PATH = "/zm/cgi-bin/zms?monitor=1"
    DEFAULT_DOMAIN = "http://localhost:8080"
    def __init__(self,
        url=None,
        domain=DEFAULT_DOMAIN,
        path=DEFAULT_PATH
    ):
        if url:
            self.url = url
        else:
            self.url = domain + path
        self.cap = cv2.VideoCapture()
        assert self.cap.open(self.url) is True

    def get_image(self):
        """Return color image from web api"""
        rc, img_color = self.cap.read()
        return img_color

    def get_image_model(self):
        """
        Create TF function for model inference

        NOTE. My camera is no longer up so this hasn't been tested since the
        migration to TF2.
        """
        img = self.get_image()
        img = cv2.resize(img, (256, 256), )
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img_array = keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)
        return img_array


if __name__ == '__main__':
    parser = ArgumentParser('perform garage door open training')
    parser.add_argument('--steps', type=int, default=100)
    args = parser.parse_args()

    model = get_linear_model(warm_start=True)
    # model.train(input_fn=train_input_fn('../data'), steps=args.steps)
    cap = Capture()
    print(list(model.predict(cap.make_pred_fn)))
    # export_estimator(model, 'saved_model')

