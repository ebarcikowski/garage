"""Test model with CLI script."""

from garage import nn
from argparse import ArgumentParser
from tensorflow import keras
import tensorflow as tf
import numpy as np


CLASSES = [
    "closed",
    "open"
]


def load_image(path):
    img = keras.utils.load_img(
        path,
        color_mode="grayscale",
        target_size=(256, 256)
    )
    img_array = keras.utils.img_to_array(img)

    return img_array

def main():
    parser = ArgumentParser('perform garage door open training')
    parser.add_argument(
        '--model',
        type=str,
        help='Load SAVED_MODEL',
        metavar='SAVED_MODEL',
        default=None,
        required=True
    )
    parser.add_argument(
        'images',
        nargs='?',
        help='Images to process',
        type=str,
    )
    parser.add_argument('--steps', type=int, default=100)
    args = parser.parse_args()
    if not args.model:
        raise ValueError('Must specify a model')

    model = keras.models.load_model(args.model)

    if args.images:
        # if images are specified just index through them
        for image in args.images:
            img_array = load_image(image)
            img_array = tf.expand_dims(img_array, 0)
            predicts = model.predict(img_array, verbose=0)
            idx = np.argmax(predicts)
            print(f"{image}: {CLASSES[idx]}")
    else:
        # get an image from the webcam
        cap = nn.Capture()
        img_array = cap.get_image_model()
        predicts = model.predict(img_array, verbose=0)
        idx = np.argmax(predicts)
        print(f"Door status: {CLASSES[idx]}")


if __name__ == '__main__':
    main()
