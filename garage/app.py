#!/usr/env python3
"""
Web endpoint get garage door status. This was the original purpose
of this project.

This hasn't been tested in sometime since my cameras are all offline.
"""
from garage import nn
import flask
from tensorflow import keras
import numpy as np


MODEL_PATH = './saved.keras'
model = keras.models.load_model(MODEL_PATH)
app = flask.Flask(__name__)

cap = nn.Capture()


@app.route('/', methods=["GET", "POST"])
def predict():
    data = {"closed": False}

    predicts = model.predict(cap.get_image_model())
    idx = np.argmax(predicts)
    if idx == 0:
        data["closed"] = True

    return flask.jsonify(data)

def main():
    app.run(host='0.0.0.0')

if __name__ == '__main__':
    main()
