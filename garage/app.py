#!/usr/env python3
from garage import nn
import flask

app = flask.Flask(__name__)

model = nn.get_linear_model(warm_start=True)
cap = nn.Capture()


@app.route('/', methods=["GET", "POST"])
def predict():
    data = {"closed": False}

    gen = model.predict(cap.make_pred_fn)
    result = list(gen)[0]

    if result['class_ids'] == 0:
        data['closed'] = True
    return flask.jsonify(data)


if __name__ == '__main__':
    app.run(host='0.0.0.0')