from garage import nn
from argparse import ArgumentParser


if __name__ == '__main__':
    parser = ArgumentParser('perform garage door open training')
    # parser.add_argument('--model_dir', help='Location to save model')
    parser.add_argument('--steps', type=int, default=100)
    args = parser.parse_args()

    cap = nn.Capture()
    model = nn.get_linear_model(warm_start=True)
    gen = model.predict(cap.make_pred_fn)
    result = list(gen)[0]
    if result['class_ids'] == 0:
        print("Closed")
    else:
        print("Open")
