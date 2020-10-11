from garage import nn
from argparse import ArgumentParser
import tensorflow as tf


if __name__ == '__main__':
    parser = ArgumentParser('perform garage door open training')
    parser.add_argument('--data_dir', help='Data location', default='../data')
    parser.add_argument('--model_dir', help='Location to save model')
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--saved_model', help='Save model to SAVED_MODEL',
                        metavar='SAVED_MODEL', type=str, default=None)
    args = parser.parse_args()

    model = nn.get_linear_model()
    model.train(input_fn=nn.train_input_fn(args.data_dir), steps=args.steps)

    if args.saved_model:
        tf.saved_model.save(model, args.saved_model)
