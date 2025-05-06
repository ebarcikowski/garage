"""Train the model using stored data and save it the result."""


from garage import nn
from argparse import ArgumentParser


def main():
    parser = ArgumentParser('perform garage door open training')
    parser.add_argument('--data_dir', help='Data location', default='data')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--save_model', help='Save model to SAVED_MODEL',
                        metavar='SAVED_MODEL', type=str, default=None)
    args = parser.parse_args()

    model = nn.get_model()
    ds = nn.get_training_data(args.data_dir)
    model.fit(ds, epochs=args.epochs)

    if args.save_model:
        model.save(args.save_model)

if __name__ == '__main__':
    main()
