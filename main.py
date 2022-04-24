from argparse import ArgumentParser
from importlib import import_module


def main(parsed_args):
    model = parsed_args.model
    revision = parsed_args.revision
    module_name = f'{revision}.{model}.train'
    module = import_module(module_name)
    module.main(parsed_args)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-m', '--model', choices=('plt5', 'papugapt2'), type=str.lower,
                        help='a model to use', required=True)
    parser.add_argument('-r', '--revision', choices=('baseline'), type=str.lower, required=True, help="model revision")
    parser.add_argument('-b', '--base-path', required=True, help='base path to all data')
    
    args = parser.parse_args()
    main(args)