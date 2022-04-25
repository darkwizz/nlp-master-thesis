from argparse import ArgumentParser
from importlib import import_module


def main(parsed_args):
    model_name = parsed_args.model_name
    revision = parsed_args.revision
    module_name = f'{model_name}.{revision}.train'
    module = import_module(module_name)
    module.main(parsed_args)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-n', '--model_name', choices=('plt5', 'papugapt2'), type=str.lower,
                        help='a model name to use', required=True)
    parser.add_argument('-r', '--revision', choices=('baseline'), type=str.lower, required=True, help="model revision")
    parser.add_argument('-b', '--base-data-path', required=True, help='base path to all data')
    parser.add_argument('-t', '--tokenizer-path', required=True)
    parser.add_argument('-m', '--model-path', required=True)
    parser.add_argument('-q', '--question-max-length', '--q-maxlen', default=50, type=int, help='max length for a tokenized question, during training')
    parser.add_argument('-a', '--answer-max-length', '--a-maxlen', default=10, type=int, help='max length for a tokenized answer, during training')
    parser.add_argument('-o', '--training-output-dir', default='./training-results', help='path to outputs during model training. Be careful, as the default path is one level up than models and their revisions, thus would be overwritten for every program run')
    parser.add_argument('--test-batch-size', default=50, type=int)
    parser.add_argument('--test-max-length', default=None, type=int, help='max length necessary to create a pytorch tensor during testing')
    parser.add_argument('--results-dir', default='data-results')
    
    args = parser.parse_args()
    main(args)