from argparse import ArgumentParser
from importlib import import_module
import os
from time import time
from datetime import timedelta


def write_elapsed_time(elapsed_time, parsed_args):
    finish_time_str = f'Elapsed time: {timedelta(seconds=elapsed_time)}'
    file_path = os.path.join(parsed_args.results_dir, 'elapsed.log')
    with open(file_path, 'w') as elapsed_file:
        elapsed_file.write(finish_time_str)
    print(finish_time_str)


def main(parsed_args):
    model_name = parsed_args.model_name
    revision = parsed_args.revision
    module_name = f'{model_name}.{revision}.train'
    module = import_module(module_name)
    start = time()
    module.main(parsed_args)
    end = time()
    write_elapsed_time(end - start, parsed_args)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-n', '--model_name', choices=('plt5', 'papugapt2'), type=str.lower,
                        help='a model name to use', required=True)
    parser.add_argument('-r', '--revision', choices=('baseline',), type=str.lower, required=True, help="model revision")
    parser.add_argument('-b', '--base-data-path', required=True, help='base path to all data')
    parser.add_argument('-t', '--tokenizer-path', required=True)
    parser.add_argument('-m', '--model-path', required=True)
    parser.add_argument('-q', '--question-max-length', '--q-maxlen', default=50, type=int, help='max length for a tokenized question, during training')
    parser.add_argument('-a', '--answer-max-length', '--a-maxlen', default=10, type=int, help='max length for a tokenized answer, during training')
    parser.add_argument('-o', '--training-output-dir', default='./training-results', help='path to outputs during model training. Be careful, as the default path is one level up than models and their revisions, thus would be overwritten for every program run')
    parser.add_argument('--test-batch-size', default=50, type=int)
    parser.add_argument('--test-max-length', default=None, type=int, help='max length necessary to create a pytorch tensor during testing')
    parser.add_argument('--results-dir', default='data-results')
    parser.add_argument('-s', '--save-pretrained', action='store_true')
    parser.add_argument('-S', '--skip-training', action='store_true')
    parser.add_argument('--seed', type=int, default=32351, help='randomizing seed for training. Default is 32351')
    parser.add_argument('-M', '--model-save-path', default=None, help='custom path to the saved trained model. If not passed, {model-name}/{revision}/trained-model is used instead')
    # parser.add_argument('-f', '--few-shot', action='store_true', help='for decoder-based models performs testing in a few-shot way')
    
    args = parser.parse_args()
    main(args)