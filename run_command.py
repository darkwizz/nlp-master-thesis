from importlib import import_module
from argparse import ArgumentParser
from tqdm import tqdm
import os


COMMAND_FILE = 'command.py'


def list_command_providers():
    admin_path = os.path.join(os.path.dirname(__file__), 'admin')
    result = []
    for admin_subitem in os.listdir(admin_path):
        full_subpath = os.path.join(admin_path, admin_subitem)
        if os.path.isdir(full_subpath) and COMMAND_FILE in os.listdir(full_subpath):
            result.append(admin_subitem)
    return result


def perform_merge(source_path, results_path):
    from utils.workflow import load_data_for_split
    
    subsets = ['dev', 'train', 'test']
    for subset in subsets:
        os.makedirs(os.path.join(results_path, subset), exist_ok=True)

    expected_files = {subset: open(os.path.join(results_path, subset, 'expected.tsv'), 'a') for subset in subsets}
    in_files = {subset: open(os.path.join(results_path, subset, 'in.tsv'), 'a') for subset in subsets}
    for dataset_dir in os.listdir(source_path):
        print(f'Processing dataset: {dataset_dir}...')
        for subset in subsets:
            dataset_path = os.path.join(source_path, dataset_dir, subset)
            if not os.path.exists(dataset_path):
                continue

            dataset = load_data_for_split(os.path.join(dataset_path, 'in.tsv'), 'question', os.path.join(dataset_path, 'expected.tsv'), 'answer')
            for item in tqdm(dataset):
                question_line = item["question"] + '\n'
                answer_line = "\t".join([item["answer"]] + item.get("alternatives")) + '\n'
                expected_files[subset].write(answer_line)
                in_files[subset].write(question_line)
    for exp_file, in_file in zip(expected_files.values(), in_files.values()):
        exp_file.close()
        in_file.close()


def get_dataset_stats(engine, dataset_path):
    from utils.workflow import load_data_for_split

    if engine.lower() == 'spacy':
        import spacy
        pl_nlp = spacy.load('pl_core_news_lg')
    result = {}
    return result


def print_subsets_stats(subsets_stats):
    print('#' * 50)
    print('Dataset stats')
    print('#' * 50)
    for subset in subsets_stats:
        print(f'Subset: {subset}')
        question_stats = subsets_stats[subset]['question']
        answer_stats = subsets_stats[subset]['answer']
        print(f'Questions => min: {question_stats["min"]}, max: {question_stats["max"]}')
        print(f'Answers => min: {answer_stats["min"]}, max: {answer_stats["max"]}')


def main(parsed_args, main_parser, provider_args, help_func=lambda: print('Help message')):
    available_providers = list_command_providers()
    if parsed_args.custom_help:
        help_func()
        exit(0)

    if parsed_args.list:
        for provider in available_providers:
            print(provider)
        exit(0)
    
    if parsed_args.merge_path:
        if not parsed_args.merge_result:
            print('Both source and target paths must be provided for merging datasets')
            exit(1)
        perform_merge(parsed_args.merge_path, parsed_args.merge_result)
        exit(0)
    
    if parsed_args.token_stats:
        if not all([parsed_args.engine, parsed_args.directory]):
            print('For token stats a dataset directory path and an engine must be passed')
            exit(1)
        
        subsets_stats = get_dataset_stats(parsed_args.engine, parsed_args.directory)
        print_subsets_stats(subsets_stats)
        exit(0)

    if parsed_args.source not in available_providers:
        raise ValueError('No such available dataset')
    provider_package = import_module(f'admin.{parsed_args.source}')
    subparsers = main_parser.add_subparsers()
    provider_parser: ArgumentParser = provider_package.prepare_arg_parser(subparsers)
    args = provider_parser.parse_args(provider_args)
    provider_package.command.main(args)


if __name__ == '__main__':
    parser = ArgumentParser(add_help=False)
    me_group = parser.add_mutually_exclusive_group(required=True)
    me_group.add_argument('-l', '--list', action='store_true', help='list all available data source command providers')
    me_group.add_argument('-H', '--custom_help', action='store_true', help='show this message and exit')
    me_group.add_argument('-m', '--merge_path', help='base directory with all datasets to merge them into one')
    me_group.add_argument('-s', '--source', help='name of the data source and the command package (e.g. poleval). Must be implemented under admin/ package and provide its own argument parser')
    me_group.add_argument('-T', '--token_stats', action='store_true', help='show the shortest and longest question/answer lengths')
    parser.add_argument('-M', '--merge_result', help='target path with the result of merge')
    group = parser.add_argument_group(title='Token Stats', description='Settings for counting longest and shortest questions and answers in a dataset')
    group.add_argument('-e', '--engine', default='spacy', help='which tokenizing engine to use (from spaCy or pass a path to a Transformers tokenizer)')
    group.add_argument('-d', '--directory', help='directory with the subsets of a dataset. The subsets must be grouped and stored in PolEval format')
    args, rest_args = parser.parse_known_args()
    main(args, parser, rest_args, parser.print_help)
