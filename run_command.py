from importlib import import_module
from argparse import ArgumentParser

from admin import get_dataset_stats, list_command_providers, perform_artificial_augmentation, \
    perform_merge, print_subsets_stats, perform_prompt_augmentation
from utils.data_preprocess import PROMPT_TARGETS


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
        if not all([parsed_args.tokenizer, parsed_args.source_directory]):
            print('For token stats a dataset directory path and a tokenizer must be passed')
            exit(1)
        
        subsets_stats = get_dataset_stats(parsed_args.tokenizer, parsed_args.source_directory)
        print_subsets_stats(subsets_stats)
        exit(0)
    
    if any([parsed_args.art_augment_data_path, parsed_args.prompt_augment_data_path]):
        if not parsed_args.augmentation_result:
            print('Target path for augmented data must be provided')
            exit(1)
        
        if parsed_args.art_augment_data_path:
            perform_artificial_augmentation(parsed_args.art_augment_data_path, parsed_args.augmentation_result)
        elif parsed_args.prompt_augment_data_path:
            perform_prompt_augmentation(parsed_args.prompt_augment_data_path, parsed_args.augmentation_result, parsed_args.prompt_target, parsed_args.prompt_seed)
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
    me_group.add_argument('-A', '--augment-artificial', dest='art_augment_data_path', help='path to subsets which add an artificial prefix and suffix for questions and answers for')
    me_group.add_argument('-P', '--augment-prompt', dest='prompt_augment_data_path', help='path to subsets which should be prepended with natural prompts from the specified prompts pool. Can be seeded')
    parser.add_argument('-R', '--augmentation_result', help='used with [-A | -P] parameters. Target path for the augmented data')
    parser.add_argument('-M', '--merge_result', help='target path with the result of merge')
    parser.add_argument('--prompt_seed', type=int, default=2650, help='used with -P parameter. Seeding value for sampling prompts from a pool. Default is 2650')
    parser.add_argument('-O', '--prompt_target', choices=PROMPT_TARGETS, default=PROMPT_TARGETS[0], help='used with -P parameter. Specifies whether add a prompt to both questions and answers or to one of them')
    group = parser.add_argument_group(title='Token Stats', description='Settings for counting longest and shortest questions and answers in a dataset')
    group.add_argument('-E', dest='tokenizer', default='spacy', help='which tokenizing engine to use (from spaCy or pass a path to a Transformers tokenizer)')
    group.add_argument('-S', '--source_directory', help='directory with the subsets of a dataset to calculate token stats. The subsets must be grouped and stored in PolEval format')
    args, rest_args = parser.parse_known_args()
    main(args, parser, rest_args, parser.print_help)
