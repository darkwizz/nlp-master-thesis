from genericpath import isdir
from importlib import import_module
from argparse import ArgumentError, ArgumentParser
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


def main(parsed_args, main_parser, provider_args):
    available_providers = list_command_providers()
    if parsed_args.list:
        for provider in available_providers:
            print(provider)
        exit(0)
    
    if parsed_args.source not in available_providers:
        raise ArgumentError('No such available dataset')
    provider_package = import_module(f'admin.{parsed_args.source}')
    subparsers = main_parser.add_subparsers()
    provider_parser: ArgumentParser = provider_package.prepare_arg_parser(subparsers)
    args = provider_parser.parse_args(provider_args)
    provider_package.command.main(args)


if __name__ == '__main__':
    parser = ArgumentParser(add_help=False)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-l', '--list', action='store_true', help='list all available data source command providers')
    group.add_argument('-s', '--source', help='name of the data source and the command package (e.g. poleval). Must be implemented under admin/ package and provide its own argument parser')
    args, rest_args = parser.parse_known_args()
    main(args, parser, rest_args)
