from argparse import _MutuallyExclusiveGroup, _SubParsersAction
from . import command

__all__ = ['prepare_arg_parser', 'command']


def prepare_arg_parser(subparsers: _SubParsersAction):
    parser = subparsers.add_parser('-s poquad', aliases=['--source poquad'], help='admin commands for poquad dataset')
    parser.add_argument('-d', '--directory', required=True, help='source directory with all PoQuAD .json subsets')
    me_group: _MutuallyExclusiveGroup = parser.add_mutually_exclusive_group(required=True)
    me_group.add_argument('-t', '--target_path', help='destination path of the processed dataset. Saved in the PolEval format')
    me_group.add_argument('-c', '--count_token', action='store_true', help='count tokens in the dataset. Ignores any other preprocessing flag')
    parser.add_argument('-i', '--impossible', action='store_true', help='include tokens with is_impossible in "true"')
    return parser