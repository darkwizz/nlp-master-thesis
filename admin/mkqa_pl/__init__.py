from argparse import _MutuallyExclusiveGroup, _SubParsersAction
from . import command

__all__ = ['prepare_arg_parser', 'command']


def prepare_arg_parser(subparsers: _SubParsersAction):
    parser = subparsers.add_parser('-s mkqa_pl', aliases=['--source mkqa_pl'], help='admin commands for mkqa_pl subset')
    source_me_group: _MutuallyExclusiveGroup = parser.add_mutually_exclusive_group(required=True)
    source_me_group.add_argument('-d', '--directory', help='source directory with the Polish MKQA subset')
    source_me_group.add_argument('-D', '--download', action='store_true', help='download the Polish MKQA subset. If passed --target_path, it will be downloaded there. In case of --count_token, the dataset will be placed under a temporary location and after counting, removed')
    count_me_group: _MutuallyExclusiveGroup = parser.add_mutually_exclusive_group(required=True)
    count_me_group.add_argument('-c', '--count_token', action='store_true', help='count tokens in the dataset. Ignores any other preprocessing flag')
    count_me_group.add_argument('-t', '--target_path', help='destination path of the processed dataset. It will be saved in the PolEval format')
    parser.add_argument('-n', '--nduplicates', type=int, default=0, help='number of possible duplicates of every data item. 0 means only the original item')
    parser.add_argument('-s', '--seed', default=6325, help='randomizing seed')
    return parser