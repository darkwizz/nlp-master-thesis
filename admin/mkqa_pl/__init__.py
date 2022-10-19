from argparse import _SubParsersAction
from . import command

__all__ = ['prepare_arg_parser', 'command']


def prepare_arg_parser(subparsers: _SubParsersAction):
    parser = subparsers.add_parser('-s mkqa_pl', aliases=['--source mkqa_pl'], help='admin commands for mkqa_pl subset')
    parser.add_argument('-n', '--nduplicates', type=int, default=0, help='number of possible duplicates of every data item. 0 means only the original item')
    parser.add_argument('-t', '--target_path', required=True, help='destination path of the processed dataset')
    return parser