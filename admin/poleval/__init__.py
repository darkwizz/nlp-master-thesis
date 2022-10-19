from argparse import _MutuallyExclusiveGroup, _SubParsersAction


def prepare_arg_parser(subparsers: _SubParsersAction):
    parser = subparsers.add_parser('-s poleval', aliases=['--source poleval'], help='admin commands for poleval dataset')
    me_group: _MutuallyExclusiveGroup = parser.add_mutually_exclusive_group()
    me_group.add_argument('-p', '--prompts', action='store_true', help='append prompts to all questions and answers from the same pool')
    me_group.add_argument('-g', '--gprompts', action='store_true', help='split questions on groups and each group has its own pool of prompts')
    me_group.add_argument('-a', '--artificial', action='store_true', help='enclose questions and answers by an artificial suffix and a prefix')
    parser.add_argument('-n', '--nduplicates', type=int, default=0, help='number of possible duplicates of every data item. 0 means only the original item')
    parser.add_argument('-d', '--directory', required=True, help='source directory with all available PolEval subsets')
    parser.add_argument('-t', '--target_path', required=True, help='destination path of the processed dataset')
    return parser