from argparse import _MutuallyExclusiveGroup, _SubParsersAction

from utils.data_preprocess import PROMPT_TARGETS
from . import command

__all__ = ['prepare_arg_parser', 'command']


def prepare_arg_parser(subparsers: _SubParsersAction):
    parser = subparsers.add_parser('-s mkqa_pl', aliases=['--source mkqa_pl'], help='admin commands for mkqa_pl subset')
    source_me_group: _MutuallyExclusiveGroup = parser.add_mutually_exclusive_group(required=True)
    source_me_group.add_argument('-d', '--directory', help='source directory with the Polish MKQA subset')
    source_me_group.add_argument('-D', '--download', action='store_true', help='download the Polish MKQA subset. If passed --target_path, it will be downloaded there. In case of --count_token, the dataset will not be saved anywhere')
    count_me_group: _MutuallyExclusiveGroup = parser.add_mutually_exclusive_group(required=True)
    count_me_group.add_argument('-c', '--count_token', action='store_true', help='count tokens in the dataset. Ignores any other preprocessing flag')
    count_me_group.add_argument('-t', '--target_path', help='destination path of the processed dataset. It will be saved in the PolEval format')
    process_me_group: _MutuallyExclusiveGroup = parser.add_mutually_exclusive_group()
    process_me_group.add_argument('-p', '--prompts', action='store_true', help='append prompts to all questions and answers from the same pool')
    process_me_group.add_argument('-g', '--gprompts', action='store_true', help='split questions on groups and each group has its own pool of prompts')
    process_me_group.add_argument('-a', '--artificial', action='store_true', help='enclose questions and answers by an artificial suffix and a prefix')
    parser.add_argument('-n', '--nduplicates', type=int, default=0, help='number of possible duplicates of every data item. 0 means only the original item')
    parser.add_argument('--seed', type=int, default=6325, help='randomizing seed')
    parser.add_argument('-o', '--original', action='store_true', help='include the original question and answer when downloading. By default they are not')
    parser.add_argument('--split', action='store_true', help='perform split on train/dev/test as 70/15/15')
    parser.add_argument('-e', '--engine', dest='tokenizer', default='spacy', help='used with -c flag. Tokenizer which is used to split dataset texts on tokens. Default is spaCy (case insensitive), but also can be passed a path to a Transformers tokenizer')
    parser.add_argument('-w', '--which-to-prompt', choices=PROMPT_TARGETS, default=PROMPT_TARGETS[0], dest='prompt_target', help='used with -p flag. Specifies whether both questions and answers to prompt or one of them')
    return parser