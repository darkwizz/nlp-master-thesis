from argparse import _MutuallyExclusiveGroup, _SubParsersAction

from utils.data_preprocess import PROMPT_TARGETS
from . import command

__all__ = ['prepare_arg_parser', 'command']


def prepare_arg_parser(subparsers: _SubParsersAction):
    parser = subparsers.add_parser('-s poquad', aliases=['--source poquad'], help='admin commands for poquad dataset')
    parser.add_argument('-d', '--directory', required=True, help='source directory with all PoQuAD .json subsets')
    me_group: _MutuallyExclusiveGroup = parser.add_mutually_exclusive_group(required=True)
    me_group.add_argument('-t', '--target_path', help='destination path of the processed dataset. Saved in the PolEval format')
    me_group.add_argument('-c', '--count_token', action='store_true', help='count tokens in the dataset. Ignores any other preprocessing flag')
    process_me_group: _MutuallyExclusiveGroup = parser.add_mutually_exclusive_group()
    process_me_group.add_argument('-p', '--prompts', action='store_true', help='append prompts to all questions and answers from the same pool')
    process_me_group.add_argument('-g', '--gprompts', action='store_true', help='split questions on groups and each group has its own pool of prompts')
    process_me_group.add_argument('-a', '--artificial', action='store_true', help='enclose questions and answers by an artificial suffix and a prefix')
    parser.add_argument('-i', '--impossible', action='store_true', help='include tokens with is_impossible in "true"')
    parser.add_argument('-e', '--engine', dest='tokenizer', default='spacy', help='used with -c flag. Tokenizer which is used to split dataset texts on tokens. Default is spaCy (case insensitive), but also can be passed a path to a Transformers tokenizer')
    parser.add_argument('--seed', type=int, default=98376, help='randomizing seed. Used during every modification. Default is 98376')
    parser.add_argument('-w', '--which-to-prompt', choices=PROMPT_TARGETS, default=PROMPT_TARGETS[0], dest='prompt_target', help='used with -p flag. Specifies whether both questions and answers to prompt or one of them')
    return parser