import os
from admin import ANSWER_FEATURE, QUESTION_FEATURE

from utils.data_preprocess import get_artificially_augmented_dataset, get_prompt_augmented_dataset, get_special_prompt_augmented_dataset, PL_QUESTION_PROMPTS


def load_mkqa_subset(base_dir):
    from datasets import Dataset

    in_path = os.path.join(base_dir, 'in.tsv')
    expected_path = os.path.join(base_dir, 'expected.tsv')

    result = []
    in_file = open(in_path)
    expected_file = open(expected_path)
    for in_line, expected_line in zip(in_file, expected_file):
        formatted_in = in_line.strip()
        formatted_expected = expected_line.strip()
        if all((formatted_in, formatted_expected)):
            question, answer_type = formatted_in.split('\t')

            answer_parts = formatted_expected.split('\t')
            answer = answer_parts[0]
            alternatives = answer_parts[1:] if len(answer_parts) > 1 else []

            result.append({'question': question, 'answer': answer, 'alternatives': alternatives, 'answer_type': answer_type})
    return Dataset.from_list(result)


def print_number_of_tokens(data, tokenizer):
    if not tokenizer:
        print('A tokenizer must be specified')
        exit(1)

    from utils import get_total_number_of_tokens_in_datasets

    total_n_tokens = get_total_number_of_tokens_in_datasets(data, [QUESTION_FEATURE, ANSWER_FEATURE], tokenizer)
    print(f'Number of tokens in the downloaded MKQA dataset: {total_n_tokens}')


def get_prompts_for_question(item):
    from utils.mkqa_pl.prompting import MKQA_QUESTION_PROMPTS

    answer_type = item.get('answer_type', 'unknown')
    return MKQA_QUESTION_PROMPTS.get(answer_type, PL_QUESTION_PROMPTS)


def main(args):
    from utils import compose_subsets_paths
    from utils.mkqa_pl import download_pl_subset, save_dataset
    from utils.workflow import save_data
    from datasets import DatasetDict

    print('Inside mkqa_pl subset main')
    seed = args.seed
    target_dir = args.target_path
    if args.download:
        data = download_pl_subset(seed=seed, split=args.split)
    else:
        data_base_dir = args.directory
        subset_dirs = compose_subsets_paths(data_base_dir)
        subsets_dict = {}
        for subset, path in subset_dirs.items():
            subset_data = load_mkqa_subset(path)
            subsets_dict[subset] = subset_data
        data = DatasetDict(subsets_dict)
    
    if args.count_token:
        print_number_of_tokens(data, args.tokenizer)
        exit(0)
    
    if args.artificial:
        data = get_artificially_augmented_dataset(data, QUESTION_FEATURE, ANSWER_FEATURE)
    elif args.prompts:
        data = get_prompt_augmented_dataset(data, QUESTION_FEATURE, ANSWER_FEATURE, prompt_target=args.prompt_target, seed=args.seed)
    elif args.gprompts:
        data = get_special_prompt_augmented_dataset(data, get_prompts_for_question, QUESTION_FEATURE, ANSWER_FEATURE, prompt_target=args.prompt_target, seed=args.seed)
    
    print('Saving MKQA as CSV...')
    if args.download:
        for subset in data:
            save_dataset(data[subset], os.path.join(target_dir, subset), keep_originals=args.original)
    else:
        save_data(data, target_dir)
    