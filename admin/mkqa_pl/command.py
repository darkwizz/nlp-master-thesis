import os
from admin import ANSWER_FEATURE, QUESTION_FEATURE

from utils.data_preprocess import get_artificially_augmented_dataset, get_prompt_augmented_dataset


def print_number_of_tokens(data, tokenizer):
    if not tokenizer:
        print('A tokenizer must be specified')
        exit(1)

    from utils import get_total_number_of_tokens_in_datasets

    total_n_tokens = get_total_number_of_tokens_in_datasets(data, [QUESTION_FEATURE, ANSWER_FEATURE], tokenizer)
    print(f'Number of tokens in the downloaded MKQA dataset: {total_n_tokens}')


def main(args):
    from utils import compose_subsets_paths
    from utils.mkqa_pl import download_pl_subset, save_dataset
    from utils.workflow import load_datasets, save_data

    print('Inside mkqa_pl subset main')
    seed = args.seed
    target_dir = args.target_path
    if args.download:
        data = download_pl_subset(seed=seed, split=args.split)
    else:
        data_base_dir = args.directory
        subset_dirs = compose_subsets_paths(data_base_dir)
        data = load_datasets(seed=seed, **subset_dirs)
    
    if args.count_token:
        print_number_of_tokens(data, args.tokenizer)
        exit(0)
    
    if args.artificial:
        data = get_artificially_augmented_dataset(data, QUESTION_FEATURE, ANSWER_FEATURE)
    elif args.prompts:
        data = get_prompt_augmented_dataset(data, QUESTION_FEATURE, ANSWER_FEATURE, prompt_target=args.prompt_target, seed=args.seed)
    
    print('Saving MKQA as CSV...')
    if args.download:
        for subset in data:
            save_dataset(data[subset], os.path.join(target_dir, subset), keep_originals=args.original)
    else:
        save_data(data, target_dir)
    