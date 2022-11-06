import os


def print_number_of_tokens(data, tokenizer):
    if not tokenizer:
        print('A tokenizer must be specified')
        exit(1)

    from utils import get_total_number_of_tokens_in_datasets

    total_n_tokens = get_total_number_of_tokens_in_datasets(data, ['question', 'answer'], tokenizer)
    print(f'Number of tokens in the downloaded MKQA dataset: {total_n_tokens}')


def main(args):
    from utils import compose_subsets_paths
    from utils.mkqa_pl import download_pl_subset, save_dataset
    from utils.workflow import load_datasets

    print('Inside mkqa_pl subset main')
    seed = args.seed
    target_dir = args.target_path
    if args.download:
        data = download_pl_subset(seed=seed, split=args.split)
        if args.count_token:
            print_number_of_tokens(data, args.tokenizer)
        else:
            print('Saving MKQA as CSV...')
            for subset in data:
                save_dataset(data[subset], os.path.join(target_dir, subset), keep_originals=args.original)
        return
    data_base_dir = args.directory
    subset_dirs = compose_subsets_paths(data_base_dir)
    data = load_datasets(seed=seed, **subset_dirs)
    if args.count_token:
        print_number_of_tokens(data, args.tokenizer)
    