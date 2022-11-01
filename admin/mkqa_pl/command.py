import os
from utils.mkqa_pl import download_pl_subset, save_dataset
from utils.workflow import load_datasets


def print_number_of_tokens(data):
    from utils import get_number_of_tokens_in_dataset

    total_n_tokens = 0
    for subset in data:
        n_tokens = get_number_of_tokens_in_dataset(data[subset], ['question', 'answer'])
        total_n_tokens += n_tokens
    print(f'Number of tokens in the downloaded MKQA dataset: {total_n_tokens}')


def main(args):
    from utils import compose_subsets_paths

    print('Inside mkqa_pl subset main')
    seed = args.seed
    target_dir = args.target_path
    if args.download:
        data = download_pl_subset(seed=seed)
        if args.count_token:
            print_number_of_tokens(data)
        else:
            print('Saving MKQA as CSV...')
            for subset in data:
                save_dataset(data[subset], os.path.join(target_dir, subset), keep_originals=args.original)
        return
    data_base_dir = args.directory
    subset_dirs = compose_subsets_paths(data_base_dir)
    data = load_datasets(seed=seed, **subset_dirs)
    if args.count_token:
        print_number_of_tokens(data)
    