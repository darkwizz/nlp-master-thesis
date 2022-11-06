import os


def main(args):
    from utils import compose_subsets_paths, get_total_number_of_tokens_in_datasets
    from utils.workflow import load_datasets
    
    print('Inside poleval subset main')
    data_base_dir = args.directory
    subset_dirs = compose_subsets_paths(data_base_dir)
    data = load_datasets(**subset_dirs)
    if args.count_token:
        if not args.engine:
            print('A tokenizer must be specified')
            exit(1)
        total_n_tokens = get_total_number_of_tokens_in_datasets(data, ['question', 'answer'], args.engine)
        print(f'Number of tokens in {data_base_dir}: {total_n_tokens}')
        return