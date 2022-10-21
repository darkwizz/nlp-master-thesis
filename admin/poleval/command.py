import os
from utils import get_number_of_tokens_in_dataset

from utils.workflow import load_datasets


def main(args):
    print('Inside poleval subset main')
    data_base_dir = args.directory
    subset_dirs = {}
    for item in os.listdir(data_base_dir):
        key = item.lower().replace('-', '_').replace(' ', '_')
        path = os.path.join(data_base_dir, item)
        subset_dirs[key] = path
    data = load_datasets(**subset_dirs)
    if args.count_token:
        total_n_tokens = 0
        for subset in data:
            n_tokens = get_number_of_tokens_in_dataset(data[subset], ['question', 'answer'])
            total_n_tokens += n_tokens
        print(f'Number of tokens in {data_base_dir}: {total_n_tokens}')