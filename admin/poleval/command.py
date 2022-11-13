
def main(args):
    from utils import compose_subsets_paths, get_total_number_of_tokens_in_datasets
    from utils.workflow import load_datasets, save_data
    from admin import ANSWER_FEATURE, QUESTION_FEATURE
    from utils.data_preprocess import get_artificially_augmented_dataset
    
    print('Inside poleval subset main')
    data_base_dir = args.directory
    subset_dirs = compose_subsets_paths(data_base_dir)
    data = load_datasets(**subset_dirs)
    if args.count_token:
        if not args.tokenizer:
            print('A tokenizer must be specified')
            exit(1)
        total_n_tokens = get_total_number_of_tokens_in_datasets(data, [QUESTION_FEATURE, ANSWER_FEATURE], args.tokenizer)
        print(f'Number of tokens in {data_base_dir}: {total_n_tokens}')
        exit()
    
    if args.artificial:
        data = get_artificially_augmented_dataset(data, QUESTION_FEATURE, ANSWER_FEATURE)
    
    target_path = args.target_path
    save_data(data, target_path)
