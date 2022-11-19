def get_prompts_for_question(item):
    from utils.data_preprocess import PL_QUESTION_PROMPTS
    from utils.poleval import question_filters, POLEVAL_QUESTION_PROMPTS, POLEVAL_QUESTION_TYPES
    
    prompts = PL_QUESTION_PROMPTS
    if question_filters.fill_gap_filter(item):
        prompts = POLEVAL_QUESTION_PROMPTS[POLEVAL_QUESTION_TYPES.GAP_FILLING]
    elif question_filters.boolean_filter(item):
        prompts = POLEVAL_QUESTION_PROMPTS[POLEVAL_QUESTION_TYPES.BOOLEAN]
    elif question_filters.multiple_choice_filter(item):
        prompts = POLEVAL_QUESTION_PROMPTS[POLEVAL_QUESTION_TYPES.MULTIPLE_CHOICE]
    elif question_filters.person_entity_filter(item):
        prompts = POLEVAL_QUESTION_PROMPTS[POLEVAL_QUESTION_TYPES.PERSON_ENTITY]
    elif question_filters.named_entity_filter(item):
        prompts = POLEVAL_QUESTION_PROMPTS[POLEVAL_QUESTION_TYPES.NAMED_ENTITY]
    elif question_filters.numeric_entity_filter(item):
        prompts = POLEVAL_QUESTION_PROMPTS[POLEVAL_QUESTION_TYPES.NUMERIC]
    elif question_filters.propn_filter(item):
        prompts = POLEVAL_QUESTION_PROMPTS[POLEVAL_QUESTION_TYPES.PROPER_NOUN]
    
    return prompts


def main(args):
    from utils import compose_subsets_paths, get_total_number_of_tokens_in_datasets
    from utils.workflow import load_datasets, save_data
    from admin import ANSWER_FEATURE, QUESTION_FEATURE
    from utils.data_preprocess import get_artificially_augmented_dataset, get_prompt_augmented_dataset, get_special_prompt_augmented_dataset
    
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
    elif args.prompts:
        data = get_prompt_augmented_dataset(data, QUESTION_FEATURE, ANSWER_FEATURE, prompt_target=args.prompt_target, seed=args.seed)
    elif args.gprompts:
        data = get_special_prompt_augmented_dataset(data, get_prompts_for_question, QUESTION_FEATURE, ANSWER_FEATURE, prompt_target=args.prompt_target, seed=args.seed)
    
    target_path = args.target_path
    save_data(data, target_path)
