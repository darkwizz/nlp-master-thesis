from pathlib import Path
import os
from admin import QUESTION_FEATURE, ANSWER_FEATURE


def read_poquad_data(base_directory, include_impossible):
    if not os.path.exists(base_directory):
        print(f'Path {base_directory} does not exist')
        exit(-1)
    
    from datasets import load_dataset, Dataset, DatasetDict
    
    poquad_jsons = os.listdir(base_directory)
    data_files = {Path(json).stem: os.path.join(base_directory, json) for json in poquad_jsons}
    data = load_dataset('json', data_files=data_files, field='data')
    result_dict = {}
    for subset in data:
        subset_dict = {
            QUESTION_FEATURE: [],
            ANSWER_FEATURE: [],
            'alternatives': []
        }
        for item in data[subset]:
            for qas in item['paragraphs'][0]['qas']:
                if qas['is_impossible'] and not include_impossible:
                    continue
                
                question = qas['question'].replace('\n', ' ')
                answer_field = qas['plausible_answers'][0] if qas['is_impossible'] else qas['answers'][0]
                answer = answer_field['generative_answer'].replace('\n', ' ')
                alternatives = [] if '\n' in answer_field['text'] else [answer_field['text']]
                subset_dict[QUESTION_FEATURE].append(question)
                subset_dict[ANSWER_FEATURE].append(answer)
                subset_dict['alternatives'].append(alternatives)
        result_dict[subset] = Dataset.from_dict(subset_dict)
    return DatasetDict(result_dict)


def main(args):
    from utils.workflow import save_data
    from utils.data_preprocess import get_artificially_augmented_dataset, get_prompt_augmented_dataset

    base_directory = args.directory
    data = read_poquad_data(base_directory, args.impossible)
    if args.count_token:
        if not args.tokenizer:
            print('Tokenizer must be specified')
            exit(1)

        from utils import get_total_number_of_tokens_in_datasets

        total_n_tokens = get_total_number_of_tokens_in_datasets(data, [QUESTION_FEATURE, ANSWER_FEATURE], args.tokenizer)
        print(f'Total number of tokens in {base_directory}: {total_n_tokens}')
        exit(0)
    
    if args.artificial:
        data = get_artificially_augmented_dataset(data, QUESTION_FEATURE, ANSWER_FEATURE)
    elif args.prompts:
        data = get_prompt_augmented_dataset(data, QUESTION_FEATURE, ANSWER_FEATURE, prompt_target=args.prompt_target, seed=args.seed)
    
    target_path = args.target_path
    save_data(data, target_path)