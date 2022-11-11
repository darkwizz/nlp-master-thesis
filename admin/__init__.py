import os
import sys
from tqdm import tqdm

COMMAND_FILE = 'command.py'


def list_command_providers():
    admin_path = os.path.dirname(__file__)
    result = []
    for admin_subitem in os.listdir(admin_path):
        full_subpath = os.path.join(admin_path, admin_subitem)
        if os.path.isdir(full_subpath) and COMMAND_FILE in os.listdir(full_subpath):
            result.append(admin_subitem)
    return result


def perform_merge(source_path, results_path):
    from utils.workflow import load_data_for_split
    
    subsets = ['dev', 'train', 'test']
    for subset in subsets:
        os.makedirs(os.path.join(results_path, subset), exist_ok=True)

    expected_files = {subset: open(os.path.join(results_path, subset, 'expected.tsv'), 'a') for subset in subsets}
    in_files = {subset: open(os.path.join(results_path, subset, 'in.tsv'), 'a') for subset in subsets}
    for dataset_dir in os.listdir(source_path):
        print(f'Processing dataset: {dataset_dir}...')
        for subset in subsets:
            dataset_path = os.path.join(source_path, dataset_dir, subset)
            if not os.path.exists(dataset_path):
                continue

            dataset = load_data_for_split(os.path.join(dataset_path, 'in.tsv'), 'question', os.path.join(dataset_path, 'expected.tsv'), 'answer')
            for item in tqdm(dataset):
                question_line = item["question"] + '\n'
                answer_line = "\t".join([item["answer"]] + item.get("alternatives")) + '\n'
                expected_files[subset].write(answer_line)
                in_files[subset].write(question_line)
    for exp_file, in_file in zip(expected_files.values(), in_files.values()):
        exp_file.close()
        in_file.close()


def get_dataset_stats(engine, dataset_path):
    from utils.workflow import load_data_for_split

    if engine.lower() == 'spacy':
        import spacy
        from utils import get_number_of_tokens_in_txt_spacy
        pl_nlp = spacy.load('pl_core_news_lg')
        get_tokens_number = lambda txt: get_number_of_tokens_in_txt_spacy(pl_nlp, txt)
    else:
        from transformers import AutoTokenizer
        from utils import get_number_of_tokens_in_txt_tokenizer
        tokenizer = AutoTokenizer.from_pretrained(engine)
        get_tokens_number = lambda txt: get_number_of_tokens_in_txt_tokenizer(tokenizer, txt)
    
    result = {}
    for subset in os.listdir(dataset_path):
        result[subset] = {
            'question': { 'min': sys.maxsize, 'max': 0, 'longest': '' },
            'answer': { 'min': sys.maxsize, 'max': 0, 'longest': '' },
        }
        in_path = os.path.join(dataset_path, subset, 'in.tsv')
        expected_path = os.path.join(dataset_path, subset, 'expected.tsv')
        dataset = load_data_for_split(in_path, 'question', expected_path, 'answer')
        for item in tqdm(dataset):
            question_len = get_tokens_number(item['question'])
            answer_len = get_tokens_number(item['answer'])
            result[subset]['question']['min'] = min(result[subset]['question']['min'], question_len)
            result[subset]['question']['max'] = max(result[subset]['question']['max'], question_len)
            result[subset]['answer']['min'] = min(result[subset]['answer']['min'], answer_len)
            result[subset]['answer']['max'] = max(result[subset]['answer']['max'], answer_len)
            if result[subset]['question']['max'] == question_len:
                result[subset]['question']['longest'] = item['question']
            if result[subset]['answer']['max'] == answer_len:
                result[subset]['answer']['longest'] = item['answer']
    return result


def print_subsets_stats(subsets_stats):
    print('#' * 50)
    print('Dataset stats')
    print('#' * 50)
    for subset in subsets_stats:
        print(f'Subset: {subset}')
        question_stats = subsets_stats[subset]['question']
        answer_stats = subsets_stats[subset]['answer']
        print(f'Questions => min: {question_stats["min"]}, max: {question_stats["max"]}, longest question: {question_stats["longest"]}')
        print(f'Answers => min: {answer_stats["min"]}, max: {answer_stats["max"]}, longest answer: {answer_stats["longest"]}')
