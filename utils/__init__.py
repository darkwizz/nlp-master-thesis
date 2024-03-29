import os
import numpy as np
from tqdm import tqdm

from utils.workflow import info_message


def compose_subsets_paths(base_path):
    subset_dirs = {}
    for item in os.listdir(base_path):
        key = item.lower().replace('-', '_').replace(' ', '_')
        path = os.path.join(base_path, item)
        subset_dirs[key] = path
    return subset_dirs


def plot_questions_distribution_into_file(grouped_questions, question_types, file_path, ax=None):
    import seaborn as sns

    sns.set_theme()
    subsets_lens = np.array([item.num_rows for item in grouped_questions.values()])
    if ax:
        bp = sns.barplot(ax=ax, x=question_types, y=subsets_lens / subsets_lens.sum())
    else:
        bp = sns.barplot(x=question_types, y=subsets_lens / subsets_lens.sum())
    bp.set_xticklabels(bp.get_xticklabels(), rotation=45)
    bp.bar_label(bp.containers[0])
    bp.set_ylabel(r'% of total number of questions')
    bp.get_figure().savefig(file_path, bbox_inches='tight')


def split_dataset_by_filters(data, **filters):
    result = {}
    remaining = data
    for key, filter_func in filters.items():
        result[key] = remaining.filter(filter_func)
        remaining = remaining.filter(lambda item: not filter_func(item))
    result['rest'] = remaining
    return result


def get_number_of_tokens_in_txt_spacy(nlp, text):
    return len(nlp(text))


def get_number_of_tokens_in_txt_tokenizer(tokenizer, text):
    eos_id = tokenizer(tokenizer.eos_token)['input_ids'][0]
    tokens_ids = tokenizer(text)['input_ids']
    result = len(tokens_ids)
    if eos_id in tokens_ids:
        result -= 1
    return result


@info_message('Counting number of tokens in a dataset using spaCy')
def get_number_of_tokens_in_dataset_spacy(nlp, dataset, fields_to_count):
    result = 0
    for line in tqdm(dataset):
        for field in fields_to_count:
            result += get_number_of_tokens_in_txt_spacy(nlp, line[field])
    return result


@info_message('Counting number of tokens in a dataset using AutoTokenizer')
def get_number_of_tokens_in_dataset_tokenizer(tokenizer, dataset, fields_to_count):
    result = 0
    for line in tqdm(dataset):
        for field in fields_to_count:
            result += get_number_of_tokens_in_txt_tokenizer(tokenizer, line[field])
    return result


@info_message('Counting total number of tokens in data')
def get_total_number_of_tokens_in_datasets(data, fields_to_count, tokenizer_type='spacy'):
    if tokenizer_type.lower() == 'spacy':
        import spacy
        pl_nlp = spacy.load('pl_core_news_lg')
    else:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_type)

    total_n_tokens = 0
    for subset in data:
        if tokenizer_type.lower() == 'spacy':
            n_tokens = get_number_of_tokens_in_dataset_spacy(pl_nlp, data[subset], fields_to_count)
        else:
            n_tokens = get_number_of_tokens_in_dataset_tokenizer(tokenizer, data[subset], fields_to_count)
        total_n_tokens += n_tokens
    return total_n_tokens
