import os
import spacy
import numpy as np
import seaborn as sns
from tqdm import tqdm

from utils.workflow import info_message

sns.set_theme()

_pl_nlp = spacy.load('pl_core_news_lg')


def compose_subsets_paths(base_path):
    subset_dirs = {}
    for item in os.listdir(base_path):
        key = item.lower().replace('-', '_').replace(' ', '_')
        path = os.path.join(base_path, item)
        subset_dirs[key] = path
    return subset_dirs


def plot_questions_distribution_into_file(grouped_questions, question_types, file_path, ax=None):
    subsets_lens = np.array([item.num_rows for item in grouped_questions.values()])
    if ax:
        bp = sns.barplot(ax=ax, x=question_types, y=subsets_lens / subsets_lens.sum())
    else:
        bp = sns.barplot(x=question_types, y=subsets_lens / subsets_lens.sum())
    bp.set_xticklabels(bp.get_xticklabels(), rotation=45)
    bp.bar_label(bp.containers[0])
    bp.set_ylabel(r'% of total number of questions')
    bp.get_figure().savefig(file_path, bbox_inches='tight')


def get_number_of_tokens_in_txt(text):
    return len(_pl_nlp(text))


@info_message('Counting number of tokens in a dataset')
def get_number_of_tokens_in_dataset(dataset, fields_to_count):
    result = 0
    for line in tqdm(dataset):
        for field in fields_to_count:
            result += get_number_of_tokens_in_txt(line[field])
    return result


@info_message('Counting total number of tokens in data')
def get_total_number_of_tokens_in_datasets(data, fields_to_count):
    total_n_tokens = 0
    for subset in data:
        n_tokens = get_number_of_tokens_in_dataset(data[subset], fields_to_count)
        total_n_tokens += n_tokens
    return total_n_tokens


def save_data(data, target_directory):
    '''
    Used to save datasets in PolEval format into .tsv files of PolEval format.
    Data must be a `DatasetDict` (even with one split).
    '''
    for subset in data:
        target_subset_path = os.path.join(target_directory, subset)
        os.makedirs(target_subset_path, exist_ok=True)
        in_file = open(os.path.join(target_subset_path, 'in.tsv'), 'w')
        expected_file = open(os.path.join(target_subset_path, 'expected.tsv'), 'w')
        print(f'Saving subset: {subset}...')
        for item in tqdm(data[subset]):
            question_line = item['question'] + '\n'
            answer_line = '\t'.join([item['answer']] + item.get('alternatives', [])) + '\n'
            in_file.write(question_line)
            expected_file.write(answer_line)
        in_file.close()
        expected_file.close()
