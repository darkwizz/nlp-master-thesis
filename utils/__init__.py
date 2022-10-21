import spacy
import numpy as np
import seaborn as sns
from tqdm import tqdm

sns.set_theme()

_pl_nlp = spacy.load('pl_core_news_lg')


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


def get_number_of_tokens_in_dataset(dataset, fields_to_count):
    result = 0
    for line in tqdm(dataset):
        for field in fields_to_count:
            result += get_number_of_tokens_in_txt(line[field])
    return result