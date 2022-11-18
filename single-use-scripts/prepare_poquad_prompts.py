from datasets import load_dataset
from utils.workflow import save_data
from utils.data_preprocess import PL_ANSWER_PROMPTS


HAS_LABEL_QUESTION_PROMPTS = [
]


def get_group_augmented_dataset(poquad):
    return poquad


def main():
    data_files = {'dev': './poquad-with-wikidata/dev.csv', 'train': './poquad-with-wikidata/train.csv'}
    wikidata_poquad = load_dataset('csv', data_files=data_files, sep='\t')
    prompted_poquad = {}
    for subset in wikidata_poquad:
        prompted_subset = get_group_augmented_dataset(wikidata_poquad[subset])
        prompted_poquad[subset] = prompted_subset
    save_data(prompted_poquad, './tmp')


if __name__ == '__main__':
    main()