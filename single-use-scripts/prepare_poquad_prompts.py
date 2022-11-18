from argparse import ArgumentParser
from datasets import load_dataset, Dataset
import sys

print(sys.path)

from utils.workflow import save_data
from utils.data_preprocess import PL_ANSWER_PROMPTS, PROMPT_PLACEHOLDER
from random import Random
from tqdm import tqdm


HAS_LABEL_QUESTION_PROMPTS = [
    f'Dane pytanie dotyczy obiektu "<title>" typu "<label>". Proszę podać odpowiedź: {PROMPT_PLACEHOLDER}',
    f'To pytanie z rozdziału "<title>", kategoria "<label>": {PROMPT_PLACEHOLDER}',
    f'W kategorii "<label>" pytanie poniżej odnosi się do "<title>": {PROMPT_PLACEHOLDER}',
    f'Proszę odpowiedzieć na pytanie o "<title>" kategorii "<label>": {PROMPT_PLACEHOLDER}'
]

NO_LABEL_QUESTION_PROMPTS = [
    f'Odpowiedź na pytanie poniżej odnosi się do informacji o "<title>": {PROMPT_PLACEHOLDER}',
    f'Proszę podać poprawną odpowiedź na pytanie dotyczące "<title>": {PROMPT_PLACEHOLDER}',
    f'Odpowiedź na pytanie poniżej jest zawierana na stonie Wiki o "<title>". Proszę odpowiedzieć na pytanie: {PROMPT_PLACEHOLDER}'
]


def get_group_augmented_dataset(poquad, include_impossible=False, seed=9237):
    rs = Random(x=seed)
    result = []
    for item in tqdm(poquad):
        if not include_impossible and item['is_impossible']:
            continue

        question = item['question']
        answer = item['answer']
        title = item['title']
        label = item.get('pl_label', None)
        if label in ('', '<nope>', None, 'niewiadome'):
            prompt_template = rs.choice(NO_LABEL_QUESTION_PROMPTS).replace('<title>', title)
        else:
            prompt_template = rs.choice(HAS_LABEL_QUESTION_PROMPTS).replace('<title>', title).replace('<label>', label)
        answer_prompt = rs.choice(PL_ANSWER_PROMPTS)
        question = prompt_template.replace(PROMPT_PLACEHOLDER, question)
        answer = answer_prompt.replace(PROMPT_PLACEHOLDER, answer)
        result.append({'question': question, 'answer': answer})
    return Dataset.from_list(result)


def main(include_impossible=False, seed=9237):
    data_files = {'dev': './poquad-with-wikidata/dev.csv', 'train': './poquad-with-wikidata/train.csv'}
    wikidata_poquad = load_dataset('csv', data_files=data_files, sep='\t')
    prompted_poquad = {}
    for subset in wikidata_poquad:
        print(f'Augmenting {subset}...')
        prompted_subset = get_group_augmented_dataset(wikidata_poquad[subset])
        prompted_poquad[subset] = prompted_subset
    save_data(prompted_poquad, './tmp')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-i', '--impossible', action='store_true', help='Include questions which heavily depend on the context which is attached in the original PoQuAD (they have "is_impossible" in true)')
    parser.add_argument('-s', '--seed', type=int, default=9237, help='randomizing seed when sampling prompts. Default is 9237')
    args = parser.parse_args()
    main(args.impossible, args.seed)