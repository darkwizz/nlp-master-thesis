import numpy as np

from utils.workflow import info_message


PROMPT_TARGETS = ('both', 'question', 'answer')
PROMPT_PLACEHOLDER = '|&|'

PL_QUESTION_PROMPTS = [
    f'Proszę odpowiedzieć na pytanie: {PROMPT_PLACEHOLDER}',
    f'Podane niżej pytanie wymaga odpowiedzi. {PROMPT_PLACEHOLDER}',
    f'{PROMPT_PLACEHOLDER} Proszę podać poprawną odpowiedź na dane pytanie.',
    f'Pytanie: {PROMPT_PLACEHOLDER} Proszę podać odpowiedź.',
    f'Dane pytanie: {PROMPT_PLACEHOLDER}'
]

PL_ANSWER_PROMPTS = [
    f'Odpowiedź na dane pytanie to: {PROMPT_PLACEHOLDER}.',
    f'Poprawna odpowiedź: {PROMPT_PLACEHOLDER}.',
    f'Odpowiedź na pytanie - {PROMPT_PLACEHOLDER}.',
    f'Odpowiedzią na to pytanie jest: {PROMPT_PLACEHOLDER}.',
    f'Odpowiedź: {PROMPT_PLACEHOLDER}.'
]

QUESTION_PREFIX = '<QUESTION>'
QUESTION_SUFFIX = '</QUESTION>'
ANSWER_PREFIX = '<ANSWER>'
ANSWER_SUFFIX = '</ANSWER>'
PL_QUESTION_PREFIX = '<PYTANIE>'
PL_QUESTION_SUFFIX = '</PYTANIE>'
PL_ANSWER_PREFIX = '<ODPOWIEDŹ>'
PL_ANSWER_SUFFIX = '</ODPOWIEDŹ>'


def split_data_by_filters(data, **filters):
    '''
    Returns a dictionary of subsets of the passed data. Each subset is
    created from the part not filtered yet. That is, the order of passing
    filters matters.
    '''
    result = {}
    remaining = data
    for key, filter_func in filters.items():
        result[key] = remaining.filter(filter_func)
        remaining = remaining.filter(lambda item: not filter_func(item))
    result['rest'] = remaining
    return result


def get_prompt_augmented_question(question, prompts, placeholder=PROMPT_PLACEHOLDER, rs=None):
    if not rs:
        rs = np.random.get_state()
    prompt = rs.choice(prompts)
    result = prompt.replace(placeholder, question)
    return result


def get_artificially_augmented_question(question, prefix=QUESTION_PREFIX, suffix=QUESTION_SUFFIX):
    return f'{prefix}{question}{suffix}'


def get_prompt_augmented_answer(answer, prompts, placeholder=PROMPT_PLACEHOLDER, rs=None):
    if not rs:
        rs = np.random.get_state()
    prompt = rs.choice(prompts)
    result = prompt.replace(placeholder, answer)
    return result


def get_artificially_augmented_answer(answer, prefix=ANSWER_PREFIX, suffix=ANSWER_SUFFIX):
    return f'{prefix}{answer}{suffix}'


def _get_item_preprocessor(question_feature, question_processor, answer_feature, answer_processor):
    def item_preprocessor(item):
        result = {}
        for key in item:
            if key == question_feature:
                result[key] = question_processor(item[key])
            elif key == answer_feature:
                result[key] = answer_processor(item[key])
            else:
                result[key] = item[key]
        return result
    return item_preprocessor


@info_message('Enclosing questions and asnwers by an artificial prefix and suffix')
def get_artificially_augmented_dataset(dataset, question_feature='question', answer_feature='answer'):
    preprocessor = _get_item_preprocessor(question_feature, get_artificially_augmented_question, answer_feature, get_artificially_augmented_answer)
    result = dataset.map(lambda item: preprocessor(item))
    return result


@info_message('Prepending questions and answers by a natural prompt')
def get_prompt_augmented_dataset(dataset, question_feature='question', answer_feature='answer', prompt_target=PROMPT_TARGETS[0], seed=2327):
    rs = np.random.RandomState(seed=seed)
    question_prompt_processor = lambda question: get_prompt_augmented_question(question, PL_QUESTION_PROMPTS, rs=rs)
    answer_prompt_processor = lambda answer: get_prompt_augmented_answer(answer, PL_ANSWER_PROMPTS, rs=rs)
    if prompt_target == PROMPT_TARGETS[1]:
        answer_prompt_processor = lambda item: item
    elif prompt_target == PROMPT_TARGETS[2]:
        question_prompt_processor = lambda item: item
    preprocessor = _get_item_preprocessor(question_feature, question_prompt_processor, answer_feature, answer_prompt_processor)
    result = dataset.map(lambda item: preprocessor(item))
    return result
