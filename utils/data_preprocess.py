import numpy as np

POLEVAL_QUESTION_TYPES = ['Gap filling', 'Yes/no', 'Multiple choice', 'Living named entity', 'Named entity', 'Numeric', 'Proper noun', 'Rest']

PL_QUESTION_PROMPTS = {
    POLEVAL_QUESTION_TYPES[0]: ['Proszę uzupełnić lukę: |&|'],
    POLEVAL_QUESTION_TYPES[1]: ['Proszę podać odpowiedź "tak" czy "nie": |&|'],
    POLEVAL_QUESTION_TYPES[2]: ['Proszę wybrać poprawną opcję z kilku podanych: |&|', 'Zastanawiam się nad tym, którą odpowiedź z podanych jest poprawną. Proszę pomóc z pytaniem: |&|'],
    POLEVAL_QUESTION_TYPES[3]: ['Dane pytanie jest o żywym podmiocie. Proszę powiedzieć, |&|'],
    POLEVAL_QUESTION_TYPES[4]: ['To pytanie jest po prostu o podmiocie nazwanym. Proszę powiedzieć, |&|'],
    POLEVAL_QUESTION_TYPES[5]: ['Prosze podać liczbę: |&|', 'W tym pytaniu odpowiedź zawiera liczbę. Proszę powiedzieć, |&|'],
    POLEVAL_QUESTION_TYPES[6]: ['W odpowiedzi na dane pytanie jest nazwa własna. Proszę powiedzieć: |&|'],
    POLEVAL_QUESTION_TYPES[7]: ['Proszę odpowiedzieć na pytanie niżej: |&|', 'Zawsze zastanawiałem się: |&|', 'Jaka jest odpowiedź na następujące pytanie: |&|', 'Jest pytanie: |&|']
}

PL_ANSWER_PROMPTS = [
    'Odpowiedź na dane pytanie to |&|',
    'Poprawna odpowiedź: |&|',
    'Proszę podać odpowiedź: |&|',
    'Odpowiedzią na to pytanie jest |&|'
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


def get_prompt_augmented_question(question, prompts, placeholder='|&|', seed=7343):
    rs = np.random.RandomState(seed=seed)
    prompt = rs.choice(prompts)
    result = prompt.replace(placeholder, question)
    return result


def get_artificially_augmented_question(question, prefix=QUESTION_PREFIX, suffix=QUESTION_SUFFIX):
    return f'{prefix}{question}{suffix}'


def get_prompt_augmented_answer(answer, prompts, placeholder='|&|', seed=1362):
    rs = np.random.RandomState(seed=seed)
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


def get_artificially_augmented_dataset(dataset, question_feature='question', answer_feature='answer'):
    preprocessor = _get_item_preprocessor(question_feature, get_artificially_augmented_question, answer_feature, get_artificially_augmented_answer)
    result = dataset.map(lambda item: preprocessor(item))
    return result
