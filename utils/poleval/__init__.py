
from utils.data_preprocess import PROMPT_PLACEHOLDER


class POLEVAL_QUESTION_TYPES:
    GAP_FILLING = 'Gap filling'
    BOOLEAN = 'Yes/no'
    MULTIPLE_CHOICE = 'Multiple choice'
    PERSON_ENTITY = 'Person entity'
    NAMED_ENTITY = 'Named entity'
    NUMERIC = 'Numeric'
    PROPER_NOUN = 'Proper noun'
    REST = 'Rest'

POLEVAL_QUESTION_PROMPTS = {
    POLEVAL_QUESTION_TYPES.GAP_FILLING: [f'Proszę uzupełnić lukę: {PROMPT_PLACEHOLDER}'],
    POLEVAL_QUESTION_TYPES.BOOLEAN: [f'Proszę podać odpowiedź "tak" czy "nie": {PROMPT_PLACEHOLDER}'],
    POLEVAL_QUESTION_TYPES.MULTIPLE_CHOICE: [f'Proszę wybrać poprawną opcję z kilku podanych: {PROMPT_PLACEHOLDER}', f'Zastanawiam się nad tym, którą odpowiedź z podanych jest poprawną. Proszę pomóc z pytaniem: {PROMPT_PLACEHOLDER}'],
    POLEVAL_QUESTION_TYPES.PERSON_ENTITY: [f'Dane pytanie odnosi się do człowieka. Proszę powiedzieć, {PROMPT_PLACEHOLDER}'],
    POLEVAL_QUESTION_TYPES.NAMED_ENTITY: [f'To pytanie jest po prostu o podmiocie nazwanym. Proszę powiedzieć, {PROMPT_PLACEHOLDER}'],
    POLEVAL_QUESTION_TYPES.NUMERIC: [f'Prosze podać liczbę: {PROMPT_PLACEHOLDER}', f'W tym pytaniu odpowiedź zawiera liczbę. Proszę powiedzieć, {PROMPT_PLACEHOLDER}'],
    POLEVAL_QUESTION_TYPES.PROPER_NOUN: [f'W odpowiedzi na dane pytanie jest nazwa własna. Proszę powiedzieć: {PROMPT_PLACEHOLDER}'],
    POLEVAL_QUESTION_TYPES.REST: [f'Proszę odpowiedzieć na pytanie niżej: {PROMPT_PLACEHOLDER}', f'Zawsze zastanawiałem się: {PROMPT_PLACEHOLDER}', f'Jaka jest odpowiedź na następujące pytanie: {PROMPT_PLACEHOLDER}', f'Jest pytanie: {PROMPT_PLACEHOLDER}']
}

