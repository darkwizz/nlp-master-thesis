

from utils.data_preprocess import PROMPT_PLACEHOLDER


class MKQA_QUESTION_TYPES:
    BINARY = 'binary'
    DATE = 'date'
    ENTITY = 'entity'
    NUMBER = 'number'
    NUMBER_WITH_UNIT = 'number_with_unit'
    SHORT_PHRASE = 'short_phrase'

MKQA_QUESTION_PROMPTS = {
    MKQA_QUESTION_TYPES.BINARY: [f'Proszę podać odpowiedź "tak" czy "nie": {PROMPT_PLACEHOLDER}', f'Dane pytanie jest binarne, proszę odpowiedzieć: {PROMPT_PLACEHOLDER}'],
    MKQA_QUESTION_TYPES.DATE: [f'Proszę podać datę w odpowiedzi: {PROMPT_PLACEHOLDER}'],
    MKQA_QUESTION_TYPES.ENTITY: [f'Dane pytanie jest o podmiocie. Proszę powiedzieć: {PROMPT_PLACEHOLDER}', f'Odpowiedź na pytnie poniżej zakłada podmiot: {PROMPT_PLACEHOLDER}'],
    MKQA_QUESTION_TYPES.NUMBER: [f'Prosze podać liczbę: {PROMPT_PLACEHOLDER}', f'W tym pytaniu odpowiedź zawiera liczbę. Proszę powiedzieć, {PROMPT_PLACEHOLDER}'],
    MKQA_QUESTION_TYPES.NUMBER_WITH_UNIT: [f'Proszę podać liczbę i jednostkę w odpowiedzi na pytanie: {PROMPT_PLACEHOLDER}'],
    MKQA_QUESTION_TYPES.SHORT_PHRASE: [f'Odpowiedzią na pytnie poniżej powinno być krótkie zdanie. Proszę powiedzieć: {PROMPT_PLACEHOLDER}']
}