import spacy


_pl_nlp = spacy.load('pl_core_news_lg')


def fill_gap_filter(data_item):
    return '...' in data_item['question']


def boolean_filter(data_item):
    return 'Czy ' in data_item['question'] and ' czy ' not in data_item['question']


def multiple_choise_filter(data_item):
    return ' czy ' in data_item['question']


def living_entity_filter(data_item):
    return data_item['question'].startswith('Kto ') or ' kto jest' in data_item['question']


def named_entity_filter(data_item):
    return bool(_pl_nlp(data_item['answer']).ents)
