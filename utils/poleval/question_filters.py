import spacy


_pl_nlp = spacy.load('pl_core_news_lg')

# TODO: describe in the thesis


def fill_gap_filter(data_item):
    return '...' in data_item['question']


def boolean_filter(data_item):
    return 'Czy ' in data_item['question'] and ' czy ' not in data_item['question']


def multiple_choice_filter(data_item):
    return ' czy ' in data_item['question']


def person_entity_filter(data_item):
    return data_item['question'].startswith('Kto ') or ' kto jest' in data_item['question']


def named_entity_filter(data_item):
    return bool(_pl_nlp(data_item['answer'], disable=["tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer"]).ents)


def numeric_entity_filter(data_item):
    ans_tokens = _pl_nlp(data_item['answer'], disable=["tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer"])
    like_number = ans_tokens[0].like_num
    for alt in data_item['alternatives']:
        if like_number:
            break
        ans_tokens = _pl_nlp(alt, disable=["tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer"])
        like_number = ans_tokens[0].like_num
    return like_number


def propn_filter(data_item):
    doc = _pl_nlp(data_item['answer'], disable=["tagger", "parser", "lemmatizer"])
    for token in doc:
        if token.pos_ == 'PROPN':
            return True
    return False
