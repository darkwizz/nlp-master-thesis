import spacy


_pl_nlp = spacy.load('pl_core_news_lg')


def get_number_of_tokens_in_txt(text):
    return len(_pl_nlp(text))


def get_number_of_tokens_in_dataset(dataset, fields_to_count):
    result = 0
    for line in dataset:
        for field in fields_to_count:
            result += get_number_of_tokens_in_txt(line[field])
    return result