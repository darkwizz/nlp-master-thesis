from collections import defaultdict
from datasets import load_dataset
import requests

WIKIMEDIA_URL_TEMPLATE = 'https://pl.wikipedia.org/w/api.php?action=query&prop=pageprops&ppprop=wikibase_item&redirects=1&titles=|&|&format=json'
WIKIDATA_CLAIMS_URL_TEMPLATE = 'https://www.wikidata.org/w/api.php?action=wbgetentities&ids=|&|&languages=en|pl&props=claims&format=json'
WIKIDATA_P31_RESOLVE_TEMPLATE = 'https://www.wikidata.org/w/api.php?action=wbgetentities&ids=|&|&languages=en|pl&props=labels|descriptions&format=json'
WIKIDATA_ITEMS_LIMIT = 50


def save_labelled_poquad(labelled_poquad):
    pass


def fill_initial_buffer_for_item(buffer, item):
    wiki_url_title = item['url'].split('/wiki/')[1]
    title = item['title']
    item_qas = []
    for qa in item['paragraphs'][0]['qas']:
        question = qa['question']
        answer_key = 'answers' if not qa['is_impossible'] else 'plausible_answers'
        answer = qa[answer_key][0]['generative_answer']
        item_qas.append({'question': question, 'answer': answer, 'is_impossible': qa['is_impossible']})
    
    if title in buffer:
        buffer[title]['qas'].extend(item_qas)
    else:
        buffer[title] = {
            'wiki_title': wiki_url_title,
            'title': title,
            'qas': item_qas
        }


def retrieve_claims_ids(buffer):
    titles = '|'.join([item['wiki_title'] for item in buffer.values()])
    wikimedia_url = WIKIMEDIA_URL_TEMPLATE.replace('|&|', titles)
    response = requests.get(wikimedia_url)
    response.raise_for_status()
    claims_ids = {}
    response_data = response.json()
    for page in response_data['query']['pages'].values():
        item_id = page['pageprops']['wikibase_item']
        claims_ids[item_id] = page['title']
    return claims_ids


def retrieve_p31_ids(buffer, claims_ids):
    claims = '|'.join(claims_ids)
    wikidata_url = WIKIDATA_CLAIMS_URL_TEMPLATE.replace('|&|', claims)
    response = requests.get(wikidata_url)
    response.raise_for_status()
    p31_ids = defaultdict(list)
    for entity in response.json()['entities'].values():
        p31 = entity['claims'].get('P31', None)
        if not p31:
            claim_id = entity['id']
            entity_title = claims_ids[claim_id]
            buffer[entity_title]['en_label'] = 'unknown'
            buffer[entity_title]['pl_label'] = 'niewiadome'
            buffer[entity_title]['description'] = ''
            continue
        p31_id = p31[0]['mainsnak']['datavalue']['value']['id']
        p31_ids[p31_id].append(entity['id'])
    return p31_ids


def fill_final_data_to_buffer(buffer, claims_ids, p31_ids):
    p31s = '|'.join(p31_ids)
    p31_url = WIKIDATA_P31_RESOLVE_TEMPLATE.replace('|&|', p31s)
    response = requests.get(p31_url)
    response.raise_for_status()
    response_data = response.json()
    for entity in response_data['entities'].values():
        entity_id = entity['id']
        entity_claims = p31_ids[entity_id]
        for claim_id in entity_claims:
            entity_title = claims_ids[claim_id]
            en_label = entity['labels'].get('en', {}).get('value', '<en-nope>')
            pl_label = entity['labels'].get('pl', {}).get('value', '<nope>')
            description = entity['descriptions'].get('pl', {}).get('value', '')
            buf_item = buffer[entity_title]
            buf_item['en_label'] = en_label
            buf_item['pl_label'] = pl_label
            buf_item['description'] = description
            buf_item['title_id'] = claim_id


def fill_subset_from_buffer(result_subset, buffer):
    for buf_item in buffer.values():
        for qa in buf_item['qas']:
            result_item = {
                'question': qa['question'],
                'answer': qa['answer'],
                'title': buf_item['title'],
                'en_label': buf_item['en_label'],
                'pl_label': buf_item['pl_label'],
                'description': buf_item['description'],
                'is_impossible': qa['is_impossible']
            }
            result_subset.append(result_item)


def main():
    print('Downloading PoQuAD and add labels from WikiData')
    data_files = {'dev': './poquad/dev.json', 'train': './poquad/train.json'}
    poquad = load_dataset('json', data_files=data_files, field='data')
    result = {}
    for subset in poquad:
        result_subset = []
        buffer = {}
        for item in poquad[subset]:
            fill_initial_buffer_for_item(buffer, item)

            if len(buffer) == WIKIDATA_ITEMS_LIMIT:
                claims_ids = retrieve_claims_ids(buffer)
                p31_ids = retrieve_p31_ids(buffer, claims_ids)
                fill_final_data_to_buffer(buffer, claims_ids, p31_ids)
                fill_subset_from_buffer(result_subset, buffer)
                
                buffer.clear()
                claims_ids.clear()
                p31_ids.clear()
        result[subset] = result_subset
    
    save_labelled_poquad(result)


if __name__ == '__main__':
    main()