from datasets import load_dataset
import requests

WIKIMEDIA_URL_TEMPLATE = 'https://pl.wikipedia.org/w/api.php?action=query&prop=pageprops&ppprop=wikibase_item&redirects=1&titles=|&|&format=json'


def save_labelled_poquad(labelled_poquad):
    pass


def main():
    print('Downloading PoQuAD and add labels from WikiData')
    data_files = {'dev': './poquad/dev.json', 'train': './poquad/train.json'}
    poquad = load_dataset('json', data_files=data_files, field='data')
    result = {}
    for subset in poquad:
        result_subset = []
        for item in poquad[subset]:
            pass
    
    save_labelled_poquad(result)


if __name__ == '__main__':
    main()