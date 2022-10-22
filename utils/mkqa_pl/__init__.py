import os
from tqdm import tqdm
from datasets import load_dataset_builder, load_dataset, Dataset

_mkqa_spec = load_dataset_builder('mkqa').info


def map_numeric_question_type_to_str(int_question_type):
    type_definition = _mkqa_spec.features['answers']['pl'][0]['type']
    return type_definition.int2str(int_question_type)


def _extract_pl_mkqa(mkqa):
    result = []
    for item in tqdm(mkqa):
        answers = item['answers']['pl']
        if len(answers) > 1:
            continue
        
        ans_type = answers[0]['type']
        if type(ans_type) is int:
            ans_type = map_numeric_question_type_to_str(ans_type)
        
        if ans_type == 'unanswerable':
            continue

        if not answers[0]['text']:
            continue
        
        result.append({
            'question': f'{item["queries"]["pl"]}?',
            'orig_question': f'{item["query"]}?',
            'answer': answers[0]['text'],
            'answer_type': ans_type,
            'alternatives': answers[0]['aliases']
        })
    return Dataset.from_list(result)


def download_pl_subset(train_test_split=0.7, seed=42621):
    mkqa = load_dataset('mkqa', split='train')  # MKQA has only 'train' split, but the result is still returned as DatasetDict
    cleaned_mkqa = _extract_pl_mkqa(mkqa)
    result_mkqa = cleaned_mkqa.train_test_split(train_size=train_test_split, seed=seed)
    dev_test_result = result_mkqa['test'].train_test_split(0.5, seed=seed)
    result_mkqa['dev'] = dev_test_result['train']
    result_mkqa['test'] = dev_test_result['test']
    return result_mkqa


def save_dataset(dataset, target_dir, extension='tsv', sep='\t'):
    if not (os.path.exists(target_dir) and os.path.isdir(target_dir)):
        os.mkdir(target_dir)
    in_out = open(os.path.join(target_dir, f'in.{extension}'), 'w')
    expected_out = open(os.path.join(target_dir, f'expected.{extension}'), 'w')
    for item in dataset:
        question_line = f'{item["question"]}\n'
        answer_line = sep.join([item['answer'],] + (item['alternatives'] or [])) + '\n'
        in_out.write(question_line)
        expected_out.write(answer_line)
    in_out.close()
    expected_out.close()