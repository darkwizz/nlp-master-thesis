import os
import random
import torch
from functools import wraps


def info_message(message):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            print(f'{message}...')
            return func(*args, **kwargs)
        return wrapper
    return decorator


@info_message('Saving the model')
def save_trained_model(args, model):
    target_dir = os.path.join('.', args.model_name, args.revision, 'trained-model')
    model.save_pretrained(target_dir)


@info_message('Loading all data from the specified location')
def load_datasets(extension='tsv', seed=93682, **dataset_paths):
    from datasets import DatasetDict
    
    result_dict = {}
    for dataset_key, dataset_base_dir in dataset_paths.items():
        questions_path = os.path.join(dataset_base_dir, f'in.{extension}')
        answers_path = os.path.join(dataset_base_dir, f'expected.{extension}')
        if not os.path.exists(answers_path):
            dataset = load_data_for_split(questions_path, 'question', seed=seed)
        else:
            dataset = load_data_for_split(questions_path, 'question', answers_path, 'answer', seed=seed)
        result_dict[dataset_key] = dataset
    result = DatasetDict(result_dict)
    return result


def load_data_for_split(questions_path, questions_feature_name, answers_path=None,
                        answers_feature_name=None, sep='\t', seed=25462):
    from datasets import Dataset

    rand = random.Random(x=seed)
    result = {
        questions_feature_name: [],
        'alternatives': []
    }
    with open(questions_path) as questions_file:
        for line in questions_file:
            formatted_line = line.strip()
            if formatted_line:
                question = formatted_line.split(sep)[0]
                result[questions_feature_name].append(question)
    
    if answers_path and answers_feature_name:
        result[answers_feature_name] = []
        with open(answers_path) as answers_file:
            for line in answers_file:
                available_answers = line.strip().split(sep)
                alternatives = []
                if available_answers:
                    answer_position = rand.randint(0, len(available_answers) - 1)
                    choice = available_answers[answer_position]
                    result[answers_feature_name].append(choice)
                    for i in list(range(answer_position)) + list(range(answer_position + 1, len(available_answers))):
                        alternatives.append(available_answers[i])
                    result['alternatives'].append(alternatives)
    return Dataset.from_dict(result)


def write_data_to_tsv(path, data):
    with open(path, 'w') as tsv_file:
        lines = [line.replace("\n", "").strip() + '\n' for line in data]
        tsv_file.writelines(lines)


@info_message('Writing model testing results')
def write_results_to_tsv(results_base_path, questions, answers, expected):
    if not os.path.exists(results_base_path):
        os.mkdir(results_base_path)

    in_path = os.path.join(results_base_path, 'in.tsv')
    out_path = os.path.join(results_base_path, 'out.tsv')
    expected_path = os.path.join(results_base_path, 'expected.tsv')
    for path, data in zip([in_path, out_path, expected_path], [questions, answers, expected]):
        write_data_to_tsv(path, data)


@info_message('Testing model')
def get_answered_questions(dataset, model, tokenizer, batch_size=50, max_len=None):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    start = 0
    outputs = []
    questions = []
    expected = []
    while start < len(dataset['input_ids']):
        end = start + batch_size
        batch = dataset['input_ids'][start:end]
        if max_len is None:
            batch_outs = model.generate(input_ids=torch.tensor(batch, device=device))
        else:
            batch_outs = model.generate(input_ids=torch.tensor(batch), max_length=max_len, device=device)
        decoded = [tokenizer.decode(item, skip_special_tokens=True) for item in batch_outs]
        outputs.extend(decoded)
        questions.extend(dataset['question'][start:end])
        if 'answer' in dataset.features:
            expected.extend(dataset['answer'][start:end])
        start = end
    return outputs, questions, expected
