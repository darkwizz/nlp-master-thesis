import os
import random
from datasets import Dataset, DatasetDict


def load_datasets(extension='tsv', **dataset_paths):
    result_dict = {}
    for dataset_key, dataset_base_dir in dataset_paths.items():
        questions_path = os.path.join(dataset_base_dir, f'in.{extension}')
        answers_path = os.path.join(dataset_base_dir, f'expected.{extension}')
        if not os.path.exists(answers_path):
            dataset = load_data_for_split(questions_path, 'question')
        else:
            dataset = load_data_for_split(questions_path, 'question', answers_path, 'answer')
        result_dict[dataset_key] = dataset
    result = DatasetDict(result_dict)
    return result


def load_data_for_split(questions_path, questions_feature_name, answers_path=None,
                        answers_feature_name=None, sep='\t', seed=25462):
    rand = random.Random(x=seed)
    result = {
        questions_feature_name: []
    }
    with open(questions_path) as questions_file:
        for line in questions_file:
            formatted_line = line.strip()
            if formatted_line:
                result[questions_feature_name].append(formatted_line)
    
    if answers_path and answers_feature_name:
        result[answers_feature_name] = []
        with open(answers_path) as answers_file:
            for line in answers_file:
                available_answers = line.strip().split(sep)
                if available_answers:
                    choice = rand.choice(available_answers)
                    result[answers_feature_name].append(choice)
    return Dataset.from_dict(result)