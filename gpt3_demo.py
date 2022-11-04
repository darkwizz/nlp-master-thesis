import os
from time import sleep
import openai
import tqdm

from utils.workflow import load_datasets, write_results_to_tsv, info_message


def prepare_prompt(question_item, few_shot_set, examples_count, seed=12904):
    few_shot_examples = few_shot_set.shuffle(seed=seed)[:examples_count]
    examples_list = []
    for question, answer in zip(few_shot_examples['question'], few_shot_examples['answer']):
        example_prompt = f'Q: {question}' + '\n' + f'A: {answer}.' + '\n'
        examples_list.append(example_prompt)
    question_prompt = f'Q: {question_item["question"]}' + '\nA: '
    examples_list.append(question_prompt)
    result = '\n'.join(examples_list)
    return result


@info_message('Answering question using GPT3')
def get_gpt3_answered_questions(api_key, data, few_shot_set, seed=29197, few_shot_examples_count=10):
    questions = []
    answers = []
    expected = []
    openai.api_key = api_key
    for item in tqdm.tqdm(data):
        questions.append(item['question'])
        expected.append(item['answer'])
        prompt = prepare_prompt(item, few_shot_set, few_shot_examples_count, seed)
        response = openai.Completion.create(
            model='text-davinci-002',
            prompt=prompt,
            temperature=0.3,
            max_tokens=64,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        try:
            text = response['choices'][0]['text']
            if 'A: ' in text:
                answer = text.split('A: ')[1]
            else:
                answer = text.rstrip()
            answer = answer.strip('.')
            answer = answer.replace('\n', ' ')
            answers.append(answer)
            sleep(0.5)
        except:
            print(response)
            raise
    return answers, questions, expected


def main():
    base_path = os.getenv('BASE_DATA_PATH', './data/poleval')
    api_key = os.getenv('OPENAI_API_KEY', None)
    seed = int(os.getenv('OPENAI_SEED', '9723'))
    few_shot_examples_count = int(os.getenv('OPENAI_FEW_SHOT_COUNT', '10'))

    if not api_key:
        print('No OpenAI API key provided')
        exit(-1)
    
    data = load_datasets(test=f'{base_path}/test', dev=f'{base_path}/dev')
    few_shot_set = data['dev'].train_test_split(test_size=0.1, seed=seed)['test']
    answers, questions, expected = get_gpt3_answered_questions(api_key, data['test'], few_shot_set, seed, few_shot_examples_count)
    results_dir = './gpt3-results'
    write_results_to_tsv(results_dir, questions, answers, expected)


if __name__ == "__main__":
    main()