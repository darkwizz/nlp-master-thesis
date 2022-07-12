import os
from time import sleep
import openai
import tqdm

from utils import load_datasets, write_results_to_tsv, info_message


@info_message('Answering question using GPT3')
def get_gpt3_answered_questions(api_key, data):
    questions = []
    answers = []
    expected = []
    openai.api_key = api_key
    for item in tqdm.tqdm(data):
        questions.append(item['question'])
        expected.append(item['answer'])
        response = openai.Completion.create(
            model='text-davinci-002',
            prompt=f'Q: {item["question"]}\n',
            temperature=0.3,
            max_tokens=64,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        try:
            text = response['choices'][0]['text']
            if '\nA: ' in text:
                answer = text.split('\nA: ')[1]
            else:
                answer = text.strip()
            answer = answer.replace('\n', ' ')
            answers.append(answer)
            sleep(3)
        except:
            print(response)
            raise
    return answers, questions, expected


def main():
    base_path = os.getenv('BASE_DATA_PATH', './data')
    api_key = os.getenv('OPENAI_API_KEY', None)

    if not api_key:
        print('No OpenAI API key provided')
        exit(-1)
    
    test_data = load_datasets(test_B=f'{base_path}/test-B')['test_B']
    answers, questions, expected = get_gpt3_answered_questions(api_key, test_data)
    results_dir = './gpt3-results'
    write_results_to_tsv(results_dir, questions, answers, expected)


if __name__ == "__main__":
    main()