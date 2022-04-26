from datasets import Dataset
import numpy as np
from utils import info_message


def get_gpt2_tokenizer_function(tokenizer, max_question_len=45, max_answer_len=10):
    def tokenize(examples):
        if 'answer' in examples:
            prompts = [f'pytanie: {q} Odpowiedź: {a}' for q, a in zip(examples['question'], examples['answer'])]
            max_len = max_question_len + max_answer_len
        else:
            prompts = [f'pytanie: {q} Odpowiedź: ' for q in examples['question']]
            max_len = max_question_len
        results = tokenizer(prompts, padding='max_length', max_length=max_len)
        return results
    return tokenize


@info_message("Splitting the dataset for few-shot testing")
def get_divided_datasets(base_dataset, seed=3673):
    shuffled = base_dataset.shuffle(seed=seed)
    separation_index = int(len(shuffled) * 0.9)
    main, few_shot = shuffled[:separation_index], shuffled[separation_index:]
    return Dataset.from_dict(main), Dataset.from_dict(few_shot)


@info_message("Preparing few-shot prompts")
def get_gpt2_few_shot_prompts(questions, few_shot_dataset, seed=7315, few_shot_size=2, end_seq_sep='\n###\n'):
    rs = np.random.RandomState(seed=seed)
    result = []
    for question in questions:
        indices = rs.randint(0, len(few_shot_dataset), size=few_shot_size)
        few_shots = few_shot_dataset[indices]
        few_shot_prompt = end_seq_sep.join([f'pytanie: {q}\nOdpowiedź: {a}' for q, a in \
                                            zip(few_shots['question'], few_shots['answer'])])
        final_prompt = f'{few_shot_prompt}{end_seq_sep}pytanie: {question}\nOdpowiedź: '
        result.append(final_prompt)
    return result


@info_message("Generating answers for few-shot questions")
def get_few_shot_answers(data, generator, end_sequence='###', max_new_tokens=10, temperature=0.5, batch_size=64):
    answers = []
    start = 0
    while start < len(data):
        end = start + batch_size
        batch = data[start:end]
        generated = generator(batch, end_sequence=end_sequence, return_full_text=False,
                              max_new_tokens=max_new_tokens, temperature=temperature)
        texts = [item[0]['generated_text'] for item in generated]
        answers.extend(texts)
        start = end
    return answers
