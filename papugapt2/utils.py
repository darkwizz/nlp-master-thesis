

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