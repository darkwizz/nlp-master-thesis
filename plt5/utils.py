import torch


def get_t5_tokenizer_function(tokenizer, question_max_len=50, answer_max_len=10):
    def tokenize_function(examples):
        model_inputs = tokenizer(examples['question'], padding='max_length', max_length=question_max_len)
        
        if 'answer' in examples:
            labels = tokenizer(examples['answer'], padding='max_length', max_length=answer_max_len)
            labels_ids = [[(id if id != tokenizer.pad_token_id else -100) for id in example] for example in labels['input_ids']]
            model_inputs['labels'] = labels_ids
        return model_inputs
    return tokenize_function


def get_answered_questions(dataset, model, tokenizer, batch_size=50, max_len=None):
    start = 0
    outputs = []
    questions = []
    expected = []
    while start < len(dataset['input_ids']):
        end = start + batch_size
        batch = dataset['input_ids'][start:end]
        if max_len is None:
            batch_outs = model.generate(input_ids=torch.tensor(batch))
        else:
            batch_outs = model.generate(input_ids=torch.tensor(batch), max_length=max_len)
        decoded = [tokenizer.decode(item, skip_special_tokens=True) for item in batch_outs]
        outputs.extend(decoded)
        questions.extend(dataset['question'][start:end])
        if 'answer' in dataset:
            expected.extend(dataset['answer'][start:end])
        start = end
    return outputs, questions, expected
