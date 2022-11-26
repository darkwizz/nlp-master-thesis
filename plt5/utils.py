

def get_t5_metric_eval_preprocess(tokenizer):
    def metric_eval_preprocess(expected_ids):
        expected_ids[expected_ids == -100] = tokenizer.pad_token_id
        return expected_ids
    
    return metric_eval_preprocess


def get_t5_tokenizer_function(tokenizer, question_max_len=50, answer_max_len=10):
    def tokenize_function(examples):
        model_inputs = tokenizer(examples['question'], padding='max_length', max_length=question_max_len)
        
        if 'answer' in examples:
            labels = tokenizer(examples['answer'], padding='max_length', max_length=answer_max_len)
            labels_ids = [[(id if id != tokenizer.pad_token_id else -100) for id in example] for example in labels['input_ids']]
            model_inputs['labels'] = labels_ids
        return model_inputs
    return tokenize_function
