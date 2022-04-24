from utils import load_datasets
from transformers import T5Tokenizer, DataCollatorForSeq2Seq, \
                         T5ForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer


def get_t5_tokenizer_function(tokenizer, question_max_len=50, answer_max_len=10):
    def tokenize_function(examples):
        model_inputs = tokenizer(examples['question'], padding='max_length', max_length=question_max_len)
        
        if 'answer' in examples:
            labels = tokenizer(examples['answer'], padding='max_length', max_length=answer_max_len)
            labels_ids = [[(id if id != tokenizer.pad_token_id else -100) for id in example] for example in labels['input_ids']]
            model_inputs['labels'] = labels_ids
        return model_inputs
    return tokenize_function


def main(parsed_args):
    base_path = parsed_args.base_path
    data = load_datasets(test_A=f'{base_path}/test-A', test_B=f'{base_path}/test-B', train=f'{base_path}/train')
    print(data)
