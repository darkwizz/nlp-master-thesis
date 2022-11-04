from utils.workflow import load_datasets, write_results_to_tsv, get_answered_questions, save_trained_model
from plt5.utils import get_t5_tokenizer_function
from transformers import T5Tokenizer, DataCollatorForSeq2Seq, \
                         T5ForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer


def main(parsed_args):
    base_data_path = parsed_args.base_data_path
    data = load_datasets(test=f'{base_data_path}/test', dev=f'{base_data_path}/dev',
                         train=f'{base_data_path}/train')
    
    tokenizer = T5Tokenizer.from_pretrained(parsed_args.tokenizer_path)
    q_maxlen = parsed_args.question_max_length
    a_maxlen = parsed_args.answer_max_length
    tokenized_data = data.map(get_t5_tokenizer_function(tokenizer, q_maxlen, a_maxlen), batched=True)

    model = T5ForConditionalGeneration.from_pretrained(parsed_args.model_path)
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    training_args = Seq2SeqTrainingArguments(
        output_dir=parsed_args.training_output_dir,
        evaluation_strategy='steps',
        eval_steps=150,
        learning_rate=3e-4,
        do_train=True,
        do_eval=True,
        generation_max_length=8,
        predict_with_generate=True,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=1,
        adafactor=True,
        warmup_ratio=0,
        weight_decay=0.01,
        save_total_limit=3,
        overwrite_output_dir=True,
        num_train_epochs=30
        # fp16=True
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_data['train'],
        eval_dataset=tokenized_data['dev'],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    if not parsed_args.skip_training:
        trainer.train()

    test_batch_size = parsed_args.test_batch_size
    test_max_len = parsed_args.test_max_length
    answers, questions, expected = get_answered_questions(tokenized_data['test'], model,
                                            tokenizer, test_batch_size, test_max_len)
    results_base_path = f'./{parsed_args.model_name}/{parsed_args.revision}/{parsed_args.results_dir}'
    write_results_to_tsv(results_base_path, questions, answers, expected)

    if parsed_args.save_pretrained:
        save_trained_model(parsed_args, model)
