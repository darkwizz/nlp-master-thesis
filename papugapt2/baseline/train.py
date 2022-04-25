from utils import load_datasets, write_results_to_tsv, get_answered_questions, save_trained_model
from papugapt2.utils import get_gpt2_tokenizer_function
from transformers import AutoTokenizer, AutoModelWithLMHead, DataCollatorForLanguageModeling, TrainingArguments, Trainer


def main(parsed_args):
    base_data_path = parsed_args.base_data_path
    data = load_datasets(test_A=f'{base_data_path}/test-A', test_B=f'{base_data_path}/test-B',
                         train=f'{base_data_path}/train')
    
    tokenizer = AutoTokenizer.from_pretrained(parsed_args.tokenizer_path)
    tokenizer.pad_token = tokenizer.eos_token
    q_maxlen = parsed_args.question_max_length
    a_maxlen = parsed_args.answer_max_length
    tokenized_data = data.map(get_gpt2_tokenizer_function(tokenizer, q_maxlen, a_maxlen), batched=True)

    model = AutoModelWithLMHead.from_pretrained(parsed_args.model_path)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    training_args = TrainingArguments(
        output_dir=parsed_args.training_output_dir,
        overwrite_output_dir=True,
        evaluation_strategy='steps',
        per_device_train_batch_size=32,
        per_device_eval_batch_size=24,
        do_train=True,
        do_eval=True,
        learning_rate=1e-3,
        warmup_ratio=5e-4,
        eval_steps=150,
        gradient_accumulation_steps=1,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=10
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_data['train'],
        eval_dataset=tokenized_data['test_A'],
        tokenizer=tokenizer,
        data_collator=data_collator
    )
    if not parsed_args.skip_training:
        trainer.train()

    test_batch_size = parsed_args.test_batch_size
    test_max_len = parsed_args.test_max_length
    answers, questions, expected = get_answered_questions(tokenized_data['test_A'], model, tokenizer, test_batch_size, test_max_len)
    results_base_path = f'./{parsed_args.model_name}/{parsed_args.revision}/{parsed_args.results_dir}'
    write_results_to_tsv(results_base_path, questions, answers, expected)

    if parsed_args.save_pretrained:
        save_trained_model