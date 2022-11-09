from papugapt2 import PapuGaPT2Runner
from utils.workflow import save_trained_model
from papugapt2.utils import get_divided_datasets, get_gpt2_few_shot_prompts, get_few_shot_answers
from transformers import TrainingArguments, pipeline


def test_model_few_shot(args, model, tokenizer, dataset):
    main_dataset, few_shot = get_divided_datasets(dataset)
    few_shot_prompts = get_gpt2_few_shot_prompts(main_dataset['question'], few_shot)
    generator = pipeline('text-generation', model=model, tokenizer=tokenizer)
    test_max_length = args.answer_max_length or 10
    test_batch_size = args.test_batch_size
    answers = get_few_shot_answers(few_shot_prompts, generator, max_new_tokens=test_max_length, batch_size=test_batch_size)
    questions = main_dataset['question']
    expected = []
    if 'answer' in main_dataset.features:
        expected = main_dataset['answer']
    return answers, questions, expected


def main(parsed_args):
    few_shot_txt = '-few-shot' if parsed_args.few_shot else ''
    results_base_path = f'./{parsed_args.model_name}/{parsed_args.revision}/{parsed_args.results_dir}{few_shot_txt}'
    papugapt2_runner = PapuGaPT2Runner(parsed_args, results_base_path, True)
    papugapt2_runner.prepare_data()
    papugapt2_runner.prepare_model()
    
    if not parsed_args.skip_training:
        training_args = TrainingArguments(
            output_dir=parsed_args.training_output_dir,
            overwrite_output_dir=True,
            evaluation_strategy='steps',
            per_device_train_batch_size=64,
            per_device_eval_batch_size=64,
            eval_steps=100,
            do_train=True,
            do_eval=True,
            learning_rate=1e-3,
            warmup_steps=1000,
            gradient_accumulation_steps=1,
            weight_decay=0.01,
            save_total_limit=3,
            num_train_epochs=10,
            seed=parsed_args.seed
        )
        papugapt2_runner.train(training_args)
    
    papugapt2_runner.test()
    # answers, questions, expected = test_model_few_shot(parsed_args, model, tokenizer, tokenized_data['test'])

    if parsed_args.save_pretrained:
        save_trained_model(parsed_args, papugapt2_runner.model)