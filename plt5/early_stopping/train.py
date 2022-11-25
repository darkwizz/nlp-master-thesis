from plt5 import PlT5Runner
from utils.workflow import save_trained_model
from transformers import Seq2SeqTrainingArguments


def main(parsed_args):
    results_base_path = f'./{parsed_args.model_name}/{parsed_args.revision}/{parsed_args.results_dir}'
    t5_runner = PlT5Runner(parsed_args, results_base_path, True)
    t5_runner.prepare_data()
    t5_runner.prepare_model()
    
    if not parsed_args.skip_training:
        training_args = Seq2SeqTrainingArguments(
            output_dir=parsed_args.training_output_dir,
            evaluation_strategy='steps',
            eval_steps=150,
            learning_rate=3e-4,
            do_train=True,
            do_eval=True,
            generation_max_length=8,
            predict_with_generate=True,
            per_device_train_batch_size=parsed_args.train_batch,
            per_device_eval_batch_size=parsed_args.eval_batch,
            gradient_accumulation_steps=1,
            adafactor=True,
            warmup_ratio=0,
            weight_decay=0.01,
            save_total_limit=3,
            overwrite_output_dir=True,
            num_train_epochs=5,
            seed=parsed_args.seed,
            fp16=True
        )
        t5_runner.train(training_args)
    
    if parsed_args.save_pretrained:
        save_trained_model(parsed_args, t5_runner.model)
    
    t5_runner.test()

