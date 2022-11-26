from papugapt2 import PapuGaPT2Runner
from papugapt2.utils import get_gpt2_metric_eval_preprocess, metric_eval_preprocess
from utils.training import get_compute_metrics
from utils.workflow import save_trained_model
from transformers import TrainingArguments, EarlyStoppingCallback


def main(parsed_args):
    results_base_path = f'./{parsed_args.model_name}/{parsed_args.revision}/{parsed_args.results_dir}'
    papugapt2_runner = PapuGaPT2Runner(parsed_args, results_base_path, True)
    papugapt2_runner.prepare_data()
    papugapt2_runner.prepare_model()
    
    if not parsed_args.skip_training:
        training_args = TrainingArguments(
            output_dir=parsed_args.training_output_dir,
            overwrite_output_dir=True,
            evaluation_strategy='steps',
            per_device_train_batch_size=parsed_args.train_batch,
            per_device_eval_batch_size=parsed_args.eval_batch,
            eval_steps=150,
            save_steps=450,
            do_train=True,
            do_eval=True,
            learning_rate=5e-6,
            warmup_steps=1000,
            gradient_accumulation_steps=1,
            weight_decay=0.01,
            save_total_limit=3,
            num_train_epochs=5,
            seed=parsed_args.seed,
            fp16=parsed_args.fp16,
            load_best_model_at_end=True,
            metric_for_best_model='loss',
            greater_is_better=False
        )
        expected_preprocess = get_gpt2_metric_eval_preprocess(papugapt2_runner.tokenizer)
        # compute_metrics = get_compute_metrics(papugapt2_runner.tokenizer, expected_ids_preprocess=expected_preprocess, predicted_ids_preprocess=metric_eval_preprocess, exact_match='EM/Accuracy', google_bleu='GLEU')
        papugapt2_runner.train(training_args, callbacks=[EarlyStoppingCallback(early_stopping_patience=1, early_stopping_threshold=0.1)])
    
    if parsed_args.save_pretrained:
        save_trained_model(parsed_args, papugapt2_runner.model)
    
    papugapt2_runner.test()
