from plt5 import PlT5Runner
from plt5.utils import get_t5_metric_eval_preprocess
from utils.workflow import save_trained_model
from utils.training import get_compute_metrics
from transformers import Seq2SeqTrainingArguments, EarlyStoppingCallback


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
            save_steps=450,
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
            load_best_model_at_end=True,
            metric_for_best_model='loss',
            greater_is_better=False,
            num_train_epochs=5,
            seed=parsed_args.seed,
            fp16=parsed_args.fp16
        )
        # Observation: enabling fp16 for plT5 breaks training (eval_loss becomes `nan`)
        metric_eval_preprocess = get_t5_metric_eval_preprocess(t5_runner.tokenizer)
        compute_metrics = get_compute_metrics(t5_runner.tokenizer, expected_ids_preprocess=metric_eval_preprocess, exact_match='EM/Accuracy', google_bleu='GLEU')
        t5_runner.train(training_args, compute_metrics, [EarlyStoppingCallback(early_stopping_patience=parsed_args.patience, early_stopping_threshold=0.1)])
    
    if parsed_args.save_pretrained:
        save_trained_model(parsed_args, t5_runner.model)
    
    t5_runner.test()

