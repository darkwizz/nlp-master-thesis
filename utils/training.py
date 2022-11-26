
def get_compute_metrics(tokenizer, expected_ids_preprocess=None, **metrics):
    import evaluate

    loaded_metrics = {}
    for metric, metric_pretty in metrics.items():
        loaded_metrics[metric_pretty] = evaluate.load(metric)
    
    def compute_metrics(pred):
        expected_ids = pred.label_ids
        if expected_ids_preprocess:
            expected_ids = expected_ids_preprocess(expected_ids)
        preds_ids = pred.predictions
        expected = tokenizer.batch_decode(expected_ids, skip_special_tokens=True)
        predicted = tokenizer.batch_decode(preds_ids, skip_special_tokens=True)
        metric_results = {}
        for metric_pretty, metric in loaded_metrics.items():
            result_key = metric.name
            metric_results[metric_pretty] = metric.compute(predictions=predicted, references=expected)[result_key]
        return metric_results
    
    return compute_metrics