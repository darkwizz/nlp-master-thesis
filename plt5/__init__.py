

class PlT5Runner:
    def __init__(self, parsed_args, results_base_path, batched_tokenization=True):
        self._base_data_path = parsed_args.base_data_path
        self._tokenizer_path = parsed_args.tokenizer_path
        self._model_path = parsed_args.model_path
        self._q_maxlen = parsed_args.question_max_length
        self._a_maxlen = parsed_args.answer_max_length
        self._batched = batched_tokenization
        self._training_output_dir = parsed_args.training_output_dir
        self._test_batch_size = parsed_args.test_batch_size
        self._test_max_len = parsed_args.test_max_length
        self._data = None