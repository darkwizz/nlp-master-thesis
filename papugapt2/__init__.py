import os
import torch
from transformers import AutoTokenizer, AutoModelWithLMHead, DataCollatorForLanguageModeling, Trainer
from papugapt2.utils import get_gpt2_tokenizer_function, EOS_TOKEN
from utils.workflow import get_answered_questions, info_message, load_datasets, write_results_to_tsv

class PapuGaPT2Runner:
    def __init__(self, parsed_args, results_base_path, batched_tokenization=True):
        self._base_data_path = parsed_args.base_data_path
        self._tokenizer_path = parsed_args.tokenizer_path
        self._model_path = parsed_args.model_path
        self._q_maxlen = parsed_args.question_max_length
        self._a_maxlen = parsed_args.answer_max_length
        self._test_batch_size = parsed_args.test_batch_size
        self._test_max_len = parsed_args.test_max_length
        self._results_base_path = results_base_path
        self._batched = batched_tokenization
        self._data = None
        self._tokenizer = None
        self._model = None
        self._device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    @property
    def data(self):
        return self._data
    
    @property
    def tokenizer(self):
        return self._tokenizer
    
    @property
    def model(self):
        return self._model
    
    @info_message('Preparing data')
    def prepare_data(self):
        data_files = {
            'train': os.path.join(self._base_data_path, 'train'),
            'dev': os.path.join(self._base_data_path, 'dev'),
            'test': os.path.join(self._base_data_path, 'test')
        }
        data = load_datasets(**data_files)
        if 'answer' in data['test'].features:
            data['test-no-ans'] = data['test'].remove_columns('answer')
        self._tokenizer = AutoTokenizer.from_pretrained(self._tokenizer_path, padding_side='left')
        self._tokenizer.eos_token = EOS_TOKEN
        self._tokenizer.pad_token = self._tokenizer.eos_token
        self._data = data.map(get_gpt2_tokenizer_function(self._tokenizer, self._q_maxlen, self._a_maxlen), batched=self._batched)
        self._data.set_format("pt", columns=["input_ids", 'attention_mask'], device=self._device, output_all_columns=True)
    
    @info_message('Preparing model')
    def prepare_model(self):
        if not self._tokenizer:
            raise ValueError('For papuGaPT2 tokenizer must be initialized before the model')
        
        self._model = AutoModelWithLMHead.from_pretrained(self._model_path).to(self._device)
        self._model.resize_token_embeddings(len(self._tokenizer))
        self._model.tie_weights()
    
    @info_message('Training')
    def train(self, training_args):
        if not all([self._data, self._model, self._tokenizer]):
            raise ValueError('Model and data must be prepared')
        
        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False)
        trainer = Trainer(
            model=self._model,
            args=training_args,
            train_dataset=self.data['train'],
            eval_dataset=self.data['dev'],
            tokenizer=self.tokenizer,
            data_collator=data_collator
        )
        trainer.train()
    
    def _get_questions_answer_retriever(self):
        def get_answers(raw_batch):
            batch = {
                'input_ids': raw_batch['input_ids'],
                'attention_mask': raw_batch['attention_mask']
            }
            if not self._test_max_len:
                batch_outs = self.model.generate(**batch, pad_token_id=self.tokenizer.eos_token_id)
            else:
                batch_outs = self.model.generate(**batch, pad_token_id=self.tokenizer.eos_token_id, max_new_tokens=self._test_max_len)
            batch_outs = batch_outs[:, batch['input_ids'].shape[1]:]
            decoded = self.tokenizer.batch_decode(batch_outs, skip_special_tokens=True)
            return decoded
        return get_answers
    
    @info_message('Testing model and writing results')
    def test(self):
        if not all([self._data, self._model, self._tokenizer]):
            raise ValueError('Model and data must be prepared')
        
        test_feature_name = 'test' if 'answer' not in self.data['test'].features else 'test-no-ans'
        answers, questions, expected = get_answered_questions(self.data[test_feature_name], self._get_questions_answer_retriever(), self._test_batch_size)

        if not expected and 'answer' in self.data['test'].features:
            expected = self.data['test']['answer']
        write_results_to_tsv(self._results_base_path, questions, answers, expected)
