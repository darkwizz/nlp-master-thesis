# Closed-Book QA

Python scripts to run model training and testing for different revisions.  
Artur Sokol

---


### Briefly about the problem
Closed-book Question Answering means that a trained model in order to answer on a question does not need any provided context - it uses its own knowledge:

| ![MachineReadingComprehension](./resources/open-book-QA.png) | ![CloseBookQA](./resources/close-book-QA.png) |
| :----: | :----: |
| Machine Reading Comprehension (MRC) | Closed-book QA |

On the contrary, Open-book QA uses an external source of knowledge (e.g. a DB of Wikipedia articles in raw format). In this case an Open-book QA model searches for _k_ best matching documents and then uses MRC to extract the answer


### General help
```
usage: main.py [-h] -n {plt5,papugapt2} -r {baseline} -b BASE_DATA_PATH -t TOKENIZER_PATH -m MODEL_PATH
               [-q QUESTION_MAX_LENGTH] [-a ANSWER_MAX_LENGTH] [-o TRAINING_OUTPUT_DIR]
               [--test-batch-size TEST_BATCH_SIZE] [--test-max-length TEST_MAX_LENGTH] [--results-dir RESULTS_DIR]
               [-s] [-S] [-f]

optional arguments:
  -h, --help            show this help message and exit
  -n {plt5,papugapt2}, --model_name {plt5,papugapt2}
                        a model name to use. In this master thesis most likely T5 and GPT2 only will be considered
  -r {baseline}, --revision {baseline}
                        model revision, the names might be weird as they should follow Python naming conventions
  -b BASE_DATA_PATH, --base-data-path BASE_DATA_PATH
                        base path to all data. Every kind of data should be placed there. Right now their names are hardcoded: test-A, test-B, train
  -t TOKENIZER_PATH, --tokenizer-path TOKENIZER_PATH
                        can be a name on huggingface or a local path
  -m MODEL_PATH, --model-path MODEL_PATH
                        can be a name on huggingface or a local path. If a local path, then it's most likely already fine-tuned and it's good to use `-S` flag too
  -q QUESTION_MAX_LENGTH, --question-max-length QUESTION_MAX_LENGTH, --q-maxlen QUESTION_MAX_LENGTH
                        max length for a tokenized question, during training
  -a ANSWER_MAX_LENGTH, --answer-max-length ANSWER_MAX_LENGTH, --a-maxlen ANSWER_MAX_LENGTH
                        max length for a tokenized answer, during training
  -o TRAINING_OUTPUT_DIR, --training-output-dir TRAINING_OUTPUT_DIR
                        path to outputs during model training. Be careful, as the default path is one level up than
                        models and their revisions, thus would be overwritten for every program run
  --test-batch-size TEST_BATCH_SIZE
                        batch size during testing
  --test-max-length TEST_MAX_LENGTH
                        max length necessary to create a pytorch tensor during testing
  --results-dir RESULTS_DIR
                        the name of a directory where results of testing would be stored. Its path is combined of the current working directory, model name (-n parameter) and revision (-r)
  -s, --save-pretrained
                        a flag to signal whether to save a fine-tuned model locally. If set then the model will be saved under "./{model_name}/{revision}/trained-model"
  -S, --skip-training
                        a model is not going to be trained
  -f, --few-shot        for decoder-based models performs testing in a few-shot way. If set for a model name/revision which does not have implemented few-shot, then the flag is just ignored
```

### Note about the environment
When running plT5 one more library should be installed:
```bash
$ pip install sentencepiece
```


### Examples

for plT5:
```
python main.py -n plt5 -r baseline -b code/data -t allegro/plt5-small -m allegro/plt5-small --results-dir ./plt5-results
```
or for papuGaPT2:
```
python main.py -S -n papugapt2 -r baseline -q 45 -a 10 -b data -t dkleczek/papuGaPT2 -m dkleczek/papuGaPT2 --results-dir papugapt2-results --test-batch-size 250 --few-shot
```

### Results
For results description (metrics, generated data by models) go to README.md under respective model names.


### Evaluation
For this purpose [geval](https://gitlab.com/filipg/geval) is used, because the testing data is taken from [PolEval-2021 task 4](https://github.com/poleval/2021-question-answering/tree/secret), where geval was used for evaluation. Moreover the tool is easy to use and it has a lot of different supported metrics.

```bash
$ ./geval --list-metrics  # outputs the complete list of available metrics and their description
$ ./geval -t <path-to-results-dir> --metric GLEU --metric Accuracy [--metric ...]  # evaluate results
$ ./geval -t <path-to-results-dir> --alt-metric <metric> --line-by-line --reverse-sort | less  # inspect every actual-expected pair sorted by the most accurate
```
