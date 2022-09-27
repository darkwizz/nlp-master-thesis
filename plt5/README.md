# plT5 Experiments
Here are presented the model description, revisions' details and the results

## The model
[The original paper](https://arxiv.org/abs/1910.10683)  
[One of the later papers](https://arxiv.org/pdf/2002.08910.pdf), which describes the extra techniques which were applied hoping to improve the results  

It is based on Google's T5 (Text-to-Text Transfer Transformer). The original T5 models are much larger (the largest has 11B parameters against 1.2B of `plT5-large`)

**Transfer learning** (in NLP) - an approach when a language model is _pre-trained_ for a more low-level (general) objective on very large amount of text, so during _fine-tuning_ it needs significantly less data to fit to a specific NLP task.


## Checkpoints
[plT5-large model on huggingface](https://huggingface.co/allegro/plt5-large)