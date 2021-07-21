# Contextual Answer Sentence Selection

[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)
[![Python 3.8](https://img.shields.io/badge/python-3.8-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![Python 3.7](https://img.shields.io/badge/python-3.7-blue.svg)](https://www.python.org/downloads/release/python-370/)



**Coala** is a python package for Contextual Answer Sentence Selection.


Answer Sentence Selection (AS2) is a sub-task of question answering, and it aims at finding the sentence containing the answer for a given input question from a pool of possible candidate sentences (e.g. retrieved from a search engine).

In our contextual AS2 task, we provide models that leverage *contextual* information coming from the documents (or the paragraphs) containing the candidate answers.

If you need further information, you can see our recent papers [1,2].


- [1] Ivano Lauriola and Alessandro Moschitti: *"[Answer sentence selection using local and global context in transformer models](https://www.amazon.science/publications/answer-sentence-selection-using-local-and-global-context-in-transformer-models)"*, ECIR 2021.
- [2] Rujun Han, Luca Soldaini, and Alessandro Moschitti: *"[Modeling Context in Answer Sentence Selection Systems on a Latency Budget](https://www.amazon.science/publications/modeling-context-in-answer-sentence-selection-systems-on-a-latency-budget)"*, EACL 2021.

This code requires `torch 1.7` and `transformers 3.5`


## Cite

If you use this code for scientific research, please cite the following paper:

```
@inproceedings{lauriola2021answer,
  title={Answer sentence selection using local and global context in transformer models},
  author={Lauriola, Ivano and Moschitti, Alessandro},
  year={2021},
  organization={ECIR}
}
```



## Introduction & main use cases

In the following we show the main use cases of this library. You can find additional examples in the `examples` folder.


### Data


Let's start from **data**.

We provide a script to convert SQuAD-2.0 from Machine Reading to AS2.

```
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json

python <repository-path>/scripts/convert_squad2.0-to-AS2.py train-v2.0.json squad2.0-train.csv squad2.0-train-docs.json
python <repository-path>/scripts/convert_squad2.0-to-AS2.py dev-v2.0.json squad2.0-dev.csv squad2.0-dev-docs.json
```

The code generates two files for training and two files for development. These files contain AS2 examples (q/a/ctx) and paragraphs to run experiments involving global contexts.


The same can be done to pre-process NQ.

Firstly, download the raw `NQ` dataset [here](https://ai.google.com/research/NaturalQuestions) (we need raw files to build contextual data).
Then, download `ASNQ`, a version of `NQ` designed for AS2

```
https://wqa-public.s3.amazonaws.com/tanda-aaai-2020/data/asnq.tar
tar -xvf asnq.tar
```

You should find two files in a folder `data/asnq/`, namely `train.tsv` and `test.tsv`
Now, you can run the script to add context, that is

```
python get_contextual_ASNQ.py <PATH_NQ>/v1.0/dev data/asnq/dev.tsv asnq-dev.csv asnq-dev-docs.json
python get_contextual_ASNQ.py <PATH_NQ>/v1.0/train data/asnq/train.tsv asnq-train.csv asnq-train-docs.json
```


We consider input observations as a list of dictionaries with fields:

| Field | Description | Mandatory | Example |
| :---- | :---------- | :-------: | :-------|
| `question` | the asked question | yes | Who is Pippo Franco? | 
| `answer`   | a candidate answer sentence | yes | Pippo Franco is an Italian actor |
| `previous` | the sentence that preceeds the candidate | yes | - |
| `successive` | the sentence next to the candidate | yes | He was born in Rome|
| `title` | the title of the document containing the candidate | no | Bibliography of Pippo Franco|
| `doc_id`| an univoke index of the document | no | 32531672372
| `label` | the output class, 1 good answer 0 else |  yes | 1 

If your dataset is stored in a `csv` file with the columns described in the previous table, you can simply read it 

```
from coala.loader import read_dataset
data = read_dataset(path_to_my_file)
```

Note that, in the simple non-contextualized AS2 task you only need the fields `question`, `answer`, and `label`.

We provide specialyzed **datasets** to handle contextual AS2 data, e.g:

```
from transformers import RobertaTokenizer
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

from coala.datasets import AS2LocalDataset
dataset = AS2LocalDataset(
	data, 				# our q/a/ctx tuples
	tokenizer, 			# the tokenizer
	max_seq_len=256, 	# the maximum length of the encoded text
	task_label='as2')	# as2 or mlm
```

We can also play with our custom **dataloaders** to handle batches and datasets. In the following example, we use a trivial PyTorch dataloader

```
from torch.utils.data import DataLoader
from coala.utils import batch_padding_as2

my_padding_fn = batch_padding_as2(tokenizer.pad_token_id)
dataloader = DataLoader(
	dataset,
	batch_size  = 64,		# global batch size
	shuffle     = True, 	# this should be False for validation/test dataloader!!!
	num_workers = workers,	# number of threads
	collate_fn  = my_padding_fn)	# custom collate function to create batches
```




### Models


Coala defines several **models** for AS2 based on Transformer architecture. These models can be imported from `coala.models`, and they are summarized in the following table.

| Model Class | Description | Encoded text |
| :---------- | :---------- | :-------: |
| `BaseModelForAS2` | The simple model that encodes q/a pairs without context | *[CLS] question [SEP] answer* | 
| `LocalModelForAS2` | An extended model encoding the surrounding sentences of a candidate answer | *[CLS] question [SEP] previous [SEP] answer [SEP] successive* |
| `LocalOrdModelForAS2` | A model with re-ordered local context. Suitable for low-resource scenarios | *[CLS] question [SEP] answer [SEP] previous [SEP] successive* |

The simplest way to use a Contextual AS2 model is starting from pre-trained public checkpoints from HuggingFace

```
from coala.models import LocalModelForAS2
model = LocalModelForAS2('roberta-base')
# or
model = LocalModelForAS2('path_to_model_or_dir')
```



We provide specialyzed wrappers to train our models with a few coding lines. Our wrapper can be configured with various parameters as showed in the following example

```
from coala import AS2Trainer as Trainer
from transformers import get_constant_schedule
import torch

trainer = Trainer(
	model,				# our contextual AS2 model
	optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-5),
	scheduler = get_constant_schedule(self.optimizer),
	epochs    = 5,		# maximum number of epochs
	patience  = 2,		# early stop after 2 epochs
	loss_fct  = torch.nn.CrossEntropyLoss(),
	val_metric= 'p@1',	# the metric used for selecting the best model
	debug     = False,  
	save_path = None,	# the name of the output model
	device    = 'cpu')
```

Now, we are ready to train our contextual AS2 model!

```
trainer = trainer.fit(dataset, dataset_validation)
```

The trainer shows some information each epoch, including the computational time, loss, and validation performance. 
If you specified the parameter `save_path`, you should find a `pt` file in that location containing the fine-tuned model.



### Evaluation 

After training, you can simply load a fine-tuned contextual model with:


```
model = torch.load(my_contextual_path.pt)
```

We can use our `Trainer` to predict the label of new examples (be sure that the parameter `shuffle` of the dataloader is set to `False`)

```
predictions = Trainer(model).predict(dataset_test)
```

For the sake of usability, we provide mechanisms to compute a few metrics and statistics

```
from coala.evaluation import report

labels = dataset_test.labels
questions = dataset_test.questions

print (report(labels, predictions, questions))
>{'examples':2000, 'questions':150, 'AUC': 91.77, 'p@1': 80.23, 'MAP':86.19, ...}
```


We can also print a simple precision-coverage curve

```
from coala.evaluation import precision_curve
precision, ans_rate, thresholds = precision_curve(labels, predictions, questions)
```

The function `precision_curve ` returns the points of the precision/answer-rate curve as triplets with values precision-answerrate-threshold.




## Work in progress
The content of this library represents a small portion of the current research in contextual AS2. 

In the near future, we will add additional modules to cover different types of contexts (including global contexts). 

If you have feedbacks, requests, or bug reports, please let us know <lauivano@amazon.com>
