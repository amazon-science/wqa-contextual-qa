'''

Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
SPDX-License-Identifier: CC-BY-NC-4.0

'''

import sys
sys.path.insert(0, "..")
import torch

import argparse, os
parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model',         default='roberta-base',   help='model path')
parser.add_argument('-x', '--context',       default='base',           help='context type',
                    choices=['base','local', 'local-ord', 'positional'])
parser.add_argument('-c', '--cuda',          default=-2,   type=int,   help='assigned gpu id, -1 means all, -2 means CPU')
parser.add_argument('-t', '--test',          required=True,            help='test data file')
parser.add_argument('-w', '--workers',       default=0,    type=int,   help='num threads for data generation')
parser.add_argument('-b', '--batch_size',    default=32,   type=int,   help='global batch size')
parser.add_argument('-s', '--max_seq_length',default=128,  type=int,   help='max sequence length (256 for contextual models is sufficient)')
parser.add_argument('-o', '--output_file',   default=None,             help='output path if you want to store the answers and scores')
parser.add_argument('--debug',               action='store_true',      help='if true, shows debug information')
args = parser.parse_args()
print(args)


from coala.utils import get_logger
logger = get_logger(args.debug)
logger.info('This script can be used to fine-tune a contextual AS2 model')


from coala.utils import dispatcher
#get the right model given the context type
model_class, dataset_class = dispatcher(args.context)
model = model_class(args.model)
model_name = model.transformer.config._name_or_path #e.g. roberta-base, bert-base-uncased

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
logger.info('Using %s tokenizer' % model_name) 

device = torch.device('cpu' if args.cuda==-2 else 'cuda')
if args.cuda >=0:
    logger.info('Predictions will be computed on a single GPU (id: %d)' % args.cuda)
    os.environ["CUDA_VISIBLE_DEVICES"]='%d' % args.cuda
elif args.cuda == -1:
    logger.info('Predictions will be computed on %d GPUs' % torch.cuda.device_count())
    model = torch.nn.DataParallel(model)
elif args.cuda == -2:
    logger.info('Predictions will be computed on CPUs')
model.to(device)

from coala.loader import read_dataset
data_te = read_dataset(args.test)
dataset_te = dataset_class(data_te, tokenizer, max_seq_len=args.max_seq_length)

from torch.utils.data import DataLoader
from coala.utils import get_batch_padding_as2
my_padding_fn = get_batch_padding_as2(tokenizer.pad_token_id)
dataloader_te = DataLoader(dataset_te, batch_size=args.batch_size, shuffle=False,  num_workers=args.workers, collate_fn=my_padding_fn)


from coala import AS2Trainer as Trainer
scores = Trainer(model, device=device).predict(dataloader_te)
labels    = dataset_te.labels
questions = dataset_te.questions

from coala.evaluation import report
results = report(labels, scores, questions)
logger.info('Evaluation report:')
logger.info('  ' + ', '.join('%s:%d' % (k,v) if type(v)==int else '%s:%.4f' % (k,v) for k,v in results.items()))

from coala.evaluation import precision_curve
pre,cov, thr = precision_curve(labels, scores, questions)
