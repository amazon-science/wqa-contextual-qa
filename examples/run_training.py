'''

Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
SPDX-License-Identifier: CC-BY-NC-4.0

'''

import sys
sys.path.insert(0, "..")
import torch

import argparse, os
parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model',         default='bert-base',      help='model name or path')
parser.add_argument('-x', '--context',       default='base',           help='context type',
                    choices=['base','local', 'local-ord', 'positional'])
parser.add_argument('-c', '--cuda',          default=-2,   type=int,   help='assigned gpu id, -1 means all, -2 means CPU')
parser.add_argument('-t', '--training',      required=True,            help='training data file')
parser.add_argument('-v', '--validation',    required=False,           help='validation data file')
parser.add_argument('-e', '--max_epochs',    default=5,    type=int,   help='max number of training epochs')
parser.add_argument('-w', '--workers',       default=0,    type=int,   help='num threads for data generation')
parser.add_argument('-r', '--run',           default=0,    type=int,   help='current run')
parser.add_argument('-b', '--batch_size',    default=32,   type=int,   help='global batch size')
parser.add_argument('-l', '--learningrate',  default=1e-5, type=float, help='learning rate')
parser.add_argument('-s', '--max_seq_length',default=128,  type=int,   help='max sequence length (256 for contextual models is sufficient)')
parser.add_argument('-o', '--output_file',   default=None,             help='output file name')
parser.add_argument('-p', '--patience',      default=5,    type=int,   help='earlystopping epochs')
parser.add_argument('--warmup_peak',         default=.2,   type=float, help='warmup peak in epochs')
parser.add_argument('--debug',               action='store_true',      help='if true, shows debug information')
parser.add_argument('--val_metric',          default='p@1',            help='validation metric',
                    choices=['loss','val_loss','roc_auc','p@1'])
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
    logger.info('The model will be trained on a single GPU (id: %d)' % args.cuda)
    os.environ["CUDA_VISIBLE_DEVICES"]='%d' % args.cuda
elif args.cuda == -1:
    logger.info('The model will be trained on %d GPUs' % torch.cuda.device_count())
    model = torch.nn.DataParallel(model)
elif args.cuda == -2:
    logger.info('The model will be trained on CPUs')
model.to(device)


from coala.loader import read_dataset
data_tr = read_dataset(args.training)
data_va = read_dataset(args.validation)

dataset_tr = dataset_class(data_tr, tokenizer, max_seq_len=args.max_seq_length)
dataset_va = dataset_class(data_va, tokenizer, max_seq_len=args.max_seq_length)

from torch.utils.data import DataLoader
from coala.utils import get_batch_padding_as2
my_padding_fn = get_batch_padding_as2(tokenizer.pad_token_id)
dataloader_tr = DataLoader(dataset_tr, batch_size=args.batch_size, shuffle=True,  num_workers=args.workers, collate_fn=my_padding_fn)
dataloader_va =	DataLoader(dataset_va, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, collate_fn=my_padding_fn)


from transformers import get_linear_schedule_with_warmup
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=args.learningrate)
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps   = int(len(dataloader_tr) * args.warmup_peak),
    num_training_steps = len(dataloader_tr)*args.max_epochs
)


from coala import AS2Trainer as Trainer
trainer = Trainer(
    model,
    optimizer = optimizer,
    scheduler = scheduler,
    epochs    = args.max_epochs,
    patience  = args.patience,
    loss_fct  = criterion,
    val_metric= args.val_metric,
    debug     = args.debug,
    save_path = args.output_file,
    device    = device,
).fit(dataloader_tr, dataloader_va)


    
