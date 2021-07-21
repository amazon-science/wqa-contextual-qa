'''

Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
SPDX-License-Identifier: CC-BY-NC-4.0

'''


import random

import torch
from torch.utils.data import Dataset

from . import utils


class AS2Dataset(Dataset):
    '''Dataset base (abstract) class for AS2'''

    def __init__(self,
                 data,
                 tokenizer,
                 max_seq_len = 256,
                 device      = 'cpu',
                 task_label  = 'as2', #as2 or mlm
                 p_mask      = 0.15,
                 **kwargs,
    ):
        #input parameters
        self.data        = data
        self.device      = device
        self.max_seq_len = max_seq_len
        self.tokenizer   = tokenizer
        self.task_label  = task_label
        self.p_mask      = p_mask
        self.kwargs      = kwargs

        #internal fields
        self.cls = self.tokenizer.cls_token_id
        self.sep = self.tokenizer.sep_token_id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        '''internal function used to process an input example'''
        if torch.is_tensor(idx):
            idx = idx.tolist()
        row     = self.data[idx]
        example = self.encode_example(row)
        example = {k:v.to(self.device) for k,v in example.items()}

        if self.task_label == 'as2':
            # binary one-hot label for as2
            label = torch.tensor(utils.single_label_encoder(row['label']), device=self.device)
        elif self.task_label == 'mlm':
            # input ids for mlm
            label = example['input_ids']
            example['input_ids'] = torch.tensor([iid if random.uniform(0.0, 1.0) > self.p_mask else self.tokenizer.mask_token_id for iid in example['input_ids']])
        else:
            raise ValueError('Task label do not recognized. Possible values: as2 or mlm')
        return example, label

    def tokenize_segment(self, segment, max_length):
        '''internal function used to tokenize segments'''
        return self.tokenizer(
            segment,
            add_special_tokens=False,
            return_token_type_ids=None,
            max_length=max_length,
            truncation=True,
            verbose=False
        )['input_ids'][:max_length]

    def encode_example(self, row):
        '''converts a row into a dictionary of sequences representing
           the input for transformer models
        '''
        raise NotImplementedError('This method has to be implemented in the derived classes')

    @property
    def questions(self):
        return [row['question'] for row in self.data]

    @property
    def labels(self):
        return [row['label'] for row in self.data]


class AS2BaseDataset(AS2Dataset):
    '''
       This class represents the simplest AS2 dataset.
       An example is defined as a simple question/candidate pair
       [cls] question [sep] candidate answer
    '''

    def encode_example(self, row):
        ids_q = self.tokenize_segment(row['question'], self.max_seq_len - 2)
        ids_a =	self.tokenize_segment(row['answer'],   self.max_seq_len - 2 - len(ids_q))

        return {
            'input_ids'      : torch.tensor([self.cls] + ids_q + [self.sep] + ids_a),
            'token_type_ids' : torch.tensor([0] * (len(ids_q)+1) + [1] * (len(ids_a)+1)),
            'attention_mask' : torch.tensor([1] * (len(ids_q)+len(ids_a)+2)),
            'position_ids'   : torch.arange(len(ids_q) + len(ids_a) + 2),
        }



class AS2LocalDataset(AS2Dataset):
    '''
       This class defines the AS2 dataset with local context
       The local context is defined as
       [cls] question [sep] prev sentence [sep] answer [sep] successive sentence
       The priority order to fill the 512 tokens is:
         question > answer > prev > successive
    '''

    def encode_example(self, row):
        l = self.max_seq_len - 4
        ids_q = self.tokenize_segment(row['question'],   l)
        ids_a = self.tokenize_segment(row['answer'],     l - len(ids_q))
        ids_p = self.tokenize_segment(row['previous'],   l - len(ids_q) - len(ids_a))
        ids_s = self.tokenize_segment(row['successive'], l - len(ids_q) - len(ids_a) - len(ids_p))

        return {
            'input_ids'      : torch.tensor([self.cls] + ids_q + [self.sep] + ids_p + [self.sep] + ids_a + [self.sep] + ids_s),
            'token_type_ids' : torch.tensor([0] * (len(ids_q)+1) + [1] * (len(ids_p)+1) + [2] * (len(ids_a)+1) + [3] * (len(ids_s)+1)),
            'attention_mask' : torch.tensor([1] * (len(ids_q)+len(ids_p)+len(ids_a)+len(ids_s)+4)),
            'position_ids'   : torch.arange(len(ids_q) + len(ids_p) + len(ids_a) + len(ids_s) + 4),
        }



class AS2LocalOrdDataset(AS2Dataset):
    '''
       This class defines the AS2 dataset with local (re-ordered) context
       The local ordered context is defined as
       [cls] question [sep] answer [sep] prev sentence [sep] successive sentence
       The priority order to fill the 512 tokens is:
         question > answer > prev > successive
    '''

    def encode_example(self, row):
        l = self.max_seq_len - 4
        ids_q = self.tokenize_segment(row['question'],   l)
        ids_a = self.tokenize_segment(row['answer'],     l - len(ids_q))
        ids_p = self.tokenize_segment(row['previous'],   l - len(ids_q) - len(ids_a))
        ids_s = self.tokenize_segment(row['successive'], l - len(ids_q) - len(ids_a) - len(ids_p))

        return {
            'input_ids'      : torch.tensor([self.cls] + ids_q + [self.sep] + ids_a + [self.sep] + ids_p + [self.sep] + ids_s),
            'token_type_ids' : torch.tensor([0] * (len(ids_q)+1) + [1] * (len(ids_a)+1) + [2] * (len(ids_p)+1) + [3] * (len(ids_s)+1)),
            'attention_mask' : torch.tensor([1] * (len(ids_q)+len(ids_a)+len(ids_p)+len(ids_s)+4)),
            'position_ids'   : torch.arange(len(ids_q) + len(ids_a) + len(ids_p) + len(ids_s) + 4),
        }
