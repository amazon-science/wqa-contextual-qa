'''

Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
SPDX-License-Identifier: CC-BY-NC-4.0

'''


import unittest

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from coala.datasets import AS2BaseDataset, AS2LocalDataset, AS2LocalOrdDataset
from coala.models import BaseModelForAS2, LocalModelForAS2, LocalOrdModelForAS2
from coala.utils import get_batch_padding_as2


class TestBaseDataset(unittest.TestCase):
    baseClass = AS2BaseDataset
    baseModel = BaseModelForAS2
    tt_size = 2
    
    def	setUp(self):
        self.data = [
            {'question':'who is Pippo Franco?',
             'answer':'this is a random sentence',
             'previous':'this is a random previous sentence',
             'successive':'this is a random successive sentence',
             'title':'',
             'label':0},
            {'question':'who is Pippo Franco?',
             'answer':'Pippo Franco is an Italian actor',
             'previous':'',
             'successive':'He was born in Rome',
             'title':'Hello! I\'m a title!',
             'label':1},
        ]

    def	test_parameters(self):
        tokenizer = AutoTokenizer.from_pretrained('roberta-base')
        dataset = self.baseClass(self.data, tokenizer, max_seq_len=32, task='as2', p_mask=.15, device='cpu')

    def test_getitem(self):
        tokenizer = AutoTokenizer.from_pretrained('roberta-base')
        dataset = self.baseClass(self.data, tokenizer, max_seq_len=32)
        example, label = dataset[0]
        self.assertEqual(type(example), dict)
        self.assertEqual(type(label), torch.Tensor)
        self.assertSetEqual(set(example.keys()), {'input_ids', 'token_type_ids', 'attention_mask', 'position_ids'})
        self.assertEqual(example['input_ids'].size(), example['token_type_ids'].size())
        self.assertEqual(example['input_ids'].size(), example['attention_mask'].size())
        self.assertEqual(example['input_ids'].size(), example['position_ids'].size())
        self.assertEqual(example['attention_mask'].max().item(), 1)
        print (set(example['token_type_ids'].tolist()), set(range(self.tt_size)))
        self.assertSetEqual(set(example['token_type_ids'].tolist()), set(range(self.tt_size)))

    def test_maxlength(self):
        tokenizer = AutoTokenizer.from_pretrained('roberta-base')
        dataset = self.baseClass(self.data, tokenizer, max_seq_len=10)
        example, label = dataset[0]
        self.assertEqual(example['input_ids'].size()[0], 10)

    def test_batch(self):
        model = self.baseModel('bert-base-uncased')
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        dataset = self.baseClass(self.data, tokenizer, max_seq_len=32)
        padding_fnc = get_batch_padding_as2(tokenizer.pad_token_id)
        dataloader = DataLoader(dataset, batch_size=2, collate_fn=padding_fnc)
        examples, labels = next(iter(dataloader))
        device = 'cpu'
        model.to(device)
        logits = model(**examples)[0]
        self.assertEqual(list(logits.size()), [2,2])

    def test_attributes(self):
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        dataset = self.baseClass(self.data, tokenizer, max_seq_len=32)
        self.assertEqual(len(dataset.labels), 2)
        self.assertEqual(len(dataset.questions), 2)
        self.assertEqual(len(dataset.data), 2)
        

class TestLocalDataset(TestBaseDataset):
    baseClass = AS2LocalDataset
    baseModel = LocalModelForAS2
    tt_size = 4

    def test_encoding(self):
        model = self.baseModel('bert-base-uncased')
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        dataset = self.baseClass(self.data, tokenizer, max_seq_len=512)
        self.assertEqual(dataset[1][0]['position_ids'].bincount()[1], 1) #empty prev


class TestLocalOrdDataset(TestLocalDataset):
    baseClass = AS2LocalOrdDataset
    baseModel = LocalOrdModelForAS2
    tt_size = 4

    def test_encoding(self):
        model = self.baseModel('bert-base-uncased')
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        dataset = self.baseClass(self.data, tokenizer, max_seq_len=512)
        self.assertEqual(dataset[1][0]['position_ids'].bincount()[2], 1) #empty prev
        self.assertEqual(dataset[0][0]['position_ids'].bincount()[4], 1) #empty title

