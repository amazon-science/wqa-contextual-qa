'''

Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
SPDX-License-Identifier: CC-BY-NC-4.0

'''


import unittest
import os

from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from coala import AS2Trainer
from coala.datasets import AS2BaseDataset, AS2LocalDataset, AS2LocalOrdDataset
from coala.evaluation import report, precision_curve
from coala.loader import read_dataset
from coala.models import BaseModelForAS2, LocalModelForAS2, LocalOrdModelForAS2
from coala.utils import get_batch_padding_as2


class TestBasePipeline(unittest.TestCase):
    # Differently from other tests, this code only checks the syntactic correctness.
    # We will improve this aspect in the near future

    baseModel = BaseModelForAS2
    baseDataset = AS2BaseDataset
    tt_size = 2
    
    def	setUp(self):
        self.data = read_dataset('tests/toydataset.csv')[:30]
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.model = self.baseModel('bert-base-uncased')
        self.dataset = self.baseDataset(self.data, self.tokenizer, max_seq_len=32)
        print (self.dataset.labels)
        padding_fnc = get_batch_padding_as2(self.tokenizer.pad_token_id)
        self.dataloader = DataLoader(self.dataset, batch_size=2, collate_fn=padding_fnc, shuffle=False)

    def test_trainer(self):
        trainer = AS2Trainer(self.model, debug=True, device='cpu', save_path='./tmpmodel.pt').fit(self.dataloader, self.dataloader)
        os.remove('./tmpmodel.pt')

    def test_inference(self):
        scores = AS2Trainer(self.model).predict(self.dataloader)
        labels = self.dataset.labels
        questions = self.dataset.questions
        results = report(labels, scores, questions)
        pr, cov, thr = precision_curve(labels, scores, questions)
        self.assertEqual(len(pr), len(cov))
        self.assertEqual(len(thr), len(pr))
        self.assertEqual(len(set(questions)), len(pr))


        
        

class TestLocalPipeline(TestBasePipeline):
    baseModel = LocalModelForAS2
    baseDataset = AS2LocalDataset
    tt_size = 4


class TestLocalOrdPipeline(TestLocalPipeline):
    baseModel = LocalOrdModelForAS2
    baseDataset = AS2LocalOrdDataset
    tt_size = 4

    def	test_precision(self):
        labels = [1,1,0,0,0,0]
        scores = [.9, .7, .5, .4, .8, .3]
        questions = ['a','b','c','c','d','d']
        pre, cov, thr =	precision_curve(labels, scores, questions)
        self.assertListEqual(pre, [1, 0.5, 2/3, 0.5])
        self.assertListEqual(cov, [.25, .5, .75, 1])
        self.assertListEqual(thr, [.9, .8, .7, .5])

