'''

Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
SPDX-License-Identifier: CC-BY-NC-4.0

'''


import sys
sys.path.insert(0, "..")

import unittest
import torch
import os
import shutil

from coala.models import BaseModelForAS2, LocalModelForAS2, LocalOrdModelForAS2, PositionalModelForAS2
from transformers import AutoModelForSequenceClassification


def equal_parameters(model1, model2):
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        if p1.data.ne(p2.data).sum() > 0:
            return False
    return True


class TestBaseModel(unittest.TestCase):
    def setUp(self):
        self.baseClass = BaseModelForAS2
        self.tt_size = 2

    def test_initialization(self):
        model = self.baseClass('bert-base-uncased')
        #self.assertListEqual(model.transformer.bert.embeddings.token_type_embeddings.weights.size(), [self.tt_size, 768])
        self.assertEqual(model.transformer.config.model_type, 'bert')
        self.assertEqual(model.transformer.config.type_vocab_size, self.tt_size)
        model =	self.baseClass('google/electra-base-discriminator')
        #self.assertListEqual(model.transformer.electra.embeddings.token_type_embeddings.weights.size(), [self.tt_size, 768])
        self.assertEqual(model.transformer.config.model_type, 'electra')
        self.assertEqual(model.transformer.config.type_vocab_size, self.tt_size)
        #test weights
        hf_model = AutoModelForSequenceClassification.from_pretrained('google/electra-base-discriminator')
        self.assertTrue(equal_parameters(model.transformer.electra.encoder, hf_model.electra.encoder))
        

    def test_load(self):
        os.mkdir('./tmp')
        model = self.baseClass('google/electra-base-discriminator')
        torch.save(model, './tmp/model_s1.pt')
        model.save('./tmp/model_s2.pt')
        model_l1 = torch.load('./tmp/model_s1.pt')
        model_l2 = torch.load('./tmp/model_s2.pt')
        model_l3 = self.baseClass('./tmp/model_s1.pt')
        model_l4 = self.baseClass('./tmp/model_s2.pt')
        self.assertTrue(equal_parameters(model, model_l1))
        self.assertTrue(equal_parameters(model, model_l2))
        self.assertTrue(equal_parameters(model, model_l3))
        self.assertTrue(equal_parameters(model, model_l4))
        self.assertTrue(equal_parameters(model, model.transformer))
        shutil.rmtree('./tmp')
        
    def test_load_mlm(self):
        pass


    
class TestLocalModel(TestBaseModel):
    def setUp(self):
        self.baseClass = LocalModelForAS2
        self.tt_size = 4


class TestLocalOrdModel(TestLocalModel):
    def setUp(self):
        self.baseClass = LocalOrdModelForAS2
        self.tt_size = 4


class TestPositionalModel(TestLocalModel):

    def setUp(self):
        self.baseClass = PositionalModelForAS2
        self.tt_size = 5


if __name__ == '__main__':
    unittest.main()
    

