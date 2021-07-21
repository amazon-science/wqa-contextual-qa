'''

Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
SPDX-License-Identifier: CC-BY-NC-4.0

'''


import os

import torch
from torch import nn
from transformers import AutoModelForSequenceClassification
from transformers.modeling_utils import PreTrainedModel

from .utils import get_logger

logger = get_logger()


class ModelForAS2(nn.Module):

    def __init__(self, pretrained_model_name_or_path):
        '''load and convert a huggingface pretrained model'''
        super().__init__()
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        
        if os.path.isfile(pretrained_model_name_or_path):
            logger.info('Loading an existing contextual model: %s' % pretrained_model_name_or_path)
            checkpoint = torch.load(pretrained_model_name_or_path, map_location=lambda storage, loc: storage)
            if hasattr(checkpoint, 'module'):
                checkpoint = checkpoint.module
            if hasattr(checkpoint, 'model'):
                checkpoint = checkpoint.model
                assert issubclass(checkpoint.__class__, PreTrainedModel)
            if hasattr(checkpoint, 'transformer'):
                checkpoint = checkpoint.transformer
                
            if checkpoint.config.architectures[0].endswith('ForMaskedLM'):
                logger.warning('The loaded model is converted from MLM to sequence classification. The classification head has to be fine-tuned')
                as2model = AutoModelForSequenceClassification.from_pretrained(checkpoint.config._name_or_path)
                as2encoder = getattr(as2model, checkpoint.config.model_type)
                as2encoder.embeddings = checkpoint.embeddings
                as2encoder.encoder    = checkpoint.encoder
                checkpoint = as2model
            self.transformer = checkpoint
            if self.transformer.config.type_vocab_size != self.type_vocab_size:
                logger.error('The size of the type-token embedding of this ctx model do not match with the size of the loaded model!')
                raise TypeError('The size of the type-token embedding of this ctx model do not match with the size of the loaded model!')
        else:
            logger.info('Loading a pre-trained checkpoint: %s' % pretrained_model_name_or_path)
            self.transformer = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path)
            if self.transformer.config.type_vocab_size < self.type_vocab_size:
                logger.warning('The token-type embedding has only %d row(s). It is replaced with a new embedding layer with %d rows that will be trained from scratch.' % (self.transformer.config.type_vocab_size, self.type_vocab_size))
                encoder = getattr(self.transformer, self.transformer.config.model_type)
                encoder.embeddings.token_type_embeddings = nn.Embedding(self.type_vocab_size, self.transformer.config.hidden_size)
                encoder.embeddings.token_type_embeddings.weight.data.normal_(mean=0.0, std=self.transformer.config.initializer_range)
                self.transformer.config.type_vocab_size = self.type_vocab_size
                

    def forward(self, **kwargs):
        return self.transformer(**kwargs)
                

    def save(self, out_path):
        '''save the model and weights on file'''
        torch.save(self, out_path)

        
    def to_huggingface():
        '''convert the model to huggingface transformers'''
        return self.transformer


class BaseModelForAS2(ModelForAS2):

    type_vocab_size = 2

    def __init__(self, pretrained_huggingface_model_name_or_path):
        super().__init__(pretrained_huggingface_model_name_or_path)



class LocalModelForAS2(BaseModelForAS2):
    type_vocab_size = 4

    
class LocalOrdModelForAS2(LocalModelForAS2):
    pass



