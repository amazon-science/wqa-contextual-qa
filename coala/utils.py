'''

Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
SPDX-License-Identifier: CC-BY-NC-4.0

'''

import logging
import logging.config
import os.path

import torch


def get_logger(debug=False):
    '''return the main logger'''
    fileconfig = os.path.join(os.path.dirname(__file__), 'logging.conf')
    logging.config.fileConfig(fileconfig)
    logger = logging.getLogger('AS2') # if not debug else 'AS2-debug')
    if debug:
        logger.setLevel(logging.DEBUG)
    return logger


def dispatcher(context):
    '''return the correct model and dataset given the context'''
    from . import models, datasets
    if context == 'base':
        return (models.BaseModelForAS2, datasets.AS2BaseDataset)
    elif context == 'local':
        return (models.LocalModelForAS2, datasets.AS2LocalDataset)
    elif context == 'local-ord':
        return (models.LocalOrdModelForAS2, datasets.AS2LocalOrdDataset)
    else:
        raise ValueError('Context type not recognized')



def single_label_encoder(y, positive={1}):
    '''binary one-hot encoding for a single label
       Y       : the label
       positive: the set of positive class values
    '''
    return 1 if int(y) in positive else 0



def get_batch_padding_as2(pad_token_id):
    return lambda batch : batch_padding_as2(batch, pad_token_id)



def batch_padding_as2(batch, token):
    '''this function applies the minimum padding to a given batch
       token : the padding value
    '''

    examples = [t[0] for t in batch]
    labels   = [t[1] for t in batch]
    keys = batch[0][0].keys()

    sequences = [t['input_ids'] for t in examples]
    mask      = [t['attention_mask'] for t in examples]
    typetoken = [t['token_type_ids'] for t in examples]
    position  = [t['position_ids'] for t in examples]

    labels = torch.tensor(labels)
    newexamples = {
        'input_ids'     : torch.nn.utils.rnn.pad_sequence(sequences, padding_value=token, batch_first=False).T,
        'token_type_ids': torch.nn.utils.rnn.pad_sequence(typetoken, padding_value=token, batch_first=False).T,
        'attention_mask': torch.nn.utils.rnn.pad_sequence(mask,      padding_value=0,     batch_first=False).T,
        'position_ids'  : torch.nn.utils.rnn.pad_sequence(position,  padding_value=0,     batch_first=False).T,
    }    
    return newexamples, labels
