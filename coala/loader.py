'''

Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
SPDX-License-Identifier: CC-BY-NC-4.0

'''

import json

import pandas as pd


def read_dataset(path):
    '''read a dataset for AS2. Accepted formats are .json and .tsv. These files can be compressed (.gz)'''
    if path.endswith('.json') or path.endswith('.json.gz'):
        ds = read_json(path)
    elif path.endswith('.csv') or path.endswith('.csv.gz'):
        ds = read_csv(path)
    else:
        raise ValueError('File extension not recognized: %s' % ext)
    return ds




def read_json(path):
    data = []
    with open(path, 'r') as ifile:
        raw_data = json.load(ifile)
    for row in raw_data:
        question = row['question']
        answers = row['answers']
        for answer in answers:
            current    = answer['answer']
            previous   = answer['prev']
            successive = answer['next']
            label      = int(answer['label'])
            data.append({
                'label'     : label,
                'question'  : question,
                'answer'    : current,
                'previous'  : previous,
                'successive': successive,
            })
    return data

    

def read_csv(path):
    data = pd.read_csv(path,compression='gzip' if path.endswith('.gz') else None)
    data = data.fillna('')
    if 'doc_id' not in data.columns:
        data['doc_id'] = ''
    if 'title' not in data.columns:
        data['title'] = ''
    data = list(data.T.to_dict().values())
    return data
