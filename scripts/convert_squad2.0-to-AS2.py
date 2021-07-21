'''

Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
SPDX-License-Identifier: CC-BY-NC-4.0

'''


import csv
import hashlib
import json
import nltk
import sys

path_in = sys.argv[1]
path_out_candidates = sys.argv[2]
path_out_documents  = sys.argv[3]

sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')



with open(path_in, 'r') as ifile:
    alljson = json.load(ifile)
data = alljson['data']

contexts_retrieved = 0
errors = 0
positives = 0
tuples = []
documents = dict()
ss = 0
for topic in data:
    title = topic['title']
    paragraphs = topic['paragraphs']
    
    for paragraph in paragraphs:
        context = paragraph['context']
        qas = paragraph['qas']
        doc_id = hashlib.sha3_512(repr(context).encode()).hexdigest()
        documents[doc_id] = {
            'doc_id': doc_id,
            'title': title,
            'content': context,
            'domain': None,
            'url': None,
            }
        sentences_idx = list(sent_detector.span_tokenize(context))
        sentences = [context[start:end] for (start, end) in sentences_idx]

        #fix tokenization issues
        answers = [_ans for _qa in qas for _ans in _qa['answers']]
        conc_indexes = {i for _ans in answers for i,_sent in enumerate(sentences)
                        if sentences_idx[i][0] <= _ans['answer_start'] <= sentences_idx[i][1] and _ans['text'] not in _sent}
        conc_indexes = sorted(list(conc_indexes))[::-1]
        
        for ii in conc_indexes:
            sentences[ii] += ' '+sentences[ii+1]
            del sentences[ii+1]
            sentences_idx[ii] = (sentences_idx[ii][0], sentences_idx[ii+1][1])
            del sentences_idx[ii+1]
        
        
        for qa in qas:
            question = qa['question']

            for i, (sentence, (start, end)) in enumerate(zip(sentences, sentences_idx)):
                label = 0
                
                for answer in qa['answers']:
                    if start <= answer['answer_start'] <= end:
                        label = 1
                        positives += 1
                        if answer['text'] not in sentence:
                            errors += 1
                        break
                tuples.append({
                    'question': question,
                    'answer': sentence,
                    'previous': sentences[i-1] if i else '',
                    'successive': sentences[i+1] if i < len(sentences)-1 else '',
                    'doc_id': doc_id,
                    'title': title,
                    'label': label,
                    })


with open(path_out_candidates, 'w') as ofile:
    writer = csv.DictWriter(ofile, fieldnames=tuples[0].keys())
    writer.writeheader()
    for row in tuples:
        writer.writerow(row)

with open(path_out_documents, 'w') as ofile:
    json.dump(documents, ofile)

print ('Conversion done')
print ('  %d q/a pairs' % len(tuples))
print ('  %d questions' % len({t['question'] for t in tuples}))
print ('  %d positive answers' % positives)
print ('  %d tokenization warnings' % errors)
print ('  %d documents/paragraphs' % len(documents))


