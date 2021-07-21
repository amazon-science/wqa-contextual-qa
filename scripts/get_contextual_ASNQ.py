'''

Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
SPDX-License-Identifier: CC-BY-NC-4.0

'''

import csv
import gzip
import hashlib
import json
import os
import re
import sys

from nltk import word_tokenize
from nltk.tokenize import sent_tokenize

nq_path_v1_split = sys.argv[1]
asnq_pairs_path = sys.argv[2]
path_out_candidates = sys.argv[3]
path_out_documents  = sys.argv[4]

clean_from_string = lambda s: re.sub(r'[^\w\s]', '', ' '.join(word_tokenize(s))).lower().replace('  ',' ')
clean_from_list   = lambda l: re.sub(r'[^\w\s]', '', ' '.join(l)).lower().replace('  ',' ')

asnq = []
with open(asnq_pairs_path, 'r') as ifile:
    reader = csv.reader(ifile, delimiter='\t')
    for row in reader:
        asnq.append({
            'question': row[0],
            'answer': row[1],
            'label': row[2]})

all_asnq_questions = {row['question'] for row in asnq}
question_to_docid = dict()
alldocs = dict()
nq = dict()

for fi in os.listdir(nq_path_v1_split):
    
    fullpath = os.path.join(nq_path_v1_split, fi)
    with gzip.open(fullpath, 'r') as ifile:
        for j,row in enumerate(ifile):
            row      = json.loads(row)
            question = row['question_text'].lower()
            if question not in all_asnq_questions:
                continue
            
            text = ' '.join([ w['token'] for w in  row['document_tokens'] if not w['html_token'] ])
            doc_id = hashlib.sha3_512(repr(row['question_text'] + text).encode()).hexdigest()
            alldocs[doc_id] = {
                'doc_id': doc_id,
                'content': text,
                'title': row['document_title'],
                'domain': None,
                'url': row['document_url'],
            }
            question_to_docid[question] = doc_id

            sentences = sent_tokenize(text)                                                                                                                                
            clean_sentences = [clean_from_string(s) for s in sentences]
            nq[question] = {'sentences': sentences, 'clean_sentences': clean_sentences}

            if not (j%20):
                print (fi,j)
                sys.stdout.flush()
            del row



print (len(all_asnq_questions), len(alldocs), len(nq))
assert len(all_asnq_questions) == len(alldocs) == len(nq)
with open(path_out_documents, 'w') as ofile:
    json.dump(alldocs, ofile)



#create triplets
hits = 0
positives = 0
with open(path_out_candidates, 'w', encoding='utf-8') as ofile:
    writer = csv.DictWriter(ofile, fieldnames=['question','answer','previous','successive','label', 'title', 'doc_id'])
    writer.writeheader()

    for row in asnq:
        q = row['question']
        answer     = row['answer']
        clean_sent = clean_from_string(row['answer'])

        doc_id = question_to_docid[q]
        title = alldocs[doc_id]['title']
        processed_row = {
            'question':q,
            'answer': answer,
            'previous': '',
            'successive': '',
            'title': title,
            'doc_id': doc_id,
            'label': int(row['label'])
        }
        positives += processed_row['label'] == 1

        if clean_sent in nq[q]['clean_sentences']:
            
            id_sent    = nq[q]['clean_sentences'].index(clean_sent)
            processed_row['previous']   = '' if id_sent == 0 else nq[q]['sentences'][id_sent-1]
            processed_row['successive'] = '' if id_sent == len(nq[q]['sentences'])-1 else nq[q]['sentences'][id_sent+1]
            hits += 1
        writer.writerow(processed_row)

        

print ('Pre-processing done')
print ('  %d q/a pairs' % len(asnq))
print ('  %d questions' % len(all_asnq_questions))
print ('  %d positive answers' % positives)
print ('  %d context warnings' % (len(asnq)-hits))
print ('  %d documents' % len(alldocs))
