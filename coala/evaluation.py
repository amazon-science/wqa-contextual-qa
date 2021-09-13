'''

Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
SPDX-License-Identifier: CC-BY-NC-4.0

'''


import numpy as np
from sklearn.metrics import roc_auc_score



def report(labels, scores, questions, return_metadata=True):
    '''computes various statistics and metrics'''
    results = {}
    if len(np.unique(labels)) == 2:
        results['roc_auc'] = roc_auc_score(labels, scores)
    dict_questions = {q:{'labels':[], 'scores':[]} for q in questions}
    for label, score, q in zip(labels, scores, questions):
        dict_questions[q]['labels'].append(label)
        dict_questions[q]['scores'].append(score)
    results['P@1'] = np.mean([precision_at_k(v['labels'], v['scores'], 1) for v in dict_questions.values()])
    results['MAP'] = np.mean([average_precision(v['labels'], v['scores']) for v in dict_questions.values()])
    results['MRR'] = np.mean([reciprocal_rank(v['labels'], v['scores']) for v in dict_questions.values()])
    results['HIT@3'] = np.mean([hit_at_k(v['labels'], v['scores'], 3) for v in dict_questions.values()])
    results['HIT@5'] = np.mean([hit_at_k(v['labels'], v['scores'], 5) for v in dict_questions.values()])
    results['AUPC'] = area_under_precision_curve(labels, scores, questions)
    if return_metadata:
        results['questions'] = len(dict_questions)
        results['examples']  = len(labels)
    return results



### ranking metrics for single question (question + all candidates)

def precision_at_k(labels, scores, k, pos_label=1):
    '''precision at k given a question'''
    scores, labels = zip(*sorted(zip(scores, labels),reverse=True))
    n =len(scores)
    n_pos = labels[:k].count(pos_label)
    return 0 if not n_pos else n_pos / k


def hit_at_k(labels, scores, k, pos_label=1):
    '''hit at k given a question'''
    scores, labels = zip(*sorted(zip(scores, labels),reverse=True))
    return int(pos_label in labels[:k])


def average_precision(labels, scores, pos_label=1):
    '''average precision given a question'''
    scores, labels = zip(*sorted(zip(scores, labels),reverse=True))
    n =len(scores)
    n_pos = labels.count(pos_label)
    if not n_pos:
        return 0
    ap = sum([precision_at_k(labels, scores, i+1, pos_label) if labels[i]==pos_label else 0 for i in range(n)]) / n_pos
    return ap


def reciprocal_rank(labels, scores, pos_label=1):
    '''reciprocal rank given a question'''
    scores, labels = zip(*sorted(zip(scores, labels),reverse=True))
    return 1./(labels.index(pos_label)+1) if pos_label in labels else 0






### global metrics (that involve all questions)

def precision_curve(labels, scores, questions, pos_label=1):
    dict_questions = dict()
    for question, label, score in zip(questions, labels, scores):
        if question not in dict_questions or dict_questions[question]['score'] < score:
            dict_questions[question] = {'score': score, 'label': label}

    m_labels = [v['label'] for k,v in dict_questions.items()]
    m_scores = [v['score'] for k,v in dict_questions.items()]
    m_scores, m_labels = (list(t)[::-1] for t in zip(*sorted(zip(m_scores, m_labels))))
    
    precision, answer_rate = [], []
    pos = 0
    for i, s, l in zip(range(1,len(m_labels)+1), m_scores, m_labels):
        pos += l==pos_label
        precision.append(pos*1./i)
        answer_rate.append(i/len(m_labels))
    thresholds = m_scores

    return precision, answer_rate, thresholds


def area_under_precision_curve(labels, scores, questions, pos_label=1):
    precision, answer_rate, thresholds = precision_curve(labels, scores, questions, pos_label=pos_label)
    return sum(precision) / len(precision)