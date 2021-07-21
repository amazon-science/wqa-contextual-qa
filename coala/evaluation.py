'''

Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
SPDX-License-Identifier: CC-BY-NC-4.0

'''


import numpy as np
from sklearn.metrics import roc_auc_score


def report(labels, scores, questions):
    '''computes various statistics and metrics'''
    results = {}
    if len(np.unique(labels)) == 2:
        results['roc_auc'] = roc_auc_score(labels, scores)
    dict_questions = {q:{'labels':[], 'scores':[]} for q in questions}
    for label, score, q in zip(labels, scores, questions):
        dict_questions[q]['labels'].append(label)
        dict_questions[q]['scores'].append(score)
    results['p@1'] = np.mean([precision_at_k(v['scores'], v['labels'], 1, 1) for v in dict_questions.values()])
    results['map'] = np.mean([average_precision(v['scores'], v['labels'], 1) for v in dict_questions.values()])
    results['questions'] = len(dict_questions)
    results['examples']  = len(labels)
    return results


def reciprocal_rank(scores, labels, pos_label=None):
    '''reciprocal rank given a query'''
    pos_label = pos_label or max(labels)
    scores, labels = zip(*sorted(zip(scores, labels),reverse=True))
    n = len(scores)

    return 1./(labels.index(1)+1)
    #return sum([1./i for i in range(n) if labels[i] == pos_label])


def average_precision(scores, labels, pos_label=None):
    '''average precision given a query'''
    pos_label = pos_label or max(labels)
    scores, labels = zip(*sorted(zip(scores, labels),reverse=True))
    n =len(scores)

    n_pos = labels.count(pos_label)
    if not n_pos:
        return 1
    ap = sum([precision_at_k(scores, labels, i+1, pos_label) if labels[i]==pos_label else 0 for i in range(n)]) / n_pos
    #print (labels, n_pos, ap, [precision_at_k(scores, labels, i+1, pos_label) for i in range(6)])
    return ap

def precision_at_k(scores, labels, k, pos_label=None):
    '''precision at k'''
    pos_label = pos_label or max(labels)
    scores, labels = zip(*sorted(zip(scores, labels),reverse=True))
    n =len(scores)
    
    n_pos = labels[:k].count(pos_label)
    if not n_pos:
        return 0.
    return sum([1. if l==pos_label else 0 for l in labels[:k]]) / k


def precision_curve(labels, scores, questions, title='Precision curve'):
    dict_questions = dict()
    for question, label, score in zip(questions, labels, scores):
        if question not in dict_questions or dict_questions[question]['score'] < score:
            dict_questions[question] = {'score': score, 'label': label}

    m_labels = [v['label'] for k,v in dict_questions.items()]
    m_scores = [v['score'] for k,v in dict_questions.items()]
    m_scores, m_labels = (list(t)[::-1] for t in zip(*sorted(zip(m_scores, m_labels))))
    
    precision, coverage = [], []
    pos = 0
    for i, s, l in zip(range(1,len(m_labels)+1), m_scores, m_labels):
        pos += l
        precision.append(pos*1./i)
        coverage.append(i/len(m_labels))
    thresholds = m_scores
    
    try:
        from matplotlib import pyplot as plt
        plt.plot(coverage, precision)
        plt.xlabel('recall')
        plt.ylabel('coverage')
        plt.title(title)
        plt.show()
    except ImportError:
        from .utils import get_logger
        logger = get_logger()
        logger.error('Matplotlib is not available here')

    return precision, coverage, thresholds
