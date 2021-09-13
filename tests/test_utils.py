'''

Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
SPDX-License-Identifier: CC-BY-NC-4.0

'''


import unittest

from coala.utils import dispatcher
from coala.datasets import AS2BaseDataset, AS2LocalDataset, AS2LocalOrdDataset
from coala.models import BaseModelForAS2, LocalModelForAS2, LocalOrdModelForAS2
from coala import evaluation as ev

class TestUtils(unittest.TestCase):
    # Evaluate other utilities outside the scope of previous unittests
    
    def	setUp(self):
        return

    def test_dispatcher(self):
        model, dataset = dispatcher('base')
        self.assertEqual(model, BaseModelForAS2)
        self.assertEqual(dataset, AS2BaseDataset)
        model, dataset = dispatcher('local')
        self.assertEqual(model, LocalModelForAS2)
        self.assertEqual(dataset, AS2LocalDataset)
        model, dataset = dispatcher('local-ord')
        self.assertEqual(model, LocalOrdModelForAS2)
        self.assertEqual(dataset, AS2LocalOrdDataset)
        with self.assertRaises(ValueError):
            dispatcher('random-context')
            dispatcher(0)

class TestMetrics(unittest.TestCase):

    def setUp(self):
        return

    def	test_precision_curve(self):
        labels = [1,1,0,0,0,0]
        scores = [.9, .7, .5, .4, .8, .3]
        questions = ['a','b','c','c','d','d']
        pre, cov, thr =	ev.precision_curve(labels, scores, questions)
        self.assertListEqual(pre, [1, 0.5, 2/3, 0.5])
        self.assertListEqual(cov, [.25, .5, .75, 1])
        self.assertListEqual(thr, [.9, .8, .7, .5])

    def test_precision(self):
        self.assertEqual(0, ev.precision_at_k([0,0,0],[1,2,3], 1) )
        self.assertEqual(0, ev.precision_at_k([0,0,0],[1,2,3], 3) )
        self.assertEqual(1, ev.precision_at_k([1,0,0],[3,2,1], 1) )
        self.assertEqual(1, ev.precision_at_k([0,0,1,0],[1,2,3,0], 1) )
        self.assertEqual(0.5, ev.precision_at_k([0,0,1,0],[1,2,3,0], 2) )
        self.assertEqual(0.25, ev.precision_at_k([0,0,1,0],[1,2,3,0], 4) )
        self.assertEqual(1/3, ev.precision_at_k([0,0,1,1],[1,2,3,0], 3) )
        self.assertEqual(1, ev.precision_at_k([1,1,1,1],[1,2,3,0], 2) )
        self.assertEqual(1, ev.precision_at_k([0,1,1,0],[1,2,3,0], 2) )

    def test_hit_rate(self):
        self.assertEqual(1, ev.hit_at_k([1,1,1,1],[1,2,3,0], 2) )
        self.assertEqual(0, ev.hit_at_k([0,0,0,1],[1,2,3,0], 1) )
        self.assertEqual(0, ev.hit_at_k([0,0,0,1],[1,2,3,0], 2) )
        self.assertEqual(0, ev.hit_at_k([0,0,0,1],[1,2,3,0], 3) )
        self.assertEqual(1, ev.hit_at_k([0,0,0,1],[1,2,3,0], 4) )
        self.assertEqual(0, ev.hit_at_k([0,0,0],[1,3,0], 3) )

    def test_average_precision(self):
        self.assertAlmostEqual(0.77083333, ev.average_precision([1,0,1,1,0,1,0,0],[0.9, 0.85, .71, 0.63, 0.47, 0.36, 0.24, 0.16]))
        self.assertAlmostEqual(0.7, ev.average_precision([1,0,0,1,1,0,0],[10,9,8,7,6,5,4]))
        self.assertAlmostEqual(0, ev.average_precision([0,0,0,0,0,0,0],[10,9,8,7,6,5,4]))
        self.assertAlmostEqual(1, ev.average_precision([1,1,0,0,0,0,0],[10,9,8,7,6,5,4]))
        self.assertAlmostEqual(0.7, ev.average_precision([1,0,0,1,0,1,0],[10,5,8,7,9,6,4]))

    def test_reciprocal_rank(self):
        self.assertAlmostEqual(1, ev.reciprocal_rank([1,1,0,0],[4,3,2,1]))
        self.assertAlmostEqual(1, ev.reciprocal_rank([0,1,0,0],[0,3,2,1]))
        self.assertAlmostEqual(0.5, ev.reciprocal_rank([0,1,0,0],[4,3,2,1]))
        self.assertAlmostEqual(0.5, ev.reciprocal_rank([0,1,0,1],[4,3,2,1]))
        self.assertAlmostEqual(1/3, ev.reciprocal_rank([0,0,1,1],[4,3,2,1]))
        self.assertAlmostEqual(0, ev.reciprocal_rank([0,0,0,0],[4,3,2,1]))

    def test_area_under_precision_curve(self):
        self.assertAlmostEqual()