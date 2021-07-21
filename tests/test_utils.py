'''

Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
SPDX-License-Identifier: CC-BY-NC-4.0

'''


import unittest

from coala.utils import dispatcher
from coala.datasets import AS2BaseDataset, AS2LocalDataset, AS2LocalOrdDataset
from coala.models import BaseModelForAS2, LocalModelForAS2, LocalOrdModelForAS2


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

