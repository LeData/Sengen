from unittest import TestCase
import random
import torch


class Test(TestCase):

    def test_get_slicer(self):
        from classifier.tensor_utils import get_slicer
        t = torch.rand([10, 5])
        n = random.randint(0, t.dim()-1)
        slicer = get_slicer(t, n)
        self.assertIsInstance(slicer, tuple,
                              msg="should return a tuple")
        self.assertEqual(len(slicer), t.shape[n],
                         msg="wrong length of tuple")

    def test_chain_compose(self):
        from classifier.tensor_utils import chain_compose
        t = torch.rand([10, 5])
        correct_chain = [t, t.transpose(0,1)]
        output_right = chain_compose(*correct_chain)
        self.assertIsInstance(output_right, torch.Tensor,
                              msg="The output should be a tensor")

    def test_optimize(self):
        pass