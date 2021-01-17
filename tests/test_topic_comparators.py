from unittest import TestCase
from unittest.mock import patch
import numpy as np
from classifier.topic_comparators import TopicComparator


class TestTopicComparator(TestCase):
    words = ["you", "are", "being", "tested"]

    @patch.multiple(TopicComparator, __abstractmethods__=set())
    def test_init(self):
        instance = TopicComparator(self.words)
        self.assertIn("", instance.le.classes_,
                      msg="The label encoder should contain the empty word")
        self.assertEqual("", instance.le.classes_[0],
                         msg="The empty word should be the first element of the encoder")
        self.assertEqual(instance.n_words, len(instance.le.classes_),
                         msg="The classes of the encoder shouldn't exceed the n_words attribute")

    @patch.multiple(TopicComparator, __abstractmethods__=set())
    def test_report(self):
        instance = TopicComparator(self.words)
        rep = instance.report(np.array([True, False, False]), np.array([0.4, 0.2, 0.6]))
        self.assertIsInstance(rep, tuple, msg="The report should have two components.")


class TestMPSComparator(TestCase):
    words = ["you", "are", "being", "tested"]

    def test_init(self):
        from classifier.topic_comparators import MPSComparator
        comp = MPSComparator(self.words)
        self.assertEqual(comp.S_tensor.shape[0], comp.n_bond,
                         msg="The vector for the dictionary unit is of the wrong dimension")
        self.assertTrue(comp.D.requires_grad, msg="The dictionary tensor must be ready for back propagation.")
        self.assertEqual(comp.D.shape[0], comp.D.shape[-1],
                         msg="The first and last dimension of the dictionary must be equal")
        self.assertEqual(comp.D.shape[0], comp.n_bond,
                         msg="The first dimension of the dictionary must be the dimension of the bond.")
        self.assertEqual(comp.D.shape[1], comp.n_words,
                         msg="The middle dimension of the dictionary must match the number of words.")

    def test_train(self):
        from classifier.topic_comparators import MPSComparator
        comp = MPSComparator(self.words)
        self.fail()

    def test_prepare_data(self):
        from classifier.topic_comparators import MPSComparator
        comp = MPSComparator(self.words)
        self.fail()

    def test_build_mps(self):
        from classifier.topic_comparators import MPSComparator
        comp = MPSComparator(self.words)
        self.fail()

    def test_similarity(self):
        from classifier.topic_comparators import MPSComparator
        comp = MPSComparator(self.words)
        self.fail()

    def test_cross_entropy_loss(self):
        from classifier.topic_comparators import MPSComparator
        comp = MPSComparator(self.words)
        self.fail()

    def test_compare_sentences(self):
        from classifier.topic_comparators import MPSComparator
        comp = MPSComparator(self.words)
        self.fail()

    def test_define_loss_fct(self):
        from classifier.topic_comparators import MPSComparator
        comp = MPSComparator(self.words)
        self.fail()
