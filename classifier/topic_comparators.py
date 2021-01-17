import torch
import numpy as np
import pandas as pd
import nltk
from itertools import chain
from collections import defaultdict, Counter
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from abc import ABC, abstractmethod
from classifier.tensor_utils import optimize, chain_compose, get_slicer


class TopicComparator(ABC):

    def __init__(self, words: list):

        words = list(set(words))
        try:
            empty_index = words.index("")
        except ValueError:
            pass
        else:
            words.pop(empty_index)
        finally:
            words = ["", *words]

        self.n_words = len(words)
        self.le = LabelEncoder()
        self.le.fit(words)

    @abstractmethod
    def train(self, X: pd.DataFrame, y: pd.Series, **kwargs):
        pass

    @staticmethod
    def report(y_true: np.array, y_pred: np.array, threshold:float = 0.5, verbose:bool = True):
        cm = confusion_matrix(y_true, y_pred > threshold, labels=[0, 1])
        cr = classification_report(y_true, y_pred, output_dict=True)
        if verbose:
            print(classification_report(y_true, y_pred, output_dict=False))
        return cm, cr


class TreeComparator(TopicComparator):

    def __init__(self):
        super().__init__()
        self.freq = defaultdict(int)
        self.len_corpus = 0
        self.reports = {}
        self.model = None

    def update_freq_with(self, sentences: pd.Series):
        n = self.len_corpus
        freq = defaultdict(int)
        freq.update({w: f * n for w, f in self.freq.items()})
        for x in chain.from_iterable(sentences.apply(nltk.word_tokenize)):
            n += 1
            freq[x] += 1
        for x in freq.keys():
            freq[x] /= n
        self.len_corpus = n
        self.freq = freq
        return self

    def get_freq(self, X: pd.Series):
        x_counts = X.apply(nltk.word_tokenize).apply(Counter)

        output = (pd
                  .DataFrame
                  .from_records(x_counts)
                  .fillna(0)
                  .reindex(self.freq.keys(), axis=1)
                  .apply(lambda x: x / len(x))
                  )

        return output

    def prepare_data(self, sentences_1: pd.Series, sentences_2: pd.Series, training=False):
        assert (sentences_1.index == sentences_2.index).all(), "The indices of the series don't align"
        if training:
            self.update_freq_with(pd.concat([sentences_1, sentences_2], axis=0))
        f_1 = self.get_freq(sentences_1)
        f_2 = self.get_freq(sentences_2)
        return pd.merge(left=f_1 + f_2, right=(f_1 - f_2).abs(), left_index=True, right_index=True)

    def train(self, df, y, train_split=.3):
        self.model = DecisionTreeClassifier()
        X = self.prepare_data(df.sentence_1, df.sentence_2, training=True)
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=train_split)
        self.model.fit(X_train, y_train)
        self.reports["validation"] = self.report(y_true=y_val.values,
                    y_pred=self.model.predict(X_val.values))[1]
        return X_val, y_val


class MPSComparator(TopicComparator):

    def __init__(self, words:list, bond_dimension:int = 6):
        """
        :param words: entire vocabulary of the language to consider
        :param bond_dimension: dimension of the bond for the Matrix Product State model
        """
        super().__init__(words)
        self.n_bond = bond_dimension
        # Setting up the tensors representing language (dictionary and sentence start)
        self.S_tensor = torch.ones(self.n_bond) / np.sqrt(self.n_bond)
        self.D = torch.rand([self.n_bond, self.n_words, self.n_bond])
        self.D[:, 0, :] = torch.eye(self.n_bond)
        self.D.requires_grad = True

    def train(self, X: pd.DataFrame, y:pd.DataFrame, test_size:float = 0.2, print_freq: int = 100) -> tuple:
        """
        Trains the model.
        :param X: Dataframe containing the sentences to compare
        :param y: target variable
        :param test_size: size of the validation set
        :param print_freq: frequency used for logging of the training steps
        :return: reports on validation
        """
        max_sentence = (
            X
                .loc[:, ["sentence_1", "sentence_2"]]
                .applymap(lambda x: len(x.split(" ")))
                .max()
                .max()
        )
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size)
        (X_train_1, X_train_2), y_train = self.prepare_data(X_train, y_train, n_max = max_sentence)
        (X_val_1, X_val_2) , y_train= self.prepare_data(X_val, y_val, n_max = max_sentence)
        loss_fct = self.define_loss_fct(x_1=X_train_1, x_2=X_train_2, y=y_train)
        optimize(self.D, loss_fct, print_freq=print_freq)
        y_pred_val = self.compare_sentences(X_val_1, X_val_2)
        return self.report(y_val, y_pred_val)

    def prepare_data(self, X: pd.DataFrame, y: pd.Series = None, n_max: int = 30):
        """
        Prepares the data for training or scoring
        :param X: input sentences
        :param y: target variable
        :param n_max: maximum number of words to be considered per sentence
        :return:
        """
        inputs = [
            torch.from_numpy(
                X[col]
                .str.split(" ", expand=True)
                .reindex(range(n_max), axis=1)
                .fillna("")
                .apply(self.le.transform)
                .to_numpy()
            ).long() for col in ["sentence_1", "sentence_2"]]
        if y is not None:
            y = torch.from_numpy(y.to_numpy()).long()
        return inputs, y

    def build_mps(self, length):
        tensors = [self.S_tensor] + [self.D] * length
        return chain_compose(*tensors)

    @staticmethod
    def similarity(x, y):
        cos_sim = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)(x, y)
        return torch.div(torch.add(cos_sim, 1), 2)

    @staticmethod
    def cross_entropy_loss(y_pred, y_true):
        return -torch.log(y_pred.where(y_true.bool(), 1 - y_pred))

    def compare_sentences(self, s_1, s_2, max_words):
        assert s_1.shape == s_2.shape, "The input tensors should be of the same shape"
        mps = self.build_mps(s_1.shape[1])
        v_1 = mps[get_slicer(s_1)]
        v_2 = mps[get_slicer(s_2)]
        return self.similarity(v_1, v_2)

    def define_loss_fct(self, x_1, x_2, y):
        def loss_fct(model):
            self.D = model
            y_pred = self.compare_sentences(x_1, x_2, model)
            return torch.sum(self.cross_entropy_loss(y_pred, y))
        return loss_fct