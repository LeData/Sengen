from itertools import chain
from collections import defaultdict, Counter
from sklearn.tree import DecisionTreeClassifier as dct
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

class Topic:

    def __init__(self, name, vocabulary):
        self.name = name
        self.vocabulary = vocabulary


class TopicComparator:

    def __init__(self):
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

    @staticmethod
    def get_freq(x: pd.Series):
        try:
            x_counts = x.apply(nltk.word_tokenize).apply(Counter)
        except:
            raise

        output = (pd
                  .DataFrame
                  .from_records(x_counts)
                  .fillna(0)
                  .reindex(tc.freq.keys(), axis=1)
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
        self.model = dct()
        X = self.prepare_data(df.sentence_1, df.sentence_2, training=True)
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=train_split)
        self.model.fit(X_train, y_train)
        self.report(y_true=y_val.values,
                    y_pred=self.model.predict(X_val.values),
                    name='validation')
        return X_val, y_val

    def report(self, y_true, y_pred, name):
        CM = confusion_matrix(y_true, y_pred, labels=[0, 1])
        print(CM)
        TN = CM[0][0]
        FN = CM[1][0]
        TP = CM[1][1]
        FP = CM[0][1]
        self.reports[name] = pd.Series({
            'tnr': TN / (TN + FP),
            'tpr': TP / (TP + FN)
        })