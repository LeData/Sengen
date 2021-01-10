import random
import logging
import pandas as pd
from classifier.topics import TopicComparator, Topic
from generator.gen_CFG import SimpleGenerator
from nltk.grammar import CFG


topics = [Topic(t, None) for t in ["food", "IT"]]

if __name__ == "__main__":
    gr1 = CFG.fromstring(
        """
        S -> NP VP
        NP -> Adj N | N
        VP -> VB NP | Adv VB NP
        """)
    Part1 = {t.name: SimpleGenerator(gr1, t.name[::-1]).add_vocabulary(t.vocabulary) for t in topics}

    n = 300


    def get_n_topics(n, topics):
        return [random.choice(topics) for n in range(n)]


    dataset = (
        pd
            .DataFrame(data={
            'topic_1': get_n_topics(300, [t.name for t in topics]),
            "topic_2": get_n_topics(300, [t.name for t in topics])})
            .assign(target=lambda x: x.topic_1 == x.topic_2)
    )
    dataset["sentence_1"] = dataset.apply(lambda x: Part1.get(x["topic_1"]).generate_sentence(), axis=1)
    dataset["sentence_2"] = dataset.apply(lambda x: Part1.get(x["topic_2"]).generate_sentence(), axis=1)

    tc = TopicComparator()

    a, b = tc.train(dataset, dataset["target"])

    logging.info("Validation report :", tc.reports['validation'])
