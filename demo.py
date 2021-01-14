import random
import logging
import pandas as pd
from classifier.topics import TopicComparator
from generator.gen_CFG import Topic

if __name__ == "__main__":
    gr_s = """
        S -> NP VP
        NP -> Adj N | N
        VP -> VB NP | Adv VB NP
        """
    vocab_IT = {
        "Adj": ["gigantic", "scalable"],
        "N": ["Jobs", "line", "device", "leak"],
        "VB": ["resets", "develops", "commits"],
        "Adv": ["skillfully", "laborously", "quickly"]}

    vocab_food = {
        "Adj": ["aromatic", "delicious", "rotten"],
        "N": ["Jamie", "knife", "recipe", "leak"],
        "VB": ["cooks", "develops", "commits"],
        "Adv": ["skillfully", "enough", "later", "quickly"]}

    topics = [Topic("IT", vocab_IT, gr_s),
              Topic("food", vocab_food, gr_s)
              ]
    n_sentences = 300

    df = (
        pd
        .DataFrame(data={
            'topic_1': random.choices(topics, n_sentences),
            "topic_2": random.choices(topics, n_sentences)})
        .assign(target=lambda x: x.topic_1 == x.topic_2,
                sentence_1=lambda x: x.apply(lambda y: y["topic_1"].get_sentence(), axis=1),
                sentence_2=lambda x: x.apply(lambda y: y["topic_2"].get_sentence(), axis=1))
        .drop(["topic_1", "topic_2"], axis=1))

    tc = TopicComparator()

    a, b = tc.train(df, df["target"])

    logging.info("Validation report :", tc.reports['validation'])
