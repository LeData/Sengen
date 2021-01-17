import random
import logging
from itertools import chain
import pandas as pd
from classifier.topic_comparators import MPSComparator
from generator.gen_CFG import Topic

if __name__ == "__main__":
    gr_s = """
        S -> NP VP
        NP -> Adj N | N
        VP -> VB NP | Adv VB NP
        """
    vocab_IT = {
        "Adj": ["gigantic", "scalable"],
        "N": ["Jobs", "line", "device", "leak", "network", "computer"],
        "VB": ["resets", "develops", "commits"],
        "Adv": ["skillfully", "laboriously", "quickly"]}

    vocab_food = {
        "Adj": ["aromatic", "delicious", "rotten", "hot", "spicy"],
        "N": ["Jamie", "knife", "recipe", "Apricot"],
        "VB": ["cooks", "develops", "commits", "stirs", "fries"],
        "Adv": ["skillfully", "enough", "later", "quickly", "slowly", "immediately", "carefully"]}

    words = [""] + list(set(chain(*vocab_IT.values(), *vocab_food.values())))

    topics = [Topic("IT", vocab_IT, gr_s),
              Topic("food", vocab_food, gr_s)
              ]
    n_sentences = 300
    df = (
        pd
        .DataFrame(data={
            'topic_1': random.choices(topics, k=n_sentences),
            "topic_2": random.choices(topics, k=n_sentences)})
        .assign(target=lambda x: x.topic_1 == x.topic_2,
                sentence_1=lambda x: x.apply(lambda y: y["topic_1"].get_sentence(), axis=1),
                sentence_2=lambda x: x.apply(lambda y: y["topic_2"].get_sentence(), axis=1))
        .drop(["topic_1", "topic_2"], axis=1))

    tc = MPSComparator(words=words, bond_dimension=6)
    losses, val_cm, val_cr = tc.train(X=df.drop("target", axis=1), y=df["target"], print_freq=5)

    logging.info("Validation report :", val_cr)
