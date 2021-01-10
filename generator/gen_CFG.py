import nltk

class SimpleGenerator:

    def __init__(self, grammar: nltk.grammar.CFG, word):
        self.grammar = grammar
        self.word = word


    def add_vocabulary(self, vocab: dict):
        return self

    def generate_sentence(self, max_depth = 5):
        """
            generates a random sentence in the vocabulary
        :return:
        """
        return f"I am a sentence with {self.word}" 