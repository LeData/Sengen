import nltk
import re
from collections import defaultdict
from itertools import chain
import logging
import random

production_line = re.compile(r"([a-zA-Z]+) -> ([\w|\s]+)")
items = r"[^|\s][^\|]*[^|\s]*"


def grammar_str_to_dict(g_string: str) -> dict:
    """
    Takes a string representing a grammar and returns a dictionary of branches by root.

    :param g_string: string representing the CFG
    :return:
    """
    lines = [re.findall(production_line, line) for line in g_string.strip().split("\n")]
    grammar_dict = defaultdict(list, {line[0][0]: re.findall(items, line[0][1]) for line in lines})
    return grammar_dict


def dict_to_grammar(grammar_dict: dict, starting_symbol: str) -> nltk.CFG:
    """
    Takes a dictionary of branches and returns the CFG that corresponds to it
    :param grammar_dict:
    :param starting_symbol: symbol to use as the start of the semantic trees.
    :return:
    """
    d = grammar_dict.copy()
    lines = []
    try:
        lines.append(starting_symbol + " -> " + " | ".join(d.pop(starting_symbol)))
    except KeyError as e:
        raise e
    else:
        lines.extend([lhs + " -> " + " | ".join(rhs) for lhs, rhs in d.items()])
    grammar_string = "\n".join(lines)
    return nltk.grammar.CFG.fromstring(grammar_string)


class Topic:
    """
    Simple class representing a topic.
    """

    def __init__(self, name: str, vocabulary: dict, grammar: str):
        self.name = name
        self.vocab = defaultdict(list, vocabulary)
        self.CFG = self._make_grammar(grammar)

    def _make_grammar(self, grammar: str):
        """
        Creates the grammar of the topic from a CFG as string.

        :param grammar: string representing the CFG
        :return:
        """
        gr_d = grammar_str_to_dict(grammar)
        v = self.vocab.copy()
        for k in gr_d:
            v[k].extend(gr_d[k])
        return dict_to_grammar(v, 'S')

    def get_sentence(self) -> str:
        """
        Generates a random sentence in the vocabulary.

        :return: string
        """
        t = Tree(self.CFG)
        t.grow()
        words = [nt.symbol() for nt in t.leaves]
        return " ".join(words)


class Tree:
    """
    Helper class for tree growth within a given CFG.
    """

    def __init__(self, grammar: nltk.CFG):
        self.tree = []  # list of tuples of branches
        self.branch_generator = grammar
        self.leafs = []  # list of triples, (item, depth, position)

    @property
    def height(self) -> int:
        """
        Height of the tree
        :return: int
        """
        return len(self.tree)

    @property
    def leaves(self) -> tuple:
        """
        Tuple of current leaves of the tree
        :return:  list
        """
        try:
            return self.tree[-1]
        except IndexError:
            return ()

    def grow_branch_from(self, n_leaf) -> 'Tree':
        """
        Randomly grow a single branch from a given leaf
        :param n_leaf:
        :return:
        """
        # TODO: implement
        print(n_leaf)
        return self

    def next_depth_for(self, leaf: str = None) -> list:
        """
        Returns all the possible branches that can stem from a given leaf.
        :param leaf: symbol/word representing the leaf
        :return:
        """
        if leaf is None:
            return [(self.branch_generator.start(),)]
        else:
            options = self.branch_generator.productions(lhs=leaf)
            if len(options) > 0:
                return [option.rhs() for option in options]
            else:
                return [(leaf,)]

    def grow_level(self):
        """
        Randomly grows the tree to the next height/depth.
        :return:
        """
        logging.info("Growing tree")
        try:
            leaves = self.tree[-1]
        except IndexError:
            leaves = (None,)
        branch_choices = [self.next_depth_for(leaf) for leaf in leaves]
        new_leaves = []
        for choices in branch_choices:
            try:
                choice = random.choice(choices)
            except IndexError:
                choice = [(choices,)]
            new_leaves.append(choice)
        if len(new_leaves) > 1:
            new_leaves = tuple(chain(*new_leaves))
        else:
            new_leaves = new_leaves[0]
        if new_leaves != leaves:
            self.tree = [*self.tree, new_leaves]

    def grow(self, max_height=None) -> 'Tree':
        """
        Randomly grows the tree to completion.

        :param max_height: height to stop growing the tree at.
        :return: self
        """
        if self.height == 0:
            self.grow_level()
        n = 0
        if max_height is not None:
            # TODO: implement
            pass
        else:
            while self.height - n > 0:
                logging.info(f"Growing level {self.height + 1}")
                n = self.height
                self.grow_level()
        return self
