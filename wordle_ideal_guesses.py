"""
wordle_ideal_guesses.py

This Python program uses a minimax implimentation with to evaluate the wordle guess list against the wordle selection
list to produce a outputs of "best" words. I know I might have talked to you recently and said that I had a solve for
Wordle. Unfortunately, I do not. I had a but where I was only evaluating a subset of the permuations and over-valuing
some word combinations. This implementation still can produce guesses of length 2 that have a worst case scenerio of
26 words remaining. Here is a table of some high quality guesses.

Words Remaining | w1 | w2
26 | tegua | colts
28 | fayne | borts
12 | clods | tyran

Words Remaining | w1 | w2 | w3
8 | guars | chile | daurs
8 | antas | chile | daurs
11 | erven | lotsa | daurs

Ex:
    Using the word POINT as a solution with an initial 3 guesses of guars chile daurs we get the following state
    BYYBB
    BBGBB
    BBBBB
With this information in a regex on the word list there are 4 valid words left
    joint
    point
    twink
    tying

Recommendations for future work:
Wrap this model in UI so a player can work with it interactively, rather than making 2-3 guesses in the dark.
Modify the variants calculation so it behaves recursively. That is word_2's variants are scored after suppoosing an
arbitrary variant from word_1. Likewise for word_3 to word_2.
"""
from functools import lru_cache
import random
import itertools as it


def get_possible_puzzle_solutions() -> list[str]:
    """
    Gets all possible solutions for wordle. This is different than the list of
    accepted guesses. The set of words than can be used for guesses is a
    strict superset of the set of words that can be the solution.
    """
    words = open("wordles.txt", "r").read().split("\n")
    words = list(map(str.lower, words))
    return words


def get_valid_wordle_guesses():
    """
    Gets all possible guesses for wordle. This is different than the list of solutions.
    See `get_possible_puzzle_solutions`.
    """
    words = open("guesses.txt", "r").read().split("\n")
    words = list(map(str.lower, words))
    return words


class PrefixTree:
    """
    A prefix tree structure that allows a player to doo a dictionary-like
    query and get a list of words that satisfy that wildcard expression.

    e.g. PrefixTree[".at"] -> ["rat", "cat", "bat"]

    Technically this dosn't need to be implemented as generally as I chose to
    do because we know all word lengths to be 5.
    """

    def __init__(self, data: dict["str", "PrefixTree"] | None = None) -> None:
        """
        Set up the data structures needed for this implementation.

        is_terminal: indicates whether this point in the traversal of the tree constitutes as a valid word.
        words: all inserted words that correspond to this location in the tree
        data: the recursive prefix tree itself
        """
        self.is_terminal = False
        self.words: list[str] = []
        if data is None:
            self.data: dict["str", "PrefixTree"] = dict()
        else:
            self.data = data

    @lru_cache(None)
    def __getitem__(self, word: str) -> frozenset[str]:
        # if there is no word queried we will get the words for this node as if
        # we terminated at this point. self.words will onyl be populated when
        # self.is_terminal = True
        if word == "":
            return frozenset(self.words)

        # DFS Traversal
        # https://en.wikipedia.org/wiki/Depth-first_search
        curr = self
        for i, c in enumerate(word):
            if c == ".":
                words: set[str] = set()
                for k in curr.data.keys():
                    words |= curr.data[k][word[i + 1 :]]
                return frozenset(words)
            else:
                if c not in curr.data:
                    return frozenset()
                curr = curr.data[c]
        return frozenset(curr.words)

    def __contains__(self, item: str) -> bool:
        # DFS Traversal
        curr = self
        for i, c in enumerate(item):
            if c == ".":
                for k in self.data.keys():
                    if self.data[k][item[i + 1 :]]:
                        return True
                return False
            else:
                if c not in curr.data:
                    return False
                curr = curr.data[c]
        return curr.is_terminal

    def add(self, word: str):
        """
        Insertion method for new words. I chose not to use a defaultdict for
        this implementation so I would have some query safety and get
        KeyNotFoundExceptions on any programming errors.
        """
        curr = self
        for c in word:
            if c not in curr.data:
                curr.data[c] = PrefixTree()
            curr = curr.data[c]
        curr.is_terminal = True
        curr.words.append(word)
        return

    def __repr__(self) -> str:
        return f"Prefixtree[{self.data}, {self.words}]"


def construct_prefix_tree_from_word_list():
    """
    Builds the prefix tree from the wordle word list
    """
    a = PrefixTree()
    for w in get_possible_puzzle_solutions():
        a.add(w)
    return a


@lru_cache(None)
def variants_of(word: str) -> list[tuple[str, frozenset[str]]]:
    """
    Gets the possible correct/incorrect permuations for a word. This pattern
    is essentially include/exclude so there are 2^n variants of a word of
    length n.
    """
    other = "." * len(word)
    variants = list(map("".join, it.product(*zip(word, other))))
    sw = set(word)
    excludes = [frozenset(sw - set(v)) for v in variants]
    return list(zip(variants, excludes))


@lru_cache(None)
def get_valid_word_set_after_exclusions(
    word_set: set[str], excluded_letters_from_word: set[str]
):
    return set(
        [w for w in word_set if all(c not in excluded_letters_from_word for c in w)]
    )


def main():
    tree = construct_prefix_tree_from_word_list()
    guess_words = get_valid_wordle_guesses()
    pair_scores: list[tuple[int, tuple[str, ...]]] = []
    # For better computations times we don't iterate over the entire word set
    # multiple times. This strategy produces ~3 worst case words at 3 words and ~23-30 worst case words at 2 guesses.
    # Play with the sample sizes to see if you can find a good balance between speed and effectiveness
    for selected_words in it.product(
        random.sample(guess_words, 20),
        random.sample(guess_words, 10),
        random.sample(guess_words, 3),
    ):
        # Stores the most poorly performing set of variants for the selected
        # words. The strategy implemented here is the minimax strategy, where
        # solutions are "better" when their worst case is the "least bad".
        # https://en.wikipedia.org/wiki/Minimax
        worst_case_variant_score = 0
        # Some dense functional programming and list comprehensions here.
        # The just of it is to get all possible "variants" of each word,
        # permute over all variant combinations in the words, and score that
        # situation.
        for variants_and_exclusions in it.product(*map(variants_of, selected_words)):
            trees = [tree[ve[0]] for ve in variants_and_exclusions]
            remaining_words_for_each_variant: list[set[str]] = [
                get_valid_word_set_after_exclusions(t, ve[1])
                for t, ve in zip(trees, variants_and_exclusions)
            ]
            intersection: set[str] = set.intersection(*remaining_words_for_each_variant)  # type: ignore
            score = len(intersection)
            if score > worst_case_variant_score:
                worst_case_variant_score = score

        score_for_word_selection = worst_case_variant_score
        pair_scores.append((score_for_word_selection, selected_words))

    # Finally we output the selected words.
    print(*sorted(pair_scores[:100], reverse=True), sep="\n")


if __name__ == "__main__":
    main()
