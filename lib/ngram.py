import random
from collections import Counter, defaultdict
from typing import Any, Optional
from tqdm import tqdm
from itertools import pairwise
import re
from concurrent.futures import ProcessPoolExecutor
from functools import partial

class Node:
    def __init__(self, value: Any = None) -> None:
        self.next_: Optional[list[Node]] = None
        self.value = value

    def _average_length_helper(self) -> list[int]:
        vals = []
        if self.next_:
            for n in self.next_:
                vals.extend([i + 1 for i in n._average_length_helper()])
        else:
            vals.append(0)

        return vals

    def average_length(self) -> float:
        if self.next_:
            lengths = self._average_length_helper()
            return sum(lengths)/len(lengths)
        else:
            return 0

def _count_token_pairs(tokens: list[str]) -> Counter:
    return Counter(pairwise(tokens))

def _tokenize_helper(i: int, text: str, vocab_tree: Node) -> tuple[str, int]:
    curr = vocab_tree
    token = ""
    il = i
    while i < len(text):
        if not curr.next_:
            break
        for n in curr.next_:
            if text[i] == n.value:
                token += text[i]
                curr = n
                i += 1
                break
        if il == i:
            break
        il = i
    return token, i

def _tokenize(text: str, vocab_tree: Node) -> list[str]:
    if not vocab_tree.next_:
        return []

    tokens = []
    len_text = len(text)
    with tqdm(total=len_text, desc="Tokenizing", dynamic_ncols=True) as pbar:
        i = 0
        while i < len_text:            
            token, i = _tokenize_helper(i, text, vocab_tree)
            if len(token) == 0:
                break
            pbar.update(len(token))
            tokens.append(token)

    return tokens

class Model:
    def __init__(self, n: int=24, vocab_size: int=8096) -> None:
        self.n = n
        self.vocab_size = vocab_size
        self.vocab: list[str] = []
        self.vocab_tree: Node = Node()
        self.model: dict[tuple[str, ...], dict[str, int]] = {}
        self.start_token = "♍"
        self.end_token = "♎"

    def get_next(self, history: list[str]=[], rng: Optional[random.Random] = None) -> str:
        gram = tuple(history[-self.n:])
        w = 10
        possible = []
        for i in range(len(gram)):
            gram_i = gram[i:]
            if gram_i in self.model:
                for text in self.model[gram_i]:
                    possible.extend([text] * self.model[gram_i][text] * w)
            if len(possible) > 0:
                break

        possible += list(random.sample(self.vocab.copy(), self.vocab_size//1000))
        return rng.choice(possible) if rng else random.choice(possible)

    def tokenize(self, text: str) -> list[str]:
        return _tokenize(text, self.vocab_tree)

    def create_bpe(self, texts: list[str]) -> None:
        if len(self.vocab) == self.vocab_size:
            return
            
        self.vocab = sorted(set("".join(texts)))
        self.vocab += [self.start_token, self.end_token]
        self.build_vocab_tree()

        with ProcessPoolExecutor() as executor:
            tokens = list(executor.map(partial(_tokenize, vocab_tree=self.vocab_tree), texts))
        
        if self.vocab_size - len(self.vocab) <= 0:
            return

        with tqdm(total=self.vocab_size, desc="Building BPE", initial=len(self.vocab), dynamic_ncols=True, smoothing=1) as pbar:
            while len(self.vocab) < self.vocab_size:
                with ProcessPoolExecutor() as executor:
                    results = list(executor.map(_count_token_pairs, tokens))

                pairs = sum(results, Counter())

                if not pairs:
                    break

                common_pairs = list(filter(
                    lambda x: x[-1] > 2,
                    pairs.most_common(random.randint(1, 100))
                ))

                if len(common_pairs) == 0:
                    break
                parsed_pairs = [(first, second, first + second) for (first, second), _ in common_pairs]

                new_token_seq = []
                for seq in tokens:
                    merged = []
                    i = 0
                    while i < (len_seq := len(seq)):
                        matched = False
                        seq_i = seq[i]
                        for first, second, new_token in parsed_pairs:
                            if seq_i == first and i + 1 < len_seq and seq[i+1] == second:
                                merged.append(new_token)
                                i += 2
                                matched = True
                                break
                        if not matched:
                            merged.append(seq_i)
                            i += 1
                    new_token_seq.append(merged)

                tokens = new_token_seq
                self.vocab.extend([t for _, _, t in parsed_pairs])
                pbar.update(len(parsed_pairs))
        self.vocab.sort(key=len, reverse=True)
        self.build_vocab_tree()

    def build_vocab_tree(self) -> None:
        self.vocab_tree = Node()
        with tqdm(total=len(self.vocab), desc="Building vocab tree", initial=len(self.vocab), dynamic_ncols=True) as pbar:
            for v in self.vocab:
                curr = self.vocab_tree
                while len(v) > 0:
                    if curr.next_ is None:
                        curr.next_ = [Node(v[0])]
                        curr = curr.next_[0]
                        v = v[1:]
                        continue
                    elif v[0] in [n.value for n in curr.next_]:
                        curr = [n for n in curr.next_ if n.value == v[0]][0]
                        v = v[1:]
                        continue
                    else:
                        curr.next_.append(Node(v[0]))
                        curr = curr.next_[-1]
                        v = v[1:]
                        continue
                pbar.update(1)

    def pre_process(self, text: str) -> str:
        text = re.sub(r"\s+", " ", text, flags=re.MULTILINE)
        text = ". ".join([t.capitalize() for t in text.split(". ")])
        text = text.replace(self.start_token, "")
        text = text.replace(self.end_token, "")
        text = text.replace("“", "\"")
        text = text.replace("”", "\"")
        text = text.replace("and", "&")
        text = text.replace("’", "'")
        text = text.strip()
        return text

    def train(self, data_paths: list[str]) -> None:
        texts = []
        with tqdm(total=len(data_paths), desc="Loading files", dynamic_ncols=True) as pbar:
            for path in data_paths:
                try:
                    with open(path, "r") as f:
                        content = f.read().strip()
                        if content:
                            texts.append(self.start_token + self.pre_process(content) + self.end_token)
                except Exception as e:
                    print(f"Error reading {path}: {e}")
                pbar.update(1)

        self.create_bpe(texts)

        with ProcessPoolExecutor() as executor:
            tokens = list(executor.map(partial(_tokenize, vocab_tree=self.vocab_tree), texts))

        del texts

        model = defaultdict(lambda: defaultdict(int))
        
        for seq in tokens:
            for i in range(len(seq) - 1):
                context_len = min(self.n, i)
                if context_len == 0:
                    continue
                gram = tuple(seq[max(0, i - context_len + 1):i + 1])
                next_token = seq[i + 1]
                model[gram][next_token] += 1

        self.model = {k: dict(v) for k, v in model.items()}
        del tokens