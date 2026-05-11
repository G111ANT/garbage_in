import random
from collections import Counter
from typing import Any
from tqdm import tqdm

class Model:
    def __init__(self, n: int=24, vocab_size: int=8096) -> None:
        self.n = n
        self.vocab_size = vocab_size
        self.vocab: list[str] = []
        self.model: dict[Any, dict[str, int]] = {}

    def get_next(self, history: list[str]=[], seed=None) -> str:
        rng = random.Random()
        rng.seed(seed)

        gram = tuple(history[:-self.n])

        possible = self.vocab.copy()
        if gram in self.model:
            for text in self.model[gram]:
                possible.extend([text] * self.model[gram][text] * 10)
            
        else:
            for gram_ in self.model:
                if " ".join(gram).endswith(" ".join(gram_)):
                    for text in self.model[gram_]:
                        possible.extend([text] * self.model[gram_][text] * 10)

        return rng.choice(possible)

    def tokenize(self, text: str) -> list[str]:
        if len(self.vocab) == 0:
            return [text]
        elif max(map(len, self.vocab)) == 1:
            return list(text)

        tokens = []
        vocab = sorted(self.vocab, key=len, reverse=True)
        while len(text) > 0:
            for v in vocab:
                if text.startswith(v):
                    tokens.append(v)
                    text = text[len(v):]
                    break
            else:
                return tokens
        return tokens

    def create_bpe(self, texts: list[str]) -> None:
        if len(self.vocab) == self.vocab_size:
            return
        self.vocab = list(set("".join(texts)))
        self.vocab += ["<START>", "</START>"]

        tokens = []
        for text in texts:
            tokens.append(self.tokenize(text))

        with tqdm(total=self.vocab_size - len(self.vocab)) as pbar:
            while len(self.vocab) < self.vocab_size:
                pairs = Counter()
                for text in tokens:
                    pairs.update(zip(text, text[1:]))

                if len(pairs) == 0:
                    return
        
                first, freq = pairs.most_common(1)[0]
                del pairs
                if freq < 2:
                    break

                for t in range(len(tokens)):
                    i = 0
                    while i < len(tokens[t]) - 1:
                        if tokens[t][i] == first[0] and tokens[t][i + 1] == first[1]:
                            tokens[t][i] = first[0] + first[1]
                            tokens[t].pop(i + 1)
                        i += 1

                self.vocab.append(first[0] + first[1])
                pbar.update(1)

        print("VOCAB")
        print(self.vocab)
        print("VOCAB")
        return

    def train(self, data_paths: list[str]) -> None:
        texts = []
        for path in data_paths:
            try:
                with open(path, "r") as f:
                    texts.append(f.read())
                    texts[-1] = "<START>" + texts[-1] + "</START>"
            except Exception as e:
                print(e)

        self.create_bpe(texts)

        tokens: list[list[str]] = []
        for text in texts:
            tokens.append(self.tokenize(text))

        del texts

        for text in tokens:
            for i in range(len(text) - self.n - 1):
                gram = tuple(text[max(0, i - self.n):i])

                if gram not in self.model:
                    self.model[gram] = {}
                self.model[gram][text[i + 1]] = self.model[gram].get(text[i + 1], 0) + 1
        
        del tokens

        return
