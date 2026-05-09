import random
from collections import Counter
from typing import Optional, Any, Tuple

class Model:
    def __init__(self, n: int=12, vocab_size: int=1024) -> None:
        self.n = n
        self.vocab_size = vocab_size
        self.vocab: list[str] = []
        self.model: dict[Any, dict[str, int]] = {}

    def get_next(self, history: list[str]=[], seed=None) -> Optional[str]:
        rng = random.Random()
        rng.seed(seed)

        gram = tuple(history[:-self.n])

        possible = self.vocab.copy()
        if gram in self.model:
            for text in self.model[gram]:
                possible.extend([text] * self.model[gram][text])
            
        else:
            for gram_ in self.model:
                if " ".join(gram).endswith(" ".join(gram_)):
                    for text in self.model[gram_]:
                        possible.extend([text] * self.model[gram_][text])

        return rng.choice(possible) if len(possible) else None

    def tokenize(self, text: str) -> list[str]:
        if len(self.vocab) == 0:
            return [text]
        tokens = []
        while len(text) > 0:
            for v in sorted(self.vocab, key=len, reverse=True):
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
        while len(self.vocab) < self.vocab_size:
            pairs = Counter()
            for text in texts:
                tokens = self.tokenize(text)
                pairs.update(zip(tokens, tokens[1:]))
                del tokens
            if len(pairs) == 0:
                return
            most_common = pairs.most_common(1)[0]
            del pairs
            if most_common[1] < 2:
                return
            self.vocab.append(most_common[0][0] + most_common[0][1])
        return

    def train(self, data_paths: list[str]) -> None:
        texts = []
        for path in data_paths:
            with open(path, "r") as f:
                texts.append(f.read())
                texts[-1] = "<START>" + texts[-1] + "</START>"

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
