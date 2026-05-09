import random
from collections import Counter
from typing import Optional, Any, Tuple

class Model:
    def __init__(self) -> None:
        self.vocab: list[str] = []
        self.model: dict[Any, dict[str, int]] = {}

    def get_next(self, history: list[str]=[], seed=None) -> str:
        rng = random.Random()
        rng.seed(seed)


        return ""

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

    def create_bpe(self, texts: list[str], vocab_size:int=1024) -> None:
        if len(self.vocab) == vocab_size:
            return
        self.vocab = list(set("".join(texts)))
        while len(self.vocab) < vocab_size:
            tokens = []
            for text in texts:
                tokens.extend(self.tokenize(text))
            pairs = Counter(zip(tokens, tokens[1:]))
            del tokens
            try:
                most_common: str = pairs.most_common(1)[0][0] # type: ignore
                del pairs
            except Exception as e:
                print(e)
                return
            self.vocab.append(most_common)
        return

    def train(self, data_paths: list[str], vocab_size:int=1024, n: int = 12) -> None:
        texts = []
        for path in data_paths:
            with open(path, "r") as f:
                texts.append(f.read())
                texts[-1] = "<START>" + texts[-1] + "</START>"
        
        self.create_bpe(texts, vocab_size)

        tokens: list[list[str]] = []
        for text in texts:
            tokens.append(self.tokenize(text))

        del texts

        for text in tokens:
            for i in range(len(text) - n - 1):
                gram = tuple([None] * max(0, n - i - 1) + text[max(0, i - n):i])

                assert type(gram) is Tuple

                if gram not in self.model:
                    self.model[gram] = {}
                self.model[gram][text[i + 1]] = self.model[gram].get(text[i + 1], 0) + 1
        
        del tokens

        return
