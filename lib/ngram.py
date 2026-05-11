import random
from collections import Counter, defaultdict
from typing import Optional
from tqdm import tqdm
from itertools import pairwise
import re
from numba import njit

class Model:
    def __init__(self, n: int=24, vocab_size: int=8096) -> None:
        self.n = n
        self.vocab_size = vocab_size
        self.vocab: list[str] = []
        self.model: dict[tuple[str, ...], dict[str, int]] = {}
        self.start_token = "♍"
        self.end_token = "♎"

    def get_next(self, history: list[str]=[], rng: Optional[random.Random] = None) -> str:
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

        return rng.choice(possible) if rng else random.choice(possible)

    def tokenize(self, text: str) -> list[str]:
        if not self.vocab:
            return [text]
        elif max(map(len, self.vocab)) == 1:
            return list(text)

        tokens = []
        vocab_sorted = sorted(self.vocab, key=len, reverse=True)
        with tqdm(total=len(text), desc="Tokenizing") as pbar:
            idx = 0
            lidx = idx
            while text:
                for v in vocab_sorted:
                    if text[idx:].startswith(v):
                        tokens.append(v)
                        idx += (len_v := len(v))
                        pbar.update(len_v)
                        break
                if idx == lidx:
                    break
                lidx = idx
        return tokens

    def create_bpe(self, texts: list[str]) -> None:
        if len(self.vocab) == self.vocab_size:
            return
            
        self.vocab = sorted(set("".join(texts)))
        self.vocab += [self.start_token, self.end_token]

        tokens = [self.tokenize(t) for t in texts]
        
        if self.vocab_size - len(self.vocab) <= 0:
            return

        with tqdm(total=self.vocab_size, desc="Building BPE", initial=len(self.vocab)) as pbar:
            while len(self.vocab) < self.vocab_size:
                pairs = Counter()
                for seq in tokens:
                    if len(seq) > 1:
                        pairs.update(pairwise(seq))
                        
                if not pairs:
                    break
                
                (first, second), freq = pairs.most_common(1)[0]
                if freq < 2:
                    break

                new_token_seq = []
                new_token = first + second
                for seq in tokens:
                    merged = []
                    i = 0
                    while i < (len_seq := len(seq)):
                        if (seq_i := seq[i]) == first and i + 1 < len_seq and seq[i+1] == second:
                            merged.append(new_token)
                            i += 2
                        else:
                            merged.append(seq_i)
                            i += 1
                    new_token_seq.append(merged)
                tokens = new_token_seq

                self.vocab.append(first + second)
                pbar.update(1)

        print("VOCAB", self.vocab, "VOCAB")

    def pre_process(self, text: str) -> str:
        text = re.sub(r"\s+", " ", text, flags=re.MULTILINE)
        text = ". ".join([t.capitalize() for t in text.split(". ")])
        text.replace(self.start_token, "")
        text.replace(self.end_token, "")
        text = text.strip()
        return text

    def train(self, data_paths: list[str]) -> None:
        texts = []
        with tqdm(total=len(data_paths), desc="Loading files") as pbar:
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

        tokens: list[list[str]] = [self.tokenize(t) for t in texts]
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
