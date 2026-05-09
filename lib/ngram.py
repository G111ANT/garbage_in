import random

class Model:
    def __init__(self) -> None:
        pass

    def get_next(self, history: list[str]=[], seed=None) -> str:
        rng = random.Random()
        rng.seed(seed)
        return ""

    def train(self, data_paths: list[str]) -> None:
        