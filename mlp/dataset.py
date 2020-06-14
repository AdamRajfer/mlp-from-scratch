import random
from typing import List, Tuple, Union, cast


class Dataset:
    def __init__(self, X: List[List[Union[int, float]]], y: List[Union[int, float]]) -> None:
        self._X = X
        self._y = y

    def __len__(self) -> int:
        return len(self._X)

    def __getitem__(self, i: int) -> Tuple[List[Union[int, float]], Union[int, float]]:
        return self._X[i], self._y[i]

    def iloc(self, start: int, end: int) -> "Dataset":
        return Dataset(self._X[start:end], self._y[start:end])

    def on_epoch_end(self) -> None:
        data_transposed = list(zip(self._X, self._y))
        random.shuffle(data_transposed)
        data = cast(Tuple[Tuple[List[Union[int, float]]], Tuple[Union[int, float], ...]], tuple(zip(*data_transposed)))
        self._X, self._y = list(data[0]), list(data[1])
