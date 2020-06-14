import random
from typing import Callable, Dict, List, Tuple, Union


class DataGenerator:
    def __init__(self, fn: Callable, dataset_size: int, train_test_ratio: float) -> None:
        self._fn = fn
        self._dataset_size = dataset_size
        self._train_test_ratio = train_test_ratio
        self._train_size = int(dataset_size * train_test_ratio)

    def _generate_data(
        self, dataset_size: int, ranges: Dict[str, Tuple[Union[int, float], Union[int, float]]]
    ) -> Tuple[List[List[Union[int, float]]], List[Union[int, float]]]:
        X = [[random.uniform(*ranges[k]) for k in sorted(ranges)] for _ in range(dataset_size)]
        y = [self._fn(*x) for x in X]
        return X, y

    def train_test_split_same_ranges(
        self, ranges: Dict[str, Tuple[Union[int, float], Union[int, float]]]
    ) -> Tuple[
        List[List[Union[int, float]]], List[Union[int, float]], List[List[Union[int, float]]], List[Union[int, float]]
    ]:
        X, y = self._generate_data(self._dataset_size, ranges)
        size = self._train_size
        return X[:size], y[:size], X[size:], y[size:]

    def train_test_split_different_ranges(
        self,
        train_ranges: Dict[str, Tuple[Union[int, float], Union[int, float]]],
        test_ranges: Dict[str, Tuple[Union[int, float], Union[int, float]]],
    ) -> Tuple[
        List[List[Union[int, float]]], List[Union[int, float]], List[List[Union[int, float]]], List[Union[int, float]]
    ]:
        X_train, y_train = self._generate_data(self._train_size, train_ranges)
        X_test, y_test = self._generate_data(self._dataset_size - self._train_size, test_ranges)
        return X_train, y_train, X_test, y_test
