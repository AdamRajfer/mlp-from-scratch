from typing import Iterable, List, Optional, Tuple, Union

from mlp.dataset import Dataset
from mlp.engine import Value
from mlp.losses import absolute_error, squared_error
from mlp.nn import MLP
from mlp.optimizer import SGD


class Model:
    def __init__(
        self,
        layer_sizes: List[int],
        optimizer: SGD,
        loss: str,
        patience: int,
        min_delta: float,
        display_freq: int,
        verbose: int,
    ) -> None:
        self._layer_sizes = layer_sizes
        self._optimizer = optimizer
        self._patience = patience
        self._min_delta = min_delta
        self._display_freq = display_freq
        self._verbose = verbose
        self._mlp: Optional[MLP] = None
        if loss == "mse":
            self._loss_fn = squared_error
        elif loss == "mae":
            self._loss_fn = absolute_error

    def __call__(self, X: Iterable[Iterable[Union[int, float]]]) -> List[Value]:
        return self._predict(X)

    def fit(self, train_dataset: Dataset, test_dataset: Dataset, epochs: int, batch_size: int) -> "Model":
        if self._mlp is None:
            self._mlp = MLP(len(train_dataset[0][0]), self._layer_sizes + [1])
        num_steps = int(len(train_dataset) / batch_size)
        display_after = int(self._display_freq * num_steps)
        self._optimizer.compile(num_steps, self._mlp.parameters)
        best_epoch, best_train_loss, best_test_loss, waiting = 0, float("inf"), float("inf"), 0
        for epoch in range(epochs):
            self._print_before_epoch(epoch, epochs)
            train_loss = 0.0
            for step in range(num_steps):
                train_loss += self._step(train_dataset, step, batch_size)
                self._print_after_step(step, train_loss, display_after)
            train_loss /= num_steps
            test_loss = self.score(test_dataset)
            best_epoch, best_train_loss, best_test_loss, waiting = self._early_stopping(
                best_epoch, best_train_loss, best_test_loss, waiting, epoch, train_loss, test_loss
            )
            train_dataset.on_epoch_end()
            self._optimizer.on_epoch_end()
            self._print_after_epoch(train_loss, test_loss)
            if waiting == self._patience:
                self._print_if_early_stopping(best_epoch)
                break
        self._print_after_training(best_train_loss, best_test_loss)
        return self

    def predict(self, X: Iterable[Iterable[Union[int, float]]]) -> List[float]:
        return list(map(self.predict_one, X))

    def predict_one(self, x: Iterable[Union[int, float]]) -> float:
        return self._predict_one(x).data

    def score(
        self, X: Union[Iterable[Iterable[Union[int, float]]], Dataset], labels: Iterable[Union[int, float]] = ()
    ) -> float:
        return self._score(X, labels).data

    def score_one(self, x: Iterable[Union[int, float]], label: Union[int, float]) -> float:
        return self._score_one(x, label).data

    def _predict(self, X: Iterable[Iterable[Union[int, float]]]) -> List[Value]:
        return list(map(self._predict_one, X))

    def _predict_one(self, x: Iterable[Union[int, float]]) -> Value:
        assert self._mlp is not None
        return self._mlp(list(map(Value, x)))

    def _score(
        self, X: Union[Iterable[Iterable[Union[int, float]]], Dataset], labels: Iterable[Union[int, float]] = ()
    ) -> Value:
        dataset = X if isinstance(X, Dataset) else zip(X, labels)
        scores = [self._score_one(x, label) for x, label in dataset]
        return sum(scores) / len(scores)

    def _score_one(self, x: Iterable[Union[int, float]], label: Union[int, float]) -> Value:
        return self._loss_fn(Value(label), self._predict_one(x))

    def _step(self, dataset: Dataset, step: int, batch_size: int) -> float:
        batch = dataset.iloc(step * batch_size, (step + 1) * batch_size)
        batch_loss = self._score(batch)
        batch_loss.backward()
        self._optimizer.update_parameters()
        return batch_loss.data

    def _early_stopping(
        self,
        best_epoch: int,
        best_train_loss: float,
        best_test_loss: float,
        waiting: int,
        epoch: int,
        train_loss: float,
        test_loss: float,
    ) -> Tuple[int, float, float, int]:
        if best_test_loss - test_loss > self._min_delta:
            return epoch, train_loss, test_loss, 0
        return best_epoch, best_train_loss, best_test_loss, waiting + 1

    def _print_before_epoch(self, epoch: int, epochs: int) -> None:
        if self._verbose > 1:
            print("Epoch {} / {}".format(epoch + 1, epochs))

    def _print_after_epoch(self, train_loss: float, test_loss: float) -> None:
        if self._verbose > 1:
            verbose_format = "{}\tTRAIN_LOSS: {:.6f}\n\tTEST_LOSS: {:.6f}\n"
            print(verbose_format.format("\n" * (self._verbose > 2), train_loss, test_loss))

    def _print_after_step(self, step: int, loss: float, display_after: int) -> None:
        if step % display_after == 0 and self._verbose > 2:
            print("\tstep: {}, train_loss: {:.6f}".format(step, loss / (step + 1)))

    def _print_if_early_stopping(self, best_epoch: int) -> None:
        if self._verbose > 0:
            print("Early stopping on epoch {}".format(best_epoch))

    def _print_after_training(self, train_loss: float, test_loss: float) -> None:
        if self._verbose > 0:
            print("TRAIN_LOSS: {:.6f}\nTEST_LOSS: {:.6f}".format(train_loss, test_loss))
