from typing import List

from mlp.engine import Value


class SGD:
    def __init__(self, start_learning_rate: float, end_learning_rate: float, momentum: float) -> None:
        self._start_learning_rate = start_learning_rate
        self._end_learning_rate = end_learning_rate
        self._momentum = momentum
        self._current_lr = start_learning_rate
        self._iterations = 0
        self._velocities: List[float] = []
        self._parameters: List[Value] = []

    def compile(self, iterations: int, parameters: List[Value]) -> None:
        self._iterations = iterations
        self._parameters = parameters

    def update_parameters(self) -> None:
        if not self._velocities:
            for parameter in self._parameters:
                parameter.update_data(-self._current_lr * parameter.grad)
                self._velocities.append(parameter.grad)
                parameter.set_grad(0)
        else:
            for i, parameter in enumerate(self._parameters):
                self._velocities[i] = self._momentum * self._velocities[i] + (1 - self._momentum) * parameter.grad
                parameter.update_data(-self._current_lr * self._velocities[i])
                parameter.set_grad(0)
        self._current_lr -= (self._start_learning_rate - self._end_learning_rate) / self._iterations

    def on_epoch_end(self) -> None:
        self._current_lr = self._start_learning_rate
