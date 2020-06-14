from typing import Union

from mlp.base import BaseValue


class Op:
    def __init__(self, out_value: BaseValue) -> None:
        self._out_value = out_value
        self.forward()

    def forward(self) -> None:
        pass

    def backward(self) -> None:
        pass


class AddOp(Op):
    def __init__(self, value: BaseValue, other_value: BaseValue, out_value: BaseValue) -> None:
        self._value = value
        self._other_value = other_value
        super().__init__(out_value)

    def forward(self) -> None:
        self._out_value.set_data(self._value.data + self._other_value.data)

    def backward(self) -> None:
        self._value.update_grad(self._out_value.grad)
        self._other_value.update_grad(self._out_value.grad)


class MulOp(Op):
    def __init__(self, value: BaseValue, other_value: BaseValue, out_value: BaseValue) -> None:
        self._value = value
        self._other_value = other_value
        super().__init__(out_value)

    def forward(self) -> None:
        self._out_value.set_data(self._value.data * self._other_value.data)

    def backward(self) -> None:
        self._value.update_grad(self._other_value.data * self._out_value.grad)
        self._other_value.update_grad(self._value.data * self._out_value.grad)


class PowOp(Op):
    def __init__(self, value: BaseValue, exp_value: Union[int, float], out_value: BaseValue) -> None:
        self._value = value
        self._exp_value = float(exp_value)
        super().__init__(out_value)

    def forward(self) -> None:
        self._out_value.set_data(self._value.data ** self._exp_value)

    def backward(self) -> None:
        self._value.update_grad((self._exp_value * self._value.data ** (self._exp_value - 1)) * self._out_value.grad)


class AbsOp(Op):
    def __init__(self, value: BaseValue, out_value: BaseValue) -> None:
        self._value = value
        super().__init__(out_value)

    def forward(self) -> None:
        self._out_value.set_data(abs(self._value.data))

    def backward(self) -> None:
        self._value.update_grad((1 if self._value.data > 0.0 else -1) * self._out_value.grad)


class ReLUOp(Op):
    def __init__(self, value: BaseValue, out_value: BaseValue) -> None:
        self._value = value
        super().__init__(out_value)

    def forward(self) -> None:
        self._out_value.set_data(max(self._value.data, 0.0))

    def backward(self) -> None:
        self._value.update_grad((self._value.data > 0.0) * self._out_value.grad)
