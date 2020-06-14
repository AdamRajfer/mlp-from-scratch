from typing import Union


class BaseValue:
    def __init__(self, data: Union[int, float] = 0) -> None:
        self.set_data(data)
        self.set_grad(0)

    def set_data(self, data: Union[int, float]) -> None:
        self.data = float(data)

    def set_grad(self, grad: Union[int, float]) -> None:
        self.grad = float(grad)

    def update_data(self, value: Union[int, float]) -> None:
        self.data += value

    def update_grad(self, value: Union[int, float]) -> None:
        self.grad += value
