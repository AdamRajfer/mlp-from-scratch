import random
from abc import ABC, abstractmethod
from typing import Iterable, List, Union

from mlp.engine import Value


class Module(ABC):
    @abstractmethod
    def __call__(self, x: Iterable[Value]) -> Union[Value, List[Value]]:
        pass

    @property
    @abstractmethod
    def parameters(self) -> List[Value]:
        pass


class Neuron(Module):
    def __init__(self, input_size: int, linear: bool) -> None:
        self.input_size = input_size
        self.linear = linear
        self.w = [Value(random.uniform(-1, 1)) for _ in range(input_size)]
        self.b = Value(0)

    def __call__(self, x: Iterable[Value]) -> Value:
        z = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return z.relu() if not self.linear else z

    @property
    def parameters(self) -> List[Value]:
        return self.w + [self.b]


class Layer(Module):
    def __init__(self, input_size: int, output_size: int, linear) -> None:
        self.input_size = input_size
        self.output_size = output_size
        self.linear = linear
        self.neurons = [Neuron(input_size, linear) for _ in range(output_size)]

    def __call__(self, x: Iterable[Value]) -> Union[Value, List[Value]]:
        out = [neuron(x) for neuron in self.neurons]
        return out[0] if len(out) == 1 else out

    @property
    def parameters(self) -> List[Value]:
        return [parameter for neuron in self.neurons for parameter in neuron.parameters]


class MLP(Module):
    def __init__(self, input_size: int, layer_sizes: List[int]) -> None:
        self.input_size = input_size
        self.layer_sizes = layer_sizes
        self.sizes = [input_size] + layer_sizes
        self.layers = [
            Layer(self.sizes[i], self.sizes[i + 1], i == len(layer_sizes) - 1) for i in range(len(self.layer_sizes))
        ]

    def __call__(self, x: Iterable[Value]) -> Union[Value, List[Value]]:
        for layer in self.layers:
            x = layer(x)
        return x

    @property
    def parameters(self) -> List[Value]:
        return [parameter for layer in self.layers for parameter in layer.parameters]
