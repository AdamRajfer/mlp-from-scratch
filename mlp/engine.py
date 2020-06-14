from typing import List, Set, Tuple, Union

from mlp.base import BaseValue
from mlp.op import AbsOp, AddOp, MulOp, Op, PowOp, ReLUOp


class Value(BaseValue):
    def __init__(
        self, data: Union[int, float] = 0, op: str = "", children: Tuple[Union["Value", Union[int, float]], ...] = ()
    ) -> None:
        super().__init__(data)
        self._op = self._initialize_op(op, children)
        self._children: Set[Value] = set(x for x in children if isinstance(x, Value))

    def __add__(self, other: Union["Value", Union[int, float]]) -> "Value":
        if not isinstance(other, Value):
            other = Value(other)
        return Value(op="+", children=(self, other))

    def __mul__(self, other: Union["Value", Union[int, float]]) -> "Value":
        if not isinstance(other, Value):
            other = Value(other)
        return Value(op="*", children=(self, other))

    def __pow__(self, other: Union[int, float]) -> "Value":
        return Value(op="**", children=(self, other))

    def __abs__(self) -> "Value":
        return Value(op="abs", children=(self,))

    def __neg__(self) -> "Value":
        return self * -1

    def __radd__(self, other: Union["Value", Union[int, float]]) -> "Value":
        return self + other

    def __sub__(self, other: Union["Value", Union[int, float]]) -> "Value":
        return self + (-other)

    def __rsub__(self, other: Union["Value", Union[int, float]]) -> "Value":
        return other + (-self)

    def __rmul__(self, other: Union["Value", Union[int, float]]) -> "Value":
        return self * other

    def __truediv__(self, other: Union["Value", Union[int, float]]) -> "Value":
        return self * other ** -1

    def __rtruediv__(self, other: Union["Value", Union[int, float]]) -> "Value":
        return other * self ** -1

    def relu(self) -> "Value":
        return Value(op="ReLU", children=(self,))

    def backward(self) -> None:
        topology: List[Value] = []
        visited_values: Set[Value] = set()
        self._find_topology(topology, visited_values)
        self.set_grad(1)
        for value in reversed(topology):
            value._op.backward()

    def _find_topology(self, topology: List["Value"], visited_values: Set["Value"]) -> None:
        if self not in visited_values:
            visited_values.add(self)
            for child in self._children:
                child._find_topology(topology, visited_values)
            topology.append(self)

    def _initialize_op(self, op: str, children: Tuple[Union["Value", Union[int, float]], ...]) -> Op:
        if op == "+":
            value, other_value = children
            return AddOp(value=value, other_value=other_value, out_value=self)
        if op == "*":
            value, other_value = children
            return MulOp(value=value, other_value=other_value, out_value=self)
        if op == "**":
            value, exp_value = children
            return PowOp(value=value, exp_value=exp_value, out_value=self)
        if op == "abs":
            (value,) = children
            return AbsOp(value=value, out_value=self)
        if op == "ReLU":
            (value,) = children
            return ReLUOp(value=value, out_value=self)
        return Op(out_value=self)
