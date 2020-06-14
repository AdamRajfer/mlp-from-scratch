from mlp.engine import Value


def squared_error(y_true: Value, y_pred: Value) -> Value:
    return (y_true - y_pred) ** 2


def absolute_error(y_true: Value, y_pred: Value) -> Value:
    return abs(y_true - y_pred)
