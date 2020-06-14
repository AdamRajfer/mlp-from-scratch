from mlp.engine import Value


def test_value_single_input() -> None:
    x = Value(-4)
    z = 2 * x + 2 + x
    q = z.relu() + z * x
    h = (z ** 2).relu()
    y = h + q + q * x
    y.backward()
    assert abs(y.data - -20.0) < 1e-6
    assert abs(x.grad - 46.0) < 1e-6


def test_value_multiple_inputs() -> None:
    a = Value(-4)
    b = Value(2)
    c = a + b
    d = a * b + b ** 3
    c += c + 1
    c += 1 + c - a
    d += d * 2 + (b + a).relu()
    d += 3 * d + (b - a).relu()
    e = c - d
    f = e ** 2
    g = f / 2
    g += 10 / f
    g.backward()
    assert abs(g.data - 24.704081633) < 1e-6
    assert abs(a.grad - 138.833819242) < 1e-6
    assert abs(b.grad - 645.577259475) < 1e-6
