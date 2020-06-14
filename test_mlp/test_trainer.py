from typing import Any, Dict

from mlp.trainer import Trainer


def get_kwargs() -> Dict[str, Any]:
    return {
        "function": "lambda x1, x2, x3: x1 + x2 + x3 + 5.0",
        "dataset_size": 64,
        "train_test_ratio": 0.5,
        "data_ranges": {
            "same": {"x1": [-5, 5], "x2": [-5, 5], "x3": [-5, 5]},
            "different": {
                "train": {"x1": [-5, 5], "x2": [-5, 5], "x3": [-5, 5]},
                "test": {"x1": [15, 25], "x2": [15, 25], "x3": [15, 25]},
            },
        },
        "model_parameters": {"batch_size": 4, "epochs": 3, "verbose": 0},
    }


def test_trainer() -> None:
    Trainer("").run(**get_kwargs())
