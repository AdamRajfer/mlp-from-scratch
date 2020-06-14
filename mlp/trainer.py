import argparse
import math  # noqa
import random
from typing import Any, Dict

from mlp.dataset import Dataset
from mlp.generator import DataGenerator
from mlp.model import Model
from mlp.optimizer import SGD


class Trainer:
    def __init__(self, *args: str) -> None:
        self.args = self._parse_arguments(*args)

    def run(self, **kwargs: Any) -> None:
        if not self.args.not_load_parameters:
            for parameter_name, parameter_value in kwargs["model_parameters"].items():
                setattr(self.args, parameter_name, parameter_value)
        if self.args.verbose > 0:
            self._display_parameters(**kwargs)
        random.seed(self.args.random_state)
        data_generator = DataGenerator(
            fn=eval(kwargs["function"]),
            dataset_size=kwargs["dataset_size"],
            train_test_ratio=kwargs["train_test_ratio"],
        )
        if self.args.different_ranges:
            X_train, y_train, X_test, y_test = data_generator.train_test_split_different_ranges(
                train_ranges=kwargs["data_ranges"]["different"]["train"],
                test_ranges=kwargs["data_ranges"]["different"]["test"],
            )
        else:
            X_train, y_train, X_test, y_test = data_generator.train_test_split_same_ranges(
                ranges=kwargs["data_ranges"]["same"]
            )
        train_dataset = Dataset(X_train, y_train)
        test_dataset = Dataset(X_test, y_test)
        optimizer = SGD(
            start_learning_rate=self.args.start_learning_rate,
            end_learning_rate=self.args.end_learning_rate,
            momentum=self.args.momentum,
        )
        Model(
            layer_sizes=self.args.layer_sizes,
            optimizer=optimizer,
            loss=self.args.loss_function,
            patience=self.args.patience,
            min_delta=self.args.min_delta,
            display_freq=self.args.display_freq,
            verbose=self.args.verbose,
        ).fit(
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            epochs=self.args.epochs,
            batch_size=self.args.batch_size,
        )

    def _parse_arguments(self, *args: str) -> argparse.Namespace:
        parser = argparse.ArgumentParser(
            description="MLP LEARNING DEMO",
            formatter_class=lambda prog: argparse.ArgumentDefaultsHelpFormatter(prog, max_help_position=130),
        )
        parser.add_argument("function_file", type=str, help="json file with function to be learned")
        parser.add_argument(
            "-d",
            "--different-ranges",
            action="store_true",
            help="whether to train and test model on different data ranges",
        )
        parser.add_argument(
            "-n",
            "--not-load-parameters",
            action="store_true",
            help="whether not to load model parameters from json file",
        )
        parser.add_argument(
            "-L",
            "--layer-sizes",
            type=int,
            default=[],
            nargs="+",
            help="MLP layer sizes EXCLUDING first and last layer",
        )
        parser.add_argument(
            "-S", "--start-learning-rate", type=float, default=0.01, help="learning rate at the beginning of an epoch"
        )
        parser.add_argument(
            "-E", "--end-learning-rate", type=float, default=0.001, help="learning rate at the end of an epoch"
        )
        parser.add_argument("-M", "--momentum", type=float, default=0.8, help="momentum")
        parser.add_argument(
            "-l", "--loss-function", type=str, choices=["mse", "mae"], default="mae", help="loss function"
        )
        parser.add_argument("-e", "--epochs", type=int, default=20, help="number of training epochs")
        parser.add_argument("-b", "--batch-size", type=int, default=32, help="size of a training batch")
        parser.add_argument("-p", "--patience", type=int, default=5, help="patience before early stopping")
        parser.add_argument("-m", "--min-delta", type=float, default=0.01, help="early stopping sensivity")
        parser.add_argument(
            "-f",
            "--display-freq",
            type=float,
            default=0.2,
            help="frequency of displaying training loss during an epoch",
        )
        parser.add_argument("-v", "--verbose", type=int, default=3, help="verbosity mode")
        parser.add_argument("-r", "--random-state", type=int, default=42, help="random state")
        return parser.parse_args(args or None)

    def _display_parameters(self, **kwargs: Any) -> None:
        print("MLP LEARNING DEMO\n")
        print("function: {}".format(kwargs["function"]))
        print("dataset size: {}".format(kwargs["dataset_size"]))
        print("train test ratio: {}".format(kwargs["train_test_ratio"]))
        print("same data ranges: {}".format(not self.args.different_ranges))
        self._display_data_ranges(kwargs["data_ranges"])
        print("\nload model parameters from file: {}".format(not self.args.not_load_parameters))
        model_parameters = kwargs["model_parameters"]
        self._display_parameter("layer_sizes", model_parameters)
        self._display_parameter("start_learning_rate", model_parameters)
        self._display_parameter("end_learning_rate", model_parameters)
        self._display_parameter("momentum", model_parameters)
        self._display_parameter("loss_function", model_parameters)
        self._display_parameter("epochs", model_parameters)
        self._display_parameter("batch_size", model_parameters)
        self._display_parameter("patience", model_parameters)
        self._display_parameter("min_delta", model_parameters)
        self._display_parameter("display_freq", model_parameters)
        self._display_parameter("verbose", model_parameters)
        self._display_parameter("random_state", model_parameters)
        print()

    def _display_data_ranges(self, data_ranges: Dict[str, Any]) -> None:
        if self.args.different_ranges:
            print("train data ranges:")
            for k, v in data_ranges["different"]["train"].items():
                print("\t{}: {}".format(k, v))
            print("test data ranges:")
            for k, v in data_ranges["different"]["test"].items():
                print("\t{}: {}".format(k, v))
        else:
            print("data ranges:")
            for k, v in data_ranges["same"].items():
                print("\t{}: {}".format(k, v))

    def _display_parameter(self, name: str, model_parameters: Dict[str, Any]) -> None:
        is_loaded = name in model_parameters and not self.args.not_load_parameters
        print("{}{}: {}".format(name, is_loaded * " (loaded)", getattr(self.args, name)))
