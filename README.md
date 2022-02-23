# MLP network for function approximation: implementation and experiments

The goal of the project was to develop a program to study the impact of number of MLP network layers, as well as the number of neurons in individual layers, on the quality of function approximation. Moreover, the influence of the parameterization of the backpropagation algorithm on the value of the model error was investigated. The research was carried out for three nonlinear multi-argument functions returning real numbers (number of arguments: 3, 5 and 7, respectively). The program also enables tracking the accuracy of approximation for successive epochs and learning iterations.

The program has been implemented using *only* Python 3 standard library.

## Input file

In order to run the program, you need to pass a JSON file to it. The file must have the following structure:

```yaml
{
  "function": "lambda x1, x2, x3: x1 + x2 + x3 + 5.0",
  "dataset_size": 8192,
  "train_test_ratio": 0.75,
  "data_ranges": {
    "same": {
      "x1": [-8, 15],
      "x2": [-3, 28],
      "x3": [4, 10]
    },
    "different": {
      "train": {
        "x1": [13, 48],
        "x2": [2, 7],
        "x3": [-4, 18]
      },
      "test": {
        "x1": [-2, 6],
        "x2": [18, 27],
        "x3": [19, 35]
      }
    }
  },
  "model_parameters": {
    "layer_sizes": [5, 3],
    "start_learning_rate": 0.01,
    "end_learning_rate": 0.001,
    "momentum": 0.8,
    "loss_function": "mse",
    "epochs": 15,
    "batch_size": 32,
    "patience": 5,
    "min_delta": 0.005,
    "display_freq": 0.25,
    "verbose": 2,
    "random_state": 42
  }
}
```
- The *function* field must contain the definition of the given function in the form of a lambda expression stored in the form of a text variable. Function arguments must be named [x1, x2, ..., x*n*], where *n* is the number of arguments. This field must be assigned a value.
- The *dataset_size* field means the total number of pairs (**x**, y) that will be generated. This field must be assigned a value.
- The *train_test_ratio* field determines the percentage of all pairs that will be assigned to the training set. This field must be assigned a value.
- The *data_ranges* field defines the intervals from which the function arguments will be drawn. This field must be assigned a value. It has the following fields:
  - the *same* field defines the ranges of arguments that will be randomized, same for training and testing data,
  - the *different* field defines the ranges of arguments that will be randomized, different for training and testing data:
	  - the *train* field means training intervals,
	  - the *test* field means testing intervals.
- The *model_parameters* field defines the default values of the model parameters. Here you can put keys such as: *layer_sizes*, *start_learning_rate*, *end_learning_rate*, *momentum*, *loss_function*, *epochs*, *batch_size*, *patience*, *min_delta*, *display_freq*, *verbose* and *random_state*. You don't have to define any parameter (pass an empty dictionary). By default, all model parameters defined in this field will be loaded. If any parameter is not defined in this field, then it will take the default value.

## Description of the executable

The *demo.py* executable is located in the main project directory. In order to see the program options, you can use the following command:

```
python demo.py --help
```

The following instruction is displayed:

```
usage: demo.py [-h] [-d] [-n] [-L LAYER_SIZES [LAYER_SIZES ...]] [-S START_LEARNING_RATE] [-E END_LEARNING_RATE] [-M MOMENTUM] [-l {mse,mae}] [-e EPOCHS] [-b BATCH_SIZE] [-p PATIENCE] [-m MIN_DELTA]
               [-f DISPLAY_FREQ] [-v VERBOSE] [-r RANDOM_STATE]
               function_file

MLP LEARNING DEMO

positional arguments:
  function_file                                                                  json file with function to be learned

optional arguments:
  -h, --help                                                                     show this help message and exit
  -d, --different-ranges                                                         whether to train and test model on different data ranges (default: False)
  -n, --not-load-parameters                                                      whether not to load model parameters from json file (default: False)
  -L LAYER_SIZES [LAYER_SIZES ...], --layer-sizes LAYER_SIZES [LAYER_SIZES ...]  MLP layer sizes EXCLUDING first and last layer (default: [])
  -S START_LEARNING_RATE, --start-learning-rate START_LEARNING_RATE              learning rate at the beginning of an epoch (default: 0.01)
  -E END_LEARNING_RATE, --end-learning-rate END_LEARNING_RATE                    learning rate at the end of an epoch (default: 0.001)
  -M MOMENTUM, --momentum MOMENTUM                                               momentum (default: 0.8)
  -l {mse,mae}, --loss-function {mse,mae}                                        loss function (default: mae)
  -e EPOCHS, --epochs EPOCHS                                                     number of training epochs (default: 20)
  -b BATCH_SIZE, --batch-size BATCH_SIZE                                         size of a training batch (default: 32)
  -p PATIENCE, --patience PATIENCE                                               patience before early stopping (default: 5)
  -m MIN_DELTA, --min-delta MIN_DELTA                                            early stopping sensivity (default: 0.01)
  -f DISPLAY_FREQ, --display-freq DISPLAY_FREQ                                   frequency of displaying training loss during an epoch (default: 0.2)
  -v VERBOSE, --verbose VERBOSE                                                  verbosity mode (default: 3)
  -r RANDOM_STATE, --random-state RANDOM_STATE                                   random state (default: 42)
```

In order to run the program, pass the path to the JSON file as the first argument:

```
python demo.py <your-function-file-name>.json
```

This will start the program with the parameters contained in the uploaded file.

By default, the dataset will be generated from the defined common data distribution, and then the train-test split of the data will be performed. If the user wants to perform training on training data generated from the defined training interval, and testing on testing data generated from the defined testing interval, then they should pass the *- different-ranges* flag:

```
python demo.py <your-function-file-name>.json --different-ranges
```

By default, model parameters will be loaded from the *model_parameters* field of the input file. Parameters that are not present in this field will be assigned default values. If the user decides not to pass parameters to the model from the file, then they should pass the *--not-load-parameters* flag. Then they will have the manual control over the parameters passed to the model:

```
python demo.py <your-function-file-name>.json --not-load-parameters
```

The remaining parameters of the program have been discussed earlier and their values can be modified by assigning appropriate values to the flags. For example, if the user wants to train the model:

- on the input file named *3-arg-function.json*,
- train it on the common data distribution,
- do not load the parameters from the file,
- use layers with sizes: [10, 20, 5],
- use MSE cost function,
- train model on 15 epochs,
- set the remaining parameters to default,

then they should type the following command:

```
python demo.py 3-arg-function.json --not-load-parameters -L 10 20 5 -l mae -e 15
```

After starting the program, the used parameter configuration will be displayed, and then the results of the subsequent learning stages will be listed. The model will end in two cases:

- all training periods will be performed,
- *early stopping* will occur due to the lack of improvement in the quality of the model.

After the program is finished, the achieved quality of approximation of the function by the model, both on the training set and on the testing set, will be displayed.

The user can control the amount of information displayed during training using the *--verbose* flag:

 - value 0 means that no information is displayed,
 - value 1 means that only the parameter configuration and the final result are displayed,
 - value 2 means displaying the configuration of parameters, the final result and the results after subsequent epochs,
 - value 3 means displaying all information: parameter configuration, final result, results after successive epochs and results during the epoch.

The default is 3.

## Launching the experiments

The following commands will allow you to recreate the experiments performed for the configurations in which the quality of the function approximation was the highest.

### 3-argument function

<img src="https://render.githubusercontent.com/render/math?math=f(x)=0.24x_1^2x_3-1.05x_2%2Bx_2x_3^2%2B0.25x_1x_2-0.03x_2^3-x_1x_2x_3" height="25">

```
python demo.py 3-arg-function.json
```

### 5-argument function

<img src="https://render.githubusercontent.com/render/math?math=f(x)=10sin(x_1x_2\pi)%2B20x_3%2B10x_1x_4%2B5x_5^2-10" height="25">

```
python demo.py 5-arg-function.json
```

### 7-argument function

<img src="https://render.githubusercontent.com/render/math?math=f(x)=x_1^2%2B3x_2%2Bx_3x_4%2B2x_5-1.5x_6x_7" height="25">

```
python demo.py 7-arg-function.json
```

## Running tests

The following command will run tests to ensure that forward and backpropagation are working properly, as well as an integration test to verify that the program is executing correctly.

```
python test.py
```

## Acknowledgments

The [micrograd](https://github.com/karpathy/micrograd) library by [Andrej Karpathy](https://github.com/karpathy) was helpful in working on the project.
