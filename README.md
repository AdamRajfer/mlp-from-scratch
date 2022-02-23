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

## Opis programu wykonywalnego

Program wykonywalny *demo.py* znajduje się w głównym katalogu projektu. W celu zapoznania się z opcjami programu, można użyć następującej komendy:

```
python demo.py --help
```

Zostanie wyświetlona następująca instrukcja:

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

Aby uruchomić program, należy przekazać mu jako pierwszy argument ścieżkę do pliku JSON:

```
python demo.py <your-function-file-name>.json
```

Spowoduje to uruchomienie programu z parametrami, zawartymi w przekazanym pliku.

Domyślnie zbiór danych zostanie wygenerowany ze zdefiniowanego wspólnego rozkładu danych, a następnie zostanie dokonany podział treningowo-testowy danych. Jeżeli użytkownik zechce wykonać trening na danych treningowych wygenerowanych ze zdefiniowanego przedziału treningowego, a testowanie na danych testowych wygenerowanych ze zdefiniowanego przedziału testowego, wtedy powinien przekazać flagę *--different-ranges*:

```
python demo.py <your-function-file-name>.json --different-ranges
```

Domyślnie parametry modelu zostaną załadowane z pola *model_parameters* pliku wsadowego. Parametrom nieobecnym w tym polu zostaną przypisane wartości domyślne. Jeżeli użytkownik zechce zrezygnować z przekazywania modelowi parametrów z pliku, wtedy powinien przekazać flagę *--not-load-parameters*. Wtedy będzie miał manualną kontrolę nad przekazywanymi do modelu parametrami:

```
python demo.py <your-function-file-name>.json --not-load-parameters
```

Pozostałe parametry programu zostały wcześniej omówione i ich wartości mogą być modyfikowane przez przypisanie ich flagom odpowiednich wartości. Przykładowo, jeżeli użytkownik zechce przetrenować model:
- na pliku *3-arg-function.json*,
- wytrenować go na wspólnym rozkładzie danych,
- nie ładować parametrów z pliku,
- zastosować rozmiary warstw: [10, 20, 5],
- funkcję kosztu MSE,
- liczbę epok treningowych równą 15,
- pozostałe parametry domyślne,

wtedy powinien wpisać następującą komendę:

```
python demo.py 3-arg-function.json --not-load-parameters -L 10 20 5 -l mae -e 15
```

Po uruchomieniu programu zostanie wyświetlona zastosowana konfiguracja parametrów, a w dalszej kolejności wypisane zostaną wyniki kolejnych etapów uczenia. Model zakończy działanie w dwóch przypadkach:

- zostaną wykonane wszystkie epoki treningowe,
- nastąpi *early stopping* ze względu na brak poprawy jakości modelu.

Po zakończeniu działania programu wyświetlona zostanie osiągnięta jakość aproksymacji funkcji przez model zarówno na zbiorze treningowym, jak i na zbiorze testowym.

Użytkownik może kontrolować ilość wyświetlanych informacji podczas treningu za pomocą flagi *--verbose*:

 - wartość 0 oznacza brak wyświetlania jakichkolwiek informacji,
 - wartość 1 oznacza wyświetlanie jedynie konfiguracji parametrów oraz końcowego wyniku,
 - wartość 2 oznacza wyświetlanie konfiguracji parametrów, końcowego wyniku oraz wyników po kolejnych epokach,
 - wartość 3 oznacza wyświetlanie wszystkich informacji: konfiguracji parametrów, końcowego wyniku, wyników po kolejnych epokach oraz wyników w trakcie trwania epoki.

Domyślnie ustawiona jest wartość 3.

## Uruchomienie eksperymentów

Poniższe komendy umożliwią odtworzenie przeprowadzonych eksperymentów dla konfiguracji, w których jakość aproksymacji funkcji była największa.

### Funkcja 3-argumentowa

<img src="https://render.githubusercontent.com/render/math?math=f(x)=0.24x_1^2x_3-1.05x_2%2Bx_2x_3^2%2B0.25x_1x_2-0.03x_2^3-x_1x_2x_3" height="25">

```
python demo.py 3-arg-function.json
```

### Funkcja 5-argumentowa

<img src="https://render.githubusercontent.com/render/math?math=f(x)=10sin(x_1x_2\pi)%2B20x_3%2B10x_1x_4%2B5x_5^2-10" height="25">

```
python demo.py 5-arg-function.json
```

### Funkcja 7-argumentowa

<img src="https://render.githubusercontent.com/render/math?math=f(x)=x_1^2%2B3x_2%2Bx_3x_4%2B2x_5-1.5x_6x_7" height="25">

```
python demo.py 7-arg-function.json
```

## Uruchomienie testów

Poniższa komenda uruchomi testy sprawdzające, czy propagacja wprzód i propagacja wsteczna działają poprawnie, a także test integracyjny sprawdzający, czy program wykonuje się poprawnie.

```
python test.py
```

## Podziękowania

Biblioteka [micrograd](https://github.com/karpathy/micrograd), autorstwa [Andreja Karpathy'ego](https://github.com/karpathy), była pomocna podczas pracy nad projektem.
