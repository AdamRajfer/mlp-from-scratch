from test_mlp.test_engine import test_value_multiple_inputs, test_value_single_input
from test_mlp.test_trainer import test_trainer

if __name__ == "__main__":
    print("Testing class Value with single input...")
    test_value_single_input()
    print("...passed successfully!\n")
    print("Testing class Value with multiple inputs...")
    test_value_multiple_inputs()
    print("...passed successfully!\n")
    print("Testing class Trainer...")
    test_trainer()
    print("...passed successfully!")
