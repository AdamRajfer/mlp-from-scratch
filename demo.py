import json

from mlp.trainer import Trainer

if __name__ == "__main__":
    trainer = Trainer()
    with open(trainer.args.function_file, "rb") as file:
        trainer.run(**json.load(file))
