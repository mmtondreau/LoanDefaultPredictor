import os.path as ops

from lightning import LightningApp
from lightning_training_studio import Sweep
from lightning_training_studio.algorithm import RandomSearch
from lightning_training_studio.distributions import Categorical, IntUniform, LogUniform

app = LightningApp(
    Sweep(
        script_path=ops.join(ops.dirname(__file__), "ldp_cli.py"),
        total_experiments=3,
        parallel_experiments=1,
        algorithm=RandomSearch(
            distributions={
                "--model.learning_rate": LogUniform(0.001, 0.01),
                "--model.hidden_units": Categorical(["[32,12]", "[12]"]),
                "--data.batch_size": Categorical([32, 64]),
                "--trainer.max_epochs": IntUniform(1, 3),
            }
        ),
    )
)
