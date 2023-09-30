from ray.train.lightning import (
    RayDDPStrategy,
    RayFSDPStrategy,
    RayLightningEnvironment,
    RayTrainReportCallback,
    prepare_trainer,
)
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.train import RunConfig, ScalingConfig, CheckpointConfig
from ray.train.torch import TorchTrainer

from torch_loan_default.ldp_data_module import LDPDataModule
from torch_loan_default.ldp_lit_module import LDPLitModule
import lightning.pytorch as pl

default_config = {
    "hidden_units": [32, 12],
    "learning_rate": 1e-3,
}


def train_func(config):
    dm = LDPDataModule(batch_size=config["batch_size"])
    model = LDPLitModule(config)

    trainer = pl.Trainer(
        devices="auto",
        accelerator="cpu",
        strategy=RayDDPStrategy(),
        callbacks=[RayTrainReportCallback()],
        plugins=[RayLightningEnvironment()],
        enable_progress_bar=False,
    )
    trainer = prepare_trainer(trainer)
    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    search_space = {
        "hidden_units": tune.choice([[32], [64], [128], [32, 12]]),
        "learning_rate": tune.loguniform(1e-4, 1e-1),
        "batch_size": tune.choice([32, 64]),
    }
    # The maximum training epochs
    num_epochs = 5

    # Number of sampls from parameter space
    num_samples = 10

    scheduler = ASHAScheduler(max_t=num_epochs, grace_period=1, reduction_factor=2)

    scaling_config = ScalingConfig(
        num_workers=3, use_gpu=False, resources_per_worker={"CPU": 1}
    )

    run_config = RunConfig(
        checkpoint_config=CheckpointConfig(
            num_to_keep=2,
            checkpoint_score_attribute="ptl/val_accuracy",
            checkpoint_score_order="max",
        ),
    )

    # Define a TorchTrainer without hyper-parameters for Tuner
    ray_trainer = TorchTrainer(
        train_func,
        scaling_config=scaling_config,
        run_config=run_config,
    )

    def tune_asha(num_samples=10):
        scheduler = ASHAScheduler(max_t=num_epochs, grace_period=1, reduction_factor=2)

        tuner = tune.Tuner(
            ray_trainer,
            param_space={"train_loop_config": search_space},
            tune_config=tune.TuneConfig(
                metric="ptl/val_accuracy",
                mode="max",
                num_samples=num_samples,
                scheduler=scheduler,
            ),
        )
        return tuner.fit()

    results = tune_asha(num_samples=num_samples)
