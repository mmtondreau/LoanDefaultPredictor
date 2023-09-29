from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.cli import LightningCLI
from watermark import watermark
from torch_loan_default.lit_ldp_module import LitLDPModule
from torch_loan_default.ldp_data_module import LDPDataModule
import os

if __name__ == "__main__":
    print(watermark(packages="torch,lightning"))
    cli = LightningCLI(
        model_class=LitLDPModule,
        datamodule_class=LDPDataModule,
        run=False,
        save_config_callback=False,
        seed_everything_default=123,
        trainer_defaults={
            "max_epochs": 1,
            "accelerator": "auto",
            "callbacks": [ModelCheckpoint(monitor="val_acc", mode="max")],
        },
    )


lightning_model = LitLDPModule(
    pytorch_model=None,
    learning_rate=cli.model.learning_rate,
    hidden_units=cli.model.hidden_units,
)

cli.trainer.fit(lightning_model, datamodule=cli.datamodule)
acc = cli.trainer.test(lightning_model, datamodule=cli.datamodule)

path = os.path.dirname(cli.trainer.checkpoint_callback.best_model_path)

with open(os.path.join(path, "test_acc.txt"), "w") as f:
    f.write(f"Test accuracy: {acc}")
