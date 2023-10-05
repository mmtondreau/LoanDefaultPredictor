from torch_loan_default.ldp_data_module import LDPDataModule
from torch_loan_default.ldp_lit_module import LDPLitModule
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from torch_loan_default.simple_data_module import EvenOddDataModule
import torch
import pandas as pd
import os

if __name__ == "__main__":
    config = {
        "hidden_units": [48, 24, 12],
        "learning_rate": 0.001,
        "batch_size": 1280,
    }

    dm = LDPDataModule(batch_size=config["batch_size"])
    dm.prepare_data()
    dm.setup(stage="fit")

    x, y, z = next(iter(dm.train_dataloader()))
    print(x[0])
    print(y[0])

    df = dm.df

    print(df[df["ID"] == z[0].item()].to_string())

    model = LDPLitModule(config, num_features=dm.width)
    trainer = pl.Trainer(
        devices="auto",
        accelerator="auto",
        max_epochs=100,
        callbacks=[
            EarlyStopping(
                monitor="ptl/val_loss", mode="min", patience=5, min_delta=0.0001
            ),
            ModelCheckpoint(
                monitor="ptl/val_auroc", mode="max", filename="{epoch}-{val_auroc:.2f}"
            ),
        ],
    )
    trainer.fit(model, datamodule=dm)

    trainer.test(model, datamodule=dm)

    model.eval()

    x, y, _ = next(iter(dm.test_dataloader()))
    y_hat = model(x)
    print(x)
    print(torch.flatten(y))
    print(torch.flatten(y_hat))
