import torch
from torchmetrics import Accuracy, MeanMetric, AUROC
import pytorch_lightning as pl

import torch.nn.functional as F
from torch_loan_default.ldp_model import LDPModel


class LDPLitModule(pl.LightningModule):
    def __init__(
        self,
        config,
        num_features,
        pytorch_model=None,
        num_classes=1,
    ):
        super().__init__()
        self.example_input_array = torch.Tensor(32, num_features)
        self.hidden_units = config["hidden_units"]
        self.learning_rate = config["learning_rate"]
        if pytorch_model is not None:
            self.model = pytorch_model
        else:
            self.model = LDPModel(
                num_features=num_features,
                num_classes=num_classes,
                hidden_units=self.hidden_units,
            )

        self.auroc = AUROC(task="binary")

        self.val_loss = []
        self.val_auroc = []
        self.save_hyperparameters()

    def forward(self, x):
        x = self.model(x)
        return x

    def training_step(self, batch, batch_idx):
        loss, auroc = self._shared_eval(batch, batch_idx)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_auroc", auroc, prog_bar=True)
        return loss

    def on_validation_epoch_start(self):
        self.val_loss.clear()
        self.val_auroc.clear()

    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        loss, auroc = self._shared_eval(batch, batch_idx)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_auroc", auroc, prog_bar=True)
        self.val_auroc.append(auroc)
        self.val_loss.append(loss)

    def on_validation_epoch_end(self):
        avg_loss = torch.stack(self.val_loss).mean()
        avg_auroc = torch.stack(self.val_auroc).mean()
        self.log("ptl/val_loss", avg_loss, sync_dist=True)
        self.log("ptl/val_auroc", avg_auroc, sync_dist=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        # this is the test loop
        loss, auroc = self._shared_eval(batch, batch_idx)
        self.log("test_loss", loss)
        self.log("test_auroc", auroc)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.learning_rate, weight_decay=0.001
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer, patience=5, factor=0.1, mode="min"
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "train_loss",
            },
        }

    def _shared_eval(self, batch, batch_idx):
        x, y, _ = batch
        predictions = self(x)
        loss = F.binary_cross_entropy(predictions, y)
        auroc = self.auroc(predictions, y)
        return loss, auroc
