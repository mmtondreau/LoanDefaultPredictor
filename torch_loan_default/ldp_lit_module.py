import torch
from torchmetrics import Accuracy, MeanMetric
import lightning.pytorch as pl
from torch_loan_default.ldp_model import LDPModel


class LDPLitModule(pl.LightningModule):
    def __init__(
        self,
        config,
        pytorch_model=None,
        num_features=29,
        num_classes=1,
    ):
        super().__init__()
        self.hidden_units = config["hidden_units"]
        self.learning_rate = config["learning_rate"]
        self.example_input_array = torch.Tensor(32, 29)
        if pytorch_model is not None:
            self.model = pytorch_model
        else:
            self.model = LDPModel(
                num_features=num_features,
                num_classes=num_classes,
                hidden_units=self.hidden_units,
            )
        self.loss = torch.nn.BCELoss()

        self.accuracy = Accuracy("binary")
        self.val_accuracy = Accuracy("binary")
        self.val_loss = MeanMetric()

    def forward(self, x):
        x = self.model(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        predictions = self.model(x)
        loss = self.loss(predictions, y)
        self.log("ptl/train_loss", loss, prog_bar=True)
        self.log("ptl/train_accuracy", self.accuracy(predictions, y))
        return loss

    def on_validation_epoch_start(self):
        self.val_accuracy.reset()
        self.val_loss.reset()

    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        x, y = batch
        predictions = self.model(x)
        val_loss = self.loss(predictions, y)

        self.log("val_acc", self.accuracy(predictions, y))
        self.log("val_loss", val_loss, prog_bar=True)
        self.val_accuracy.update(predictions, y)
        self.val_loss.update(val_loss)

    def on_validation_epoch_end(self):
        self.log("ptl/val_loss", self.val_loss.compute(), sync_dist=True)
        self.log("ptl/val_accuracy", self.val_accuracy.compute(), sync_dist=True)

    def test_step(self, batch, batch_idx):
        # this is the test loop
        x, y = batch
        predictions = self.model(x)
        test_loss = self.loss(predictions, y)
        self.log("test_loss", test_loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
