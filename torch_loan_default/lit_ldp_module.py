import torch
from torchmetrics import Accuracy
import lightning.pytorch as pl
from torch_loan_default.ldp_model import LDPModel


class LitLDPModule(pl.LightningModule):
    def __init__(
        self,
        pytorch_model,
        hidden_units=[32, 12],
        num_features=29,
        num_classes=1,
        learning_rate=0.001,
    ):
        super().__init__()
        self.hidden_units = hidden_units
        self.example_input_array = torch.Tensor(32, 29)
        if pytorch_model is not None:
            self.model = pytorch_model
        else:
            self.model = LDPModel(
                num_features=num_features,
                num_classes=num_classes,
                hidden_units=hidden_units,
            )
        self.loss = torch.nn.BCELoss()
        self.accuracy = Accuracy("binary")
        self.learning_rate = learning_rate

    def forward(self, x):
        x = self.model(x)
        return x

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x, y = batch
        # x = x.view(x.size(0), -1)
        self.accuracy.reset()
        predictions = self.model(x)
        loss = self.loss(predictions, y)
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss, prog_bar=True)
        self.accuracy.update(predictions, y)
        self.log("train_acc", self.accuracy.compute(), prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        x, y = batch
        self.accuracy.reset()
        predictions = self.model(x)
        val_loss = self.loss(predictions, y)

        self.accuracy.update(predictions, y)
        self.log("val_acc", self.accuracy.compute(), prog_bar=True)
        self.log("val_loss", val_loss, prog_bar=True)

    def test_step(self, batch, batch_idx):
        # this is the test loop
        x, y = batch
        predictions = self.model(x)
        test_loss = self.loss(predictions, y)
        self.log("test_loss", test_loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
