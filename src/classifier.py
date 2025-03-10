import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
import timm

class EfficientNetLungClassifier(pl.LightningModule):
    def __init__(self, model_name='efficientnet_b0', num_classes=2, lr=1e-3):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
        self.loss_fn = nn.CrossEntropyLoss()
        self.lr = lr

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        loss = self.loss_fn(preds, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        loss = self.loss_fn(preds, y)
        acc = (preds.argmax(dim=1) == y).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)
