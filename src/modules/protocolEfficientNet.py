import lightning as pl
import torch
from torch import optim, nn
from efficientnet_pytorch import EfficientNet

class ProtocolEfficientNet(pl.LightningModule):
    """
    EfficientNet-based classifier for lung nodule classification.

    Supports both scratch training and fine-tuning with a pretrained encoder.
    """

    def __init__(self, num_classes=1, num_channels=3, model_name="efficientnet-b0", pretrained=True):
        """
        Args:
            num_classes (int): Number of output classes (1 for binary classification).
            num_channels (int): Number of input image channels (1 for grayscale, 3 for RGB).
            model_name (str): Name of EfficientNet model variant.
            pretrained (bool): Whether to use ImageNet pretrained weights.
        """
        super().__init__()
        self.save_hyperparameters()

        # Load EfficientNet with or without pretrained weights
        self.model = EfficientNet.from_pretrained(model_name) if pretrained else EfficientNet.from_name(model_name)

        # Modify the first convolutional layer to match input channels
        self.model._conv_stem = nn.Conv2d(num_channels, self.model._conv_stem.out_channels, 
                                          kernel_size=3, stride=self.model._conv_stem.stride, 
                                          padding=1, bias=False)

        # Modify the final fully connected layer for binary classification
        self.model._fc = nn.Linear(self.model._fc.in_features, num_classes)

        # Define loss function
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, x):
        """Forward pass."""
        return self.model(x)
    
    def step(self, batch):
        """Compute loss and accuracy for a batch."""
        images, labels = batch
        images = images['input_image']
        labels = labels['lnm']['mean']

        logits = self(images)
        loss = self.loss_fn(logits, labels)

        preds = torch.sigmoid(logits).round()  # Convert logits to binary predictions
        acc = (preds == labels).float().mean()  # Compute accuracy

        return loss, acc

    def training_step(self, batch, batch_idx):
        """Training step."""
        loss, acc = self.step(batch)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step."""
        loss, acc = self.step(batch)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        """Test step."""
        loss, acc = self.step(batch)
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        """Configure the optimizer."""
        return optim.Adam(self.parameters(), lr=1e-3)
