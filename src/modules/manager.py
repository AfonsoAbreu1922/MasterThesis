from os.path import abspath, dirname, join
import sys
from lightning import Trainer

# Import data loading utilities (modify based on your project structure)
sys.path.append(abspath(join(dirname(__file__), "../data_loading/")))
from data_loading.src.modules.data.dataloader.preprocessed_data_loader import LIDCIDRIPreprocessedKFoldDataLoader
from data_loading.src.modules.data.metadata import LIDCIDRIPreprocessedMetaData
from data_loading.src.modules.utils.paths import PYTHON_PROJECT_DIR_PATH

from pytorch_lightning.utilities.combined_loader import CombinedLoader
from data_loading.src.modules.data.dataloader.preprocessed_data_loader_luna25 import LUNA25PreprocessedDataLoader
import pandas as pd
import numpy as np

# Import model protocols
from modules.protocolEfficientNet import ProtocolEfficientNet
# from modules.ProtocolUnsupervised import ProtocolUnsupervised  # For later

class DataLoaderManager:
    """Manages dataset loading and k-fold cross-validation setup."""

    def __init__(self, config):
        self.config = config
        self.metadata = LIDCIDRIPreprocessedMetaData(config=config.metadata.preprocessed)
        self.dataloader = None

    def setup_k_fold_dataloader(self):
        """Set up k-fold cross-validation data loader."""
        self.config.data.preprocessed.loader.number_of_k_folds = 5
        self.config.data.preprocessed.loader.test_fraction_of_entire_dataset = None
        self.dataloader = LIDCIDRIPreprocessedKFoldDataLoader(
            config=self.config.data.preprocessed.loader, 
            lung_nodule_image_metadataframe=self.metadata.get_lung_nodule_image_metadataframe()
        )
        #adicionar o outro self.dataloader
        #combinar os dois dataloaders usando CombinedLoader da biblioteca pytorch_lightning.utilities

    def get_data_loaders_by_subset(self):
        """Return train/validation/test data loaders."""
        return self.dataloader.get_data_loaders_by_subset()

class TrainerManager:
    """Handles model training and testing."""

    def __init__(self, config, dataloader_manager):
        self.config = config
        self.dataloader_manager = dataloader_manager
        self.trainer = Trainer(limit_train_batches=100, max_epochs=1)
        self.model = None

    def setup_model(self, mode="supervised"):
        """
        Initialize the model. 
        Supports different training modes.
        
        Args:
            mode (str): "supervised" or "unsupervised".
        """
        if mode == "supervised":
            self.model = ProtocolEfficientNet(num_classes=1, num_channels=1)
        elif mode == "unsupervised":
            print("Unsupervised model setup is not yet implemented.")
            # self.model = ProtocolUnsupervised(...)  # Later integration




    def train_model(self):
        """Train the model."""
        self.trainer.fit(model=self.model, train_dataloaders=self.dataloader_manager.get_data_loaders_by_subset()["train"][0])

    def test_model(self):
        """Evaluate the model on test data."""
        self.trainer.test(model=self.model, dataloaders=self.dataloader_manager.get_data_loaders_by_subset()["test"][0])
