from os.path import abspath, dirname, join
import sys
from lightning import Trainer

# Import data loading utilities (modify based on your project structure)
sys.path.append(abspath(join(dirname(__file__), "../data_loading/")))
from data_loading.src.modules.data.dataloader.preprocessed_data_loader_LIDCIDRI import LIDCIDRIPreprocessedKFoldDataLoader
from data_loading.src.modules.data.metadata_LIDCIDRI import LIDCIDRIPreprocessedMetaData
from data_loading.src.modules.utils.paths import PYTHON_PROJECT_DIR_PATH

from data_loading.src.modules.data.dataloader.preprocessed_data_loader_luna25 import LUNA25PreprocessedDataLoader
from data_loading.src.modules.data.metadata_LIDCIDRI import LUNA25PreprocessedMetaData

from pytorch_lightning.utilities.combined_loader import CombinedLoader
from data_loading.src.modules.data.dataloader.preprocessed_data_loader_luna25 import LUNA25PreprocessedDataLoader
import pandas as pd
import numpy as np
import torch

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
        # === LIDC DataLoader ===
        self.config.data.preprocessed.loader.number_of_k_folds = 5
        self.config.data.preprocessed.loader.test_fraction_of_entire_dataset = None

        lidc_metadata_df = self.metadata.get_lung_nodule_image_metadataframe()
        lidc_loader = LIDCIDRIPreprocessedKFoldDataLoader(
            config=self.config.data.preprocessed.loader,
            lung_nodule_image_metadataframe=lidc_metadata_df
        )

        self.lidc_loader = lidc_loader

        # === LUNA25 DataLoader ===
        luna_df = pd.read_csv(self.config.luna25.metadata_path)  # Adjust path source
        luna_train_files = luna_df["file_name"].tolist()

        luna_dataset = LUNA25PreprocessedDataLoader(
            config=self.config.luna25,
            file_names=luna_train_files,
            image_numpy_arrays_dir_path=self.config.luna25.image_numpy_arrays_dir_path,
            label_dataframe=luna_df,
            subset_type="train"
        )

        luna_loader = torch.utils.data.DataLoader(
            dataset=luna_dataset,
            batch_size=self.config.luna25.batch_size,
            shuffle=True,
            num_workers=4
        )

        # === Combine them ===
        combined = {
            "lidc": lidc_loader.get_data_loaders_by_subset()["train"][0],
            "luna": luna_loader
        }
        self.combined_loader = CombinedLoader(combined, mode="min_size")  # or "max_size_cycle"


    def get_data_loaders_by_subset(self):
        """Return train/validation/test data loaders."""
        return self.dataloader.get_data_loaders_by_subset()
    
    def get_combined_train_loader(self):
        return self.combined_loader


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
        self.trainer.fit(model=self.model, train_dataloaders=self.dataloader_manager.get_combined_train_loader())

    def test_model(self):
        """Evaluate the model on test data."""
        self.trainer.test(model=self.model, dataloaders=self.dataloader_manager.get_data_loaders_by_subset()["test"][0])
