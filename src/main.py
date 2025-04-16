import hydra
from modules.manager import DataLoaderManager, TrainerManager

class MainApplication:
    def __init__(self):
        """Initialize configuration and data managers."""
        hydra.initialize(config_path='./config', version_base=None)
        self.config = hydra.compose(
            config_name="config", 
            overrides=[
                "data/preprocessed/loader=lidc_idri_preprocessed_data_loader_jn_demo",
                "metadata/preprocessed=lidc_idri_preprocessed_metadata_jn_demo"
            ]
        )
        self.dataloader_manager = DataLoaderManager(self.config)
        self.trainer_manager = TrainerManager(self.config, self.dataloader_manager)

    def run(self):
        """Run the full training pipeline."""
        print("\n--- Setting Up K-Fold Data Loader ---\n")
        self.dataloader_manager.setup_k_fold_dataloader()

        print("\n--- Training Model (Supervised) ---\n")
        self.trainer_manager.setup_model(mode="supervised")  # Later, we'll add "unsupervised"
        self.trainer_manager.train_model()

        print("\n--- Testing Model ---\n")
        self.trainer_manager.test_model()

if __name__ == "__main__":
    app = MainApplication()
    app.run()
