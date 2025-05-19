from src.modules.data.dataloader.preprocessed_dataloaders \
    .combined_preprocessed_dataloader \
    import CombinedPreprocessedKFoldDataLoader
from src.modules.data.dataloader.preprocessed_dataloaders \
    .lidc_idri_preprocessed_dataloader \
    import LIDCIDRIPreprocessedKFoldDataLoader
from src.modules.data.dataloader.preprocessed_dataloaders \
    .luna25_preprocessed_dataloader \
    import LUNA25PreprocessedKFoldDataLoader
from src.modules.data.metadataframe.preprocessed_metadataframe \
    import PreprocessedMetadataFrame


class PreprocessedDataLoader:
    def __new__(cls, config, experiment_execution_paths):
        if config.combined_dataloader.apply:
            lung_nodule_metadataframes = {}
            for dataset in config.dataset:
                metadataframe = PreprocessedMetadataFrame(
                    config=config.metadataframe[dataset.name],
                    preprocessed_data_dir_path=
                        experiment_execution_paths
                            .preprocessed_data_dir_path[dataset.name]
                )
                lung_nodule_metadataframes[dataset.name] = \
                    metadataframe.get_lung_nodule_metadataframe()
            return CombinedPreprocessedKFoldDataLoader(
                config=config.dataloader.combined,
                lung_nodule_metadataframes=lung_nodule_metadataframes
            )
        else:
            metadataframe = PreprocessedMetadataFrame(
                config=config.data.metadataframe,
                preprocessed_data_dir_path=
                    experiment_execution_paths.preprocessed_data_dir_path
            )
            if config.dataset_name == "LIDC-IDRI":
                return LIDCIDRIPreprocessedKFoldDataLoader(
                    config=config,
                    lung_nodule_metadataframe=
                        metadataframe.get_lung_nodule_metadataframe()
                )
            elif config.dataset_name == "LUNA25":
                return LUNA25PreprocessedKFoldDataLoader(
                    config=config,
                    lung_nodule_metadataframe=
                        metadataframe.get_lung_nodule_metadataframe()
                )
            else:
                raise ValueError(
                    f"Invalid dataset name: {config.dataset_name}. "
                    f"Supported dataset names are 'LIDC-IDRI' or 'LUNA25'."
                )
