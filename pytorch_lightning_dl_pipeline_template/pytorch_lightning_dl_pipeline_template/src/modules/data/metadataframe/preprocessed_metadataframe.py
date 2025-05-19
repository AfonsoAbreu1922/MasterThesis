from src.modules.data.metadataframe.preprocessed_metadataframes \
    .lidc_idri_metadataframe import LIDCIDRIPreprocessedMetaDataFrame
from src.modules.data.metadataframe.preprocessed_metadataframes \
    .luna25_metadataframe import LUNA25PreprocessedMetaDataFrame

class PreprocessedMetadataFrame:
    def __new__(cls, config, preprocessed_data_dir_path):
        if config.dataset.name == "lidc_idri":
            return LIDCIDRIPreprocessedMetaDataFrame(
                config,
                preprocessed_data_dir_path
            )
        elif config.dataset.name == "luna25":
            return LUNA25PreprocessedMetaDataFrame(
                config,
                preprocessed_data_dir_path
            )
        else:
            raise ValueError(
                f"Invalid dataset name: {config.dataset.name}. "
                f"Supported datasets are 'lidc_idri' and 'luna25'."
            )