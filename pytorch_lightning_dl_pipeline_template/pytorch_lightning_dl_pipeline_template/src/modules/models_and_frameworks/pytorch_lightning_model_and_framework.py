from src.modules.models_and_frameworks.efficient_net \
    .pytorch_lightning_efficient_net_model \
    import PyTorchLightningEfficientNetModel
from src.modules.models_and_frameworks.fixmatch \
    .pytorch_lightning_fixmatch_framework \
    import PyTorchLightningFixMatchFramework


class PyTorchLightningModelAndFramework:
    def __new__(cls, config, experiment_execution_paths):
        if config.model_or_framework_name == "EfficientNet":
            return PyTorchLightningEfficientNetModel(
                config=config.hyperparameters,
                experiment_execution_paths=experiment_execution_paths
            )
        elif config.model_or_framework_name == "FixMatch":
            return PyTorchLightningFixMatchFramework(
                config=config.hyperparameters,
                experiment_execution_paths=experiment_execution_paths
            )
        else:
            raise ValueError(
                f"Invalid model name: {config.model_or_framework_name}. "
                f"Supported datasets are 'EfficientNet'."
            )