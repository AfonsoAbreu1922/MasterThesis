from datetime import datetime, UTC
from itertools import product
from os.path import abspath, dirname, join
import hydra
import sys

sys.path.append(abspath(join(dirname('.'), "../../")))
from src.modules.experiment_execution import setup
setup.disable_warning_messages()
setup.set_precision(level="high")

from src.modules.data.dataloader.preprocessed_dataloader \
    import PreprocessedDataLoader
from src.modules.experiment_execution.datetimes \
    import ExperimentExecutionDatetimes
from src.modules.experiment_execution.config import experiment_execution_config
from src.modules.experiment_execution.info import ExperimentExecutionInfo
from src.modules.experiment_execution.prints import ExperimentExecutionPrints
from src.modules.models_and_frameworks.model_and_framework_pipeline \
    import ModelAndFrameworkPipeline
from src.modules.results_analysis.model_training_performance_metrics_figure \
    import ModelTrainingPerformanceMetricsFigure
from src.modules.results_analysis.model_test_performance_metrics_dataframe \
    import ModelTestPerformanceMetricsDataframe
experiment_execution_prints = ExperimentExecutionPrints()


@hydra.main(
    version_base=None,
    config_path="../../config_files",
    config_name="main"
)
def run_hyperparameter_grid_based_execution_pipeline(config):
    experiment_execution_config.set_experiment_id(config)

    if config.hyperparameter_grid_based_execution.apply:
        all_hyperparameter_combinations = list(product(
            *config.hyperparameter_grid_based_execution
                .hyperparameter_grid.values()
        ))
        for (
                hyperparameter_combination_index,
                current_hyperparameter_combination
        ) in enumerate(all_hyperparameter_combinations, 1):
            experiment_execution_config.set_used_hyperparameter_combination(
                config, current_hyperparameter_combination
            )
            experiment_execution_config.set_hyperparameter_combination_index(
                config,  hyperparameter_combination_index
            )
            experiment_execution_config.update_hyperparameter_combination(
                config
            )
            run_experiment_pipeline(config)
    else:
        experiment_execution_config.delete_key(
            config, key='hyperparameter_grid_based_execution'
        )
        run_experiment_pipeline(config)

    experiment_info = ExperimentExecutionInfo(
        config=config.experiment_execution.info,
        experiment_execution_ids=config.experiment_execution.ids,
        experiment_execution_paths=config.experiment_execution.paths
    )
    experiment_info.set_dataframe()
    experiment_info.save_as_md()
    experiment_info.save_dataframe_as_csv()

def run_experiment_pipeline(config):
    experiment_execution_config.set_experiment_version_id(config)
    experiment_execution_config.set_paths(config)
    setup.create_experiment_dir(
        dir_path=config.experiment_execution.paths.experiment_version_dir_path
    )

    experiment_execution_prints.experiment_version_start(
        experiment_id=config.experiment_execution.ids.experiment_id,
        experiment_version_id=
            config.experiment_execution.ids.experiment_version_id
    )

    experiment_execution_datetimes = ExperimentExecutionDatetimes(
        experiment_execution_paths=config.experiment_execution.paths
    )
    experiment_execution_datetimes.add_event(
        event_name="overall_execution",
        subevent_name="start",
        datetime=str(datetime.now(UTC).replace(microsecond=0))
    )

    dataloader = PreprocessedDataLoader(
        config=config.data,
        experiment_execution_paths=config.experiment_execution.paths
    )
    kfold_dataloaders = dataloader.get_dataloaders()
    kfold_data_names = dataloader.get_data_names()

    setup.set_seed(seed_value=config.seed_value)
    for datafold_id in range(1, len(kfold_dataloaders['training']) + 1):
        experiment_execution_prints.datafold_start(datafold_id)

        experiment_execution_datetimes.add_event(
            event_name=f"datafold_{datafold_id}_execution",
            subevent_name="start",
            datetime=str(datetime.now(UTC).replace(microsecond=0))
        )

        model_and_framework_pipeline = ModelAndFrameworkPipeline(
            config=config.models_and_frameworks.model_and_framework_pipeline,
            datafold_id=datafold_id,
            dataloaders=dict(
                training=kfold_dataloaders['training'][datafold_id - 1],
                validation=kfold_dataloaders['validation'][datafold_id - 1],
                test=kfold_dataloaders['test'][datafold_id - 1]
            ),
            data_file_names=dict(
                training=kfold_data_names['training'][datafold_id - 1],
                validation=kfold_data_names['validation'][datafold_id - 1],
                test=kfold_data_names['test'][datafold_id - 1]
            ),
            experiment_execution_ids=config.experiment_execution.ids,
            experiment_execution_paths=config.experiment_execution.paths
        )
        model_and_framework_pipeline.train_model()
        model_and_framework_pipeline.test_model()
        model_and_framework_pipeline.finalize()

        experiment_execution_datetimes.add_event(
            event_name=f"datafold_{datafold_id}_execution",
            subevent_name="end",
            datetime=str(datetime.now(UTC).replace(microsecond=0))
        )

        experiment_execution_prints.datafold_end(datafold_id)

    model_training_performance_metrics_figure = \
        ModelTrainingPerformanceMetricsFigure(
            config=config.results_analysis
                .model_training_performance_metrics_figure,
            experiment_execution_ids=config.experiment_execution.ids,
            experiment_execution_paths=config.experiment_execution.paths
        )
    model_training_performance_metrics_figure.set()
    model_training_performance_metrics_figure.save_image()

    model_test_performance_metrics_dataframe = (
        ModelTestPerformanceMetricsDataframe(
            config=config.results_analysis
                .model_test_performance_metrics_dataframe,
            experiment_execution_ids=config.experiment_execution.ids,
            experiment_execution_paths=config.experiment_execution.paths
        )
    )
    model_test_performance_metrics_dataframe.set_dataframe()
    model_test_performance_metrics_dataframe.export_as_html()

    experiment_execution_config.save(config)

    experiment_execution_datetimes.add_event(
        event_name=f"overall_execution",
        subevent_name="end",
        datetime=str(datetime.now(UTC).replace(microsecond=0))
    )
    experiment_execution_datetimes.save()

    experiment_execution_prints.experiment_version_end(
        experiment_id=config.experiment_execution.ids.experiment_id,
        experiment_version_id=
            config.experiment_execution.ids.experiment_version_id
    )

if __name__ == "__main__":
    run_hyperparameter_grid_based_execution_pipeline()

