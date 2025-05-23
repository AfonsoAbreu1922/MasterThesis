class ExperimentExecutionPrints:

    @staticmethod
    def datafold_start(datafold_id):
        print(f'\n    Running model pipeline for data fold {datafold_id}...')

    @staticmethod
    def datafold_end(datafold_id):
        print(f'    ...model pipeline for data fold {datafold_id} has run!')

    @staticmethod
    def experiment_version_start(experiment_id, experiment_version_id):
        print("\nStarting experiment {} version {} ".format(
            experiment_id,
            experiment_version_id
        ))

    @staticmethod
    def experiment_version_end(experiment_id, experiment_version_id):
        print("\n...experiment {} version {} has ended!".format(
            experiment_id,
            experiment_version_id
        ))