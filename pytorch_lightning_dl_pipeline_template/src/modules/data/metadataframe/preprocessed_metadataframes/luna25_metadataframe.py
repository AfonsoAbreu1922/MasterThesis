import pandas


class LUNA25PreprocessedMetaDataFrame:
    def __init__(self, config, preprocessed_data_dir_path):
        self.config = config

        self.lung_nodule_metadataframe = pandas.read_csv(
            filepath_or_buffer=
                f"{preprocessed_data_dir_path}/metadata_csvs"
                f"/lung_nodule_metadata.csv"
        )
        self._apply_lung_nodule_metadataframe_transformations()

    def get_lung_nodule_metadataframe(self):
        return self.lung_nodule_metadataframe

    def _apply_lung_nodule_metadataframe_transformations(self):
        self.lung_nodule_metadataframe['file_name'] = \
            self.lung_nodule_metadataframe.apply(
            lambda row:
                f"P{row['patient_id']}-"
                f"S{row['series_instance_uid'][-5:]}-"
                f"N{str(row['nodule_id']).zfill(2)}",
            axis=1
        )

        label_columns = []
        for label in self.config.labels:
            label_column = f"label_{label.name}"
            if label.name == "malignancy":
                self.lung_nodule_metadataframe.insert(
                    loc=len(self.lung_nodule_metadataframe.columns),
                    column=label_column,
                    value=self.lung_nodule_metadataframe['label'].astype(int)
                )

            label_columns.append(label_column)

        self.lung_nodule_metadataframe = \
            self.lung_nodule_metadataframe[['file_name', *label_columns]]
