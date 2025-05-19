import ast
import numpy
import pandas
import statistics


class LIDCIDRIPreprocessedMetaDataFrame:
    def __init__(self, config, preprocessed_data_dir_path):
        self.config = config

        self.lung_nodule_metadataframe = pandas.read_csv(
            filepath_or_buffer=
                f"{preprocessed_data_dir_path}/metadata_csvs"
                "/lung_nodule_image_metadata.csv"
        )
        self._apply_lung_nodule_metadataframe_transformations()

    # def get_file_names(self):
    #     file_names = self.lung_nodule_metadataframe['file_name'].tolist()
    #     return file_names

    def get_lung_nodule_metadataframe(self):
        return self.lung_nodule_metadataframe

    # def get_visual_attribute_score_means_dataframe(self):
    #     visual_attribute_score_means_dataframe = self.lung_nodule_metadataframe.copy()
    #     filtered_df = self.lung_nodule_metadataframe.loc[:,
    #         'mean' in self.lung_nodule_metadataframe.columns
    #         & 'file_name' in self.lung_nodule_metadataframe.columns]
    #     return visual_attribute_score_means_dataframe

    def _apply_lung_nodule_metadataframe_transformations(self):

        # insert nodule file names
        self.lung_nodule_metadataframe.insert(
            loc=0,
            column='file_name',
            value=(
                self.lung_nodule_metadataframe['Patient ID']
                + "-N" + self.lung_nodule_metadataframe['Nodule ID']
                    .astype(str).str.zfill(2)
            )
        )

        # set up nodule malignancy columns
        # self.lung_nodule_metadataframe['Nodule Malignancy'] = \
        #     self.lung_nodule_metadataframe['Nodule Malignancy'] \
        #         .apply(ast.literal_eval)
        # self.lung_nodule_metadataframe.insert(
        #     loc=self.lung_nodule_metadataframe.columns
        #         .get_loc('Nodule Malignancy') + 1,
        #     column=f"Mean Nodule Malignancy",
        #     value=self.lung_nodule_metadataframe['Nodule Malignancy']
        #         .apply(numpy.mean)
        # )
        # self.lung_nodule_metadataframe.insert(
        #     loc=self.lung_nodule_metadataframe.columns
        #         .get_loc('Nodule Malignancy') + 2,
        #     column=f"Nodule Malignancy StD",
        #     value=self.lung_nodule_metadataframe['Nodule Malignancy']
        #         .apply(numpy.std)
        # )

        # set up nodule visual attribute columns
        # for semantic_characteristic_name \
        #         in self.config.semantic_characteristic.names:
        #     self.lung_nodule_metadataframe[
        #         f'Nodule {semantic_characteristic_name.replace("_", " ").title()}'
        #     ] = self.lung_nodule_metadataframe[
        #         f'Nodule {semantic_characteristic_name.replace("_", " ").title()}'
        #     ].apply(ast.literal_eval)
        #
        #     statistical_operation = numpy.mean
        #     if self.config.use_mode_for_internal_structure_and_calcification:
        #         if semantic_characteristic_name \
        #                 in ["internal_structure", "calcification"]:
        #             statistical_operation = statistics.mode
        #     self.lung_nodule_metadataframe.insert(
        #         loc=self.lung_nodule_metadataframe.columns.get_loc(
        #             f'Nodule {semantic_characteristic_name.replace("_", " ").title()}'
        #         ) + 1,
        #         column=f'Mean Nodule {semantic_characteristic_name.replace("_", " ").title()}',
        #         value=self.lung_nodule_metadataframe[
        #             f'Nodule {semantic_characteristic_name.replace("_", " ").title()}'
        #         ].apply(statistical_operation))

        # filter nodules that have been labeled by at least three radiologists
        self.lung_nodule_metadataframe = self.lung_nodule_metadataframe[
            self.lung_nodule_metadataframe['Nodule Malignancy'].apply(
                ast.literal_eval
            ).apply(
                lambda x:
                len(x) >= self.config.minimum_number_of_annotations_per_nodule
            )
        ]

        # filter nodules with mean nodule malignancy score != 3
        # self.lung_nodule_metadataframe = self.lung_nodule_metadataframe[
        #     self.lung_nodule_metadataframe[f"Mean Nodule Malignancy"] != 3]

        # reset index due to applied filters
        self.lung_nodule_metadataframe.reset_index(drop=True, inplace=True)

        # self.lung_nodule_metadataframe.insert(
        #     loc=len(self.lung_nodule_metadataframe.columns),
        #     column=f"label",
        #     value=(
        #         self.lung_nodule_metadataframe['Mean Nodule Malignancy'] > 3
        #     ).astype(int)
        # )
        label_columns = []
        for label in self.config.labels:
            if label.stratification_label:
                label_column_name = f'label_{label.name}'
            else:
                label_column_name = \
                    f'label_{label.name}_{label.statistical_operation}'
            mean_label_score_value = (
                1 + self.config.semantic_characteristic
                    .score_counts[label.name]
            ) / 2
            statistical_operation = getattr(numpy, label.statistical_operation)

            self.lung_nodule_metadataframe.insert(
                loc=len(self.lung_nodule_metadataframe.columns),
                column=label_column_name,
                value=self.lung_nodule_metadataframe[
                    f'Nodule {label.name.replace("_", " ").title()}'
                ].apply(ast.literal_eval).apply(statistical_operation)
            )

            if label.filter_out_average_score:
                self.lung_nodule_metadataframe = \
                    self.lung_nodule_metadataframe[
                        self.lung_nodule_metadataframe[label_column_name]
                        != mean_label_score_value
                    ]

            if label.mode != "float":
                if label.mode == "binary":
                    self.lung_nodule_metadataframe[label_column_name] = (
                        self.lung_nodule_metadataframe[label_column_name]
                        > mean_label_score_value
                    ).astype(int)
                elif label.mode == "round_and_convert_to_int":
                    self.lung_nodule_metadataframe[label_column_name] = \
                        self.lung_nodule_metadataframe[label_column_name] \
                            .apply(lambda x: int(x + 0.5))
                else:
                    raise ValueError(
                        f"Invalid label mode {label.mode}. "
                        "Supported label modes are"
                        " 'binary' and 'round_and_convert_to_int'"
                    )

            label_columns.append(label_column_name)

        self.lung_nodule_metadataframe = \
            self.lung_nodule_metadataframe[['file_name', *label_columns]]
