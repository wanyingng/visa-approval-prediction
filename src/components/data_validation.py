import sys

import pandas as pd
from pandas import DataFrame
import json

from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

from src.exception import CustomException
from src.logger import logging
from src.utils import read_yaml_file, write_yaml_file
from src.constants import SCHEMA_FILE_PATH

from src.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from src.entity.config_entity import DataValidationConfig


class DataValidation:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact, data_validation_config: DataValidationConfig):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            self._schema_config = read_yaml_file(file_path=SCHEMA_FILE_PATH)
        except Exception as e:
            raise CustomException(e, sys)


    def validate_number_of_columns(self, dataframe: DataFrame) -> bool:
        """Validates the number of columns and returns a bool value based on the validation outcome."""
        try:
            status = len(dataframe.columns) == len(self._schema_config["columns"])
            logging.info(f"All required columns are present: [{status}]")
            return status
        except Exception as e:
            raise CustomException(e, sys)


    def is_column_exist(self, df: DataFrame) -> bool:
        """Validates the existence of both numerical and categorical columns."""
        try:
            dataframe_columns = df.columns
            missing_numerical_columns = []
            missing_categorical_columns = []

            for column in self._schema_config["numerical_columns"]:
                if column not in dataframe_columns:
                    missing_numerical_columns.append(column)
            if len(missing_numerical_columns) > 0:
                logging.info(f"Missing numerical column: {missing_numerical_columns}")

            for column in self._schema_config["categorical_columns"]:
                if column not in dataframe_columns:
                    missing_categorical_columns.append(column)
            if len(missing_categorical_columns) > 0:
                logging.info(f"Missing categorical column: {missing_categorical_columns}")

            return False if len(missing_categorical_columns) > 0 or len(missing_numerical_columns) > 0 else True
        except Exception as e:
            raise CustomException(e, sys)


    @staticmethod
    def read_data(file_path) -> DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise CustomException(e, sys)


    def detect_dataset_drift(self, reference_df: DataFrame, current_df: DataFrame) -> bool:
        """Validates for data drift and returns a bool value based on the validation outcome."""
        try:
            # Create a report
            report = Report(metrics=[DataDriftPreset()])

            # Calculate the report
            report.run(current_data=current_df, reference_data=reference_df)
            json_report = json.loads(report.json())
            write_yaml_file(file_path=self.data_validation_config.drift_report_file_path, content=json_report)
            n_columns = json_report["metrics"][0]["result"]["number_of_columns"]
            n_drifted_columns = json_report["metrics"][0]["result"]["number_of_drifted_columns"]
            logging.info(f"{n_drifted_columns}/{n_columns} drift detected.")
            drift_status = json_report["metrics"][0]["result"]["dataset_drift"]
            return drift_status
        except Exception as e:
            raise CustomException(e, sys)


    def initiate_data_validation(self) -> DataValidationArtifact:
        """Initiates the data validation component of training pipeline."""
        try:
            validation_error_msg = ""
            logging.info("Starting data validation")
            train_df, test_df = (DataValidation.read_data(file_path=self.data_ingestion_artifact.train_file_path),
                                 DataValidation.read_data(file_path=self.data_ingestion_artifact.test_file_path))
            status = self.validate_number_of_columns(dataframe=train_df)
            logging.info(f"All required columns are present in train dataframe: {status}")
            if not status:
                validation_error_msg += f"Columns are missing in train dataframe."
            status = self.validate_number_of_columns(dataframe=test_df)
            logging.info(f"All required columns are present in test dataframe: {status}")
            if not status:
                validation_error_msg += f"Columns are missing in test dataframe."

            status = self.is_column_exist(df=train_df)
            if not status:
                validation_error_msg += f"Columns are missing in train dataframe."
            status = self.is_column_exist(df=test_df)
            if not status:
                validation_error_msg += f"Columns are missing in test dataframe."

            is_validated = len(validation_error_msg) == 0
            if is_validated:
                drift_exist = self.detect_dataset_drift(train_df, test_df)
                if drift_exist:
                    logging.info(f"Drift detected")
                    validation_error_msg = "Drift detected."
                else:
                    validation_error_msg = "Drift not detected."
            else:
                logging.info(f"Validation_error: {validation_error_msg}")

            data_validation_artifact = DataValidationArtifact(
                is_validated=is_validated,
                message=validation_error_msg,
                drift_report_file_path=self.data_validation_config.drift_report_file_path
            )
            logging.info(f"Data validation artifact: {data_validation_artifact}")
            return data_validation_artifact
        except Exception as e:
            raise CustomException(e, sys)
