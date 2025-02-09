import sys

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, PowerTransformer
from sklearn.compose import ColumnTransformer

from src.constants import TARGET_COLUMN, SCHEMA_FILE_PATH, DATASET_YEAR
from src.entity.config_entity import DataTransformationConfig
from src.entity.artifact_entity import DataTransformationArtifact, DataIngestionArtifact, DataValidationArtifact
from src.entity.estimator import TargetValueMapping

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, save_numpy_array_data, read_yaml_file, drop_columns


class DataTransformation:
    def __init__(self,
                 data_ingestion_artifact: DataIngestionArtifact,
                 data_transformation_config: DataTransformationConfig,
                 data_validation_artifact: DataValidationArtifact):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_transformation_config = data_transformation_config
            self.data_validation_artifact = data_validation_artifact
            self._schema_config = read_yaml_file(file_path=SCHEMA_FILE_PATH)
        except Exception as e:
            raise CustomException(e, sys)


    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise CustomException(e, sys)


    def get_data_transformer_object(self) -> Pipeline:
        """Create and return a data transformer object."""
        logging.info("Entered get_data_transformer_object method of DataTransformation class")
        try:
            numeric_transformer = StandardScaler()
            oh_transformer = OneHotEncoder()
            ordinal_encoder = OrdinalEncoder()
            logging.info("Initialized StandardScaler, OneHotEncoder, OrdinalEncoder")

            oh_columns = self._schema_config['oh_columns']
            or_columns = self._schema_config['or_columns']
            transform_columns = self._schema_config['transform_columns']
            num_features = self._schema_config['num_features']
            logging.info("Retrieved numerical cols from schema config")

            transform_pipeline = Pipeline(steps=[
                ('transformer', PowerTransformer(method='yeo-johnson'))
            ])
            logging.info("Initialized PowerTransformer")

            preprocessor = ColumnTransformer(
                [
                    ("OneHotEncoder", oh_transformer, oh_columns),
                    ("Ordinal_Encoder", ordinal_encoder, or_columns),
                    ("Transformer", transform_pipeline, transform_columns),
                    ("StandardScaler", numeric_transformer, num_features)
                ]
            )
            logging.info("Created preprocessor object from ColumnTransformer")
            logging.info("Exited get_data_transformer_object method of DataTransformation class")
            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)


    def initiate_data_transformation(self) -> DataTransformationArtifact:
        """Initiate the data transformation component of training pipeline."""
        try:
            if self.data_validation_artifact.is_validated:
                logging.info("Starting data transformation")
                preprocessor = self.get_data_transformer_object()
                logging.info("Retrieved the preprocessor object")

                train_df = DataTransformation.read_data(file_path=self.data_ingestion_artifact.train_file_path)
                test_df = DataTransformation.read_data(file_path=self.data_ingestion_artifact.test_file_path)

                input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN], axis=1)
                target_feature_train_df = train_df[TARGET_COLUMN]
                logging.info("Retrieved both input and target features of Train dataset")
                input_feature_train_df['company_age'] = DATASET_YEAR - input_feature_train_df['yr_of_estab']
                logging.info("Added company_age column to the Train dataset")
                drop_cols = self._schema_config['drop_columns']
                input_feature_train_df = drop_columns(df=input_feature_train_df, cols=drop_cols)
                target_feature_train_df = target_feature_train_df.replace(TargetValueMapping()._asdict())
                logging.info("Dropped the columns in drop_cols of Train dataset")

                input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN], axis=1)
                target_feature_test_df = test_df[TARGET_COLUMN]
                logging.info("Retrieved both input and target features of Test dataset")
                input_feature_test_df['company_age'] = DATASET_YEAR - input_feature_test_df['yr_of_estab']
                logging.info("Added company_age column to the Test dataset")
                input_feature_test_df = drop_columns(df=input_feature_test_df, cols=drop_cols)
                target_feature_test_df = target_feature_test_df.replace(TargetValueMapping()._asdict())
                logging.info("Dropped the columns in drop_cols of Test dataset")

                logging.info("Applying preprocessor on both train and test dataframe")
                input_feature_train_arr = preprocessor.fit_transform(input_feature_train_df)
                logging.info("Applied fit transform to the train features")
                input_feature_test_arr = preprocessor.transform(input_feature_test_df)
                logging.info("Applied transform to the test features")

                logging.info("Creating train array and test array")
                train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
                test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
                save_object(
                    file_path=self.data_transformation_config.transformed_object_file_path,
                    obj=preprocessor
                )
                save_numpy_array_data(
                    file_path=self.data_transformation_config.transformed_train_file_path,
                    array=train_arr
                )
                save_numpy_array_data(
                    file_path=self.data_transformation_config.transformed_test_file_path,
                    array=test_arr
                )
                logging.info("Saved the preprocessor, train array, and test array")

                data_transformation_artifact = DataTransformationArtifact(
                    transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                    transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                    transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
                )
                logging.info("Exited initiate_data_transformation method of DataTransformation class")
                return data_transformation_artifact
            else:
                raise Exception(self.data_validation_artifact.message)
        except Exception as e:
            raise CustomException(e, sys)
