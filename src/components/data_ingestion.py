import os
import sys

from pandas import DataFrame
from sklearn.model_selection import train_test_split

from src.data_access.visa_data import VisaData
from src.entity.config_entity import DataIngestionConfig
from src.entity.artifact_entity import DataIngestionArtifact

from src.exception import CustomException
from src.logger import logging


class DataIngestion:
    def __init__(self, data_ingestion_config=DataIngestionConfig()):
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise CustomException(e, sys)


    def export_data_into_feature_store(self) -> DataFrame:
        """Export data from MongoDB to CSV file."""
        try:
            logging.info(f"Exporting data from MongoDB")
            visa_data = VisaData()
            dataframe = visa_data.export_collection_as_dataframe(collection_name=
                                                                 self.data_ingestion_config.collection_name)
            logging.info(f"Shape of dataframe: {dataframe.shape}")
            feature_store_file_path  = self.data_ingestion_config.feature_store_file_path
            dir_path = os.path.dirname(feature_store_file_path)
            os.makedirs(dir_path, exist_ok=True)
            logging.info(f"Saving exported data into feature store file path: {feature_store_file_path}")
            dataframe.to_csv(feature_store_file_path, index=False, header=True)
            return dataframe
        except Exception as e:
            raise CustomException(e, sys)


    def split_data_as_train_test(self, dataframe: DataFrame) -> None:
        """Split the dataframe into train set and test set based on the split ratio."""
        logging.info("Entered split_data_as_train_test method of DataIngestion class")
        try:
            train_set, test_set = train_test_split(dataframe, test_size=self.data_ingestion_config.train_test_split_ratio)
            logging.info("Performed train test split on the dataframe")

            dir_path = os.path.dirname(self.data_ingestion_config.training_file_path)
            os.makedirs(dir_path,exist_ok=True)

            logging.info(f"Exporting train and test file path.")
            train_set.to_csv(self.data_ingestion_config.training_file_path,index=False,header=True)
            test_set.to_csv(self.data_ingestion_config.testing_file_path,index=False,header=True)
            logging.info(f"Exported train and test file path.")
            logging.info("Exited split_data_as_train_test method of DataIngestion class")
        except Exception as e:
            raise CustomException(e, sys)


    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        """Initiate the data ingestion components of training pipeline."""
        logging.info("Entered initiate_data_ingestion method of DataIngestion class")
        try:
            dataframe = self.export_data_into_feature_store()
            logging.info("Retrieved data from MongoDB")

            self.split_data_as_train_test(dataframe)
            logging.info("Performed train test split on the dataset")

            data_ingestion_artifact = DataIngestionArtifact(train_file_path=self.data_ingestion_config.training_file_path,
                                                            test_file_path=self.data_ingestion_config.testing_file_path)
            logging.info(f"Data ingestion artifact: {data_ingestion_artifact}")
            logging.info("Exited initiate_data_ingestion method of DataIngestion class")
            return data_ingestion_artifact
        except Exception as e:
            raise CustomException(e, sys)
