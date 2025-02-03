import sys

from src.logger import logging
from src.exception import CustomException

from src.components.data_ingestion import DataIngestion
from src.entity.config_entity import DataIngestionConfig
from src.entity.artifact_entity import DataIngestionArtifact


class TrainPipeline:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()


    def start_data_ingestion(self) -> DataIngestionArtifact:
        """Kickstart data ingestion component."""
        try:
            logging.info("Entered the start_data_ingestion method of TrainPipeline class")
            logging.info("Retrieving data from MongoDB")
            data_ingestion = DataIngestion(data_ingestion_config=self.data_ingestion_config)
            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
            logging.info("Retrieved train_set and test_set from MongoDB")
            logging.info("Exited the start_data_ingestion method of TrainPipeline class")
            return data_ingestion_artifact
        except Exception as e:
            raise CustomException(e, sys)


    def run_pipeline(self, ) -> None:
        """Run the complete train pipeline."""
        try:
            data_ingestion_artifact = self.start_data_ingestion()
        except Exception as e:
            raise CustomException(e, sys)
