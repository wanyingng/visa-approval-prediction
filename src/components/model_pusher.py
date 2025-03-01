import sys

from src.entity.artifact_entity import ModelPusherArtifact, ModelEvaluationArtifact
from src.entity.config_entity import ModelPusherConfig
from src.entity.s3_estimator import VisaEstimator
from src.cloud_storage.aws_storage import SimpleStorageService

from src.exception import CustomException
from src.logger import logging


class ModelPusher:
    def __init__(self,
                 model_evaluation_artifact: ModelEvaluationArtifact,
                 model_pusher_config: ModelPusherConfig):
        self.s3 = SimpleStorageService()
        self.model_evaluation_artifact = model_evaluation_artifact
        self.model_pusher_config = model_pusher_config
        self.visa_estimator = VisaEstimator(bucket_name=model_pusher_config.bucket_name,
                                            model_path=model_pusher_config.s3_model_key_path)


    def initiate_model_pusher(self) -> ModelPusherArtifact:
        """Initiates all steps of the model pusher component and returns the model pusher artifact."""
        logging.info("Entered initiate_model_pusher method of ModelPusher class")
        try:
            logging.info("Uploading trained model in artifact folder to s3 bucket")
            self.visa_estimator.save_model(from_file=self.model_evaluation_artifact.trained_model_path)
            model_pusher_artifact = ModelPusherArtifact(bucket_name=self.model_pusher_config.bucket_name,
                                                        s3_model_path=self.model_pusher_config.s3_model_key_path)
            logging.info("Uploaded trained model to s3 bucket")
            logging.info(f"Model pusher artifact: [{model_pusher_artifact}]")
            logging.info("Exited initiate_model_pusher method of ModelPusher class")
            return model_pusher_artifact
        except Exception as e:
            raise CustomException(e, sys) from e
