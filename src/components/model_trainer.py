import sys

import numpy as np
from typing import Tuple

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from src.models.model_factory import ModelFactory

from src.entity.config_entity import ModelTrainerConfig
from src.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact, ClassificationMetricArtifact
from src.entity.estimator import VisaModel

from src.exception import CustomException
from src.logger import logging
from src.utils import load_numpy_array_data, load_object, save_object


class ModelTrainer:
    def __init__(self, data_transformation_artifact: DataTransformationArtifact,
                 model_trainer_config: ModelTrainerConfig):
        self.data_transformation_artifact = data_transformation_artifact
        self.model_trainer_config = model_trainer_config


    def get_model_report(self, train: np.array, test: np.array) -> Tuple[object, object]:
        """Retrieves the best model report."""
        try:
            model_factory = ModelFactory(model_config_path=self.model_trainer_config.model_config_file_path)
            logging.info("Retrieved best model object and its report")

            X_train, y_train, X_test, y_test = train[:, :-1], train[:, -1], test[:, :-1], test[:, -1]
            best_model_report = model_factory.get_best_model(
                X=X_train, y=y_train, base_accuracy=self.model_trainer_config.expected_accuracy
            )
            best_model = best_model_report.best_model
            y_pred = best_model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            metric_artifact = ClassificationMetricArtifact(accuracy=accuracy,
                                                           f1_score=f1,
                                                           precision=precision,
                                                           recall=recall)
            return best_model_report, metric_artifact
        except Exception as e:
            raise CustomException(e, sys) from e


    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        """Initiates the model trainer steps and returns model trainer artifact."""
        logging.info("Entered initiate_model_trainer method of ModelTrainer class")
        try:
            train_arr = load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_train_file_path)
            test_arr = load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_test_file_path)
            best_model_report, metric_artifact = self.get_model_report(train=train_arr, test=test_arr)
            preprocessor = load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)

            # Set an evaluation threshold that aligns with business goals
            if best_model_report.best_score < self.model_trainer_config.expected_accuracy:
                logging.info("No best model found")
                raise CustomException("No best model found")
            logging.info("Best model found")

            visa_model = VisaModel(preprocessor=preprocessor,
                                   trained_model=best_model_report.best_model)
            logging.info("Created VisaModel object with preprocessor and best model")
            save_object(self.model_trainer_config.trained_model_file_path, visa_model)
            logging.info("Saved the VisaModel object")

            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                metric_artifact=metric_artifact
            )
            logging.info(f"Model trainer artifact: {model_trainer_artifact}")
            return model_trainer_artifact
        except Exception as e:
            raise CustomException(e, sys) from e
