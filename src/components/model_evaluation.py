import sys
import pandas as pd
from typing import Optional
from dataclasses import dataclass

from sklearn.metrics import f1_score

from src.entity.config_entity import ModelEvaluationConfig
from src.entity.artifact_entity import ModelTrainerArtifact, DataIngestionArtifact, ModelEvaluationArtifact
from src.entity.estimator import VisaModel
from src.entity.s3_estimator import VisaEstimator

from src.exception import CustomException
from src.logger import logging
from src.constants import TARGET_COLUMN, CURRENT_YEAR


@dataclass
class EvaluateModelResponse:
    trained_model_f1_score: float
    best_model_f1_score: float
    is_model_accepted: bool
    difference: float


class ModelEvaluation:
    def __init__(self,
                 model_eval_config: ModelEvaluationConfig,
                 data_ingestion_artifact: DataIngestionArtifact,
                 model_trainer_artifact: ModelTrainerArtifact):
        try:
            self.model_eval_config = model_eval_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.model_trainer_artifact = model_trainer_artifact
        except Exception as e:
            raise CustomException(e, sys) from e


    def get_best_model(self) -> Optional[VisaEstimator]:
        """Retrieves production model and returns the model object if available in s3 storage."""
        try:
            bucket_name = self.model_eval_config.bucket_name
            model_path = self.model_eval_config.s3_model_key_path
            visa_estimator = VisaEstimator(bucket_name=bucket_name,
                                           model_path=model_path)
            if visa_estimator.is_model_present(model_path=model_path):
                return visa_estimator
            return None
        except Exception as e:
            raise CustomException(e, sys) from e


    def evaluate_model(self) -> EvaluateModelResponse:
        """Evaluates trained model against production model and returns the evaluation result."""
        try:
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)
            test_df['company_age'] = CURRENT_YEAR - test_df['yr_of_estab']
            X_test, y_test = test_df.drop(TARGET_COLUMN, axis=1), test_df[TARGET_COLUMN]
            y_test = (y_test == 'Certified').astype(int)

            trained_model_f1_score = self.model_trainer_artifact.metric_artifact.f1_score
            best_model_f1_score = None
            best_model = self.get_best_model()
            if best_model is not None:
                y_pred_best_model = best_model.predict(X_test)
                best_model_f1_score = f1_score(y_test, y_pred_best_model)
            tmp_best_model_score = 0 if best_model_f1_score is None else best_model_f1_score
            result = EvaluateModelResponse(trained_model_f1_score=trained_model_f1_score,
                                           best_model_f1_score=best_model_f1_score,
                                           is_model_accepted=trained_model_f1_score > tmp_best_model_score,
                                           difference=trained_model_f1_score - tmp_best_model_score)
            logging.info(f"Result: {result}")
            return result
        except Exception as e:
            raise CustomException(e, sys) from e


    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        """Initiates the steps of the model evaluation component and returns a model evaluation artifact."""
        try:
            evaluate_model_response = self.evaluate_model()
            s3_model_path = self.model_eval_config.s3_model_key_path
            model_evaluation_artifact = ModelEvaluationArtifact(
                is_model_accepted=evaluate_model_response.is_model_accepted,
                s3_model_path=s3_model_path,
                trained_model_path=self.model_trainer_artifact.trained_model_file_path,
                accuracy_difference=evaluate_model_response.difference
            )
            logging.info(f"Model evaluation artifact: {model_evaluation_artifact}")
            return model_evaluation_artifact
        except Exception as e:
            raise CustomException(e, sys) from e
