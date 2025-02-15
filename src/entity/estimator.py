import sys

from pandas import DataFrame
from sklearn.pipeline import Pipeline

from src.exception import CustomException
from src.logger import logging


class VisaModel:
    def __init__(self, preprocessor: Pipeline, trained_model: object):
        self.preprocessor = preprocessor
        self.trained_model = trained_model


    def __repr__(self):
        return (f"{self.__class__.__name__}()")


    def predict(self, dataframe: DataFrame) -> DataFrame:
        """Preprocess raw input and predict using the transformed features."""
        logging.info("Entered predict method of VisaModel class")
        try:
            logging.info("Using the trained model to get predictions")
            transformed_features = self.preprocessor.transform(dataframe)
            logging.info("Used the trained model to get predictions")
            return self.trained_model.predict(transformed_features)
        except Exception as e:
            raise CustomException(e, sys)
