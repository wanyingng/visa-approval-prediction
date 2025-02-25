import sys
from pandas import DataFrame
import os
from src.utils import load_object

from src.entity.config_entity import VisaPredictonConfig
from src.entity.s3_estimator import VisaEstimator

from src.exception import CustomException
from src.logger import logging
from src.constants import CURRENT_YEAR


class VisaData:
    def __init__(self, continent: str, employee_education: str, has_job_experience: str,
                 requires_job_training: str, no_of_employees: int, region_of_employment: str,
                 prevailing_wage: float, unit_of_wage: str, full_time_position: str, yr_of_estab: int):
        try:
            # Store the user inputs collected by the web app form
            self.continent = continent
            self.employee_education = employee_education
            self.has_job_experience = has_job_experience
            self.requires_job_training = requires_job_training
            self.no_of_employees = no_of_employees
            self.region_of_employment = region_of_employment
            self.prevailing_wage = prevailing_wage
            self.unit_of_wage = unit_of_wage
            self.full_time_position = full_time_position
            self.company_age = CURRENT_YEAR - yr_of_estab
        except Exception as e:
            raise CustomException(e, sys)


    def convert_to_dataframe(self) -> DataFrame:
        """Converts visa data to DataFrame and returns the DataFrame."""
        try:
            visa_dict = self.convert_to_dict()
            return DataFrame(visa_dict)
        except Exception as e:
            raise CustomException(e, sys)


    def convert_to_dict(self):
        """Converts visa data to dictionary and returns the dictionary."""
        logging.info("Entered convert_to_dict method of VisaData class")
        try:
            visa_dict = {
                "continent": [self.continent],
                "education_of_employee": [self.employee_education],
                "has_job_experience": [self.has_job_experience],
                "requires_job_training": [self.requires_job_training],
                "no_of_employees": [self.no_of_employees],
                "region_of_employment": [self.region_of_employment],
                "prevailing_wage": [self.prevailing_wage],
                "unit_of_wage": [self.unit_of_wage],
                "full_time_position": [self.full_time_position],
                "company_age": [self.company_age],
            }
            logging.info("Created visa dictionary")
            logging.info("Exited convert_to_dict method of VisaData class")
            return visa_dict
        except Exception as e:
            raise CustomException(e, sys)


class VisaClassifier:
    def __init__(self, prediction_pipeline_config: VisaPredictonConfig = VisaPredictonConfig()) -> None:
        try:
            self.prediction_pipeline_config = prediction_pipeline_config
        except Exception as e:
            raise CustomException(e, sys)


    def predict_s3(self, dataframe: DataFrame) -> str:
        """Returns the prediction result in string format for production use (AWS S3)."""
        try:
            logging.info("Entered predict method of VisaClassifier class")
            model = VisaEstimator(
                bucket_name=self.prediction_pipeline_config.model_bucket_name,
                model_path=self.prediction_pipeline_config.model_file_path
            )
            result = model.predict(dataframe)
            return result
        except Exception as e:
            raise CustomException(e, sys)


    def predict_local(self, dataframe: DataFrame) -> str:
        """Returns the prediction result in string format for local deployment."""
        try:
            logging.info("Entered predict method of VisaClassifier class")
            model_path = os.path.join("artifact/02_20_2025_13_04_04/model_trainer/trained_model", "model.pkl")
            print("Start loading")
            model = load_object(file_path=model_path)
            print("Completed loading")
            print(dataframe)
            result = model.predict(dataframe)
            print("Completed predicting: ", result)
            return result
        except Exception as e:
            raise CustomException(e, sys)
