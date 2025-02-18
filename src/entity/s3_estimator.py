import sys
from pandas import DataFrame

from src.cloud_storage.aws_storage import SimpleStorageService
from src.entity.estimator import VisaModel
from src.exception import CustomException


class VisaEstimator:
    """Represents a visa estimator.

    This class provides methods for checking, loading, saving, and retrieving visa model in s3 bucket
    (production) for prediction.
    """
    def __init__(self, bucket_name, model_path):
        self.bucket_name = bucket_name  # Name of your model bucket
        self.s3 = SimpleStorageService()
        self.model_path = model_path    # Location of the model in bucket
        self.loaded_model: VisaModel = None


    def is_model_present(self, model_path):
        try:
            return self.s3.s3_key_path_available(bucket_name=self.bucket_name, s3_key=model_path)
        except CustomException as e:
            print(e)
            return False


    def load_model(self) -> VisaModel:
        """Loads the model from the model_path."""
        return self.s3.load_model(self.model_path, bucket_name=self.bucket_name)


    def save_model(self, from_file, remove: bool = False) -> None:
        """Saves the model to the model_path.

        Args:
            from_file: The local system model path.
            remove: The flag to indicate whether to remove the model in local system folder.

        Returns:
            None.

        Raises:
            CustomException: If fail to save the model to the model_path.

        """
        try:
            self.s3.upload_file(from_file,
                                to_filename=self.model_path,
                                bucket_name=self.bucket_name,
                                remove=remove)
        except Exception as e:
            raise CustomException(e, sys)


    def predict(self, dataframe: DataFrame):
        try:
            if self.loaded_model is None:
                self.loaded_model = self.load_model()
            return self.loaded_model.predict(dataframe=dataframe)
        except Exception as e:
            raise CustomException(e, sys)
