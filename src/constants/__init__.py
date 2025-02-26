import os
from datetime import date


DATABASE_NAME: str = "US_VISA"
COLLECTION_NAME: str = "visa_data"
MONGODB_URL_KEY: str = "MONGODB_URL"

PIPELINE_NAME: str = "visa"
ARTIFACT_DIR: str = "artifact"

SEED: int = 42

FILE_NAME: str = "data.csv"
TRAIN_FILE_NAME: str = "train.csv"
TEST_FILE_NAME: str = "test.csv"

MODEL_FILE_NAME: str = "model.pkl"
PREPROCESSOR_FILE_NAME = "preprocessor.pkl"

TARGET_COLUMN = "case_status"
DATASET_YEAR = 2016
CURRENT_YEAR = date.today().year

SCHEMA_FILE_PATH = os.path.join("config", "schema.yaml")

GRID_SEARCH_KEY = "grid_search"
MODULE_KEY = "module"
CLASS_KEY = "class"
PARAM_KEY = "params"
MODEL_SELECTION_KEY = "model_selection"
SEARCH_PARAM_GRID_KEY = "search_param_grid"
MODEL_CONFIG_FILE_NAME = "model.yaml"

AWS_ACCESS_KEY_ID_ENV_KEY = "AWS_ACCESS_KEY_ID"
AWS_SECRET_ACCESS_KEY_ENV_KEY = "AWS_SECRET_ACCESS_KEY"
REGION_NAME = "ap-southeast-1"

# Constants for Data Ingestion
DATA_INGESTION_COLLECTION_NAME: str = "visa_data"
DATA_INGESTION_DIR_NAME: str = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR: str = "feature_store"
DATA_INGESTION_INGESTED_DIR: str = "ingested"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO: float = 0.2

# Constants for Data Validation
DATA_VALIDATION_DIR_NAME: str = "data_validation"
DATA_VALIDATION_DRIFT_REPORT_DIR: str = "drift_report"
DATA_VALIDATION_DRIFT_REPORT_FILE_NAME: str = "report.yaml"

# Constants for Data Transformation
DATA_TRANSFORMATION_DIR_NAME: str = "data_transformation"
DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR: str = "transformed"
DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR: str = "transformed_object"

# Constants for Model Trainer
MODEL_TRAINER_DIR_NAME: str = "model_trainer"
MODEL_TRAINER_TRAINED_MODEL_DIR: str = "trained_model"
MODEL_TRAINER_TRAINED_MODEL_NAME: str = "model.pkl"
MODEL_TRAINER_EXPECTED_SCORE: float = 0.7
MODEL_TRAINER_MODEL_CONFIG_FILE_PATH: str = os.path.join("config", MODEL_CONFIG_FILE_NAME)

# Constants for Model Evaluation
MODEL_EVALUATION_CHANGED_THRESHOLD_SCORE: float = 0.02
MODEL_BUCKET_NAME = "visa-model2025"
MODEL_PUSHER_S3_KEY = "model-registry"

# Constants for FastAPI
APP_HOST = "0.0.0.0"
APP_PORT = 9696
