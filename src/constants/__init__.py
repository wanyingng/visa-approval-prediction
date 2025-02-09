import os


DATABASE_NAME: str = "US_VISA"
COLLECTION_NAME: str = "visa_data"
MONGODB_URL_KEY: str = "MONGODB_URL"

PIPELINE_NAME: str = "visa"
ARTIFACT_DIR: str = "artifact"

FILE_NAME: str = "data.csv"
TRAIN_FILE_NAME: str = "train.csv"
TEST_FILE_NAME: str = "test.csv"

MODEL_FILE_NAME: str = "model.pkl"
PREPROCESSOR_FILE_NAME = "preprocessor.pkl"

TARGET_COLUMN = "case_status"
DATASET_YEAR = 2016

SCHEMA_FILE_PATH = os.path.join("config", "schema.yaml")

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
