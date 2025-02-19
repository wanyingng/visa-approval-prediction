from dataclasses import dataclass


@dataclass
class DataIngestionArtifact:
    train_file_path: str
    test_file_path: str


@dataclass
class DataValidationArtifact:
    is_validated: bool
    message: str
    drift_report_file_path: str


@dataclass
class DataTransformationArtifact:
    transformed_object_file_path: str
    transformed_train_file_path: str
    transformed_test_file_path: str


@dataclass
class ClassificationMetricArtifact:
    accuracy: float
    f1_score: float
    precision: float
    recall: float


@dataclass
class ModelTrainerArtifact:
    trained_model_file_path: str
    metric_artifact: ClassificationMetricArtifact


@dataclass
class ModelEvaluationArtifact:
    is_model_accepted: bool # Whether current model is better than production model
    accuracy_difference: float  # The difference in accuracy between current model and production model
    s3_model_path: str  # The model.pkl location in s3 bucket
    trained_model_path: str # The model.pkl location in artifact folder


@dataclass
class ModelPusherArtifact:
    bucket_name: str
    s3_model_path: str
