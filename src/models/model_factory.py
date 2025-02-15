import sys
import yaml
import importlib

from collections import namedtuple
from typing import List

from src.exception import CustomException
from src.logger import logging
from src.constants import (GRID_SEARCH_KEY, MODULE_KEY, CLASS_KEY, PARAM_KEY,
                           MODEL_SELECTION_KEY, SEARCH_PARAM_GRID_KEY)


InitializedModelDetail = namedtuple("InitializedModelDetail",
                                    ["model_serial_number", "model", "param_grid_search", "model_name"])

GridSearchedBestModel = namedtuple("GridSearchedBestModel", ["model_serial_number",
                                                             "model",
                                                             "best_model",
                                                             "best_parameters",
                                                             "best_score"])

BestModel = namedtuple("BestModel", ["model_serial_number",
                                     "model",
                                     "best_model",
                                     "best_parameters",
                                     "best_score"])


class ModelFactory:
    def __init__(self, model_config_path: str = None):
        try:
            self.config: dict = ModelFactory.read_params(model_config_path)
            self.grid_search_cv_module: str = self.config[GRID_SEARCH_KEY][MODULE_KEY]
            self.grid_search_class_name: str = self.config[GRID_SEARCH_KEY][CLASS_KEY]
            self.grid_search_property_data: dict = dict(self.config[GRID_SEARCH_KEY][PARAM_KEY])
            self.models_initialization_config: dict = dict(self.config[MODEL_SELECTION_KEY])
            self.initialized_model_list = None
            self.grid_searched_best_model_list = None
        except Exception as e:
            raise CustomException(e, sys)


    @staticmethod
    def update_property_of_class(instance_ref, property_data: dict):
        try:
            if not isinstance(property_data, dict):
                raise CustomException("property_data parameter needs to be a dictionary")
            logging.info(property_data)
            for key, value in property_data.items():
                setattr(instance_ref, key, value)
            return instance_ref
        except Exception as e:
            raise CustomException(e, sys)


    @staticmethod
    def read_params(config_path: str) -> dict:
        try:
            with open(config_path) as yaml_file:
                config = yaml.safe_load(yaml_file)
            return config
        except Exception as e:
            raise CustomException(e, sys)


    @staticmethod
    def class_for_name(module_name, class_name):
        """
        Import the specified class of a module (equivalent to from module_name import class_name).

        Args:
            module_name: The name of the module to be imported.
            class_name: The name of the class of the imported module.

        Returns:
            The class reference of the imported module.

        Raises:
            ImportError: If fail to import the module.
            AttributeError: If class cannot be found.

        """
        try:
            # Load the module, will raise ImportError if module cannot be loaded
            module = importlib.import_module(module_name)
            # Get the class, will raise AttributeError if class cannot be found
            class_ref = getattr(module, class_name)
            return class_ref
        except Exception as e:
            raise CustomException(e, sys)


    def execute_grid_search_operation(self, initialized_model: InitializedModelDetail, input_feature,
                                      output_feature) -> GridSearchedBestModel:
        """
        Perform parameter search operation and return the best model with its set of best parameters.

        Args:
            initialized_model: The InitializedModelDetail object.
            input_feature: The features used for training the model.
            output_feature: The target/dependent feature in the prediction.

        Returns:
            The GridSearchedBestModel object.

        Raises:
            CustomException: If GridSearchCV operation failed.

        """
        try:
            # Instantiate the GridSearchCV class
            message = "*" * 50, f"training {type(initialized_model.model).__name__}", "*" * 50
            logging.info(message)
            grid_search_cv_ref = ModelFactory.class_for_name(module_name=self.grid_search_cv_module,
                                                             class_name=self.grid_search_class_name)

            grid_search_cv = grid_search_cv_ref(estimator=initialized_model.model,
                                                param_grid=initialized_model.param_grid_search)
            grid_search_cv = ModelFactory.update_property_of_class(grid_search_cv,
                                                                   self.grid_search_property_data)
            grid_search_cv.fit(input_feature, output_feature)
            grid_searched_best_model = GridSearchedBestModel(model_serial_number=initialized_model.model_serial_number,
                                                             model=initialized_model.model,
                                                             best_model=grid_search_cv.best_estimator_,
                                                             best_parameters=grid_search_cv.best_params_,
                                                             best_score=grid_search_cv.best_score_)
            return grid_searched_best_model
        except Exception as e:
            raise CustomException(e, sys)


    def get_initialized_model_list(self) -> List[InitializedModelDetail]:
        """
        Retrieve a list of model details.

        Returns:
            The list of model details.

        Raises:
            CustomException: If fail to retrieve the list of model details.

        """
        try:
            initialized_model_list = []
            for model_serial_number in self.models_initialization_config.keys():
                model_initialization_config = self.models_initialization_config[model_serial_number]
                model_obj_ref = ModelFactory.class_for_name(module_name=model_initialization_config[MODULE_KEY],
                                                            class_name=model_initialization_config[CLASS_KEY])
                model = model_obj_ref()

                if PARAM_KEY in model_initialization_config:
                    model_obj_property_data = dict(model_initialization_config[PARAM_KEY])
                    model = ModelFactory.update_property_of_class(instance_ref=model,
                                                                  property_data=model_obj_property_data)

                param_grid_search = model_initialization_config[SEARCH_PARAM_GRID_KEY]
                model_name = f"{model_initialization_config[MODULE_KEY]}.{model_initialization_config[CLASS_KEY]}"
                model_initialization_config = InitializedModelDetail(model_serial_number=model_serial_number,
                                                                     model=model,
                                                                     param_grid_search=param_grid_search,
                                                                     model_name=model_name)
                initialized_model_list.append(model_initialization_config)

            self.initialized_model_list = initialized_model_list
            return self.initialized_model_list
        except Exception as e:
            raise CustomException(e, sys)


    def initiate_best_parameter_search_for_initialized_model(self,
                                                             initialized_model: InitializedModelDetail,
                                                             input_feature,
                                                             output_feature) -> GridSearchedBestModel:
        """
        Initiate parameter search operation to return the best model with its set of best parameters.

        Args:
            initialized_model: The InitializedModelDetail object.
            input_feature: The features used for training the model.
            output_feature: The target/dependent feature in the prediction.

        Returns:
            The GridSearchedBestModel object.

        Raises:
            CustomException: If GridSearchCV operation failed.

        """
        try:
            return self.execute_grid_search_operation(initialized_model=initialized_model,
                                                      input_feature=input_feature,
                                                      output_feature=output_feature)
        except Exception as e:
            raise CustomException(e, sys)


    def initiate_best_parameter_search_for_initialized_models(self,
                                                              initialized_model_list: List[InitializedModelDetail],
                                                              input_feature,
                                                              output_feature) -> List[GridSearchedBestModel]:
        try:
            self.grid_searched_best_model_list = []
            for initialized_model_list in initialized_model_list:
                grid_searched_best_model = self.initiate_best_parameter_search_for_initialized_model(
                    initialized_model=initialized_model_list,
                    input_feature=input_feature,
                    output_feature=output_feature)
                self.grid_searched_best_model_list.append(grid_searched_best_model)
            return self.grid_searched_best_model_list
        except Exception as e:
            raise CustomException(e, sys)


    @staticmethod
    def get_model_detail(model_details: List[InitializedModelDetail],
                         model_serial_number: str) -> InitializedModelDetail:
        try:
            for model_data in model_details:
                if model_data.model_serial_number == model_serial_number:
                    return model_data
        except Exception as e:
            raise CustomException(e, sys)


    @staticmethod
    def get_best_model_from_grid_searched_best_model_list(grid_searched_best_model_list: List[GridSearchedBestModel],
                                                          base_accuracy: float = 0.6) -> BestModel:
        try:
            best_model = None
            for grid_searched_best_model in grid_searched_best_model_list:
                if base_accuracy < grid_searched_best_model.best_score:
                    logging.info(f"Acceptable model found: {grid_searched_best_model}")
                    base_accuracy = grid_searched_best_model.best_score
                    best_model = grid_searched_best_model
            if not best_model:
                raise CustomException(f"None of the model has base accuracy: {base_accuracy}")
            logging.info(f"Best model: {best_model}")
            return best_model
        except Exception as e:
            raise CustomException(e, sys)


    def get_best_model(self, X, y, base_accuracy: float = 0.6) -> BestModel:
        """
        Retrieve the best model.

        Args:
            X: The input features.
            y: The target feature.
            base_accuracy: The expected baseline accuracy.

        Returns:
            The BestModel object.

        Raises:
            CustomException: If none of the models met the expected baseline accuracy.

        """
        try:
            logging.info("Initializing model from config file")
            initialized_model_list = self.get_initialized_model_list()
            logging.info(f"Initialized model: {initialized_model_list}")
            grid_searched_best_model_list = self.initiate_best_parameter_search_for_initialized_models(
                initialized_model_list=initialized_model_list,
                input_feature=X,
                output_feature=y)
            return ModelFactory.get_best_model_from_grid_searched_best_model_list(grid_searched_best_model_list,
                                                                                  base_accuracy=base_accuracy)
        except Exception as e:
            raise CustomException(e, sys)
