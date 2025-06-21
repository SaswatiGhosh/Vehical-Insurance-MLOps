import sys
from typing import Tuple
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score,f1_score,recall_score

from src.exception import MyException
from src.utils.main_utils import load_numpy_array_data, save_object,load_object
from src.entity.config_entity import ModelTrainerConfig
from src.entity.artifact_entity  import DataTransformationArtifact, ClassificationMetricArtifact, ModelTrainerArtifact
from src.entity.estimator import MyModel
from src.logger import logging
from sklearn.pipeline import Pipeline

class ModelTrainer:
    def __init__(self, data_transformation_artifact: DataTransformationArtifact, model_trainer_config: ModelTrainerConfig):
        
        self.data_transformation_artifact = data_transformation_artifact
        self.model_trainer_config = model_trainer_config
    
    def get_model_object_report(self, train:np.array, test:np.array):
        try:
            logging.info("Training RandomClassifier with specified paramters")
            x_train,y_train,x_test,y_test= train[: , :-1], train[:,-1] , test[:,:-1], test[:,-1]
            logging.info("Train &Test split done")
            model=RandomForestClassifier(
                n_estimators=self.model_trainer_config._n_estimators,
                random_state=self.model_trainer_config._random_state,
                max_depth=self.model_trainer_config._max_depth,
                min_samples_split=self.model_trainer_config._min_samples_split,
                min_samples_leaf=self.model_trainer_config._min_samples_leaf,
                criterion=self.model_trainer_config._criterion,

            )
            logging.info("Model Training going on..")
            model.fit(x_train,y_train)
            logging.info("Model Training done")

            y_pred=model.predict(x_test)
            accuracy=accuracy_score(y_test,y_pred=y_pred)
            precision=precision_score(y_test,y_pred=y_pred)
            f1=f1_score(y_test,y_pred=y_pred)
            recall=recall_score(y_test,y_pred=y_pred)

            metric_artifact= ClassificationMetricArtifact(f1_score=f1, precision_score=precision, recall_score=recall)
            return model , metric_artifact
        except Exception as e:
            raise MyException(e,sys) from e
        
    def initiate_model_trainer(self):
        try:
            print(".........................................................................")
            print("Model Trainer initiated")
            train_arr= load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_train_file_path)
            test_arr = load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_test_file_path)
            logging.info("test-train data loaded")

            trained_model, model_artifact = self.get_model_object_report(train=train_arr, test= test_arr)
            logging.info("Model object and artifact loaded")

            preprocessing_obj:object= load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)
            logging.info("Preprocessing object loaded")

            if accuracy_score(train_arr[:,-1],trained_model.predict(train_arr[:, :-1])) < self.model_trainer_config.expected_accuracy: #tb checked
                logging.info("Model is not good enough, retraining the model")
                raise Exception("No model found with score above base score.")
            
            logging.info("Saving new model as performance is better than previous one.")
            my_model=MyModel(preprocessing_object=preprocessing_obj, trained_model_object= trained_model)
            save_object(self. model_trainer_config.trained_model_file_path, my_model)
            logging.info("Saved final model object that includes preprocessing and trained model")

            model_trainer_artifact=ModelTrainerArtifact(trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                                                        metric_artifact=model_artifact)
            logging.info(f"Model Trainer Artifact created{model_trainer_artifact}")
            return model_trainer_artifact


        except Exception as e:
            raise MyException(e,sys) from e
