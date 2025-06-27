import sys
import pandas as pd
from pandas import DataFrame
from sklearn.pipeline import Pipeline

from src.exception import MyException
from src.logger import logging

class TargetValueMapping:
    def __init__(self):
        self.yes:str=0
        self.no:str=1
    def _asdict(self):
        return self.__dict__
    def reverse_mapping(self):
        mapping_response =self._asdict
        return dict(zip(mapping_response.values(), mapping_response.keys()))
    
class MyModel:
    def __init__(self, preprocessing_object: object,trained_model_object:object):
        self.preprocessing_object=preprocessing_object
        self.trained_model_object=trained_model_object

    def predict(self, dataframe: pd.DataFrame):
        try:
            logging.info("Starting Prediction process.")
            logging.info(dataframe.columns)
            transformed_feature= self.preprocessing_object.transform(dataframe)
            logging.info("Using the trained model to get predictions")
            predictions=self.trained_model_object(transformed_feature)
            return predictions
        
        except MyException as e:
            raise MyException(e,sys) from e
    

    def __repr__(self):
        return f"{type(self.trained_model_object).__name__}()"
    
    def __str__(self):
        return f"{type(self.trained_model_object).__name__}()"
