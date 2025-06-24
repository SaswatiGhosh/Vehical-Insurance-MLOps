from src.entity.config_entity import ModelEvaluationConfig
from src.entity.artifact_entity import ModelTrainerArtifact, DataIngestionArtifact, ModelEvaluationArtifact
from sklearn.metrics import f1_score
from src.constants import TARGET_COLUMN
from src.exception import MyException
from src.logger import logging
from src.utils.main_utils import load_object
import sys
import pandas as pd
from typing import  Optional
from src.entity.s3_estimator import Proj1Estimator
from dataclasses import dataclass

@dataclass
class ModelEvaluationResponse:
    trained_model_f1_score:float
    best_model_f1_score:float
    is_model_accepted:bool
    difference:float

class ModelEvaluation:
    def __init__(self, model_evaluation_config: ModelEvaluationConfig ,data_ingestion_artifact :DataIngestionArtifact,model_trainer_artifact: ModelTrainerArtifact):
        try:
            self.model_evaluation_config = model_evaluation_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.model_trainer_artifact = model_trainer_artifact
        except Exception as e:
            raise MyException(e,sys) from e
    
    def get_best_model(self)-> Optional[Proj1Estimator]:
        try:
            bucket_name=self.model_evaluation_config.bucket_name
            model_path=self.model_evaluation_config.s3_model_key_path
            proj1_estimator= Proj1Estimator(bucket_name=bucket_name, model_path=model_path)
            if proj1_estimator.is_model_present(model_path=model_path):
                return proj1_estimator
            return None
        except Exception as e:
            raise MyException(e,sys) from e
        
    def _map_gender_column(self,df):
        logging.info("Mapping 'Gender' column to binary values")
        df['Gender'] = df['Gender'].map({'Male': 1, 'Female':0}).astype('int')
        return df
    
    def _create_dummy_columns(self,df):
        logging.info("Creating dummy columns for categorical features")
        df=pd.get_dummies(df,drop_first=True)
        return df
    
    def rename_columns(self,df):
        logging.info("Renaming columns to match the target column name")
        df=df.rename(columns={
            "Vehicle_Age_< 1 Year" : "Vehicle_Age_lt_1_Year",
            "Vehicle_Age_> 2 Years" : "Vehicle_Age_gt_2_Years",
        })
        for col in ["Vehicle_Age_lt_1_Year", "Vehicle_Age_gt_2_Years" , "Vehicle_Damage_Yes"]:
            df[col] = df[col].astype(int)
            return df
        
    def _drop_id_column(self,df):
        logging.info("Dropping 'id' column")
        if "_id" in df.columns:
            df.drop(columns=["_id"], inplace=True)
        return df
    
    def evaluate_model(self):
        try:
            test_df=pd.read_csv(self.data_ingestion_artifact.test_file_path)
            x,y=test_df.drop(TARGET_COLUMN,axis=1),test_df[TARGET_COLUMN]

            logging.info("Test data loaded and now transforming it for prediction...")
            x=self._map_gender_column(x)
            x=self._drop_id_column(x)
            x=self._create_dummy_columns(x)
            x=self.rename_columns(x)

            trained_model= load_object(file_path=self.model_trainer_artifact.trained_model_file_path)
            logging.info("Trained model loaded/exist")
            trained_model_f1_score=self.model_trainer_artifact.metric_artifact.f1_score
            logging.info(f"f1_score for this model: {trained_model_f1_score}")
            best_model_f1_score=None
            best_model=self.get_best_model()
            if best_model is not None:
                logging.info("Best model loaded/exist")
                y_hat_best_model= best_model.predict(x)
                best_model_f1_score=f1_score(y,y_hat_best_model)
                logging.info(f"f1_score for best model: {best_model_f1_score} , f1_score-New trained Model: {trained_model_f1_score}")
            temp_best_model= 0 if best_model_f1_score is None else best_model_f1_score
            result=ModelEvaluationResponse(trained_model_f1_score=trained_model_f1_score, best_model_f1_score=best_model_f1_score, is_model_accepted=trained_model_f1_score>temp_best_model,
                                           difference=trained_model_f1_score- temp_best_model)
            logging.info(f"Result:{result}")
            return result
        except Exception as e:
            raise MyException(e,sys)
    

    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        try:
            print("-----------------------------------------------")
            logging.info("Model evaluation initiated")
            evaluate_model_response= self.evaluate_model()
            
            s3_model_path= self.model_evaluation_config.s3_model_key_path
            model_evaluation_artifact=ModelEvaluationArtifact(is_model_accepted=evaluate_model_response.is_model_accepted, 
                                                      s3_model_path=s3_model_path,
                                                      trained_model_path=self.model_trainer_artifact.trained_model_file_path,
                                                      changed_accuracy=evaluate_model_response.difference)
            logging.info(f"Model evaluation completed : {model_evaluation_artifact}")
            return model_evaluation_artifact
        except Exception as e:
            raise MyException(e,sys)

        
        




