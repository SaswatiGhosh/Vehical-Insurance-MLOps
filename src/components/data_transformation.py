import sys
import numpy as np
import pandas as pd
from imblearn.combine import SMOTEENN
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer


from src.constants import TARGET_COLUMN,SCHEMA_FILE_PATH,CURRENT_YEAR
from src.entity.config_entity import DataTransformationConfig
from src.entity.artifact_entity import DataTransformationArtifact, DataIngestionArtifact,DataValidationArtifact
from src.exception import MyException
from src.logger import logging
from src.utils.main_utils import save_object,save_numpy_array_data,read_yaml_file

class DataTransformation:
# This method is automatically called when you create an object of the class.
# Parameters it accepts:
#     data_ingestion_atrifact: This object contains info about where your raw training and test data is stored.
#     data_transformation_config: This holds file paths or settings for saving transformed data and the preprocessor.
#     data_validation_artifact: This tells whether data validation passed or not, and has metadata about the data.
#     👉 The names like DataIngestionArtifact, DataTransformationConfig, etc., are custom classes used for organizing your project’s data flow.
    
    def __init__(self, data_ingestion_atrifact: DataIngestionArtifact, data_transformation_config: DataTransformationConfig, 
                 data_validation_artifact: DataValidationArtifact):
        
        try:
            self.data_ingestion_artifact= data_ingestion_atrifact # You're storing the incoming data_ingestion_atrifact into the object as self.data_ingestion_artifact, so you can use it later in other methods like initiate_data_transformation().🧠 self. means: "This belongs to the object being created."
            self.data_tranformation_config=data_transformation_config
            self.data_validation_artifact=data_validation_artifact
            self._schema_config= read_yaml_file(file_path=SCHEMA_FILE_PATH)

        except Exception as e:
            raise MyException(e, sys) from e

    @staticmethod
    def read_data(file_path:str):
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise MyException(e, sys) from e
        

    def get_data_transformer_object(self) -> Pipeline:
        logging.info("Entered get_data_transformed_object method of DataTransformation class")

        try:
            numeric_transformer= StandardScaler()
            min_max_scaler= MinMaxScaler()

            logging.info("Transformers initialized: Standard Scaler- MinMaxScaler")

            num_features= list(self._schema_config['num_features']) #to be understood
            mm_columns=list(self._schema_config['mm_columns']) #to be understood

            logging.info("Cols loaded from schema")
            preprocessor= ColumnTransformer(transformers=[("Standard Scaler", numeric_transformer, num_features),
                                                          ("MinMaxScaler", min_max_scaler, mm_columns)],
                                                          remainder='passthrough') # tb understood
            
            final_pipeline=Pipeline(steps=[("Preprocessor", preprocessor)])
            logging.info("Final Pipeline ready")
            logging.info("Exited get_data_transformer_object method of Data Tranformation class")
            return final_pipeline
        except Exception as e:
            raise MyException(e,sys) from e
        
    def _map_gender_column(self,df):
        logging.info("Mapping 'Gender' column to binary values")
        df['Gender']= df['Gender'].map({'Female': 0 , 'Male' : 1}).astype(int)
        return df
    
    def _create_dummies(self,df):
        logging.info("Creating dummies for categorical columns")
        df=pd.get_dummies(df,drop_first=True) #to be understood
        return df
    
    def _rename_columns(self,df):
        logging.info("Renaming columns as per schema and casting to int")
        df=df.rename(columns={
            "Vehical_Age_< 1 Year" :"Vehicle_Age_lt_1_Years",
            "Vehicle_Age_> 2_Years":"Vehicle_Age_gt_2_Years",
        })
        for col in["Vehicle_Age_lt_1_Years", "Vehicle_Age_gt_2_Years", "Vehicle_Damage_Yes"]:
            if col in df.columns:
                df[col]=df[col].astype('int')
            return df
        
    def _drop_id_columns(self,df):
        logging.info("Dropping 'id' columns")
        drop_col= self._schema_config['drop_columns']
        if drop_col in df.columns:
            df=df.drop(drop_col, axis=1)
        return df
    
    def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:
            logging.info("Data Transformation started !!")
            if not self.data_validation_artifact.validation_status:
                raise Exception(self.data_validation_artifact.message)
            train_df = self.read_data(file_path=self.data_ingestion_artifact.trained_file_path)
            test_df = self.read_data(file_path=self.data_ingestion_artifact.test_file_path)
            logging.info("Train and test data loaded")

            input_feature_train_df= train_df.drop(columns=[TARGET_COLUMN],axis=1) # to be understood
            target_feature_train_df= train_df[TARGET_COLUMN]

            input_feature_test_df= test_df.drop(columns=[TARGET_COLUMN],axis=1)
            target_feature_test_df= test_df[TARGET_COLUMN]

            input_feature_train_df= self._map_gender_column(input_feature_train_df)
            input_feature_train_df=self._drop_id_columns(input_feature_train_df)
            input_feature_train_df=self._create_dummies(input_feature_train_df)
            input_feature_train_df=self._rename_columns(input_feature_train_df)

            input_feature_test_df=self._map_gender_column(input_feature_test_df)
            input_feature_test_df=self._drop_id_columns(input_feature_test_df)
            input_feature_test_df=self._create_dummies(input_feature_test_df)
            input_feature_test_df=self._rename_columns(input_feature_test_df)

            logging.info("Custom transformation applied to test and train data")
            logging.info("Starting data Transfromation!!")
            preprocessor= self.get_data_transformer_object()
            logging.info("Got the preprocessor object")

            logging.info("Initializing transformation fro training data")
            input_feature_train_arr= preprocessor.fit_transform(input_feature_train_df)
            logging.info("Initiazliing transformation for testing data ")
            input_feature_test_arr =preprocessor.transform(input_feature_test_df)
            logging.info("Transformation completed!!")

            logging.info("Applying SMOTTEENN for unhanding unbalanced dataset")
            smt=SMOTEENN(sampling_strategy='minority')
            input_feature_train_final, target_feature_train_final= smt.fit_resample(input_feature_train_arr, target_feature_train_df)
            input_feature_test_final, target_feature_test_final= smt.fit_resample(input_feature_test_arr, target_feature_test_df)
            logging.info("SMOTEENN applie to train-test df.")

            train_arr=np.c_[input_feature_train_final, np.array(target_feature_train_final)]
            test_arr=np.c_[input_feature_test_final, np.array(target_feature_test_final)]
            logging.info("feature-target concatenation done for test-train df")

            save_object(self.data_tranformation_config.transformed_object_file_path, preprocessor)
            save_numpy_array_data(self.data_tranformation_config.transformed_train_file_path, array=train_arr)
            save_numpy_array_data(self.data_tranformation_config.transformed_test_file_path, array=test_arr)
            logging.info("Saving transformed objet and transformed files")
            logging.info("Data Transformation completed!!")
            return DataTransformationArtifact(transformed_object_file_path=self.data_tranformation_config.transformed_object_file_path,
                                              transformed_train_file_path=self.data_tranformation_config.transformed_train_file_path,
                                              transformed_test_file_path=self.data_tranformation_config.transformed_test_file_path)
        

        except Exception as e:
            raise MyException(e,sys) from e




            


            

            


