import boto3
from src.configuration.aws_connection import S3Client
from io import StringIO
from typing import Union,List
import os,sys
from src.logger import logging
from mypy_boto3_s3.service_resource import Bucket
from src.exception import MyException
from pandas import DataFrame, read_csv
from botocore.exceptions import ClientError
import pickle

class SimpleStorageService:
    def __init__(self):
        s3_client= S3Client()
        self.s3_client = s3_client.s3_client
        self.s3_resource= s3_client.s3_resource

    def s3_key_path_available(self, bucket_name,s3_key)->bool:
        try:
            bucket=self.get_bucket(bucket_name)
            file_objects=[file_object for file_object in bucket.objects.filter(Prefix=s3_key)]
            return len(file_objects)> 0
        except Exception as e:
            raise MyException(e,sys)

    @staticmethod
    def read_object(object_name:str,decode:bool=True, make_readable= False) -> Union[StringIO,str]:
        try:
            func=(
                lambda: object_name.get()["Body"].read().decode()
                if decode else object_name.get()["Body"].read()
            )
            conv_func= lambda: StringIO(func()) if make_readable else func()
            logging.info("Exited the read_objec")
            return conv_func()
        except Exception as e:
            raise MyException(e,sys)
        
    def upload_file(self, from_filename:str,to_filename:str,bucket_name:str, remove:bool=True):
        logging.info("Entered the upload_file method of SimpleststorageService ")
        try:
            logging.info(f"Uploading {from_filename} to {to_filename} in {bucket_name}")
            self.s3_resource.meta.client.upload_file(from_filename, bucket_name, to_filename)
            logging.info(f"Uploaded {from_filename} to {to_filename} in {bucket_name}")
            if remove:
                os.remove(from_filename)
                logging.info(f"Removed {from_filename}")
            logging.info("Exited the upload_file method of SimpleStorageService")
        except Exception as e:
            raise MyException(e,sys)
    
    def load_model(self, model_name:str, bucket_name:str, model_dir:str=None) -> Union[List[object], object]:
        try:
            model_file= model_dir +"/" + model_name if model_dir else model_name
            file_object= self.get_file_object(model_file, bucket_name)
            model_obj= self.read_object(file_object, decode=False)
            model= pickle.loads(model_obj)
            logging.info("Production model loaded from s3 bucket")
            return model
        except Exception as e:
            raise MyException(e,sys)
        
    def get_file_object(self, filename:str, bucket_name:str)-> Union[List[object], object]:
        try:
            bucket=self.get_bucket(bucket_name)
            file_objects=[file_object for file_object in bucket.objects.filter(Prefix=filename)]
            func=lambda x:x[0] if len(x) ==1 else x
            file_objs=func(file_objects)
            logging.info("Exited the get_file_object method of SimpleStorageService class")
            return file_objs
        except Exception as e:
            raise MyException(e,sys)
    
    def get_bucket(self, bucket_name:str) -> Bucket:
        logging.info("Entered the get_bucket method of SimpleStorageService")
        try:
            bucket= self.s3_resource.Bucket(bucket_name)
            logging.info("Exited the get_bucket method of SimpleStorageService")
            return bucket
        except Exception as e:
            raise MyException(e,sys)
    
    


