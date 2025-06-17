import sys
import pandas as pd
import numpy as np
from typing import Optional
from src.logger import logging

from src.configuration.mongo_db_connection import MongoDBClient
from src.constants import DATABASE_NAME

from src.exception import MyException

class Proj1Data:
    def __init__(self) -> None:
        # print(DATABASE_NAME)
        try:
            self.mongo_client = MongoDBClient(database_name=DATABASE_NAME)
            # print(self.mongo_client.database_name)
        except Exception as e:
            raise MyException(e,sys)
        
    def export_collection_as_DataFrame(self, collection_name :str , database_name:Optional[str]= None) -> pd.DataFrame:

        try:
            if database_name is None:
                collection= self.mongo_client.database[collection_name]
            else:
                collection=self.mongo_client[database_name][collection_name]
            logging.info("Fetching Data from MongoDB")
            df=pd.DataFrame(list(collection.find()))
            logging.info(f" Data fetched with len : {len(df)}")
            if "id" in df.columns.to_list():
                df=df.drop(columns=["id"], axis=1)
            df.replace({"na" : np.nan}, inplace=True)
            return df
        except MyException as e:
            raise MyException(e,sys)
# s=Proj1Data()
#  print(os.getenv('COLLECTION_NAME') , os.getenv('DATABASE_NAME'))
# s.export_collection_as_DataFrame(COLLECTION_NAME , DATABASE_NAME)
        
        


