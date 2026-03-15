import sys  #used for system-level error handling in custom exceptions
from dataclasses import dataclass  #simplifies creation of configuration classes.

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.exceptions import CustomException
from src.logger import logging
import os  #used for file path handling.

from src.utils import save_object  #custom utility function that saves Python objects (like pipelines) as .pkl files.

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"proprocessor.pkl")  # Path created: artifacts/preprocessor.pkl
    #This file stores the trained preprocessing pipeline. Later, during prediction or deployment, the same transformations must be applied.
class DataTransformation: #handles all preprocessing operations
    '''Creates an instance of the configuration class so the 
       pipeline knows where to save the preprocessor object.'''
    def __init__(self):  
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_object(self):  #builds the preprocessing pipeline
        '''
        This function si responsible for data transformation
        
        '''
        try:
            # first we define Column Types
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]

            # This pipeline performs two steps on numeric columns
            num_pipeline= Pipeline(
                steps=[
                ("imputer",SimpleImputer(strategy="median")), # handles missing values
                ("scaler",StandardScaler())                   # Standardization

                ]
            )

            #This pipeline processes categorical variable
            cat_pipeline=Pipeline(

                steps=[
                ("imputer",SimpleImputer(strategy="most_frequent")), # Handle Missing Categories
                ("one_hot_encoder",OneHotEncoder()),                 # Encoding into model understandable values
                ("scaler",StandardScaler(with_mean=False))           # Standardization
                ]

            )

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            preprocessor=ColumnTransformer(
                [
                ("num_pipeline",num_pipeline,numerical_columns),
                ("cat_pipelines",cat_pipeline,categorical_columns)

                ]


            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path): #This method applies the preprocessing pipeline to the training and test data.

        try:
            # Load data
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")

            preprocessing_obj=self.get_data_transformer_object() #Creates the pipeline defined earlier.

            # defining Target Variable
            target_column_name="math_score"
            numerical_columns = ["writing_score", "reading_score"]

            input_feature_train_df=train_df.drop(columns=[target_column_name])
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name])
            target_feature_test_df=test_df[target_column_name]

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object.")

            '''
               Save Preprocessing Object

               Purpose:
               During model inference or deployment, new data must go through the same preprocessing steps.
            '''
            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj

            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e,sys)
        


        '''preprocessing file(here pkl (pickle) format ) is used to store the preprocessing pipeline in the disk
           (in byte stream) so that when used later in memory during model training , it can correctly guide the 
            new or old data to be encoded and scaled in the same way to avoid any kind of ambiguity'''