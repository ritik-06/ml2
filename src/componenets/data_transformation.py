import os,sys
from src.exception import CustomeException
from src.logger import logging
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.impute import SimpleImputer
from dataclasses import dataclass
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')
class DataTransformation:
    def __init__(self):
        self.data_transfromation_config = DataTransformationConfig()
    def get_data_tranformer_object(self):

        '''  
            this function is responsible for data transfromation
        '''
        try:
            numerical_features = ['writing_score','reading_score']
            categorical_features = ['gender','race_ethnicity','parental_level_of_education','lunch','test_preparation_course']
            num_pipeline = Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='median')),
                    ('scaler',StandardScaler())
                ]
            )
            cat_pipeline = Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='most_frequent')),
                    ('ohe',OneHotEncoder())
                    
                ]
            )
            preporcessor = ColumnTransformer(
                [
                    ('num_pipeline',num_pipeline,numerical_features),
                    ('cat_pipeline',cat_pipeline,categorical_features)
                ]
            )
            return preporcessor

        except Exception as e:
            raise CustomeException(e,sys)
        



    def initiate_data_transfromation(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("reading of train and test data completed")

            logging.info("obtaining preprocessing object")
            preprocessing_obj = self.get_data_tranformer_object()

            target_column_name = 'math_score' 
            ## x_train
            input_features_train_df = train_df.drop(columns=[target_column_name],axis=1)
            ## y_train
            target_feature_train = train_df[target_column_name]

            ##x_test

            input_features_test_df = test_df.drop(columns=[target_column_name],axis=1)
            ## y_test
            target_feature_test = test_df[target_column_name]

            logging.info("Applying preprocessing object on training dataframe and testing dataframe")


            ## Xtrain and Xtest tranfromation 
            input_feature_train_arr = preprocessing_obj.fit_transform(input_features_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_features_test_df)

            ## np.c_ helps to column-wise concatenate two arrays  
            train_arr = np.c_[
                input_feature_train_arr,np.array(target_feature_train)
            ]
            test_arr = np.c_[
                input_feature_test_arr,np.array(target_feature_test)
            ]

            logging.info('Saved preprocessing object.')
            ## saving pickle file of preprocessor object
            save_object(
                file_path = self.data_transfromation_config.preprocessor_obj_file_path,
                obj = preprocessing_obj
            )


            return(
                train_arr,
                test_arr,
                self.data_transfromation_config.preprocessor_obj_file_path,
            )


        except Exception as e:
            raise CustomeException(e,sys)