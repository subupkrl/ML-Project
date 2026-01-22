import os
import sys
from dataclasses import dataclass
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object,evaluate_models
from sklearn.metrics import r2_score

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artefacts',"model.pkl")
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        
    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split train and test input data")
            X_train,y_train,X_test,y_test = (
                train_array[:,:-1],  #features for training
                train_array[:,-1],   #target variable for training
                test_array[:,:-1],   #features for training
                test_array[:,-1]     #target variable for training
            )
            models = {
                'Random Forest': RandomForestRegressor(),
                'Linear Regression': LinearRegression(),
                'Decision Tree': DecisionTreeRegressor(),
                'Gradient Boosting':GradientBoostingRegressor(),
                'K-Neighbors': KNeighborsRegressor(),
                'XGBRegressor':XGBRegressor(),
                'AdaBoostRegressor':AdaBoostRegressor()
            }
            
            params = {
                "Decision Tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "K-Neighbors Regressor": {

                    "n_neighbors": [3, 5, 7, 9, 11],  # Number of neighbors to use
                    "weights": ["uniform", "distance"],  # Weighting function for predictions
                    "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],  # Algorithm for nearest neighbor 1  search
                    #"leaf_size": [10, 20, 30, 40],  # Leaf size for BallTree or KDTree
                    #"p": [1, 2]  # Power parameter for the Minkowski metric
                },
                "Random Forest":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Gradient Boosting":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Linear Regression":{},
                "XGBRegressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "AdaBoost Regressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                } 
            }
            
            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models,param=params)
            
            #to get best model score from dictionary
            best_model_score = max(sorted(model_report.values()))
            
            #to get the best model name 
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            
            best_model = models[best_model_name]
            
            #define threshold
            if best_model_score < 0.6:
                raise CustomException('No best model found')
            logging.info(f'Best model is  {best_model}')
            
            #save the model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            
            predicted_result = best_model.predict(X_test)
            r2_score_value = r2_score(y_test,predicted_result)
            logging.info(f'The probability r2 score is {r2_score_value}')
            logging.info(f'Final model saved succesfully')
            return r2_score_value,best_model
            
        except Exception as e:
            return CustomException(e,sys)
         