import datetime
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import numpy as np
import glob
import cianparser
import logging
from typing import Tuple, Optional
import json

def setup_logging(log_file: str = './pabd25/logs/app.log'):
    
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

logger = logging.getLogger(__name__)

def train_and_evaluate_model():

    try:
        
        logger.info("The beginning of model training")

        df_train = pd.read_csv('./pabd25/data/processed/train.csv')

        df_test = pd.read_csv('./pabd25/data/processed/test.csv')
        
        # Разделение на тренировочную и тестовую выборки
        X_train = df_train[['total_meters', "floor", "floors_count", "rooms_count", "location", "district", "underground"]]
        X_test =  df_test[['total_meters', "floor", "floors_count", "rooms_count", "location", "district", "underground"]]
        y_train = df_train['price'] 
        y_test = df_test['price']
        
        # Обучение модели
        model = GradientBoostingRegressor()
        model.fit(X_train, y_train)
        logger.info("The model has been successfully trained")
        
        try:
            t = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
            model_path = f'./pabd25/models/house_price_model_{t}.pkl'
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            joblib.dump(model, model_path)
            logger.info(f"The model was successfully saved in {model_path}")
        except Exception as e:
            logger.error(f"Error saving the model: {str(e)}")
            raise
    
    except Exception as e:
        logger.error(f"Error when training the model: {str(e)}")
        raise

if __name__ == '__main__':

    setup_logging()
    train_and_evaluate_model()