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

def setup_logging(log_file: str = './pabd25/logs/app.log') -> None:
    
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

def test_model():

    try:
        
        files = [f for f in os.listdir("./pabd25/models/") if f.endswith('.pkl')]
        latest_file = max(files, key=lambda x: os.path.getmtime(os.path.join("./pabd25/models/", x)))

        model = joblib.load("./pabd25/models/" + latest_file)
        model_path = "./pabd25/models/" + latest_file
        logger.info(f"The model was successfully loaded from {model_path}")

        df_test = pd.read_csv('./pabd25/data/processed/test.csv')
        
        # Разделение на тренировочную и тестовую выборки
        X_test =  df_test[['total_meters', "floor", "floors_count", "rooms_count", "location", "district", "underground"]]
        y_test = df_test['price']
        
        y_pred = model.predict(X_test)
        
        # Оценка модели
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        mean_abs_error = np.mean(np.abs(y_test - y_pred))
        
        # Логирование метрик
        logger.info(f"The metrics of the model:\n"
                   f"MSE: {mse:.2f}\n"
                   f"RMSE: {rmse:.2f}\n"
                   f"R²: {r2:.6f}\n"
                    f"Average absolute error: {mean_abs_error:.2f} rubles\n")
        
        return model
    
    except Exception as e:
        logger.error(f"Model prediction error: {str(e)}")
        raise


if __name__ == '__main__':
    setup_logging()
    test_model()