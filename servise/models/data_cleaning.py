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

# Настройка логирования
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

def clean_and_prepare_data():

    try:
        
        logger.info("The beginning of data processing")

        # Загрузка всех CSV файлов из директории
        raw_data_path = './pabd25/data/raw'
        file_list = glob.glob(raw_data_path + "/*.csv")
        
        if not file_list:
            logger.warning("There are no CSV files in the raw directory for processing")
            return pd.DataFrame()
        
        logger.info(f"{len(file_list)} CSV files found for processing")
        
        # Объединение данных
        main_dataframe = pd.read_csv(file_list[0], delimiter=',')
        for i in range(1, len(file_list)):
            data = pd.read_csv(file_list[i], delimiter=',')
            df = pd.DataFrame(data)
            main_dataframe = pd.concat([main_dataframe, df], axis=0)
        
        initial_count = len(main_dataframe)
        logger.info(f"The combined dataset contains {initial_count} records")
        
        # Выбор нужных столбцов
        new_dataframe = main_dataframe[['total_meters', 'price', "floor", "floors_count", "rooms_count", "location", "district", "underground"]]
        
        # Удаление пропущенных значений
        new_dataframe = new_dataframe.dropna()
        logger.info(f"Deleted {initial_count - len(new_dataframe)} rows with missing values")
        
        # Удаление дубликатов
        new_dataframe = new_dataframe.drop_duplicates()
        logger.info(f"Deleted {len(main_dataframe) - len(new_dataframe)} duplicates")
        
        # Удаление выбросов
        def remove_outliers(df, column):
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
        
        initial_size = len(new_dataframe)
        for col in ['total_meters', 'price', 'floor', 'floors_count', 'rooms_count']:
            new_dataframe = remove_outliers(new_dataframe, col)
        
        logger.info(f"Removed {initial_size - len(new_dataframe)} emissions")
        logger.info(f"Total dataset size: {len(new_dataframe)} rows")

        #Замена категориальных признаков
        mapping_dict = {
            'location': {val: idx+1 for idx, val in enumerate(new_dataframe['location'].unique())},
            'district': {val: idx+1 for idx, val in enumerate(new_dataframe['district'].unique())},
            'underground': {val: idx+1 for idx, val in enumerate(new_dataframe['underground'].unique())}
        }
        for column in ['location', 'district', 'underground']:
            new_dataframe[column] = new_dataframe[column].map(mapping_dict[column])
        
        os.makedirs(os.path.dirname('./pabd25/indo_dataset/info.json'), exist_ok=True)

        with open('./pabd25/indo_dataset/info.json', 'w', encoding='utf-8') as f:
            json.dump(mapping_dict, f, ensure_ascii=False, indent=4)
        
        # Сохранение очищенных данных
        cleaned_path = './pabd25/data/cleaned_data.csv'
        new_dataframe.to_csv(cleaned_path, index=False)
        logger.info(f"The cleaned data is saved in {cleaned_path}")

        train_df, test_df = train_test_split(new_dataframe, test_size=0.2, random_state=42)

        cleaned_path = './pabd25/data/processed/train.csv'
        train_df.to_csv(cleaned_path, index=False)
        logger.info(f"The train_df is saved in {cleaned_path}")

        cleaned_path = './pabd25/data/processed/test.csv'
        test_df.to_csv(cleaned_path, index=False)
        logger.info(f"The test_df is saved in {cleaned_path}")

    except Exception as e:
        logger.error(f"Error when clearing data: {str(e)}")
        raise

if __name__ == '__main__':

    setup_logging()
    clean_and_prepare_data()