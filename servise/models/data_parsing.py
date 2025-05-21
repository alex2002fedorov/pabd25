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

moscow_parser = cianparser.CianParser(location="Москва")
spb_parser = cianparser.CianParser(location="Санкт-Петербург")

def parse_flats_data(output_dir: str = './pabd25/data/raw'):

    try:
        for n_rooms in [1, 2, 3]:

            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
            csv_path = f'{output_dir}/{n_rooms}к_{timestamp}.csv'
            os.makedirs(output_dir, exist_ok=True)
            
            logging.info(f"Parsing {n_rooms}-room apartments...")
            
            
            data_moscow = moscow_parser.get_flats(
                deal_type="sale",
                rooms=(n_rooms,),
                with_saving_csv=False,
                additional_settings={"start_page": 1, "end_page": 10, "object_type": "secondary"}
            )
            
            data_spb = spb_parser.get_flats(
                deal_type="sale",
                rooms=(n_rooms,),
                with_saving_csv=False,
                additional_settings={"start_page": 1, "end_page": 10, "object_type": "secondary"}
            )
            
            df = pd.concat([pd.DataFrame(data_moscow), pd.DataFrame(data_spb)], ignore_index=True)
            df.to_csv(csv_path, index=False)
            
            logging.info(f"Saved {len(df)} entries in {csv_path}")
        
    except Exception as e:
        logging.error(f"Parsing error: {str(e)}")
        raise


if __name__ == '__main__':

    setup_logging()
    parse_flats_data()