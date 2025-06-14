from flask import Flask, request, jsonify, render_template
import logging
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
import os

app = Flask(__name__)

from flask_cors import CORS
cors = CORS(app, resources={r"/api/*": {"origins": "http://localhost:5050"}})

import dotenv
config = dotenv.dotenv_values('.env')
print(config)
print(config['APP_TOKEN'])

from flask_httpauth import HTTPTokenAuth
auth = HTTPTokenAuth(scheme='Bearer')
tokens = { config['APP_TOKEN']: "user1", }

@auth.verify_token
def verify_token(token):
    if token in tokens:
        return tokens[token]

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('./logs/app.log'), 
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def get_latest_file_scandir(directory):
    """Более эффективный способ найти последний созданный файл"""
    with os.scandir(directory) as entries:

        files = [entry for entry in entries if entry.is_file()]
        
        if not files:
            return None
            
        # Находим файл с максимальным временем создания
        latest_file = max(files, key=lambda x: x.stat().st_ctime)
        return latest_file.path

# Загрузка модели при старте приложения
latest_file = get_latest_file_scandir('./models')
print(latest_file)
model = joblib.load(latest_file)

def format_rubles(amount: float) -> str:
    """
    Форматирует сумму в рублях в читаемый вид: 13 млн 684 тыс 413 руб 15 коп
    
    Args:
        amount: Сумма в рублях (float или int)
        
    Returns:
        Отформатированная строка
    """
    rubles = int(amount)
    kopecks = round((amount - rubles) * 100)
    
    millions = rubles // 1_000_000
    rubles %= 1_000_000
    
    thousands = rubles // 1_000
    rubles %= 1_000
    
    parts = []
    if millions > 0:
        parts.append(f"{millions} млн")
    if thousands > 0:
        parts.append(f"{thousands} тыс")
    if rubles > 0 or not parts:
        parts.append(f"{rubles} руб")
    if kopecks > 0:
        parts.append(f"{kopecks} коп")
    
    return ' '.join(parts)

# Маршрут для отображения формы
@app.route('/')
def index():
    return render_template('index.html')

# Маршрут для обработки данных формы
@app.route('/api/numbers', methods=['POST'])
@auth.login_required
def process_numbers():

    logger.info("Request received for /api/numberss")
    data = request.get_json()

    if not data:
        logger.error("Error: no data received")
        return {'status': 'error', 'message': 'Ошибка при получении данных. Данные не пришли'}
        
    num1 = data.get('number1')
    num2 = data.get('number2')
    num3 = data.get('number3')
    num4 = data.get('number4')
    num5 = data.get('number5')
    num6 = data.get('number6')
    num7 = data.get('number7')

    try:
        print(num5)
        print(num6)
        print(num7)
        num1 = float(num1)
        num2 = int(num2)
        num3 = int(num3)
        num4 = int(num4)
        num4 = int(num5)
        num4 = int(num6)
        num4 = int(num7)
        logger.info(f"The data has been successfully converted: {num1}, {num2}, {num3}, {num4}, {num5}, {num6}, {num7}")
    except (ValueError, TypeError) as e:
        logger.error(f"Error in data processing: {e}")
        return {'status': 'error', 'message': 'Ошибка при обработке данных'}
    
    logger.info("=== Data received ===")
    logger.info(f"The area of the apartment: {num1}")
    logger.info(f"Number of rooms: {num2}")
    logger.info(f"Floors in the house: {num3}")
    logger.info(f"Apartment floor: {num4}")
    logger.info("=====================\n")

    if model is not None:
        try:
            # Преобразуем данные в 2D массив для модели
            prediction = model.predict([[num1, num4, num3, num2, num5, num6, num7]])[0]
            predicted_price = round(float(prediction), 2)
            logger.info(f"Predicted price: {predicted_price}")
            return {'status': 'success', 'message': f'Предсказанная цена: {format_rubles(predicted_price)}'}
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return {'status': 'error', 'message': 'Ошибка при предсказании цены'}
    else:
        logger.error("Model not available")
        return {'status': 'error', 'message': 'Модель не загружена'}

    return {'status': 'success', 'message': 'Цена: ' + str(300000 *num1)}

if __name__ == '__main__':
    logger.info("The server is running")
    app.run(debug=True)

