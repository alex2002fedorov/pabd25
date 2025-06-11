# Использовать официальный образ Python для базового изображения.  
FROM python:3.13-slim  
# Установить рабочий каталог в контейнере на /app.  
WORKDIR /app  
# Скопировать файлы из текущего каталога в /app контейнера.  
ADD . /app  
# Установить необходимые пакеты, указанные в файле requirements.txt.  
RUN pip install --no-cache-dir -r requirements.txt
# Сделать порт 5000 доступным снаружи контейнера.  
EXPOSE 6000
# Запустить Gunicorn при запуске контейнера.  
CMD ["gunicorn", "-b", ":6000", "-w", "1", "servise.app:app"]
