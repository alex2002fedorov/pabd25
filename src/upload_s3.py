"""Upload selected files to S3 storage"""
import argparse
import os
from dotenv import dotenv_values
import boto3

BUCKET_NAME = 'pabd25'
YOUR_SURNAME = 'fedorov'
FILE_PATH = './models/house_price_model_2025-05-21_21-28.pkl'
os.makedirs('models', exist_ok=True)
config = dotenv_values(".env")


def main(args):
    client = boto3.client(
        's3',
        endpoint_url='https://storage.yandexcloud.net',
        aws_access_key_id=config['aws_access_key_id'],
        aws_secret_access_key=config['aws_secret_access_key']
    )

    object_name = f'{YOUR_SURNAME}/' + args.input
    client.upload_file(Bucket=BUCKET_NAME, 
                            Key=object_name, 
                            Filename=args.input)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input',
                        help='Input data files to download from S3 storage',
                        default=FILE_PATH)
    args = parser.parse_args()
    main(args)