import requests
import pandas as pd
import boto3
from config import config
from datetime import datetime

def fetch_stock_data(symbol):
    url = f'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={symbol}&interval=1min&apikey={config.alpha_vantage_api_key}'
    response = requests.get(url)
    data = response.json()
    
    if "Time Series (1min)" in data:
        time_series = data["Time Series (1min)"]
        stock_data = []
        for timestamp, values in time_series.items():
            stock_data.append({
                'timestamp': timestamp,
                'open': values['1. open'],
                'high': values['2. high'],
                'low': values['3. low'],
                'close': values['4. close'],
                'volume': values['5. volume']
            })
        return stock_data
    else:
        raise Exception("Error fetching stock data: " + data.get("Note", "Unknown error"))

def save_to_csv(stock_data, file_name):
    df = pd.DataFrame(stock_data)
    df.to_csv(file_name, index=False)

def upload_to_s3(file_name, bucket_name, object_name=None):
    if object_name is None:
        object_name = file_name
    s3_client = boto3.client(
        's3',
        aws_access_key_id=config.aws_access_key_id,
        aws_secret_access_key=config.aws_secret_access_key,
        region_name=config.aws_region_name
    )
    s3_client.upload_file(file_name, bucket_name, object_name)
