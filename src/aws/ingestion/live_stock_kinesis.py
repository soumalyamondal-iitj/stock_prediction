import requests
import pandas as pd
import boto3
from config import config
from datetime import datetime
import json
import schedule
import time

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

def upload_to_kinesis(stock_data, stream_name):
    kinesis_client = boto3.client(
        'kinesis',
        aws_access_key_id=config.aws_access_key_id,
        aws_secret_access_key=config.aws_secret_access_key,
        region_name=config.aws_region_name
    )

    for record in stock_data:
        kinesis_client.put_record(
            StreamName=stream_name,
            Data=json.dumps(record),
            PartitionKey=record['timestamp']
        )

def job():
    symbol = 'AAPL'
    stream_name = 's3-kinesis-etl-stream'
    
    try:
        stock_data = fetch_stock_data(symbol)
        upload_to_kinesis(stock_data, stream_name)
        print(f"Stock data for {symbol} uploaded to Kinesis successfully at {datetime.now()}.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

# Schedule the job every day at a specific time, e.g., 12:00 PM
schedule.every().day.at("12:00").do(job)

print("Scheduler started. Waiting for the job to run...")

# Keep the script running
# while True:
#     schedule.run_pending()
#     time.sleep(1)