import streamlit as st
import pandas as pd
from textblob import TextBlob
import joblib
import matplotlib.pyplot as plt
import datetime
import boto3
import os
# Load the data
@st.cache_data
def load_data():
    
    # Initialize a session using Amazon DynamoDB
    session = boto3.Session(
        aws_access_key_id=os.environ["key_id"],
        aws_secret_access_key=os.environ["access_key"],
        region_name=os.environ["region"]
    )

    # # Initialize DynamoDB resource
    dynamodb = session.resource('dynamodb')

    # # Specify the table
    table = dynamodb.Table('Tweets')

    # # Scan the table
    response = table.scan()

    # # Load data into a pandas DataFrame
    data = response['Items']

    # Load the datasets
    stock_data = pd.read_csv('./data/stock_yfinance_data.csv')
    # tweets_data = pd.read_csv('./data/stock_tweets.csv')
    tweets_data = pd.DataFrame(data)

    # Convert the Date columns to datetime
    stock_data['Date'] = pd.to_datetime(stock_data['Date'])
    tweets_data['Date'] = pd.to_datetime(tweets_data['Date']).dt.date

    # Perform sentiment analysis on tweets
    def get_sentiment(tweet):
        analysis = TextBlob(tweet)
        return analysis.sentiment.polarity

    tweets_data['Sentiment'] = tweets_data['Tweet'].apply(get_sentiment)

    # Aggregate sentiment by date and stock
    daily_sentiment = tweets_data.groupby(['Date', 'Stock Name']).mean(numeric_only=True).reset_index()

    # Convert the Date column in daily_sentiment to datetime64[ns]
    daily_sentiment['Date'] = pd.to_datetime(daily_sentiment['Date'])

    # Merge stock data with sentiment data
    merged_data = pd.merge(stock_data, daily_sentiment, how='left', left_on=['Date', 'Stock Name'], right_on=['Date', 'Stock Name'])

    # Fill missing sentiment values with 0 (neutral sentiment)
    merged_data['Sentiment'].fillna(0, inplace=True)

    # Sort the data by date
    merged_data.sort_values(by='Date', inplace=True)

    # Create lagged features
    merged_data['Prev_Close'] = merged_data.groupby('Stock Name')['Close'].shift(1)
    merged_data['Prev_Sentiment'] = merged_data.groupby('Stock Name')['Sentiment'].shift(1)

    # Create moving averages
    merged_data['MA7'] = merged_data.groupby('Stock Name')['Close'].transform(lambda x: x.rolling(window=7).mean())
    merged_data['MA14'] = merged_data.groupby('Stock Name')['Close'].transform(lambda x: x.rolling(window=14).mean())

    # Create daily price changes
    merged_data['Daily_Change'] = merged_data['Close'] - merged_data['Prev_Close']

    # Create volatility
    merged_data['Volatility'] = merged_data.groupby('Stock Name')['Close'].transform(lambda x: x.rolling(window=7).std())

    # Drop rows with missing values
    merged_data.dropna(inplace=True)

    return merged_data

data = load_data()
stock_names = data['Stock Name'].unique()

# Load the best model
model_filename = 'model/best_model.pkl'
model = joblib.load(model_filename)

st.title("Stock Price Prediction Using Sentiment Analysis")

# User input for stock data
st.header("Input Stock Data")
selected_stock = st.selectbox("Select Stock Name", stock_names)
days_to_predict = st.number_input("Number of Days to Predict",
min_value=1, max_value=30, value=10)

# Get the latest data for the selected stock
latest_data = data[data['Stock Name'] == selected_stock].iloc[-1]
prev_close = latest_data['Close']
prev_sentiment = latest_data['Sentiment']
ma7 = latest_data['MA7']
ma14 = latest_data['MA14']
daily_change = latest_data['Daily_Change']
volatility = latest_data['Volatility']

# Display the latest stock data in a table
latest_data_df = pd.DataFrame({
    'Metric': ['Previous Close Price', 'Previous Sentiment', '7-day Moving Average', '14-day Moving Average', 'Daily Change', 'Volatility'],
    'Value': [prev_close, prev_sentiment, ma7, ma14, daily_change, volatility]
})

st.write("Latest Stock Data:")
st.write(latest_data_df)

st.write("Use the inputs above to predict the next days close prices of the stock.")
if st.button("Predict"):
    predictions = []
    latest_date = datetime.datetime.now()

    for i in range(days_to_predict):
        X_future = pd.DataFrame({
            'Prev_Close': [prev_close],
            'Prev_Sentiment': [prev_sentiment],
            'MA7': [ma7],
            'MA14': [ma14],
            'Daily_Change': [daily_change],
            'Volatility': [volatility]
        })

        next_day_prediction = model.predict(X_future)[0]
        predictions.append(next_day_prediction)

        # Update features for next prediction
        prev_close = next_day_prediction
        ma7 = (ma7 * 6 + next_day_prediction) / 7  # Simplified rolling calculation
        ma14 = (ma14 * 13 + next_day_prediction) / 14  # Simplified rolling calculation
        daily_change = next_day_prediction - prev_close

    # st.write(f"Predicted next {days_to_predict} days close prices: {predictions}")
    # Prepare prediction data for display
    # Prepare prediction data for display
    prediction_dates = pd.date_range(start=latest_date + pd.Timedelta(days=1), periods=days_to_predict)
    prediction_df = pd.DataFrame({
        'Date': prediction_dates,
        'Predicted Close Price': predictions
    })

    st.subheader("Predicted Prices")
    st.write(prediction_df)

   # Plotting the results
    st.subheader("Prediction Chart")
    plt.figure(figsize=(10, 6))
    plt.plot(prediction_df['Date'], prediction_df['Predicted Close Price'], marker='o', linestyle='--', label="Predicted Close Price")
    plt.xlabel("Date")
    plt.ylabel("Close Price")
    plt.title(f"{selected_stock} Predicted Close Prices")
    plt.legend()
    st.pyplot(plt)
