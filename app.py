import streamlit as st
import pandas as pd
from textblob import TextBlob
import joblib

# Load the data
@st.cache_data
def load_data():
    stock_data = pd.read_csv('stock_yfinance_data.csv')
    tweets_data = pd.read_csv('stock_tweets.csv')

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
model_filename = 'best_model.pkl'
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

st.write(f"Previous Close Price: {prev_close}")
st.write(f"Previous Sentiment: {prev_sentiment}")
st.write(f"7-day Moving Average: {ma7}")
st.write(f"14-day Moving Average: {ma14}")
st.write(f"Daily Change: {daily_change}")
st.write(f"Volatility: {volatility}")

if st.button("Predict"):
    predictions = []
    latest_date = latest_data['Date']

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

    st.write(f"Predicted next {days_to_predict} days close prices: {predictions}")

st.write("Use the inputs above to predict the next days close prices of the stock.")