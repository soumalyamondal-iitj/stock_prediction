import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np
import re
from datetime import timedelta

# Load the best model
model_filename = 'best_model.pkl'
model = joblib.load(model_filename)

# Load data for reference
tweets_data_path = 'stock_tweets.csv'
stock_data_path = 'stock_yfinance_data.csv'

tweets_df = pd.read_csv(tweets_data_path, engine='python', on_bad_lines='skip')
stock_df = pd.read_csv(stock_data_path)

# Preprocess the data
tweet_column = 'Tweet'  
date_column = 'Date' 

# Clean and process tweets
def clean_tweet(tweet):
    tweet = re.sub(r'http\S+', '', tweet)
    tweet = re.sub(r'@\w+', '', tweet)
    tweet = re.sub(r'#', '', tweet)
    tweet = re.sub(r'RT[\s]+', '', tweet)
    tweet = re.sub(r'\W', ' ', tweet)
    tweet = tweet.lower()
    return tweet

tweets_df['cleaned_tweet'] = tweets_df[tweet_column].apply(clean_tweet)

# Perform sentiment analysis
from textblob import TextBlob

def get_sentiment(tweet):
    analysis = TextBlob(tweet)
    return analysis.sentiment.polarity

tweets_df['sentiment'] = tweets_df['cleaned_tweet'].apply(get_sentiment)

# Ensure date formats are consistent and without timezone information
tweets_df[date_column] = pd.to_datetime(tweets_df[date_column]).dt.tz_localize(None)
stock_df['Date'] = pd.to_datetime(stock_df['Date']).dt.tz_localize(None)

# Aggregate sentiment by date
sentiment_by_date = tweets_df.groupby(date_column)['sentiment'].mean().reset_index()

# Merge the dataframes on date
merged_df = pd.merge(stock_df, sentiment_by_date, left_on='Date', right_on=date_column, how='inner')

# Prepare data for prediction
X = merged_df[['Open', 'High', 'Low', 'Close', 'Volume', 'sentiment']]
y = merged_df['Close'].shift(-1).dropna()
X = X[:-1]

# Streamlit App
st.title('Stock Price Prediction App')

st.write('## Data Overview')
st.write('### Stock Data')
st.write(stock_df.head())

st.write('### Tweets Data')
st.write(tweets_df.head())

st.write('### Merged Data for Model')
st.write(merged_df.head())

st.write('## Predict Future Stock Prices')
days_to_predict = st.slider('Days to Predict', 1, 30, 20)

# Predict the next `days_to_predict` days
predictions = []
last_known = X.iloc[-1].values.reshape(1, -1)
last_known_df = pd.DataFrame(last_known, columns=X.columns)  # Retain feature names

for _ in range(days_to_predict):
    next_pred = model.predict(last_known_df)
    predictions.append(next_pred[0])
    last_known_df = pd.DataFrame(np.append(last_known_df.values[:, :-1], next_pred[0]).reshape(1, -1), columns=X.columns)

# Create a date range for predictions
last_date = merged_df['Date'].iloc[-1]
future_dates = [last_date + timedelta(days=i+1) for i in range(days_to_predict)]

# Plot the predictions
fig, ax = plt.subplots(figsize=(14, 7))
ax.plot(merged_df['Date'], merged_df['Close'], label='Historical Close Price')
ax.plot(future_dates, predictions, label='Predicted Close Price', linestyle='--')
ax.set_xlabel('Date')
ax.set_ylabel('Close Price')
ax.legend()
st.pyplot(fig)