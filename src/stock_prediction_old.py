# BDM Project - Stock Price Prediction based on Tweets Sentiment
import mlflow
import mlflow.sklearn
import pandas as pd
import re
import numpy as np
import joblib
from textblob import TextBlob
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# Set up MLflow
mlflow.set_experiment("Stock Price Prediction")

# Load the data
tweets_data_path = 'stock_tweets.csv'
stock_data_path = 'stock_yfinance_data.csv'

# Read tweets data with error handling
tweets_df = pd.read_csv(tweets_data_path, engine='python', on_bad_lines='skip')
stock_df = pd.read_csv(stock_data_path)

# Display the first few rows of each dataframe
tweets_df.head(), stock_df.head()


# Inspect column names
tweets_df.columns

tweet_column = 'Tweet'
date_column = 'Date'

# Clean the tweets
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
#merged_df = merged_df.drop(columns=[date_column])
merged_df.head()


# Plot the data
fig, ax1 = plt.subplots(figsize=(14, 7))

color = 'tab:blue'
ax1.set_xlabel('Date')
ax1.set_ylabel('Stock Close Price', color=color)
ax1.plot(merged_df['Date'], merged_df['Close'], color=color, label='Close Price')
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # Instantiate a second axes that shares the same x-axis
color = 'tab:orange'
ax2.set_ylabel('Sentiment', color=color)
ax2.plot(merged_df['Date'], merged_df['sentiment'], color=color, label='Sentiment')
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # Ensure the plot elements fit into the figure area
plt.title('Stock Close Price and Sentiment Over Time')
plt.show()


# Feature and target variables
X = merged_df[['Open', 'High', 'Low', 'Close', 'Volume', 'sentiment']]
y = merged_df['Close'].shift(-1).dropna()
X = X[:-1]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define models
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest Regressor": RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting Regressor": GradientBoostingRegressor(n_estimators=100, random_state=42)
}

# Train and evaluate models
best_model = None
best_rmse = float('inf')

for model_name, model in models.items():
    with mlflow.start_run(run_name=model_name):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Log parameters, metrics, and model
        mlflow.log_param("model_name", model_name)
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)
        mlflow.sklearn.log_model(model, "model")

        print(f"{model_name} - RMSE: {rmse}, MAE: {mae}, R2: {r2}")

        if rmse < best_rmse:
            best_rmse = rmse
            best_model = model
            best_model_name = model_name

# Register the best model
with mlflow.start_run(run_name=f"Best Model - {best_model_name}") as run:
    mlflow.log_param("best_model_name", best_model_name)
    mlflow.log_metric("best_rmse", best_rmse)
    mlflow.sklearn.log_model(best_model, "best_model")

    model_uri = f"runs:/{run.info.run_id}/best_model"
    mlflow.register_model(model_uri, "BestStockPricePredictionModel")

print(f"Best model {best_model_name} with RMSE {best_rmse} registered in MLflow.")

# Predict the next 20 days
predictions = []
last_known = X.iloc[-1].values.reshape(1, -1)
last_known_df = pd.DataFrame(last_known, columns=X.columns)  # Retain feature names

for _ in range(20):
    next_pred = best_model.predict(last_known_df)
    predictions.append(next_pred[0])
    last_known_df = pd.DataFrame(np.append(last_known_df.values[:, :-1], next_pred[0]).reshape(1, -1), columns=X.columns)

predictions


#**Save the trained model**:
joblib.dump(best_model, 'best_model.pkl')