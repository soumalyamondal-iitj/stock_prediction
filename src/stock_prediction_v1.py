import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import mlflow
import mlflow.sklearn
import xgboost as xgb
import joblib
import re
import boto3
import os

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
def clean_tweet(tweet):
    tweet = re.sub(r'http\S+', '', tweet)
    tweet = re.sub(r'@\w+', '', tweet)
    tweet = re.sub(r'#', '', tweet)
    tweet = re.sub(r'RT[\s]+', '', tweet)
    tweet = re.sub(r'\W', ' ', tweet)
    tweet = tweet.lower()
    return tweet

tweets_data['cleaned_tweet'] = tweets_data['Tweet'].apply(clean_tweet)

def get_sentiment(tweet):
    analysis = TextBlob(tweet)
    return analysis.sentiment.polarity

tweets_data['Sentiment'] = tweets_data['cleaned_tweet'].apply(get_sentiment)

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

# Define features and target
X = merged_data[['Prev_Close', 'Prev_Sentiment', 'MA7', 'MA14', 'Daily_Change', 'Volatility']]
y = merged_data['Close']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Plotting sentiment vs. stock price
def plot_sentiment_vs_price(stock_name):
    stock_subset = merged_data[merged_data['Stock Name'] == stock_name]

    fig, ax1 = plt.subplots()

    ax1.set_xlabel('Date')
    ax1.set_ylabel('Stock Price', color='tab:blue')
    ax1.plot(stock_subset['Date'], stock_subset['Close'],
color='tab:blue', label='Close Price')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Sentiment', color='tab:orange')
    ax2.plot(stock_subset['Date'], stock_subset['Sentiment'],
color='tab:orange', label='Sentiment')
    ax2.tick_params(axis='y', labelcolor='tab:orange')

    fig.tight_layout()
    plt.title(f'Sentiment vs Stock Price for {stock_name}')
    plt.show()

# Plot sentiment vs. price for each stock
unique_stocks = merged_data['Stock Name'].unique()
for stock in unique_stocks:
    plot_sentiment_vs_price(stock)

# Define models and hyperparameter grids
models = {
    "Random Forest": RandomForestRegressor(random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    "SVR": SVR(),
    "XGBoost": xgb.XGBRegressor(random_state=42)
}

param_grids = {
    "Random Forest": {
        'n_estimators': [100, 200],
        'max_depth': [3, 5, 7]
    },
    "Gradient Boosting": {
        'n_estimators': [100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7]
    },
    "SVR": {
        'C': [0.1, 1, 10],
        'epsilon': [0.01, 0.1, 0.2]
    },
    "XGBoost": {
        'n_estimators': [100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7]
    }
}

# Compare models
best_model = None
best_params = None
best_mse = float("inf")
mlflow.set_experiment("Stock Analysis")
for model_name, model in models.items():
    grid_search = GridSearchCV(model, param_grids[model_name], cv=3, n_jobs=-1, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)
    model_best = grid_search.best_estimator_
    model_best_params = grid_search.best_params_
    y_pred = model_best.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    with mlflow.start_run(run_name=model_name):
        # Train the model
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)

        # Log parameters and metrics
        mlflow.log_param("model_name", model_name)
        mlflow.log_metric("rmse", mse)
        mlflow.log_metric("mae", mean_absolute_error(y_test, y_pred))
        mlflow.log_metric("r2", r2_score(y_test, y_pred))
        # Log the model
        mlflow.sklearn.log_model(model, "model")

    print(f'{model_name} Best Parameters: {model_best_params}')
    print(f'{model_name} Mean Squared Error: {mse}')

    if mse < best_mse:
        best_mse = mse
        best_model = model_best
        best_params = model_best_params

print(f'Best Model: {best_model} with MSE: {best_mse}')

# Register the final model with MLflow
with mlflow.start_run(run_name="Best Model"):
    mlflow.log_param("model_name", best_model)
    mlflow.log_params(best_params)
    mlflow.log_metric("mse", best_mse)
    mlflow.sklearn.log_model(best_model, "model")

    # Predict future stock prices for the next 10 days
    latest_date = merged_data['Date'].max()
    latest_data = merged_data[merged_data['Date'] == latest_date]
    stock_name = latest_data['Stock Name'].values[0]

    predictions = []
    for i in range(10):
        future_close = latest_data['Close'].values[0]
        future_sentiment = latest_data['Sentiment'].values[0]
        future_ma7 = latest_data['MA7'].values[0]
        future_ma14 = latest_data['MA14'].values[0]
        future_change = latest_data['Daily_Change'].values[0]
        future_volatility = latest_data['Volatility'].values[0]

        X_future = pd.DataFrame({
            'Prev_Close': [future_close],
            'Prev_Sentiment': [future_sentiment],
            'MA7': [future_ma7],
            'MA14': [future_ma14],
            'Daily_Change': [future_change],
            'Volatility': [future_volatility]
        })

        next_day_prediction = best_model.predict(X_future)[0]
        predictions.append(next_day_prediction)

        # Update latest_data for next iteration
        new_row = pd.DataFrame({
            'Date': [latest_date + pd.Timedelta(days=i+1)],
            'Close': [next_day_prediction],
            'Sentiment': [future_sentiment],
            'MA7': [(future_ma7 * 6 + next_day_prediction) / 7],  # Simplified rolling calculation
            'MA14': [(future_ma14 * 13 + next_day_prediction) / 14],  # Simplified rolling calculation
            'Daily_Change': [next_day_prediction - future_close],
            'Volatility': [future_volatility]  # Assuming volatility stays the same for simplicity
        })
        latest_data = pd.concat([latest_data, new_row], ignore_index=True)

    print(f'Predicted next 10 days close prices for {stock_name}: {predictions}')
    for day, price in enumerate(predictions, start=1):
        mlflow.log_metric(f"day_{day}_prediction", price)


#**Save the trained model**:
joblib.dump(best_model, './model/best_model.pkl')