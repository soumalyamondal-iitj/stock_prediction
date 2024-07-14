import tweepy
import pandas as pd
import boto3
from config import config
from datetime import datetime
from twitter.client import get_twitter_client
def fetch_tweets(query, count=100):
    api = get_twitter_client()
    tweets = api.search_tweets(q=query, count=count, tweet_mode='extended')
    tweet_data = []
    for tweet in tweets:
        tweet_data.append({
            'id': tweet.id_str,
            'created_at': tweet.created_at,
            'text': tweet.full_text,
            'user': tweet.user.screen_name,
            'user_followers': tweet.user.followers_count,
            'retweet_count': tweet.retweet_count,
            'favorite_count': tweet.favorite_count,
        })
    return tweet_data

def save_to_csv(tweet_data, file_name):
    df = pd.DataFrame(tweet_data)
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