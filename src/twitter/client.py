import tweepy
import pandas as pd
import boto3
from config import config
from datetime import datetime

def get_twitter_client():
    auth = tweepy.OAuth1UserHandler(
        config.twitter_api_key, 
        config.twitter_api_key_secret, 
        config.twitter_access_token, 
        config.twitter_access_token_secret
    )
    api = tweepy.API(auth)
    return api