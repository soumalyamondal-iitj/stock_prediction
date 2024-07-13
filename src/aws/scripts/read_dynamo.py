import pandas as pd
from aws_session.py import get_aws_session
from config import config

def read_from_dynamodb():
    # Initialize AWS session
    session = get_aws_session()

    # Initialize DynamoDB resource
    dynamodb = session.resource('dynamodb')

    # Specify the table
    table = dynamodb.Table(config.dynamodb_table_name)

    # Scan the table
    response = table.scan()

    # Load data into a pandas DataFrame
    data = response['Items']
    df = pd.DataFrame(data)

    return df