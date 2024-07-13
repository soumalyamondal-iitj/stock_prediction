import boto3
import csv
import os

# to-do - use in aws env. handler to load data from s3 to dynamo
def lambda_handler(event, context):
    s3_client = boto3.client('s3')
    dynamodb = boto3.resource('dynamodb')
    
    # Get the bucket and file name from the event
    bucket = event['Records'][0]['s3']['bucket']['name']
    key = event['Records'][0]['s3']['object']['key']
    
    # Download the CSV file from S3
    download_path = f'/tmp/{key}'
    s3_client.download_file(bucket, key, download_path)
    
    # Read the CSV file
    table = dynamodb.Table('YourDynamoDBTableName')
    with open(download_path, 'r') as csvfile:
        csv_reader = csv.DictReader(csvfile)
        for row in csv_reader:
            # Put item into DynamoDB
            table.put_item(Item=row)
    
    return {
        'statusCode': 200,
        'body': 'Data loaded successfully!'
    }
