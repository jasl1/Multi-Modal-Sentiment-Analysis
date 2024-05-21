import boto3
import json

# Serialize the model
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

# Serialize the tokenizer
tokenizer_json = tokenizer.to_json()
with open("tokenizer.json", "w") as json_file:
    json_file.write(tokenizer_json)

# Upload model files to S3
s3 = boto3.client('s3', region_name='your-region')
bucket_name = 'your-bucket-name'
s3.upload_file('model.json', bucket_name, 'model.json')
s3.upload_file('tokenizer.json', bucket_name, 'tokenizer.json')

# Create a Lambda function
lambda_client = boto3.client('lambda', region_name='your-region')
lambda_function_name = 'sentiment-analysis-function'

with open('lambda_function.py', 'r') as file:
    lambda_code = file.read()

response = lambda_client.create_function(
    FunctionName=lambda_function_name,
    Runtime='python3.8',
    Role='your-lambda-role-arn',
    Handler='lambda_function.lambda_handler',
    Code={
        'ZipFile': lambda_code.encode()
    },
    Environment={
        'Variables': {
            'S3_BUCKET': bucket_name,
            'MODEL_FILE': 'model.json',
            'TOKENIZER_FILE': 'tokenizer.json'
        }
    }
)

# Expose an API endpoint using API Gateway
api_gateway = boto3.client('apigateway', region_name='your-region')
api_name = 'sentiment-analysis-api'

response = api_gateway.create_rest_api(
    name=api_name,
    endpointConfiguration={
        'types': ['REGIONAL']
    }
)

api_id = response['id']

root_resource_id = api_gateway.get_resources(restApiId=api_id)['items'][0]['id']

lambda_arn = response_lambda['FunctionArn']

response = api_gateway.put_integration(
    restApiId=api_id,
    resourceId=root_resource_id,
    httpMethod='POST',
    type='AWS_PROXY',
    integrationHttpMethod='POST',
    uri=lambda_arn
)

response = api_gateway.put_method(
    restApiId=api_id,
    resourceId=root_resource_id,
    httpMethod='POST',
    authorizationType='NONE'
)

response = api_gateway.create_deployment(
    restApiId=api_id,
    stageName='prod'
)

api_url = f'https://{api_id}.execute-api.{region}.amazonaws.com/prod'

print("API Endpoint:", api_url)
