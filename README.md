# Multi-Modal-Sentiment-Analysis
In this, project we develop a multi-modal sentiment analysis model to predict the sentiment (positive/negative) of movie reviews. We'll create a sentiment analysis model for movie reviews using RNN/LSTM, deploy it on AWS, manage the codebase with Git, and utilize multi-modal analysis by combining text data with metadata like genre or release year.

### Dataset: 
We'll use the IMDB movie review dataset, which contains movie reviews labeled as positive or negative.

### Tools and Libraries:

1. RNN/LSTM: TensorFlow or PyTorch for building the sentiment analysis model.
2. AWS: Deploy the model on AWS Lambda for serverless inference.
3. Git: Version control for managing the project codebase.
4. Multi-Modal: Combine text data with metadata like genre or release year.


### Data Collection: 
Gather the IMDB movie review dataset along with metadata like genre and release year. This step ensures we have the necessary data to train the sentiment analysis model.
```python
import pandas as pd
import requests
from bs4 import BeautifulSoup

# Function to scrape IMDb for movie metadata
def scrape_imdb_metadata(movie_title):
    url = f"https://www.imdb.com/find?q={movie_title}&s=tt&ttype=ft&ref_=fn_ft"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    first_result = soup.find('td', class_='result_text')
    if first_result:
        movie_link = first_result.find('a')['href']
        movie_id = movie_link.split('/')[2]
        movie_url = f"https://www.imdb.com/title/{movie_id}/"
        metadata_response = requests.get(movie_url)
        metadata_soup = BeautifulSoup(metadata_response.content, 'html.parser')
        genre = metadata_soup.find('div', class_='subtext').find('a').text
        release_year = metadata_soup.find('span', id='titleYear').find('a').text
        return genre, release_year
    else:
        return None, None

# Download IMDB movie review dataset
data = pd.read_csv('IMDB Dataset.csv')

# Collect metadata like genre and release year
metadata = []

for movie_title in data['title']:
    genre, release_year = scrape_imdb_metadata(movie_title)
    metadata.append({'title': movie_title, 'genre': genre, 'release_year': release_year})

metadata_df = pd.DataFrame(metadata)

# Merge metadata with the original dataset
data_with_metadata = pd.merge(data, metadata_df, on='title', how='left')

```

### Data Preprocessing: 
Clean and preprocess the text data by tokenizing, removing stopwords, and converting to lowercase. Encode the sentiment labels into numerical values and combine text data with metadata for multi-modal analysis.

```python
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.preprocessing import LabelEncoder

# Download NLTK resources
nltk.download('stopwords')
nltk.download('punkt')

# Preprocess text data: tokenization, removing stopwords, lowercase conversion
def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text.lower())
    filtered_text = [word for word in word_tokens if word not in stop_words]
    return " ".join(filtered_text)

data['cleaned_review'] = data['review'].apply(preprocess_text)

# Encode sentiment labels into numerical values
label_encoder = LabelEncoder()
data['encoded_sentiment'] = label_encoder.fit_transform(data['sentiment'])

# Combine text data with metadata (genre, release year)
data['combined_data'] = data['cleaned_review'] + ' ' + data['genre'] + ' ' + data['release_year']

```

### Model Development: 
Design and train an RNN/LSTM model for sentiment analysis using TensorFlow or PyTorch. This model will learn to predict whether a movie review is positive or negative based on the text data.

```python
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

# Tokenize text data
tokenizer = Tokenizer()
tokenizer.fit_on_texts(data['cleaned_review'])
sequences = tokenizer.texts_to_sequences(data['cleaned_review'])
max_sequence_length = max([len(seq) for seq in sequences])
padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length)

# Define RNN/LSTM model architecture
model = Sequential([
    Embedding(input_dim=len(tokenizer.word_index)+1, output_dim=100, input_length=max_sequence_length),
    LSTM(units=128),
    Dense(units=1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(padded_sequences, data['encoded_sentiment'], epochs=5, validation_split=0.2)

```

### Model Evaluation: 
Evaluate the performance of the trained model using standard metrics like accuracy, precision, recall, and F1-score. Cross-validation ensures that the model's performance is consistent across different subsets of the data.

```python
from sklearn.metrics import accuracy_score, classification_report

# Evaluate model performance
predictions = model.predict_classes(padded_sequences)
accuracy = accuracy_score(data['encoded_sentiment'], predictions)
report = classification_report(data['encoded_sentiment'], predictions)

print("Accuracy:", accuracy)
print("Classification Report:\n", report)

```

### Deployment on AWS: 
Deploy the trained model on AWS Lambda for serverless inference. Expose an endpoint for making predictions, allowing users to input movie reviews and receive sentiment predictions in real-time.

```python
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

```

### Version Control with Git: 
Use Git for version control to track changes to the codebase, collaborate with team members, and manage new features or bug fixes efficiently.

```python
# Commands for initializing Git repository, adding files, committing changes, and creating branches
git init
git add .
git commit -m "Initial commit"
git branch feature-1
git checkout feature-1

```

By following these steps, we can create a multi-modal sentiment analysis project that incorporates RNN/LSTM models on the AWS platform.
