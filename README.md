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

### Model Development: 
Design and train an RNN/LSTM model for sentiment analysis using TensorFlow or PyTorch. This model will learn to predict whether a movie review is positive or negative based on the text data.

### Model Evaluation: 
Evaluate the performance of the trained model using standard metrics like accuracy, precision, recall, and F1-score. Cross-validation ensures that the model's performance is consistent across different subsets of the data.

### Deployment on AWS: 
Deploy the trained model on AWS Lambda for serverless inference. Expose an endpoint for making predictions, allowing users to input movie reviews and receive sentiment predictions in real-time.

### Version Control with Git: 
Use Git for version control to track changes to the codebase, collaborate with team members, and manage new features or bug fixes efficiently.

By following these steps, we can create a multi-modal sentiment analysis project that incorporates RNN/LSTM models on the AWS platform.
