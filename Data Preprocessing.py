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
