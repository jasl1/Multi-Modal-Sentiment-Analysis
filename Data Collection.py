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
