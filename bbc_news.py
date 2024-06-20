import requests
from bs4 import BeautifulSoup
import pandas as pd

# URL of the page to scrape
url = 'https://www.bbc.com/news'

# Function to get the BeautifulSoup object of a webpage
def get_soup(url):
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        print(f"Successfully fetched the page: {url}")
    else:
        print(f"Failed to fetch the page: {url}, status code: {response.status_code}")
    return BeautifulSoup(response.content, "html.parser")

# Function to scrape the latest news titles
def scrape_latest_news(url):
    soup = get_soup(url)
    news_data = []

    # Find all the news headlines
    for item in soup.find_all('h2', {'data-testid': 'card-headline'}):
        title = item.get_text().strip()
        if title:
            print(f"Found title: {title}")
            news_data.append({'title': title})

    if not news_data:
        print("No news titles found.")
    return news_data

# Main function
if __name__ == "__main__":
    news_data = scrape_latest_news(url)

    # Save to CSV
    if news_data:
        df = pd.DataFrame(news_data)
        df.to_csv('latest_news.csv', index=False)
        print('Saved latest news titles to latest_news.csv')
    else:
        print('No data to save to CSV.')