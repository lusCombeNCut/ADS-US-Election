import requests
import csv
from datetime import datetime, timedelta
import os

# Replace with your actual TheNewsAPI token
API_TOKEN = 'ZgBtUNJ5j0fuYsC5xiVRt1eQd6xfibOAcupwJb5j'
BASE_URL = 'https://api.thenewsapi.com/v1/news/all'

# Debugging option
DEBUG = True  # Set to False when you want to run the full 2024

# Search settings
KEYWORDS = [
    "2024 Elections",
    "2024 Presidential Election",
    "Biden",
    "Biden2024",
    "conservative",
    "CPAC",
    "Donald Trump",
    "GOP",
    "Joe Biden and Kamala Harris",
    "Joe Biden",
    "Joseph Biden",
    "KAG",
    "MAGA",
    "Nikki Haley",
    "RNC",
    "Ron DeSantis",
    "Snowballing",
    "Trump2024",
    "trumpsupporters",
    "trumptrain",
    "US Elections",
    "thedemocrats",
    "DNC",
    "Kamala Harris",
    "Marianne Williamson",
    "Dean Phillips",
    "williamson2024",
    "phillips2024",
    "Democratic party",
    "Republican party",
    "Third Party",
    "Green Party",
    "Independent Party",
    "No Labels",
    "RFK Jr",
    "Robert F. Kennedy Jr.",
    "Jill Stein",
    "Cornel West",
    "ultramaga",
    "voteblue2024",
    "letsgobrandon",
    "bidenharris2024",
    "makeamericagreatagain",
    "Vivek Ramaswamy"
]


def build_search_query(keywords):
    """
    Builds a search query string for the API.
    Example: 'election | government | senate | president | congress'
    """
    return ' | '.join(keywords)

def fetch_top_headline(start_date, end_date):
    """
    Fetches the top U.S. politics headline between start_date and end_date.
    Returns a dictionary with headline information.
    """
    search_query = build_search_query(KEYWORDS)

    params = {
    'api_token': API_TOKEN,
    'language': 'en',
    'categories': 'politics',
    'search': search_query,
    'search_fields': 'title,description,main_text',  # <--- updated here
    'published_after': start_date.strftime('%Y-%m-%d'),
    'published_before': end_date.strftime('%Y-%m-%d'),
    'sort': 'relevance_score',
    'limit': 1,}


    try:
        response = requests.get(BASE_URL, params=params)
        response.raise_for_status()
        data = response.json()
        articles = data.get('data', [])
        if articles:
            article = articles[0]
            title = article.get('title', 'No Title')
            source = article.get('source', 'Unknown Source')
            published_at = article.get('published_at', '')
            url = article.get('url', '')
            description = article.get('description', '')
            
            print(f"{start_date.date()} - {title} ({source})")
            
            return {
                'week_start': start_date.strftime('%Y-%m-%d'),
                'week_end': end_date.strftime('%Y-%m-%d'),
                'title': title,
                'source': source,
                'published_at': published_at,
                'url': url,
                'description': description
            }
        else:
            print(f"{start_date.date()} - No articles found.")
            return {
                'week_start': start_date.strftime('%Y-%m-%d'),
                'week_end': end_date.strftime('%Y-%m-%d'),
                'title': 'No articles found',
                'source': '',
                'published_at': '',
                'url': '',
                'description': ''
            }
    except requests.exceptions.RequestException as e:
        print(f"Error fetching headline for {start_date.date()} to {end_date.date()}: {e}")
        return {
            'week_start': start_date.strftime('%Y-%m-%d'),
            'week_end': end_date.strftime('%Y-%m-%d'),
            'title': f'Error: {str(e)}',
            'source': '',
            'published_at': '',
            'url': '',
            'description': ''
        }

def main():
    """
    Fetches the top headline for each week of 2024 and saves to a CSV file.
    """
    headlines = []
    
    if DEBUG:
        # Only fetch one week's headline for debugging
        current_date = datetime(2024, 6, 14)
        week_end = current_date + timedelta(days=6)
        headline_data = fetch_top_headline(current_date, week_end)
        headlines.append(headline_data)
    else:
        # Fetch for all weeks of 2024
        current_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 12, 31)

        while current_date <= end_date:
            week_end = current_date + timedelta(days=6)
            if week_end > end_date:
                week_end = end_date

            headline_data = fetch_top_headline(current_date, week_end)
            headlines.append(headline_data)

            current_date += timedelta(weeks=1)
    
    # Different CSV filenames depending on mode
    if DEBUG:
        csv_filename = 'weekly_headlines_debug.csv'
    else:
        csv_filename = 'weekly_headlines_full.csv'

    fieldnames = ['week_start', 'week_end', 'title', 'source', 'published_at', 'url', 'description']
    
    with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(headlines)
    
    print(f"\nHeadlines saved to {os.path.abspath(csv_filename)}")

if __name__ == "__main__":
    main()
