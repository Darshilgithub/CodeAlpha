import requests
from bs4 import BeautifulSoup
import csv
import time

base_url = "https://quotes.toscrape.com/page/{}/"

with open("quotes_datasets.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Quote", "Author", "Tags"])  # header row
    
    for page in range(1, 11):
        print(f"Scraping page {page}...")
        response = requests.get(base_url.format(page))
        
        if response.status_code != 200:
            print("No more pages or error occurred.")
            break
        
        soup = BeautifulSoup(response.text, "html.parser")
        quotes = soup.find_all("div", class_="quote")
        
        for q in quotes:
            text = q.find("span", class_="text").get_text()
            author=q.find("small",class_="author").get_text()
            tags = [tag.get_text() for tag in q.find_all("a", class_="tag")]
            
            writer.writerow([text, author, ", ".join(tags)])
            
        time.sleep(1)  # polite delay
