import os
import requests
import backoff
import bs4
from bs4 import BeautifulSoup
from bs4.element import Comment
from pprint import pprint


CX = os.getenv('GOOGLE_API_CX')
API_KEY = os.getenv('GOOGLE_API_KEY')
 
API_URL = 'https://www.googleapis.com/customsearch/v1?key={api_key}&cx={cx}&q={{query}}'.format(api_key=API_KEY, cx=CX)


def tag_visible(element):
    if element.parent.name in ['style', 'script', 'head', 'title', 'meta', '[document]']:
        return False
    if isinstance(element, Comment):
        return False
    return True


def text_from_html(body):
    soup = BeautifulSoup(body, 'html.parser')
    texts = soup.findAll(text=True)
    visible_texts = filter(tag_visible, texts)  
    return u" ".join(t.strip() for t in visible_texts)


@backoff.on_exception(backoff.expo,
                    (requests.exceptions.RequestException,
                     bs4.builder.ParserRejectedMarkup),
                    max_time=15,
                    raise_on_giveup=False)
def link_to_text(url):
    resp = requests.get(url)
    return text_from_html(resp.text)


class GoogleSearcher:
    def __init__(self):
        pass

    def batch_search(self, queries, *args, **kwargs):
        results = [self.search(query) for query in queries]
        return results

    @backoff.on_exception(backoff.expo,
                      requests.exceptions.RequestException,
                      max_time=15,
                      raise_on_giveup=False)
    def search(self, query, *args, **kwargs):
        # query = requests.utils.quote(query, safe='')
        resp = requests.get(API_URL.format(query=query))
        data = resp.json()
        if 'items' in data:
            return data['items']
        return None
