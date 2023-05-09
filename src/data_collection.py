import os
import re
from pathlib import Path
from urllib.request import urlopen, Request

from bs4 import BeautifulSoup, Tag
import mlrun

ARTICLE_TOKEN = "Article: "
HEADER_TOKEN = "Subject: "


def normalize(s):
    return s.replace("\n", "").replace("\t", "")


def mark_header_tags(soup):
    nodes = soup.find_all(re.compile("^h[1-6]$"))
    # Tagging headers in html to identify in text files:
    if nodes:
        content_type = type(nodes[0].contents[0])
        nodes[0].string = content_type(ARTICLE_TOKEN + normalize(str(nodes[0].contents[0])))
        for node in nodes[1:]:
            if node.string:
                content_type = type(node.contents[0])
                if content_type == Tag:
                    node.string = HEADER_TOKEN + normalize(node.string)
                else:
                    node.string = content_type(HEADER_TOKEN + str(node.contents[0]))


def get_html_as_string(url, mark_headers):
    # read html source:
    req = Request(
        url=url,
        headers={'User-Agent': 'Mozilla/5.0'}
    )
    web_html_content = urlopen(req).read().decode("utf-8")
    soup = BeautifulSoup(web_html_content, features="html.parser")
    if mark_headers:
        mark_header_tags(soup)
    return soup.get_text()


@mlrun.handler(outputs=["html-as-text-files:directory"])
def collect_html_to_text_files(urls, mark_headers=True):
    directory = "html_as_text_files"
    os.makedirs(directory, exist_ok=True)
    # Writing html files as text files:
    for url in urls:
        page_name = Path(url).name
        with open(f"{directory}/{page_name}.txt", "w") as f:
            f.write(get_html_as_string(url, mark_headers))
    return directory
