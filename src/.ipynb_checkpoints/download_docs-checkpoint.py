from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.error import HTTPError
from urllib.request import urlopen
import os
import requests
from bs4 import BeautifulSoup
import mlrun


def get_all_links(base_url, parent_url):
    download_urls = []
    link_to_parent = False
    page = requests.get(base_url).text  # get the raw HTML of the page
    soup = BeautifulSoup(page, features="html.parser")  # make our page easy to navigate
    for a in soup.find_all('a', href=True):  # iterate through every <a> tag on the page
        href = a['href']  # get the href attribute of the tag
        if "#" in href:
            href = href.split("#")[0]
        if "://" not in href and href.endswith(".html"):
            if href.startswith("./"):
                href = href[2:]
            if href.startswith("../"):  # in case the ref is to the parent path
              link_to_parent = True
              link = parent_url + href[3:]
            else:
                link = base_url + href
            if link not in download_urls:
                download_urls.append(link) # add the link to our array
    return download_urls, link_to_parent


def download_link(url, target_dir, base_url):
    try:
        with urlopen(url) as connection:
            # read the contents of the url as bytes and return it
            data = connection.read()
    except HTTPError:
        return
    filename = url[len(base_url):].replace("/", "_")
    path = os.path.join(target_dir, filename)
    
    # TBD - may want to convert html to text using bs4 here
    
    with open(path, 'wb') as file:
        # write all provided data to the file
        file.write(data)
    return path


@mlrun.handler(outputs=["docs_dir:directory"])
def download_all_files(url, target_dir):
    """download all files on the provided webpage to the provided path"""
    parent_url = url.rsplit("/", 2)[0] + "/"
    download_urls, link_to_parent = get_all_links(url, parent_url)
    # download_urls.sort()
    root_url = parent_url if link_to_parent else url
    os.makedirs(target_dir, exist_ok=True)
    print(f'Found {len(download_urls)} links in {url}')

    # create the pool of worker threads
    with ThreadPoolExecutor(max_workers=20) as exe:
        # dispatch all download tasks to worker threads
        futures = [exe.submit(download_link, link, target_dir, root_url) for link in download_urls]
        # report results as they become available
        for future in as_completed(futures):
            # retrieve result
            path = future.result()
            if path:
                print("wrote file:", path)
    return target_dir
    