import json
from bs4 import BeautifulSoup
from git import Repo
import tempfile
from pathlib import Path
from typing import Dict, List, Optional
import os
import mlrun
from datasets import Dataset, load_dataset
import zipfile


def _ignore(filename, ignored_files) -> bool:
    return filename.stem in ignored_files


def _parse_html_files(root, files_to_ignore: List[str] = []):
    # Iterating over all html files:
    pathlist = Path(root).glob("**/*.html")
    values = []
    ignored_files = []
    problematic_files = []
    text_key = "text"
    for path in pathlist:
        if _ignore(path, files_to_ignore):
            ignored_files.append(path)
            continue
        soup = BeautifulSoup(open(path, "r").read(), features="html.parser")
        txt = soup.get_text()
        txt = txt.split("\n")
        clean_content = [line.strip() for line in txt if line != "" if line.strip() != ""]
        start = -1
        end = 0
        for i, a in enumerate(clean_content):
            if "#" in a and start == -1:
                start = i
            if start != -1 and a in ["previous", "By Iguazio"]:
                end = i
                break
        if start == -1 or not end:
            problematic_files.append((path, start, end))
            continue
        encoded = [u"{raw}".format(raw=raw) for raw in clean_content[start: end]]
        print(f"Added to dataset: {path.stem}")
        encoded_string = u"\n".join(encoded)
        values.append({text_key: encoded_string})
    if ignored_files:
        print("===== Ignored files: =====")
    else:
        print("===== No ignored files =====")
    for ignored in ignored_files:
        print(ignored)
    if problematic_files:
        print("===== Problematic files: =====")
        for problematic, start, end in problematic_files:
            print(f"- filename = {problematic}\nstart = {start}, end = {end}")
    return values


def _prepare_mlrun_documentation_dataset(dataset_path, docs_source, ignored_files):
    with tempfile.TemporaryDirectory() as temp_folder:
        path_to_zip_file = os.path.join(temp_folder, "docs.zip")
        docs_source.download(path_to_zip_file)
        with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
            zip_ref.extractall(temp_folder)
        parsed_htmls = _parse_html_files(temp_folder, ignored_files)
        with open(dataset_path, "w", encoding='utf8') as f:
            for item in parsed_htmls:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")


def _edit_columns(
    dataset: Dataset,
    drop_columns: List[str] = None,
    rename_columns: Dict[str, str] = None,
):
    if drop_columns:
        dataset = dataset.remove_columns(drop_columns)
    if rename_columns:
        dataset = dataset.rename_columns(rename_columns)
    return dataset


@mlrun.handler(outputs=["train_dataset:dataset", "test_dataset:dataset"])
def prepare_dataset(
    docs_source: mlrun.DataItem,
    target_dir: str,
    drop_columns: Optional[List[str]] = None,
    rename_columns: Optional[Dict[str, str]] = None,
    ignored_files: Optional[List[str]] = None,
):
    """
    Loading the dataset and editing the columns and logs the datasets

    :param dataset_name:    The name of the dataset to get from the HuggingFace hub
    :param drop_columns:    The columns to drop from the dataset.
    :param rename_columns:  The columns to rename in the dataset.
    """
    dataset_path = os.path.join(target_dir, "mlrun_docs.jsonl")
    os.makedirs(target_dir, exist_ok=True)
    if not os.path.exists(dataset_path):
        _prepare_mlrun_documentation_dataset(dataset_path, docs_source, ignored_files)
    # Loading and editing dataset:
    dataset = load_dataset(target_dir)
    dataset = dataset["train"].train_test_split(test_size=0.2)
    small_train_dataset = dataset["train"].shuffle(seed=42)
    small_train_dataset = _edit_columns(
        small_train_dataset, drop_columns, rename_columns
    )
    small_test_dataset = dataset["test"].shuffle(seed=42)
    small_test_dataset = _edit_columns(small_test_dataset, drop_columns, rename_columns)

    return small_train_dataset.to_pandas(), small_test_dataset.to_pandas()
