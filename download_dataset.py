import requests
import polars as pl
from os.path import exists


DATASET_URL = "https://huggingface.co/datasets/lang-uk/recruitment-dataset-candidate-profiles-ukrainian/resolve/main/data/train-00000-of-00001.parquet?download=true"
FILENAME = "recruitment-dataset-candidate-profiles-ukrainian.parquet"
JSON_FILENAME = "documents.json"


def download_file(url, local_filename):
    with requests.get(url, stream=True) as response:
        response.raise_for_status()
        with open(local_filename, "wb") as f:
            chunk_size = 8192
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)

    print(f"File downloaded successfully: {local_filename}")


if not exists(FILENAME):
    download_file(DATASET_URL, FILENAME)

df = pl.read_parquet(FILENAME)

print(df)

df.write_json(JSON_FILENAME)

print(f"Dataframe exported to JSON file: {JSON_FILENAME}")
