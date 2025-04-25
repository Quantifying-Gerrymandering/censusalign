import os
import yaml
import requests
import zipfile
import tempfile
import pandas as pd


class Fetch:
    """
    Fetch and load SWDB census data and election metadata from a YAML config.
    """

    def __init__(self, key: str, config_path: str = "./config/swdb.yaml"):
        """
        Initialize the Fetch object using a key from the YAML config.

        Args:
            key (str): Key to retrieve the URL and metadata (e.g., "g22_state_sov")
            config_path (str): Path to the YAML config file.
        """
        self.config = self._load_yaml(config_path)
        self.meta = self.config[key]
        self.url = self.meta["url"]
        self.elections = self.meta.get("election", {})
        self.df = self._download_and_load()

    def _load_yaml(self, path: str) -> dict:
        with open(path, "r") as f:
            return yaml.safe_load(f)

    def _download_and_load(self) -> pd.DataFrame:
        response = requests.get(self.url)
        response.raise_for_status()

        with tempfile.TemporaryDirectory() as tmpdir:
            zip_path = os.path.join(tmpdir, "data.zip")
            with open(zip_path, "wb") as f:
                f.write(response.content)

            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(tmpdir)
                extracted_files = zip_ref.namelist()

            data_file = next(
                (f for f in extracted_files if f.endswith((".csv", ".txt"))), None
            )
            if not data_file:
                raise FileNotFoundError("No CSV or TXT file found in the archive.")

            full_path = os.path.join(tmpdir, data_file)

            try:
                return pd.read_csv(full_path)
            except pd.errors.ParserError:
                return pd.read_csv(full_path, delimiter="\t")
