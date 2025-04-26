import os
import yaml
import requests
import zipfile
import tempfile
import pandas as pd
from io import BytesIO
import geopandas as gpd
import importlib.resources


class Harvest:
    """A utility class for fetching and loading various datasets (vote data, conversion data,
    census data, and shapefile data) from URLs specified in a YAML configuration file.

    This class provides methods to load data from remote sources, handle compressed
    archives, and parse the data into pandas DataFrames. It is designed to work with
    a YAML configuration file that specifies the URLs for the required datasets.

    Attributes:
        config (dict): The parsed YAML configuration containing dataset URLs.
        vote_url (str): URL for the vote data.
        conversion_url (str): URL for the conversion data.
        census_url (str): URL for the census data.
        shapefile_url (str): URL for the shapefile data.

    Methods:
        load_vote() -> pd.DataFrame:
            Fetch and load the vote data as a pandas DataFrame.

        load_conversion() -> pd.DataFrame:
            Fetch and load the conversion data as a pandas DataFrame.

        load_census() -> pd.DataFrame:
            Fetch and load the census data as a pandas DataFrame.

        load_shapefile() -> pd.DataFrame:
            Fetch and load the shapefile data as a pandas DataFrame.

    Private Methods:
        _load_yaml(path: str) -> dict:
            Load and parse a YAML configuration file.

        _load(data_url: str) -> pd.DataFrame:
            Fetch data from a URL, extract it if compressed, and load it into a pandas DataFrame.

    """

    def __init__(self, year=2022):
        """
        Initialize the Fetch object using a key from the YAML config.

        Args:
            year (int): The year for which to fetch data. Default is 2022.

        Raises:
            ValueError: If the year is not 2022.
        """

        # TODO: Add support for other years
        if year != 2022:
            raise ValueError("Only the year 2022 is supported at this time.")

        self.year = year
        self.config = self._load_config(year)

        self.vote_url = self.config["SRPREC_vote_url"]
        self.conversion_url = self.config["conversion_url"]
        self.census_url = self.config["census_url"]
        self.shapefile_url = self.config["shapefile_url"]

    def _load_config(self, year):
        yaml_file = importlib.resources.files("censusalign.config").joinpath(
            f"ca_{year}.yaml"
        )
        with yaml_file.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def fetch_and_store(self, out_dir: str):
        """
        Fetches the data from the URLs specified in the YAML config and stores them as local files.
        The files are saved in the specified output directory, which is created if it doesn't exist.

        Args:
            out_dir (str): The directory where the data files will be stored.
        """
        os.makedirs(out_dir, exist_ok=True)

        vote_df = self.load_vote()
        conversion_df = self.load_conversion()
        census_df = self.load_census()
        shapefile_gdf = self.load_shapefile()

        vote_df.to_csv(os.path.join(out_dir, f"vote_data_{self.year}.csv"), index=False)
        conversion_df.to_csv(
            os.path.join(out_dir, f"conversion_data_{self.year}.csv"),
            index=False,
        )
        census_df.to_csv(
            os.path.join(out_dir, f"census_data_{self.year}.csv"), index=False
        )
        shapefile_gdf.to_file(os.path.join(out_dir, f"shapefile_data_{self.year}.shp"))

    def load_vote(self) -> pd.DataFrame:
        """
        Load the vote data from the URL specified in the YAML config.

        Returns:
            pd.DataFrame: DataFrame containing the vote data.
        """
        return self._load(self.vote_url)

    def load_conversion(self) -> pd.DataFrame:
        """
        Load the conversion data from the URL specified in the YAML config.

        Returns:
            pd.DataFrame: DataFrame containing the conversion data.
        """
        return self._load(self.conversion_url)

    def load_census(self) -> pd.DataFrame:
        """
        Load the census data from the URL specified in the YAML config.

        Returns:
            pd.DataFrame: DataFrame containing the census data.
        """
        return self._load(self.census_url)

    def load_shapefile(self) -> gpd.GeoDataFrame:
        """
        Load the shapefile data from the URL specified in the YAML config.

        Returns:
            gpd.GeoDataFrame: GeoDataFrame containing the shapefile data.
        """
        response = requests.get(self.shapefile_url)
        response.raise_for_status()

        content_type = response.headers.get("Content-Type", "")

        if "zip" in content_type or response.content[:4] == b"PK\x03\x04":
            # It's a zip archive
            with tempfile.TemporaryDirectory() as tmpdir:
                zip_path = os.path.join(tmpdir, "shapefile.zip")
                with open(zip_path, "wb") as f:
                    f.write(response.content)

                with zipfile.ZipFile(zip_path, "r") as zip_ref:
                    zip_ref.extractall(tmpdir)
                    extracted_files = zip_ref.namelist()

                # Look for a .shp file
                shp_file = next(
                    (f for f in extracted_files if f.endswith(".shp")), None
                )
                if not shp_file:
                    raise FileNotFoundError("No .shp file found in the archive.")

                full_path = os.path.join(tmpdir, shp_file)
                return gpd.read_file(full_path)

        else:
            raise ValueError("Expected a zip archive containing shapefile components.")

    def _load_yaml(self, path: str) -> dict:
        with open(path, "r") as f:
            return yaml.safe_load(f)

    def _load(self, data_url: str) -> pd.DataFrame:
        response = requests.get(data_url)
        response.raise_for_status()

        # First, check if the response looks like a zip file
        content_type = response.headers.get("Content-Type", "")

        if "zip" in content_type or response.content[:4] == b"PK\x03\x04":
            # It's a zip file
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

        else:
            # It's a raw CSV or TXT file, not zipped
            try:
                return pd.read_csv(BytesIO(response.content))
            except pd.errors.ParserError:
                return pd.read_csv(BytesIO(response.content), delimiter="\t")
