import sys
import yaml
import numpy as np
import pandas as pd
import networkx as nx
import geopandas as gpd
from .harvest import Harvest
import importlib.resources
from gerrychain import Graph


class Cultivate:
    """
    The `Cultivate` class is designed to process, clean, and transform election, population,
    and geographic data into a graph format. It provides methods for aggregating election
    data to various geographic levels, merging population data with geographic shapes, and
    creating graph representations of the data.

    Attributes:
        election_df (pd.DataFrame): DataFrame containing election data with columns for
            precinct keys and vote counts for Democratic and Republican candidates.
        conversion_df (pd.DataFrame): DataFrame containing conversion data mapping precinct
            keys to census block keys, along with registration data.

    Methods:
        __init__(election_file: str, conversion_file: str):
            Initializes the `Cultivate` object by loading and preprocessing election and
            conversion data from the specified CSV files.

        hamilton_floor(values: pd.Series) -> pd.Series:
            Applies Hamilton rounding to a series of values, ensuring that the sum of the
            rounded values matches the sum of the original values.

        blockify(level: str = "blockgroup") -> pd.DataFrame:
            Aggregates precinct-level election data to a specified census geography level
            (e.g., block, blockgroup, tract, or county) and returns the aggregated data.

        merge_population_and_geometry(shape_path: str, blockfile_path: str) -> gpd.GeoDataFrame:
            Merges population data from a block-level CSV file with geographic shapes from
            a shapefile, returning a GeoDataFrame with population and geometry data.

        graphify(shape_path: str, blockfile_path: str, level: str = "blockgroup", custom_edges: list) -> nx.Graph:
            Creates a graph representation of the election data aggregated to the specified
            geographic level, optionally adding custom edges between specified geographic units.

        _add_edge_by_geoid(graph: nx.Graph, gdf: gpd.GeoDataFrame, geoid1: str, geoid2: str):
            Adds an edge to the graph between two geographic units identified by their GEOID20
            values, if both units are present in the GeoDataFrame.
    """

    def __init__(
        self,
        year: str = "2022",
        election: str = "governor",
    ):
        """
        Initialize Cultivate.

        Args:
            year (str, optional): Year to fetch if files are not provided. Defaults to "2022".
            election:
        """

        self.config = self._load_config(year)
        harvester = Harvest(year=year)
        self._print_status(f"Loading election data")
        self.election_df = harvester.load_vote()
        self._clear_status()
        self._print_status(
            f"Loading conversion data",
        )
        self.conversion_df = harvester.load_conversion()
        self._clear_status()
        self._print_status(f"Loading census data")
        self.census_df = harvester.load_census()
        self._clear_status()
        self._print_status(f"Loading shapefile data")
        self.shapefile_df = harvester.load_shapefile()
        self._clear_status()

        self.year = year

        # ➔ Pull election columns from config
        election_columns = self.config["election"][election]

        # Always include SRPREC_KEY
        subset_columns = ["SRPREC_KEY"] + election_columns

        # Process election data
        self.election_df = self.election_df[subset_columns].copy()
        self.election_df.rename(
            columns={
                election_columns[0]: "dem",
                election_columns[1]: "rep",
            },
            inplace=True,
        )
        self.election_df[["dem", "rep"]] = self.election_df[["dem", "rep"]].astype(int)
        self._print_status("All data loaded successfully!")

    def _load_config(self, year):
        yaml_file = importlib.resources.files("censusalign.config").joinpath(
            f"ca_{year}.yaml"
        )
        with yaml_file.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    @staticmethod
    def hamilton_floor(values: pd.Series) -> pd.Series:
        if values.isna().all():
            return pd.Series([0] * len(values), index=values.index)
        floored = np.floor(values).astype(int)
        remainder = values - floored
        n_remaining = int(round(remainder.sum()))
        if n_remaining > 0:
            top_indices = remainder.nlargest(n_remaining).index
            floored.loc[top_indices] += 1
        return floored

    def blockify(self, level: str = "blockgroup") -> pd.DataFrame:
        """
        Aggregates precinct-level election data to specified census geography.

        Args:
            race (str): Type of election data to process.

            level (str): Aggregation level.
                One of 'block', 'blockgroup', 'tract', or 'county'.

        Returns:
            pd.DataFrame: Aggregated DataFrame with columns [GEOID_<LEVEL>, tot, dem, rep].
        """
        # Merge election and conversion data
        merged = self.conversion_df.merge(self.election_df, on="SRPREC_KEY", how="left")

        # Drop rows with missing or zero registration data
        merged = merged.dropna(subset=["BLKREG", "SRTOTREG", "dem", "rep"])
        merged = merged[merged["SRTOTREG"] > 0]

        # Compute raw proportional allocation
        merged["dem_raw"] = merged["dem"] * merged["BLKREG"] / merged["SRTOTREG"]
        merged["rep_raw"] = merged["rep"] * merged["BLKREG"] / merged["SRTOTREG"]

        # Replace any infinite results and drop remaining NaNs
        merged.replace([np.inf, -np.inf], np.nan, inplace=True)
        merged.dropna(subset=["dem_raw", "rep_raw"], inplace=True)

        # Apply Hamilton rounding within each precinct
        merged["dem"] = merged.groupby("SRPREC_KEY")["dem_raw"].transform(
            self.hamilton_floor
        )
        merged["rep"] = merged.groupby("SRPREC_KEY")["rep_raw"].transform(
            self.hamilton_floor
        )

        # Aggregate to block level
        block_votes = merged.groupby("BLOCK_KEY")[["dem", "rep"]].sum().reset_index()
        block_votes["tot"] = block_votes["dem"] + block_votes["rep"]

        # Remove water-only blocks (block groups ending in '0')
        block_votes["BLOCK_KEY"] = block_votes["BLOCK_KEY"].astype(str)
        block_votes["blockgroup"] = block_votes["BLOCK_KEY"].str[:11]
        block_df = block_votes[block_votes["blockgroup"].str[-1] != "0"].copy()
        block_df = block_df[["BLOCK_KEY", "tot", "dem", "rep"]].sort_values("BLOCK_KEY")

        geoid_col = f"GEOID_{level}"
        # Handle aggregation level
        if level == "blockgroup":
            block_df[geoid_col] = block_df["BLOCK_KEY"].str[:11]
        else:
            raise ValueError(
                "Invalid level. Only 'blockgroup', supported at this time."
            )

        # Aggregate and return
        agg = block_df.groupby(geoid_col)[["dem", "rep"]].sum().reset_index()
        agg["tot"] = agg["dem"] + agg["rep"]

        # Rename columns
        agg.rename(
            columns={"tot": "total_vote", "dem": "dem_vote", "rep": "rep_vote"},
            inplace=True,
        )

        return agg[[geoid_col, "total_vote", "dem_vote", "rep_vote"]].sort_values(
            geoid_col
        )

    def merge_population_and_geometry(self) -> gpd.GeoDataFrame:
        """
        Merges population data from a block-level CSV file with geographic shapes from a shapefile.

        Returns:
            gpd.GeoDataFrame: A GeoDataFrame containing merged population and geometry data,
                              with columns ['GEOID20', 'geometry', 'FIPS', 'pop_total'].
        """
        gdf = self.shapefile_df.copy()
        block_df = self.census_df.copy()

        gdf["geometry"] = gdf["geometry"].buffer(0)
        gdf["FIPS"] = gdf["STATEFP20"] + gdf["COUNTYFP20"]
        gdf.drop(["STATEFP20", "COUNTYFP20"], axis=1, inplace=True)

        block_df["BLOCK20"] = block_df["BLOCK20"].astype(str)
        block_df["GEOID20"] = block_df["BLOCK20"].str[:11]
        blockgroup_data = block_df.groupby("GEOID20").sum(numeric_only=True)
        blockgroup_data = blockgroup_data.reset_index()

        blockgroup_data["GEOID20"] = "0" + blockgroup_data["GEOID20"]
        blockgroup_data = blockgroup_data.rename(columns={"CIT_22": "pop_total"})[
            ["GEOID20", "pop_total"]
        ]

        gdf = pd.merge(gdf, blockgroup_data, on="GEOID20", how="inner")
        gdf = gdf.to_crs(3310)
        return gdf.sort_values(by="GEOID20").reset_index(drop=True)

    def graphify(
        self,
        level: str = "blockgroup",
        custom_edges: list = [
            # Named connections
            ("060759804011", "060750604002"),  # Farallon Islands ↔ Port
            ("060750179032", "060750101011"),  # Alcatraz ↔ Pier 39
            ("060750615072", "060750179031"),  # Example connection
            # Channel islands
            ("060839801001", "061110025003"),  # Channel 0 ↔ Ventura Harbor
            ("061110036181", "061110036172"),  # Channel 1 ↔ Shared port
            ("061119800001", "061110036172"),  # Channel 2 ↔ Shared port
            ("060375991001", "060375990001"),  # Channel 3 ↔ 5
            ("060375991001", "060375991002"),  # Channel 3 ↔ 4
            ("060375990001", "060375760011"),  # Channel 5 ↔ Port 1
            ("060375990001", "060379800311"),  # Channel 5 ↔ Port 2
            ("060375990001", "061110025003"),  # Channel 5 ↔ Ventura Harbor
            # More manual connections
            ("060730050001", "060730110001"),
            ("060014272005", "060014060001"),
            ("060730101091", "060730102011"),
            ("060590995143", "060590995141"),
            ("060590995145", "060590995141"),
            ("060375775011", "060375776041"),
            ("060590630051", "060590630061"),
            ("060590635001", "060590629001"),
        ],
    ) -> nx.Graph:
        """
        Creates a GeoDataFrame with election data aggregated to the specified level.

        Args:
            level (str): Aggregation level. One of 'block', 'blockgroup', 'tract', or 'county'.

        Returns:
            An nx.Graph object representing the election data at the specified level.
        """
        self._print_status(f"Blockifying election data")
        vote_by_block_df = self.blockify(level="blockgroup")
        geoid_col = f"GEOID_{level}"
        vote_by_block_df.rename(columns={geoid_col: "GEOID20"}, inplace=True)
        vote_by_block_df["GEOID20"] = "0" + vote_by_block_df["GEOID20"]
        self._clear_status()
        self._print_status(f"Merging population and geometry data")
        gdf = self.merge_population_and_geometry()
        gdf = pd.merge(
            gdf,
            vote_by_block_df,
            on="GEOID20",
            how="inner",
        )
        self._clear_status()

        graph = Graph.from_geodataframe(gdf, ignore_errors=False)
        for geoid1, geoid2 in custom_edges:
            self._add_edge_by_geoid(graph, gdf, geoid1, geoid2)

        return graph

    def _add_edge_by_geoid(self, graph, gdf, geoid1, geoid2, warnings=False):
        """Add edge to the graph using GEOID20 values."""
        try:
            index1 = gdf.index[gdf["GEOID20"] == geoid1].tolist()[0]
            index2 = gdf.index[gdf["GEOID20"] == geoid2].tolist()[0]
            graph.add_edge(index1, index2)
        except IndexError:
            if warnings:
                print(
                    f"Warning: One of the GEOIDs {geoid1}, {geoid2} not found in gdf."
                )

    def _print_status(self, message: str):
        """
        Print a status message that stays on the same line.
        """
        sys.stdout.write(f"\r{message}")
        sys.stdout.flush()

    def _clear_status(self):
        """
        Clear the current line from the terminal.
        """
        sys.stdout.write("\r" + " " * 100 + "\r")
        sys.stdout.flush()
