import os
import numpy as np
import pandas as pd
import networkx as nx
import geopandas as gpd
from gerrychain import Graph


class Graphify:
    """
    Transforms, cleans, combines, and aggregates shapefile, elections, and population data and converts to a graph format.
    """

    def __init__(self, election_file: str, conversion_file: str):
        if not os.path.exists(election_file):
            raise FileNotFoundError(f"Election file not found: {election_file}")
        if not os.path.exists(conversion_file):
            raise FileNotFoundError(f"Conversion file not found: {conversion_file}")

        self.election_df = pd.read_csv(election_file, dtype={"SRPREC_KEY": str})
        self.election_df = self.election_df[
            ["SRPREC_KEY", "GOVDEM01", "GOVREP01"]
        ].copy()
        self.election_df.rename(
            columns={"GOVDEM01": "dem", "GOVREP01": "rep"}, inplace=True
        )
        self.election_df[["dem", "rep"]] = self.election_df[["dem", "rep"]].astype(int)

        self.conversion_df = pd.read_csv(
            conversion_file, dtype={"SRPREC_KEY": str, "BLOCK_KEY": str}
        )
        self.conversion_df = self.conversion_df[
            self.conversion_df["SRPREC_KEY"].notna()
            & self.conversion_df["BLOCK_KEY"].notna()
        ].copy()
        self.conversion_df[["BLKREG", "SRTOTREG"]] = self.conversion_df[
            ["BLKREG", "SRTOTREG"]
        ].astype(float)

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
        block_votes["blockgroup"] = block_votes["BLOCK_KEY"].str[:12]
        block_df = block_votes[block_votes["blockgroup"].str[-1] != "0"].copy()
        block_df = block_df[["BLOCK_KEY", "tot", "dem", "rep"]].sort_values("BLOCK_KEY")

        geoid_col = f"GEOID_{level}"
        # Handle aggregation level
        if level == "block":
            block_df[geoid_col] = block_df["BLOCK_KEY"]
            return block_df
        if level == "blockgroup":
            block_df[geoid_col] = block_df["BLOCK_KEY"].str[:12]
        elif level == "tract":
            geoid_col = "GEOID_TRACT"
            block_df[geoid_col] = block_df["BLOCK_KEY"].str[:11]
        elif level == "county":
            geoid_col = "GEOID_COUNTY"
            block_df[geoid_col] = block_df["BLOCK_KEY"].str[:5]
        else:
            raise ValueError(
                "Invalid level. Choose from 'block', 'blockgroup', 'tract', or 'county'."
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

    def merge_population_and_geometry(
        self,
        shape_path: str,
        blockfile_path: str,
    ) -> gpd.GeoDataFrame:
        """
        Merges population data from a block-level CSV file with geographic shapes from a shapefile.

        Args:
            shape_path (str): Path to the shapefile containing geographic shapes.
            blockfile_path (str): Path to the CSV file containing block-level population data.

        Returns:
            gpd.GeoDataFrame: A GeoDataFrame containing merged population and geometry data,
                              with columns ['GEOID20', 'geometry', 'FIPS', 'pop_total'].
        """
        block_df = pd.read_csv(blockfile_path)
        gdf = gpd.read_file(shape_path)[
            ["GEOID20", "geometry", "STATEFP20", "COUNTYFP20"]
        ]
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

    def create(
        self,
        shape_path,
        blockfile_path,
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
            # District connection fixes
            ("060839801001", "061110025003"),
            ("060375991001", "060375990001"),
            ("060990008061", "060952506042"),  # District 4
            ("060990009082", "060710103002"),  # District 7
            ("060770021001", "060750132002"),  # District 9
            ("060759804011", "060750171011"),  # District 22
            ("060990005063", "060050003011"),  # District 25
            ("060770051332", "060372371012"),  # District 27
            ("060990019002", "060650408092"),  # District 39
            ("060990004042", "060190041002"),  # District 43
            ("060770022012", "060530118022"),  # District 43 (additional)
            ("060990010022", "060371434013"),  # District 44
        ],
    ) -> nx.Graph:
        """
        Creates a GeoDataFrame with election data aggregated to the specified level.

        Args:
            level (str): Aggregation level. One of 'block', 'blockgroup', 'tract', or 'county'.

        Returns:
            An nx.Graph object representing the election data at the specified level.
        """
        block_df = self.blockify(level)
        geoid_col = f"GEOID_{level}"
        block_df.rename(columns={geoid_col: "GEOID20"}, inplace=True)

        gdf = self.merge_population_and_geometry(shape_path, blockfile_path)

        gdf = pd.merge(
            gdf,
            block_df,
            on="GEOID20",
            how="inner",
        )

        graph = Graph.from_geodataframe(gdf, ignore_errors=False)
        for geoid1, geoid2 in custom_edges:
            self._add_edge_by_geoid(graph, gdf, geoid1, geoid2)

        return graph

    def _add_edge_by_geoid(self, graph, gdf, geoid1, geoid2):
        """Add edge to the graph using GEOID20 values."""
        try:
            index1 = gdf.index[gdf["GEOID20"] == geoid1].tolist()[0]
            index2 = gdf.index[gdf["GEOID20"] == geoid2].tolist()[0]
            graph.add_edge(index1, index2)
        except IndexError:
            print(f"Warning: One of the GEOIDs {geoid1}, {geoid2} not found in gdf.")
