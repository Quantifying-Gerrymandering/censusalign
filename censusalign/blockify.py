import os
import pandas as pd
import numpy as np
import geopandas as gpd


class Blockify:
    """
    Transforms election data from the precinct level to the block level using a conversion file
    that maps precincts to census blocks and distributes votes proportionally based on registered
    voter counts. Uses the Hamilton method to handle non-integer rounding.
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

    def rollup(self, race="gov", level: str = "blockgroup") -> pd.DataFrame:
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

        # Handle aggregation level
        if level == "block":
            return block_df

        if level == "blockgroup":
            geoid_col = "GEOID_BLOCKGROUP"
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
        return agg[[geoid_col, "tot", "dem", "rep"]].sort_values(geoid_col)

    # def map(self, save_shapefile=True, plot_fips=True):
    #     gdf = gpd.read_file(self.shapefile_path)[
    #         ["GEOID20", "geometry", "STATEFP20", "COUNTYFP20"]
    #     ]

    #     cbgdf = pd.read_csv(
    #         usecols=[
    #             "GEOID20",
    #             "Name",
    #             "T_20_CENS_ADJ_Total",
    #             "E_20_PRES_Total",
    #             "E_20_PRES_Dem",
    #             "E_20_PRES_Rep",
    #         ],
    #         dtype={"GEOID20": str},
    #     )

    #     gdf["FIPS"] = gdf["STATEFP20"] + gdf["COUNTYFP20"]
    #     gdf.drop(["STATEFP20", "COUNTYFP20"], axis=1, inplace=True)
    #     gdf["geometry"] = gdf["geometry"].buffer(0)

    #     cbgdf = cbgdf.rename(
    #         columns={
    #             "T_20_CENS_ADJ_Total": "CENS_Total",
    #             "E_20_PRES_Total": "PRES_Total",
    #             "E_20_PRES_Dem": "PRES_Dem",
    #             "E_20_PRES_Rep": "PRES_Rep",
    #         }
    #     )

    #     # Use rollup vote data
    #     blockgroup_votes = self.rollup(level="blockgroup")
    #     blockgroup_votes.rename(columns={"GEOID_BLOCKGROUP": "GEOID20"}, inplace=True)

    #     # Merge everything together
    #     gdf = pd.merge(gdf, cbgdf, on="GEOID20", how="inner")
    #     gdf = pd.merge(gdf, blockgroup_votes, on="GEOID20", how="left")

    #     if save_shapefile:
    #         os.makedirs(self.output_dir, exist_ok=True)
    #         gdf.to_file(
    #             os.path.join(self.output_dir, "cbg_pop_and_voting_2020.shp"),
    #             driver="ESRI Shapefile",
    #         )

    #     return gdf
