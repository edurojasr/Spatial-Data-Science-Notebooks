# %%
from typing import Any

import esda
import geopandas as gpd
import libpysal as lps
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely.geometry import Point

# %% [markdown]
# This is a followup of the tutorial form PySAL [Exploratory Analysis of Spatial Data:
# Spatial Autocorrelation](https://pysal.org/esda/notebooks/spatialautocorrelation.html)
# Our data set comes from the Berlin airbnb scrape taken in April 2018. This dataframe
# was constructed as part of the GeoPython 2018 workshop by Levi Wolf and Serge Rey. As
# part of the workshop a geopandas data frame was constructed with one of the columns
# reporting the median listing price of units in each neighborhood in Berlin:

# %% [markdown]
# ## Data preparation

# %%
gdf: gpd.GeoDataFrame = gpd.read_file("data/berlin-neighbourhoods.geojson")
gdf.info()
# %%
bl_df: pd.DataFrame = pd.read_csv("data/berlin-listings.csv")

# Create a geometry object from the lat and log in the csv file
geometry: list[Point] = [Point(xy) for xy in zip(bl_df.longitude, bl_df.latitude)]
crs: str = "EPSG:4326"
bl_gdf = gpd.GeoDataFrame(bl_df, crs=crs, geometry=geometry)  # type: ignore

# %%
# Check the data
# bl_gdf.info()
# bl_gdf.plot()

# %%
# Cast as new type
# bl_gdf['price'] = bl_gdf['price'].astype('float32')

# %% [markdown]
# ## Spatial join

# %%
sj_gdf: gpd.GeoDataFrame = gpd.sjoin(
    gdf, bl_gdf, how="inner", predicate="intersects", lsuffix="left", rsuffix="rigth"
)
sj_gdf.info()

# %% [markdown]
# Calculate the mean price grouping by the neighbourhood_group
# %%
mean_price_gb: Any = sj_gdf["price"].groupby([sj_gdf["neighbourhood_group"]]).mean()  # type: ignore
mean_price_gb

gdf = gdf.join(mean_price_gb, on="neighbourhood_group")  # type: ignore

gdf.rename(columns={"price": "median_pri"}, inplace=True)
gdf.head(15)

# %% [markdown]
# We need to deal with the null values
# %%
pd.isnull(gdf["median_pri"]).sum()  # type: ignore

# %%
gdf["median_pri"].fillna((gdf["median_pri"].mean()), inplace=True)  # type: ignore

gdf.plot(column="median_pri")

# %%
fig, ax = plt.subplots(figsize=(12, 10), subplot_kw={"aspect": "equal"})
gdf.plot(column="median_pri", scheme="Quantiles", k=5, cmap="GnBu", legend=True, ax=ax)
