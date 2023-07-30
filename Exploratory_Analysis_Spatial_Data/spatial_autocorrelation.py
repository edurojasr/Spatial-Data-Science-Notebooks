# %%
from typing import Any

import esda
import geopandas as gpd
import libpysal as lps
import mapclassify as mc
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

# %% [markdown]
# ## Spatial autocorrelation
# The concept of spatial autocorrelation relates to the combination of two types of
# similarity: spatial similarity and attribute similarity. Although there are many
# different measures of spatial autocorrelation, they all combine these two types of
# simmilarity into a summary measure.

# ### Spatial similarity
# Using spatial weights
# %%
df: gpd.GeoDataFrame = gdf
wq: Any = lps.weights.Queen.from_dataframe(df=df)

wq.transform = "r"

# %% [markdown]
# ## Attribute Similarity
# measure of attribute similarity to pair up with this concept of spatial similarity.
# The spatial lag
# %%
y = df["median_pri"]
ylag = lps.weights.lag_spatial(w=wq, y=y)
print(ylag)

# %%
# make quantiles
ylagq5 = mc.Quantiles(y=ylag, k=5)

ylagq5
# %%
f, ax = plt.subplots(1, figsize=(9, 9))
df.assign(cl=ylagq5.yb).plot(
    column="cl",
    categorical=True,
    k=5,
    cmap="GnBu",
    linewidth=0.1,
    ax=ax,
    edgecolor="white",
    legend=True,
)
ax.set_axis_off()
plt.title("Spatial Lag Median Price (Quintiles)")

plt.show()
