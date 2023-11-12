# %%

import esda
import fiona
import geopandas as gpd
import libpysal as lps
import mapclassify as mc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sbn
from matplotlib import colors
from shapely.geometry import Point

plt.ion

# %% [markdown]
# # Costa Rica Housing Deficit Spatial autocorrelation and Moran's I analysis

# %%
# Listing gpkg layers
gpkg = "./data/censo_2011.gpkg"
layers = fiona.listlayers(gpkg)
print(layers)

# Importing the geopackage data
gdf = gpd.read_file(
    "./data/censo_2011.gpkg", layer="deficit_habitacional_distritos_2011"
)

print(gdf.info())

gdf

# %% [markdown]
# ## Data wrangling
# Since 2011 Costa Rica districts have changed, so we expect to have some missing
# values. First we need to check how many rows have missing data.

# %%
column = "df_déficit_habitacional_(dh)"
na_sum = pd.isna(gdf[column]).sum()
print(f"We have {na_sum} empty rows in {column} column")

# %% [markdown]
# ### Missing data handling
# We have several ways to deal with missing data, we can:
# - Drop the row with missing values.
# - Fill missing values.
#   - By adding the average of the canton.
#   - By adding a arbitrary value.
#
# Dropping the row have a mayor problem, since we are using the 2023 districts, in a
# **multipart** geometry the dropping of the entire row result in the deletion of the
# entire geometry. Because of this we are going to choose filling the missign values
# with the avegare of the canton, this have its own problems, but for now its not a
# concern.
# %%
# Get the average of the groupby canton
gdf.groupby("canton")["df_déficit_habitacional_(dh)"].transform("mean").round()
# %%
# fill the missing values with the avg of the canton
gdf["df_déficit_habitacional_(dh)"].fillna(
    (gdf.groupby("canton")["df_déficit_habitacional_(dh)"].transform("mean").round()),
    inplace=True,
)

gdf

# %%
na_sum = pd.isna(gdf[column]).sum()
print(f"Now we end up with {na_sum} missing values")
# %% [markdown]
# ### Plot the deficit

# %%
gdf.plot(column="df_déficit_habitacional_(dh)")

# %%
fig, ax = plt.subplots(figsize=(20, 20), subplot_kw={"aspect": "equal"})

gdf.plot(
    column="df_déficit_habitacional_(dh)",
    scheme="Quantiles",
    k=5,
    cmap="GnBu",
    legend=True,
    ax=ax,
)

# %% [markdown]
# ## Spatial autocorrelation
# The concept of spatial autocorrelation relates to the combination of two types of
# similarity: spatial similarity and attribute similarity. Although there are many
# different measures of spatial autocorrelation, they all combine these two types of
# simmilarity into a summary measure.
#
# ## Spatial Similarity

# %%
wq = lps.weights.Queen.from_dataframe(gdf, use_index=True)
wq.transform = "r"  # type: ignore

# %% [markdown]
# ## Attribute Similarity

y = gdf["df_déficit_habitacional_(dh)"]
ylag = lps.weights.lag_spatial(wq, y)
ylag

# %%
ylagq5 = mc.Quantiles(ylag, k=5)
ylagq5

# %%
f, ax = plt.subplots(1, figsize=(20, 20))
gdf.assign(cl=ylagq5.yb).plot(
    column="cl",
    categorical=True,
    k=5,
    cmap="GnBu",
    linewidth=0.1,
    ax=ax,
    edgecolor="black",
    legend=True,
)
ax.set_axis_off()
plt.title("Spatial Lag Median Price (Quintiles)")

plt.show()
