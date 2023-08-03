# %%
from typing import Any

import esda
import geopandas as gpd
import libpysal as lps
import mapclassify as mc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sbn
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
y: pd.Series = df["median_pri"]  # type: ignore
ylag = lps.weights.lag_spatial(w=wq, y=y)
print("This is the spatial lag")
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

# %% [markdown]
# The quintile map for the spatial lag tends to enhance the impression of value
# similarity in space. It is, in effect, a local smoother.
# %%
df["lag_median_pri"] = ylag

f, ax = plt.subplots(nrows=1, ncols=2, figsize=(2.16 * 4, 4))
df.plot(
    column="median_pri",
    ax=ax[0],
    edgecolor="k",
    scheme="quantiles",
    k=5,
    cmap="GnBu",
)
ax[0].axis(df.total_bounds[np.asarray([0, 2, 1, 3])])
ax[0].set_title("Price")

df.plot(
    column="lag_median_pri",
    ax=ax[1],
    edgecolor="k",
    scheme="quantiles",
    k=5,
    cmap="GnBu",
)
ax[1].axis(df.total_bounds[np.asarray([0, 2, 1, 3])])
ax[1].set_title("Price")

ax[0].axis("off")
ax[1].axis("off")
plt.show()

# %% [markdown]
# However, we still have the challenge of visually associating the value of the prices
# in a neighborhod with the value of the spatial lag of values for the focal unit. The
# latter is a weighted average of homicide rates in the focal county’s neighborhood. To
# complement the geovisualization of these associations we can turn to formal
# statistical measures of spatial autocorrelation.

# ## Global Spatial Autocorrelation
# We begin with a simple case where the variable under consideration is binary. This is
# useful to unpack the logic of spatial autocorrelation tests. So even though our
# attribute is a continuously valued one, we will convert it to a binary case to
# illustrate the key concepts:
#
# ### Binary Case
# %%
y.median()

# %%
yb: pd.Series = y > y.median()
sum(yb)

# %% [markdown]
# We have 68 neighborhoods with list prices above the median and 70 below the median
# (recall the issue with ties).
# %%
labels: list[str] = ["0 Low", "1 High"]
yb = [labels[i] for i in 1 * yb]  # type: ignore

df["yb"] = yb

# %% [markdown]
# The spatial distribution of the binary variable immediately raises questions about the
# juxtaposition of the “black” and “white” areas.

# %%
fig, ax = plt.subplots(
    figsize=(12, 18),
    subplot_kw={"aspect": "equal"},
)
df.plot(column="yb", cmap="binary", edgecolor="grey", legend=True, ax=ax)
plt.show()

# %% [markdown]
# Join counts One way to formalize a test for spatial autocorrelation in a binary
# attribute is to consider the so-called joins. A join exists for each neighbor pair of
# observations, and the joins are reflected in our binary spatial weights object wq.
#
# Each unit can take on one of two values “Black” or “White”, and so for a given pair of
# neighboring locations there are three different types of joins that can arise:
#
# - Black Black (BB)
# - White White (WW)
# - Black White (or White Black) (BW)
#
# Given that we have 68 Black polygons on our map, what is the number of Black Black
# (BB) joins we could expect if the process were such that the Black polygons were
# randomly assigned on the map? This is the logic of join count statistics.
#
# We can use the esda package from PySAL to carry out join count analysis:

# %%
yb = 1 * (y > y.median())  # convert back to binary
wq = lps.weights.Queen.from_dataframe(df)
wq.transform = "b"
np.random.seed(12345)
jc = esda.join_counts.Join_Counts(yb, wq)

# %% [markdown]
# The resulting object stores the observed counts for the different types of joins:
# %%
jc.bb
# %%
jc.ww
# %%
jc.bw
# %%
all_posibilities: float = jc.bb + jc.ww + jc.bw
print(all_posibilities)

# %% [markdown]
# which is the unique number of joins in the spatial weights object.

# Our object tells us we have observed 121 BB joins:
# %%
jc.bb
# %% [markdown]
# The critical question for us, is whether this is a departure from what we would expect
# if the process generating the spatial distribution of the Black polygons were a
# completely random one? To answer this, PySAL uses random spatial permutations of the
# observed attribute values to generate a realization under the null of complete spatial
# randomness (CSR). This is repeated a large number of times (999 default) to construct
# a reference distribution to evaluate the statistical significance of our observed
# counts.

# The average number of BB joins from the synthetic realizations is:

# %%
jc.mean_bb
# %%
# which is less than our observed count. The question is whether our observed value is
# so different from the expectation that we would reject the null of CSR?

# %%
sbn.kdeplot(jc.sim_bb, fill=True)
plt.vlines(jc.bb, 0, 0.075, color="r")
plt.vlines(jc.mean_bb, 0, 0.075)
plt.xlabel("BB Counts")

# %% [markdown]
# The density portrays the distribution of the BB counts, with the black vertical line
# indicating the mean BB count from the synthetic realizations and the red line the
# observed BB count for our prices. Clearly our observed value is extremely high. A
# pseudo p-value summarizes this:

# %%
jc.p_sim_bb

# %% [markdown]
# Since this is below conventional significance levels, we would reject the null of
# complete spatial randomness in favor of spatial autocorrelation in market prices.

# %% [markdown]
# ## Continuous Case
# The join count analysis is based on a binary attribute, which can cover many
# interesting empirical applications where one is interested in presence and absence
# type phenomena. In our case, we artificially created the binary variable, and in the
# process we throw away a lot of information in our originally continuous attribute.
# Turning back to the original variable, we can explore other tests for spatial
# autocorrelation for the continuous case.

# First, we transform our weights to be row-standardized, from the current binary state:
# %%
wq.transform = "r"

# %%
y: pd.Series = df["median_pri"]  # type: ignore

# %% [markdown]
# **Moran’s I** is a test for global autocorrelation for a continuous attribute:

# %%
np.random.seed(12345)
mi = esda.moran.Moran(y, wq)

mi.I

# %% [markdown]
# Again, our value for the statistic needs to be interpreted against a reference
# distribution under the null of CSR. PySAL uses a similar approach as we saw in the
# join count analysis: random spatial permutations.
# %%
sbn.kdeplot(mi.sim, fill=True)
plt.vlines(mi.I, 0, 1, color="r")
plt.vlines(mi.EI, 0, 1)
plt.xlabel("Moran's I")
plt.show()

# %% [markdown]
# Here our observed value is again in the upper tail, although visually it does not look
# as extreme relative to the binary case. Yet, it is still statistically significant:
# %%
mi.p_sim

# %% [markdown]
# # Local Autocorrelation: Hot Spots, Cold Spots, and Spatial Outliers
# In addition to the Global autocorrelation statistics, PySAL has many local
# autocorrelation statistics. Let’s compute a local Moran statistic for the same d
# %%
np.random.seed(12345)
wq.transform = "r"
lag_price = lps.weights.lag_spatial(wq, df["median_pri"])

# %%
price: pd.Series = df["median_pri"]  # type: ignore
b, a = np.polyfit(price, lag_price, 1)
f, ax = plt.subplots(1, figsize=(9, 9))

plt.plot(
    price,
    lag_price,
    ".",
    color="firebrick",
)

# dashed vert at mean of the price
plt.vlines(
    price.mean(),
    lag_price.min(),
    lag_price.max(),
    linestyle="--",
)
# dashed horizontal at mean of lagged price
plt.hlines(
    lag_price.mean(),
    price.min(),
    price.max(),
    linestyle="--",
)

# red line of best fit using global I as slope
plt.plot(price, a + b * price, "r")
plt.title("Moran Scatterplot")
plt.ylabel("Spatial Lag of Price")
plt.xlabel("Price")
plt.show()
