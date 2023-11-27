# %%

import esda
import fiona
import folium
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
# Fallowing the Exploratory Analysis of Spatial Data: Spatial Autocorrelation
# tutorial of [PySAL](https://pysal.org/esda/notebooks/spatialautocorrelation.html)
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

# %% [markdown]
# Comparation between Distribution and the Spatial Lag for housing deficit
# %%
gdf["lag_deficit"] = ylag

f, ax = plt.subplots(1, 2, figsize=(20, 20))
gdf.plot(
    column="df_déficit_habitacional_(dh)",
    scheme="Quantiles",
    k=5,
    cmap="GnBu",
    edgecolor="black",
    ax=ax[0],
)

ax[0].axis(gdf.total_bounds[np.asarray([0, 2, 1, 3])])
ax[0].set_title("Deficit habitacional")

gdf.plot(
    column="lag_deficit",
    scheme="Quantiles",
    k=5,
    cmap="GnBu",
    edgecolor="black",
    ax=ax[1],
)
ax[1].axis(gdf.total_bounds[np.asarray([0, 2, 1, 3])])
ax[1].set_title("Spatial Lag Deficit")

ax[0].axis("off")
ax[1].axis("off")
plt.show()

# %% [markdown]
# ## Global Spatial Autocorrelation
# ### The binary case

# %%
y.median()
# %%
yb = y > y.median()
print(f"Districts above median {sum(yb)}")

# %%
print(
    f"We have {y.count()} districts where {sum(yb)} are above median and {y.count() - sum(yb)} are below"
)

# %%
yb = y > y.median()
labels = ["0 Low", "1 High"]
yb = [labels[i] for i in 1 * yb]
gdf["yb"] = yb
# %%
fig, ax = plt.subplots(figsize=(20, 20), subplot_kw={"aspect": "equal"})


gdf.plot(column="yb", cmap="binary", edgecolor="grey", legend=True, ax=ax)

# %% [markdown]
# ## Join counts
#
# One way to formalize a test for spatial autocorrelation in a binary attribute is to
# consider the so-called joins. A join exists for each neighbor pair of observations,
# and the joins are reflected in our binary spatial weights object *wq*

# %%
yb = 1 * (y > y.median())
wq = lps.weights.Queen.from_dataframe(gdf)
wq.transform = "b"  # type: ignore
np.random.seed(12345)
jc = esda.join_counts.Join_Counts(yb, wq)

# %% [markdown]
# Join counts results
# %%
print(f"Black on black join {jc.bb}")
print(f"White on white join {jc.ww}")
print(f"Black on white join {jc.bw}")

print(f"All cases with all posibilites {jc.bb + jc.ww + jc.bw}")

print(f"{wq.s0 / 2} unique number of joins in the spatial weights object.")

# %% [markdown]
# The critical question for us, is whether this is a departure from what we would expect
# if the process generating the spatial distribution of the Black polygons were a
# completely random one?

# %%
print(f"The average number of BB joins from the synthetic realizations is {jc.mean_bb}")

# %% [markdown]
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
print(f"pseudo p-value {jc.p_sim_bb}")

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
# autocorrelation for the continuous case. First, we transform our weights to be
# row-standardized, from the current binary state:
# %%
wq.transform = "r"  # type: ignore
y = gdf["df_déficit_habitacional_(dh)"]

# %% [markdown]
# Moran’s I is a test for global autocorrelation for a continuous attribute:

# %%
np.random.seed(12345)
mi = esda.moran.Moran(y, wq)
mi.I

# %% [markdown]
# Again, our value for the statistic needs to be interpreted against a reference
# distribution under the null of CSR. PySAL uses a similar approach as we saw in the
# join count analysis: random spatial permutations.

# %%
sbn.kdeplot(mi.sim, shade=True)
plt.vlines(mi.I, 0, 1, color="r")
plt.vlines(mi.EI, 0, 1)
plt.xlabel("Moran's I")

# %% [markdown]
# Here our observed value is again in the upper tail, although visually it does not look
# as extreme relative to the binary case. Yet, it is still statistically significant:

# %%
mi.p_sim

# %% [markdown]
# ## Local Autocorrelation: Hot Spots, Cold Spots, and Spatial Outliers
np.random.seed(12345)
wq.transform = "r"  # type: ignore

lag_deficit = lps.weights.lag_spatial(wq, gdf["df_déficit_habitacional_(dh)"])
deficit = gdf["df_déficit_habitacional_(dh)"]
b, a = np.polyfit(deficit, lag_deficit, 1)
f, ax = plt.subplots(1, figsize=(10, 10))

plt.plot(deficit, lag_deficit, ".", color="firebrick")

# dashed vert at mean of the deficit
plt.vlines(deficit.mean(), lag_deficit.min(), lag_deficit.max(), linestyle="--")
# dashed horizontal at mean of lagged deficit
plt.hlines(lag_deficit.mean(), deficit.min(), deficit.max(), linestyle="--")

# red line of best fit using global I as slope
plt.plot(deficit, a + b * deficit, "r")
plt.title("Moran Scatterplot")
plt.ylabel("Spatial Lag of Deficit")
plt.xlabel("Deficit")
plt.show()

# %% [markdown]
# Now, instead of a single statistic, we have an array of local statistics, stored in
# the .Is attribute, and p-values from the simulation are in p_sim.
# %%
li = esda.moran.Moran_Local(y, wq)
li.q

# %% [markdown]
# We can again test for local clustering using permutations, but here we use conditional
# random permutations (different distributions for each focal location)

# %%
(li.p_sim < 0.05).sum()

# %% [markdown]
# We can distinguish the specific type of local spatial association reflected in the
# four quadrants of the Moran Scatterplot above:

# %%
sig = li.p_sim < 0.05
hotspot = sig * li.q == 1
coldspot = sig * li.q == 3
doughnut = sig * li.q == 2
diamond = sig * li.q == 4

# %%
spots = ["n.sig.", "hot spot"]
labels = [spots[i] for i in hotspot * 1]

# %%
hmap = colors.ListedColormap(["red", "lightgrey"])
f, ax = plt.subplots(1, figsize=(15, 15))
gdf.assign(cl=labels).plot(
    column="cl",
    categorical=True,
    k=2,
    cmap=hmap,
    linewidth=0.1,
    ax=ax,
    edgecolor="white",
    legend=True,
)
ax.set_axis_off()
plt.show()

# %%
spots = ["n.sig.", "cold spot"]
labels = [spots[i] for i in coldspot * 1]

# %%
hmap = colors.ListedColormap(["blue", "lightgrey"])
f, ax = plt.subplots(1, figsize=(15, 15))
gdf.assign(cl=labels).plot(
    column="cl",
    categorical=True,
    k=2,
    cmap=hmap,
    linewidth=0.1,
    ax=ax,
    edgecolor="white",
    legend=True,
)
ax.set_axis_off()
plt.show()

# %%
spots = ["n.sig.", "doughnut"]
labels = [spots[i] for i in doughnut * 1]

# %%
hmap = colors.ListedColormap(["lightblue", "lightgrey"])
f, ax = plt.subplots(1, figsize=(15, 15))
gdf.assign(cl=labels).plot(
    column="cl",
    categorical=True,
    k=2,
    cmap=hmap,
    linewidth=0.1,
    ax=ax,
    edgecolor="white",
    legend=True,
)
ax.set_axis_off()
plt.show()

# %%
spots = ["n.sig.", "diamond"]
labels = [spots[i] for i in diamond * 1]

# %%
hmap = colors.ListedColormap(["pink", "lightgrey"])
f, ax = plt.subplots(1, figsize=(15, 15))
gdf.assign(cl=labels).plot(
    column="cl",
    categorical=True,
    k=2,
    cmap=hmap,
    linewidth=0.1,
    ax=ax,
    edgecolor="white",
    legend=True,
)
ax.set_axis_off()
plt.show()

# %%
sig = 1 * (li.p_sim < 0.05)
hotspot = 1 * (sig * li.q == 1)
coldspot = 3 * (sig * li.q == 3)
doughnut = 2 * (sig * li.q == 2)
diamond = 4 * (sig * li.q == 4)
spots = hotspot + coldspot + doughnut + diamond
spots

# %%
spot_labels = ["0 ns", "1 hot spot", "2 doughnut", "3 cold spot", "4 diamond"]
labels = [spot_labels[i] for i in spots]

# %%
hmap = colors.ListedColormap(["lightgrey", "red", "lightblue", "blue", "pink"])
f, ax = plt.subplots(1, figsize=(15, 15))
gdf.assign(cl=labels).plot(
    column="cl",
    categorical=True,
    k=5,
    cmap=hmap,
    linewidth=0.1,
    ax=ax,
    edgecolor="white",
    legend=True,
)
ax.set_axis_off()
plt.show()

# %% [markdown]
# ## Interactive mapping with GeoPandas

# %%
hmap = colors.ListedColormap(["lightgrey", "red", "lightblue", "blue", "pink"])
m = gdf.assign(cl=labels).explore(
    column="cl",
    categorical=True,
    k=5,
    cmap=hmap,
    linewidth=0.1,
    edgecolor="white",
    legend=True,
)

folium.TileLayer("CartoDB positron", show=False).add_to(m)
folium.LayerControl().add_to(m)

m
