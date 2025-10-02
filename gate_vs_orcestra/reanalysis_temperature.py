# %%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import easygems.healpix as egh
from scipy import signal
import cartopy.crs as ccrs
import cmocean
import cartopy.feature as cf

import utilities.preprocessing as pp
from utilities.settings_and_colors import colors

# %%

reanalysis_sfc = pp.preprocess_sfc_temperatures()
reanalysis_sfc["BEST"] = pp.get_tsrf_berkeley()

# %%
detrended = {}

for name in ["ERA5", "MERRA2", "JRA3Q"]:
    ds = reanalysis_sfc[name]
    detrended[name] = signal.detrend(ds.mean("cell").sel(year=slice(1974, None)).values)


detrended["BEST"] = signal.detrend(reanalysis_sfc["BEST"].values)
# %%
std = np.nanstd(np.concatenate([vals for vals in detrended.values()]))

# %%
cw = 190 / 25.4
sns.set_context("paper")
fig, ax = plt.subplots(figsize=(cw, cw / 2))

for name, vals in detrended.items():
    ax.plot(
        reanalysis_sfc[name].sel(year=slice(1974, None)).year,
        vals,
        label=name,
        color=colors[name.lower()] if name.lower() in colors else "k",
    )


ax.fill_between(
    np.array([1974, 2025]),
    [-std, -std],
    [std, std],
    color="gray",
    alpha=0.2,
)


ax.axhline(0, color="k", linestyle="-", linewidth=0.5)
ax.legend()
ax.set_xlabel("Year")
ax.set_ylabel("August/September mean anomaly / K")
ax.set_title("Detrended 2m/sfc air temperature for lon: [-27, -20] and lat: [5, 12]")
sns.despine(offset=10)
xticks = ax.get_xticks()
xticks = np.append(xticks, [1974, 2024])
ax.set_xticks(xticks)
yticks = np.concatenate([ax.get_yticks(), [-std, std]])
ax.set_yticks(
    [
        tick
        for tick in np.where(np.abs(np.abs(yticks) - 0.2) > 1e-3, yticks, np.nan)
        if not np.isnan(tick)
    ],
    labels=[
        f"{tick:.2f}"
        for tick in np.where(np.abs(np.abs(yticks) - 0.2) > 1e-3, yticks, np.nan)
        if not np.isnan(tick)
    ],
)
ax.set_xlim(1973, 2025)
fig.savefig(
    "images/sfc_temperature_anomaly.pdf",
)

# %%

projection = ccrs.Robinson(central_longitude=10)
fig, ax = plt.subplots(
    figsize=(8, 4), subplot_kw={"projection": projection}, constrained_layout=True
)
ax.set_extent([-67, -10, -2, 22], crs=ccrs.PlateCarree())

egh.healpix_show(reanalysis_sfc["JRA3Q"].mean("year"), ax=ax, cmap=cmocean.cm.thermal)
ax.add_feature(cf.COASTLINE, linewidth=0.8)
ax.add_feature(cf.BORDERS, linewidth=0.4)

# %%
ds = reanalysis_sfc["BEST"].to_dataframe()
# %%

fig, ax = plt.subplots(figsize=(16, 9))
sns.barplot(
    x=ds.index,
    y=1,
    hue=ds.temperature,
    palette="RdBu_r",
    linewidth=0,
    edgecolor=None,
    width=1,
    fill=True,
    legend=False,
)
ax.set_axis_off()
plt.margins(0, 0)
fig.savefig("images/berkeley_sfc_temperature.pdf", bbox_inches="tight", pad_inches=0)
