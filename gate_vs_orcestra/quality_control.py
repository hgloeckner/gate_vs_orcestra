# %% [markdown]
## Quality control of sonde data
# Select just the Meteor circle or the whole day.  Doing so doesn't make a difference, the closer comparision is actually worse. It suggests that on this day the radio sondes measure colder temperatures, by about 1 K, than either HALO or the dropsondes

import xarray as xr
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import moist_thermodynamics.constants as mtc

import utilities.data_utils as dus
import utilities.preprocessing as pre
from utilities.settings_and_colors import colors, percusion_E

cids = dus.get_cids()
colors["halo"] = "red"

setup = {
    "september-3-percusion_e": {
        "area": percusion_E,
        "days": slice("2024-09-03", "2024-09-03"),
        "alts": slice(13875, 13925),
    },
    "september-3-meteor": {
        "area": np.array([[-34.0, 10.0], [-20.0, 10.0], [-20.0, 7.0], [-34.0, 7.0]]),
        "days": slice("2024-09-03T14:50", "2024-09-03T16:45"),
        "alts": slice(13875, 13925),
    },
    "east": {
        "area": percusion_E,
        "days": slice("2024-08-10", "2024-09-06"),
        "alts": slice(14300, 14500),
    },
    "west": {
        "area": np.array([[-61.0, 5.0], [-40.0, 5.0], [-40.0, 15.0], [-61.0, 15.0]]),
        "days": slice("2024-09-05", "2024-09-30"),
        "alts": slice(14300, 14500),
    },
    "september-23-gate-bco": {
        "area": np.array([[-61.0, 12.8], [-57.5, 12.8], [-57.5, 13.5], [-61.0, 13.5]]),
        "days": slice("2024-09-23", "2024-09-23"),
        "alts": slice(13000, 13250),
    },
}

alts2 = slice(12000, 12100)
sns.set_context("paper")

for key, coords in setup.items():
    domain = "september-3-meteor"
    halo = (
        xr.open_dataset(
            "ipfs://bafybeif52irmuurpb27cujwpqhtbg5w6maw4d7zppg2lqgpew25gs5eczm",
            engine="zarr",
        )
        .rename_vars(
            {
                "IRS_LAT": "latitude",
                "IRS_LON": "longitude",
                "IRS_ALT": "altitude",
            }
        )
        .set_coords(({"latitude", "longitude", "altitude"}))
        .pipe(
            pre.sel_sub_domain,
            coords["area"],
            item_var="TIME",
            lon_var="longitude",
            lat_var="latitude",
        )
        .sel(TIME=coords["days"])
        .swap_dims({"TIME": "altitude"})
        .sortby("altitude")
        .sel(altitude=coords["alts"])
    )

    raps_14km = (
        dus.open_radiosondes(cids["radiosondes"])
        .pipe(pre.sel_sub_domain, coords["area"])
        .swap_dims({"sonde": "launch_time"})
        .sel(launch_time=coords["days"])
        .sel(altitude=coords["alts"])
    )

    raps_12km = (
        dus.open_radiosondes(cids["radiosondes"])
        .pipe(pre.sel_sub_domain, coords["area"])
        .swap_dims({"sonde": "launch_time"})
        .sel(launch_time=coords["days"])
        .sel(altitude=alts2)
    )

    drps = (
        dus.open_dropsondes(cids["dropsondes"])
        .pipe(pre.sel_sub_domain, coords["area"])
        .swap_dims({"sonde": "launch_time"})
        .sel(launch_time=coords["days"])
        .sel(altitude=alts2)
    )

    fig, axs = plt.subplots(2, 2, figsize=(8, 6))

    axs[0, 0].scatter(
        halo.longitude, halo.latitude, s=0.1, color=colors["halo"], zorder=3
    )
    axs[0, 0].scatter(
        raps_14km.launch_lon,
        raps_14km.launch_lat,
        alpha=0.5,
        s=50,
        zorder=2,
        color=colors["rapsodi"],
    )
    axs[0, 0].scatter(
        drps.launch_lon,
        drps.launch_lat,
        alpha=0.5,
        s=50,
        zorder=2,
        color=colors["beach"],
    )
    axs[0, 0].set_xlabel("longitude / deg E")
    axs[0, 0].set_ylabel("latitude / deg N")

    halo.altitude.plot.hist(
        ax=axs[1, 0],
        bins=np.arange(12500, 15000, 50),
        density=True,
        color=colors["halo"],
    )
    axs[1, 0].set_xlabel("altitude / m")

    halo.TS.plot.hist(ax=axs[0, 1], color=colors["halo"], alpha=0.5, density=True)
    raps_14km.ta.plot.hist(
        ax=axs[0, 1], color=colors["rapsodi"], alpha=0.5, density=True
    )
    axs[0, 1].set_xticks(
        [raps_14km.ta.quantile(0.5), halo.TS.quantile(0.5)], minor=True
    )
    print(
        f"Region {key}:\n --halo minus rapsodi {(halo.TS.quantile(0.5) - raps_14km.ta.quantile(0.5)).values:.2f} K"
    )

    drps.ta.plot.hist(ax=axs[1, 1], color=colors["beach"], alpha=0.5, density=True)
    raps_12km.ta.plot.hist(
        ax=axs[1, 1],
        color=colors["rapsodi"],
        alpha=0.5,
        density=True,
    )
    axs[1, 1].set_xlabel("$T$ / K")
    axs[1, 1].set_xticks(
        [raps_12km.ta.quantile(0.5), drps.ta.quantile(0.5)], minor=True
    )
    print(
        f" --beach minus rapsodi {(drps.ta.quantile(0.5) - raps_12km.ta.quantile(0.5)).values:.2f} K"
    )

    axs[0, 1].set_title("")
    axs[1, 1].set_title("")
    sns.despine(offset=5)
    fig.suptitle(key, fontsize=14)
    fig.tight_layout()


# %% [markdown]
## Radiosonde Diel Cycle
# compare sondes near the overpass at 30W, which would have been about 14 local time, or 16 UTC
raps = (
    dus.open_radiosondes(cids["radiosondes"])
    .pipe(pre.sel_sub_domain, percusion_E)
    .swap_dims({"sonde": "launch_time"})
    .sel(altitude=slice(12000, 12100))
)

day_mask = (raps.launch_time.dt.hour > 12) & (raps.launch_time.dt.hour < 18)
raps_day = raps.where(day_mask, drop=True).ta

ngt_mask = raps.launch_time.dt.hour < 6
raps_ngt = raps.where(ngt_mask, drop=True).ta

raps_ngt.plot.hist(bins=30, alpha=0.5, density=True)
raps_day.plot.hist(bins=30, alpha=0.5, density=True)
plt.gca().set_xticks([raps_ngt.quantile(0.5), raps_day.quantile(0.5)], minor=True)
print(
    f"Day night difference {(raps_day.quantile(0.5) - raps_ngt.quantile(0.5)).values:.2f} K"
)
sns.despine(offset=5)

# %% [markdown]
## Pressure differences
# compare pressure in radio and dropsondes at the same height level
raps = (
    dus.open_radiosondes(cids["radiosondes"])
    .pipe(pre.sel_sub_domain, percusion_E)
    .swap_dims({"sonde": "launch_time"})
    .sel(altitude=slice(12000, 12100))
)
drps = (
    dus.open_dropsondes(cids["dropsondes"])
    .pipe(pre.sel_sub_domain, percusion_E)
    .swap_dims({"sonde": "launch_time"})
    .sel(altitude=slice(12000, 12100))
)

day_mask = (raps.launch_time.dt.hour > 12) & (raps.launch_time.dt.hour < 18)
raps_day = raps.where(day_mask, drop=True).p

ngt_mask = raps.launch_time.dt.hour < 6
raps_ngt = raps.where(ngt_mask, drop=True).p

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(7, 3))

raps.p.plot.hist(ax=ax1, bins=30, alpha=0.5, density=True, color=colors["rapsodi"])
drps.p.plot.hist(ax=ax1, bins=30, alpha=0.5, density=True, color=colors["beach"])
ax1.set_xticks([drps.p.quantile(0.5), raps.p.quantile(0.5)], minor=True)
ax1.set_xlabel("$P$ / hPa")
print("Distributions at 12 km")
print(
    f"--Pressure difference: {((drps.p.quantile(0.5) - raps.p.quantile(0.5)).values) / 100:.2f} hPa"
)

raps.ta.plot.hist(ax=ax2, bins=30, alpha=0.5, density=True, color=colors["rapsodi"])
drps.ta.plot.hist(ax=ax2, bins=30, alpha=0.5, density=True, color=colors["beach"])
ax2.set_xticks([raps.ta.quantile(0.5), drps.ta.quantile(0.5)], minor=True)
ax2.set_xlabel("$T$ / K")
print(
    f"--Temperature difference: {((drps.ta.quantile(0.5) - raps.ta.quantile(0.5)).values):.2f} K"
)

raps.theta.plot.hist(ax=ax3, bins=30, alpha=0.5, density=True, color=colors["rapsodi"])
drps.theta.plot.hist(ax=ax3, bins=30, alpha=0.5, density=True, color=colors["beach"])
ax3.set_xticks([raps.theta.quantile(0.5), drps.theta.quantile(0.5)], minor=True)
ax3.set_xlabel("$\\Theta$ / K")
print(
    f"--Potential temperature difference: {((drps.theta.quantile(0.5) - raps.theta.quantile(0.5)).values):.2f} K"
)
sns.despine(offset=5)

# %% [markdown]
## Deviations in  hydrostaticity
# We calculate the hydrostatic altitude, Z, and compare this to the sonde-altitude z.  Then do the same for the pressure

raps = (
    dus.open_radiosondes(cids["radiosondes"])
    .pipe(pre.sel_sub_domain, percusion_E)
    .swap_dims({"sonde": "launch_time"})
    .sel(altitude=slice(None, 25100))
)
drps = (
    dus.open_dropsondes(cids["dropsondes"])
    .pipe(pre.sel_sub_domain, percusion_E)
    .swap_dims({"sonde": "launch_time"})
    .sel(altitude=slice(None, 25100))
)

datasets = {"rapsodi": raps, "beach": drps}

sns.set_context("paper")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 3))

dZ = {}
for key, dx in datasets.items():
    ds = dx
    gamma = ds.ta.diff(dim="altitude", label="lower") / 10.0
    q = 0.5 * ds.ta.diff(dim="altitude", label="lower") + ds.q
    R = mtc.Rd + (mtc.Rv - mtc.Rd) * q
    chi = R / mtc.gravity_earth
    dlnp = np.log(ds.p).diff(dim="altitude", label="lower")
    dZ[key] = (ds.ta[:-1] / gamma * (np.exp(-chi * gamma * dlnp) - 1)).sel(
        altitude=slice(None, 12000)
    )
    val = 100 * (dZ[key].mean(dim="launch_time").quantile(0.5) - 10).values
    dZ[key].mean(dim="launch_time").plot.hist(
        ax=ax1,
        bins=np.arange(9, 11, 0.01),
        alpha=0.4,
        color=colors[key],
        density=True,
        label=f"{key} ({val:.0f} cm)",
    )

    Z = dZ[key].mean(dim="launch_time").cumsum(dim="altitude")
    ax2.plot(Z.altitude[1:], Z - Z.altitude[1:], color=colors[key])

ax1.set_xlabel(" $\\Delta Z$ / m")
ax2.set_xlabel("$z$ / m")
ax2.set_ylabel("$(Z-z)$ / m")
ax1.legend(fontsize=8, loc="upper left")
ax2.set_ylim(-250, 250)
sns.despine(offset=10)

# %%

sns.set_context("paper")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 3))

dP = {}
for key, dx in datasets.items():
    ds = dx
    gamma = ds.ta.diff(dim="altitude", label="lower") / 10
    q = 0.5 * ds.ta.diff(dim="altitude", label="lower") + ds.q
    R = mtc.Rd + (mtc.Rv - mtc.Rd) * q
    chi = R / mtc.gravity_earth
    dP[key] = (
        -ds.p[:-1]
        * (np.exp(np.log(10 * gamma / ds.ta[:-1] + 1) / (chi * gamma)) - 1)
        / ds.p.diff(dim="altitude", label="lower")
    )
    val = (dP[key].mean(dim="launch_time").quantile(0.5)).values
    dP[key].mean(dim="launch_time").plot.hist(
        ax=ax1,
        bins=np.arange(0.95, 1.2, 0.001),
        alpha=0.4,
        color=colors[key],
        density=True,
        label=f"{key} ({val:.2f} cm)",
    )

    P = ((dP[key] - 1) * ds.p.diff(dim="altitude", label="lower")).mean(
        dim="launch_time"
    ).cumsum(dim="altitude") / 100
    ax2.plot(P.altitude, P, color=colors[key])

ax1.set_xlabel(" $\\Delta Z$ / m")
ax2.set_xlabel("$z$ / m")
ax2.set_ylabel("$(P-p)$ / hPa")
ax1.legend(fontsize=8, loc="upper left")
sns.despine(offset=10)
# %% [markdown]
## Summary
# What I think I learned was that
# - the radiosondes are a bit colder than the aircraft and the dropsondes at the same height
# - some of this difference, but I don't think all, can be the diurnal cycle
# - at 12 km the radiosondes measure lower (2 hPa) pressure than the dropsondes
# - this lower pressure compensates for the colder sondes, i.e., differences in $\theta$ are smaller
# - the hydrostatic altitude of the dropsondes doesn't increase as much as the altitude
# - the opposite is true for the radiosondes, primarily above 5km
# It could be good to do the above, but separating out only the ascending radiosondes.  But I also don't see a consistent picture that would lead me to the error, except to note that if I multiply $\chi = R/g$ by 1.02 the dropsondes hydrostatic relationships look perfect. This is the same as multiplying $\mathrm{dln}p$ by 1.02, which seems hard to justify.

# %%
