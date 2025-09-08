# %% [markdown]
## Quality control of sonde data
# Select different regions for analysis and see if there seems to be systematic variations. The picture is broadly consisten

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import moist_thermodynamics.constants as mtc

import utilities.data_utils as dus
import utilities.preprocessing as pre
from utilities.settings_and_colors import colors, percusion_E

cids = dus.get_cids()
colors["halo"] = "red"
g = mtc.gravity_earth

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
        "alts": slice(13800, 14000),
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
    halo = (
        dus.open_halo()
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
        bins=np.arange(coords["alts"].start, coords["alts"].stop, 5),
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
# Compare sondes in the percusion_E area.  At 30W, an overpass would have been at 16 UTC
raps = (
    dus.open_radiosondes(cids["radiosondes"])
    .pipe(pre.sel_sub_domain, percusion_E)
    .swap_dims({"sonde": "launch_time"})
    .sel(altitude=slice(12000, 12100))
)

day_mask = (raps.launch_time.dt.hour > 14) & (raps.launch_time.dt.hour < 18)
raps_day = raps.where(day_mask, drop=True).ta
raps_all = raps.ta

raps_all.plot.hist(bins=30, alpha=0.5, density=True)
raps_day.plot.hist(bins=30, alpha=0.5, density=True)
plt.gca().set_xticks([raps_all.quantile(0.5), raps_day.quantile(0.5)], minor=True)
print(f"Daytime bias {(raps_day.quantile(0.5) - raps_all.quantile(0.5)).values:.2f} K")
sns.despine(offset=5)

# %% [markdown]
## Differences at common heights
# compare radiosondes and dropsondes at 12km

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
## Differences at common heights
# compare radiosondes and halo at 14km

raps = (
    dus.open_radiosondes(cids["radiosondes"])
    .pipe(pre.sel_sub_domain, percusion_E)
    .swap_dims({"sonde": "launch_time"})
    .sel(altitude=slice(13800, 14000))
)

halo = (
    dus.open_halo()
    .pipe(
        pre.sel_sub_domain,
        percusion_E,
        item_var="TIME",
        lon_var="longitude",
        lat_var="latitude",
    )
    .swap_dims({"TIME": "altitude"})
    .sortby("altitude")
    .sel(altitude=slice(13800, 14000))
)

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(7, 3))

raps.p.plot.hist(ax=ax1, bins=30, alpha=0.5, density=True, color=colors["rapsodi"])
(halo.PS * 100).plot.hist(
    ax=ax1, bins=30, alpha=0.5, density=True, color=colors["halo"]
)
ax1.set_xticks([halo.PS.quantile(0.5) * 100, raps.p.quantile(0.5)], minor=True)
ax1.set_xlabel("$P$ / hPa")
ax1.set_xlim(15500, 16500)
print("Distributions at 14 km")
print(
    f"--Pressure difference: {((halo.PS.quantile(0.5) * 100 - raps.p.quantile(0.5)).values) / 100:.2f} hPa"
)

raps.ta.plot.hist(ax=ax2, bins=30, alpha=0.5, density=True, color=colors["rapsodi"])
halo.TS.plot.hist(ax=ax2, bins=30, alpha=0.5, density=True, color=colors["halo"])
ax2.set_xticks([halo.TS.quantile(0.5), drps.ta.quantile(0.5)], minor=True)
ax2.set_xlabel("$T$ / K")
ax2.set_xlim(205, 215)

print(
    f"--Temperature difference: {((halo.TS.quantile(0.5) - raps.ta.quantile(0.5)).values):.2f} K"
)

raps.theta.plot.hist(ax=ax3, bins=30, alpha=0.5, density=True, color=colors["rapsodi"])
halo.THETA.plot.hist(ax=ax3, bins=30, alpha=0.5, density=True, color=colors["halo"])
ax3.set_xticks([halo.THETA.quantile(0.5), halo.THETA.quantile(0.5)], minor=True)
ax3.set_xlabel("$\\Theta$ / K")
print(
    f"--Potential temperature difference: {((halo.THETA.quantile(0.5) - raps.theta.quantile(0.5)).values):.2f} K"
)
sns.despine(offset=5)


# %% [markdown]
## Summary
# What I think I learned was that
# - the radiosondes are a bit colder than the aircraft and the dropsondes at the same height
# - some of this difference, maybe 1/3 can be attributed to the diurnal cycle
# - at 12 km the radiosondes measure lower (2 hPa) pressure than the dropsondes, similar to halo radio sondes at 14 km.  At this height this corresponds to a hdyrostatic pressure difference of about 50m
# - this lower pressure compensates for the colder sondes, i.e., differences in $\theta$ are smaller
# - integrating hydrostatically would accumulate much larger height/pressure differences than are actually observed at altitude
# - errors in estimate pressures are expected due to the binning, but the bias is more systematic (and hence the accumulation larger) than I would have expected.
# - not shown but I compared radio and dropsondes at 200m - 300m.  The dropsondes were 0.2 K warmer (diurnal?) and pressure was 0.25 hPa lower
# - there is the semi-diurnal tide, maybe this explains some of the difference?
# Based on this I have the feeling that things are broadly okay, but that even after accounting for diurnal effects the radiosondes are a bit cold, or their altitude is a bit under-estimated.  If they were mapped to the dropsonde altitude at the same pressure the cold bias would largely vanish.
