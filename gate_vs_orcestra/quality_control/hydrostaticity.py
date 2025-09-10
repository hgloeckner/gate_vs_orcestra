# %%
#
import xarray as xr
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import moist_thermodynamics.constants as mtc
import moist_thermodynamics.functions as mtf
import moist_thermodynamics.saturation_vapor_pressures as svp
import utilities.data_utils as dus
import utilities.preprocessing as pre
from utilities.settings_and_colors import colors, percusion_E

cids = dus.get_cids()

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
# %%

# %% [markdown]
## Simple Example
# illustrates how grouping gives more spread in observed values

zmax = 10000.0
dz_obs = 8.0
dz_bin = 10
hgt = np.arange(0, zmax, dz_obs)
hgt_obs = hgt + np.random.rand(len(hgt)) * dz_obs / 5
hgt_bin = np.arange(0, zmax, dz_bin)
z = xr.DataArray(
    hgt_obs,
    dims=[
        "z",
    ],
    coords={"z": hgt},
)
dz = z.groupby_bins("z", hgt_bin).mean().diff(dim="z_bins")
dz.plot.hist(bins=30)  # , label=f"{1.0 - dx.quantile(0.5).values:.2f}")
sns.despine(offset=10)
# %%
gamma = 8e-3
dz_12 = (
    225
    / gamma
    * (np.exp(mtc.Rd / mtc.gravity_earth * gamma * np.log(21570 / 21375)) - 1)
)
dz_14 = (
    211
    / gamma
    * (np.exp(mtc.Rd / mtc.gravity_earth * gamma * np.log(16300 / 16125)) - 1)
)
print(
    f"effective height differences between radisondes and:\n dropsondes {dz_12:.2f} m\n halo {dz_14:.2f} m"
)

# %% [markdown]
## effect of binning on dropsonde hydrostaticity
# takes some time to run unfortunately :(

l2res = []
files = dus.fsglob(
    f"ipfs://{cids['orcestra']}/products/HALO/dropsondes/Level_2/HALO-*/*.zarr",
)
files


# %%
def get_alt_diff(ds, altdim="altitude"):
    dsn = ds.assign(
        q=mtf.relative_humidity_to_specific_humidity(
            ds.rh, ds.p, ds.ta, es=svp.liq_wagner_pruss
        )
    )
    dsn = dsn.dropna(dim=altdim, how="any", subset=["ta", "p", "q"])
    hydro = np.diff(
        mtf.pressure_altitude(
            dsn.p.values,
            dsn.ta.values,
            qv=dsn.q.values,
        )
    )
    dz = dsn[altdim].diff(dim=altdim)
    return dz - hydro  # .where(dz < 100)


# %%
for file in files:
    l2 = xr.open_dataset(f"ipfs://{file}", engine="zarr")
    # print(file)
    if l2.sonde_qc.values == 0:
        l2res.append(get_alt_diff(l2.swap_dims({"time": "gpsalt"}), altdim="gpsalt"))
# %%
l3 = dus.open_dropsondes(cids["dropsondes"])
ds = l3.where(l3.sonde_qc == 0)[["altitude", "ta", "p", "rh"]]
ds = ds.dropna(dim="sonde", how="all")

# %%
l3res = []
for sonde in ds.sonde:
    l3res.append(get_alt_diff(ds.sel(sonde=sonde)))
# %% rapsodi
rapsres = []
raps = raps.swap_dims({"launch_time": "sonde"})
for sonde in raps.sonde:
    rapsres.append(get_alt_diff(raps.sel(sonde=sonde)))

# %%
sns.histplot(
    np.concatenate(l3res),
    bins=300,
    binrange=(-5, 5),
    stat="probability",
    label="L3 Beach: median {:.2f} m".format(np.nanmedian(np.concatenate(l3res))),
)
sns.histplot(
    np.concatenate(l2res),
    bins=300,
    binrange=(-5, 5),
    stat="probability",
    label="L2 Beach: median {:.2f} m".format(np.nanmedian(np.concatenate(l2res))),
)
sns.histplot(
    np.concatenate(rapsres),
    bins=300,
    binrange=(-5, 5),
    stat="probability",
    label="Rapsodi: median {:.2f} m".format(np.nanmedian(np.concatenate(rapsres))),
)
plt.legend()
