# %%
# - Define some functions for plotting and evaluating atmospheric soundings
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

import utilities.data_utils as dus
import utilities.preprocessing as pre

# %%
# - load data
#
# %%
# - load data
#
cids = dus.get_cids()
beach = dus.open_dropsondes(cids["dropsondes"])
rapsodi = dus.open_radiosondes(cids["radiosondes"])
gate = dus.open_gate(cids["gate"])

platforms = np.unique(gate.platform_id.values).tolist()
platforms.append("ALL")
# %%
# - plot gate data
#
sns.set_context(context="paper")

for RV in platforms:
    fig, ax = plt.subplots(1, 5, figsize=(5, 3), sharey=True)
    i = 0
    if RV == "ALL":
        ds = gate.pipe(pre.sel_gate_A)
    else:
        ds = gate.isel(sonde=(gate.platform_id == RV).compute()).pipe(pre.sel_gate_A)
    for fld in ["ta", "rh", "u", "v", "p"]:
        x = ds[fld].stack(points=["sonde", "altitude"])
        ax[i].scatter(x, x.altitude, s=1, color="k", alpha=0.003)
        ds[fld].mean(dim="sonde").plot(ax=ax[i], y="altitude", lw=1.0, color="w")

        ax[i].set_ylim(0, 32000)
        ax[i].set_ylabel(None)
        i += 1

    ax[0].set_ylabel("altitude / km")
    ax[0].set_ylabel("altitude / km")
    ax[0].set_yticks(np.arange(0, 32000, 3000))
    ax[0].set_yticklabels([0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30])
    ax[0].set_xlabel("$T$ / K")
    ax[1].set_xlabel("$RH$ / -")
    ax[2].set_xlabel("$u$ / m s$^{-1}$")
    ax[3].set_xlabel("$v$ / m s$^{-1}$")
    ax[4].set_xlabel("$P$ / Pa")

    sns.despine(offset=5)
    plt.suptitle(RV)
    plt.savefig(f"plots/{RV}_outliers.png", dpi=600)

# %%

# %%
# - plot gate data
#
sns.set_context(context="paper")

for RV in [
    "GILLISS",
]:  # platforms:
    fig, ax = plt.subplots(1, 1, figsize=(5, 3), sharey=True)
    i = 0
    if RV == "ALL":
        ds = gate.pipe(pre.sel_gate_A)
    else:
        ds = gate.isel(sonde=(gate.platform_id == RV).compute()).pipe(pre.sel_gate_A)
    for fld in ["q"]:
        for sonde in ds[fld].sonde[15:18]:
            x = ds[fld].sel(sonde=sonde)
            x = x.where(x > 0, drop=True)  # .stack(points=["sonde", "altitude"])
            ax.scatter(x, x.altitude, s=2, alpha=1)
        #            ds[fld].mean(dim="sonde").plot(ax=ax, y="altitude", lw=1.0, color="w")

        ax.set_ylim(0, 32000)
        ax.set_ylabel(None)
        i += 1

    ax.set_xscale("log")
    ax.set_ylim(5000, 24500)
    ax.set_xlim(5e-6, 5e-4)
    ax.set_ylabel("altitude / km")
    ax.set_ylabel("altitude / km")
    #    ax.set_yticks(np.arange(0, 32000, 3000))
    #    ax.set_yticklabels([6, 9, 12, 15, 18, 21, 24, 27, 30])
    ax.set_xlabel("$q$ / kg/kg")

    sns.despine(offset=5)
    plt.suptitle(RV)
    plt.savefig(f"plots/{RV}_outliers.png", dpi=600)
# %%
for sonde in ds[fld].sonde:
    print(ds[fld].sel(sonde=sonde))
# %%
gate = xr.open_dataset("/Users/m219063/data/gate-l2.zarr", engine="zarr")
# %%
