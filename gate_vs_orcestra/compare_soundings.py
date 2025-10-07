#  %%
# -Define some functions for plotting and evaluating atmospheric soundings
#
import utilities.data_utils as dus
import utilities.preprocessing as dpp
from utilities.settings_and_colors import colors

import xarray as xr
import matplotlib.pyplot as plt
import numpy as np

from moist_thermodynamics import saturation_vapor_pressures as svp
from moist_thermodynamics import functions as mtf
from moist_thermodynamics import constants as mtc
import utilities.thermo as thermo

import seaborn as sns

es = svp.liq_wagner_pruss
# %%
# - load data
#
cids = dus.get_cids()
beach = dus.open_dropsondes(cids["dropsondes"])
rapsodi = dus.open_radiosondes(cids["radiosondes"])
gate = dus.open_gate(cids["gate"])
# %%

# %%

# %%
# - localize data into different domains
#
gs_PE = dpp.sel_percusion_E(gate)
rs_PE = dpp.sel_percusion_E(rapsodi)
bs_PE = dpp.sel_percusion_E(beach)

gs_GA = dpp.sel_gate_A(gs_PE)
rs_GA = dpp.sel_gate_A(rs_PE)
bs_GA = dpp.sel_gate_A(bs_PE)

gs_PE_bar = gs_PE.mean(dim="sonde").coarsen(altitude=10, boundary="trim").mean()
rs_PE_bar = rs_PE.mean(dim="sonde").coarsen(altitude=10, boundary="trim").mean()
bs_PE_bar = bs_PE.mean(dim="sonde").coarsen(altitude=10, boundary="trim").mean()

gs_GA_bar = gs_GA.mean(dim="sonde").coarsen(altitude=10, boundary="trim").mean()
rs_GA_bar = rs_GA.mean(dim="sonde").coarsen(altitude=10, boundary="trim").mean()
bs_GA_bar = bs_GA.mean(dim="sonde").coarsen(altitude=10, boundary="trim").mean()

# For subsequent comparisons decide which region
gs_bar = gs_PE_bar
rs_bar = rs_PE_bar
bs_bar = bs_PE_bar

# %%
# - get aircraft data
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
).pipe(dpp.sel_percusion_E, item_var="TIME", lon_var="longitude", lat_var="latitude")

hal = (
    halo.swap_dims({"TIME": "altitude"})
    .sortby("altitude")
    .sel(altitude=slice(14000, None))
)
halo_rh = halo.RELHUM.where(halo.RELHUM < 70, drop=True) / 100.0
halo_tk = halo.TS.where(halo.RELHUM < 70, drop=True)
x = hal.swap_dims({"altitude": "TIME"})
zbar = x.altitude.mean().values
# %%
# - parameters for soundings
#
sfc_vals = {}
sondes = {"gate": gs_PE, "rapsodi": rs_PE, "beach": bs_PE}
for sset in ["gate", "rapsodi", "beach"]:
    xs = sondes[sset]
    P_sfc = xs.p.sel(altitude=slice(10, 50)).max(dim="altitude") + 100
    RH_sfc = xs.rh.sel(altitude=slice(0, 50)).max(dim="altitude")
    T_sfc = mtf.theta2T(
        xs.theta.sel(altitude=slice(None, 400)).mean(dim="altitude"), P_sfc
    )
    q_sfc = mtf.partial_pressure_to_specific_humidity(RH_sfc * es(T_sfc), P_sfc)
    sfc_vals[sset] = {
        "P": P_sfc.mean(),
        "RH": RH_sfc.mean(),
        "T": T_sfc.mean(),
        "q": q_sfc.mean(),
    }
    print(
        f"{sset:10s}: RH = {RH_sfc.mean().values:.3f}, T = {T_sfc.mean().values:.2f} K, P = {P_sfc.mean().values / 100:.1f} hPa"
    )
# %%
# - zero-degree isotherms
#
datasets = {"rapsodi": rs_PE, "gate": gs_PE, "beach": bs_PE}
print("Height of 0˚ isotherm:")
zp_ticks = {}
for key, ds in datasets.items():
    Tx = (ds.ta - mtc.T0) ** 2
    z_T0 = Tx.idxmin(dim="altitude")
    if ds.title[0] == "G":
        z_T0_gate = z_T0
    print(
        f"  {key}: {z_T0.median().values:.2f}m ({(z_T0.quantile(0.9) - z_T0.quantile(0.1)).values / 2:.2f} m)"
    )
    zp_ticks[key] = z_T0.median().values

zp_ticks

# %%
# - cold point
#
datasets = {"rapsodi": rs_PE, "gate": gs_PE}
sns.set_context(context="paper")
fig, ax = plt.subplots(1, 1, figsize=(3, 2), sharey=True)

print("Height and temerature of cold point:")
cp_ticks = {}
for key, ds in datasets.items():
    mask = (
        ds["ta"].sel(altitude=slice(18000, None)).count(dim="altitude") > 1
    ).compute()
    dsx = ds.isel(sonde=mask)

    z_cp = dsx.ta.idxmin(dim="altitude")
    t_cp = dsx.ta.min(dim="altitude")
    print(f"  {key}: n={np.sum(mask).values} sondes")
    print(
        f"   z at cold point: {z_cp.median().values:.2f} m ({(z_cp.quantile(0.9) - z_cp.quantile(0.1)).values / 2:.2f} m)"
    )
    print(
        f"   T at cold point: {t_cp.median().values:.2f} K ({(t_cp.quantile(0.9) - t_cp.quantile(0.1)).values / 2:.2f} K)"
    )

    z_cp.plot.hist(
        bins=50, density=True, xlim=(14000, 19000), color=colors[key], alpha=0.5
    )

    cp_ticks[key] = z_cp.median().values
ax.set_xlabel("z / m")
sns.despine(offset=10)

Tx = rs_bar.ta
RHice = svp.ice_wagner_etal(Tx) / svp.liq_wagner_pruss(Tx)
# %%
# - convective top
#
datasets = {"rapsodi": rs_bar, "beach": bs_bar, "gate": gs_bar}

ct_ticks = {}
print("Height and temerature of convective top:")
for key, ds in datasets.items():
    z_ct = (
        mtf.brunt_vaisala_frequency(ds.theta, ds.q, ds.altitude)
        .sel(altitude=slice(5e3, 15e3))
        .coarsen(altitude=2, boundary="trim")
        .mean()
        .idxmin(dim="altitude")
    )
    ct_ticks[key] = z_ct.values
    print(f" {key}: z at convective top: {z_ct.values:.2f} m ")

# %%
# - create theoretical soundings
#
Px = 100900.0
P = np.arange(Px, 4000.0, -500)

T1 = sfc_vals["gate"]["T"].values
RHx = sfc_vals["gate"]["RH"].values
q1 = mtf.partial_pressure_to_specific_humidity(RHx * es(T1), Px)

T2 = sfc_vals["rapsodi"]["T"].values
RHx = sfc_vals["gate"]["RH"].values
q2 = mtf.partial_pressure_to_specific_humidity(RHx * es(T2), Px)

pseudo1 = thermo.make_sounding_from_adiabat(P, T1, q1, thx=mtf.theta_e_bolton, Tmin=195)
pseudo2 = thermo.make_sounding_from_adiabat(P, T2, q2, thx=mtf.theta_e_bolton, Tmin=195)
consrv2 = thermo.make_sounding_from_adiabat(P, T2, q2)

zz = np.arange(0, 15000, 10)
delta_pseudo = pseudo2.T - pseudo1.T.interp(altitude=pseudo2.altitude)

# %%
# - plot profiles
#
cw = 190 / 25.4
sns.set_context(context="paper")
fig, ax = plt.subplots(1, 3, figsize=(cw, cw / 2), sharey=True)

ylim = (0, 23000)
rlim = (0, 1)
dlim = (295, 375)
#
# temperature profiles and adiabats
kwargs = {"ax": ax[0], "y": "altitude", "ylim": ylim, "xlim": (185, 200)}

gs_bar.ta.plot(c=colors["gate"], ls="-", label="GATE-RS", **kwargs)
rs_bar.ta.plot(c=colors["rapsodi"], ls="-", label="ORCESTRA-RS", **kwargs)
bs_bar.ta.plot(c=colors["beach"], ls="-", label="ORCESTRA-DS", **kwargs)

pseudo2["T"].plot(c="grey", ls="--", **kwargs)
consrv2["T"].plot(c="grey", ls=":", **kwargs)

ax[0].plot(
    [halo_tk.quantile(0.35), halo_tk.quantile(0.65)],
    [zbar, zbar],
    lw=2.5,
    c="k",
    label="HALO",
)

ax[0].plot(
    [halo_tk.quantile(0.1), halo_tk.quantile(0.9)],
    [zbar, zbar],
    lw=0.5,
    c="k",
)
ax[0].plot(
    [halo_tk.quantile(0.49), halo_tk.quantile(0.51)], [zbar, zbar], lw=2.5, c="w"
)

ax[0].annotate("adiabats", xy=(250, 10500), fontsize=8)

ax[0].annotate(
    "$z_\\mathrm{cp}$",
    xy=(230, (cp_ticks["rapsodi"] + cp_ticks["gate"]) / 2),
    fontsize=8,
)

ax[0].set_xlim(190, 305)
ax[0].set_xlabel("$T$ / K")
ax[0].set_ylabel("z / km")
ax[0].set_xticks(
    [np.round(rs_bar.ta.min().values, 1), 250, np.round(rs_bar.ta.max().values, 1)]
)
ax[0].set_yticks(np.arange(0, 22000, 3000))
ax[0].set_yticklabels([0, 3, 6, 9, 12, 15, 18, 21])
ax[0].legend(fontsize=8, loc="lower left")
sns.despine(ax=ax[0], offset={"bottom": 0, "left": 10})
#
# rh profiles and ice-saturation
kwargs = {"ax": ax[1], "y": "altitude", "ylim": ylim, "xlim": (0, 1)}
rs_bar.rh.plot(c=colors["rapsodi"], ls="-", **kwargs)
bs_bar.sel(altitude=slice(None, 12000)).rh.plot(c=colors["beach"], ls="-", **kwargs)
gs_bar.rh.where(gs_bar.ta > 220).plot(c=colors["gate"], ls="-", **kwargs)

ax[1].plot(
    [halo_rh.quantile(0.35), halo_rh.quantile(0.65)],
    [zbar, zbar],
    lw=2.5,
    c="k",
)

ax[1].plot(
    [halo_rh.quantile(0.1), halo_rh.quantile(0.9)],
    [zbar, zbar],
    lw=0.5,
    c="k",
)
ax[1].plot(
    [halo_rh.quantile(0.49), halo_rh.quantile(0.51)], [zbar, zbar], lw=2.5, c="w"
)

ax[1].annotate("HALO", xy=(0.1, 13500), fontsize=8)

ax[1].annotate(
    "ice-saturation",
    xy=(0.52, 18500),
    color=colors["rapsodi"],
    fontsize=8,
)

ax[1].annotate(
    "$z_0$",
    xy=(0.5, (zp_ticks["rapsodi"] + zp_ticks["gate"]) / 2),
    fontsize=8,
)

RHice.plot(c=colors["rapsodi"], ls="dotted", **kwargs)

ax[1].set_xlabel("RH / 1")
ax[1].set_ylabel("")
ax[1].set_xticks(
    [
        np.round(
            rs_bar.rh.sel(altitude=cp_ticks["rapsodi"], method="nearest").values, 2
        ),
        np.round(rs_bar.rh.sel(altitude=slice(9000, 12000)).mean().values, 2),
        np.round(rs_bar.rh.sel(altitude=slice(None, 400)).min().values, 1),
    ]
)

for x in ["rapsodi", "gate"]:
    ax[0].axhline(cp_ticks[x], xmin=0.05, xmax=0.3, lw=1, ls=":", c=colors[x])
    ax[1].hlines(zp_ticks[x], xmin=0.6, xmax=0.9, lw=1, ls=":", color=colors[x])

sns.despine(ax=ax[1], offset=0)
#
# brunt-vaisala frequency profiles and adiabats
kwargs = {"ax": ax[2], "y": "altitude", "ylim": ylim, "xlim": (0, 0.015)}
mtf.brunt_vaisala_frequency(rs_bar.theta, rs_bar.q, rs_bar.altitude, axis=0).plot(
    c=colors["rapsodi"], ls="-", label="rapsodi", **kwargs
)
mtf.brunt_vaisala_frequency(bs_bar.theta, bs_bar.q, bs_bar.altitude).plot(
    c=colors["beach"], ls="-", label="beach", **kwargs
)
mtf.brunt_vaisala_frequency(gs_bar.theta, gs_bar.q, gs_bar.altitude).plot(
    c=colors["gate"], ls="-", label="gate", **kwargs
)

mtf.brunt_vaisala_frequency(pseudo2["theta"], pseudo2["q"], pseudo2["altitude"]).plot(
    c="grey", ls="--", label="pseudo", **kwargs
)
mtf.brunt_vaisala_frequency(consrv2["theta"], consrv2["q"], consrv2["altitude"]).plot(
    c="grey", ls=":", label="moist", **kwargs
)


nt1 = mtf.brunt_vaisala_frequency(rs_bar.theta, rs_bar.q, rs_bar.altitude, axis=0).sel(
    altitude=slice(None, 5000)
)
nt2 = mtf.brunt_vaisala_frequency(rs_bar.theta, rs_bar.q, rs_bar.altitude, axis=0).sel(
    altitude=slice(9000, None)
)

for x in ["rapsodi", "gate"]:
    print(ct_ticks[x])
    ax[2].axhline(ct_ticks[x], xmin=0.5, xmax=0.8, lw=1, ls=":", c=colors[x])

ax[2].annotate(
    "$z_\\mathrm{ct}$",
    xy=(0.0135, (ct_ticks["rapsodi"] + ct_ticks["gate"]) / 2),
    fontsize=8,
)

ax[2].set_xticks([np.round(nt2.min().values, 3), np.round(nt1.max().values, 3)])
ax[2].set_xlim(0.005, 0.015)
ax[2].set_xlabel("$N$ / s")
ax[2].set_ylabel("")

sns.despine(ax=ax[2], offset={"bottom": 0, "left": 10})

fig.tight_layout()
plt.savefig("plots/sounding.pdf")
plt.show()
# %%
# -- plot differencex wrt GATE
#
ylim = (0, 23000)

delta_T_rs = rs_bar.ta - gs_bar.ta
delta_T_bs = bs_bar.ta - gs_bar.ta

sns.set_context("talk", font_scale=0.8)

plt.style.use("./dark.mplstyle")
# sns.set_context("paper")
fig, ax = plt.subplots(1, 1, figsize=(cw / 2, cw / 2 * 1.333), sharey=True)

delta_T_rs.plot(
    ax=ax, y="altitude", ylim=ylim, label="ORCESTRA-RS", color=colors["rapsodi"]
)
delta_T_bs.sel(altitude=slice(None, 14200)).plot(
    ax=ax, y="altitude", ylim=ylim, label="ORCESTRA-DS", color=colors["beach"]
)

delta_pseudo.where(delta_pseudo > 0.1, drop=True).plot(
    ax=ax, y="altitude", color="k", ls="dotted", lw=1.5
)
# x = x.where(x > 0.01)
# x.plot(ax=ax, y="altitude", color="k", ls="dotted", lw=1)

ax.axvline(0, color="grey", lw=0.5, ls="--")
# ax.vlines(-5.6, ymin=20e3, ymax=23e3, color="grey", lw=0.5, ls="-")
ax.plot([-1, -3], [21.0e3, 23e3], color="grey", lw=0.5, ls="-")

ax.plot(
    np.asarray([-1, 3.5]),
    np.asarray([z_T0.quantile(0.5), z_T0.quantile(0.5)]),
    color="grey",
    lw="1",
    ls="-",
)
ax.annotate(
    "$z_0$",
    xy=(-2, z_T0.quantile(0.5)),
    fontsize=10,
)

ax.set_xlabel("$\\Delta T$ / K")
ax.set_ylabel("z / m")
ax.set_ylabel("z / km")

dth_st = delta_T_rs.sel(altitude=slice(23000, 24500)).mean().values
dth_bl = delta_T_bs.sel(altitude=slice(None, 400)).mean().values

ax.annotate("pseudo adiabat", xy=(2.6, 16200), color=colors["rapsodi"], fontsize=8)
ax.annotate("RCE", xy=(-2.8, 21200), color="k", fontsize=8)

ax.set_xticks(
    [
        np.round(delta_T_rs.min().values, 2),
        np.round(dth_st, 2),
        np.round(dth_bl, 2),
        np.round(delta_T_rs.max().values, 2),
    ]
)
ax.set_xticks([0], minor=True)

ax.set_yticks(np.arange(0, 21500, 3000))
ax.set_yticklabels([0, 3, 6, 9, 12, 15, 18, 21])
ax.set_yticks([z_T0_gate.quantile(0.5), z_T0.quantile(0.5)], minor=True)

plt.legend(fontsize=10, loc="upper right")
plt.tight_layout()
sns.despine(offset={"bottom": 0, "left": 5})
plt.savefig("plots/DeltaT.pdf")
# %%
# - write diagnostics
#
rx = rs_bar.sel(altitude=slice(0, 12000)).set_coords("p").swap_dims({"altitude": "p"})
gx = gs_bar.sel(altitude=slice(0, 12000)).set_coords("p").swap_dims({"altitude": "p"})
bx = bs_bar.sel(altitude=slice(0, 12000)).set_coords("p").swap_dims({"altitude": "p"})
pgrid = np.asarray([100000, 70000, 25000])
dr = (rx.interp(p=pgrid) - gx.interp(p=pgrid))["ta"]
db = (bx.interp(p=pgrid) - gx.interp(p=pgrid))["ta"]

print(
    f"Amplificaiton factors\n Rapsodi: {dr.values[2] / dr.values[1]:.2f}, and {dr.values[2] / dr.values[0]:.2f}"
)
print(
    f" Beach: {db.values[2] / db.values[1]:.2f}, and {db.values[2] / db.values[0]:.2f}"
)
rp = pseudo2.sel(altitude=slice(0, 12000)).set_coords("P").swap_dims({"altitude": "P"})
gp = pseudo1.sel(altitude=slice(0, 12000)).set_coords("P").swap_dims({"altitude": "P"})
dp = (rp.interp(P=pgrid) - gp.interp(P=pgrid))["T"]
print(
    f" Pseudo-adiabat: {dp.values[2] / dp.values[1]:.2f}, and {dp.values[2] / dp.values[0]:.2f}"
)

pgrid = np.asarray([100000, 70000, 35000, 20000])
heights = gp.interp(P=pgrid).altitude
print(heights.values)

# %%
# -- plot difference depending on domain
#
sns.set_context("paper")
fig, ax = plt.subplots(1, 2, figsize=(5, 3), sharey=True)

label = "beach"
delta_bs = bs_GA_bar.theta - bs_PE_bar.theta
delta_bar_bs = delta_bs.sel(altitude=slice(0, 14000)).mean().values
delta_bs.plot(ax=ax[0], y="altitude", ylim=ylim, label=label, c=colors[label])
delta_bs = bs_GA_bar.u - bs_PE_bar.u
delta_bs.plot(ax=ax[1], y="altitude", ylim=ylim, label=label, c=colors[label])

label = "rapsodi"
delta_rs = rs_GA_bar.theta - rs_PE_bar.theta
delta_bar_rs = delta_rs.sel(altitude=slice(0, 14000)).mean().values
delta_rs.plot(ax=ax[0], y="altitude", ylim=ylim, label=label, c=colors[label])
delta_rs = rs_GA_bar.u - rs_PE_bar.u
delta_rs.plot(ax=ax[1], y="altitude", ylim=ylim, label=label, c=colors[label])

ax[0].set_ylim(0, 14300)
ax[0].set_xlim(-1, 1)
ax[0].set_xlabel("$\\Delta \\theta$ / K")
ax[0].set_ylabel("z / km")
ax[0].set_xticks([delta_bar_bs, delta_bar_rs], minor=True)
ax[0].set_yticks(np.arange(0, 12300, 3000))
ax[0].set_yticklabels([0, 3, 6, 9, 12])
ax[1].set_ylabel(None)

plt.legend()

sns.despine(offset=10)
plt.savefig("plots/DeltaT2.pdf")
print(f"differences: beach {delta_bar_bs:.2f}, rapsodi {delta_bar_rs:.2f}")
# %%
# - look at distributions
#
sns.set_context("paper")
fig, ax = plt.subplots(1, 3, figsize=(10, 3))

gate["theta"].sel(altitude=slice(0, 300)).plot.hist(
    ax=ax[0], bins=300, alpha=0.4, density=True, color=colors["gate"]
)

rapsodi["theta"].sel(altitude=slice(0, 300)).plot.hist(
    ax=ax[0], bins=30, alpha=0.4, density=True, color=colors["rapsodi"]
)

beach["theta"].sel(altitude=slice(0, 300)).plot.hist(
    ax=ax[0], bins=30, alpha=0.4, density=True, color=colors["beach"]
)

ax[0].set_xlim(294, 304)
ax[0].set_xlabel("$\\theta_\\mathrm{bl}$ / K")

x1 = (
    rapsodi["theta"]
    .sel(altitude=slice(0, 500))
    .stack(points=["sonde", "altitude"])
    .quantile(0.5)
    .values
)
x2 = (
    gate["theta"]
    .sel(altitude=slice(0, 500))
    .stack(points=["sonde", "altitude"])
    .quantile(0.5)
    .values
)

ax[0].set_xticks([np.round(x1, 2), np.round(x2, 2)])

gate["rh"].sel(altitude=slice(10, 100)).plot.hist(
    ax=ax[1], bins=300, alpha=0.4, density=True, color=colors["gate"]
)
rapsodi["rh"].sel(altitude=slice(10, 100)).plot.hist(
    ax=ax[1], bins=30, alpha=0.4, density=True, color=colors["rapsodi"]
)
beach["rh"].sel(altitude=slice(10, 100)).plot.hist(
    ax=ax[1], bins=30, alpha=0.4, density=True, color=colors["beach"]
)
ax[1].set_xlim(0.5, 1.1)
ax[1].set_xlabel("RH$_\\mathrm{sfc}$ / $-$")

x1 = (
    rapsodi["rh"]
    .sel(altitude=slice(0, 50))
    .stack(points=["sonde", "altitude"])
    .quantile(0.5)
    .values
)
x2 = (
    gate["rh"]
    .sel(altitude=slice(0, 50))
    .stack(points=["sonde", "altitude"])
    .quantile(0.5)
    .values
)

ax[1].set_xticks([0.75, np.round((x1 + x2) / 2, 2), 0.9])

gate["p"].sel(altitude=slice(10, 50)).plot.hist(
    ax=ax[2], bins=30, alpha=0.4, density=True, color=colors["gate"]
)
rapsodi["p"].sel(altitude=slice(10, 50)).plot.hist(
    ax=ax[2], bins=30, alpha=0.4, density=True, color=colors["rapsodi"]
)
beach["p"].sel(altitude=slice(10, 50)).plot.hist(
    ax=ax[2], bins=30, alpha=0.4, density=True, color=colors["beach"]
)
ax[2].set_xlim(100000, 102000)
ax[2].set_xlabel("$p_\\mathrm{sfc}$ / $-$")

x1 = (
    rapsodi["p"]
    .sel(altitude=slice(0, 50))
    .stack(points=["sonde", "altitude"])
    .quantile(0.5)
    .values
)
x2 = (
    gate["p"]
    .sel(altitude=slice(0, 50))
    .stack(points=["sonde", "altitude"])
    .quantile(0.5)
    .values
)

ax[2].set_xticks([100000, np.round((x1 + x2) / 2, 2), 102000])

sns.despine(offset=10)
# %%
# - plot theta profiles
#
sns.set_context(context="paper")
fig, ax = plt.subplots(1, 1, figsize=(3, 3), sharey=True)

ylim = (0, 15000)
tlim = (296, 356)
nlim = (0, 0.02)
rlim = (0, 1)

rs_bar.theta.plot(
    ax=ax, y="altitude", ylim=ylim, xlim=tlim, label="rapsodi", color=colors["rapsodi"]
)
bs_bar.theta.plot(
    ax=ax, y="altitude", ylim=ylim, xlim=tlim, label="beach", color=colors["beach"]
)
gs_bar.theta.plot(
    ax=ax, y="altitude", ylim=ylim, xlim=tlim, label="gate", color=colors["gate"]
)

ax.set_xlabel("$\\theta$ / K")

ax.legend(fontsize=10)
fig.tight_layout()
sns.despine(offset=10)
plt.savefig("plots/theta.pdf")
# %%
# - plot zonal and meridional winds
#
sns.set_context(context="paper")
fig, ax = plt.subplots(1, 2, figsize=(5, 3), sharey=True)

rs_bar.u.plot(ax=ax[0], y="altitude", label="rapsodi", color=colors["rapsodi"])
gs_bar.u.plot(ax=ax[0], y="altitude", label="gate", color=colors["gate"])
bs_bar.u.plot(ax=ax[0], y="altitude", label="beach", color=colors["beach"])

rs_bar.v.plot(ax=ax[1], y="altitude", ylim=ylim, color=colors["rapsodi"])
gs_bar.v.plot(ax=ax[1], y="altitude", color=colors["gate"])
bs_bar.v.plot(ax=ax[1], y="altitude", color=colors["beach"])

ax[0].axvline(x=0.0, ls=":", lw=1)
ax[1].axvline(x=0.0, ls=":", lw=1)

ax[0].set_xlabel("$u$ / ms$^{-1}$")
ax[0].set_ylabel("z / km")
ax[1].set_xlabel("$v$ / ms$^{-1}$")
ax[1].set_ylabel(None)
ax[0].set_xlim(-23, 10)

ax[0].set_yticks(np.arange(0, 18300, 3000))
ax[0].set_yticklabels([0, 3, 6, 9, 12, 15, 18])

umn = rs_bar.u.min().values
umx = bs_bar.u.max().values
vmx = bs_bar.v.sel(altitude=slice(0, 5000)).max().values

z0 = (bs_bar.u**2).idxmin(dim="altitude")
z1 = (bs_bar.u).idxmin(dim="altitude")
ax[0].set_yticks([z0, z1], minor=True)
ax[0].set_xticks([np.round(umn, 1), 0, np.round(umx, 1)])

ax[1].set_xticks([0, np.round(vmx, 1)])
ax[1].set_xlim(-3, 7)

sns.despine(offset=10)
plt.savefig("plots/zonal-wind.pdf")
