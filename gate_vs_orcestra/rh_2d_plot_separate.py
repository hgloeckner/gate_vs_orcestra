# %%
import matplotlib.pyplot as plt
import seaborn as sns
import xarray as xr
import numpy as np
import moist_thermodynamics.functions as mtf
import utilities.thermo as thermo
from moist_thermodynamics import saturation_vapor_pressures as svp
import moist_thermodynamics.constants as mtc
import utilities.data_utils as dus
import utilities.preprocessing as pp
import utilities.modify_ds as md
from utilities.settings_and_colors import colors  # noqa

# %%
cids = dus.get_cids()
datasets = {
    "rapsodi": dus.open_radiosondes(cids["radiosondes"]),
    "beach": dus.open_dropsondes(cids["dropsondes"]),
    "gate": dus.open_gate(cids["gate"]),
}
datasets["orcestra"] = xr.concat(
    [datasets["rapsodi"], datasets["beach"]],
    dim="sonde",
)

for name, ds in datasets.items():
    datasets[name] = (
        ds.pipe(pp.interpolate_gaps).pipe(pp.extrapolate_sfc).pipe(pp.sel_percusion_E)
    )


def density_from_q(p, T, q):
    Rd = mtc.dry_air_gas_constant
    Rv = mtc.water_vapor_gas_constant
    return p / ((Rd + (Rv - Rd) * q) * T)


def calc_iwv(ds):
    ds = ds.assign(rho=density_from_q(ds.p, ds.ta, ds.q))
    iwv = (ds.q * ds.rho).fillna(0).integrate(coord="altitude")
    ds = ds.assign(iwv=xr.where(iwv > 0, iwv, np.nan))
    return ds


iwv = {}
for name, ds in datasets.items():
    iwv[name] = ds.where(
        np.all(~np.isnan(ds.sel(altitude=slice(0, 8000)).q), axis=1)
        & (ds.q > 0)
        & np.all(~np.isnan(ds.sel(altitude=slice(0, 8000)).p), axis=1)
        & np.all(~np.isnan(ds.sel(altitude=slice(0, 8000)).ta), axis=1),
    )
    iwv[name] = calc_iwv(iwv[name])

P = np.arange(100900.0, 4000.0, -500)

orc_sfc = datasets["orcestra"].sel(altitude=0).mean("sonde")
gate_sfc = datasets["gate"].sel(altitude=0).mean("sonde")

orc_pseudo = thermo.make_sounding_from_adiabat(
    P, orc_sfc.ta.values, orc_sfc.q.values, thx=mtf.theta_e_bolton
).rename({"T": "ta", "P": "p"})
orc_pseudo = orc_pseudo.where(orc_pseudo.ta > 200, drop=True)
gate_pseudo = thermo.make_sounding_from_adiabat(
    P, gate_sfc.ta.values, gate_sfc.q.values, thx=mtf.theta_e_bolton
).rename({"T": "ta", "P": "p"})
gate_pseudo = gate_pseudo.where(gate_pseudo.ta > 200, drop=True)


# %%
def get_rh(datasets, rhname, pseudo_ds):
    mean_ta = datasets[rhname].mean(dim="sonde").dropna(dim="altitude", subset=["ta"])
    _, idx = np.unique(mean_ta.ta, return_index=True)
    lcl = (
        pseudo_ds.swap_dims({"altitude": "p"})
        .sel(
            p=mtf.plcl_bolton(
                mean_ta.ta.sel(altitude=0),
                mean_ta.p.sel(altitude=0),
                mean_ta.q.sel(altitude=0),
            ),
            method="nearest",
        )
        .altitude
    )
    return xr.concat(
        [
            mean_ta.rh.interp(
                altitude=pseudo_ds.altitude.sel(altitude=slice(None, lcl))
            ).isel(altitude=slice(0, -1)),
            (
                mean_ta.isel(altitude=idx)
                .swap_dims({"altitude": "ta"})
                .rh.interp(ta=pseudo_ds.ta.values)
                .to_dataset()
                .assign(
                    altitude=("ta", pseudo_ds.altitude.values),
                )
                .swap_dims({"ta": "altitude"})
                .sel(altitude=slice(lcl, None))
                .reset_coords("ta")
                .rh
            ),
        ],
        dim="altitude",
    )


ref_rh = "orcestra"
gate_pseudo = gate_pseudo.assign(
    rh=("altitude", get_rh(datasets, ref_rh, gate_pseudo).values)
).sel(altitude=slice(0, 11500))
orc_pseudo = orc_pseudo.assign(
    rh=("altitude", get_rh(datasets, ref_rh, orc_pseudo).values)
).sel(altitude=slice(0, 11500))
# %%
(ta_datasets["orcestra"].mean("sonde").rh - ta_datasets["gate"].mean("sonde").rh)
# %%
gate_pseudo = calc_iwv(
    gate_pseudo.assign(
        q=mtf.relative_humidity_to_specific_humidity(
            RH=gate_pseudo.rh,
            p=gate_pseudo.p,
            T=gate_pseudo.ta,
        )
    )
)
orc_pseudo = calc_iwv(
    orc_pseudo.assign(
        q=mtf.relative_humidity_to_specific_humidity(
            RH=orc_pseudo.rh,
            p=orc_pseudo.p,
            T=orc_pseudo.ta,
        )
    )
)
# %%

# %%
cw = 190 / 25.4
sns.set_context("talk", font_scale=0.8)
cs_threshold = 0.95
fig, ax = plt.subplots(
    figsize=(cw, cw * 0.75 * 0.3),
)

for name, label in [
    ("rapsodi", "ORCESTRA-RS"),
    ("beach", "ORCESTRA-DS"),
    ("gate", "GATE"),
]:
    sns.histplot(
        data=iwv[name].iwv,
        bins=40,
        binrange=(32, 73),
        element="step",
        stat="density",
        label=label,
        color=colors[name],
        ax=ax,
    )

ax.legend(fontsize=8)
print("orcestra median", iwv["orcestra"].iwv.median().values)
print("gate median", iwv["gate"].iwv.median().values)

ax.axvline(
    x=iwv["orcestra"].iwv.median(),
    ymax=0.91,
    color=colors["orcestra"],
    linestyle="-",
    linewidth=2,
    alpha=0.5,
)
ax.text(
    x=iwv["orcestra"].iwv.median() + 0.2,
    y=0.2,
    fontsize=8,
    s="ORC",
    color=colors["orcestra"],
    ha="left",
    va="top",
    rotation=90,
)

ax.axvline(
    x=iwv["gate"].iwv.median(),
    ymax=0.91,
    color=colors["gate"],
    linestyle="-",
    alpha=0.5,
)
ax.text(
    x=iwv["gate"].iwv.median(),
    y=0.2,
    fontsize=8,
    s="GATE",
    color=colors["gate"],
    ha="right",
    va="top",
    rotation=90,
)
ax.axvline(orc_pseudo.iwv, ymax=0.9, color=colors["orcestra"], linestyle="--")

ax.axvline(gate_pseudo.iwv, ymax=0.9, color=colors["gate"], linestyle="--")

print("orcestra pseudo", orc_pseudo.iwv.values)
print("gate pseudo", gate_pseudo.iwv.values)

mean_pseudo = (orc_pseudo.iwv + gate_pseudo.iwv).values / 2
diff_pseudo = (orc_pseudo.iwv - gate_pseudo.iwv).values


ax.annotate(
    "{:.2f}".format(diff_pseudo),
    xy=(mean_pseudo, 0.205),
    xytext=(mean_pseudo, 0.215),
    fontsize=10,
    ha="center",
    va="bottom",
    arrowprops=dict(arrowstyle="-[, widthB=3.5, lengthB=.1", lw=2.0),
)
mean_campaigns = (iwv["orcestra"].iwv.median() + iwv["gate"].iwv.median()).values / 2
diff_campaigns = iwv["orcestra"].iwv.median() - iwv["gate"].iwv.median()

ax.annotate(
    "{:.2f}".format(diff_campaigns),
    xy=(mean_campaigns, 0.215),
    xytext=(mean_campaigns, 0.225),
    fontsize=10,
    ha="center",
    va="bottom",
    alpha=0.5,
    arrowprops=dict(arrowstyle="-[, widthB=3.5, lengthB=.1", lw=2.0, color="gray"),
)
ax.set_ylim(0, 0.218)
ax.set_yticks(ticks=np.arange(0, 0.2, 0.08))
ax.spines["left"].set_bounds(-0.0, 0.19)

ax.set_ylabel("prob. density")
ax.set_xticks(
    [
        40.0,
        np.round(iwv["gate"].iwv.median(), 1),
        np.round(iwv["orcestra"].iwv.median(), 1),
    ]
)
ax.set_xlim(32, 73)

for axis in ["top", "bottom", "left", "right"]:
    ax.spines[axis].set_linewidth(1)
ax.tick_params(width=0.5, which="both")
sns.despine(offset={"left": 10})
ax.set_xlabel("IWV / kg m$^{-2}$")
fig.savefig(
    "plots/iwv_histogram_full.pdf",
    bbox_inches="tight",
)
# %%

# %%


ta_bin_num = 50
var_bin_num = 50
ta_datasets = {name: [] for name in datasets.keys()}
for name, ds in datasets.items():
    ta_datasets[name].append(
        md.get_hist_of_ta(
            ds.ta.sel(altitude=slice(0, 14000)),
            ds.rh.sel(altitude=slice(0, 14000)),
            var_binrange=(0, 1.1),
            ta_binrange=(220, 305),
            var_bin_num=var_bin_num,
            ta_bin_num=ta_bin_num,
        ).rename("rh")
    )
for name, ds in datasets.items():
    ta_datasets[name].append(
        md.get_hist_of_ta(
            ds.ta.sel(altitude=slice(0, 14000)),
            ds.p.sel(altitude=slice(0, 14000)),
            var_binrange=(10000, 100000),
            ta_binrange=(220, 305),
            var_bin_num=var_bin_num,
            ta_bin_num=ta_bin_num,
        ).rename("p")
    )
for name, ds in datasets.items():
    ta_datasets[name] = xr.merge(ta_datasets[name])
# %%
ta_2d_datasets = {}
for name, ds in datasets.items():
    ta_2d_datasets[name] = md.get_hist_of_ta_2d(
        ds.ta.sel(altitude=slice(0, 14000)),
        ds.rh.sel(altitude=slice(0, 14000)),
        var_binrange=(0, 1.1),
        ta_binrange=(220, 305),
        var_bin_num=var_bin_num,
        ta_bin_num=ta_bin_num,
    )
# %%
# %%


es_liq = svp.liq_wagner_pruss
es_ice = svp.ice_wagner_etal
es_mixed = mtf.make_es_mxd(es_liq=es_liq, es_ice=es_ice)


ice = mtf.relative_humidity_to_specific_humidity(
    1.0,
    p=datasets["orcestra"].p.mean(dim="sonde"),
    T=datasets["orcestra"].ta.mean(dim="sonde"),
    es=es_ice,
)
ice_line = mtf.specific_humidity_to_relative_humidity(
    ice,
    p=datasets["orcestra"].p.mean(dim="sonde"),
    T=datasets["orcestra"].ta.mean(dim="sonde"),
    es=es_liq,
)

rh = 0.95

p = 100000
T = 300
q_low = mtf.relative_humidity_to_specific_humidity(
    rh,
    p=p,
    T=T,
    es=es_liq,
)
print("low q", q_low)
fix_q_rh_low = mtf.specific_humidity_to_relative_humidity(
    q_low,
    p=ta_datasets["orcestra"]
    .p.mean(dim="sonde")
    .interpolate_na(dim="ta", fill_value="extrapolate", method="linear"),
    T=ta_datasets["orcestra"].ta,
    es=es_liq,
)

T = 230
p = 23000
rh = 0.4
q_high = mtf.relative_humidity_to_specific_humidity(
    rh,
    p=p,
    T=T,
    es=es_liq,
)
print("high q ", q_high)
fix_q_rh_high = mtf.specific_humidity_to_relative_humidity(
    q_high,
    p=ta_datasets["orcestra"].p.mean(dim="sonde"),
    T=ta_datasets["orcestra"].ta,
    es=es_mixed,
)
# %%

gate_cmap = sns.light_palette(colors["gate"], as_cmap=True)
orc_cmap = sns.light_palette("cornflowerblue", as_cmap=True)
fig, axes = plt.subplots(
    ncols=2,
    figsize=(cw, cw * 0.75 * 0.7),
    sharey=True,
)
for data_name in ["gate", "orcestra"]:
    for ax in axes:
        ta_datasets[data_name].mean("sonde").rh.plot(
            label=data_name.upper(),
            y="ta",
            color=colors[data_name],
            linewidth=2,
            ax=ax,
        )

for idx, (name, cmap) in enumerate([("gate", gate_cmap), ("orcestra", orc_cmap)]):

    (ta_2d_datasets[name] / ta_2d_datasets[name].sum("rh_bin")).plot.contourf(
        levels=[0, 0.01, 0.02, 0.03, 0.05],
        cmap=cmap,
        ax=axes[idx],
        y="ta_bin",
        x="rh_bin",
        alpha=0.8,
        add_colorbar=False,
    )

bbox_args = dict(boxstyle="round", fc="white", alpha=0.3)
for ax in axes:
    ax.set_ylabel("")
    ax.set_xlabel("RH / 1")
    ax.axhline(273.15, color="k", linestyle="--", linewidth=1)
    ax.plot(
        ice_line.values,
        datasets["orcestra"].ta.mean("sonde"),
        color="black",
        linewidth=2,
    )
    ax.annotate(
        r"RH$_{\text{ice}} = 1$",
        xy=(1.1, 250),
        xycoords="data",
        ha="right",
        va="top",
        bbox=bbox_args,
        fontsize=8,
    )
    ax.plot(
        fix_q_rh_low.values,
        ta_datasets["orcestra"].ta,
        color="black",
        linestyle="--",
        linewidth=2,
    )
    ax.annotate(
        "q = {:.4f}".format(q_low),
        xy=(1.1, 302),
        xycoords="data",
        ha="right",
        va="top",
        fontsize=8,
        bbox=bbox_args,
    )
    ax.plot(
        fix_q_rh_high.values,
        ta_datasets["orcestra"].ta,
        color="black",
        linestyle=":",
        linewidth=2,
    )
    ax.annotate(
        "q = {:.5f}".format(q_high),
        xy=(1.1, 220),
        xycoords="data",
        ha="right",
        va="top",
        fontsize=8,
        bbox=bbox_args,
    )
axes[0].invert_yaxis()
axes[0].legend(loc=3)
axes[0].set_ylabel("$T$ / K")
axes[0].set_xlim(0, 1.05)
axes[0].set_ylim(None, 220)

axes[1].set_yticklabels([])
axes[1].set_xlim(0, 1.1)


axes[0].set_yticks(
    [300, 273.15, 250, 220],
    labels=["300", "273.15", "250", "220"],
)
for ax in axes:
    for axis in ["top", "bottom", "left", "right"]:
        ax.spines[axis].set_linewidth(1)
    ax.tick_params(width=0.5, which="both")
sns.despine(offset={"bottom": 10})
fig.savefig(
    "plots/rh_histograms_full.pdf",
    bbox_inches="tight",
)
# %%
fig, ax = plt.subplots(
    figsize=(cw, cw),
)
(
    (ta_datasets["orcestra"].mean("sonde").rh - ta_datasets["gate"].mean("sonde").rh)
    / (1.31)
    * 100
).plot(
    y="ta",
    color="k",
    linewidth=2,
    ax=ax,
)
ax.invert_yaxis()
ax.set_xlabel("RH change / % K$^{-1}$ sfc warming")
ax.set_ylabel("$T$ / K")
yticks = [300, 280, 270, 260, 240, 220]
ax.set_yticks(yticks, labels=yticks)
ax.axvline(0, color="k", linestyle="--", linewidth=1, alpha=0.5)
sns.despine(offset=10)
# %%
pltcolors = sns.color_palette("Paired", n_colors=8)
cs_threshold = 0.98
fig, ax = plt.subplots(figsize=(5, 5))

for name, color_idx in [("orcestra", 0), ("gate", 4)]:
    ds = ta_datasets[name].rh
    ds.where(ds.max(dim="ta") < cs_threshold).mean("sonde").rolling(ta=5).mean().plot(
        y="ta",
        ax=ax,
        label=f"{name} rh_max < {cs_threshold:.2f}",
        c=pltcolors[color_idx],
    )
    ds.mean("sonde").rolling(ta=5).mean().plot(
        label=name, y="ta", ax=ax, c=pltcolors[color_idx + 1]
    )


ax.invert_yaxis()
ax.legend(loc="upper right", fontsize=12)
ax.set_xlabel("RH / 1")
ax.set_ylabel("$T$ / K")
sns.despine(offset=10)

for axis in ["top", "bottom", "left", "right"]:
    ax.spines[axis].set_linewidth(1)
ax.tick_params(width=0.5, which="both")
fig.savefig(
    "plots/total_vs_clear_sky.pdf",
)
