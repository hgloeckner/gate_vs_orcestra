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
sns.set_context("talk", font_scale=0.8)

plt.style.use("./dark.mplstyle")
cw = 190 / 25.4
gate_cmap = sns.light_palette(colors["gate"], as_cmap=True)
orc_cmap = sns.light_palette("cornflowerblue", as_cmap=True)
fig, axes = plt.subplots(
    ncols=3,
    figsize=(cw * 1.5, cw * 0.75 * 0.7),
    sharey=True,
)
for data_name in ["gate", "orcestra"]:
    for ax in axes[:2]:
        ta_datasets[data_name].mean("sonde").rh.plot(
            label=data_name.upper(),
            y="ta",
            color=colors[data_name],
            linewidth=2,
            ax=ax,
        )

for idx, (name, cmap) in enumerate([("gate", gate_cmap), ("orcestra", orc_cmap)]):

    (ta_2d_datasets[name] / ta_2d_datasets[name].sum("rh_bin")).plot.contourf(
        levels=[0, 0.005, 0.01, 0.02, 0.03, 0.05],
        cmap=cmap,
        ax=axes[idx],
        y="ta_bin",
        x="rh_bin",
        alpha=0.8,
        add_colorbar=False,
    )

bbox_args = dict(boxstyle="round", fc="white", alpha=0.3)
for ax in axes[:2]:
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

(
    (ta_datasets["orcestra"].mean("sonde").rh - ta_datasets["gate"].mean("sonde").rh)
    / 1.32
).plot(
    y="ta",
    color="white",
)
"""
axes[2].axvline(0, color="k", linestyle="--", linewidth=1, alpha=0.5)
"""
axes[2].set_ylabel("")
axes[2].set_xlabel("RH change / % K$^{-1}$")
axes[2].set_axis_off()

axes[0].invert_yaxis()
axes[0].legend(loc=3)
axes[0].set_ylabel("$T$ / K")
axes[0].set_xlim(0, 1.05)
axes[0].set_ylim(None, 220)

axes[1].set_yticklabels([])
axes[1].set_xlim(0, 1.1)
for ax in axes[:2]:
    ax.set_xticks([0, 0.5, 1])

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
    "plots/rh_histograms_left.pdf",
    bbox_inches="tight",
)
# %%
fig, ax = plt.subplots(
    figsize=(cw * 0.3, cw * 0.75 * 0.7),
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
ax.set_xlabel("RH change / % K$^{-1}$")
ax.set_ylabel("")
yticks = [300, 273, 250, 220]
ax.set_yticks(yticks, labels=[])
for axis in ["top", "bottom", "left", "right"]:
    ax.spines[axis].set_linewidth(1)
ax.tick_params(width=0.5, which="both")
ax.axvline(0, color="k", linestyle="--", linewidth=1, alpha=0.5)
sns.despine(offset=10)

fig.savefig(
    "plots/rh_difference.pdf",
    bbox_inches="tight",
)
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
