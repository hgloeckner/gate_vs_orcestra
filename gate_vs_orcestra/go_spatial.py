# %%
import seaborn as sns

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.lines as mlines

import cartopy.crs as ccrs
import cartopy.feature as cfeature

import utilities.data_utils as dus
import utilities.preprocessing as dpp
from utilities.settings_and_colors import colors, gate_A, percusion_E

# %%
# - load gate sounding data
#
cids = dus.get_cids()
beach = dus.open_dropsondes(cids["dropsondes"])
rapsodi = dus.open_radiosondes(cids["radiosondes"])
gate = dus.open_gate(cids["gate"])

gs_PE = dpp.sel_percusion_E(gate)
rs_PE = dpp.sel_percusion_E(rapsodi)
bs_PE = dpp.sel_percusion_E(beach)

gs_GA = dpp.sel_gate_A(gs_PE)
rs_GA = dpp.sel_gate_A(rs_PE)
bs_GA = dpp.sel_gate_A(bs_PE)
# %%
# - plot geographic distribution of soundings
#
cw = 190 / 25.4
sns.set_context(context="talk", font_scale=0.8)
plt.style.use("./dark.mplstyle")
line_color = "k"
face_color = "white"
small = 3
big = 6
fig, ax = plt.subplots(
    figsize=(cw, cw * 2 / 3), subplot_kw={"projection": ccrs.PlateCarree()}
)

ngso = len(gs_PE.sonde)
nrso = len(rs_PE.sonde)
nbso = len(bs_PE.sonde)
ngs = len(gs_GA.sonde)
nrs = len(rs_GA.sonde)
nbs = len(bs_GA.sonde)

kwargs = {"transform": ccrs.PlateCarree(), "marker": "o"}
ax.scatter(
    gs_PE.launch_lon,
    gs_PE.launch_lat,
    s=small,
    color=colors["gate"],
    alpha=0.5,
    label=f"GATE (n={ngso})",
    **kwargs,
)
ax.scatter(
    rs_PE.launch_lon,
    rs_PE.launch_lat,
    s=small,
    color=colors["rapsodi"],
    alpha=0.5,
    label=f"ORCESTRA (Meteor, n={nrso})",
    **kwargs,
)
ax.scatter(
    bs_PE.launch_lon,
    bs_PE.launch_lat,
    s=small,
    color=colors["beach"],
    alpha=0.5,
    label=f"ORCESTRA (HALO, n={nbso})",
    **kwargs,
)
ax.scatter(
    gs_GA.launch_lon,
    gs_GA.launch_lat,
    s=big,
    color=colors["gate"],
    alpha=0.5 / 3,
    label=f"GATE (n={ngs})",
    **kwargs,
)
ax.scatter(
    rs_GA.launch_lon,
    rs_GA.launch_lat,
    s=big,
    color=colors["rapsodi"],
    alpha=0.5,
    label=f"ORCESTRA (Meteor, n={nrs})",
    **kwargs,
)
ax.scatter(
    bs_GA.launch_lon,
    bs_GA.launch_lat,
    s=big,
    color=colors["beach"],
    alpha=0.5,
    label=f"ORCESTRA (HALO, n={nbs})",
    **kwargs,
)

tr = rs_GA.launch_time.mean().dt.strftime("%m-%d").values
td = bs_GA.launch_time.mean().dt.strftime("%m-%d").values
tg = gs_GA.launch_time.mean().dt.strftime("%m-%d").values

print(
    f"RAPSODI: {rs_GA.launch_lon.mean().values:.2f}, {rs_GA.launch_lat.mean().values:.2f}, {tr}"
)
print(
    f"BEACH:   {bs_GA.launch_lon.mean().values:.2f}, {bs_GA.launch_lat.mean().values:.2f}, {td}"
)
print(
    f"GATE:    {gs_GA.launch_lon.mean().values:.2f}, {gs_GA.launch_lat.mean().values:.2f}, {tg}"
)
#
# Add map features
#
ax.coastlines(color="black")
ax.add_feature(cfeature.LAND, facecolor="lightgray")

ax.set_xlim(-37.0, -13.0)
ax.set_ylim(2.0, 18.0)

gl = ax.gridlines(
    draw_labels=True, linewidth=1, color="none", alpha=0.5, linestyle="--"
)
gl.top_labels = False
gl.right_labels = False
gl.xlocator = mticker.FixedLocator([])
gl.ylocator = mticker.FixedLocator([])

ax.set_xlabel("longitude / $^\\circ$W")
ax.set_ylabel("latitude / $^\\circ$N")

ax.plot(
    gate_A[:3, 0],
    gate_A[:3, 1],
    color=line_color,
    lw=1,
    ls="solid",
    label="GATE A/B Array",
)
ax.plot(gate_A[3:, 0], gate_A[3:, 1], color=line_color, lw=1, ls="solid")
ax.plot(percusion_E[:2, 0], percusion_E[:2, 1], color=line_color, lw=1, ls="solid")
ax.plot(percusion_E[2:, 0], percusion_E[2:, 1], color=line_color, lw=1, ls="solid")

h_g1 = mlines.Line2D(
    [],
    [],
    color=colors["gate"],
    marker="o",
    linestyle="None",
    markersize=3,
    alpha=0.3,
    label=f"GATE ($n=$ {ngs})",
)
h_g2 = mlines.Line2D(
    [],
    [],
    color=colors["gate"],
    marker="o",
    linestyle="None",
    markersize=1,
    alpha=0.75,
    label=f"GATE ($n=$ {ngso})",
)
h_r1 = mlines.Line2D(
    [],
    [],
    color=colors["rapsodi"],
    marker="o",
    linestyle="None",
    markersize=3,
    alpha=0.9,
    label=f"METEOR ($n=${nrs})",
)
h_r2 = mlines.Line2D(
    [],
    [],
    color=colors["rapsodi"],
    marker="o",
    linestyle="None",
    markersize=1,
    alpha=0.9,
    label=f"METEOR ($n=${nrso})",
)
h_b1 = mlines.Line2D(
    [],
    [],
    color=colors["beach"],
    marker="o",
    linestyle="None",
    markersize=3,
    alpha=0.9,
    label=f"HALO ($n=${nbs})",
)
h_b2 = mlines.Line2D(
    [],
    [],
    color=colors["beach"],
    marker="o",
    linestyle="None",
    markersize=1,
    alpha=0.9,
    label=f"HALO ($n=${nbso})",
)

h_p1 = mlines.Line2D(
    [],
    [],
    color=colors["pirata12"],
    marker="*",
    linestyle="None",
    markersize=5,
    alpha=1,
    label="PIRATA (-23˚E, 12˚N)",
)

h_p2 = mlines.Line2D(
    [],
    [],
    color=colors["pirata4"],
    marker="*",
    linestyle="None",
    markersize=4,
    alpha=1,
    label="PIRATA (-23˚E, 4˚N)",
)

ax.legend(
    bbox_to_anchor=(-0.15, 1.10),
    title="ORCESTRA East              GATE A/B             ",
    ncol=2,
    loc="upper left",
    framealpha=1,
    handles=[h_g2, h_b2, h_r2, h_p1, h_g1, h_b1, h_r1],
    fontsize=10,
    title_fontsize=12,
    facecolor=face_color,
)

xticks = [-34, -27.0, -23.5, -20]
yticks = [4.5, 8.5, 12.5]
ax.set_xticks(xticks)
ax.set_xlabel("longitude / $^\\circ$W", fontsize=12)
ax.set_yticks(yticks)
ax.set_ylabel("latitude / $^\\circ$N", fontsize=12)
for xlat in [12]:
    ax.scatter(
        [-23],
        [xlat],
        s=5,
        edgecolors=colors["pirata" + str(xlat)],
        facecolors=colors["pirata" + str(xlat)],
        linewidth=1.0,
        transform=ccrs.PlateCarree(),
        marker="*",
    )

ax.annotate(
    "Dakar",
    xy=(-17.467686, 14.716677),
    xytext=(-20.5, 15.5),
    fontsize=10,
    color=line_color,
    arrowprops=dict(arrowstyle="->", color=line_color),
)

ax.annotate(
    "Sal",
    xy=(-22.916663, 16.8499966),
    xytext=(-21.5, 17.0),
    fontsize=10,
    color=line_color,
    arrowprops=dict(arrowstyle="->", color=line_color),
)

ax.annotate("GATE A/B array", xy=(-20.3, 9.55), fontsize=10)
ax.annotate("ORCESTRA East", xy=(-35.5, 11.5), fontsize=10)


ax.tick_params(width=0.5, which="both", colors=line_color)

fig.tight_layout()
plt.savefig("plots/gate-orcestra-sondes.pdf", transparent=True)

# %%
