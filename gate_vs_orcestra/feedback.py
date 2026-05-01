# %%
import xarray as xr
import moist_thermodynamics.functions as mtf
import moist_thermodynamics.saturation_vapor_pressures as svp
import numpy as np
import matplotlib.pyplot as plt
import utilities.rad_helper as rh
import utilities.preprocessing as pp
from xhistogram.xarray import histogram

from pyrte_rrtmgp.rrtmgp import GasOptics
from pyrte_rrtmgp.rrtmgp_data_files import GasOpticsFiles
# %%

es_liq = svp.liq_hardy  # liq_wagner_pruss
es_ice = svp.ice_wagner_etal
es_mixed = mtf.make_es_mxd(es_liq=es_liq, es_ice=es_ice)
es = es_mixed


itcz = np.array([[-34.0, 6], [-20, 6], [-20, 11], [-34.0, 11]])
beach = (
    xr.open_dataset(
        "ipfs://bafybeiczbv7mycr2jois6t4dq3zwiltycomwo5xxvjqcjz2ot3newzar6q",
        engine="zarr",
    )
    .pipe(pp.interpolate_gaps)
    .pipe(pp.extrapolate_sfc)
    .pipe(pp.sel_sub_domain, itcz)
    .reset_coords()
)
rapsodi = (
    xr.open_dataset(
        "ipfs://QmcQRuqCgLRUVyCXjzmKfRVL34xxnxzL91PWTJSELrtQxa", engine="zarr"
    )
    .pipe(pp.interpolate_gaps)
    .pipe(pp.extrapolate_sfc)
    .pipe(pp.sel_sub_domain, itcz)
)
gate = (
    xr.open_dataset(
        "ipfs://QmWZryTDTZu68MBzoRDQRcUJzKdCrP2C4VZfZw1sZWMJJc", engine="zarr"
    )
    .pipe(pp.interpolate_gaps)
    .pipe(pp.extrapolate_sfc)
    .pipe(pp.sel_sub_domain, itcz)
)
orcestra = xr.concat([rapsodi, beach], dim="sonde")
# %%
ozone = xr.open_dataset("../data/ozone_profile.nc")
# %%


datasets = {"orc": orcestra, "gate": gate}
max_alt = 13000
ta_bin_num = 500
ta_datasets = {name: [] for name in datasets.keys()}
for name, ds in datasets.items():
    dsmax = ds.ta.mean("sonde").sel(altitude=0).values

    ta_datasets[name].append(
        (
            histogram(
                ds.ta.sel(altitude=slice(0, max_alt)),
                bins=[ta_bin_num],
                range=[(220, dsmax + 0.1)],
                weights=ds.rh.sel(altitude=slice(0, max_alt)),
                dim=["altitude"],
            )
            / histogram(
                ds.ta.sel(altitude=slice(0, max_alt)),
                bins=[ta_bin_num],
                range=[(220, dsmax + 0.1)],
                dim=["altitude"],
            )
        ).rename("rh")
    )
for name, ds in datasets.items():
    ta_datasets[name].append(
        (
            histogram(
                ds.ta.sel(altitude=slice(0, max_alt)),
                bins=[ta_bin_num],
                range=[(220, dsmax + 0.1)],
                weights=ds.p.sel(altitude=slice(0, max_alt)),
                dim=["altitude"],
            )
            / histogram(
                ds.ta.sel(altitude=slice(0, max_alt)),
                bins=[ta_bin_num],
                range=[(220, dsmax + 0.1)],
                dim=["altitude"],
            )
        ).rename("p")
    )
for name, ds in datasets.items():
    ta_datasets[name] = xr.merge(ta_datasets[name]).rename({"ta_bin": "ta"})
# %%
mean_orc = (
    ta_datasets["orc"].mean("sonde").interpolate_na("ta", fill_value="extrapolate")
)
mean_orc = mean_orc.assign(
    O3=ozone.o3.isel(temp_layer=slice(0, 180)).interp(
        temp_layer=mean_orc.ta, kwargs=dict(fill_value="extrapolate")
    ),
    H2O=rh.specific_humidity2vmr(
        mtf.relative_humidity_to_specific_humidity(
            mean_orc.rh, mean_orc.p, mean_orc.ta, es=es
        )
    ),
)
mean_gate = (
    ta_datasets["gate"].mean("sonde").interpolate_na("ta", fill_value="extrapolate")
)


# change which rh is used here!
gate_rh = mean_gate.rh.interpolate_na("ta")  # mean_orc.rh.interpolate_na("ta")
mean_gate = mean_gate.assign(
    O3=ozone.o3.isel(temp_layer=slice(0, 180)).interp(
        temp_layer=mean_gate.ta, kwargs=dict(fill_value="extrapolate")
    ),
    H2O=rh.specific_humidity2vmr(
        mtf.relative_humidity_to_specific_humidity(
            gate_rh, mean_gate.p, mean_gate.ta, es=es
        ),
    ),
).dropna("ta", how="any")

mean_gate = (
    mean_gate.swap_dims({"ta": "p"})
    .interp(
        p=np.insert(mean_gate.p.values, 0, 0),
        method="nearest",
        kwargs=dict(fill_value="extrapolate"),
    )
    .swap_dims({"p": "ta"})
)
mean_orc = mean_orc.dropna("ta", how="any")
mean_orc = (
    mean_orc.swap_dims({"ta": "p"})
    .interp(
        p=np.insert(mean_orc.p.values, 0, 0),
        method="nearest",
        kwargs=dict(fill_value="extrapolate"),
    )
    .swap_dims({"p": "ta"})
)

# %%
flxs = {}
for label, ds in [("orc", mean_orc), ("gate", mean_gate)]:
    ds = (
        ds.sortby("ta", ascending=False)
        .drop_vars(["temp_layer"])
        .swap_dims({"ta": "level"})
        .reset_coords(["ta", "p"])
        .expand_dims("column", axis=0)
    )

    atmosphere = rh.make_atmosphere(
        ds.p.values, ds.ta.values, ds.H2O.values, ds.O3.values[0, :]
    )
    gas_optics_lw = GasOptics(gas_optics_file=GasOpticsFiles.LW_G256)
    gas_optics_sw = GasOptics(gas_optics_file=GasOpticsFiles.SW_G224)
    optical_props = gas_optics_lw.compute(atmosphere, add_to_input=False)

    optical_props = optical_props.assign(surface_emissivity=0.98)
    clr_fluxes = optical_props.rte.solve(add_to_input=False)
    flxs[label] = xr.merge([clr_fluxes, atmosphere]).rename(
        {"temp_level": "ta", "pres_level": "p"}
    )
    flxs[label] = flxs[label].assign(
        rh=mtf.specific_humidity_to_relative_humidity(
            rh.vmr2specific_humidity(flxs[label].h2o),
            flxs[label].pres_layer,
            flxs[label].temp_layer,
            es=es,
        )
    )

fig, ax = plt.subplots()
for ds in [flxs["orc"], flxs["gate"]]:
    ds = ds.sel(column=0)
    ax.plot(ds.p, ds.lw_flux_down - ds.lw_flux_up)

ax.set_xlim(20000, 25000)
ax.set_ylim(-260, -240)
# %%

flxorc = flxs["orc"].squeeze().swap_dims({"level": "p"}).sortby("p", ascending=False)
flxgate = flxs["gate"].squeeze().swap_dims({"level": "p"}).sortby("p", ascending=False)


p = 21000

print(
    "lw fb:",
    (
        (
            (flxorc.lw_flux_down - flxorc.lw_flux_up).sel(p=p, method="nearest")
            - (flxgate.lw_flux_down - flxgate.lw_flux_up).sel(p=p, method="nearest")
        )
        / (flxs["orc"].ta.isel(level=0).squeeze() - flxs["gate"].ta.isel(level=0))
        .squeeze()
        .values
    ).values,
)
print(
    "flx diff ",
    (
        (flxorc.lw_flux_down - flxorc.lw_flux_up).sel(p=p, method="nearest")
        - (flxgate.lw_flux_down - flxgate.lw_flux_up).sel(p=p, method="nearest")
    ).values,
)
print(
    "temp diff ",
    (flxs["orc"].ta.isel(level=0).squeeze() - flxs["gate"].ta.isel(level=0))
    .squeeze()
    .values,
)
# %%

pltvar = "rh"
temp_var = "temp_layer"

fig, ax = plt.subplots()

ax.plot(
    flxs["orc"].isel(column=0)[pltvar],
    flxs["orc"].isel(column=0)[temp_var],
    label="orc",
)
ax.plot(
    flxs["gate"].isel(column=0)[pltvar],
    flxs["gate"].isel(column=0)[temp_var],
    label="gate",
)
ax.legend()
ax.invert_yaxis()
# ax.set_xlim(290, None)
# ax.set_xlim(0, )
