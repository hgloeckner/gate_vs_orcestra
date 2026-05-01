import numpy as np
import xarray as xr
import moist_thermodynamics.constants as mtc


def make_atmosphere(p, T, h2o_vmr, o3, T_s=None):
    """Create a pyRTE-RRTMG atmosphere from pressure, temperature and humidity arrays."""
    if np.any(p[..., 0] < p[..., -1]):
        raise ValueError("Arrays need to be passed in ascending order")

    if T_s is None:
        T_s = T[..., 0]

    atmosphere = xr.Dataset(
        data_vars={
            "pres_level": (("column", "level"), p),
            "temp_level": (("column", "level"), T),
            "pres_layer": (("column", "layer"), 0.5 * (p[..., 1:] + p[..., :-1])),
            "temp_layer": (("column", "layer"), 0.5 * (T[..., 1:] + T[..., :-1])),
            "surface_temperature": (("column",), T_s),
            "h2o": (("column", "layer"), 0.5 * (h2o_vmr[..., 1:] + h2o_vmr[..., :-1])),
            "o3": (("layer"), 0.5 * (o3[..., 1:] + o3[..., :-1])),
            "co2": 422e-6,
            "ch4": 1650e-9,
            "n2o": 306e-9,
            "n2": 0.7808,
            "o2": 0.2095,
            "co": 0.0,
        },
    )

    return atmosphere


def mixing_ratio2vmr(w):
    Md = mtc.md
    Mw = mtc.molar_mass_h2o

    return w / (w + Mw / Md)


def specific_humidity2vmr(q):
    Md = mtc.md
    Mw = mtc.molar_mass_h2o

    return q / ((1 - q) * Mw / Md + q)


def vmr2specific_humidity(x):
    Md = mtc.md
    Mw = mtc.molar_mass_h2o

    return x / ((1 - x) * Md / Mw + x)
