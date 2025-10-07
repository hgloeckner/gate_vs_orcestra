from xhistogram.xarray import histogram
import numpy as np


def get_hist_of_ta(
    da_t, da_var, var_binrange, ta_binrange=(240, 305), var_bin_num=100, ta_bin_num=200
):
    bins_ta = np.linspace(ta_binrange[0], ta_binrange[1], ta_bin_num)
    bins_var = np.linspace(var_binrange[0], var_binrange[1], var_bin_num)
    bin_name = da_var.name + "_bin"
    ta_name = da_t.name
    hist = histogram(da_t, da_var, bins=[bins_ta, bins_var], dim=["altitude"]).compute()
    return (
        ((hist * hist[bin_name]).sum(dim=bin_name) / hist.sum(bin_name))
        .interpolate_na(dim=f"{ta_name}_bin")
        .rename({f"{ta_name}_bin": ta_name})
    )


def get_hist_of_ta_2d(
    da_t, da_var, var_binrange, ta_binrange=(240, 305), var_bin_num=100, ta_bin_num=200
):
    bins_ta = np.linspace(ta_binrange[0], ta_binrange[1], ta_bin_num)
    bins_var = np.linspace(var_binrange[0], var_binrange[1], var_bin_num)
    return histogram(da_t, da_var, bins=[bins_ta, bins_var], dim=da_t.dims)
