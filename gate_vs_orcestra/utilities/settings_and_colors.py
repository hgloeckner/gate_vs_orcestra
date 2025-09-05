import numpy as np

colors = {
    "rapsodi": "navy",
    "beach": "teal",
    "gate": "orangered",
    "orcestra": "royalblue",
    "jra3q": "#435D00",
    "merra2": "#6C9600",
    "era5": "#98D500",
    "best": "#00A9A0",
    "pirata12": "#939494",
    "pirata4": "#D9F407",
}


gate_A = np.array(
    [
        [-27.0, 6.5],
        [-23.5, 5.0],
        [-20.0, 6.5],
        [-20.0, 10.5],
        [-23.5, 12.0],
        [-27.0, 10.5],
    ]
)

gate_B = np.array(
    [
        [-23.5, 7.0],
        [-24.8, 7.8],
        [-24.8, 9.2],
        [-23.5, 10.0],
        [-22.2, 9.2],
        [-22.2, 7.8],
        [-23.5, 7.0],
    ]
)

percusion_E = np.array([[-34.0, 12.5], [-20.0, 12.5], [-20.0, 4.5], [-34.0, 4.5]])
