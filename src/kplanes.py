import numpy as np
import xarray as xr

from config import (Constants)
from pathlib import Path


class KPlane0500:

    def __init__(self) -> None:

        self.temp = xr.open_dataset(Path("../data/KPlane0500Temperature160000.nc"))
        self.w = xr.open_dataset(Path("../data/KPlane0500VerticalVelocity.nc"))

    @property
    def temp_anom(self):
        return self.temp.isel(t=0, z=0).T.values - self.temp.isel(t=0, z=0).mean(dim=("x", "y")).T.values

    @property
    def temp_anom_flat(self):
        return self.temp_anom.ravel()

    @property
    def temp_anom_flat_pdf(self):

        pdf, bin_edges = np.histogram(
            self.temp_anom_flat, bins=Constants.bins, density=True
        )

        return pdf, bin_edges

    @property
    def w_flat(self):
        return self.w.isel(t=0, z=0).w.values.ravel()


class KPlane0947:

    def __init__(self) -> None:

        self.temp = xr.open_dataset(Path("../data/KPlane0947Temperature160000.nc"))
        self.w = xr.open_dataset(Path("../data/KPlane0947VerticalVelocity.nc"))

    @property
    def temp_anom(self):
        return self.temp.isel(t=0, z=0).T.values - self.temp.isel(t=0, z=0).mean(dim=("x", "y")).T.values

    @property
    def temp_anom_flat(self):
        return self.temp_anom.ravel()

    @property
    def temp_anom_flat_pdf(self):

        pdf, bin_edges = np.histogram(
            self.temp_anom_flat, bins=Constants.bins, density=True
        )

        return pdf, bin_edges

    @property
    def w_flat(self):
        return self.w.isel(t=0, z=0).w.values.ravel()
