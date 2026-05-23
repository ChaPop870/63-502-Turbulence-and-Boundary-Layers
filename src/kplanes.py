from typing import Tuple, Any

import numpy as np
import xarray as xr
from numpy import ndarray, dtype

from config import (Constants)
from pathlib import Path
from utils import AreaFractionStats, RootMeanSquares


class KPlane0500:

    def __init__(self, temp_path: Path, w_path: Path) -> None:

        self.temp = xr.open_dataset(temp_path)
        self.w = xr.open_dataset(w_path)

    @property
    def temp_anom(self) -> np.ndarray:
        return self.temp.isel(t=0, z=0).T.values - self.temp.isel(t=0, z=0).mean(dim=("x", "y")).T.values

    @property
    def temp_anom_flat(self) -> np.ndarray:
        return self.temp_anom.ravel()

    @property
    def temp_anom_flat_pdf(self) -> tuple[ndarray[tuple[Any, ...], dtype[Any]], ndarray[tuple[Any, ...], dtype[Any]]]:

        pdf, bin_edges = np.histogram(
            self.temp_anom_flat, bins=Constants.bins, density=True
        )

        return pdf, bin_edges

    @property
    def w_flat(self):
        return self.w.isel(t=0, z=0).w.values.ravel()

    @property
    def area_stats(self) -> AreaFractionStats:

        area_total = len(self.w_flat)

        area_of_updrafts = len(self.w_flat[self.w_flat > 0])

        area_of_downdrafts = len(self.w_flat[self.w_flat < 0])

        updraft_area_fraction = area_of_updrafts / area_total

        downdraft_area_fraction = area_of_downdrafts / area_total

        mean_updraft_velocity = np.mean(self.w_flat[self.w_flat > 0])

        mean_downdraft_velocity = np.mean(self.w_flat[self.w_flat < 0])

        diff_mean_updraft_downdraft = np.abs(mean_updraft_velocity) - np.abs(mean_downdraft_velocity)

        return AreaFractionStats(
            area_of_updrafts,
            area_of_downdrafts,
            updraft_area_fraction,
            downdraft_area_fraction,
            mean_updraft_velocity,
            mean_downdraft_velocity,
            diff_mean_updraft_downdraft,
        )

    @property
    def rms(self) -> RootMeanSquares:
        return RootMeanSquares


class KPlane0947:

    def __init__(self, temp_path: Path, w_path: Path) -> None:
        self.temp = xr.open_dataset(temp_path)
        self.w = xr.open_dataset(w_path)

    @property
    def temp_anom(self) -> np.ndarray:
        return self.temp.isel(t=0, z=0).T.values - self.temp.isel(t=0, z=0).mean(dim=("x", "y")).T.values

    @property
    def temp_anom_flat(self) -> np.ndarray:
        return self.temp_anom.ravel()

    @property
    def temp_anom_flat_pdf(self) -> tuple[ndarray[tuple[Any, ...], dtype[Any]], ndarray[tuple[Any, ...], dtype[Any]]]:

        pdf, bin_edges = np.histogram(
            self.temp_anom_flat, bins=Constants.bins, density=True
        )

        return pdf, bin_edges

    @property
    def w_flat(self) -> np.ndarray:
        return self.w.isel(t=0, z=0).w.values.ravel()

    @property
    def area_stats(self) -> AreaFractionStats:
        area_total = len(self.w_flat)

        area_of_updrafts = len(self.w_flat[self.w_flat > 0])

        area_of_downdrafts = len(self.w_flat[self.w_flat < 0])

        updraft_area_fraction = area_of_updrafts / area_total

        downdraft_area_fraction = area_of_downdrafts / area_total

        mean_updraft_velocity = np.mean(self.w_flat[self.w_flat > 0])

        mean_downdraft_velocity = np.mean(self.w_flat[self.w_flat < 0])

        diff_mean_updraft_downdraft = np.abs(mean_updraft_velocity) - np.abs(mean_downdraft_velocity)

        return AreaFractionStats(
            area_of_updrafts,
            area_of_downdrafts,
            updraft_area_fraction,
            downdraft_area_fraction,
            mean_updraft_velocity,
            mean_downdraft_velocity,
            diff_mean_updraft_downdraft,
        )
