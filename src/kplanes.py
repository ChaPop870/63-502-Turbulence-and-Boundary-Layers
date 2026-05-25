from typing import Tuple, Any

import numpy as np
import xarray as xr
from numpy import ndarray, dtype

from config import (Constants, AreaFractionStats, RootMeanSquares)
from scipy.stats import skew
from pathlib import Path


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

        w_rms = np.sqrt(np.mean(self.w ** 2))
        T_rms = np.sqrt(np.mean(self.temp ** 2))

        return RootMeanSquares(
            w_rms,
            T_rms
        )

    @property
    def temp_skew(self):
        return skew(self.temp_anom_flat)

    @property
    def w_skew(self):
        return skew(self.w_flat)


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

    @property
    def rms(self) -> RootMeanSquares:
        w_rms = np.sqrt(np.mean(self.w ** 2))
        T_rms = np.sqrt(np.mean(self.temp ** 2))

        return RootMeanSquares(
            w_rms,
            T_rms
        )

    @property
    def temp_skew(self):
        return skew(self.temp_anom_flat)

    @property
    def w_skew(self):
        return skew(self.w_flat)


class KPlaneCombined:

    def __init__(self, temp_path_500: Path, w_path_500: Path, temp_path_947: Path, w_path_947: Path) -> None:
        self.temp_path_500 = temp_path_500
        self.w_path_500 = w_path_500
        self.temp_path_947 = temp_path_947
        self.w_path_947 = w_path_947

        self.kplane_0500 = KPlane0500(self.temp_path_500, self.w_path_500)
        self.kplane_0947 = KPlane0947(self.temp_path_947, self.w_path_947)

    @property
    def temp_anom_flat(self) -> np.ndarray:
        return np.concatenate((self.kplane_0500.temp_anom_flat, self.kplane_0947.temp_anom_flat))

    @property
    def temp_anom_flat_pdf(self) -> tuple[ndarray[tuple[Any, ...], dtype[Any]], ndarray[tuple[Any, ...], dtype[Any]]]:

        pdf, bin_edges = np.histogram(
            self.temp_anom_flat, bins=Constants.bins, density=True
        )

        return pdf, bin_edges

    @property
    def w_flat(self) -> np.ndarray:
        return np.concatenate((self.kplane_0500.w_flat, self.kplane_0947.w_flat))

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
    def temp_skew(self):
        return skew(self.temp_anom_flat)

    @property
    def w_skew(self):
        return skew(self.w_flat)
