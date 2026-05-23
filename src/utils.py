import numpy as np

from dataclasses import dataclass


@dataclass
class AreaFractionStats:
    area_of_updrafts: int
    area_of_downdrafts: int
    updraft_area_fraction: float
    downdraft_area_fraction: float
    mean_updraft_velocity: float | np.floating
    mean_downdraft_velocity: float | np.floating
    diff_mean_updraft_downdraft: float


@dataclass
class RootMeanSquares:
    w_rms: int | float | np.floating
    T_rms: int | float | np.floating


def plot_path_maker():
    pass

