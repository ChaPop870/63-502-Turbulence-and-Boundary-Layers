import numpy as np

from pathlib import Path


class Constants:

    bins = 100

    plot_storage_directory = Path("../plots")

    r_500_temp_path = Path("../data/KPlane0500Temperature160000.nc")
    r_500_w_path = Path("../data/KPlane0500VerticalVelocity.nc")

    r_947_temp_path = Path("../data/KPlane0947Temperature160000.nc")
    r_947_w_path = Path("../data/KPlane0947VerticalVelocity.nc")

    p_temp_path = Path("KPlane0500Temperature160000.nc")
    p_w_path = Path("KPlane0500VerticalVelocity.nc")


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

