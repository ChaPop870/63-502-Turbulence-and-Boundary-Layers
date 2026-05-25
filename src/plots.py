import matplotlib.pyplot as plt

from config import Constants
from kplanes import (KPlane0500, KPlane0947)
from utils import plot_joint_marginal_pdfs


class Plotter:

    def __init__(self, kplane0500: KPlane0500, kplane0947: KPlane0947) -> None:
        self.kplane0500 = kplane0500
        self.kplane0947 = kplane0947

    def plot_horizontal_cross_section(self):
        """Plot horizontal cross-section of temperature anomalies and vertical velocity."""

        fig, axes = plt.subplots(2, 2, figsize=(12, 13), dpi=300)
        ax1, ax2, ax3, ax4 = axes.flatten()

        ax1.set_title('Vertical Velocity in kPlane0500')

        kplane0500_w = ax1.pcolormesh(
            self.kplane0500.w.isel(t=0, z=0).w.values,
            cmap="bwr", vmin=-2.5, vmax=2.5
        )

        fig.colorbar(
            kplane0500_w,
            ax=ax1,
            orientation='horizontal',
            label=r"Vertical Velocity / m s$^{-1}$",
            extend='both',
            fraction=0.05,
            pad=0.1
        )

        ax2.set_title('Temperature anomaly in KPlane0500')

        kplane0500_t = ax2.pcolormesh(
            self.kplane0947.temp_anom,
            cmap="bwr", vmin=-0.15, vmax=0.15
        )

        fig.colorbar(
            kplane0500_t,
            ax=ax2,
            orientation='horizontal',
            label="Temperature anomaly / K",
            extend='both',
            fraction=0.05,
            pad=0.1
        )

        ax3.set_title('Vertical Velocity in kPlane0947')

        kplane_0947_w = ax3.pcolormesh(
            self.kplane0947.w.isel(t=0, z=0).w.values,
            cmap="bwr", vmin=-2.5, vmax=2.5
        )

        fig.colorbar(
            kplane_0947_w,
            ax=ax3,
            orientation='horizontal',
            label=r"Vertical Velocity / m s$^{-1}$",
            extend='both',
            fraction=0.05,
            pad=0.1
        )

        ax4.set_title('Temperature anomaly in KPlane0947')

        kplane0947_t = ax4.pcolormesh(
            self.kplane0947.temp_anom,
            cmap="bwr", vmin=-0.15, vmax=0.15
        )

        fig.colorbar(
            kplane0947_t,
            ax=ax4,
            orientation='horizontal',
            label="Temperature anomaly / K",
            extend='both',
            fraction=0.05,
            pad=0.1
        )

        for ax in axes.flatten():
            ax.set_aspect('equal')
            ax.set_xlabel('x / m')
            ax.set_ylabel('y / m')

        plt.tight_layout()

        plt.savefig(
            Constants.plot_storage_directory / "w_T_horizontal.png"
        )

        return None

    def plot_joint_marginal_pdfs(self):

        plot_joint_marginal_pdfs("KPlane0500", self.kplane0500.w_flat, self.kplane0500.temp_anom_flat)

        plt.savefig(
            Constants.plot_storage_directory / "joint_marginal_pdfs_0500.png"
        )

        plt.close()

        plot_joint_marginal_pdfs("KPlane0947", self.kplane0947.w_flat, self.kplane0947.temp_anom_flat)

        plt.savefig(
            Constants.plot_storage_directory / "joint_marginal_pdfs_0947.png"
        )

        plt.close()

