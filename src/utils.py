import numpy as np
import matplotlib.pyplot as plt

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


def plot_joint_marginal_pdfs(fig, ax_joint, ax_top, ax_right, w_data, T_data):

    cmap = plt.get_cmap("YlOrRd").copy()
    cmap.set_under("white")

    r = np.corrcoef(w_data, T_data)[0, 1]

    joint_pdf, w_joint_edges, T_joint_edges = np.histogram2d(
        w_data,
        T_data,
        bins=500,
        density=True
    )

    w_marginal_pdf, w_marginal_edges = np.histogram(
        w_data,
        bins=500,
        density=True
    )

    T_marginal_pdf, T_marginal_edges = np.histogram(
        w_data,
        bins=500,
        density=True
    )

    w_centers = 0.5 * (w_marginal_edges[:-1] + w_marginal_edges[1:])
    T_centers = 0.5 * (T_marginal_edges[:-1] + T_marginal_edges[1:])

    pdf = ax_joint.pcolormesh(
        w_joint_edges,
        T_joint_edges,
        joint_pdf.T,
        shading='auto',
        cmap=cmap
    )

    ax_joint.text(
        0.88,
        0.97,
        rf'$r = {r:.3f}$',
        transform=ax_joint.transAxes,
        ha='left',
        va='top',
        fontsize=12,
        bbox=dict(
            facecolor='white',
            edgecolor='black',
            alpha=0.8
        )
    )

    ax_top.plot(w_centers, w_marginal_pdf)
    ax_right.plot(T_marginal_pdf, T_centers)

    ax_joint.set_xlabel(r'Vertical Velocity $w$ / m s$^{-1}$')
    ax_joint.set_ylabel(r'Temperature Anomaly $T^\prime$ / K')
    ax_joint.set_xlim(-2.5, 2.5)
    ax_joint.set_ylim(-0.15, 0.15)

    ax_top.set_ylabel(r'Probability Density $p(w)$')
    ax_top.set_title(r'Marginal PDF of $w$', fontsize=10)

    ax_right.set_xlabel(r'Probability Density $p(T\prime)$')
    ax_right.set_ylabel(r'Marginal PDF of $T^\prime$', fontsize=10, rotation=270, labelpad=20)
    ax_right.yaxis.set_label_position('right')
    ax_right.xaxis.set_label_position('top')

    ax_top.tick_params(labelbottom=False)
    ax_right.tick_params(labelleft=False, labeltop=True, top=True, bottom=False, labelbottom=False)

    # Colorbar
    fig.subplots_adjust(bottom=0.12)

    cax = fig.add_axes([0.125, 0.01, 0.60, 0.025])

    cbar = fig.colorbar(
        pdf,
        cax=cax,
        orientation='horizontal'
    )

    cbar.set_label('Joint Probability Density Function')

    fig.suptitle("kPlane0500 Joint PDF for vertical velocity and temperature anomaly with marginal PDFs", y=0.95)

    return fig, ax_joint, ax_right, ax_top

