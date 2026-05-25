import numpy as np
import matplotlib.pyplot as plt

from kplanes import (KPlane0500, KPlane0947, KPlaneCombined)


def plot_joint_marginal_pdfs(kplane_name: str, kplane: KPlane0500 | KPlane0947 | KPlaneCombined):
    w = kplane.w_flat
    T = kplane.temp_anom_flat

    fig = plt.figure(figsize=(12, 12), dpi=300)

    gs = fig.add_gridspec(
        2, 2,
        width_ratios=[4, 1],
        height_ratios=[1, 4],
        hspace=0.05,
        wspace=0.05
    )

    ax_joint = fig.add_subplot(gs[1, 0])
    ax_top = fig.add_subplot(gs[0, 0], sharex=ax_joint)
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax_joint)

    cmap = plt.get_cmap("inferno").copy()
    cmap.set_under("white")

    r = np.corrcoef(w, T)[0, 1]

    t_95 = np.percentile(T, 0.95)
    w_95 = np.percentile(w, 0.95)

    joint_pdf, w_joint_edges, T_joint_edges = np.histogram2d(
        w,
        T,
        bins=500,
        density=True
    )

    w_marginal_pdf, w_marginal_edges = np.histogram(
        w,
        bins=500,
        density=True
    )

    T_marginal_pdf, T_marginal_edges = np.histogram(
        T,
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
        cmap=cmap,
    )

    if type(kplane) == KPlaneCombined:
        ax_joint.text(
            0.75,
            0.98,
            rf'$r = {r:.3f}$' + f'\ntemp_skew = {kplane.temp_skew:.3f}' + f'\nw_skew = {kplane.w_skew:.3f}',
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
    else:
        ax_joint.text(
            0.80,
            0.98,
            rf'$r = {r:.3f}$' + f'\ntemp_skew = {kplane.temp_skew:.3f}' + f'\nw_skew = {kplane.w_skew:.3f}',
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
    ax_top.axvline(w_95, linestyle="--", label="95% percentile")
    ax_right.plot(T_marginal_pdf, T_centers)
    ax_right.axhline(t_95, linestyle="--", label="95% percentile")

    ax_joint.set_xlabel(r'Vertical Velocity $w$ / m s$^{-1}$')
    ax_joint.set_ylabel(r'Temperature Anomaly $T^\prime$ / K')
    ax_joint.set_xlim(-2.5, 2.5)
    # Change y lim depending on the type of data
    if type(kplane) == KPlane0500:
        ax_joint.set_ylim(-0.15, 0.15)
    elif type(kplane) == KPlane0947:
        ax_joint.set_ylim(-0.2, 0.2)
    elif type(kplane) == KPlaneCombined:
        ax_joint.set_ylim(-0.2, 0.2)

    ax_top.set_ylabel(r'Probability Density $p(w)$')
    ax_top.set_title(r'Marginal PDF of $w$', fontsize=10)

    ax_right.set_xlabel(r'Probability Density $p(T\prime)$')
    ax_right.set_ylabel(r'Marginal PDF of $T^\prime$', fontsize=10, rotation=270, labelpad=20)
    ax_right.yaxis.set_label_position('right')
    ax_right.xaxis.set_label_position('top')

    ax_top.tick_params(labelbottom=False)
    ax_right.tick_params(labelleft=False, labeltop=True, top=True, bottom=False, labelbottom=False)

    # Colorbar
    fig.subplots_adjust(bottom=0.15)

    cax = fig.add_axes([0.125, 0.05, 0.60, 0.025])

    cbar = fig.colorbar(
        pdf,
        cax=cax,
        orientation='horizontal'
    )

    cbar.set_label('Joint Probability Density Function')

    fig.suptitle(f"{kplane_name} Joint PDF for vertical velocity and temperature anomaly with marginal PDFs", y=0.95)
