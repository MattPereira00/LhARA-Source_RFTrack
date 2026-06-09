from matplotlib.ticker import FormatStrFormatter, ScalarFormatter, MaxNLocator
from scipy.stats import binned_statistic as _bin
import matplotlib.pyplot as _plt
import numpy as _np


def plot_tracking_residuals(
    ref,                   # dict: x,y,px,py (already prepared/scaled)
    other,                 # dict: x,y,px,py (already prepared/scaled)
    out_pdf,
    ref_name="RFTrack",
    other_name="GPT",
    bins=20,
    showPlot=False,
):
    """Fixed-style 2x2 residual plot for x, y, p_x, p_y (preprocessed upstream)."""

    def _res(a, b):
        n = min(len(a), len(b))
        if len(a) != len(b):
            print(f"Warning: unequal lengths ({len(a)} vs {len(b)}), using n={n}")
        return a[:n] - b[:n], n

    # Residuals: ref - other (no in-function normalization)
    resx, _ = _res(ref["x"], other["x"])
    resy, _ = _res(ref["y"], other["y"])
    respx, _ = _res(ref["px"], other["px"])
    respy, _ = _res(ref["py"], other["py"])

    plot_vars = [
        (other["x"],  resx,  "$x$ [m]",   "$x$ residual [m]",   "b"),
        (other["y"],  resy,  "$y$ [m]",   "$y$ residual [m]",   "r"),
        (other["px"], respx, r"$p_{x} / p_{ref}$", r"$p_{x}$ residual",   "g"),
        (other["py"], respy, r"$p_{y} / p_{ref}$", r"$p_{y}$ residual",   "m"),
    ]

    fig = _plt.figure(figsize=(11, 8), dpi=100, facecolor="w", edgecolor="k")
    _plt.clf()

    for i, (xv_all, rv_all, xlabel, ylabel, color) in enumerate(plot_vars):
        n = min(len(xv_all), len(rv_all))
        xv = xv_all[:n]
        rv = rv_all[:n]

        means, bin_edges, _ = _bin(xv, rv, statistic="mean", bins=bins)
        stds = _bin(xv, rv, statistic=_np.std, bins=bins)[0]
        counts = _bin(xv, rv, statistic="count", bins=bins)[0]

        errs = _np.zeros_like(means)
        valid = counts > 0
        errs[valid] = stds[valid] / _np.sqrt(counts[valid])

        ax = fig.add_subplot([221, 222, 223, 224][i])
        bw = bin_edges[1] - bin_edges[0]
        ax.bar(
            bin_edges[:-1], means, 0.85 * bw, yerr=errs,
            color=color, ec=color, alpha=0.45, align="edge"
        )

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.1e"))

        # Keep your current style: x,y scientific; px,py plain decimal
        if i in (2, 3):
            sf = ScalarFormatter(useMathText=False)
            sf.set_scientific(False)
            ax.xaxis.set_major_formatter(sf)
        else:
            ax.xaxis.set_major_formatter(FormatStrFormatter("%.1e"))

    fig.suptitle(f"{ref_name} - {other_name} residuals")
    _plt.tight_layout()
    _plt.savefig(out_pdf)
    if showPlot:
        _plt.show()
    else:
        _plt.close("all")
