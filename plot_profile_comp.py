import numpy as np
import matplotlib.pyplot as plt
import RF_Track as rft
from matplotlib.gridspec import GridSpec

import pybdsim
import pygpt

import bdsim2rftrack

def get_coords_6DT(bunch):
    ps = bunch.get_phase_space()
    return {
        "x": ps[:,0],
        "p_x": ps[:,1],
        "y": ps[:,2],
        "p_y": ps[:,3],
        "z": ps[:,4],
        "p_z": ps[:,5],
    }

# Define the reference particle properties
mass = 938.2720885731878 #con.physical_constants["proton mass energy equivalent in MeV"][0]
Ek_ref = 15 # MeV
G_ref = 1 + (Ek_ref/mass)
B_ref = np.sqrt(1 - (1/G_ref**2))
V_ref = B_ref * rft.clight
p_ref = np.sqrt((mass+15)**2 - mass**2)

# RF_Track
bunch_from_file = bdsim2rftrack.bdsim_userfile_to_rftrack_bunch6DT(
    filename="Beams/LhARA_10cm_pm2-SC-rftrack.dat",
    particle_mass=mass,
    particle_charge=1,
    bunch_charge=1e9
)

bunch = get_coords_6DT(bunch_from_file)

x_rftrack_SC = bunch["x"]
y_rftrack_SC = bunch["y"]
px = bunch["p_x"]
py = bunch["p_y"]
norm_px_rftrack_SC = px / p_ref
norm_py_rftrack_SC = py / p_ref

bunch_from_file = bdsim2rftrack.bdsim_userfile_to_rftrack_bunch6DT(
    filename="Beams/LhARA_10cm_pm2-rftrack.dat",
    particle_mass=mass,
    particle_charge=1,
    bunch_charge=1e9
)

bunch = get_coords_6DT(bunch_from_file)

x_rftrack = bunch["x"]
y_rftrack = bunch["y"]
px = bunch["p_x"]
py = bunch["p_y"]
norm_px_rftrack = px / p_ref
norm_py_rftrack = py / p_ref


# GPT
gpt_data = pygpt.Reader.LoadGptData("../GPT/IdealTNSA/pm2/Source/LhARA_10cm_pm2-SC.txt").times[0]
x_gpt = gpt_data.GetColumn('x')*1e3
y_gpt = gpt_data.GetColumn('y')*1e3
px_gpt = gpt_data.GetAbsolutexp()
py_gpt = gpt_data.GetAbsoluteyp()
norm_px_gpt = px_gpt / p_ref
norm_py_gpt = py_gpt / p_ref

# Organize data into dictionaries
rftrack_data = {
    "x": x_rftrack,
    "y": y_rftrack,
    "px": norm_px_rftrack,
    "py": norm_py_rftrack,
}

rftrack_data_SC = {
    "x": x_rftrack_SC,
    "y": y_rftrack_SC,
    "px": norm_px_rftrack_SC,
    "py": norm_py_rftrack_SC,
}

gpt_data_dict = {
    "x": x_gpt,
    "y": y_gpt,
    "px": norm_px_gpt,
    "py": norm_py_gpt,
}

# Plotting
def plot_histogram_comparison(rftrack_data, gpt_data, output_pdf="histogram_comparison.pdf",
                              nbins=50, alpha=0.6, figsize=(12, 10)):
    """
    Plot side-by-side histograms comparing RF_Track and GPT distributions.

    Parameters
    ----------
    rftrack_data : dict
        Dictionary with keys 'x', 'y', 'px', 'py' (RF_Track data in mm and normalized momentum)
    gpt_data : dict
        Dictionary with keys 'x', 'y', 'px', 'py' (GPT data in mm and normalized momentum)
    output_pdf : str
        Path to save the output PDF
    nbins : int
        Number of bins for histograms
    alpha : float
        Transparency for histogram bars
    figsize : tuple
        Figure size (width, height)
    """

    fig = plt.figure(figsize=figsize)
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

    # Define the variables to plot
    vars_to_plot = [
        ('x', 'x [mm]', 'RF_Track x'),
        ('y', 'y [mm]', 'RF_Track y'),
        ('px', 'px / p_ref', 'RF_Track px'),
        ('py', 'py / p_ref', 'RF_Track py'),
    ]

    # Create subplots
    for idx, (key, label, title) in enumerate(vars_to_plot):
        ax = fig.add_subplot(gs[idx // 2, idx % 2])

        # Get data
        rf_vals = rftrack_data[key]
        gpt_vals = gpt_data[key]

        # Calculate common range for better comparison
        all_vals = np.concatenate([rf_vals, gpt_vals])
        vmin, vmax = np.percentile(all_vals, [0.5, 99.5])

        # Plot histograms
        ax.hist(rf_vals, bins=nbins, range=(vmin, vmax), alpha=alpha,
                label='RF_Track', color='blue', histtype="step", log=True)
        ax.hist(gpt_vals, bins=nbins, range=(vmin, vmax), alpha=alpha,
                label='GPT', color='red', histtype='step', log=True)

        # Labels and formatting
        ax.set_xlabel(label, fontsize=11)
        ax.set_ylabel('Count', fontsize=11)
        ax.legend(fontsize=10, loc='upper left')
        ax.grid(True, alpha=0.3)

        # Add statistics
        rf_mean, rf_std = np.mean(rf_vals), np.std(rf_vals)
        gpt_mean, gpt_std = np.mean(gpt_vals), np.std(gpt_vals)

        stats_text = (f'RF: σ={rf_std:.4f}\n'
                      f'GPT: σ={gpt_std:.4f}')
        ax.text(0.98, 0.97, stats_text, transform=ax.transAxes,
                fontsize=9, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.savefig(output_pdf, dpi=150, bbox_inches='tight')
    print(f"Saved histogram comparison to {output_pdf}")
    plt.close()

def plot_histogram_comparison_SC(rftrack_data, rftrack_data_SC, gpt_data, output_pdf="histogram_comparison.pdf",
                              nbins=50, alpha=0.6, figsize=(12, 10)):
    """
    Plot side-by-side histograms comparing RF_Track and GPT distributions.

    Parameters
    ----------
    rftrack_data : dict
        Dictionary with keys 'x', 'y', 'px', 'py' (RF_Track data in mm and normalized momentum)
    gpt_data : dict
        Dictionary with keys 'x', 'y', 'px', 'py' (GPT data in mm and normalized momentum)
    output_pdf : str
        Path to save the output PDF
    nbins : int
        Number of bins for histograms
    alpha : float
        Transparency for histogram bars
    figsize : tuple
        Figure size (width, height)
    """

    fig = plt.figure(figsize=figsize)
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

    # Define the variables to plot
    vars_to_plot = [
        ('x', 'x [mm]', 'RF_Track x'),
        ('y', 'y [mm]', 'RF_Track y'),
        ('px', 'px / p_ref', 'RF_Track px'),
        ('py', 'py / p_ref', 'RF_Track py'),
    ]

    # Create subplots
    for idx, (key, label, title) in enumerate(vars_to_plot):
        ax = fig.add_subplot(gs[idx // 2, idx % 2])

        # Get data
        rf_vals = rftrack_data[key]
        rf_vals_SC = rftrack_data_SC[key]
        gpt_vals = gpt_data[key]

        # Calculate common range for better comparison
        all_vals = np.concatenate([rf_vals, rf_vals_SC, gpt_vals])
        vmin, vmax = np.percentile(all_vals, [0.5, 99.5])

        # Plot histograms
        ax.hist(rf_vals, bins=nbins, range=(vmin, vmax), alpha=alpha,
                label='RF_Track', color='black', histtype="step", log=True)
        ax.hist(rf_vals_SC, bins=nbins, range=(vmin, vmax), alpha=alpha,
                label='RF_Track_SC', color='blue', histtype="step", log=True)
        ax.hist(gpt_vals, bins=nbins, range=(vmin, vmax), alpha=alpha,
                label='GPT', color='red', histtype='step', log=True)

        # Labels and formatting
        ax.set_xlabel(label, fontsize=11)
        ax.set_ylabel('Count', fontsize=11)
        ax.legend(fontsize=10, loc='upper left')
        ax.grid(True, alpha=0.3)

        # Add statistics
        rf_mean, rf_std = np.mean(rf_vals), np.std(rf_vals)
        gpt_mean, gpt_std = np.mean(gpt_vals), np.std(gpt_vals)

        stats_text = (f'RF: σ={rf_std:.4f}\n'
                      f'GPT: σ={gpt_std:.4f}')
        ax.text(0.98, 0.97, stats_text, transform=ax.transAxes,
                fontsize=9, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.savefig(output_pdf, dpi=150, bbox_inches='tight')
    print(f"Saved histogram comparison to {output_pdf}")
    plt.close()

def plot_histogram_difference(rftrack_data, gpt_data, output_pdf="histogram_difference.pdf",
                              nbins=50, figsize=(12, 10)):
    """
    Plot signed histogram differences: RF_Track bin counts minus GPT bin counts.

    Parameters
    ----------
    rftrack_data : dict
        Keys: 'x', 'y', 'px', 'py'
    gpt_data : dict
        Keys: 'x', 'y', 'px', 'py'
    output_pdf : str
        Output PDF path.
    nbins : int
        Number of bins.
    figsize : tuple
        Figure size.
    """
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

    vars_to_plot = [
        ('x', 'x [mm]'),
        ('y', 'y [mm]'),
        ('px', 'p_x / p_ref'),
        ('py', 'p_y / p_ref'),
    ]

    for idx, (key, label) in enumerate(vars_to_plot):
        ax = fig.add_subplot(gs[idx // 2, idx % 2])

        rf_vals = rftrack_data[key]
        gpt_vals = gpt_data[key]

        # Common binning
        all_vals = np.concatenate([rf_vals, gpt_vals])
        vmin, vmax = np.percentile(all_vals, [0.5, 99.5])
        bins = np.linspace(vmin, vmax, nbins + 1)

        rf_counts, bin_edges = np.histogram(rf_vals, bins=bins)
        gpt_counts, _ = np.histogram(gpt_vals, bins=bins)

        diff_counts = (rf_counts - gpt_counts)/rf_counts * 100
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        bin_widths = np.diff(bin_edges)

        ax.bar(
            bin_centers,
            diff_counts,
            width=bin_widths,
            align='center',
            color='purple',
            edgecolor='black',
            linewidth=0.5
        )

        ax.axhline(0, color='k', linewidth=1)
        ax.set_xlabel(label, fontsize=11)
        ax.set_ylabel('Relative % Difference (RFT - GPT)', fontsize=11)
        ax.grid(True, alpha=0.3)

    plt.savefig(output_pdf, dpi=150, bbox_inches='tight')
    print(f"Saved histogram difference plot to {output_pdf}")
    plt.close()


# Plot comparisons
plot_histogram_comparison(
    rftrack_data_SC,
    gpt_data_dict,
    output_pdf="plots/histogram_comparison.pdf"
)

plot_histogram_difference(
    rftrack_data_SC,
    gpt_data_dict,
    output_pdf="plots/histogram_differenceSC.pdf"
)


# Plot comparisons
plot_histogram_comparison_SC(
    rftrack_data,
    rftrack_data_SC,
    gpt_data_dict,
    output_pdf="plots/histogram_comparison_SC.pdf"
)