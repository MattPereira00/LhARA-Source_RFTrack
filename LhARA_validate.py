#!/usr/bin/env python3
"""
LhARA tracking validation: RF_Track vs GPT residuals and BDSIM phase-space plots.

Loads the intermediate and final beam files produced by LhARA_source_to_nozzle.py
and compares them against GPT reference data at each pipeline stage.

Stages validated:
  A — Source    (0 cm):  picosecond RF_Track step vs GPT at source
  B — Entrance  (5 cm, pre-cut):  RF_Track vs GPT at 5 cm
  C — Entrance  (5 cm, post-cut + picostep): RF_Track vs GPT
  D — Exit      (10 cm, with SC): RF_Track vs GPT at 10 cm

Optionally runs BDSIM for phase-space comparisons at each stage (--bdsim).

Usage:
    python LhARA_validate.py
    python LhARA_validate.py --label LhARA_0cm_pm2 --gpt-dir ../GPT/IdealTNSA/pm2/Source
    python LhARA_validate.py --bdsim
"""
import argparse
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import RF_Track as rft

import bdsim2rftrack
from plot_residuals import plot_tracking_residuals


# ── Reference particle ────────────────────────────────────────────────────────
PROTON_MASS  = 938.2720885731878   # MeV/c^2
EK_REF       = 15.0                # MeV
BUNCH_CHARGE = 1e9

# ── Geometry ──────────────────────────────────────────────────────────────────
SOURCE_TO_ENTRANCE = 0.05   # m
NOZZLE_LENGTH      = 0.05   # m
CUT_RADIUS_5CM     = 0.002
CUT_RADIUS_10CM    = 0.00287

# ── Space-charge settings (must match pipeline) ───────────────────────────────
SC_GRID  = (50, 50, 125)
SC_DT_MM = 0.3   # mm/c


# ─────────────────────────────────────────────────────────────────────────────

def _ref_particle(mass, Ek):
    G = 1.0 + Ek / mass
    B = np.sqrt(1.0 - 1.0 / G**2)
    V = B * rft.clight
    p = np.sqrt((mass + Ek)**2 - mass**2)
    return G, B, V, p


def _load_bunch(path, mass=PROTON_MASS, charge=1, bunch_charge=BUNCH_CHARGE):
    return bdsim2rftrack.bdsim_userfile_to_rftrack_bunch6DT(
        filename=path, particle_mass=mass,
        particle_charge=charge, bunch_charge=bunch_charge,
    )


def _bunch_to_dict(bunch, p_ref):
    ps = bunch.get_phase_space()
    return {
        "x":  ps[:, 0] * 1e-3,          # mm -> m
        "y":  ps[:, 2] * 1e-3,
        "px": ps[:, 1] / p_ref,
        "py": ps[:, 3] / p_ref,
    }


def _load_gpt(path, p_ref):
    import pygpt
    data = pygpt.Reader.LoadGptData(path).times[0]
    return {
        "x":  data.GetColumn('x'),
        "y":  data.GetColumn('y'),
        "px": data.GetAbsolutexp() / p_ref,
        "py": data.GetAbsoluteyp() / p_ref,
    }


def _track_drift(bunch, length_m, t_max_mm, *, space_charge=False, sc_dt_mm=None):
    if space_charge:
        SC = rft.SpaceCharge_PIC_FreeSpace(*SC_GRID)
        rft.cvar.SC_engine = SC
    drift = rft.Drift(length=length_m)
    line  = rft.Lattice()
    line.append(drift)
    world = rft.Volume()
    world.add(line, 0, 0, 0, 'entrance')
    world.odeint_algorithm = "rk4"
    world.t_max_mm = t_max_mm
    if space_charge and sc_dt_mm is not None:
        world.sc_dt_mm = sc_dt_mm
        world.dt_mm    = sc_dt_mm
    return world.track(bunch)


def _run_bdsim(gmad, outfile, n, optics=True, plots_prefix=None):
    import pybdsim, pygpt
    print(f"    Running BDSIM: {gmad} ...")
    pybdsim.Run.Bdsim(gmadpath=gmad, outfile=outfile, ngenerate=n, silent=True)
    if optics:
        pybdsim.Run.RebdsimOptics(
            rootpath=outfile + ".root",
            outpath=outfile + "-optics.root",
            silent=True,
        )
    if plots_prefix is not None:
        pygpt.Plot.Phasespace.BDSIMPhaseSpace(
            filename=outfile + ".root",
            outputfilename=plots_prefix,
            coordsTitle=" ",
            correlationTitle=" ",
        )
        plt.close("all")


def plot_histogram_comparison(datasets, labels, colors, out_pdf, nbins=50):
    """Overlaid histogram comparison for x, y, px/p_ref, py/p_ref."""
    keys   = ["x",          "y",          "px",              "py"]
    xlbls  = ["$x$ [m]",   "$y$ [m]",   "$p_x / p_{ref}$", "$p_y / p_{ref}$"]

    fig = plt.figure(figsize=(12, 10))
    gs  = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

    for idx, (key, xlabel) in enumerate(zip(keys, xlbls)):
        ax = fig.add_subplot(gs[idx // 2, idx % 2])
        all_vals = np.concatenate([d[key] for d in datasets])
        vmin, vmax = np.percentile(all_vals, [0.5, 99.5])

        for d, label, color in zip(datasets, labels, colors):
            ax.hist(d[key], bins=nbins, range=(vmin, vmax),
                    histtype="step", log=True, label=label, color=color)

        ax.set_xlabel(xlabel)
        ax.set_ylabel("Count")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        stats = "\n".join(f"{lbl}: σ={np.std(d[key]):.4f}"
                          for d, lbl in zip(datasets, labels))
        ax.text(0.98, 0.97, stats, transform=ax.transAxes, fontsize=8,
                va="top", ha="right",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    fig.suptitle(" vs ".join(labels))
    plt.tight_layout()
    plt.savefig(out_pdf, dpi=150, bbox_inches="tight")
    plt.close("all")
    print(f"    Saved: {out_pdf}")


def plot_histogram_difference(rf_dict, gpt_dict, out_pdf, nbins=50):
    """Signed relative % difference (RF_Track - GPT) per bin."""
    keys  = ["x",         "y",         "px",              "py"]
    xlbls = ["$x$ [m]",  "$y$ [m]",  "$p_x / p_{ref}$", "$p_y / p_{ref}$"]

    fig = plt.figure(figsize=(12, 10))
    gs  = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

    for idx, (key, xlabel) in enumerate(zip(keys, xlbls)):
        ax = fig.add_subplot(gs[idx // 2, idx % 2])

        all_vals = np.concatenate([rf_dict[key], gpt_dict[key]])
        vmin, vmax = np.percentile(all_vals, [0.5, 99.5])
        bins = np.linspace(vmin, vmax, nbins + 1)

        rf_counts,  edges = np.histogram(rf_dict[key],  bins=bins)
        gpt_counts, _     = np.histogram(gpt_dict[key], bins=bins)

        with np.errstate(divide="ignore", invalid="ignore"):
            diff = np.where(rf_counts > 0,
                            (rf_counts - gpt_counts) / rf_counts * 100, 0.0)

        centers = 0.5 * (edges[:-1] + edges[1:])
        ax.bar(centers, diff, width=np.diff(edges), align="center",
               color="purple", edgecolor="black", linewidth=0.5)
        ax.axhline(0, color="k", linewidth=1)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Relative % difference (RFT − GPT)")
        ax.grid(True, alpha=0.3)

    fig.suptitle("RF_Track − GPT (relative %)")
    plt.tight_layout()
    plt.savefig(out_pdf, dpi=150, bbox_inches="tight")
    plt.close("all")
    print(f"    Saved: {out_pdf}")


# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Validate LhARA tracking: RF_Track vs GPT at each pipeline stage"
    )
    parser.add_argument("--label",   default="LhARA_0cm_pm2",
                        help="Filename stem used by LhARA_source_to_nozzle.py (default: LhARA_0cm_pm2)")
    parser.add_argument("--beamdir", default="Beams",
                        help="Directory containing RF_Track beam files (default: Beams/)")
    parser.add_argument("--plotdir", default="plots",
                        help="Directory for output plots (default: plots/)")
    parser.add_argument("--gpt-dir", default="../GPT/IdealTNSA/pm2/Source",
                        help="Directory containing GPT reference files")

    # Optional per-stage GPT filename overrides
    parser.add_argument("--gpt-0cm",      default=None, help="GPT file at source (0 cm)")
    parser.add_argument("--gpt-5cm",      default=None, help="GPT file at nozzle entrance (5 cm)")
    parser.add_argument("--gpt-5cm-step", default=None, help="GPT file for picostep at 5 cm")
    parser.add_argument("--gpt-10cm-sc",  default=None, help="GPT file at nozzle exit (10 cm, SC)")

    parser.add_argument("--bdsim", action="store_true",
                        help="Run BDSIM phase-space comparisons at each stage")
    parser.add_argument("--no-residuals", action="store_true",
                        help="Skip RF_Track vs GPT residual plots")
    args = parser.parse_args()

    os.makedirs(args.plotdir, exist_ok=True)

    _, _, V_ref, p_ref = _ref_particle(PROTON_MASS, EK_REF)

    def gpt_path(default_name, override):
        return override if override else os.path.join(args.gpt_dir, default_name)

    def beam_path(suffix):
        return os.path.join(args.beamdir, f"{args.label}{suffix}")

    # Derive a short beam label without the leading path name e.g. "0cm_pm2"
    # for use in output plot filenames
    plot_stem = args.label

    # ── Stage A: source (0 cm) picostep ──────────────────────────────────────
    print("\n[A] Source (0 cm) — picostep validation ...")
    src_file = beam_path("-bdsimin.dat")
    if not os.path.exists(src_file):
        print(f"    WARNING: {src_file} not found, skipping stage A.")
    else:
        bunch_src = _load_bunch(src_file)
        t_pico    = 1e-12 * rft.clight * 1e3   # 1 ps in mm/c
        bunch_0cm = _track_drift(bunch_src, SOURCE_TO_ENTRANCE, t_pico)
        rf_0cm    = _bunch_to_dict(bunch_0cm, p_ref)

        gpt_file_0cm = gpt_path(f"LhARA_0cm_pm2.txt", args.gpt_0cm)
        if not os.path.exists(gpt_file_0cm):
            print(f"    WARNING: GPT file not found ({gpt_file_0cm}), skipping residuals.")
        elif not args.no_residuals:
            gpt_0cm = _load_gpt(gpt_file_0cm, p_ref)
            plot_tracking_residuals(
                ref=rf_0cm, other=gpt_0cm,
                out_pdf=os.path.join(args.plotdir, f"{plot_stem}_Residual_RF_GPT_0cm.pdf"),
                ref_name="RFTrack", other_name="GPT",
            )
            print(f"    Saved residual plot.")

        if args.bdsim:
            _run_bdsim(
                gmad="BDSIM/0cmprim.gmad",
                outfile=beam_path("-bdsim").replace(args.beamdir, args.beamdir),
                n=len(rf_0cm["x"]),
                plots_prefix=os.path.join(args.plotdir, f"{plot_stem}_BDSPS_0cm"),
            )

    # ── Stage B: nozzle entrance (5 cm, pre-cut) ─────────────────────────────
    print("\n[B] Nozzle entrance (5 cm, pre-cut) ...")
    path_5cm = beam_path("-5cm-rftrack.dat")
    if not os.path.exists(path_5cm):
        print(f"    WARNING: {path_5cm} not found — run LhARA_source_to_nozzle.py first.")
    else:
        bunch_5cm = _load_bunch(path_5cm)
        rf_5cm    = _bunch_to_dict(bunch_5cm, p_ref)

        gpt_file_5cm = gpt_path("LhARA_5cm_pm2.txt", args.gpt_5cm)
        if not os.path.exists(gpt_file_5cm):
            print(f"    WARNING: GPT file not found ({gpt_file_5cm}), skipping residuals.")
        elif not args.no_residuals:
            gpt_5cm = _load_gpt(gpt_file_5cm, p_ref)
            plot_tracking_residuals(
                ref=rf_5cm, other=gpt_5cm,
                out_pdf=os.path.join(args.plotdir, f"{plot_stem}_Residual_RF_GPT_5cm.pdf"),
                ref_name="RFTrack", other_name="GPT",
            )
            print(f"    Saved residual plot.")

        if args.bdsim:
            _run_bdsim(
                gmad="BDSIM/5cmprim.gmad",
                outfile=os.path.join(args.beamdir, f"{plot_stem}-5cm-bdsim"),
                n=len(rf_5cm["x"]),
                plots_prefix=os.path.join(args.plotdir, f"{plot_stem}_BDSPS_5cm"),
            )

    # ── Stage C: nozzle entrance post-cut + picostep ─────────────────────────
    print("\n[C] Nozzle entrance (5 cm, post-cut + picostep) ...")
    path_5cm_cut = beam_path("-5cm-cut-rftrack.dat")
    if not os.path.exists(path_5cm_cut):
        print(f"    WARNING: {path_5cm_cut} not found, skipping stage C.")
    else:
        bunch_5cm_cut = _load_bunch(path_5cm_cut)
        t_pico        = 1e-12 * rft.clight * 1e3
        bunch_5cm_step = _track_drift(bunch_5cm_cut, SOURCE_TO_ENTRANCE, t_pico)
        rf_5cm_step    = _bunch_to_dict(bunch_5cm_step, p_ref)

        gpt_file_5cm_step = gpt_path("LhARA_5cm_pm2-step.txt", args.gpt_5cm_step)
        if not os.path.exists(gpt_file_5cm_step):
            print(f"    WARNING: GPT file not found ({gpt_file_5cm_step}), skipping residuals.")
        elif not args.no_residuals:
            gpt_5cm_step = _load_gpt(gpt_file_5cm_step, p_ref)
            plot_tracking_residuals(
                ref=rf_5cm_step, other=gpt_5cm_step,
                out_pdf=os.path.join(args.plotdir, f"{plot_stem}_Residual_RF_GPT_5cm_step.pdf"),
                ref_name="RFTrack", other_name="GPT",
            )
            print(f"    Saved residual plot.")

        if args.bdsim:
            _run_bdsim(
                gmad="BDSIM/5cmprimcut.gmad",
                outfile=os.path.join(args.beamdir, f"{plot_stem}-5cm-cut-bdsim"),
                n=len(rf_5cm_step["x"]),
                plots_prefix=os.path.join(args.plotdir, f"{plot_stem}_BDSPS_5cm_cut"),
            )

    # ── Stage D: nozzle exit (10 cm, with SC) ────────────────────────────────
    print("\n[D] Nozzle exit (10 cm, space charge) ...")
    path_10cm = beam_path("-10cm-rftrack.dat")
    if not os.path.exists(path_10cm):
        print(f"    WARNING: {path_10cm} not found — run LhARA_source_to_nozzle.py first.")
    else:
        bunch_10cm = _load_bunch(path_10cm)
        rf_10cm    = _bunch_to_dict(bunch_10cm, p_ref)

        gpt_file_10cm = gpt_path("LhARA_10cm_pm2-SC.txt", args.gpt_10cm_sc)
        if not os.path.exists(gpt_file_10cm):
            print(f"    WARNING: GPT file not found ({gpt_file_10cm}), skipping residuals.")
        elif not args.no_residuals:
            gpt_10cm = _load_gpt(gpt_file_10cm, p_ref)
            plot_tracking_residuals(
                ref=rf_10cm, other=gpt_10cm,
                out_pdf=os.path.join(args.plotdir, f"{plot_stem}_Residual_RF_GPT_10cm_SC.pdf"),
                ref_name="RFTrack", other_name="GPT",
            )
            plot_histogram_comparison(
                datasets=[rf_10cm, gpt_10cm],
                labels=["RFTrack (SC)", "GPT"],
                colors=["blue", "red"],
                out_pdf=os.path.join(args.plotdir, f"{plot_stem}_histogram_10cm_SC.pdf"),
            )
            plot_histogram_difference(
                rf_dict=rf_10cm, gpt_dict=gpt_10cm,
                out_pdf=os.path.join(args.plotdir, f"{plot_stem}_histogram_diff_10cm_SC.pdf"),
            )
            print(f"    Saved residual and histogram plots.")

        if args.bdsim:
            _run_bdsim(
                gmad="BDSIM/10cmprim.gmad",
                outfile=os.path.join(args.beamdir, f"{plot_stem}-10cm-bdsim"),
                n=len(rf_10cm["x"]),
                plots_prefix=os.path.join(args.plotdir, f"{plot_stem}_BDSPS_10cm"),
            )

    # ── Final cut: BDSIM phase space at nozzle exit ───────────────────────────
    path_10cm_cut = beam_path("-10cm-cut-rftrack.dat")
    if args.bdsim and os.path.exists(path_10cm_cut):
        print("\n[E] BDSIM phase space for final cut beam at 10 cm ...")
        bunch_final = _load_bunch(path_10cm_cut)
        ps = bunch_final.get_phase_space()
        _run_bdsim(
            gmad="BDSIM/10cmprimcut.gmad",
            outfile=os.path.join(args.beamdir, f"{plot_stem}-10cm-cut-bdsim"),
            n=len(ps),
            plots_prefix=os.path.join(args.plotdir, f"{plot_stem}_BDSPS_10cm_cut"),
        )

    print("\n── Validation complete ────────────────────────────────────────────")


if __name__ == "__main__":
    main()
