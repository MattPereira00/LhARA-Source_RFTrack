#!/usr/bin/env python3
"""
LhARA source-to-nozzle tracking pipeline.

Tracks a proton bunch from the laser-plasma source (0 cm) to the nozzle exit
(10 cm) in four stages:

  1. Free drift:  0 cm  -> 5 cm  (no space charge)
  2. Radial cut:  2 mm  at nozzle entrance
  3. SC drift:    5 cm  -> 10 cm (space charge via PIC)
  4. Radial cut:  2.87 mm at nozzle exit

Intermediate and final beam files are written to --outdir.
The final output (BDSIM format + GPT input) is ready for downstream tracking.

Usage:
    python LhARA_source_to_nozzle.py Beams/LhARA_0cm_pm2-bdsimin.dat
    python LhARA_source_to_nozzle.py <input_file> [--outdir DIR] [--label LABEL]
"""
import argparse
import os

import numpy as np
import RF_Track as rft

import bdsim2rftrack


# ── Physical / beam constants ─────────────────────────────────────────────────
PROTON_MASS  = 938.2720885731878   # MeV/c^2
EK_REF       = 15.0                # MeV  (reference kinetic energy)
BUNCH_CHARGE = 1e9                 # elementary charges

# ── Geometry ──────────────────────────────────────────────────────────────────
SOURCE_TO_ENTRANCE = 0.05   # m  (source → nozzle entrance)
NOZZLE_LENGTH      = 0.05   # m  (nozzle entrance → nozzle exit)
CUT_RADIUS_5CM     = 0.002  # m  (2.00 mm — aperture at nozzle entrance)
CUT_RADIUS_10CM    = 0.00287  # m (2.87 mm — aperture at nozzle exit)

# ── Space-charge settings ─────────────────────────────────────────────────────
SC_GRID    = (50, 50, 125)   # PIC grid (nx, ny, nz)
SC_DT_MM   = 0.3             # mm/c  (≈ 1 ps time step)


# ─────────────────────────────────────────────────────────────────────────────

def _ref_particle(mass, Ek):
    """Return (gamma, beta, velocity [mm/c units], |p| [MeV/c]) for reference."""
    G = 1.0 + Ek / mass
    B = np.sqrt(1.0 - 1.0 / G**2)
    V = B * rft.clight          # m/s
    p = np.sqrt((mass + Ek)**2 - mass**2)
    return G, B, V, p


def _track_drift(bunch, length_m, t_max_mm, *, space_charge=False, sc_dt_mm=None):
    """Track bunch through a drift of length_m, stopping at t_max_mm."""
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


def _bunch_summary(bunch):
    ps = bunch.get_phase_space()
    x = ps[:, 0] * 1e-3   # mm -> m
    y = ps[:, 2] * 1e-3
    print(f"    N = {len(x)},  sigma_x = {np.std(x)*1e3:.4f} mm,"
          f"  sigma_y = {np.std(y)*1e3:.4f} mm")


# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="LhARA source-to-nozzle tracking: 0 cm → 10 cm"
    )
    parser.add_argument(
        "input_file",
        help="BDSIM user file at 0 cm  [x  y  s  xp  yp  E]"
    )
    parser.add_argument(
        "--outdir", default="Beams",
        help="Directory for output beam files (default: Beams/)"
    )
    parser.add_argument(
        "--label", default=None,
        help="Stem for output filenames (default: derived from input filename)"
    )
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # Build output filename stem from input if not given
    if args.label is not None:
        label = args.label
    else:
        stem = os.path.splitext(os.path.basename(args.input_file))[0]
        # Strip '-bdsimin' suffix if present so outputs read e.g. LhARA_0cm_pm2-5cm-...
        label = stem.replace("-bdsimin", "")

    _, _, V_ref, _ = _ref_particle(PROTON_MASS, EK_REF)

    # ── Stage 1: 0 cm → 5 cm, free drift ─────────────────────────────────────
    print("\n[1/4] Tracking 0 cm → 5 cm (free drift, no space charge) ...")
    bunch_source = bdsim2rftrack.bdsim_userfile_to_rftrack_bunch6DT(
        filename=args.input_file,
        particle_mass=PROTON_MASS,
        particle_charge=1,
        bunch_charge=BUNCH_CHARGE,
    )

    t_max_5cm = (SOURCE_TO_ENTRANCE / V_ref) * rft.clight * 1e3   # mm/c
    bunch_5cm = _track_drift(
        bunch_source, SOURCE_TO_ENTRANCE, t_max_5cm, space_charge=False
    )

    path_5cm = os.path.join(args.outdir, f"{label}-5cm-rftrack.dat")
    bdsim2rftrack.rftrack_bunch6DT_to_bdsim_userfile(
        bunch=bunch_5cm, particle_mass=PROTON_MASS, filename=path_5cm
    )
    _bunch_summary(bunch_5cm)
    print(f"    Saved: {path_5cm}")

    # ── Stage 2: radial cut at nozzle entrance ────────────────────────────────
    print(f"\n[2/4] Applying {CUT_RADIUS_5CM*1e3:.1f} mm radial cut at nozzle entrance ...")
    bunch_5cm_cut = bdsim2rftrack.apply_radial_cut_bunch6DT(
        bunch_5cm, cutradius=CUT_RADIUS_5CM
    )

    path_5cm_cut = os.path.join(args.outdir, f"{label}-5cm-cut-rftrack.dat")
    bdsim2rftrack.rftrack_bunch6DT_to_bdsim_userfile(
        bunch=bunch_5cm_cut, particle_mass=PROTON_MASS, filename=path_5cm_cut
    )
    _bunch_summary(bunch_5cm_cut)
    print(f"    Saved: {path_5cm_cut}")

    # ── Stage 3: 5 cm → 10 cm, drift with space charge ───────────────────────
    print("\n[3/4] Tracking 5 cm → 10 cm (space charge active) ...")
    t_max_10cm = (NOZZLE_LENGTH / V_ref) * rft.clight * 1e3   # mm/c
    bunch_10cm = _track_drift(
        bunch_5cm_cut, NOZZLE_LENGTH, t_max_10cm,
        space_charge=True, sc_dt_mm=SC_DT_MM
    )

    path_10cm = os.path.join(args.outdir, f"{label}-10cm-rftrack.dat")
    bdsim2rftrack.rftrack_bunch6DT_to_bdsim_userfile(
        bunch=bunch_10cm, particle_mass=PROTON_MASS, filename=path_10cm
    )
    _bunch_summary(bunch_10cm)
    print(f"    Saved: {path_10cm}")

    # ── Stage 4: radial cut at nozzle exit ────────────────────────────────────
    print(f"\n[4/4] Applying {CUT_RADIUS_10CM*1e3:.2f} mm radial cut at nozzle exit ...")
    bunch_final = bdsim2rftrack.apply_radial_cut_bunch6DT(
        bunch_10cm, cutradius=CUT_RADIUS_10CM
    )

    path_final = os.path.join(args.outdir, f"{label}-10cm-cut-rftrack.dat")
    bdsim2rftrack.rftrack_bunch6DT_to_bdsim_userfile(
        bunch=bunch_final, particle_mass=PROTON_MASS, filename=path_final
    )
    _bunch_summary(bunch_final)
    print(f"    Saved: {path_final}")

    # GPT input file for downstream tracking
    t_nozzle_exit = 0.10 / V_ref   # s
    path_gpt = os.path.join(args.outdir, f"{label}-10cm-cut-gptin.txt")
    bdsim2rftrack.bdsim_userfile_to_gptin(
        filename=path_final,
        particle_mass=PROTON_MASS,
        particle_charge=1,
        time_init=t_nozzle_exit,
        output_filename=path_gpt,
    )
    print(f"    Saved GPT input: {path_gpt}")

    print("\n── Pipeline complete ──────────────────────────────────────────────")
    print(f"  Final beam file : {path_final}")
    print(f"  GPT input file  : {path_gpt}")


if __name__ == "__main__":
    main()
