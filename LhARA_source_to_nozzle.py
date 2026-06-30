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

Running a subset of stages:
    Use --stages to run only some of the four stages (1=drift 0->5cm,
    2=cut at 5cm, 3=drift 5->10cm with SC, 4=cut at 10cm). <input_file> must
    then be the beam file matching the state *before* the first selected
    stage, e.g.:

        # Run only stage 3 (5cm-cut -> 10cm with SC)
        python LhARA_source_to_nozzle.py Beams/LhARA_0cm_pm2-5cm-cut-rftrack.dat \\
            --stages 3 --label LhARA_0cm_pm2
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
        "--outdir", default="10-Beams",
        help="Directory for output beam files (default: 10-Beams/)"
    )
    parser.add_argument(
        "--label", default=None,
        help="Stem for output filenames (default: derived from input filename)"
    )
    parser.add_argument(
        "--stages", default="1,2,3,4",
        help=(
            "Comma-separated list of stages to run (subset of 1,2,3,4). "
            "1=drift 0->5cm, 2=cut at 5cm, 3=drift 5->10cm with SC, "
            "4=cut at 10cm. <input_file> must match the beam state before "
            "the first selected stage. Default: 1,2,3,4 (run everything)."
        )
    )
    args = parser.parse_args()

    try:
        stages = sorted({int(s) for s in args.stages.split(",") if s.strip()})
    except ValueError:
        parser.error(f"--stages must be a comma-separated list of integers, got: {args.stages!r}")
    if not stages or any(s not in (1, 2, 3, 4) for s in stages):
        parser.error(f"--stages must only contain values from {{1,2,3,4}}, got: {stages}")

    os.makedirs(args.outdir, exist_ok=True)

    # Build output filename stem from input if not given
    if args.label is not None:
        label = args.label
    else:
        stem = os.path.splitext(os.path.basename(args.input_file))[0]
        # Strip '-bdsimin' suffix if present so outputs read e.g. LhARA_0cm_pm2-5cm-...
        label = stem.replace("-bdsimin", "")

    _, _, V_ref, _ = _ref_particle(PROTON_MASS, EK_REF)

    # Bunch carried forward through whichever stages are selected; the input
    # file must match the beam state expected before the first selected stage.
    bunch = bdsim2rftrack.bdsim_userfile_to_rftrack_bunch6DT(
        filename=args.input_file,
        particle_mass=PROTON_MASS,
        particle_charge=1,
        bunch_charge=BUNCH_CHARGE,
    )
    last_path = args.input_file

    # ── Stage 1: 0 cm → 5 cm, free drift ─────────────────────────────────────
    if 1 in stages:
        print("\n[1/4] Tracking 0 cm → 5 cm (free drift, no space charge) ...")
        t_max_5cm = (SOURCE_TO_ENTRANCE / V_ref) * rft.clight * 1e3   # mm/c
        bunch = _track_drift(bunch, SOURCE_TO_ENTRANCE, t_max_5cm, space_charge=False)

        last_path = os.path.join(args.outdir, f"{label}-5cm-rftrack.dat")
        bdsim2rftrack.rftrack_bunch6DT_to_bdsim_userfile(
            bunch=bunch, particle_mass=PROTON_MASS, filename=last_path
        )
        _bunch_summary(bunch)
        print(f"    Saved: {last_path}")

    # ── Stage 2: radial cut at nozzle entrance ────────────────────────────────
    if 2 in stages:
        print(f"\n[2/4] Applying {CUT_RADIUS_5CM*1e3:.1f} mm radial cut at nozzle entrance ...")
        bunch = bdsim2rftrack.apply_radial_cut_bunch6DT(bunch, cutradius=CUT_RADIUS_5CM)

        last_path = os.path.join(args.outdir, f"{label}-5cm-cut-rftrack.dat")
        bdsim2rftrack.rftrack_bunch6DT_to_bdsim_userfile(
            bunch=bunch, particle_mass=PROTON_MASS, filename=last_path
        )
        _bunch_summary(bunch)
        print(f"    Saved: {last_path}")

    # ── Stage 3: 5 cm → 10 cm, drift with space charge ───────────────────────
    if 3 in stages:
        print("\n[3/4] Tracking 5 cm → 10 cm (space charge active) ...")
        t_max_10cm = (NOZZLE_LENGTH / V_ref) * rft.clight * 1e3   # mm/c
        bunch = _track_drift(
            bunch, NOZZLE_LENGTH, t_max_10cm, space_charge=True, sc_dt_mm=SC_DT_MM
        )

        last_path = os.path.join(args.outdir, f"{label}-10cm-rftrack.dat")
        bdsim2rftrack.rftrack_bunch6DT_to_bdsim_userfile(
            bunch=bunch, particle_mass=PROTON_MASS, filename=last_path
        )
        _bunch_summary(bunch)
        print(f"    Saved: {last_path}")

    # ── Stage 4: radial cut at nozzle exit ────────────────────────────────────
    if 4 in stages:
        print(f"\n[4/4] Applying {CUT_RADIUS_10CM*1e3:.2f} mm radial cut at nozzle exit ...")
        bunch = bdsim2rftrack.apply_radial_cut_bunch6DT(bunch, cutradius=CUT_RADIUS_10CM)

        last_path = os.path.join(args.outdir, f"{label}-10cm-cut-rftrack.dat")
        bdsim2rftrack.rftrack_bunch6DT_to_bdsim_userfile(
            bunch=bunch, particle_mass=PROTON_MASS, filename=last_path
        )
        _bunch_summary(bunch)
        print(f"    Saved: {last_path}")

    print("\n── Pipeline complete ──────────────────────────────────────────────")
    print(f"  Stages run      : {stages}")
    print(f"  Final beam file : {last_path}")


if __name__ == "__main__":
    main()
