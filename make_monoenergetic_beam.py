"""
Generate a monoenergetic test beam for SC solver comparison.

Reads an existing BDSIM user file, replaces every particle's total energy
with exactly E_ref while preserving all transverse coordinates (x, y, xp, yp)
and the longitudinal position (s). The output is saved as a new BDSIM user
file and a GPT input file, both ready to run through the 5->10 cm SC segment.

Usage:
    python make_monoenergetic_beam.py
    python make_monoenergetic_beam.py --input 10-Beams/LhARA_pm2_5cm-cut-rftrack.dat
    python make_monoenergetic_beam.py --input <file> --energy 15.0 --outdir 10-Beams
"""
import argparse
import os
import numpy as np
import bdsim2rftrack

PROTON_MASS = 938.2720885731878  # MeV/c^2
E_REF       = 15.0               # MeV kinetic energy
BUNCH_CHARGE = 1e9


def main():
    parser = argparse.ArgumentParser(
        description="Replace particle energies with a single reference value"
    )
    parser.add_argument(
        "--input", default="10-Beams/LhARA_pm2_5cm-cut-rftrack.dat",
        help="BDSIM user file to read (default: 10-Beams/LhARA_pm2_5cm-cut-rftrack.dat)"
    )
    parser.add_argument(
        "--energy", type=float, default=E_REF,
        help=f"Reference kinetic energy in MeV (default: {E_REF})"
    )
    parser.add_argument(
        "--outdir", default="10-Beams",
        help="Directory for output files (default: 10-Beams/)"
    )
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    E_total_ref = PROTON_MASS + args.energy  # MeV (total energy)

    # ── Load input file ───────────────────────────────────────────────────────
    data = np.genfromtxt(args.input, dtype=float)
    if data.ndim == 1:
        data = data[None, :]

    x   = data[:, 0]  # m
    y   = data[:, 1]  # m
    s   = data[:, 2]  # m
    xp  = data[:, 3]  # rad
    yp  = data[:, 4]  # rad
    # column 5 (E) is replaced

    N = len(x)
    print(f"Loaded {N} particles from {args.input}")
    print(f"Original energy range: {data[:, 5].min():.4f} – {data[:, 5].max():.4f} MeV")
    print(f"Setting all energies to {E_total_ref:.6f} MeV total ({args.energy} MeV kinetic)")

    E_mono = np.full(N, E_total_ref)

    # ── Save BDSIM user file ──────────────────────────────────────────────────
    stem = os.path.splitext(os.path.basename(args.input))[0]
    bdsim_out = os.path.join(args.outdir, f"{stem}-mono{int(args.energy)}MeV.dat")

    output = np.column_stack([x, y, s, xp, yp, E_mono])
    np.savetxt(bdsim_out, output)
    print(f"Saved BDSIM user file: {bdsim_out}")

    # ── Save GPT input file ───────────────────────────────────────────────────
    # GPT format: x  y  z  GBx  GBy  GBz  t  G  m[kg]  Q
    P_ref = np.sqrt(E_total_ref**2 - PROTON_MASS**2)  # MeV/c

    norm  = np.sqrt(1 + xp**2 + yp**2)
    p_z   = np.full(N, P_ref) / norm
    p_x   = xp * p_z
    p_y   = yp * p_z

    G_ref = E_total_ref / PROTON_MASS
    GBx   = p_x / PROTON_MASS
    GBy   = p_y / PROTON_MASS
    GBz   = p_z / PROTON_MASS

    eV_to_J = 1.602176634e-19
    c       = 299792458.0
    m_kg    = (PROTON_MASS * 1e6 * eV_to_J) / c**2

    t_gpt = np.zeros(N)

    gpt_out = os.path.join(args.outdir, f"{stem}-mono{int(args.energy)}MeV-gpt.txt")
    header  = "x\ty\tz\tGBx\tGBy\tGBz\tt\tG\tm\tQ"
    gpt_array = np.column_stack([
        x, y, s,
        GBx, GBy, GBz,
        t_gpt,
        np.full(N, G_ref),
        np.full(N, m_kg),
        np.full(N, 1.0),   # charge in units of e — GPT handles settotalcharge separately
    ])
    np.savetxt(gpt_out, gpt_array, header=header, comments="")
    print(f"Saved GPT input file:  {gpt_out}")

    print(f"\nNext steps:")
    print(f"  RF_Track: python LhARA_source_to_nozzle.py {bdsim_out} --stages 3,4 "
          f"--label LhARA_pm2_mono{int(args.energy)}MeV")
    print(f"  GPT:      use {gpt_out} as the input distribution for the 5->10 cm SC run")


if __name__ == "__main__":
    main()
