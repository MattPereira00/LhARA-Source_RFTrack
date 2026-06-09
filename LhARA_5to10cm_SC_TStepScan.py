import numpy as np
import matplotlib.pyplot as plt
import RF_Track as rft
import pygpt
import pybdsim

import bdsim2rftrack
from plot_residuals import plot_tracking_residuals

import sys

np.set_printoptions(precision=14)


def get_coords_6DT(bunch):
    ps = bunch.get_phase_space()
    return {
        "x": ps[:, 0],
        "p_x": ps[:, 1],
        "y": ps[:, 2],
        "p_y": ps[:, 3],
        "z": ps[:, 4],
        "p_z": ps[:, 5],
    }


# Define the reference particle properties
mass = 938.2720885731878
Ek_ref = 15  # MeV
G_ref = 1 + (Ek_ref / mass)
B_ref = np.sqrt(1 - (1 / G_ref ** 2))
V_ref = B_ref * rft.clight
p_ref = np.sqrt((mass + 15) ** 2 - mass ** 2)

# Space Charge
SC = rft.SpaceCharge_PIC_FreeSpace(50, 50, 125)
rft.cvar.SC_engine = SC

# Build the beam from BDSIM userfile (done once, reused for each iteration)
bunch_nozzle_entrance = bdsim2rftrack.bdsim_userfile_to_rftrack_bunch6DT(
    filename="Beams/LhARA_5cm_pm2-cut-rftrack.dat",
    particle_mass=mass,
    particle_charge=1,
    bunch_charge=1e9
)

# Load GPT reference data (done once, reused for each iteration)
gpt_data = pygpt.Reader.LoadGptData("../GPT/IdealTNSA/pm2/Source/LhARA_10cm_pm2-SC.txt").times[0]
x_gpt = gpt_data.GetColumn('x')
y_gpt = gpt_data.GetColumn('y')
px_gpt = gpt_data.GetAbsolutexp()
py_gpt = gpt_data.GetAbsoluteyp()
norm_px_gpt = px_gpt / p_ref
norm_py_gpt = py_gpt / p_ref

gpt = {
    "x": x_gpt,
    "y": y_gpt,
    "px": norm_px_gpt,
    "py": norm_py_gpt,
}

# Define sc_dt_mm values to scan: from 1mm (1e-1) down to 1μm (1e-6)
# Using logarithmic spacing for better coverage
sc_dt_values = np.logspace(-1, -6, num=10)  # Adjust num for more/fewer points

results = {}

for sc_dt_mm in sc_dt_values:
    print(f"\nTracking with sc_dt_mm = {sc_dt_mm:.2e}")

    # Create fresh bunch for this iteration
    bunch_entrance = bdsim2rftrack.bdsim_userfile_to_rftrack_bunch6DT(
        filename="Beams/LhARA_5cm_pm2-cut-rftrack.dat",
        particle_mass=mass,
        particle_charge=1,
        bunch_charge=1e9
    )

    # Elements
    nozzle = rft.Drift(length=0.05)

    # Lattice
    line = rft.Lattice()
    line.append(nozzle)

    # Volume
    world = rft.Volume()
    world.add(line, 0, 0, 0, 'entrance')

    # Tracking Options
    world.odeint_algorithm = "rk4"
    world.t_max_mm = (0.05 / V_ref) * rft.clight * 1e3
    world.sc_dt_mm = sc_dt_mm

    # Tracking
    ps_init = get_coords_6DT(bunch_entrance)
    bunch_exit = world.track(bunch_entrance)
    ps_tracked = get_coords_6DT(bunch_exit)

    # RF_Track results
    x_rftrack = ps_tracked["x"] * 1e-3
    y_rftrack = ps_tracked["y"] * 1e-3
    px = ps_tracked["p_x"]
    py = ps_tracked["p_y"]
    norm_px_rftrack = px / p_ref
    norm_py_rftrack = py / p_ref

    rf = {
        "x": x_rftrack,
        "y": y_rftrack,
        "px": norm_px_rftrack,
        "py": norm_py_rftrack,
    }

    # Store results
    results[sc_dt_mm] = rf

    # Generate output filename with sc_dt_mm value
    # Format: 0.1mm, 0.01mm, 0.001mm, 1e-4mm, 1e-5mm, 1e-6mm
    formatted_value = f"{sc_dt_mm:.0e}".replace("+", "").replace("-0", "-").replace("-", "nm")
    out_pdf = f"plots/sc_dt_scan/LhARA_5to10cm_pm2_Residual_RF_GPT_sc_dt_{formatted_value}.pdf"

    # Plot residuals for this sc_dt_mm value
    plot_tracking_residuals(
        ref=rf,
        other=gpt,
        out_pdf=out_pdf,
        ref_name="RFTrack",
        other_name="GPT",
        showPlot=False  # Set False to avoid interactive plots in loop
    )

    print(f"  Saved plot to {out_pdf}")

print("\nScanning complete!")
print(f"Generated {len(results)} plots for sc_dt_mm values: {[f'{v:.2e}' for v in sc_dt_values]}")