import os
import time
import numpy as np
import matplotlib.pyplot as plt
import RF_Track as rft
import bdsim2rftrack

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

def format_sc_dt_label(mm_value):
    if mm_value >= 1.0:
        return f"{mm_value:g}mm"
    return f"{mm_value * 1e3:g}um"


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

# Scan range
sc_dt_values = np.logspace(1, -1, num=10)

# Output directory
out_dir = "plots/scan_dt_Sxy_0to5cm"
os.makedirs(out_dir, exist_ok=True)

results = {
    "sc_dt_mm": [],
    "sigma_x": [],
    "sigma_y": [],
    "sigma_z": [],
    "runtime_s": [],
}

for sc_dt_mm in sc_dt_values:
    print(f"\nTracking with sc_dt_mm = {sc_dt_mm:.3e} [mm/c]")

    bunch_entrance = bdsim2rftrack.bdsim_userfile_to_rftrack_bunch6DT(
        filename="Beams/LhARA_0cm_pm2-bdsimin.dat",
        particle_mass=mass,
        particle_charge=1,
        bunch_charge=1e9
    )

    # Elements
    source_to_nozzle = rft.Drift(length=0.05)

    # Lattice
    line = rft.Lattice()
    line.append(source_to_nozzle)

    # Volume
    world = rft.Volume()
    world.add(line, 0, 0, 0, 'entrance')

    # Tracking Options
    world.odeint_algorithm = "rk4"
    world.t_max_mm = (0.05 / V_ref) * rft.clight * 1e3
    world.sc_dt_mm = sc_dt_mm
    world.dt_mm = sc_dt_mm

    # Tracking
    start = time.perf_counter()
    bunch_exit = world.track(bunch_entrance)
    runtime_s = time.perf_counter() - start
    ps_tracked = get_coords_6DT(bunch_exit)

    # Beam Sizes
    x = ps_tracked["x"] * 1e-3
    y = ps_tracked["y"] * 1e-3
    z = ps_tracked["z"] * 1e-3
    sigma_x = np.std(x)
    sigma_y = np.std(y)
    sigma_z = np.std(z)

    # append results
    results["sc_dt_mm"].append(sc_dt_mm)
    results["sigma_x"].append(sigma_x)
    results["sigma_y"].append(sigma_y)
    results["sigma_z"].append(sigma_z)
    results["runtime_s"].append(runtime_s)

    print(f"  sigma_x = {sigma_x:.6e}, sigma_y = {sigma_y:.6e}, sigma_z = {sigma_z:.6e}, runtime = {runtime_s:.3f} s")


sc_dt = np.array(results["sc_dt_mm"])
sigma_x = np.array(results["sigma_x"])
sigma_y = np.array(results["sigma_y"])
sigma_z = np.array(results["sigma_z"])
runtime_s = np.array(results["runtime_s"])

fig, ax1 = plt.subplots(figsize=(8, 5))

# left axis: sigma values
ax1.semilogx(sc_dt, sigma_x, "o-", label=r"$\sigma_x$")
ax1.semilogx(sc_dt, sigma_y, "s-", label=r"$\sigma_y$")
ax1.semilogx(sc_dt, sigma_z, "s-", label=r"$\sigma_z$")
ax1.set_xlabel(r"Space Charge Time Step [mm/c]")
ax1.set_ylabel(r"Beam size [m]")
ax1.grid(True, which="both", alpha=0.3)

# right axis: computational runtime
ax2 = ax1.twinx()
ax2.semilogx(sc_dt, runtime_s, "^-", color="tab:red", label="runtime")
ax2.set_ylabel("Computational time [s]", color="tab:red")
ax2.tick_params(axis="y", labelcolor="tab:red")

# combined legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper center")

fig.tight_layout()
fig.savefig(f"{out_dir}/sigma_runtime_vs_sc_dt.pdf")
plt.show()