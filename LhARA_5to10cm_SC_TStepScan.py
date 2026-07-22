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

# Wide sweep from coarse to fine to demonstrate convergence (~50 → 0.05 mm/c)
sc_dt_values = np.logspace(1.7, -1.3, num=15)

# Output directory
out_dir = "11-Plots/scan_dt_Sxy_5to10cm"
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
        filename="10-Beams/LhARA_pm2_5cm-cut-rftrack.dat",
        particle_mass=mass,
        particle_charge=1,
        bunch_charge=1e9,
    )

    nozzle = rft.Drift(length=0.05)
    line = rft.Lattice()
    line.append(nozzle)
    world = rft.Volume()
    world.add(line, 0, 0, 0, 'entrance')
    world.odeint_algorithm = "rk4"
    world.t_max_mm = (0.05 / V_ref) * rft.clight * 1e3
    world.sc_dt_mm = sc_dt_mm
    world.dt_mm = sc_dt_mm

    start = time.perf_counter()
    bunch_exit = world.track(bunch_entrance)
    runtime_s = time.perf_counter() - start
    ps_tracked = get_coords_6DT(bunch_exit)

    x = ps_tracked["x"] * 1e-3
    y = ps_tracked["y"] * 1e-3
    z = ps_tracked["z"] * 1e-3
    sigma_x = np.std(x)
    sigma_y = np.std(y)
    sigma_z = np.std(z)

    results["sc_dt_mm"].append(sc_dt_mm)
    results["sigma_x"].append(sigma_x)
    results["sigma_y"].append(sigma_y)
    results["sigma_z"].append(sigma_z)
    results["runtime_s"].append(runtime_s)

    print(f"  sigma_x = {sigma_x:.6e}, sigma_y = {sigma_y:.6e}, sigma_z = {sigma_z:.6e}, runtime = {runtime_s:.3f} s")


sc_dt     = np.array(results["sc_dt_mm"])
sigma_x   = np.array(results["sigma_x"])
sigma_y   = np.array(results["sigma_y"])
sigma_z   = np.array(results["sigma_z"])
runtime_s = np.array(results["runtime_s"])

c_mm_per_ns = 299.792458          # mm/ns
sc_dt_ns    = sc_dt / c_mm_per_ns  # convert mm/c -> ns

# ── Save numerical results to log file ───────────────────────────────────────
log_path = f"{out_dir}/scan_results.txt"
header = f"{'sc_dt_mm':>14}  {'sc_dt_ns':>14}  {'sigma_x':>14}  {'sigma_y':>14}  {'sigma_z':>14}  {'runtime_s':>12}"
with open(log_path, "w") as f:
    f.write(header + "\n")
    for i in range(len(sc_dt)):
        f.write(f"{sc_dt[i]:>14.6e}  {sc_dt_ns[i]:>14.6e}  {sigma_x[i]:>14.6e}  {sigma_y[i]:>14.6e}  "
                f"{sigma_z[i]:>14.6e}  {runtime_s[i]:>12.3f}\n")
print(f"\nSaved numerical results: {log_path}")

# ── Plot 1: sigma_x and sigma_y vs sc_dt, runtime on twin axis ───────────────
fig1, ax1 = plt.subplots(figsize=(8, 5))
ax1.semilogx(sc_dt_ns, sigma_x, "o-", label=r"$\sigma_x$")
ax1.semilogx(sc_dt_ns, sigma_y, "s-", label=r"$\sigma_y$")
ax1.set_xlabel(r"Space Charge Time Step [ns]")
ax1.set_ylabel(r"Beam size [m]")
ax1.grid(True, which="both", alpha=0.3)

ax1b = ax1.twinx()
ax1b.semilogx(sc_dt_ns, runtime_s, "^-", color="tab:red", label="runtime")
ax1b.set_ylabel("Computational time [s]", color="tab:red")
ax1b.tick_params(axis="y", labelcolor="tab:red")

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax1b.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper center")
ax1.invert_xaxis()
fig1.tight_layout()
fig1.savefig(f"{out_dir}/sigmaXY_runtime_vs_sc_dt.pdf")

# ── Plot 2: sigma_z vs sc_dt, runtime on twin axis ───────────────────────────
fig2, ax2 = plt.subplots(figsize=(8, 5))
ax2.semilogx(sc_dt_ns, sigma_z, "^-", label=r"$\sigma_z$")
ax2.set_xlabel(r"Space Charge Time Step [ns]")
ax2.set_ylabel(r"Beam size [m]")
ax2.grid(True, which="both", alpha=0.3)

ax2b = ax2.twinx()
ax2b.semilogx(sc_dt, runtime_s, "^-", color="tab:red", label="runtime")
ax2b.set_ylabel("Computational time [s]", color="tab:red")
ax2b.tick_params(axis="y", labelcolor="tab:red")

lines3, labels3 = ax2.get_legend_handles_labels()
lines4, labels4 = ax2b.get_legend_handles_labels()
ax2.legend(lines3 + lines4, labels3 + labels4, loc="upper center")
ax2.invert_xaxis()
fig2.tight_layout()
fig2.savefig(f"{out_dir}/sigmaZ_runtime_vs_sc_dt.pdf")

# ── Plot 3: % difference from coarsest step, runtime on twin axis ────────────
pct_x = (sigma_x - sigma_x[0]) / sigma_x[0] * 100
pct_y = (sigma_y - sigma_y[0]) / sigma_y[0] * 100
pct_z = (sigma_z - sigma_z[0]) / sigma_z[0] * 100

fig3, ax3 = plt.subplots(figsize=(8, 5))
ax3.semilogx(sc_dt_ns, pct_x, "o-", label=r"$\sigma_x$")
ax3.semilogx(sc_dt_ns, pct_y, "s-", label=r"$\sigma_y$")
ax3.semilogx(sc_dt_ns, pct_z, "^-", label=r"$\sigma_z$")
ax3.axhline(0, color="k", linewidth=0.8, linestyle="--")
ax3.set_xlabel(r"Space Charge Time Step [ns]")
ax3.set_ylabel(r"$\Delta\sigma$ / $\sigma_{\mathrm{coarse}}$ [%]")
ax3.grid(True, which="both", alpha=0.3)

ax3b = ax3.twinx()
ax3b.semilogx(sc_dt_ns, runtime_s, "^-", color="tab:red", label="runtime")
ax3b.set_ylabel("Computational time [s]", color="tab:red")
ax3b.tick_params(axis="y", labelcolor="tab:red")

lines5, labels5 = ax3.get_legend_handles_labels()
lines6, labels6 = ax3b.get_legend_handles_labels()
ax3.legend(lines5 + lines6, labels5 + labels6, loc="upper center")
ax3.invert_xaxis()
fig3.tight_layout()
fig3.savefig(f"{out_dir}/sigma_pct_diff_runtime_vs_sc_dt.pdf")

plt.show()
