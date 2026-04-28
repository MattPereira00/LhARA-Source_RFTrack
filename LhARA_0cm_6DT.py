import numpy as np
import RF_Track as rft
import pygpt
import bdsim2rftrack
from scipy import constants as con

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
mass = con.physical_constants["proton mass energy equivalent in MeV"][0]
# Build the beam from BDSIM userfile
# bunch_init = bdsim2rftrack.bdsim_userfile_to_rftrack_bunch6DT(
#     "Beams/LhARA_0cm_pm2-bdsimin.dat", mass, 1, 1e9)

# Two Particle Validation
x = [0.0, 0.0, 0.0]
y = [0.0, 0.0, 0.0]
z = [0.0, 0.0, 0.0]

def momentum_components(P, xp, yp):
    denom = np.sqrt(1 + xp**2 + yp**2)
    pz = P / denom
    px = xp * pz
    py = yp * pz
    return px, py, pz

P1 = np.sqrt((mass+15.3)**2 - mass**2)
P2 = np.sqrt((mass+14.7)**2 - mass**2)
P3 = np.sqrt((mass+15.0)**2 - mass**2)
xp = -0.1112133160895747
yp = 0.0
px1, py1, pz1 = momentum_components(P1, xp, yp)
px2, py2, pz2 = momentum_components(P2, xp, yp)
px3, py3, pz3 = momentum_components(P3, xp, yp)

Px = np.array([px1, px2, px3])
Py = np.array([py1, py2, py3])
Pz = np.array([pz1, pz2, pz3])
m = np.array([mass, mass, mass])
Q = np.array([1.0, 1.0, 1.0])
N_macro = 1

bunch_array = np.column_stack([
    x, Px,
    y, Py,
    z, Pz,
    m,
    Q,
    np.full(len(x), N_macro),
])
bunch_init = rft.Bunch6dT(bunch_array)

# Export two particles to GPT
# normalized momentum (dimensionless)
# GBx = Px / m
# GBy = Py / m
# GBz = Pz / m
#
# GB = np.sqrt(GBx**2 + GBy**2 + GBz**2)
# gamma = np.sqrt(1 + GB**2)
#
# t = [0.0, 0.0, 0.0]
# m_kg = mass * 1.78266192e-30
#
# particles = np.column_stack([
#     x,
#     y,
#     z,
#     GBx,
#     GBy,
#     GBz,
#     t,
#     gamma,
#     np.full(len(x), m_kg)
# ])
#
# np.savetxt(
#     "Beams/LhARA_0cm_twopart-gptin.txt",
#     particles,
#     fmt="%.18e",
#     delimiter="\t",
#     header="x\ty\tz\tGBx\tGBy\tGBz\tt\tG\tm"
# )

# Elements
source_to_nozzle = rft.Drift(length=0.05)

# Lattice
line = rft.Lattice()
line.append(source_to_nozzle)

# Volume
world = rft.Volume()
world.add(line, 0, 0, 0)

# Tracking options
world.t_max_mm=1e-12*rft.clight*1e3
world.odeint_algorithm = "rk4"

# Tracking
ps_init = get_coords_6DT(bunch_init)
GB_X_init = ps_init["p_x"]/mass

pico_step = world.track(bunch_init)
step_ps = get_coords_6DT(pico_step)

# Step dist comparison
px_rftrack = step_ps["p_x"]
py_rftrack = step_ps["p_y"]
pz_rftrack = step_ps["p_z"]

p_ref = np.sqrt((mass+15)**2 - mass**2)

norm_px_rftrack = px_rftrack / p_ref
norm_py_rftrack = py_rftrack / p_ref
norm_pz_rftrack = pz_rftrack / p_ref

sigma_px_rftrack = np.std(norm_px_rftrack)
sigma_py_rftrack = np.std(norm_py_rftrack)
sigma_pz_rftrack = np.std(norm_pz_rftrack)

initial_gdfa_data = pygpt.Reader.LoadGdfaData("../GPT/IdealTNSA/pm2/Source/LhARA_0cm_twopart-gdfa.txt")
sigma_xp_gpt = initial_gdfa_data.Sigma_xp()[-1]
sigma_yp_gpt = initial_gdfa_data.Sigma_yp()[-1]
sigma_zp_gpt = initial_gdfa_data.Sigma_zp()[-1]

initial_gpt_data = pygpt.Reader.LoadGptData("../GPT/IdealTNSA/pm2/Source/LhARA_0cm_twopart.txt").times[0]
xp_gpt = initial_gpt_data.Getxp()
yp_gpt = initial_gpt_data.Getyp()
zp_gpt = initial_gpt_data.Getzp()


x_diff_gpt = xp_gpt - norm_px_rftrack
y_diff_gpt = yp_gpt - norm_py_rftrack
z_diff_gpt = zp_gpt - norm_pz_rftrack

print("RF_TRACK: ", norm_px_rftrack, norm_py_rftrack, norm_pz_rftrack)
print("GPT: ", xp_gpt, yp_gpt, zp_gpt)

print("--Step--\nSigma_X_diff_gpt = ", x_diff_gpt, "\nSigma_Y_diff_gpt = ", y_diff_gpt, "\nSigma_Z_diff_gpt = ", z_diff_gpt)