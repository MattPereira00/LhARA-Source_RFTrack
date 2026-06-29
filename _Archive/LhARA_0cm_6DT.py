import numpy as np
import matplotlib.pyplot as plt
import RF_Track as rft
import pygpt
import pybdsim

import bdsim2rftrack
from plot_residuals import plot_tracking_residuals

np.set_printoptions(precision=14)

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

# Convert the -bdsimin file to a -gptin file
bdsim2rftrack.bdsim_userfile_to_gptin(
    "Beams/LhARA_0cm_pm2-bdsimin.dat",
    mass, 1, 0.0,
    "LhARA_0cm_pm2-gptin.txt")

# Build the bunch from BDSIM userfile
bunch_init = bdsim2rftrack.bdsim_userfile_to_rftrack_bunch6DT(
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
world.add(line, 0, 0, 0)

# Tracking options
world.t_max_mm=1e-12*rft.clight*1e3
world.odeint_algorithm = "rk4"

# Tracking
ps_init = get_coords_6DT(bunch_init)
pico_step = world.track(bunch_init)
step_ps = get_coords_6DT(pico_step)

bdsim2rftrack.rftrack_bunch6DT_to_bdsim_userfile(
    bunch=pico_step,
    particle_mass=mass,
    filename="Beams/LhARA_0cm_pm2-step-rftrack.dat"
)

# dist comparison
x_rftrack = step_ps["x"]*1e-3
y_rftrack = step_ps["y"]*1e-3
px = step_ps["p_x"]
py = step_ps["p_y"]
norm_px_rftrack = px / p_ref
norm_py_rftrack = py / p_ref

gpt_data = pygpt.Reader.LoadGptData("../GPT/IdealTNSA/pm2/Source/LhARA_0cm_pm2.txt").times[0]
x_gpt = gpt_data.GetColumn('x')
y_gpt = gpt_data.GetColumn('y')
px_gpt = gpt_data.GetAbsolutexp()
py_gpt = gpt_data.GetAbsoluteyp()
norm_px_gpt = px_gpt / p_ref
norm_py_gpt = py_gpt / p_ref

rf = {
    "x": x_rftrack,
    "y": y_rftrack,
    "px": norm_px_rftrack,
    "py": norm_py_rftrack,
}

gpt = {
    "x": x_gpt,
    "y": y_gpt,
    "px": norm_px_gpt,
    "py": norm_py_gpt,
}

plot_tracking_residuals(
    ref=rf,
    other=gpt,
    out_pdf="plots/LhARA_0cm_pm2_Residual_RF_GPT.pdf",
    ref_name="RFTrack",
    other_name="GPT",
    showPlot=False,
)

# Run BDSIM Primaries for the 0cm Beam
pybdsim.Run.Bdsim(gmadpath="BDSIM/0cmprim.gmad",
                  outfile="Beams/LhARA_0cm_pm2-bdsim",
                  ngenerate=len(x_rftrack),
                  silent=True,
                  )
pybdsim.Run.RebdsimOptics(rootpath="Beams/LhARA_0cm_pm2-bdsim.root",
                          outpath="Beams/LhARA_0cm_pm2-bdsim-optics.root",
                          silent=True
                          )
pygpt.Plot.Phasespace.BDSIMPhaseSpace(filename="Beams/LhARA_0cm_pm2-bdsim.root",
                                      outputfilename="plots/LhARA_0cm_pm2"+"_BDSPS",
                                      coordsTitle=" ",
                                      correlationTitle=" ",
                                      )
plt.close('all')
