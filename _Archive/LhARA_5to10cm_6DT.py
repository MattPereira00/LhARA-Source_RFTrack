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

# Space Charge
SC = rft.SpaceCharge_PIC_FreeSpace(50, 50, 125)
rft.cvar.SC_engine = SC

# Build the beam from BDSIM userfile
bunch_nozzle_entrance = bdsim2rftrack.bdsim_userfile_to_rftrack_bunch6DT(
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
world.t_max_mm=(0.05/V_ref) * rft.clight * 1e3 # 5cm for reference particle (10 cm in full model)
world.sc_dt_mm = 1e-12 * rft.clight * 1e3

# Tracking
ps_init = get_coords_6DT(bunch_nozzle_entrance)
bunch_nozzle_exit = world.track(bunch_nozzle_entrance)
ps_nozzle = get_coords_6DT(bunch_nozzle_exit)

bdsim2rftrack.rftrack_bunch6DT_to_bdsim_userfile(
    bunch=bunch_nozzle_exit,
    particle_mass=mass,
    filename="Beams/LhARA_10cm_pm2-SC-rftrack.dat"
)

# RF_Track
x_rftrack = ps_nozzle["x"]*1e-3
y_rftrack = ps_nozzle["y"]*1e-3
px = ps_nozzle["p_x"]
py = ps_nozzle["p_y"]
norm_px_rftrack = px / p_ref
norm_py_rftrack = py / p_ref

# GPT
gpt_data = pygpt.Reader.LoadGptData("../GPT/IdealTNSA/pm2/Source/LhARA_10cm_pm2-SC.txt").times[0]
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
    out_pdf="plots/LhARA_5to10cm_pm2_Residual_RF_GPT.pdf",
    ref_name="RFTrack",
    other_name="GPT",
    showPlot=True
)
sys.exit()
# Run BDSIM Primaries for the 10cm Beam
pybdsim.Run.Bdsim(gmadpath="BDSIM/10cmprim.gmad",
                  outfile="Beams/LhARA_10cm_pm2-bdsim",
                  ngenerate=len(x_rftrack),
                  silent=True,
                  )
pybdsim.Run.RebdsimOptics(rootpath="Beams/LhARA_10cm_pm2-bdsim.root",
                          outpath="Beams/LhARA_10cm_pm2-bdsim-optics.root",
                          silent=True
                          )
pygpt.Plot.Phasespace.BDSIMPhaseSpace(filename="Beams/LhARA_10cm_pm2-bdsim.root",
                                      outputfilename="plots/LhARA_10cm_pm2"+"_BDSPS",
                                      coordsTitle=" ",
                                      correlationTitle=" ",
                                      )
plt.close('all')