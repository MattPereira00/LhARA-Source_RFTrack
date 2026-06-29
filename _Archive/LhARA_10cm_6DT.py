import numpy as np
import matplotlib.pyplot as plt
import RF_Track as rft
import pygpt
import pybdsim

import bdsim2rftrack

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
t_init = (0.1/V_ref)

# Build the beam from BDSIM userfile
bunch_init = bdsim2rftrack.bdsim_userfile_to_rftrack_bunch6DT(
    "Beams/LhARA_10cm_pm2-rftrack.dat", mass, 1, 1e9)

# 2mm cut at nozzle entrance
bunch_cut = bdsim2rftrack.apply_radial_cut_bunch6DT(
    bunch=bunch_init,
    cutradius=0.00287
)
cut_ps = get_coords_6DT(bunch_cut)
x_rftrack = cut_ps["x"]

# Convert the cut rftrack beam at 10cm to a bdsim infile
bdsim2rftrack.rftrack_bunch6DT_to_bdsim_userfile(
    bunch=bunch_cut,
    particle_mass=mass,
    filename="Beams/LhARA_10cm_pm2-cut-rftrack.dat"
)

# Convert the cut rftrack beam at 10cm to a -gpt infile
bdsim2rftrack.bdsim_userfile_to_gptin(
    filename="Beams/LhARA_10cm_pm2-cut-rftrack.dat",
    particle_mass=mass,
    particle_charge=1,
    time_init=t_init,
    output_filename="LhARA_10cm_pm2-cut-gpt.dat",
)

# Run BDSIM Primaries for the 5cm cut Beam
pybdsim.Run.Bdsim(gmadpath="BDSIM/10cmprimcut.gmad",
                  outfile="Beams/LhARA_10cm_pm2-cut-bdsim",
                  ngenerate=len(x_rftrack),
                  silent=True,
                  )
pybdsim.Run.RebdsimOptics(rootpath="Beams/LhARA_10cm_pm2-cut-bdsim.root",
                          outpath="Beams/LhARA_10cm_pm2-cut-bdsim-optics.root",
                          silent=True
                          )
pygpt.Plot.Phasespace.BDSIMPhaseSpace(filename="Beams/LhARA_10cm_pm2-cut-bdsim.root",
                                      outputfilename="plots/LhARA_10cm_pm2-cut"+"_BDSPS",
                                      coordsTitle=" ",
                                      correlationTitle=" ",
                                      )
plt.close('all')