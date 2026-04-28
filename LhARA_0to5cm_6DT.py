import numpy as np
import RF_Track as rft
import pygpt
import pybdsim
import bdsim2rftrack


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

# Build the beam from BDSIM userfile
bunch_init = bdsim2rftrack.bdsim_userfile_to_rftrack_bunch6DT(
    "Beams/LhARA_0cm_pm2-bdsimin.dat", rft.protonmass, 1, 1e9)

# Elements
source_to_nozzle = rft.Drift(length=0.05)

# Lattice
line = rft.Lattice()
line.append(source_to_nozzle)

# Volume
world = rft.Volume()
world.add(line, 0, 0, 0)

# Tracking
ps_init = get_coords_6DT(bunch_init)
bunch_at_nozzle = world.track(bunch_init)
ps_nozzle = get_coords_6DT(bunch_at_nozzle)

# Initial dist comparison
sigma_x_rftrack = np.std(ps_init["x"]*1e-3)
sigma_y_rftrack = np.std(ps_init["y"]*1e-3)

initial_gdfa_data = pygpt.Reader.LoadGdfaData("../GPT/IdealTNSA/pm2/Source/LhARA_0cm_pm2-gdfa.txt")
sigma_x_gpt = initial_gdfa_data.Sigma_x()[-1]
sigma_y_gpt = initial_gdfa_data.Sigma_y()[-1]

sigma_x_diff_gpt = sigma_x_gpt - sigma_x_rftrack
sigma_y_diff_gpt = sigma_y_gpt - sigma_y_rftrack

print("--Initial--\nSigma_X_diff_gpt = ", sigma_x_diff_gpt, "\nSigma_Y_diff_gpt = ", sigma_y_diff_gpt)

# Post-track comparison
sigma_x_rftrack = np.std(ps_nozzle["x"] * 1e-3)
sigma_y_rftrack = np.std(ps_nozzle["y"] * 1e-3)

# gpt_gdfa_data = pygpt.Reader.LoadGdfaData("../GPT/IdealTNSA/pm2/Source/LhARA_5cm_pm2-screen-gdfa.txt")
# sigma_x_gpt = gpt_gdfa_data.Sigma_x()[-1]
# sigma_y_gpt = gpt_gdfa_data.Sigma_y()[-1]

arr = np.genfromtxt("../GPT/IdealTNSA/pm2/Source/LhARA_5cm_pm2-gdfa.txt", names=True)
sigma_x_gpt = float(np.atleast_1d(arr["stdx"])[-1])
sigma_y_gpt = float(np.atleast_1d(arr["stdy"])[-1])

sigma_x_diff_gpt = sigma_x_gpt - sigma_x_rftrack
sigma_y_diff_gpt = sigma_y_gpt - sigma_y_rftrack

print("\n--Final--\nSigma_X_diff_gpt = ", sigma_x_diff_gpt, "\nSigma_Y_diff_gpt = ", sigma_y_diff_gpt)

bdsim2rftrack.rftrack_bunch6DT_to_bdsim_userfile(bunch_at_nozzle, rft.protonmass, "Beams/LhARA_5cm_pm2_6DT-bdsimin.dat")