import numpy as np
import RF_Track as rft
import pygpt
import pybdsim


def get_coords(bunch):
    ps = bunch.get_phase_space()
    return {
        "x": ps[:,0],
        "xp": ps[:,1],
        "y": ps[:,2],
        "yp": ps[:,3],
        "t": ps[:,4],
        "P": ps[:,5],
    }

# Elements
nozzle = rft.Drift(length=0.05)
SC = rft.SpaceCharge_PIC_FreeSpace(50, 50, 125)
nozzle.set_sc_nsteps(50)

# Lattice
line = rft.Lattice()
line.append(nozzle)

# Build the beam
data = np.genfromtxt("Beams/LhARA_5cm_pm100_rftrackin.txt", dtype="float")
bunch_at_nozzle = rft.Bunch6d(data)

# Nozzle tracking with Space Charge
bunch_nozzle_end = line.track(bunch_at_nozzle)
ps_nozzle_end = get_coords(bunch_at_nozzle)
