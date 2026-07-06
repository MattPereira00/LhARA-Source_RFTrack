# LhARA Source Region Tracking — RF_Track Pipeline

Proton bunch tracking from the laser-plasma source (0 cm) to the LhARA nozzle exit (10 cm),
implemented in RF_Track with optional BDSIM phase-space verification.

---

## Repository Structure

```
02-Source_RFTrack/
├── 00-Documentation/       This README and any further documentation
├── 01-BDSIM/               BDSIM lattice files (.gmad) for each tracking checkpoint
├── 10-Beams/               Input and output beam files (.dat) — not tracked in git
├── 11-Plots/               Phase-space PDFs produced by BDSIM — not tracked in git
├── LhARA_source_to_nozzle.py   Main tracking pipeline (see below)
├── bdsim2rftrack.py            Coordinate conversion library
└── .gitignore
```

---

## Physics Overview

The source region transports a laser-accelerated (TNSA) proton bunch through two sections:

| Segment | Length | Description |
|---------|--------|-------------|
| Source → nozzle entrance | 5 cm | Free drift, no space charge |
| Nozzle entrance → nozzle exit | 5 cm | Drift with space-charge (PIC) |

Radial aperture cuts are applied at each boundary to model the physical beam pipe.

### Reference particle

| Parameter | Value |
|-----------|-------|
| Species | Proton |
| Rest mass | 938.272 MeV/c² |
| Kinetic energy | 15 MeV |
| Total momentum | ~168 MeV/c |
| Total energy | 953.272 MeV |

### Space-charge settings

| Parameter | Value |
|-----------|-------|
| Method | PIC, free-space (`SpaceCharge_PIC_FreeSpace`) |
| Grid | 50 × 50 × 125 cells |
| Time step (`sc_dt_mm`) | 0.3 mm/c |
| Integrator | RK4 |

The 0.3 mm/c timestep was selected from convergence scans (see
`LhARA_0to5cm_SC_TStepScan.py` and `LhARA_5to10cm_SC_TStepScan.py`).

---

## Main Pipeline: `LhARA_source_to_nozzle.py`

Runs the full 0 → 10 cm tracking in four stages:

| Stage | Action | Output beam file |
|-------|--------|-----------------|
| 1 | Free drift, 0 cm → 5 cm | `<label>_5cm-rftrack.dat` |
| 2 | Radial cut 2.0 mm at nozzle entrance | `<label>_5cm-cut-rftrack.dat` |
| 3 | SC drift, 5 cm → 10 cm | `<label>_10cm-rftrack.dat` |
| 4 | Radial cut 2.87 mm at nozzle exit | `<label>_10cm-cut-rftrack.dat` |

All beam files are written in BDSIM user-file format:
`x [m]  y [m]  s [m]  xp [rad]  yp [rad]  E [MeV]`

### Usage

```bash
# Full pipeline (all four stages)
python LhARA_source_to_nozzle.py 10-Beams/LhARA_pm2_0cm-bdsimin.dat

# With BDSIM phase-space plots at each checkpoint
python LhARA_source_to_nozzle.py 10-Beams/LhARA_pm2_0cm-bdsimin.dat --bdsim

# Custom output directories and label
python LhARA_source_to_nozzle.py <input_file> --outdir 10-Beams --plotdir 11-Plots --label LhARA_pm2

# Run only a subset of stages (input must match state before the first selected stage)
python LhARA_source_to_nozzle.py 10-Beams/LhARA_pm2_5cm-cut-rftrack.dat --stages 3,4 --label LhARA_pm2
```

### Command-line arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `input_file` | (required) | BDSIM user file at the start of the first selected stage |
| `--outdir` | `10-Beams` | Directory for output beam files |
| `--plotdir` | `11-Plots` | Directory for BDSIM phase-space PDFs |
| `--label` | derived from input filename | Stem used for all output filenames |
| `--bdsim` | off | Run BDSIM and produce phase-space plots at each stage |
| `--stages` | `1,2,3,4` | Comma-separated subset of stages to run |

The default label is derived automatically: the `_0cm-bdsimin` suffix is stripped from the
input filename stem (e.g. `LhARA_pm2_0cm-bdsimin.dat` → `LhARA_pm2`).

---

## Coordinate Conversion Library: `bdsim2rftrack.py`

Provides functions to convert between the BDSIM user-file format and RF_Track's
`Bunch6dT` Cartesian momentum phase space, and to apply aperture cuts.

### Key functions

| Function | Description |
|----------|-------------|
| `bdsim_userfile_to_rftrack_bunch6DT(filename, particle_mass, particle_charge, bunch_charge)` | Load a BDSIM user file into an RF_Track `Bunch6dT` object |
| `rftrack_bunch6DT_to_bdsim_userfile(bunch, particle_mass, filename)` | Save an RF_Track `Bunch6dT` to BDSIM user-file format |
| `apply_radial_cut_bunch6DT(bunch, cutradius)` | Remove particles outside a transverse radius (metres) |

`Bunch6dT` column order: `[x(mm), px(MeV/c), y(mm), py(MeV/c), z(mm), pz(MeV/c), m, Q, n_macro]`

---

## BDSIM Lattice Files: `01-BDSIM/`

Used by the pipeline's `--bdsim` option to produce phase-space plots at each checkpoint.
Requires BDSIM and pybdsim to be installed.

| File | Reads from | Geometry |
|------|-----------|----------|
| `0cmprim.gmad` | `10-Beams/<label>_0cm-bdsimin.dat` | 1 drift, 5 cm |
| `5cmprim.gmad` | `10-Beams/<label>_5cm-rftrack.dat` | 2 drifts, 10 cm |
| `5cmprimcut.gmad` | `10-Beams/<label>_5cm-cut-rftrack.dat` | 2 drifts, 10 cm |
| `10cmprim.gmad` | `10-Beams/<label>_10cm-rftrack.dat` | 3 drifts, 15 cm |
| `10cmprimcut.gmad` | `10-Beams/<label>_10cm-cut-rftrack.dat` | 3 drifts, 15 cm |

All lattices use `energy = 953.27231 MeV` and
`distrFileFormat = "x[m]:y[m]:S[m]:xp[rad]:yp[rad]:E[MeV]"`.

---

## Input File Format

The pipeline expects a BDSIM user file with one particle per line:

```
x[m]  y[m]  s[m]  xp[rad]  yp[rad]  E[MeV]
```

The standard input for a full run is `10-Beams/LhARA_pm2_0cm-bdsimin.dat`,
generated upstream from a laser-plasma source model.

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `RF_Track` | Particle tracking (`Bunch6dT`, `Drift`, `Lattice`, `Volume`, space charge) |
| `numpy` | Numerical operations |
| `matplotlib` | Plots |
| `pybdsim` | Run BDSIM and rebdsim from Python (required only with `--bdsim`) |
| `pygpt` | Read GPT output and produce BDSIM phase-space plots (required only with `--bdsim`) |
