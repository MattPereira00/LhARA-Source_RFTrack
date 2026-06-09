import os as _os
import numpy as _np
import RF_Track as _rft



def bdsim_userfile_to_rftrack_bunch6D(filename, particle_mass, particle_charge=1, bunch_charge=None, time_from_s=True, s_reference=None):
    """
    Load a particle coordinate file and build an RF_Track Bunch6d.

    Expected BDSIM input file columns:
        [x, y, s, xp, yp, E]

    RF_Track Bunch6D column order:
        [x, x', y, y', t, P, m, Q, n_macro]

    Parameters
    ----------
    filename : str
        Path to the text file.
    particle_mass : float
        Particle rest mass in MeV/c^2, e.g. rft.protonmass.
    particle_charge: int or float, optional
        Particle charge in units of e. Default is 1.
    bunch_charge : float
        Total bunch population in units of elementary charges.
        Example: 1e9 for a 1e9-particle bunch.
        If None, set 1.
    time_from_s: bool, optional
        Calculate time from the longitudinal coordinate s. If False, time will be set to zero. Default is True.
    s_reference: float or None, optional
        Reference s position in meters for time calculation. If None, set zero.

    Returns
    -------
    rft.Bunch6d
        RF Track 6D bunch object.
    """
    data = _np.genfromtxt(filename, dtype=float)

    if data.ndim == 1:
        data = data[None, :]

    if data.shape[1] < 6:
        raise ValueError(
            f"{filename} must have at least 6 columns: "
            "x, y, s, xp, yp, E"
        )

    x = data[:, 0] * 1e3   # m -> mm
    y = data[:, 1] * 1e3   # m -> mm
    S = data[:, 2]         # m
    xp = data[:, 3] * 1e3  # rad -> mrad
    yp = data[:, 4] * 1e3  # rad -> mrad
    E = data[:, 5]         # MeV

    m = float(particle_mass)
    Q = float(particle_charge)

    # Assumes E is total energy in MeV.
    P = _np.sqrt(E ** 2 - m ** 2)

    if time_from_s:
        beta = _np.divide(P, E, out=_np.zeros_like(P), where=E != 0)

        if s_reference is None:
            s0 = 0.0
        else:
            s0 = float(s_reference)

        # RF Track-style time coordinate
        t = (S - s0) / beta # m/c
        t = t * 1e3 # mm/c
    else:
        t = _np.zeros_like(S)

    if bunch_charge is not None:
        n_macro = float(bunch_charge) / len(x)
    else:
        n_macro = _np.ones(len(x))

    bunch_array = _np.column_stack([
        x, xp,
        y, yp,
        t, P,
        _np.full(len(x), m),
        _np.full(len(x), Q),
        _np.full(len(x), n_macro),
    ])

    return _rft.Bunch6d(bunch_array)

def bdsim_userfile_to_rftrack_bunch6DT(filename, particle_mass, particle_charge=1, bunch_charge=None):
    """
    Load a particle coordinate file and build an RF_Track Bunch6dT.

    Expected BDSIM input file columns:
        [x, y, s, xp, yp, E]

    RF_Track Bunch6DT column order:
        [x, p_x, y, p_y, z, p_z, m, Q, n_macro]

    Parameters
    ----------
    filename : str
        Path to the text file.
    particle_mass : float
        Particle rest mass in MeV/c^2, e.g. rft.protonmass.
    particle_charge: int or float, optional
        Particle charge in units of e. Default is 1.
    bunch_charge : float
        Total bunch population in units of elementary charges.
        Example: 1e9 for a 1e9-particle bunch.
        If None, set n_macro=1.

    Returns
    -------
    rft.Bunch6dT
        RF Track 6DT bunch object.
    """
    data = _np.genfromtxt(filename, dtype=float)

    if data.ndim == 1:
        data = data[None, :]

    if data.shape[1] < 6:
        raise ValueError(
            f"{filename} must have at least 6 columns: "
            "x, y, s, xp, yp, E"
        )

    x = data[:, 0] * 1e3   # m -> mm
    y = data[:, 1] * 1e3   # m -> mm
    z = data[:, 2] * 1e3   # m -> mm
    xp = data[:, 3]
    yp = data[:, 4]

    E = data[:, 5]         # MeV
    P = _np.sqrt(E ** 2 - particle_mass ** 2)
    p_z = P / (_np.sqrt(1 + xp ** 2 + yp ** 2))
    p_x = xp * p_z
    p_y = yp * p_z

    m = float(particle_mass)
    Q = float(particle_charge)
    if bunch_charge is not None:
        n_macro = float(bunch_charge) / len(x)
    else:
        n_macro = 1.0


    bunch_array = _np.column_stack([
        x, p_x,
        y, p_y,
        z, p_z,
        _np.full(len(x), m),
        _np.full(len(x), Q),
        _np.full(len(x), n_macro)
    ])

    return _rft.Bunch6dT(bunch_array)

def bdsim_userfile_to_gptin(filename, particle_mass, particle_charge=1, time_init=0.0, output_filename=None):
    """
    Load a particle coordinate file and convert to GPT input format.

    Expected BDSIM input file columns:
        [x, y, s, xp, yp, E]

    GPT input file column order:
        [x, y, z, GBx, GBy, GBz, t, G, m, Q]

    Parameters
    ----------
    filename : str
        Path to the BDSIM text file.
    particle_mass : float
        Particle rest mass in MeV/c^2, e.g. 938.3 for proton.
    particle_charge : int or float, optional
        Particle charge in units of e. Default is 1.
    time_init : float, optional
        Initial time value for all particles. Default is 0.0.
    output_filename : str, optional
        Full output path for the GPT file.
        If None, the output is written next to `filename`
        using the same base name with `-gpt.txt` appended.

    Returns
    -------
    None
        Writes output to the requested GPT input file.
    """
    data = _np.genfromtxt(filename, dtype=float)

    if data.ndim == 1:
        data = data[None, :]

    if data.shape[1] < 6:
        raise ValueError(
            f"{filename} must have at least 6 columns: "
            "x, y, s, xp, yp, E"
        )

    x = data[:, 0]  # m
    y = data[:, 1]  # m
    z = data[:, 2]  # m
    xp = data[:, 3]  # rad
    yp = data[:, 4]  # rad
    E = data[:, 5]  # MeV

    m_MeV = float(particle_mass)
    q = float(particle_charge)

    P = _np.sqrt(E**2 - m_MeV**2)  # MeV/c
    G = E / m_MeV

    norm_factor = _np.sqrt(1 + xp ** 2 + yp ** 2)
    p_z = P / norm_factor
    p_x = xp * p_z
    p_y = yp * p_z

    GBx = p_x / m_MeV
    GBy = p_y / m_MeV
    GBz = p_z / m_MeV

    t = _np.full_like(E, time_init)

    eV_to_J = 1.602176634e-19
    c = 299792458.0
    m_kg = (m_MeV * 1.0e6 * eV_to_J) / (c ** 2)

    output_array = _np.column_stack([
        x, y, z,
        GBx, GBy, GBz,
        t,
        G,
        _np.full(len(x), m_kg),
        _np.full(len(x), q)
    ])

    if output_filename is None:
        base_name = _os.path.splitext(_os.path.basename(filename))[0]
        output_filename = base_name + '-gpt.txt'
    else:
        output_filename = _os.path.join(_os.path.dirname(filename), output_filename)

    header = 'x\ty\tz\tGBx\tGBy\tGBz\tt\tG\tm\tQ'
    _np.savetxt(output_filename, output_array, header=header, comments='')

def rftrack_bunch6D_to_bdsim_userfile(bunch, particle_mass, filename, time_to_s=True, s_reference=None):
    """
    Save an RF_Track Bunch6D to a text file in BDSIM user file format.

    RF_Track Bunch6D column order:
        [x, x', y, y', t, P, m, Q, n_macro]

    Expected BDSIM output file columns:
        [x, y, s, xp, yp, E]

    Parameters
    ----------
    bunch : rft.Bunch6d
        RF Track 6D bunch object.
    filename : str
        Path to the output text file.
    time_to_s: bool, optional
        Calculate longitudinal coordinate s from the time coordinate. If False, s will be set to zero. Default is True.
    s_reference: float or None, optional
        Reference s position in meters for time calculation. If None, set zero.

    Returns
    -------
    None
    """
    ps = bunch.get_phase_space()

    x = ps[:, 0] * 1e-3   # mm -> m
    xp = ps[:, 1] * 1e-3  # mrad -> rad
    y = ps[:, 2] * 1e-3   # mm -> m
    yp = ps[:, 3] * 1e-3  # mrad -> rad
    t = ps[:, 4]          # mm/c
    P = ps[:, 5]          # MeV/c


    if time_to_s:
        beta = _np.divide(P, _np.sqrt(P ** 2 + particle_mass ** 2), out=_np.zeros_like(P), where=P != 0)

        if s_reference is None:
            s0 = 0.0
        else:
            s0 = float(s_reference)

        S = (t * beta) / 1e3 + s0  # m
    else:
        S = _np.zeros_like(t)

    E = _np.sqrt(P ** 2 + particle_mass ** 2) # MeV

    output_array = _np.column_stack([
        x, y,
        S,
        xp,
        yp,
        E,
    ])

    _np.savetxt(filename, output_array)

def rftrack_bunch6DT_to_bdsim_userfile(bunch, particle_mass, filename):
    """
    Save an RF_Track Bunch6DT to a text file in BDSIM user file format.

    RF_Track Bunch6DT column order:
        [x, p_x, y, p_y, z, p_z, m, Q, n_macro]

    Expected BDSIM output file columns:
        [x, y, s, xp, yp, E]

    Parameters
    ----------
    bunch : rft.Bunch6DT
        RF Track 6DT bunch object.
    filename : str
        Path to the output text file.


    Returns
    -------
    None
    """
    ps = bunch.get_phase_space()

    x = ps[:, 0] * 1e-3   # mm -> m
    p_x = ps[:, 1]
    y = ps[:, 2] * 1e-3   # mm -> m
    p_y = ps[:, 3]
    z = ps[:, 4] * 1e-3   # mm -> m
    p_z = ps[:, 5]

    s = z
    xp = _np.divide(p_x, p_z, out=_np.zeros_like(p_x), where=p_z != 0)
    yp = _np.divide(p_y, p_z, out=_np.zeros_like(p_y), where=p_z != 0)

    P = _np.sqrt(p_x ** 2 + p_y ** 2 + p_z ** 2)  # MeV/c
    E = _np.sqrt(P ** 2 + float(particle_mass) ** 2)  # MeV (total energy)

    output_array = _np.column_stack([x, y, s, xp, yp, E])

    _np.savetxt(filename, output_array)

def apply_radial_cut_bunch6DT(bunch, cutradius=0.00287):
    """
    Apply a radial cut to an RF_Track Bunch6DT object.

    Keeps only particles where sqrt(x^2 + y^2) < cutradius.

    Parameters
    ----------
    bunch : rft.Bunch6dT
        Input Bunch6DT object.
    cutradius : float
        Transverse radius cutoff in metres (default 0.00287).

    Returns
    -------
    rft.Bunch6dT
        New Bunch6DT object with particles inside the radius.
    """
    ps = bunch.get_phase_space("%X %Px %Y %Py %Z %Pz %m %Q %N")

    # Extract x and y (column 0 and 2 in Bunch6DT phase space)
    x = ps[:, 0] * 1e-3 # in m
    y = ps[:, 2] * 1e-3 # in m

    r = _np.sqrt((x**2) + (y**2))

    # Filter: keep particles inside cutradius
    mask = r < cutradius
    filtered_ps = ps[mask, :]
    # Create new Bunch6DT from filtered phase space
    bunch_cut = _rft.Bunch6dT(filtered_ps)

    print(f"Radial cut: {len(ps)} particles -> {len(filtered_ps)} particles (r < {cutradius} m)")

    return bunch_cut