import numpy as np

def ssfp(T1, T2, TR, TE, alpha, dphi=(0,), field_map=0, M0=1, f0=0, phi=0, useSqueeze=True):
    """ Transverse signal for ssfp mri after excitation at TE

    Parameters
    ----------
    T1 : float or array_like
        longitudinal exponential decay time constant (in seconds).
    T2 : float or array_like
        transverse exponential decay time constant (in seconds).
    TR : float
        repetition time (in seconds).
    alpha : float or array_like
        flip angle (in rad).
    dphi : float, optional
        Linear phase-cycle increment (in rad).
    field_map : float or array_like, optional
        B0 field map (in Hz).
    M0 : float or array_like, optional
        proton density.
    f0 : float, optional
        off-resonance (in Hz). Includes factors like the chemical shift
        of species w.r.t. the water peak.
    phi : float, optional
        phase offset (in rad).
    """
    
    # Convention for Ernst-Anderson based implementation from Hoff
    field_map = -field_map

    # Set T1, T2, alpha, and field_map inputs to arrays
    dphi = np.atleast_2d(dphi)
    T1, T2, alpha, f0, field_map = map(np.atleast_2d, (T1, T2, alpha, f0, field_map))

    # Compute exponential decay and handle T1, T2 of zero
    E1 = np.where(T1 > 0, np.exp(-TR/T1), 0)
    E2 = np.where(T2 > 0, np.exp(-TR/T2), 0)

    # Precompute theta and derivatives of theta and alpha
    beta = 2 * np.pi * (f0 + field_map) * TR # theta => phase per repetition time
    cos_alpha, sin_alpha = np.cos(alpha), np.sin(alpha)

    def compute_Mc(pc):
        theta = beta - pc
        cos_theta, sin_theta = np.cos(theta), np.sin(theta)

        # Calculate Mxy 
        Mbottom = (1 - E1 * cos_alpha) * (1 - E2 * cos_theta) - E2 * (E1 - cos_alpha) * (E2 - cos_theta)
        Mx = M0 * (1 - E1) * sin_alpha * (1 - E2 * cos_theta) / Mbottom
        My = M0 * (1 - E1) * E2 * sin_alpha * sin_theta / Mbottom
        Mc = Mx + 1j * My

        # Add additional phase and handle T2 of zero 
        _phi = beta * (TE / TR) + phi
        T2_decay = np.where(T2 > 0, np.exp(-TE/T2), 0)
        return Mc * np.exp(1j * _phi) * T2_decay

    # Stack pcs into M
    M = np.stack([compute_Mc(pc) for pc in dphi.flat], axis=-1)
    
    return np.squeeze(M) if useSqueeze else M