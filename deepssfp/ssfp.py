import numpy as np
from typing import Union

def ssfp(T1: Union[float, np.ndarray],
         T2: Union[float, np.ndarray],
         TR: float,
         TE: float,
         alpha: float,
         dphi: Union[float, np.ndarray] = (0,),
         field_map: Union[float, np.ndarray] = 0,
         M0: Union[float, np.ndarray] = 1,
         f0: Union[float, np.ndarray] = 0,
         phi: Union[float, np.ndarray] = 0,
         useSqueeze: bool = True) -> np.ndarray:
    """ Transverse signal for SSFP MRI after excitation at TE.

    Parameters
    ----------
    T1 : float or np.ndarray
        longitudinal exponential decay time constant (in seconds).
    T2 : float or np.ndarray
        transverse exponential decay time constant (in seconds).
    TR : float
        repetition time (in seconds).
    TE : float
        echo time (in seconds).
    alpha : float 
        flip angle (in rad).
    dphi : float or np.ndarray, optional
        Linear phase-cycle increment (in rad).
    field_map : float or np.ndarray, optional
        B0 field map (in Hz).
    M0 : float or np.ndarray, optional
        proton density.
    f0 : float or np.ndarray, optional
        off-resonance (in Hz). Includes factors like the chemical shift
        of species w.r.t. the water peak.
    phi : float or np.ndarray, optional
        phase offset (in rad).
    useSqueeze : bool, optional
        Whether to squeeze the output array.

    Returns
    -------
    np.ndarray
        Complex-valued array representing the transverse magnetization.
    """

    # Convert inputs to arrays and adjust field map convention
    inputs = [T1, T2, f0, -field_map, M0]
    T1, T2,  f0, field_map, M0 = map(np.asarray, inputs)
    dphi = np.asarray(dphi).ravel()

    # Determine the broadcasted shape and reshape dphi
    broadcast_shape = np.broadcast(*inputs).shape
    dphi = dphi.reshape((1,) * len(broadcast_shape) + (-1,))

    # Broadcast all input arrays to the common shape
    T1, T2, f0, field_map, M0 = np.broadcast_arrays(*inputs)

    # Compute exponential decays
    E1 = np.where(T1 > 0, np.exp(-TR / T1), 0)
    E2 = np.where(T2 > 0, np.exp(-TR / T2), 0)

    # Compute beta and trigonometric functions
    beta = 2 * np.pi * (f0 + field_map) * TR
    cos_alpha, sin_alpha = np.cos(alpha), np.sin(alpha)

    # Prepare for broadcasting with dphi
    expand_dims = lambda x: x[..., np.newaxis]
    beta, E1, E2, sin_alpha, cos_alpha, M0, T2 = map(expand_dims, 
                                                          [beta, E1, E2, sin_alpha, cos_alpha, M0, T2])

    # Compute theta
    theta = beta - dphi
    cos_theta, sin_theta = np.cos(theta), np.sin(theta)

    # Calculate denominator
    denominator = ((1 - E1 * cos_alpha) * (1 - E2 * cos_theta) -
                   E2 * (E1 - cos_alpha) * (E2 - cos_theta))

    # Calculate Mx and My
    common_factor = M0 * (1 - E1) * sin_alpha / denominator
    Mx = common_factor * (1 - E2 * cos_theta)
    My = common_factor * E2 * sin_theta

    # Combine Mx and My into complex magnetization and apply phase and T2 decay
    _phi = beta * (TE / TR) + phi
    T2_decay = np.where(T2 > 0, np.exp(-TE / T2), 0)
    M = ( Mx + 1j * My) * np.exp(1j * _phi) * T2_decay

    return np.squeeze(M) if useSqueeze else M


def ssfp_og(T1, T2, TR, TE, alpha, dphi=(0,), field_map=0, M0=1, f0=0, phi=0, useSqueeze=True) -> np.ndarray:
    """ Multiple acquisition ssfp """
    dphi = np.atleast_2d(dphi)

    M = []
    for ii, pc in np.ndenumerate(dphi):
        M.append(_ssfp(T1, T2, TR, TE, alpha, pc, field_map, M0, f0, phi)[..., None])
    M = np.concatenate(M, axis=-1)
    
    # Squeeze out dim of length 1, otherwise shape is [width, height, dphi]
    M = np.squeeze(M)
    if not useSqueeze and len(dphi) == 1:
        M = M[..., None]
    return M

def _ssfp(T1, T2, TR, TE, alpha, dphi=0, field_map=0, M0=1, f0=0, phi=0):
    """ transverse signal for ssfp mri after excitation at TE

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
    field_map = -1 * field_map
    
    # Set T1, T2, alpha, and field_map inputs to arrays
    T1 = np.atleast_2d(T1)
    T2 = np.atleast_2d(T2)
    alpha = np.atleast_2d(alpha)
    f0 = np.atleast_2d(f0)
    field_map = np.atleast_2d(field_map)

    # Compute exponential decay and handle T1, T2 of zero
    E1 = np.zeros(T1.shape)
    E1[T1 > 0] = np.exp(-TR/T1[T1 > 0])
    E2 = np.zeros(T2.shape)
    E2[T2 > 0] = np.exp(-TR/T2[T2 > 0])

    # Precompute theta and derivatives of theta and alpha
    beta = 2 * np.pi * (f0 + field_map) * TR
    theta = beta - dphi; # theta => phase per repetition time
    cos_alpha = np.cos(alpha)
    sin_alpha = np.sin(alpha)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    # Calculate Mxy 
    Mbottom = (1 - E1 * cos_alpha) * (1 - E2 * cos_theta) - E2 * (E1 - cos_alpha) * (E2 - cos_theta)
    Mx = M0 * (1 - E1) * sin_alpha * (1 - E2 * cos_theta) / Mbottom
    My = M0 * (1 - E1) * E2 * sin_alpha * sin_theta / Mbottom
    Mc = Mx + 1j * My

    # Add additional phase and handle T2 of zero 
    T2 = np.array(T2)
    idx = np.where(T2 > 0)
    val = np.zeros(T2.shape)
    val[idx] = -TE/T2[idx]
    _phi = beta * (TE / TR) + phi
    Mc = Mc * np.exp(1j * _phi) * np.exp(val)

    return Mc


def ssfp2(T1, T2, TR, TE, alpha, dphi=(0,), field_map=0, M0=1, f0=0, phi=0, useSqueeze=True):
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