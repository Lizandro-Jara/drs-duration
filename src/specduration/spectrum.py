import numpy as np

from .newmark import Newmark


# DRS(T)


def maxi(x):
    return float(np.max(np.abs(x)))


def compute_drs(
    ug,
    dt,
    zeta=0.05,
    Tmin=None,
    Tmax=15.0,
    dT=0.01,
):
    """
    Compute the displacement response spectrum DRS(T)
    using Newmark-beta time integration.
    """
    ug = np.asarray(ug, dtype=float)

    if ug.size < 2:
        raise ValueError(
            "At least two acceleration points are required to compute the DRS."
        )

    if not np.isfinite(dt) or dt <= 0:
        raise ValueError(
            "The time step dt must be a positive finite value."
        )

    if not np.isfinite(Tmax) or Tmax <= 0:
        raise ValueError(
            "Tmax must be a positive finite value."
        )

    if not np.isfinite(dT) or dT <= 0:
        raise ValueError(
            "dT must be a positive finite value."
        )

    if Tmin is None:
        Tmin = max(10.0 * dt, 0.01)

    if not np.isfinite(Tmin) or Tmin <= 0:
        raise ValueError(
            "Tmin must be a positive finite value."
        )

    if Tmin >= Tmax:
        raise ValueError(
            "Tmin must be smaller than Tmax."
        )

    T_values = np.arange(
        Tmin,
        Tmax + dT,
        dT,
        dtype=float,
    )

    RESP = np.zeros_like(T_values)

    for i, Tn in enumerate(T_values):
        try:
            u, _, _ = Newmark(
                ug,
                0.0,
                0.0,
                0.5,
                0.25,
                dt,
                zeta,
                float(Tn),
            )

            RESP[i] = maxi(u)

        except Exception:
            RESP[i] = np.nan

    return T_values, RESP