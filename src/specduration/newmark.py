import numpy as np
from numba import njit


# Newmark-beta time integration


@njit
def Newmark(
    ug,
    u0=0.0,
    up0=0.0,
    gamma=0.5,
    beta=0.25,
    dt=1 / 100,
    zeta=0.05,
    Tn=0.1,
):
    if Tn <= 0:
        raise ValueError("Tn must be greater than zero.")

    m = 1.0
    w = 2.0 * np.pi / Tn
    k = w * w * m
    c = 2.0 * zeta * w * m
    p = -m * ug

    n = len(ug)

    u = np.zeros(n)
    up = np.zeros(n)
    upp = np.zeros(n)

    u[0] = u0
    up[0] = up0
    upp[0] = (
        -ug[0]
        - (c / m) * up0
        - (k / m) * u0
    )

    a1 = (
        m / (beta * dt * dt)
        + gamma * c / (beta * dt)
    )

    a2 = (
        m / (beta * dt)
        + (gamma / beta - 1.0) * c
    )

    a3 = (
        (1.0 / (2.0 * beta) - 1.0) * m
        + dt * (
            gamma / (2.0 * beta) - 1.0
        ) * c
    )

    kt = k + a1

    for i in range(n - 1):
        p_eff = (
            p[i + 1]
            + a1 * u[i]
            + a2 * up[i]
            + a3 * upp[i]
        )

        u_next = p_eff / kt

        up_next = (
            (gamma / (beta * dt))
            * (u_next - u[i])
            + (1.0 - gamma / beta) * up[i]
            + dt
            * (
                1.0
                - gamma / (2.0 * beta)
            )
            * upp[i]
        )

        upp_next = (
            (u_next - u[i])
            / (beta * dt * dt)
            - up[i] / (beta * dt)
            - (
                1.0 / (2.0 * beta) - 1.0
            ) * upp[i]
        )

        u[i + 1] = u_next
        up[i + 1] = up_next
        upp[i + 1] = upp_next

    return u, up, upp