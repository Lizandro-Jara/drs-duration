import math
import numpy as np
from scipy.optimize import curve_fit

from .window import detect_window


# Harmonic spectral model


def spectral_sine_model(T, P, Ts, c):
    T = np.asarray(T, dtype=float)

    safe_T = np.where(
        T == 0,
        np.finfo(float).eps,
        T,
    )

    return P * np.sin(
        np.pi * Ts / safe_T - c
    )


# Initial guesses


def _build_initial_guesses(
    T_win,
    detect_res,
):
    T_win = np.asarray(
        T_win,
        dtype=float,
    )

    T_left = float(
        np.min(T_win)
    )

    T_right = float(
        np.max(T_win)
    )

    T_mean = float(
        np.mean(T_win)
    )

    T_peak = float(
        detect_res.get(
            "T_peak",
            T_mean,
        )
    )

    window_width = max(
        float(T_right - T_left),
        1e-6,
    )

    Ts_candidates = []

    for factor in [
        0.75,
        1.0,
        1.25,
        1.5,
        1.75,
        2.0,
        2.25,
        2.5,
        3.0,
    ]:
        Ts_candidates.append(
            factor * T_peak
        )

    for factor in [
        1.0,
        1.5,
        2.0,
        2.5,
    ]:
        Ts_candidates.append(
            factor * T_mean
        )

    for factor in [
        1.0,
        2.0,
        4.0,
        8.0,
        12.0,
    ]:
        Ts_candidates.append(
            factor * window_width
        )

    Ts_candidates.extend(
        [
            1.0,
            3.0,
            5.0,
            7.5,
            10.0,
            15.0,
            20.0,
            25.0,
            30.0,
            40.0,
        ]
    )

    Ts_candidates = np.asarray(
        Ts_candidates,
        dtype=float,
    )

    Ts_candidates = Ts_candidates[
        np.isfinite(Ts_candidates)
    ]

    Ts_candidates = Ts_candidates[
        (Ts_candidates >= 0.001)
        & (Ts_candidates <= 120.0)
    ]

    Ts_candidates = np.unique(
        np.round(
            Ts_candidates,
            6,
        )
    )

    c_candidates = np.asarray(
        [
            -math.pi,
            -3.0 * math.pi / 4.0,
            -math.pi / 2.0,
            -math.pi / 4.0,
            0.0,
            math.pi / 4.0,
            math.pi / 2.0,
            3.0 * math.pi / 4.0,
            math.pi,
        ],
        dtype=float,
    )

    guesses = []

    for Ts0 in Ts_candidates:
        for c0 in c_candidates:
            guesses.append(
                [
                    float(Ts0),
                    float(c0),
                ]
            )

    return guesses


# Harmonic fitting


def fit_sinusoidal_in_window(
    T_in,
    RESP_in,
    detect_res=None,
    use_detect_function=True,
    verbose=False,
):
    if (
        detect_res is None
        and use_detect_function
    ):
        detect_res = detect_window(
            T_in=T_in,
            RESP_in=RESP_in,
        )

    if detect_res is None:
        raise RuntimeError(
            "detect_res was not provided and "
            "use_detect_function is False."
        )

    T_left = float(
        detect_res["T_left"]
    )

    T_right = float(
        detect_res["T_right"]
    )

    T = np.asarray(
        T_in,
        dtype=float,
    )

    DRS = np.asarray(
        RESP_in,
        dtype=float,
    )

    # Fixed P


    P = float(
        np.nanmax(DRS)
    )

    if (
        not np.isfinite(P)
        or P <= 0
    ):
        raise RuntimeError(
            "Unable to determine a valid fixed P "
            "from the DRS maximum."
        )

    # DRS(T) inside the window


    mask = (
        np.isfinite(T)
        & np.isfinite(DRS)
        & (T >= T_left)
        & (T <= T_right)
    )

    T_win = T[mask]
    DRS_win = DRS[mask]

    if T_win.size < 4:
        raise RuntimeError(
            "Not enough points inside the detected "
            f"window for fitting: {T_win.size}"
        )

    order = np.argsort(
        T_win
    )

    T_win = T_win[order]
    DRS_win = DRS_win[order]

    # Scaling


    DRS_scaled = (
        DRS_win / P
    )

    # Initial guesses


    initial_guesses = (
        _build_initial_guesses(
            T_win=T_win,
            detect_res=detect_res,
        )
    )

    lower_bounds = [
        0.001,
        -math.pi,
    ]

    upper_bounds = [
        120.0,
        math.pi,
    ]

    # Fixed-P model


    def fixed_P_model(
        T,
        Ts,
        c,
    ):
        return spectral_sine_model(
            T,
            1.0,
            Ts,
            c,
        )

    best_popt = None
    best_ssr = np.inf

    # Nonlinear fitting


    for p0 in initial_guesses:
        try:
            popt_scaled, _ = curve_fit(
                fixed_P_model,
                T_win,
                DRS_scaled,
                p0=p0,
                bounds=(
                    lower_bounds,
                    upper_bounds,
                ),
                maxfev=200000,
            )

            residual = (
                DRS_scaled
                - fixed_P_model(
                    T_win,
                    *popt_scaled,
                )
            )

            ssr = float(
                np.sum(
                    residual ** 2
                )
            )

            if ssr < best_ssr:
                best_ssr = ssr
                best_popt = (
                    popt_scaled.copy()
                )

        except Exception:
            continue

    if best_popt is None:
        return dict(
            result=None,
            detect_res=detect_res,
            T_win=T_win,
            DRS_win=DRS_win,
        )

    # Final parameters


    Ts = float(
        best_popt[0]
    )

    c = float(
        best_popt[1]
    )

    popt = np.array(
        [
            P,
            Ts,
            c,
        ],
        dtype=float,
    )

    # Fit in original DRS units


    DRS_pred = spectral_sine_model(
        T_win,
        P,
        Ts,
        c,
    )

    ss_tot = float(
        np.sum(
            (
                DRS_win
                - np.mean(DRS_win)
            ) ** 2
        )
    )

    ss_res = float(
        np.sum(
            (
                DRS_win
                - DRS_pred
            ) ** 2
        )
    )

    R2 = (
        1.0
        - ss_res / ss_tot
        if ss_tot != 0
        else np.nan
    )

    window_width = float(
        T_right - T_left
    )

    cycles = (
        float(
            window_width / Ts
        )
        if Ts != 0
        else np.nan
    )

    result = dict(
        popt=popt,
        P=P,
        Ts=Ts,
        c=c,
        R2=float(R2),
        cycles=cycles,
        ssr=float(best_ssr),
        n_initial_guesses=int(
            len(initial_guesses)
        ),
        n_points=int(
            T_win.size
        ),
        T_left=T_left,
        T_right=T_right,
        window_width=window_width,
        window_method=detect_res.get(
            "window_method",
            detect_res.get(
                "method",
                "",
            ),
        ),
    )

    if verbose:
        print(
            f"Window: "
            f"{T_left:.6f}–{T_right:.6f} s"
        )

        print(
            f"Number of points: "
            f"{T_win.size}"
        )

        print(
            f"P = {P:.6f}"
        )

        print(
            f"Ts = {Ts:.6f} s"
        )

        print(
            f"c = {c:.6f}"
        )

        print(
            f"R2 = {R2:.6f}"
        )

        print(
            f"Cycles = {cycles:.6f}"
        )

    return dict(
        result=result,
        detect_res=detect_res,
        T_win=T_win,
        DRS_win=DRS_win,
    )