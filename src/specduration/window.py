import numpy as np
from scipy.interpolate import interp1d


DEFAULT_MIN_POINTS_IN_WINDOW = 10


# DRS(T) preparation


def preprocess_T_RESP(T_in, RESP_in):
    T = np.asarray(T_in, dtype=float)
    DRS = np.asarray(RESP_in, dtype=float)

    mask = np.isfinite(T) & np.isfinite(DRS)
    T = T[mask]
    DRS = DRS[mask]

    if T.size == 0:
        raise RuntimeError(
            "No valid period values remain after filtering NaN or Inf."
        )

    idx = np.argsort(T)

    return T[idx], DRS[idx]


def to_uniform_grid_if_needed(T, DRS, tol_rel=0.2):
    T = np.asarray(T, dtype=float)
    DRS = np.asarray(DRS, dtype=float)

    if len(T) < 3:
        return T, DRS, False

    dT = np.diff(T)
    median_dT = np.median(dT)

    if median_dT <= 0:
        return T, DRS, False

    irregularity = np.max(
        np.abs(dT - median_dT)
    ) / (median_dT + 1e-12)

    if irregularity < tol_rel:
        return T, DRS, False

    T_new = np.arange(
        T[0],
        T[-1] + 0.5 * median_dT,
        median_dT,
    )

    interpolator = interp1d(
        T,
        DRS,
        kind="cubic",
        bounds_error=False,
        fill_value="extrapolate",
    )

    DRS_new = interpolator(T_new)

    return T_new, DRS_new, True


# DRS'(T) and DRS''(T)


def compute_second_derivative(T, DRS):
    T = np.asarray(T, dtype=float)
    DRS = np.asarray(DRS, dtype=float)

    if len(T) < 5:
        raise RuntimeError(
            "At least five DRS points are required to compute DRS''(T)."
        )

    DRS_prime = np.gradient(
        DRS,
        T,
    )

    DRS_second = np.gradient(
        DRS_prime,
        T,
    )

    return DRS_prime, DRS_second


# Zero-crossing detection


def _has_real_sign_crossing(a, b):
    if not np.isfinite(a) or not np.isfinite(b):
        return False

    if a == 0.0 or b == 0.0:
        return True

    return a * b < 0.0


def _left_crossing_candidates(
    DRS_second,
    idx_peak,
):
    candidates = []

    for i in range(
        idx_peak - 1,
        0,
        -1,
    ):
        if _has_real_sign_crossing(
            DRS_second[i],
            DRS_second[i + 1],
        ):
            candidates.append(
                (
                    i + 1,
                    i,
                )
            )

    return candidates


def _right_crossing_candidates(
    DRS_second,
    idx_peak,
):
    candidates = []
    n = len(DRS_second)

    for i in range(
        idx_peak + 1,
        n - 1,
    ):
        if _has_real_sign_crossing(
            DRS_second[i - 1],
            DRS_second[i],
        ):
            candidates.append(
                (
                    i - 1,
                    i,
                )
            )

    return candidates


def _select_first_valid_crossing_pair(
    left_candidates,
    right_candidates,
    idx_peak,
    min_points,
):
    valid_pairs = []

    for i_left, left_cross in left_candidates:
        for i_right, right_cross in right_candidates:
            if (
                i_left >= idx_peak
                or i_right <= idx_peak
            ):
                continue

            n_points = (
                i_right - i_left + 1
            )

            if n_points >= min_points:
                valid_pairs.append(
                    {
                        "i_left": int(i_left),
                        "i_right": int(i_right),
                        "left_zero_index": int(
                            left_cross
                        ),
                        "right_zero_index": int(
                            right_cross
                        ),
                        "n_points": int(
                            n_points
                        ),
                        "width_index": int(
                            i_right - i_left
                        ),
                    }
                )

    if not valid_pairs:
        return None

    return sorted(
        valid_pairs,
        key=lambda x: x["width_index"],
    )[0]


# Fallback window


def _low_curvature_fallback(
    abs_DRS_second,
    idx_peak,
    n,
    min_points,
):
    min_side_distance = max(
        min_points // 2,
        3,
    )

    left_end = (
        idx_peak - min_side_distance
    )

    right_start = (
        idx_peak + min_side_distance
    )

    if left_end <= 0:
        i_left = max(
            0,
            idx_peak - min_side_distance,
        )
        left_zero_index = None

    else:
        left_range = np.arange(
            0,
            left_end + 1,
        )

        local = abs_DRS_second[
            left_range
        ]

        i_left = int(
            left_range[
                np.nanargmin(local)
            ]
        )

        left_zero_index = i_left

    if right_start >= n - 1:
        i_right = min(
            n - 1,
            idx_peak + min_side_distance,
        )
        right_zero_index = None

    else:
        right_range = np.arange(
            right_start,
            n,
        )

        local = abs_DRS_second[
            right_range
        ]

        i_right = int(
            right_range[
                np.nanargmin(local)
            ]
        )

        right_zero_index = i_right

    if i_left >= idx_peak:
        i_left = max(
            0,
            idx_peak - min_side_distance,
        )

    if i_right <= idx_peak:
        i_right = min(
            n - 1,
            idx_peak + min_side_distance,
        )

    return {
        "i_left": int(i_left),
        "i_right": int(i_right),
        "left_zero_index": left_zero_index,
        "right_zero_index": right_zero_index,
    }


# Window extension


def _extend_window_to_min_points(
    i_left,
    i_right,
    idx_peak,
    n,
    min_points,
):
    i_left = int(i_left)
    i_right = int(i_right)

    if i_left > idx_peak:
        i_left = idx_peak

    if i_right < idx_peak:
        i_right = idx_peak

    initial_left = i_left
    initial_right = i_right

    while (
        i_right - i_left + 1
    ) < min_points:
        can_left = i_left > 0
        can_right = i_right < n - 1

        if not can_left and not can_right:
            break

        left_distance = (
            idx_peak - i_left
        )

        right_distance = (
            i_right - idx_peak
        )

        if can_left and can_right:
            if left_distance <= right_distance:
                i_left -= 1
            else:
                i_right += 1

        elif can_left:
            i_left -= 1

        else:
            i_right += 1

    return (
        int(i_left),
        int(i_right),
        int(initial_left - i_left),
        int(i_right - initial_right),
    )


# Window diagnostics


def _extract_DRS_second_derivative_diagnostics(
    T,
    DRS_second,
    i_left,
    i_right,
    left_zero_index,
    right_zero_index,
):
    n = len(DRS_second)

    DRS_second_left = (
        float(DRS_second[i_left])
        if 0 <= i_left < n
        else np.nan
    )

    DRS_second_right = (
        float(DRS_second[i_right])
        if 0 <= i_right < n
        else np.nan
    )

    if (
        left_zero_index is not None
        and 0 <= int(left_zero_index) < n
    ):
        i_left_outer = int(
            left_zero_index
        )
    else:
        i_left_outer = (
            i_left - 1
            if i_left - 1 >= 0
            else None
        )

    if (
        right_zero_index is not None
        and 0 <= int(right_zero_index) < n
    ):
        i_right_outer = int(
            right_zero_index
        )
    else:
        i_right_outer = right_zero_index

        if i_right_outer is None:
            i_right_outer = (
                i_right + 1
                if i_right + 1 < n
                else None
            )

    DRS_second_left_outer = (
        float(
            DRS_second[i_left_outer]
        )
        if (
            i_left_outer is not None
            and 0 <= i_left_outer < n
        )
        else np.nan
    )

    DRS_second_right_outer = (
        float(
            DRS_second[i_right_outer]
        )
        if (
            i_right_outer is not None
            and 0 <= i_right_outer < n
        )
        else np.nan
    )

    T_left_outer = (
        float(T[i_left_outer])
        if (
            i_left_outer is not None
            and 0 <= i_left_outer < n
        )
        else np.nan
    )

    T_right_outer = (
        float(T[i_right_outer])
        if (
            i_right_outer is not None
            and 0 <= i_right_outer < n
        )
        else np.nan
    )

    return {
        "DRS_second_left": DRS_second_left,
        "DRS_second_left_outer": (
            DRS_second_left_outer
        ),
        "DRS_second_right": DRS_second_right,
        "DRS_second_right_outer": (
            DRS_second_right_outer
        ),
        "T_left_outer": T_left_outer,
        "T_right_outer": T_right_outer,
        "idx_left_outer": (
            int(i_left_outer)
            if i_left_outer is not None
            else None
        ),
        "idx_right_outer": (
            int(i_right_outer)
            if i_right_outer is not None
            else None
        ),
    }


# Main window detection


def detect_window_by_second_derivative(
    T_in,
    RESP_in,
    min_points_in_window=DEFAULT_MIN_POINTS_IN_WINDOW,
    extend_if_needed=True,
):
    T0, DRS0 = preprocess_T_RESP(
        T_in,
        RESP_in,
    )

    idx_peak_original = int(
        np.nanargmax(DRS0)
    )

    T_peak_original = float(
        T0[idx_peak_original]
    )

    T_u, DRS_u, regridded = (
        to_uniform_grid_if_needed(
            T0,
            DRS0,
        )
    )

    T_u = np.asarray(
        T_u,
        dtype=float,
    )

    DRS_u = np.asarray(
        DRS_u,
        dtype=float,
    )

    n = len(T_u)

    if n < 5:
        raise RuntimeError(
            "Too few DRS points are available to detect a window."
        )

    idx_peak = int(
        np.argmin(
            np.abs(
                T_u - T_peak_original
            )
        )
    )

    T_peak = float(
        T_u[idx_peak]
    )

    DRS_prime, DRS_second = (
        compute_second_derivative(
            T_u,
            DRS_u,
        )
    )

    abs_DRS_second = np.abs(
        DRS_second
    )

    left_candidates = (
        _left_crossing_candidates(
            DRS_second,
            idx_peak,
        )
    )

    right_candidates = (
        _right_crossing_candidates(
            DRS_second,
            idx_peak,
        )
    )

    selected = (
        _select_first_valid_crossing_pair(
            left_candidates=left_candidates,
            right_candidates=right_candidates,
            idx_peak=idx_peak,
            min_points=min_points_in_window,
        )
    )

    window_extended = False
    extension_points_left = 0
    extension_points_right = 0

    if selected is not None:
        i_left = selected["i_left"]
        i_right = selected["i_right"]
        left_zero_index = selected[
            "left_zero_index"
        ]
        right_zero_index = selected[
            "right_zero_index"
        ]

        method = (
            "second_derivative_real_zero_crossing"
        )

    else:
        fallback = _low_curvature_fallback(
            abs_DRS_second=abs_DRS_second,
            idx_peak=idx_peak,
            n=n,
            min_points=min_points_in_window,
        )

        i_left = fallback["i_left"]
        i_right = fallback["i_right"]
        left_zero_index = fallback[
            "left_zero_index"
        ]
        right_zero_index = fallback[
            "right_zero_index"
        ]

        method = (
            "second_derivative_low_curvature_fallback"
        )

    n_points_window_raw = int(
        i_right - i_left + 1
    )

    if (
        n_points_window_raw
        < min_points_in_window
        and extend_if_needed
    ):
        window_extended = True

        (
            i_left,
            i_right,
            extension_points_left,
            extension_points_right,
        ) = _extend_window_to_min_points(
            i_left=i_left,
            i_right=i_right,
            idx_peak=idx_peak,
            n=n,
            min_points=min_points_in_window,
        )

        method = (
            method + "_extended"
        )

    n_points_window = int(
        i_right - i_left + 1
    )

    T_left = float(
        T_u[i_left]
    )

    T_right = float(
        T_u[i_right]
    )

    valid_window = bool(
        n_points_window
        >= min_points_in_window
    )

    n_points_original = int(
        np.sum(
            (T0 >= T_left)
            & (T0 <= T_right)
        )
    )

    max_abs_DRS_second = (
        float(
            np.nanmax(
                abs_DRS_second
            )
        )
        if abs_DRS_second.size > 0
        else np.nan
    )

    DRS_second_diag = (
        _extract_DRS_second_derivative_diagnostics(
            T=T_u,
            DRS_second=DRS_second,
            i_left=i_left,
            i_right=i_right,
            left_zero_index=left_zero_index,
            right_zero_index=right_zero_index,
        )
    )

    return {
        "T_left": T_left,
        "T_right": T_right,
        "T_peak": T_peak,
        "idx_peak": idx_peak,
        "idx_left": int(i_left),
        "idx_right": int(i_right),
        "idx_peak_original": idx_peak_original,
        "left_zero_index": left_zero_index,
        "right_zero_index": right_zero_index,
        "T_u": T_u,
        "DRS_u": DRS_u,
        "DRS_prime": DRS_prime,
        "DRS_second": DRS_second,
        "method": method,
        "window_method": method,
        "regridded": bool(regridded),
        "n_points_orig": n_points_original,
        "n_points_window": n_points_window,
        "n_points_window_raw": (
            n_points_window_raw
        ),
        "min_points_in_window": int(
            min_points_in_window
        ),
        "valid_window": valid_window,
        "window_extended": bool(
            window_extended
        ),
        "extension_points_left": int(
            extension_points_left
        ),
        "extension_points_right": int(
            extension_points_right
        ),
        "max_abs_DRS_second": (
            max_abs_DRS_second
        ),
        "DRS_second_left": (
            DRS_second_diag[
                "DRS_second_left"
            ]
        ),
        "DRS_second_left_outer": (
            DRS_second_diag[
                "DRS_second_left_outer"
            ]
        ),
        "DRS_second_right": (
            DRS_second_diag[
                "DRS_second_right"
            ]
        ),
        "DRS_second_right_outer": (
            DRS_second_diag[
                "DRS_second_right_outer"
            ]
        ),
        "T_left_outer": (
            DRS_second_diag[
                "T_left_outer"
            ]
        ),
        "T_right_outer": (
            DRS_second_diag[
                "T_right_outer"
            ]
        ),
        "idx_left_outer": (
            DRS_second_diag[
                "idx_left_outer"
            ]
        ),
        "idx_right_outer": (
            DRS_second_diag[
                "idx_right_outer"
            ]
        ),
    }


def detect_window(
    T_in,
    RESP_in,
    min_points_in_window=DEFAULT_MIN_POINTS_IN_WINDOW,
    extend_if_needed=True,
):
    return detect_window_by_second_derivative(
        T_in=T_in,
        RESP_in=RESP_in,
        min_points_in_window=min_points_in_window,
        extend_if_needed=extend_if_needed,
    )