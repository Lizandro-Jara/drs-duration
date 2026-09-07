import numpy as np
import matplotlib.pyplot as plt


# Plot helpers


def _clean_xy(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if x.size == 0:
        raise RuntimeError("No valid data available for plotting.")

    order = np.argsort(x)
    return x[order], y[order]


def _safe_ylim(y, margin=0.08):
    y = np.asarray(y, dtype=float)
    y = y[np.isfinite(y)]

    if y.size == 0:
        return None

    ymin = float(np.nanmin(y))
    ymax = float(np.nanmax(y))

    if np.isclose(ymin, ymax):
        delta = abs(ymax) * 0.10 + 1e-12
        return ymin - delta, ymax + delta

    delta = (ymax - ymin) * margin
    return ymin - delta, ymax + delta


def _get_window_limits(
    T_values,
    T_left,
    T_right,
    left_frac=0.25,
    right_frac=0.25,
):
    T_values = np.asarray(T_values, dtype=float)

    width = max(
        float(T_right) - float(T_left),
        1e-12,
    )

    x_min = max(
        float(np.min(T_values)),
        float(T_left) - left_frac * width,
    )

    x_max = min(
        float(np.max(T_values)),
        float(T_right) + right_frac * width,
    )

    if x_max <= x_min:
        x_min = float(np.min(T_values))
        x_max = float(np.max(T_values))

    return x_min, x_max


def _get_second_derivative_array(detect_result):
    if "DRS_second" in detect_result:
        return np.asarray(
            detect_result["DRS_second"],
            dtype=float,
        )

    raise KeyError(
        "detect_result does not contain 'DRS_second'."
    )


def _get_model_function():
    from .fit import spectral_sine_model

    return spectral_sine_model


# Full DRS(T)


def save_drs_full_png(
    T_values,
    RESP,
    outpath,
    title="Displacement Response Spectrum (DRS)",
    dpi=300,
):
    T_values, RESP = _clean_xy(
        T_values,
        RESP,
    )

    plt.figure(figsize=(10.5, 5.8))

    plt.plot(
        T_values,
        RESP,
        lw=2.2,
        label="DRS",
    )

    plt.grid(
        True,
        ls="--",
        lw=0.6,
        alpha=0.65,
    )

    plt.xlabel(
        "Period T [s]",
        fontsize=13,
    )

    plt.ylabel(
        "Peak relative displacement |u| [m]",
        fontsize=13,
    )

    plt.title(
        title,
        fontsize=17,
        fontweight="bold",
    )

    plt.xlim(
        float(np.min(T_values)),
        float(np.max(T_values)),
    )

    ylim = _safe_ylim(RESP)

    if ylim is not None:
        plt.ylim(
            max(0.0, ylim[0]),
            ylim[1],
        )

    plt.legend(
        fontsize=12,
        loc="best",
    )

    plt.tight_layout()
    plt.savefig(
        outpath,
        dpi=dpi,
    )
    plt.close()


# DRS(T) window and harmonic fit


def save_drs_window_png(
    T_values,
    RESP,
    detect_result,
    outpath,
    fit_result=None,
    title="DRS window detection and harmonic fit",
    dpi=300,
):
    T_values, RESP = _clean_xy(
        T_values,
        RESP,
    )

    T_peak = float(
        detect_result["T_peak"]
    )

    T_left = float(
        detect_result["T_left"]
    )

    T_right = float(
        detect_result["T_right"]
    )

    idx_peak = int(
        np.argmin(
            np.abs(T_values - T_peak)
        )
    )

    DRS_peak = float(
        RESP[idx_peak]
    )

    mask_window = (
        (T_values >= T_left)
        & (T_values <= T_right)
    )

    x_min, x_max = _get_window_limits(
        T_values=T_values,
        T_left=T_left,
        T_right=T_right,
        left_frac=0.25,
        right_frac=0.25,
    )

    local_mask = (
        (T_values >= x_min)
        & (T_values <= x_max)
    )

    plt.figure(figsize=(10.8, 5.9))

    plt.plot(
        T_values[local_mask],
        RESP[local_mask],
        lw=2.2,
        label="DRS(T)",
    )

    if (
        fit_result is not None
        and "popt" in fit_result
    ):
        model_fun = _get_model_function()

        T_win = T_values[mask_window]

        if T_win.size >= 2:
            T_dense = np.linspace(
                float(np.min(T_win)),
                float(np.max(T_win)),
                900,
            )

            DRS_fit = model_fun(
                T_dense,
                *fit_result["popt"],
            )

            if "R2" in fit_result:
                fit_label = (
                    f"Harmonic fit "
                    f"(R² = {fit_result['R2']:.3f})"
                )
            else:
                fit_label = "Harmonic fit"

            plt.plot(
                T_dense,
                DRS_fit,
                lw=2.0,
                label=fit_label,
            )

    plt.scatter(
        [T_peak],
        [DRS_peak],
        s=80,
        zorder=5,
        label=f"T_peak = {T_peak:.3f} s",
    )

    plt.axvline(
        T_left,
        ls="--",
        lw=1.6,
        label=f"T_left = {T_left:.3f} s",
    )

    plt.axvline(
        T_right,
        ls="--",
        lw=1.6,
        label=f"T_right = {T_right:.3f} s",
    )

    plt.axvline(
        T_peak,
        ls=":",
        lw=1.8,
    )

    Ts_text = ""

    if (
        fit_result is not None
        and "Ts" in fit_result
    ):
        Ts_text = (
            f" | Ts = "
            f"{fit_result['Ts']:.3f} s"
        )

    plt.grid(
        True,
        ls="--",
        lw=0.6,
        alpha=0.65,
    )

    plt.xlabel(
        "Period T [s]",
        fontsize=13,
    )

    plt.ylabel(
        "Peak relative displacement |u| [m]",
        fontsize=13,
    )

    plt.title(
        f"{title}{Ts_text}",
        fontsize=17,
        fontweight="bold",
    )

    plt.xlim(
        x_min,
        x_max,
    )

    ylim = _safe_ylim(
        RESP[local_mask]
    )

    if ylim is not None:
        plt.ylim(
            max(0.0, ylim[0]),
            ylim[1],
        )

    plt.legend(
        fontsize=11,
        loc="best",
    )

    plt.tight_layout()
    plt.savefig(
        outpath,
        dpi=dpi,
    )
    plt.close()


# Harmonic fit


def save_drs_fit_overlay_png(
    T_values,
    RESP,
    detect_result,
    fit_result,
    outpath,
    title="Harmonic fitting inside detected window",
    dpi=300,
):
    T_values, RESP = _clean_xy(
        T_values,
        RESP,
    )

    T_left = float(
        detect_result["T_left"]
    )

    T_right = float(
        detect_result["T_right"]
    )

    mask_window = (
        (T_values >= T_left)
        & (T_values <= T_right)
    )

    T_win = T_values[mask_window]
    DRS_win = RESP[mask_window]

    if T_win.size < 2:
        raise RuntimeError(
            "Not enough points inside the detected window to plot the fit."
        )

    plt.figure(figsize=(10.8, 5.9))

    plt.plot(
        T_win,
        DRS_win,
        lw=2.2,
        label="DRS(T) window",
    )

    if (
        fit_result is not None
        and "popt" in fit_result
    ):
        model_fun = _get_model_function()

        T_dense = np.linspace(
            float(np.min(T_win)),
            float(np.max(T_win)),
            900,
        )

        DRS_fit = model_fun(
            T_dense,
            *fit_result["popt"],
        )

        plt.plot(
            T_dense,
            DRS_fit,
            lw=2.0,
            label=(
                f"Harmonic fit "
                f"(R² = {fit_result['R2']:.3f})"
            ),
        )

    Ts_text = ""

    if (
        fit_result is not None
        and "Ts" in fit_result
    ):
        Ts_text = (
            f" | Ts = "
            f"{fit_result['Ts']:.3f} s"
        )

    plt.title(
        f"{title}{Ts_text}",
        fontsize=17,
        fontweight="bold",
    )

    plt.xlabel(
        "Period T [s]",
        fontsize=13,
    )

    plt.ylabel(
        "DRS(T)",
        fontsize=13,
    )

    plt.grid(
        True,
        ls="--",
        lw=0.6,
        alpha=0.65,
    )

    ylim = _safe_ylim(
        DRS_win
    )

    if ylim is not None:
        plt.ylim(
            max(0.0, ylim[0]),
            ylim[1],
        )

    plt.legend(
        fontsize=11,
        loc="best",
    )

    plt.tight_layout()
    plt.savefig(
        outpath,
        dpi=dpi,
    )
    plt.close()


# DRS''(T)


def save_second_derivative_window_png(
    detect_result,
    outpath,
    title="Second spectral derivative detector",
    dpi=300,
):
    T = np.asarray(
        detect_result["T_u"],
        dtype=float,
    )

    DRS_second = _get_second_derivative_array(
        detect_result
    )

    T, DRS_second = _clean_xy(
        T,
        DRS_second,
    )

    T_left = float(
        detect_result["T_left"]
    )

    T_right = float(
        detect_result["T_right"]
    )

    T_peak = float(
        detect_result["T_peak"]
    )

    width = max(
        T_right - T_left,
        1e-12,
    )

    x_min = max(
        float(np.min(T)),
        T_left - 0.35 * width,
    )

    x_max = min(
        float(np.max(T)),
        T_right + 0.35 * width,
    )

    local_mask = (
        (T >= x_min)
        & (T <= x_max)
    )

    window_mask = (
        (T >= T_left)
        & (T <= T_right)
    )

    T_local = T[local_mask]
    DRS_second_local = DRS_second[
        local_mask
    ]

    T_win = T[window_mask]
    DRS_second_win = DRS_second[
        window_mask
    ]

    plt.figure(figsize=(10.8, 5.8))

    plt.plot(
        T_local,
        DRS_second_local,
        lw=2.0,
        label=r"$DRS''(T)$",
    )

    if T_win.size > 0:
        plt.plot(
            T_win,
            DRS_second_win,
            lw=2.2,
            label=(
                r"$DRS''(T)$ inside detected window"
            ),
        )

    plt.axhline(
        0.0,
        lw=1.2,
        label=r"$DRS''(T)=0$",
    )

    plt.axvline(
        T_left,
        ls="--",
        lw=1.6,
        label=f"T_left = {T_left:.3f} s",
    )

    plt.axvline(
        T_peak,
        ls=":",
        lw=1.8,
        label=f"T_peak = {T_peak:.3f} s",
    )

    plt.axvline(
        T_right,
        ls="--",
        lw=1.6,
        label=f"T_right = {T_right:.3f} s",
    )

    DRS_second_local_finite = (
        DRS_second_local[
            np.isfinite(DRS_second_local)
        ]
    )

    if DRS_second_local_finite.size > 0:
        y_low = float(
            np.nanpercentile(
                DRS_second_local_finite,
                1,
            )
        )

        y_high = float(
            np.nanpercentile(
                DRS_second_local_finite,
                99,
            )
        )

        if T_win.size > 0:
            DRS_second_win_finite = (
                DRS_second_win[
                    np.isfinite(DRS_second_win)
                ]
            )

            if DRS_second_win_finite.size > 0:
                y_low = min(
                    y_low,
                    float(
                        np.nanmin(
                            DRS_second_win_finite
                        )
                    ),
                )

                y_high = max(
                    y_high,
                    float(
                        np.nanmax(
                            DRS_second_win_finite
                        )
                    ),
                )

        y_low = min(y_low, 0.0)
        y_high = max(y_high, 0.0)

        min_range = 0.50
        current_range = y_high - y_low

        if current_range < min_range:
            center = 0.5 * (
                y_low + y_high
            )

            y_low = (
                center
                - min_range / 2.0
            )

            y_high = (
                center
                + min_range / 2.0
            )

            y_low = min(y_low, 0.0)
            y_high = max(y_high, 0.0)

        margin = 0.12 * (
            y_high - y_low
        )

        y_low_plot = y_low - margin
        y_high_plot = y_high + margin

        plt.ylim(
            y_low_plot,
            y_high_plot,
        )

        real_min = float(
            np.nanmin(
                DRS_second_local_finite
            )
        )

        real_max = float(
            np.nanmax(
                DRS_second_local_finite
            )
        )

        if (
            real_min < y_low_plot
            or real_max > y_high_plot
        ):
            plt.text(
                0.02,
                0.03,
                "Y-axis clipped for visualization",
                transform=plt.gca().transAxes,
                fontsize=9,
                alpha=0.70,
            )

    plt.grid(
        True,
        ls="--",
        lw=0.6,
        alpha=0.65,
    )

    plt.xlabel(
        "Period T [s]",
        fontsize=13,
    )

    plt.ylabel(
        r"$DRS''(T)$",
        fontsize=15,
    )

    plt.title(
        title,
        fontsize=17,
        fontweight="bold",
    )

    plt.xlim(
        x_min,
        x_max,
    )

    plt.legend(
        fontsize=10,
        loc="best",
    )

    plt.tight_layout()
    plt.savefig(
        outpath,
        dpi=dpi,
    )
    plt.close()


# Compatibility functions


def save_spectrum_fit_overlay_png(
    T_values,
    RESP,
    detect_result,
    fit_result,
    outpath,
    title="Harmonic fitting inside detected window",
    dpi=300,
):
    save_drs_fit_overlay_png(
        T_values=T_values,
        RESP=RESP,
        detect_result=detect_result,
        fit_result=fit_result,
        outpath=outpath,
        title=title,
        dpi=dpi,
    )


def save_drs_window_with_fit_inset_png(
    T_values,
    RESP,
    detect_result,
    fit_result,
    outpath,
    title="DRS window detection and harmonic fit",
    dpi=300,
):
    save_drs_window_png(
        T_values=T_values,
        RESP=RESP,
        detect_result=detect_result,
        fit_result=fit_result,
        outpath=outpath,
        title=title,
        dpi=dpi,
    )


def save_drs_zoom_png(
    T_zoom,
    RESP_zoom,
    T_peak,
    Dmax,
    outpath,
    title=None,
    dpi=300,
):
    T_zoom, RESP_zoom = _clean_xy(
        T_zoom,
        RESP_zoom,
    )

    T_peak = float(T_peak)
    Dmax = float(Dmax)

    plt.figure(figsize=(10, 5))

    plt.plot(
        T_zoom,
        RESP_zoom,
        lw=2,
        label="DRS(T) zoom",
    )

    plt.scatter(
        [T_peak],
        [Dmax],
        s=65,
        zorder=5,
        label=f"T_peak = {T_peak:.3f} s",
    )

    plt.axvline(
        T_peak,
        ls="--",
        lw=1.2,
    )

    plt.grid(
        True,
        ls="--",
        lw=0.6,
        alpha=0.65,
    )

    plt.xlabel(
        "Period T [s]",
        fontsize=12,
    )

    plt.ylabel(
        "Peak relative displacement |u| [m]",
        fontsize=12,
    )

    if title is None:
        title = (
            f"DRS(T) zoom around "
            f"T_peak = {T_peak:.3f} s"
        )

    plt.title(
        title,
        fontsize=14,
        fontweight="bold",
    )

    ylim = _safe_ylim(
        RESP_zoom
    )

    if ylim is not None:
        plt.ylim(
            max(0.0, ylim[0]),
            ylim[1],
        )

    plt.legend(
        fontsize=11,
        loc="best",
    )

    plt.tight_layout()
    plt.savefig(
        outpath,
        dpi=dpi,
    )
    plt.close()