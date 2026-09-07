import argparse

from .pipeline import PipelineConfig, run_pipeline


def main():
    parser = argparse.ArgumentParser(
        prog="specduration",
        description=(
            "Estimate spectral seismic duration from the displacement "
            "response spectrum using a second-derivative window detector."
        ),
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to the input ground-motion record in .txt format.",
    )

    parser.add_argument(
        "--out",
        required=True,
        help="Output directory where result files and figures will be saved.",
    )

    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Run the analysis without generating figures.",
    )

    parser.add_argument(
        "--zeta",
        type=float,
        default=0.05,
        help="Damping ratio used to compute the DRS. Default: 0.05.",
    )

    parser.add_argument(
        "--tmax",
        type=float,
        default=15.0,
        help="Maximum period for the DRS computation, in seconds. Default: 15.0.",
    )

    parser.add_argument(
        "--dT",
        type=float,
        default=0.01,
        help="Spectral period step used to compute the DRS, in seconds. Default: 0.01.",
    )

    parser.add_argument(
        "--min-points-window",
        type=int,
        default=10,
        help="Minimum number of points required in the detected spectral window. Default: 10.",
    )

    parser.add_argument(
        "--min-points-fit",
        type=int,
        default=10,
        help="Minimum number of points required for the harmonic fit. Default: 10.",
    )

    parser.add_argument(
        "--min-R2-fit",
        type=float,
        default=0.98,
        help="Minimum R² required to accept the harmonic fit. Default: 0.98.",
    )

    parser.add_argument(
        "--no-extend-window",
        action="store_true",
        help=(
            "Do not extend the detected window when it has fewer points "
            "than the minimum required."
        ),
    )

    args = parser.parse_args()

    cfg = PipelineConfig(
        zeta=args.zeta,
        tmax=args.tmax,
        dT=args.dT,
        min_points_window=args.min_points_window,
        min_points_fit=args.min_points_fit,
        min_R2_fit=args.min_R2_fit,
        extend_window_if_needed=(
            not args.no_extend_window
        ),
    )

    result = run_pipeline(
        input_path=args.input,
        out_dir=args.out,
        cfg=cfg,
        make_plots=(not args.no_plots),
    )

    print("\n=== SPECduration results ===")

    keys_to_show = [
        "record",
        "dt_record",
        "zeta",
        "dT_DRS",
        "T_peak",
        "T_left",
        "T_right",
        "n_points_window",
        "Ts",
        "R2",
        "valid_solution",
    ]

    for key in keys_to_show:
        if key in result:
            print(f"{key}: {result[key]}")

    print("\nMain output:")
    print("- results.xlsx")

    print("\nAdditional CSV outputs:")
    print("- summary.csv")
    print("- results_diagnostics.csv")
    print("- drs_full.csv")
    print("- second_derivative.csv")

    if not args.no_plots:
        print("\nFigures:")
        print("- drs_full.png")
        print("- drs_window.png")
        print("- second_derivative_window.png")


if __name__ == "__main__":
    main()