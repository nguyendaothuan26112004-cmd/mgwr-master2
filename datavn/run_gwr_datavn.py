"""Run GWR on datavn/features.csv and export local coefficients.

Usage examples:
  python run_gwr_datavn.py
  python run_gwr_datavn.py --y-col dens --x-cols nightlights_avg built_v_avg elevation_avg
  python run_gwr_datavn.py --map-col elevation_avg_coef --plot
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Sequence

import numpy as np
import pandas as pd
from mgwr.gwr import GWR
from mgwr.sel_bw import Sel_BW


def _validate_columns(df: pd.DataFrame, cols: Sequence[str], role: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing {role} columns: {missing}")


def _pick_default_xcols(df: pd.DataFrame) -> List[str]:
    preferred = ["nightlights_avg", "built_v_avg", "elevation_avg"]
    if all(c in df.columns for c in preferred):
        return preferred

    fallback_pool = [
        "esalc_190_avg",
        "esalc_160_avg",
        "esalc_150_avg",
        "esalc_140_avg",
        "esalc_130_avg",
        "wdpa_dist_avg",
        "coastline_dist_avg",
        "road_intr_dist_avg",
        "highway_dist_avg",
    ]
    fallback = [c for c in fallback_pool if c in df.columns][:3]
    if len(fallback) < 3:
        raise ValueError(
            "Cannot infer default x columns. Please pass --x-cols explicitly."
        )
    return fallback


def run_gwr(
    features_csv: Path,
    output_csv: Path,
    y_col: str,
    x_cols: Sequence[str],
    coords_cols: Sequence[str],
    kernel: str,
    fixed: bool,
    criterion: str,
    standardize: bool,
    n_jobs: int,
) -> dict:
    df = pd.read_csv(features_csv)

    required_cols = [y_col, *x_cols, *coords_cols]
    _validate_columns(df, required_cols, "input")

    # Keep only complete rows for model fitting.
    model_df = df.dropna(subset=required_cols).copy()
    if model_df.empty:
        raise ValueError("No rows left after dropping NaN in selected columns.")

    y = model_df[y_col].to_numpy(dtype=float).reshape((-1, 1))
    X = model_df[list(x_cols)].to_numpy(dtype=float)
    coords = list(
        zip(
            model_df[coords_cols[0]].to_numpy(dtype=float),
            model_df[coords_cols[1]].to_numpy(dtype=float),
        )
    )

    if standardize:
        X_mean = X.mean(axis=0)
        X_std = X.std(axis=0)
        X_std[X_std == 0] = 1.0
        X = (X - X_mean) / X_std

        y_mean = y.mean(axis=0)
        y_std = y.std(axis=0)
        y_std[y_std == 0] = 1.0
        y = (y - y_mean) / y_std

    selector = Sel_BW(coords, y, X, kernel=kernel, fixed=fixed, n_jobs=n_jobs)
    bw = selector.search(criterion=criterion)

    gwr_model = GWR(coords, y, X, bw, kernel=kernel, fixed=fixed, n_jobs=n_jobs)
    gwr_results = gwr_model.fit()

    params = gwr_results.params
    local_r2 = getattr(gwr_results, "localR2", None)

    # Add local coefficients back to original frame using row index alignment.
    out_df = df.copy()
    coef_names = ["Intercept_coef", *[f"{col}_coef" for col in x_cols]]
    coef_df = pd.DataFrame(params, columns=coef_names, index=model_df.index)

    for col in coef_names:
        out_df[col] = np.nan
        out_df.loc[coef_df.index, col] = coef_df[col]

    if local_r2 is not None:
        out_df["Local_R2"] = np.nan
        out_df.loc[model_df.index, "Local_R2"] = np.asarray(local_r2).reshape(-1)

    out_df.to_csv(output_csv, index=False)

    return {
        "bw": bw,
        "n_rows_used": int(model_df.shape[0]),
        "aicc": float(gwr_results.aicc),
        "r2": float(getattr(gwr_results, "R2", np.nan)),
        "has_local_r2": local_r2 is not None,
        "output_csv": str(output_csv),
        "coef_cols": coef_names,
    }


def export_map(
    shapefile_path: Path,
    gwr_csv_path: Path,
    map_col: str,
    output_png: Path,
) -> None:
    import geopandas as gpd
    import matplotlib.pyplot as plt

    gdf = gpd.read_file(shapefile_path)
    df = pd.read_csv(gwr_csv_path)

    if "id" in gdf.columns and "id" in df.columns:
        map_df = gdf.merge(df[["id", map_col]], on="id", how="left")
    elif len(gdf) == len(df):
        map_df = gdf.copy()
        map_df[map_col] = df[map_col].values
    else:
        raise ValueError(
            "Cannot align shape and result rows. Provide matching 'id' in both datasets."
        )

    ax = map_df.plot(
        column=map_col,
        cmap="RdYlBu_r",
        legend=True,
        figsize=(10, 14),
        edgecolor="black",
        linewidth=0.1,
        missing_kwds={"color": "lightgray", "label": "No data"},
    )
    ax.set_title(f"Vietnam GWR Local Coefficient: {map_col}")
    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GWR on Vietnam datavn dataset.")
    parser.add_argument(
        "--features-csv",
        default="datavn/features.csv",
        help="Path to input features CSV.",
    )
    parser.add_argument(
        "--output-csv",
        default="datavn/gwr_result.csv",
        help="Path to output CSV containing local coefficients.",
    )
    parser.add_argument(
        "--y-col",
        default="dens",
        help="Dependent variable column.",
    )
    parser.add_argument(
        "--x-cols",
        nargs="+",
        default=None,
        help="Independent variable columns.",
    )
    parser.add_argument(
        "--coords-cols",
        nargs=2,
        default=["lon", "lat"],
        metavar=("LON_COL", "LAT_COL"),
        help="Coordinate columns in features CSV.",
    )
    parser.add_argument(
        "--kernel",
        default="bisquare",
        choices=["gaussian", "bisquare", "exponential"],
        help="Kernel type.",
    )
    parser.add_argument(
        "--fixed",
        action="store_true",
        help="Use fixed bandwidth. Default is adaptive.",
    )
    parser.add_argument(
        "--criterion",
        default="AICc",
        choices=["AICc", "AIC", "BIC", "CV"],
        help="Bandwidth search criterion.",
    )
    parser.add_argument(
        "--no-standardize",
        action="store_true",
        help="Disable standardization of y and X.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Parallel jobs for bandwidth search and fit.",
    )

    parser.add_argument(
        "--plot",
        action="store_true",
        help="Also export a PNG map from shapefile + coefficient column.",
    )
    parser.add_argument(
        "--shapefile",
        default="datavn/Census2019_All_of_Vietnam.shp",
        help="Path to Vietnam shapefile.",
    )
    parser.add_argument(
        "--map-col",
        default=None,
        help="Coefficient column to map, e.g. nightlights_avg_coef.",
    )
    parser.add_argument(
        "--output-png",
        default="datavn/gwr_map.png",
        help="Output PNG path when --plot is set.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    features_csv = Path(args.features_csv)
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    if not features_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {features_csv}")

    x_cols = args.x_cols
    if x_cols is None:
        x_cols = _pick_default_xcols(pd.read_csv(features_csv, nrows=5))

    report = run_gwr(
        features_csv=features_csv,
        output_csv=output_csv,
        y_col=args.y_col,
        x_cols=x_cols,
        coords_cols=args.coords_cols,
        kernel=args.kernel,
        fixed=args.fixed,
        criterion=args.criterion,
        standardize=not args.no_standardize,
        n_jobs=args.n_jobs,
    )

    print("GWR finished")
    print(f"- Rows used: {report['n_rows_used']}")
    print(f"- Bandwidth: {report['bw']}")
    print(f"- AICc: {report['aicc']}")
    print(f"- R2: {report['r2']}")
    print(f"- Local R2 exported: {report['has_local_r2']}")
    print(f"- Output CSV: {report['output_csv']}")
    print(f"- Coefficient columns: {', '.join(report['coef_cols'])}")

    if args.plot:
        if not args.map_col:
            # Default to first non-intercept coefficient.
            args.map_col = report["coef_cols"][1]

        export_map(
            shapefile_path=Path(args.shapefile),
            gwr_csv_path=output_csv,
            map_col=args.map_col,
            output_png=Path(args.output_png),
        )
        print(f"- Output map: {args.output_png}")


if __name__ == "__main__":
    main()
