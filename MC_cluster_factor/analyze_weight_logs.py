#!/usr/bin/env python3
"""
analyze_weight_logs.py

Utility script to summarise StreamingLogger weight logs into a multi-sheet Excel
workbook for quick inspection of neighbourhood-level metrics.

For each target cluster category (e.g. non-durable, durable, services) the script:
  1. Filters log rows where the source node belongs to that cluster.
  2. Randomly selects up to five distinct source nodes from the cluster.
  3. Expands every neighbour interaction for the sampled nodes, preserving the
     similarity components, final weight, and household-level attributes that were
     added to the logger (adults, total spending on food, redmeat, cult1, income,
     HC24, cult) for both the source node and its neighbour.
  4. Writes the rows to a dedicated sheet in an Excel workbook.

Usage
-----
    python analyze_weight_logs.py \
        --logs /absolute/path/to/log_dir \
        --output /absolute/path/to/report.xlsx

Arguments
---------
--logs      Directory containing one or more `weight_logs_*.csv` files, or a
           single CSV file path.
--output   Destination Excel file. Defaults to `weight_log_summary.xlsx` in the
           working directory.
--seed     Optional RNG seed for reproducible sampling (default: 1996).
--mapping  Optional JSON or Python literal mapping (cluster_id: label). If
           omitted, a default {0: 'non_durable', 1: 'durable', 2: 'services'}
           mapping is used. Unmapped cluster ids fall back to `cluster_<id>`.
"""

from __future__ import annotations

import argparse
import ast
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd


# Default cluster label mapping (can be overridden via CLI)
DEFAULT_CLUSTER_MAPPING: Dict[int, str] = {
    0: "non_durable",
    1: "durable",
    2: "services",
}

# Columns we keep/rename in the final output
BASE_COLUMNS: List[str] = [
    "iteration",
    "node",
    "neighbor",
    "neighbor_cluster",
    "spending_similarity",
    "household_similarity",
    "redmeat_similarity",
    "cluster_similarity",
    "alpha",
    "beta",
    "component1",
    "component2",
    "final_weight",
]

NODE_ATTR_COLUMNS: List[str] = [
    "node_adults",
    "node_total_spending_on_food",
    "node_redmeat",
    "node_cult1",
    "node_income",
    "node_HC24",
    "node_cult",
    "node_consumption_rate",
    "node_dur_spend_r",
    "node_ndur_spend_r",
    "node_serv_spend_r",
]

NEIGHBOR_ATTR_COLUMNS: List[str] = [
    "neighbor_adults",
    "neighbor_total_spending_on_food",
    "neighbor_redmeat",
    "neighbor_cult1",
    "neighbor_income",
    "neighbor_HC24",
    "neighbor_cult",
    "neighbor_consumption_rate",
    "neighbor_dur_spend_r",
    "neighbor_ndur_spend_r",
    "neighbor_serv_spend_r",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarise StreamingLogger weight logs into a multi-sheet Excel report."
    )
    parser.add_argument(
        "--logs",
        required=True,
        help="Path to a weight_logs_*.csv file or a directory containing such files.",
    )
    parser.add_argument(
        "--output",
        default="weight_log_summary.xlsx",
        help="Destination Excel workbook (default: ./weight_log_summary.xlsx).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1996,
        help="Random seed for reproducible sampling (default: 1996).",
    )
    parser.add_argument(
        "--mapping",
        default=None,
        help=(
            "Optional mapping from cluster id to sheet label. "
            "Provide as JSON or Python literal, e.g. '{0:\"non_durable\"}'."
        ),
    )
    return parser.parse_args()


def load_weight_logs(path_arg: str) -> pd.DataFrame:
    path = Path(path_arg).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Provided path does not exist: {path}")

    candidate_files: Iterable[Path]
    if path.is_dir():
        candidate_files = sorted(path.glob("weight_logs_*.csv"))
    else:
        candidate_files = [path]

    files = [p for p in candidate_files if p.is_file()]
    if not files:
        raise FileNotFoundError(f"No weight log CSV files found under {path}")

    frames = []
    for csv_path in files:
        df = pd.read_csv(csv_path)
        df["source_file"] = csv_path.name
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    return combined


def normalise_mapping(mapping_arg: str | None) -> Dict[int, str]:
    if not mapping_arg:
        return DEFAULT_CLUSTER_MAPPING.copy()

    try:
        parsed = ast.literal_eval(mapping_arg)
    except (ValueError, SyntaxError) as exc:
        raise ValueError(f"Unable to parse mapping: {mapping_arg}") from exc

    if not isinstance(parsed, dict):
        raise TypeError("Cluster mapping must evaluate to a dictionary.")

    normalised: Dict[int, str] = {}
    for key, value in parsed.items():
        try:
            cluster_id = int(key)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"Cluster id keys must be integers, got {key!r}") from exc
        normalised[cluster_id] = str(value)

    return normalised


def build_sheet(
    df: pd.DataFrame,
    cluster_id: int,
    sheet_label: str,
    seed: int,
) -> pd.DataFrame:
    cluster_df = df[df["node_cluster"] == cluster_id].copy()
    if cluster_df.empty:
        return pd.DataFrame()

    unique_nodes = cluster_df["node"].drop_duplicates()
    sample_size = min(5, len(unique_nodes))
    sampled_nodes = unique_nodes.sample(n=sample_size, random_state=seed)

    sheet_df = cluster_df[cluster_df["node"].isin(sampled_nodes)].copy()
    if sheet_df.empty:
        return pd.DataFrame()

    # Keep only the columns we care about and rename for clarity
    keep_columns = BASE_COLUMNS + NODE_ATTR_COLUMNS + NEIGHBOR_ATTR_COLUMNS + ["node_cluster", "source_file"]
    missing = [col for col in keep_columns if col not in sheet_df.columns]
    if missing:
        raise KeyError(
            f"The following expected columns are missing from the weight logs: {missing}"
        )

    sheet_df = sheet_df[keep_columns]

    rename_map = {
        "node": "individual",
        "neighbor": "neighbor_individual",
        "neighbor_cluster": "neighbor_cluster_id",
        "node_cluster": "individual_cluster_id",
        "node_adults": "individual_adults",
        "neighbor_adults": "neighbor_adults",
        "node_total_spending_on_food": "individual_total_spending_on_food",
        "neighbor_total_spending_on_food": "neighbor_total_spending_on_food",
        "node_redmeat": "individual_redmeat",
        "neighbor_redmeat": "neighbor_redmeat",
        "node_cult1": "individual_cult1",
        "neighbor_cult1": "neighbor_cult1",
        "node_income": "individual_income",
        "neighbor_income": "neighbor_income",
        "node_HC24": "individual_HC24",
        "neighbor_HC24": "neighbor_HC24",
        "node_cult": "individual_cult",
        "neighbor_cult": "neighbor_cult",
        "node_consumption_rate": "individual_consumption_rate",
        "neighbor_consumption_rate": "neighbor_consumption_rate",
        "node_dur_spend_r": "individual_dur_spend_r",
        "neighbor_dur_spend_r": "neighbor_dur_spend_r",
        "node_ndur_spend_r": "individual_ndur_spend_r",
        "neighbor_ndur_spend_r": "neighbor_ndur_spend_r",
        "node_serv_spend_r": "individual_serv_spend_r",
        "neighbor_serv_spend_r": "neighbor_serv_spend_r",
        "spending_similarity": "similarity_spending",
        "household_similarity": "similarity_household_size",
        "redmeat_similarity": "similarity_redmeat",
        "cluster_similarity": "similarity_cluster",
        "component1": "weight_component_spending",
        "component2": "weight_component_redmeat",
    }

    sheet_df = sheet_df.rename(columns=rename_map)
    sheet_df.insert(0, "cluster_label", sheet_label)

    # Sort for readability: individuals grouped together, iteration ascending
    sheet_df = sheet_df.sort_values(
        by=["individual", "iteration", "neighbor_individual"], ascending=[True, True, True]
    )
    sheet_df.reset_index(drop=True, inplace=True)
    return sheet_df


def write_excel(sheets: Dict[str, pd.DataFrame], output_path: Path) -> None:
    if not sheets:
        raise ValueError("No data available to write to Excel.")

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        for sheet_name, data_frame in sheets.items():
            if data_frame.empty:
                continue
            # openpyxl limits sheet names to 31 chars
            writer_sheet_name = sheet_name[:31]
            data_frame.to_excel(writer, sheet_name=writer_sheet_name, index=False)


def main() -> None:
    args = parse_args()
    cluster_mapping = normalise_mapping(args.mapping)

    weight_logs = load_weight_logs(args.logs)

    available_clusters = sorted(weight_logs["node_cluster"].dropna().unique())
    sheets: Dict[str, pd.DataFrame] = {}
    for cluster_id in available_clusters:
        cluster_int = int(cluster_id)
        label = cluster_mapping.get(cluster_int, f"cluster_{cluster_int}")
        sheet_df = build_sheet(weight_logs, cluster_int, label, args.seed)
        if not sheet_df.empty:
            sheets[label] = sheet_df

    if not sheets:
        raise ValueError("No rows matched the requested clusters; nothing to write.")

    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    write_excel(sheets, output_path)
    print(f"Saved summary to {output_path}")


if __name__ == "__main__":
    main()

