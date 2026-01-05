import networkx as nx
import pandas as pd
import numpy as np

def compute_stats(df: pd.DataFrame, attrs):
    stats = {}
    for attr in attrs:
        if attr in df.columns:
            safe = attr.replace(" ", "_")
            stats[f"{safe}_mean"]  = df[attr].mean()
            stats[f"{safe}_var"]   = df[attr].var()
            stats[f"{safe}_total"] = df[attr].sum()
            stats[f"{safe}_count"] = df[attr].count()
    return stats


def compute_grouped_stats(df: pd.DataFrame, attrs, group_col: str = 'meat_tertile'):
    grouped_stats = {}
    # Guard against empty frames or missing group column
    if df.empty or group_col not in df.columns:
        return grouped_stats
    for group_name, df_tmp in df.groupby(group_col):
        grouped_stats[group_name] = compute_stats(df_tmp, attrs)
    return grouped_stats


def add_grouped_stats(result_row, grouped_stats_dict, prefix: str):
    """
    grouped_stats_dict: {group_name: {stat_key: value, ...}, ...}
    prefix: e.g. 'dynamic_initial', 'static_final', ...
    """
    for group_name, stats_dict in grouped_stats_dict.items():
        # Make group name safe for column labels
        group_str = str(group_name).replace(" ", "_")
        for k, v in stats_dict.items():
            # Example column: dynamic_initial_group_0_redmeat_protein_share_mean
            col_name = f"{prefix}_group_{group_str}_{k}"
            result_row[col_name] = v


def add_redmeat_share(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds 'redmeat protein share' column to df based on
    Beef, Pork and Lamb+Goat protein share columns.
    Returns the same DataFrame (for chaining).
    """
    df['redmeat protein share'] = (
        df.get('Beef and veal aggregated protein share', 0.0) +
        df.get('Pork aggregated protein share', 0.0) +
        df.get('Lamb and goat aggregated protein share', 0.0)
    )
    return df



def build_node_dfs(df : pd.DataFrame, static_node_indices : list): # -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Builds two DataFrames:
    - dynamic_df: nodes NOT in static_nodes
    - static_df:  nodes IN static_nodes
    
    Returns:
        dynamic_df, static_df
    """
    df_tmp = df.copy()

    static_df = df_tmp.loc[static_node_indices] 
    dynamic_df = df_tmp.loc[~df_tmp.index.isin(static_node_indices)]

    return dynamic_df, static_df