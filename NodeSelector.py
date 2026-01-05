import math
import random
from typing import Any, List, Optional
import pandas as pd
import numpy as np

def get_nodes_from_quantile(
    df : pd.DataFrame,
    variable: str,
    quantile: int,
    x: float,
    use_random: bool,
    seed: Optional[int]
) -> List[Any]:
    
    if not isinstance(quantile, int):
        raise TypeError("quantile must be an integer (e.g., 1, 2, 3).")
    if x <= 0:
        return []
    if x > 1:
        raise ValueError("x must be in the interval (0, 1].")

    indices_list = df.index[df[variable] == quantile].to_list()

    if not indices_list:
        return []

    k = int(np.floor(len(indices_list) * x))
    if k == 0:
        return []

    if use_random:
        rng = np.random.default_rng(seed)
        return rng.choice(indices_list, size=k, replace=False).tolist()
    else:
        degrees = df.loc[indices_list, "neighbors"].apply(len)
        sorted_nodes = degrees.sort_values(ascending=False).index
        return sorted_nodes[:k].to_list()
