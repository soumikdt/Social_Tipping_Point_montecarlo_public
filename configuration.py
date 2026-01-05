import networkx as nx
from typing import Any, Dict, List, Optional, Callable
import numpy as np
import pandas as pd

class CalculateWeight: 
    @staticmethod 
    def calculate_weight(df: pd.DataFrame, node : int, nbr : int, alpha : float, beta : float) -> float:
        spending_sim = CalculateWeight.custom_g(df, node , nbr) 
        hh_sim = CalculateWeight.cosine_similarity_on_adults(df, node , nbr)
        red_sim = CalculateWeight.my_redmeat_sim(df, node , nbr)
        cluster_quantile_sim = CalculateWeight.quantile_similarity(df, node , nbr) 

        weight_tmp = (alpha * spending_sim * hh_sim + beta * red_sim) * cluster_quantile_sim
        #print("temp weight", weight_tmp)
        return weight_tmp
    
    @staticmethod 
    def custom_g(df : pd.DataFrame, i : int, j : int, scaling_param: float = 30.0) -> float:
        x1 = df.loc[i]['total spending on food']
        x2 = df.loc[j]['total spending on food']
        re1 = 1 - (1 + np.exp(scaling_param * (x2 - x1) / x1)) ** (-1 / scaling_param)
        #print("ustom_g called from run_cluster_factor_experiment.py")
        #print("spending sim", re1)
        return re1

    @staticmethod 
    def cosine_similarity_on_adults(df : pd.DataFrame, i: int, j: int) -> float:
        features = ['adults']
        diff = sum((df.loc[i][feat] - df.loc[j][feat])**2 for feat in features)
        #print("cosine_similarity_on_adults called from run_cluster_factor_experiment.py")
        out = 1.0 / (0.1 + diff)
        #print("HH sim", out)
        return out

    @staticmethod 
    def my_redmeat_sim(df : pd.DataFrame, i: int, j: int) -> float:
        r1 = df.loc[i]['redmeat']
        r2 = df.loc[j]['redmeat']
        #if not isinstance(r1, float):
        #    print("cluster quantile number must be float")
        #if not isinstance(r2, float):
        #    print("cluster quantile number must be float")
        #print(r1, r2)
        #print("my_redmeat_sim called from run_cluster_factor_experiment.py")
        out = 1.0 / (1 + (r1 - r2)**2)
        #print("redmeat sim", out)
        return out

    @staticmethod
    def quantile_similarity(
        df : pd.DataFrame, i: int, j: int,
        same_cluster_factor: float = 1.0,
        diff_cluster_factor: float = 0.0
    ) -> float:
        """
        Quantile-based similarity using the precomputed node attribute 'meat_quantile'.
        Returns:
            same_cluster_factor  if meat_quantile(i) == meat_quantile(j)
            diff_cluster_factor  otherwise
            1.0                  if either attribute is missing
        """
        cluster_quantile_col = 'meat_tertile'
        qi = df.loc[i][cluster_quantile_col]
        qj = df.loc[j][cluster_quantile_col]
        #print(qi, qj)

        #if not isinstance(qi, int):
        #    print("cluster quantile number must be int")
        #if not isinstance(qj, int):
        #    print("cluster quantile number must be int")


        # if either node lacks the attribute → neutral similarity

        out = same_cluster_factor if qi == qj else diff_cluster_factor

        #print("cluster sim", out)

        return out

def default_degree_function(val: float, val2: float) -> float:
    """
    Default degree function for SocialNetworkAnalyzer when constructing graphs
    from the preprocessed consumption CSV.
    Mirrors the heuristic used in earlier Monte Carlo scripts.
    """
    #print(" default_degree_function called from run_cluster_factor_experiment.py")
    out = 3 * (2 * np.log(0.1 * val + 10.0) + np.log(0.001 * val2 + 1.0) - 5.0) + 5.0
    #print("degree", out)
    return out

# ==========================================================================
# CONFIGURATION - Edit these parameters for your experiment
# ==========================================================================
SAMPLE_FRACTION = 0.01 

# Cluster factors to test (different cluster vs same cluster weight multiplier)
# Using same values as local run for consistency
CLUSTER_FACTORS = (0.0,1.0)

# Number of Monte Carlo runs (increase for better statistics)
N_RUNS = 1

# Simulation parameters (usually keep these fixed)
ALPHA_0 = (0.5,)  # Base adoption parameter
ALPHA = (0.3,)    # Spending similarity weight
BETA = (5.0,)     # Red meat similarity weight
SIT = (0.005,)    # Spending increase tolerance

# Static fraction values (also used as Lancet adoption percentages)
# None = use power_partition(0, 1, 7, p=3) which gives ~9-10 values
# Custom values: (0.0, 0.002, 0.02, 0.06, 0.15, 0.29, 0.5, 0.8)
X_LIST = (0.0, 0.002, 0.02, 0.06, 0.15, 0.29, 0.5, 0.8)

# Node selection parameters
USE_RANDOM = (True,)   # Use random node selection
Q_LIST = (1,2,3)          # Cluster number

# Parallelism
N_JOBS = 1             # Number of parallel workers for joblib

# CSV-driven network construction
CSV_PATH_LOCAL = "/Users/mimuw2022/Documents/GitHub/Social_Network/Data and output/preprocessedconsumptions411_ICM.csv"
CSV_PATH_CLUSTER = "/lu/topola/home/soumik96/montecarlosim/newtry/Data and output/preprocessedconsumptions411_ICM.csv"
CSV_PATH = None