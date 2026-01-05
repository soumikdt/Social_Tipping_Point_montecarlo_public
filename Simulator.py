# simulator.py
import copy
import math
import random
import gc


import pandas as pd
import warnings
import numpy as np

warnings.filterwarnings(
    "ignore",
    category=RuntimeWarning,
    message="overflow encountered in exp"
)

np.seterr(over='ignore', invalid='ignore')

from SocialNetworkGenerator import *
from Dicts_Lists_Helpers import *
from configuration import *
from simulation_utils import *
from model_dynamics import *
from ImposeDiet import *
from NodeSelector import *



def run_simulation_with_diet_impo(df, opn_cols, alpha, beta, stubbornness_param, SIT, x : float, var : str, quantile_cluster : int, use_random_diet_impo : bool, seed :int, num_itr : int = 1,  linear_model_bool = False, confidence_col = None, confidence_param = None):
    get_nodes_from_quantile()

def run_simulation(df, opn_cols, alpha, beta, stubbornness_param, SIT, num_itr : int = 1, static_node_indices = [], linear_model_bool = False, confidence_col = None, confidence_param = None):
    x = len(static_node_indices)
    if x == 0:
        diet_impo = False
    else: 
        diet_impo = True
    print(f"Processing: alpha_0={stubbornness_param}, alpha={alpha}, beta={beta}, sit={SIT}, x={x}", flush=True)

    
    ###################################################################
    ###################################################################
    # 5) Build dynamic initial DataFrame from initial node attributes
    dynamic_initial_df, static_initial_df = build_node_dfs(df, static_node_indices)        

    add_redmeat_share(dynamic_initial_df)

    # dynamic initial
    dynamic_initial_stats = compute_stats(dynamic_initial_df, comparison_list)
    dynamic_initial_stats_grouped = compute_grouped_stats(dynamic_initial_df, comparison_list)


    # Also recompute 'redmeat protein share' for static nodes
    add_redmeat_share(static_initial_df)

    # static initial
    static_initial_stats = compute_stats(static_initial_df, comparison_list)
    static_initial_stats_grouped = compute_grouped_stats(static_initial_df, comparison_list)

    

    ###################################################################
    ###################################################################
    ###################################################################

    if diet_impo:
        impose_lanclet_diet(df, static_node_indices)
        print(f"diet imposition on {x} static agents.")

        ###################################################################

        # 5) Build dynamic initial_after_diet_imposition_ DataFrame from initial_after_diet_imposition_ node attributes
        dynamic_initial_after_diet_imposition__df, static_initial_after_diet_imposition__df = build_node_dfs(df, static_node_indices)        


        add_redmeat_share(dynamic_initial_after_diet_imposition__df)

        dynamic_initial_after_diet_imposition__stats = compute_stats(
        dynamic_initial_after_diet_imposition__df, comparison_list)

        dynamic_initial_after_diet_imposition__stats_grouped = compute_grouped_stats(
        dynamic_initial_after_diet_imposition__df, comparison_list)



        # Also recompute 'redmeat protein share' for static nodes
        add_redmeat_share(static_initial_after_diet_imposition__df)

        static_initial_after_diet_imposition__stats = compute_stats(
        static_initial_after_diet_imposition__df, comparison_list)

        static_initial_after_diet_imposition__stats_grouped = compute_grouped_stats(
        static_initial_after_diet_imposition__df, comparison_list)



    ###################################################################
    ###################################################################
    ###################################################################

    # 4) Run 50 updates
    for iteration_idx in range(num_itr):
        # Now we call weighted_update on the AllAgents instance
        update_full(df, opn_cols, alpha, beta, stubbornness_param, SIT, iteration_idx, linear_model_bool, confidence_col, confidence_param)

    ###################################################################
    ###################################################################
    ###################################################################

    print(f"after 50 update {x} static agents")

    # 5) Build dynamic final DataFrame from updated node attributes
    dynamic_final_df, static_final_df =  build_node_dfs(df, static_node_indices)

    add_redmeat_share(dynamic_final_df)

    # dynamic final
    dynamic_final_stats = compute_stats(dynamic_final_df, comparison_list)
    dynamic_final_stats_grouped = compute_grouped_stats(dynamic_final_df, comparison_list)


    # Also recompute 'redmeat protein share' for static nodes
    add_redmeat_share(static_final_df)

    # static final
    static_final_stats = compute_stats(static_final_df, comparison_list)
    static_final_stats_grouped = compute_grouped_stats(static_final_df, comparison_list)

    ###################################################################
    ###################################################################

    del dynamic_initial_df, static_initial_df
    del dynamic_final_df, static_final_df

    if diet_impo:
        del dynamic_initial_after_diet_imposition__df, static_initial_after_diet_imposition__df

    gc.collect()

    ###################################################################
    ###################################################################
    # Combine results in one row
    result_row = {
        'alpha_0': stubbornness_param,
        'alpha': alpha,
        'beta': beta,
        'sit': SIT,
        'x': x,
    }

    # --------- non-grouped stats ---------
    # Add dynamic initial stats
    for k, v in dynamic_initial_stats.items():
        result_row['dynamic_initial_' + k] = v

    # Add static initial stats
    for k, v in static_initial_stats.items():
        result_row['static_initial_' + k] = v

    # Add dynamic final stats
    for k, v in dynamic_final_stats.items():
        result_row['dynamic_final_' + k] = v

    # Add static final stats
    for k, v in static_final_stats.items():
        result_row['static_final_' + k] = v



    # dynamic initial grouped stats
    add_grouped_stats(result_row, dynamic_initial_stats_grouped, prefix="dynamic_initial")

    # static initial grouped stats
    add_grouped_stats(result_row, static_initial_stats_grouped, prefix="static_initial")

    # dynamic final grouped stats
    add_grouped_stats(result_row, dynamic_final_stats_grouped, prefix="dynamic_final")

    # static final grouped stats
    add_grouped_stats(result_row, static_final_stats_grouped, prefix="static_final")


    # --------- after-diet stats (if applicable) ---------

    if diet_impo:
        # non-grouped after-diet stats
        for k, v in dynamic_initial_after_diet_imposition__stats.items():
            result_row['afterdiet_dynamic_' + k] = v
        for k, v in static_initial_after_diet_imposition__stats.items():
            result_row['afterdiet_static_' + k] = v

        # grouped after-diet stats
        add_grouped_stats(result_row,
                        dynamic_initial_after_diet_imposition__stats_grouped,
                        prefix="afterdiet_dynamic_initial")
        add_grouped_stats(result_row,
                        static_initial_after_diet_imposition__stats_grouped,
                        prefix="afterdiet_static_initial")

    return result_row, df