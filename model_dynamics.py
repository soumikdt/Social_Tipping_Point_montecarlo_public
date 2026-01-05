import pandas as pd
import numpy as np
import networkx as nx
from typing import Tuple



from SocialNetworkGenerator import SocialNetworkAnalyzer
from utils import *
from configuration import * 
from model_utils import *


def linear_model(
    opn_order : list[str],
    node_opn: np.array,
    nbd_opns_initial: np.ndarray,
    nbd_weights_initial: np.ndarray,
    stubbornness_param: float
) -> Tuple[np.ndarray, np.ndarray, list[str]]:
    """PURE FUNCTION
    method for implement core idea of DeGroot or Friedkin-Johnson models.
    NOTETHAT: nbd_opns/consumption = [nbd1_opn, nbd2_opn, ...]
    OUTPUT: node_new_opn, pure_influence_opn, opn_order
    """

    if len(nbd_opns_initial) != len(nbd_weights_initial):
        raise ValueError(
            "Input lists 'nbd_opn' and 'nbd_weights' must have the same length."
        )

    nbd_opns = nbd_opns_initial.transpose()
    sum_nbd_weights = nbd_weights_initial.sum()

    if np.abs(sum_nbd_weights) <  1e-8:
        pass
    
    nbd_weights = nbd_weights_initial
    if np.abs(sum_nbd_weights - 1) >=  1e-8:
        # normalize weights
        nbd_weights = nbd_weights_initial / sum_nbd_weights

    pure_influence_opn = (nbd_weights * nbd_opns).sum(axis=1)

    node_new_opn = (
        stubbornness_param * node_opn
        + (1 - stubbornness_param) * pure_influence_opn
    )

    return node_new_opn, pure_influence_opn, opn_order

def bounded_confidence_model(
    opn_order : list[str],
    node_opn: np.array,
    nbd_opns: np.ndarray,
    nbd_weights: np.array,
    stubbornness_param: float,
    confidence_param: float,
    node_confidence_col_val : float,
    nbd_confidence_col_val: np.array
) -> Tuple[np.ndarray, np.ndarray, list[str]]:
    """PURE FUNCTION
    implements bounded confidence models.
    using single food item for confidence.
    """

    # find confidence group
    # Create a boolean mask based on the condition
    condition_mask = (
        np.abs(node_confidence_col_val - nbd_confidence_col_val)
        <= confidence_param
    )

    # Apply the mask to both arrays to get the filtered values


    assert nbd_opns.ndim == 2
    assert nbd_opns.shape[0] == len(nbd_weights)
    assert condition_mask.shape[0] == nbd_opns.shape[0]


    confidence_nbd_opns = nbd_opns[condition_mask, :]
    confidence_nbd_weights = nbd_weights[condition_mask]

    if confidence_nbd_opns.shape[0] == 0:
        # No neighbors within confidence → no social influence
        return node_opn, node_opn.copy(), opn_order

    return linear_model(
        opn_order,
        node_opn,
        confidence_nbd_opns,
        confidence_nbd_weights,
        stubbornness_param
    )

def update_single_node(
    df: pd.DataFrame,
    node_id,
    opn_cols: list[str],
    alpha : float,
    beta : float,
    stubbornness_param : float, 
    SIT : float,
    linear_model_bool : bool = True,
    confidence_col = None,
    confidence_param = None
) -> None:
    nbd_ids_list = df.loc[node_id]['neighbors']
    node_opn_array = df_slice_to_np(df, [node_id], opn_cols)[0]
    nbd_opns_ndarray = df_slice_to_np(df, nbd_ids_list, opn_cols)

    CW = CalculateWeight()
    nbd_weights = np.array([
    CW.calculate_weight(df, node_id, nbr, alpha, beta)
    for nbr in nbd_ids_list
    ])

    if np.sum(nbd_weights) < 1e-8:
        return 
    
    if linear_model_bool: 
        model_output = linear_model(
            opn_cols,
            node_opn_array,
            nbd_opns_ndarray,
            nbd_weights,
            stubbornness_param
        )

    else: 
        if confidence_col is None or confidence_param is None:
            raise ValueError("confidence_col and confidence_param must be provided for bounded confidence model")
        
        node_confidence_col_val = df.loc[node_id][confidence_col]

        nbd_confidence_col_val = df.loc[nbd_ids_list][confidence_col].to_numpy()

        model_output = bounded_confidence_model(
            opn_cols,
            node_opn_array,
            nbd_opns_ndarray,
            nbd_weights,
            stubbornness_param,
            confidence_param,
            node_confidence_col_val,
            nbd_confidence_col_val
        )

    pure_inflnc_dict = {x : y for x,y in zip(model_output[2], model_output[1])}

    node_dict = {x : y for x,y in zip(model_output[2], node_opn_array)}


    # calculate stubbornness_param_personal using

    proposed_secondary_update_pure_inflnc_expense = primary_updates_to_secondary_updates(df, node_id, pure_inflnc_dict, conversion_factors, emission_factors)['total spending on food']

    proposed_secondary_update_pure_node_expense = primary_updates_to_secondary_updates(df, node_id, node_dict, conversion_factors, emission_factors)['total spending on food']

    stubbornness_param_personal = calculate_stubbornness_param_personal(proposed_secondary_update_pure_node_expense, proposed_secondary_update_pure_inflnc_expense, SIT, stubbornness_param)

    node_new_opn = (
        stubbornness_param_personal * node_opn_array
        + (1 - stubbornness_param_personal) * model_output[1]
    )

    node_new_opn_dict = {x : y for x,y in zip(model_output[2], node_new_opn)}

    final_secondary_update_dict = primary_updates_to_secondary_updates(df, node_id, node_new_opn_dict, conversion_factors, emission_factors)

    for col, val in node_new_opn_dict.items():
        df.loc[node_id, col] = val

    for col, val in final_secondary_update_dict.items():
        df.loc[node_id, col] = val

def update_full(df : pd.DataFrame, opn_cols : list[str], alpha : float, beta : float , stubbornness_param : float, SIT : float, run_index : int, static_nodes : list = [], linear_model_bool : bool = True, confidence_col : str|None = None, confidence_param : float|None = None):
    rng = np.random.default_rng(seed=42 + run_index)
    node_list = df.index.to_list()
    rng.shuffle(node_list)
    for node in node_list:
        if node in static_nodes:
            continue
        update_single_node(df, node, opn_cols, alpha, beta , stubbornness_param, SIT, linear_model_bool, confidence_col, confidence_param)