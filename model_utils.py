import numpy as np
import pandas as pd
from Dicts_Lists_Helpers import *
from configuration import *



def calculate_stubbornness_param_personal(old_spending : float, new_inflc_spending : float, spending_inc_tolerance : float, alpha_0 : float) -> float:           
    diff = new_inflc_spending - old_spending
    if diff > 1e-12:
        con1 = (old_spending * spending_inc_tolerance) / diff
        if con1 < alpha_0:
            new_alpha_0 = con1
        else:
            new_alpha_0 = alpha_0
    else:
        new_alpha_0 = alpha_0
    return new_alpha_0


def normalize_protein_shares(df : pd.DataFrame, node_id : int, basic_protein_columns : list, flag_error = True) -> None:
    sum_pro_share = sum([df.loc[node_id][attrdict[key]['protein_share']] for key in basic_protein_columns])
    if sum_pro_share < 1e-10:
        if flag_error:
            print(f"[normalize] total protein share ≈0 for node {node_id}, skipping")
    elif np.abs(1.0 - sum_pro_share) > 1e-10:
        if flag_error:
            print('pro share does not sum upto 1 for node:', node_id)
        for key in basic_protein_columns:
            old_val =  df.loc[node_id][attrdict[key]['protein_share']]
            df.loc[node_id, attrdict[key]['protein_share']] = old_val / sum_pro_share

def primary_updates_to_secondary_updates(df : pd.DataFrame, node_id : int, pro_share_update_dict: Dict[str, float], conversion_factors, emission_factors) -> Dict[str, float]:
        temporary_dict = {}
        temporary_dict['total emission from food'] = 0.0
        temporary_dict['total spending on food']   = 0.0
        temporary_dict['redmeat']                  = 0.0
        temporary_dict["total_meat_protein_share"] = 0.0

        tot_prot = df.loc[node_id]['total protein content']
        adults   = df.loc[node_id]['adults']

        for com in basic_protein_cols:
            share_key  = attrdict[com]['protein_share']
            share_val  = pro_share_update_dict.get(share_key, 0.0)
            pro_val    = share_val * tot_prot
            quantity   = pro_val / conversion_factors[com]
            emission   = quantity * emission_factors[com]
            spending   = quantity * df.loc[node_id][attrdict[com]['price']]

            temporary_dict[share_key]                = share_val
            temporary_dict[attrdict[com]['protein_amount']]  = pro_val
            temporary_dict[attrdict[com]['quantity']]        = quantity
            temporary_dict[attrdict[com]['emission']]        = emission
            temporary_dict[attrdict[com]['spending']]        = spending
            temporary_dict['total emission from food']      += emission
            temporary_dict['total spending on food']        += spending

            # If com is a redmeat commodity
            if attrdict[com]['spending'] in redmeat_spending_cols:
                temporary_dict['redmeat'] += spending
            if attrdict[com]['protein_share'] in meat_protein_share:
                temporary_dict["total_meat_protein_share"] += share_val

        # Denormalize by # of adults
        temporary_dict['denormalized total emission from food'] = (
            temporary_dict['total emission from food'] * adults
        )

        return temporary_dict
    



