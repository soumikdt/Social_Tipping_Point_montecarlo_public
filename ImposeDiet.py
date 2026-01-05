#ImposeDiet.py

import networkx as nx
import numpy as np
import pandas as pd


from typing import Any, Callable, Dict, List, Optional


from SocialNetworkGenerator import SocialNetworkAnalyzer
from utils import *
from configuration import *
from model_utils import *
from model_dynamics import *
from Simulator import *
from Dicts_Lists_Helpers import *

def redistribute_pro_share(df : pd.DataFrame, node_id : int, com_list: List[str], target_share: float) -> None:
    if len(com_list) == 1:
        df.loc[node_id, attrdict[com_list[0]]['protein_share']] = target_share
        return

    attr_list = [attrdict[c]['protein_share'] for c in com_list]

    # extract current shares
    shares = df_slice_to_np(df, [node_id], attr_list)[0]

    sum_orig = shares.sum()

    if sum_orig > 1e-7: #TODO make verbose
        shares *= target_share / sum_orig
    else:
        shares.fill(target_share / len(com_list))

    update_dict = {x : y for x,y in zip(attr_list, shares) }

    # write back
    for col, val in update_dict.items():
        df.loc[node_id, col] = val


def impose_lanclet_diet(df : pd.DataFrame, static_nodes : list) -> None:
    """
    Example: sets target protein shares for a "Lanclet diet".
    """
    
    for node in static_nodes:
        redistribute_pro_share(df, node, ['egg'], 0.01381509033)
        redistribute_pro_share(df, node, ['bread', 'rice'], 0.2465462274)
        redistribute_pro_share(df, node, ['vegetables without dried'], 0.3188097768)
        redistribute_pro_share(df, node, ['Dried vegetables'], 0.07970244421)
        redistribute_pro_share(df, node, ['fish and seafood'], 0.02975557917)
        redistribute_pro_share(df , node, ['milk without cheese', 'Cheese'], 0.265674814)
        redistribute_pro_share(df, node, 
            ['Beef and veal aggregated','Pork aggregated','Lamb and goat aggregated'],
            0.01487778959)
        redistribute_pro_share(df , node, ['Poultry aggregated'], 0.03081827843)

        update_dict = df.loc[node, pro_share_list].to_dict()

        primary_updates_to_secondary_updates(df, node, update_dict, conversion_factors, emission_factors)




