import copy
import math
import random
import gc
import numpy as np
import pandas as pd
import networkx as nx
from joblib import Parallel, delayed
from AgentManipulator import *
from NodeSelector import *
from SocialNetworkGenerator import *
from Dicts_Lists_Helpers import *


# simulator.py


from AgentManipulator import AllAgents, ImposeDiet, basic_protein_cols, pro_share_list
from NodeSelector import NodeSelector
from SocialNetworkGenerator import SocialNetworkAnalyzer



class SimulationConfif:
    def __init__(
        self,
        alpha_0,
        alpha,
        beta,
        sit,
        g_func,
        hh_similarity_func,
        redmeat_similarity_func,
        diet_imp,
        prnt,
        cluster_similarity_func=None,
        logger=None,
    ):
        self.alpha_0 = alpha_0
        self.alpha = alpha
        self.beta = beta
        self.sit = sit

        self.g_func = g_func
        self.hh_similarity_func = hh_similarity_func
        self.redmeat_similarity_func = redmeat_similarity_func
        self.cluster_similarity_func = cluster_similarity_func
        self._diet_impo = diet_imp
        self.logger = logger

        self.prnt = prnt

class RunSimulationCollectData:

    def run_simulation(self, objSimulationConfig : SimulationConfif, items, objAllAgents: AllAgents):
        alpha_0 = objSimulationConfig.alpha_0
        alpha = objSimulationConfig.alpha
        beta = objSimulationConfig.beta
        sit = objSimulationConfig.sit
        x = len(objAllAgents._AM_static_nodes)
        static_nodes = objAllAgents._AM_static_nodes
        diet_impo = objSimulationConfig._diet_impo

        # 1) Get the original graph from self.AllAgents
        H = objAllAgents.network

        print(f"Processing: alpha_0={alpha_0}, alpha={alpha}, beta={beta}, sit={sit}, x={x}", flush=True)

        
        ###################################################################
        # 5) Build dynamic initial DataFrame from initial node attributes
        dynamic_initial_df = pd.DataFrame.from_dict(
            {node: data for node, data in H.nodes(data=True) if node not in static_nodes},
            orient='index'
        )
        dynamic_initial_df.index.name = 'node'
        dynamic_initial_df['redmeat protein share'] = (
            dynamic_initial_df.get('Beef and veal aggregated protein share', 0.0) +
            dynamic_initial_df.get('Pork aggregated protein share', 0.0) +
            dynamic_initial_df.get('Lamb and goat aggregated protein share', 0.0)
        )

        # Possibly compute initial stats & changes
        dynamic_initial_stats = {}
        for attr in comparison_list:
            if attr in dynamic_initial_df.columns:
                key_mean = attr.replace(" ", "_") + "_mean"
                key_var  = attr.replace(" ", "_") + "_var"
                key_total  = attr.replace(" ", "_") + "_total"
                dynamic_initial_stats[key_mean] = dynamic_initial_df[attr].mean()
                dynamic_initial_stats[key_var]  = dynamic_initial_df[attr].var()
                dynamic_initial_stats[key_total]  = dynamic_initial_df[attr].sum()
         
        # 6) Build initial static DataFrame (already done)
        static_initial_df = pd.DataFrame.from_dict(
            {node: data for node, data in H.nodes(data=True) if node in static_nodes},
            orient='index'
        )
        static_initial_df.index.name = 'node'

        # Also recompute 'redmeat protein share' for static nodes
        static_initial_df['redmeat protein share'] = (
            static_initial_df.get('Beef and veal aggregated protein share', 0.0) +
            static_initial_df.get('Pork aggregated protein share', 0.0) +
            static_initial_df.get('Lamb and goat aggregated protein share', 0.0)
        )

        # Compute statistics for static nodes separately
        static_initial_stats = {}
        for attr in comparison_list:
            if attr in static_initial_df.columns:
                key_mean = attr.replace(" ", "_") + "_mean"
                key_var  = attr.replace(" ", "_") + "_var"
                key_total = attr.replace(" ", "_") + "_total"
                static_initial_stats[key_mean] = static_initial_df[attr].mean()
                static_initial_stats[key_var]  = static_initial_df[attr].var()
                static_initial_stats[key_total] = static_initial_df[attr].sum()
        if diet_impo:
            ###################################################################
            impose_diet = ImposeDiet(conversion_factors,
                                    emission_factors,
                                    objAllAgents,
                                    basic_protein_cols)
            impose_diet.impose_lanclet_diet()

            ###################################################################

            # 5) Build dynamic initial_after_diet_imposition_ DataFrame from initial_after_diet_imposition_ node attributes
            dynamic_initial_after_diet_imposition__df = pd.DataFrame.from_dict(
                {node: data for node, data in H.nodes(data=True) if node not in static_nodes},
                orient='index'
            )
            dynamic_initial_after_diet_imposition__df.index.name = 'node'
            dynamic_initial_after_diet_imposition__df['redmeat protein share'] = (
                dynamic_initial_after_diet_imposition__df.get('Beef and veal aggregated protein share', 0.0) +
                dynamic_initial_after_diet_imposition__df.get('Pork aggregated protein share', 0.0) +
                dynamic_initial_after_diet_imposition__df.get('Lamb and goat aggregated protein share', 0.0)
            )

            # Possibly compute initial_after_diet_imposition_ stats & changes
            dynamic_initial_after_diet_imposition__stats = {}
            for attr in comparison_list:
                if attr in dynamic_initial_after_diet_imposition__df.columns:
                    key_mean = attr.replace(" ", "_") + "_mean"
                    key_var  = attr.replace(" ", "_") + "_var"
                    key_total  = attr.replace(" ", "_") + "_total"
                    dynamic_initial_after_diet_imposition__stats[key_mean] = dynamic_initial_after_diet_imposition__df[attr].mean()
                    dynamic_initial_after_diet_imposition__stats[key_var]  = dynamic_initial_after_diet_imposition__df[attr].var()
                    dynamic_initial_after_diet_imposition__stats[key_total]  = dynamic_initial_after_diet_imposition__df[attr].sum()
            
            # 6) Build initial_after_diet_imposition_ static DataFrame (already done)
            static_initial_after_diet_imposition__df = pd.DataFrame.from_dict(
                {node: data for node, data in H.nodes(data=True) if node in static_nodes},
                orient='index'
            )
            static_initial_after_diet_imposition__df.index.name = 'node'

            # Also recompute 'redmeat protein share' for static nodes
            static_initial_after_diet_imposition__df['redmeat protein share'] = (
                static_initial_after_diet_imposition__df.get('Beef and veal aggregated protein share', 0.0) +
                static_initial_after_diet_imposition__df.get('Pork aggregated protein share', 0.0) +
                static_initial_after_diet_imposition__df.get('Lamb and goat aggregated protein share', 0.0)
            )

            # Compute statistics for static nodes separately
            static_initial_after_diet_imposition__stats = {}
            for attr in comparison_list:
                if attr in static_initial_after_diet_imposition__df.columns:
                    key_mean = attr.replace(" ", "_") + "_mean"
                    key_var  = attr.replace(" ", "_") + "_var"
                    key_total = attr.replace(" ", "_") + "_total"
                    static_initial_after_diet_imposition__stats[key_mean] = static_initial_after_diet_imposition__df[attr].mean()
                    static_initial_after_diet_imposition__stats[key_var]  = static_initial_after_diet_imposition__df[attr].var()
                    static_initial_after_diet_imposition__stats[key_total] = static_initial_after_diet_imposition__df[attr].sum()



        ###################################################################

        # 4) Run 50 updates
        for iteration_idx in range(50):
            # Now we call weighted_update on the AllAgents instance
            objAllAgents.weighted_update(
                alpha=alpha,
                beta=beta,
                items=items,
                alpha_0=alpha_0,
                spending_inc_tolerance=sit,
                g_func=objSimulationConfig.g_func,
                hh_similarity_func=objSimulationConfig.hh_similarity_func,
                redmeat_similarity_func=objSimulationConfig.redmeat_similarity_func,
                cluster_similarity_func=objSimulationConfig.cluster_similarity_func,
                logger=objSimulationConfig.logger,
                iteration_num=iteration_idx,
                prnt = objSimulationConfig.prnt
            )

        ###################################################################
        # 5) Build dynamic final DataFrame from updated node attributes
        dynamic_final_df = pd.DataFrame.from_dict(
            {node: data for node, data in H.nodes(data=True) if node not in static_nodes},
            orient='index'
        )
        dynamic_final_df.index.name = 'node'
        dynamic_final_df['redmeat protein share'] = (
            dynamic_final_df.get('Beef and veal aggregated protein share', 0.0) +
            dynamic_final_df.get('Pork aggregated protein share', 0.0) +
            dynamic_final_df.get('Lamb and goat aggregated protein share', 0.0)
        )

        # Possibly compute final stats & changes
        dynamic_final_stats = {}
        for attr in comparison_list:
            if attr in dynamic_final_df.columns:
                key_mean = attr.replace(" ", "_") + "_mean"
                key_var  = attr.replace(" ", "_") + "_var"
                key_total  = attr.replace(" ", "_") + "_total"
                dynamic_final_stats[key_mean] = dynamic_final_df[attr].mean()
                dynamic_final_stats[key_var]  = dynamic_final_df[attr].var()
                dynamic_final_stats[key_total]  = dynamic_final_df[attr].sum()
                """
                change_attr = attr + "_change"
                if attr in initial_df.columns:
                    dynamic_final_df[change_attr] = np.where(
                        initial_df[attr] != 0,
                        (dynamic_final_df[attr] - initial_df[attr]) / initial_df[attr],
                        np.nan
                    )
                    key_change_mean = attr.replace(" ", "_") + "_change_mean"
                    key_change_var  = attr.replace(" ", "_") + "_change_var"
                    dynamic_final_stats[key_change_mean] = dynamic_final_df[change_attr].mean()
                    dynamic_final_stats[key_change_var]  = dynamic_final_df[change_attr].var()
                """
        # 6) Build final static DataFrame (already done)
        static_final_df = pd.DataFrame.from_dict(
            {node: data for node, data in H.nodes(data=True) if node in static_nodes},
            orient='index'
        )
        static_final_df.index.name = 'node'

        # Also recompute 'redmeat protein share' for static nodes
        static_final_df['redmeat protein share'] = (
            static_final_df.get('Beef and veal aggregated protein share', 0.0) +
            static_final_df.get('Pork aggregated protein share', 0.0) +
            static_final_df.get('Lamb and goat aggregated protein share', 0.0)
        )

        # Compute statistics for static nodes separately
        static_final_stats = {}
        for attr in comparison_list:
            if attr in static_final_df.columns:
                key_mean = attr.replace(" ", "_") + "_mean"
                key_var  = attr.replace(" ", "_") + "_var"
                key_total = attr.replace(" ", "_") + "_total"
                static_final_stats[key_mean] = static_final_df[attr].mean()
                static_final_stats[key_var]  = static_final_df[attr].var()
                static_final_stats[key_total] = static_final_df[attr].sum()

        del dynamic_initial_df, static_initial_df
        del dynamic_final_df, static_final_df
        if diet_impo:
            del dynamic_initial_after_diet_imposition__df, static_initial_after_diet_imposition__df
        gc.collect()

        ###################################################################
        # Combine results in one row
        result_row = {
            'alpha_0': alpha_0,
            'alpha': alpha,
            'beta': beta,
            'sit': sit,
            'x': x,
        }
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
        if diet_impo:
            for k,v in dynamic_initial_after_diet_imposition__stats.items():
                result_row['afterdiet_dynamic_' + k] = v
            for k,v in static_initial_after_diet_imposition__stats.items():
                result_row['afterdiet_static_'  + k] = v

        return result_row

