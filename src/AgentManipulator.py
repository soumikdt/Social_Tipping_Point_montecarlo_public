
import random
import networkx as nx
import numpy as np
import pandas as pd
from Dicts_Lists_Helpers import *
import inspect



attrdict = {}
for idx, com in enumerate(basic_protein_cols):
    sub_structure = {}
    sub_structure['protein_share'] = pro_share_list[idx] 
    sub_structure['protein_amount'] = pro_list[idx] 
    sub_structure['quantity'] = basic_protein_cols[idx] 
    sub_structure['spending'] = total_spending_cols[idx] 
    sub_structure['emission'] = emission_list[idx] 
    sub_structure['price'] = prices_col[idx] 
    attrdict[com] = sub_structure



# AgentManipulator.py

from typing import Any, Callable, Dict, List, Optional
import networkx as nx
import numpy as np
import pandas as pd


###############################################################################
# AllAgents class 
###############################################################################
class AllAgents:
    def __init__(
        self,
        _conversion_factors: Dict[str, float],
        _emission_factors: Dict[str, float],
        G: nx.Graph,
        basic_protein_columns: List[str], 
        stat_node : List = [],
        seed = 1996
    ) -> None:
        self._conversion_factors = conversion_factors
        self._emission_factors   = emission_factors
        self.network             = G
        self._basic_protein_columns = basic_protein_columns
        self._AM_static_nodes = stat_node

        
    def set_static_nodes(self, static_nodes: List[Any]) -> None:
        """Set the static nodes for the AllAgents instance."""
        self._AM_static_nodes = static_nodes
 
    def normalize_protein_shares(self, agentlist: Optional[List[Any]] = None, flag_error = False) -> None:
        if agentlist is None:
            agentlist = list(self.network.nodes)
        for node in agentlist:
            sum_pro_share = 0.0
            for key in self._basic_protein_columns:
                sum_pro_share += self.network.nodes[node].get(attrdict[key]['protein_share'], 0.0)
            if sum_pro_share < 1e-10:
                if flag_error:
                    print(f"[normalize] total protein share ≈0 for node {node}, skipping")
                continue
            if np.abs(1.0 - sum_pro_share) > 1e-10:
                if flag_error:
                    print('pro share does not sum upto 1 for node:', node)
                    break
                for key in self._basic_protein_columns:
                    old_val = self.network.nodes[node].get(attrdict[key]['protein_share'], 0.0)
                    if sum_pro_share != 0.0:
                        self.network.nodes[node][attrdict[key]['protein_share']] = old_val / sum_pro_share

    def recalc_node_attr_from_pro_share(self, agentlist: Optional[List[Any]] = None) -> None:
        """
        Recompute the node's food-related attributes from the protein shares,
        e.g. total spending, total emissions, etc.
        """
        if agentlist is None:
            agentlist = list(self.network.nodes)
        for node in agentlist:
            self.normalize_protein_shares([node])
            pro_sh_dict = {}
            # Collect the updated protein shares from the node
            for com in basic_protein_cols:
                share_key = attrdict[com]['protein_share']
                pro_sh_dict[share_key] = self.network.nodes[node].get(share_key, 0.0)

            # Then compute the rest
            update_dict = self.recalc_dict_from_pro_share(pro_sh_dict, node)
            self.network.nodes[node].update(update_dict)
    # TODO: check
    def recalc_dict_from_pro_share(self, pro_share_update_dict: Dict[str, float], node: Any) -> Dict[str, float]:
        temporary_dict = {}
        temporary_dict['total emission from food'] = 0.0
        temporary_dict['total spending on food']   = 0.0
        temporary_dict['redmeat']                  = 0.0

        tot_prot = self.network.nodes[node].get('total protein content', 0.0)
        adults   = self.network.nodes[node].get('adults', 1)

        for com in basic_protein_cols:
            share_key  = attrdict[com]['protein_share']
            share_val  = pro_share_update_dict.get(share_key, 0.0)
            pro_val    = share_val * tot_prot
            quantity   = pro_val / self._conversion_factors[com]
            emission   = quantity * self._emission_factors[com]
            spending   = quantity * self.network.nodes[node].get(attrdict[com]['price'], 0.0)

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

        # Denormalize by # of adults
        temporary_dict['denormalized total emission from food'] = (
            temporary_dict['total emission from food'] * adults
        )
        return temporary_dict
    ############################  main  update function  ###################################
    @staticmethod
    def g(x1: float, x2: float, scaling_param: float = 30.0) -> float:
        """
        Example function used for similarity or adjacency weighting.
        """
        if x1 == 0:
            return 0.0
        return 1 - (1 + np.exp(scaling_param * (x2 - x1) / x1)) ** (-1 / scaling_param)

    @staticmethod
    def calculate_similarity_Household_size(G: nx.Graph, i: Any, j: Any) -> float:
        features = ['adults']
        diff = sum((G.nodes[i][feat] - G.nodes[j][feat])**2 for feat in features)
        return 1.0 / (0.1 + diff)  # example

    @staticmethod
    def assym_sp_sym_HHsize(G: nx.Graph, i: Any, j: Any) -> float:
        xs = G.nodes[i].get('total spending on food', 0.0)
        ys = G.nodes[j].get('total spending on food', 0.0)
        return AllAgents.g(xs, ys) * AllAgents.calculate_similarity_Household_size(G, i, j)

    @staticmethod
    def redmeatspendingsim(G: nx.Graph, i: Any, j: Any) -> float:
        r1 = G.nodes[i].get('redmeat', 0.0)
        r2 = G.nodes[j].get('redmeat', 0.0)
        return 1.0 / (1 + (r1 - r2)**2)

    @staticmethod
    def cls_similarity(G: nx.Graph, i: Any, j: Any, same_cluster_factor: float = 1.0, 
                          diff_cluster_factor: float = 1.0, cluster_attr: str = 'kmeans_cluster') -> float:
        """
        Returns same_cluster_factor if both nodes are in the same lifestyle cluster,
        otherwise returns diff_cluster_factor.
        
        Parameters
        ----------
        G : nx.Graph
            The social network graph
        i, j : Any
            Node identifiers
        same_cluster_factor : float
            Multiplier when nodes are from the same cluster (default 1.0)
        diff_cluster_factor : float
            Multiplier when nodes are from different clusters (default 1.0)
        cluster_attr : str
            Node attribute name for cluster membership (default 'kmeans_cluster')
        """
        meat_list = [ 'Beef and veal aggregated protein share','Pork aggregated protein share','Lamb and goat aggregated protein share','Poultry aggregated protein share']
        cluster_i = G.nodes[i].get(cluster_attr, None)
        cluster_j = G.nodes[j].get(cluster_attr, None)
        
        if cluster_i is None or cluster_j is None:
            # If cluster info is missing, return 1.0 (no effect)
            return 1.0
        
        if cluster_i == cluster_j:
            return same_cluster_factor
        else:
            return diff_cluster_factor
        

    @staticmethod
    def meat_quantile(share):
        X = 0.2580763390058186
        Y = 0.43870006943830847
        if share < X:
            return 1
        elif share < Y:
            return 2
        else:
            return 3

    @staticmethod
    def get_meat_share(G, node):
        meat_cols = [
            'Beef and veal aggregated protein share',
            'Pork aggregated protein share',
            'Lamb and goat aggregated protein share',
            'Poultry aggregated protein share'
        ]
        return sum(G.nodes[node].get(col, 0.0) for col in meat_cols)
        
    @staticmethod
    def quantile_similarity(G: nx.Graph, i: Any, j: Any,
                            same_cluster_factor: float = 1.0,
                            diff_cluster_factor: float = 0.0) -> float:
        """
        Quantile-based similarity on meat protein share.
        Nodes are placed into quantiles 1, 2, 3 using thresholds X and Y.
        If nodes fall in the same quantile → same_cluster_factor
        else → diff_cluster_factor
        """

        # compute meat protein share for both nodes
        share_i = AllAgents.get_meat_share(G, i)
        share_j = AllAgents.get_meat_share(G, j)

        # determine quantile
        qi = AllAgents.meat_quantile(share_i)
        qj = AllAgents.meat_quantile(share_j)

        # if either is missing, return neutral
        if qi is None or qj is None:
            return 1.0

        # return similarity
        if qi == qj:
            return same_cluster_factor
        else:
            return diff_cluster_factor

    def weighted_update(
        self,
        alpha: float,
        beta: float,
        items: List[str],
        alpha_0: float,
        spending_inc_tolerance: float,
        static_nodes = None,
        noise_std: float = 0.0,
        g_func=None,
        hh_similarity_func=None,
        redmeat_similarity_func=None,
        cluster_similarity_func=None,
        logger=None,
        iteration_num: int = 0,
        prnt = False
    ) -> None:
        """
        Weighted update rule:
         - For each node in random order:
             * skip if in static_nodes
             * compute neighbor weights
             * gather neighbor influence
             * recalc node's consumption & update 
        """
        
        G = self.network
        
        if static_nodes is None:
            static_nodes = self._AM_static_nodes
        # 1) Shuffle nodes (use global RNG seeded by main script)
        nodes_in_random_order = list(G.nodes)
        random.shuffle(nodes_in_random_order)

        SKIPPED = 0
        DIFF    = 0
        UPDATED = 0
        static_nodes = self._AM_static_nodes
        
        weight_attr_keys = [
            'adults',
            'total spending on food',
            'redmeat',
            'cult1',
            'income',
            'HC24',
            'cult',
            'consumption_rate',
            'dur_spend_r',
            'ndur_spend_r',
            'serv_spend_r'
        ]

        # Log iteration start if logger provided
        if logger:
            if hasattr(logger, 'log_iterations') and logger.log_iterations:
                logger.log_iteration_stats(iteration_num, G, static_nodes)
        
        for node in nodes_in_random_order:
            neighbors = list(G.neighbors(node))
            if not neighbors or node in static_nodes:
                continue

            # Save old state
            temp_dict = G.nodes[node].copy()
            
            # Log node state before update
            if logger:
                if hasattr(logger, 'log_node_states') and logger.log_node_states:
                    logger.log_node_state(iteration_num, node, temp_dict, stage="before_update")

            # Compute neighbor weights once
            neighbor_weights = {}
            total_weight = 0.0
            for nbr in neighbors:
                
                xs = G.nodes[node].get('total spending on food', 0.0)
                ys = G.nodes[nbr].get('total spending on food', 0.0)

                g = g_func or AllAgents.g
                hh_sim = hh_similarity_func or AllAgents.calculate_similarity_Household_size
                red_sim = redmeat_similarity_func or AllAgents.redmeatspendingsim
                cluster_sim = cluster_similarity_func(G, node, nbr) if cluster_similarity_func else 1.0

                #print("beta"*10, beta)
                #print(inspect.getsource(cluster_similarity_func))

                # Calculate components
                spending_sim = g(xs, ys)
                w = (alpha * spending_sim * hh_sim(G, node, nbr) + beta * red_sim(G, node, nbr)) * cluster_sim
                
                # Log weight calculation
                if logger:
                    if hasattr(logger, 'log_weights') and logger.log_weights:
                        logger.log_weight_calculation(
                            iteration=iteration_num,
                            node=node,
                            neighbor=nbr,
                            spending_sim=spending_sim,
                            household_sim=hh_sim(G, node, nbr),
                            redmeat_sim=red_sim(G, node, nbr),
                            cluster_sim=cluster_sim,
                            alpha=alpha,
                            beta=beta,
                            final_weight=w,
                            node_cluster=G.nodes[node].get('kmeans_cluster', None),
                            neighbor_cluster=G.nodes[nbr].get('kmeans_cluster', None),
                            node_attrs={key: G.nodes[node].get(key, '') for key in weight_attr_keys},
                            neighbor_attrs={key: G.nodes[nbr].get(key, '') for key in weight_attr_keys}
                        )
                
                neighbor_weights[nbr] = w
                total_weight += w
            if total_weight < 1e-2:
                SKIPPED += 1
                continue

            # Accumulate new protein share from neighbors
            influence_dict = {}
            for com in basic_protein_cols:
                share_key = attrdict[com]['protein_share']
                item_influence = 0.0
                for nbr in neighbors:
                    item_influence += neighbor_weights[nbr] * G.nodes[nbr].get(share_key, 0.0) + noise_std*np.random.randn()
                influence_dict[share_key] = item_influence / total_weight

            # Recompute the node's food consumption
            update_dict = self.recalc_dict_from_pro_share(influence_dict, node)
            influence_dict.update(update_dict)

            # Spending change to adapt alpha_0
            new_spending = influence_dict['total spending on food']
            old_spending = temp_dict.get('total spending on food', 0.0)
            diff = new_spending - old_spending
            if diff > 0.0:
                con1 = (old_spending * spending_inc_tolerance) / diff
                if con1 < alpha_0:
                    new_alpha_0 = con1
                else:
                    new_alpha_0 = alpha_0
            else:
                new_alpha_0 = alpha_0
                DIFF += 1

            # Final smoothing update
            final_dict = {}
            for k, new_val in influence_dict.items():
                old_val = temp_dict.get(k, 0.0)
                final_dict[k] = (1 - new_alpha_0) * old_val + new_alpha_0 * new_val

            G.nodes[node].update(final_dict)
            UPDATED += 1
            
            # Log node state after update
            if logger:
                if hasattr(logger, 'log_node_states') and logger.log_node_states:
                    logger.log_node_state(iteration_num, node, final_dict, stage="after_update")
                if hasattr(logger, 'log_influence') and logger.log_influence:
                    logger.log_influence(
                        iteration=iteration_num,
                        node=node,
                        influences=neighbor_weights,
                        total_weight=total_weight,
                        old_state=temp_dict,
                        new_state=final_dict
                    )

        # Recompute denormalized
        for node in G.nodes:
            adults = G.nodes[node].get('adults', 1)
            G.nodes[node]['denormalized total emission from food'] = (
                G.nodes[node]['total emission from food'] * adults
            )
        if prnt:
            print(
                'skipped for low weight:', SKIPPED,
                "diff neg for:", DIFF,
                "updated:", UPDATED,
                "alpha:", alpha,
                "beta:", beta,
                "alpha_0:", alpha_0
            )
            
        # Log final iteration statistics
        if logger:
            if hasattr(logger, 'log_iterations') and logger.log_iterations:
                logger.log_iteration_stats(iteration_num, G, static_nodes)

    def quantity_table(self, key: str) -> None:
        """
        Example method that shows mean, std of an emission or spending attribute across all nodes.
        """
        orig_output = {com: [] for com in basic_protein_cols}
        for node in self.network.nodes:
            for com in basic_protein_cols:
                val = self.network.nodes[node].get(attrdict[com][key], 0.0)
                orig_output[com].append(val)

        # Build stats
        table_data = []
        for com in basic_protein_cols:
            arr = np.array(orig_output[com], dtype=float)
            table_data.append({
                'Commodity': com,
                'Mean': arr.mean(),
                'Std': arr.std()
            })
        df = pd.DataFrame(table_data)
        print("Mean and Std of", key, "per commodity:")
        print(df)

###############################################################################
# ImposeDiet class
###############################################################################
class ImposeDiet:
    def __init__(
        self,
        _conversion_factors: Dict[str, float],
        _emission_factors: Dict[str, float],
        objAllAgents: AllAgents,
        basic_protein_columns: List[str], 
        agentlist1 = None,
        seed = 1996
    ):
        self._conversion_factors = _conversion_factors
        self._emission_factors   = _emission_factors
        self.AllAgents           = objAllAgents
        self.network             = objAllAgents.network
        self._basic_protein_columns = basic_protein_columns
        if agentlist1 is None:
            self._agentlist = objAllAgents._AM_static_nodes
        else:
            self._agentlist = agentlist1
        random.seed(seed)
        np.random.seed(seed)

    def redistribute_pro_share(self, com_list: List[str], target_share: float, node: Any) -> None:
        """
        Scale the existing shares in com_list so their sum is target_share.
        """
        if len(com_list) == 1:
            # If only one commodity, set its share directly.
            self.network.nodes[node][attrdict[com_list[0]]['protein_share']] = target_share
            return
        sum_orig = 0.0
        for c in com_list:
            sum_orig += self.network.nodes[node].get(attrdict[c]['protein_share'], 0.0)
        if sum_orig > 1e-7:
            ratio = target_share / sum_orig
            for c in com_list:
                old_val = self.network.nodes[node].get(attrdict[c]['protein_share'], 0.0)
                self.network.nodes[node][attrdict[c]['protein_share']] = old_val * ratio
        else:
            # If sum is 0, distribute equally or whatever strategy
            eq_share = target_share / len(com_list)
            for c in com_list:
                self.network.nodes[node][attrdict[c]['protein_share']] = eq_share

    def pre_impose_lanclet_diet(self, agentlist: Optional[List[Any]] = None) -> None:
        """
        Example: sets target protein shares for a "Lanclet diet".
        """
        if agentlist is None:
            agentlist = self._agentlist
            if agentlist is None:
                agentlist = list(self.network.nodes)
        for node in agentlist:
            self.redistribute_pro_share(['egg'], 0.01381509033, node)
            self.redistribute_pro_share(['bread', 'rice'], 0.2465462274, node)
            self.redistribute_pro_share(['vegetables without dried'], 0.3188097768, node)
            self.redistribute_pro_share(['Dried vegetables'], 0.07970244421, node)
            self.redistribute_pro_share(['fish and seafood'], 0.02975557917, node)
            self.redistribute_pro_share(['milk without cheese', 'Cheese'], 0.265674814, node)
            self.redistribute_pro_share(
                ['Beef and veal aggregated','Pork aggregated','Lamb and goat aggregated'],
                0.01487778959, node
            )
            self.redistribute_pro_share(['Poultry aggregated'], 0.03081827843, node)

    def pro_share_transfer_to_veg(self, com_list: List[str], node: Any) -> None:
        """
        Transfer the entire share from com_list to veggies.
        Splits 2/3 or 1/3 as an example, etc.
        """
        transfer_share = 0.0
        for c in com_list:
            old_val = self.network.nodes[node].get(attrdict[c]['protein_share'], 0.0)
            transfer_share += old_val
            self.network.nodes[node][attrdict[c]['protein_share']] = 0.0
        # Distribute to vegetarian sources
        self.network.nodes[node][attrdict['vegetables without dried']['protein_share']] += transfer_share / 3.0
        self.network.nodes[node][attrdict['Dried vegetables']['protein_share']]         += transfer_share / 3.0
        # Possibly do more distribution logic here

    def impose_lanclet_diet(self, agentlist: Optional[List[Any]]=None) -> None:
        if agentlist is None:
            agentlist = self._agentlist
            if agentlist is None:
                agentlist = list(self.network.nodes)
        self.pre_impose_lanclet_diet(agentlist)
        self.AllAgents.recalc_node_attr_from_pro_share(agentlist)

    def impose_vegetarian_diet(self, agentlist: Optional[List[Any]]=None) -> None:
        
        if agentlist is None:
            agentlist = self._agentlist
            if agentlist is None:
                agentlist = list(self.network.nodes)
        self.pre_impose_lanclet_diet(agentlist)
        for node in agentlist:
            self.pro_share_transfer_to_veg(
                ['Beef and veal aggregated','Pork aggregated','Lamb and goat aggregated','Poultry aggregated','fish and seafood'],
                node
            )
        self.AllAgents.recalc_node_attr_from_pro_share(agentlist)

    def impose_vegan_diet(self, agentlist: Optional[List[Any]]=None) -> None:
        
        if agentlist is None:
            agentlist = self._agentlist
            if agentlist is None:
                agentlist = list(self.network.nodes)
        self.pre_impose_lanclet_diet(agentlist)
        for node in agentlist:
            self.pro_share_transfer_to_veg(
                ['Beef and veal aggregated','Pork aggregated','Lamb and goat aggregated',
                 'Poultry aggregated','fish and seafood','egg','milk without cheese','Cheese'],
                node
            )
        self.AllAgents.recalc_node_attr_from_pro_share(agentlist)

# --------------------------------------------------------------------------
# EXAMPLE USAGE (when running this file as a script)
# --------------------------------------------------------------------------
if __name__ == "__main__":
    import networkx as nx

    # Suppose we load a GraphML or build a random graph
    G = nx.read_graphml("graph_export2.graphml")

    # 1) Create an AllAgents instance 
    all_agents_obj = AllAgents(
        conversion_factors,
        emission_factors,
        G,
        basic_protein_cols
    )

    # 2) Impose a diet (example: vegetarian)
    impose_diet_obj = ImposeDiet(
        conversion_factors,
        emission_factors,
        all_agents_obj,
        basic_protein_cols
    )
    impose_diet_obj.impose_vegetarian_diet()

    # 3) For demonstration, let's call weighted_update to modify the agent's consumption
    #    We'll call it on the instance, which references self.network
    alpha   = 0.3
    beta    = 5.0
    alpha_0 = 0.3
    sit     = 0.005
    static_nodes = []
    items   = pro_share_list  # could be your list of interest

    all_agents_obj.weighted_update(
        alpha=alpha,
        beta=beta,
        items=items,
        alpha_0=alpha_0,
        spending_inc_tolerance=sit,
        static_nodes=static_nodes
    )

    # 4) Print final stats (example: emission)
    all_agents_obj.quantity_table('emission')

