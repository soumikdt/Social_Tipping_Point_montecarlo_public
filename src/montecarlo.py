#──────────────────────────────────────────────────────────────
# Environment setup: enforce single-threaded BLAS
#──────────────────────────────────────────────────────────────
import os
os.environ.update({
    "OMP_NUM_THREADS":      "1",
    "MKL_NUM_THREADS":      "1",
    "OPENBLAS_NUM_THREADS": "1",
})

#──────────────────────────────────────────────────────────────
# Seed setting for reproducibility
#──────────────────────────────────────────────────────────────
import random
import numpy as np
MASTER_SEED = 1996
random.seed(MASTER_SEED)
np.random.seed(MASTER_SEED)

#──────────────────────────────────────────────────────────────
# Standard library imports
#──────────────────────────────────────────────────────────────
import sys
import math
import pickle
import glob
import copy
from datetime import datetime
from typing import Any
import warnings

#──────────────────────────────────────────────────────────────
# Data and plotting
#──────────────────────────────────────────────────────────────
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#──────────────────────────────────────────────────────────────
# sklearn and joblib
#──────────────────────────────────────────────────────────────
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from joblib import Parallel, delayed

#──────────────────────────────────────────────────────────────
# Graph/network tools
#──────────────────────────────────────────────────────────────
import networkx as nx

#──────────────────────────────────────────────────────────────
# Local modules
#──────────────────────────────────────────────────────────────
import Dicts_Lists_Helpers
import SocialNetworkGenerator
import AgentManipulator
import NodeSelector
import Simulator

# Reload for live development
from importlib import reload
reload(Dicts_Lists_Helpers)
reload(SocialNetworkGenerator)
reload(AgentManipulator)
reload(NodeSelector)
reload(Simulator)

#──────────────────────────────────────────────────────────────
# Direct imports from local modules
#──────────────────────────────────────────────────────────────
from Dicts_Lists_Helpers import *
from SocialNetworkGenerator import *
from AgentManipulator import *
from NodeSelector import *
from Simulator import *

#──────────────────────────────────────────────────────────────
# Timestamp for logging/outputs
#──────────────────────────────────────────────────────────────
print(datetime.now())




#──────────────────────────────────────────────────────────────
# Load original Austrian household dataset
#──────────────────────────────────────────────────────────────
HH_AT_df_org = pd.read_excel('HBS_HH_AT.xlsx')

# Inspect the structure of the dataset
print(HH_AT_df_org.info())

# Create a working copy to preserve the original
HH_AT_df = HH_AT_df_org.copy()

#──────────────────────────────────────────────────────────────
# Prepare index sets for tracking
#──────────────────────────────────────────────────────────────
orig_idx = set(HH_AT_df.index)
n_orig = len(orig_idx)

HH_set_size = {}
#################################################################################################
#################################################################################################
#################################################################################################
#################################################################################################

for _ in range(30):
    sample_seed = (MASTER_SEED + _ ) % (2**32 - 1)
    HH_AT_df_sampled = HH_AT_df.sample(
        n=100_000,
        weights='HA10',
        random_state=sample_seed,
        replace=True
    )



    # 2) reset the index *and* keep the old one in a column
    HH_AT_df_sampled = (
        HH_AT_df_sampled
        .reset_index()                     # brings the old index into a column named "index"
        .rename(columns={'index':'orig_idx'})  # rename it to something meaningful
    )



    filler = pd.DataFrame(np.nan,
                            index=HH_AT_df_sampled.index,
                            columns=sorted(defrag_cols))
    HH_AT_df_sampled = pd.concat([HH_AT_df_sampled, filler], axis=1)



    HH_AT_df_sampled['totalcon']  = HH_AT_df_sampled['EUR_HE00'] 

    HH_AT_df_sampled['totalinc']  = HH_AT_df_sampled['EUR_HH099'] 


    print(len(cols_to_normalize))

    HH_AT_df_sampled['adults'] = HH_AT_df_sampled['HB05'] - HH_AT_df_sampled['HB051'] #056 contains zero values

    # Normalize these columns by dividing each row by the value in 'HB05' for that row.
    HH_AT_df_sampled[cols_to_normalize] = HH_AT_df_sampled[cols_to_normalize].div(HH_AT_df_sampled['adults'], axis=0)

    # -------------------------------------------------------------------
    #change egg no to kg
    HH_AT_df_sampled['HQ01147'] = HH_AT_df_sampled['HQ01147']*0.05

    #filling up the empty columns in the original data

    HH_AT_df_sampled['HQ0117-'] = (
        HH_AT_df_sampled['HQ01171'] +
        HH_AT_df_sampled['HQ01172'] +
        #HH_AT_df_sampled['HQ01173'] +
        HH_AT_df_sampled['HQ01174'] +
        HH_AT_df_sampled['HQ01175'] +
        HH_AT_df_sampled['HQ01176']
    )

    HH_AT_df_sampled['HQ0114-'] = (
        HH_AT_df_sampled['HQ01141'] +
        HH_AT_df_sampled['HQ01142'] +
        HH_AT_df_sampled['HQ01143'] +
        HH_AT_df_sampled['HQ01144'] +
        #HH_AT_df_sampled['HQ01145'] +
        HH_AT_df_sampled['HQ01146'] 
    )

    HH_AT_df_sampled['HQ0113'] = (
        HH_AT_df_sampled['HQ01131'] +
        HH_AT_df_sampled['HQ01132'] +
        HH_AT_df_sampled['HQ01133'] +
        HH_AT_df_sampled['HQ01134'] +
        HH_AT_df_sampled['HQ01135'] +
        HH_AT_df_sampled['HQ01136'] 
    )
    # calculating dried veggie quantity form spendiing assuming the price = 5 EUR
    HH_AT_df_sampled['HQ01173'] = HH_AT_df_sampled['EUR_HE01173']/5.0

    # Meat Distribution Logic:
    # 	1.	Sum the Meat Types:
    # 	•	Create a non_basic_meat total by adding four columns (other meats, offal, dried meats, and other preparations).
    # 	•	Create a basic_meat total by adding four columns (beef & veal, pork, lamb & goat, and poultry).
    # 	2.	Adjust Each Basic Meat Value:
    # 	•	For each basic meat column, create a new adjusted column (appending a “+” to the name).
    # 	•	If the total basic meat is zero, add 25% of the non-basic meat to the column.
    # 	•	Otherwise, add a portion of the non-basic meat proportional to that column’s share of the total basic meat.
    # 
    # 

    HH_AT_df_sampled['non_basic_meat']=HH_AT_df_sampled['HQ01125']+HH_AT_df_sampled['HQ01126']+HH_AT_df_sampled['HQ01127']+HH_AT_df_sampled['HQ01128']

    HH_AT_df_sampled['basic_meat']=HH_AT_df_sampled['HQ01121']+HH_AT_df_sampled['HQ01122']+HH_AT_df_sampled['HQ01123']+HH_AT_df_sampled['HQ01124']

    # Update columns based on whether 'basic_meat' is 0 or not
    for col in ['HQ01121', 'HQ01122', 'HQ01123', 'HQ01124']:
        stcol = col + '+'
        HH_AT_df_sampled[stcol] = np.where(
            HH_AT_df_sampled['basic_meat'] == 0,
            HH_AT_df_sampled[col] + HH_AT_df_sampled['non_basic_meat'] * 0.25, # shall I change the 0.25 to 0.33 because lamb my not be in sausage etc.
            HH_AT_df_sampled[col] + HH_AT_df_sampled['non_basic_meat'] * (HH_AT_df_sampled[col] / HH_AT_df_sampled['basic_meat'])
        )

    HH_AT_df_sampled.rename(columns=protein_share_dict, inplace=True)
    print("%&*&^*&^*&"*10)
    # ## price
    def print_duplicates(lst):
        seen = set()
        duplicates = set(x for x in lst if x in seen or seen.add(x))
        for d in duplicates:
            print(d)

    print_duplicates(list(HH_AT_df_sampled.columns))

    dups = HH_AT_df_sampled.index[HH_AT_df_sampled.index.duplicated()]
    print(dups.unique())
    print("%&*&^*&^*&"*10)

    # spendings for meat items are handled in similar way
    HH_AT_df_sampled['non_basic_meat_spending']=HH_AT_df_sampled['EUR_HE01125']+HH_AT_df_sampled['EUR_HE01126']+HH_AT_df_sampled['EUR_HE01127']+HH_AT_df_sampled['EUR_HE01128']

    HH_AT_df_sampled['basic_meat_spending']=HH_AT_df_sampled['EUR_HE01121']+HH_AT_df_sampled['EUR_HE01122']+HH_AT_df_sampled['EUR_HE01123']+HH_AT_df_sampled['EUR_HE01124']

    HH_AT_df_sampled['Beef and veal aggregated_total_spending'] = np.where(HH_AT_df_sampled['basic_meat_spending'] == 0,
                                                                        HH_AT_df_sampled['EUR_HE01121'] + HH_AT_df_sampled['non_basic_meat_spending']*(0.25),
                                                                            HH_AT_df_sampled['EUR_HE01121'] + HH_AT_df_sampled['non_basic_meat_spending']*(HH_AT_df_sampled['EUR_HE01121']/HH_AT_df_sampled['basic_meat_spending']))
    HH_AT_df_sampled['Pork aggregated_total_spending'] = np.where(HH_AT_df_sampled['basic_meat_spending'] == 0,
                                                                HH_AT_df_sampled['EUR_HE01122'] + HH_AT_df_sampled['non_basic_meat_spending']*(0.25),
                                                                HH_AT_df_sampled['EUR_HE01122'] + HH_AT_df_sampled['non_basic_meat_spending']*(HH_AT_df_sampled['EUR_HE01122']/HH_AT_df_sampled['basic_meat_spending']))
    HH_AT_df_sampled['Lamb and goat aggregated_total_spending'] = np.where(HH_AT_df_sampled['basic_meat_spending'] == 0,
                                                                HH_AT_df_sampled['EUR_HE01123'] + HH_AT_df_sampled['non_basic_meat_spending']*(0.25),
                                                                HH_AT_df_sampled['EUR_HE01123'] + HH_AT_df_sampled['non_basic_meat_spending']*(HH_AT_df_sampled['EUR_HE01123']/HH_AT_df_sampled['basic_meat_spending']))
    HH_AT_df_sampled['Poultry aggregated_total_spending'] = np.where(HH_AT_df_sampled['basic_meat_spending'] == 0,
                                                                HH_AT_df_sampled['EUR_HE01124'] + HH_AT_df_sampled['non_basic_meat_spending']*(0.25),
                                                                HH_AT_df_sampled['EUR_HE01124'] + HH_AT_df_sampled['non_basic_meat_spending']*(HH_AT_df_sampled['EUR_HE01124']/HH_AT_df_sampled['basic_meat_spending']))

    HH_AT_df_sampled['vegetables without dried_total_spending'] = (HH_AT_df_sampled['EUR_HE01171'] + HH_AT_df_sampled['EUR_HE01172'] + HH_AT_df_sampled['EUR_HE01174'] + HH_AT_df_sampled['EUR_HE01175'] + HH_AT_df_sampled['EUR_HE01176'])
    HH_AT_df_sampled['milk without cheese_total_spending'] = (HH_AT_df_sampled['EUR_HE01141'] + HH_AT_df_sampled['EUR_HE01142'] + HH_AT_df_sampled['EUR_HE01143'] + HH_AT_df_sampled['EUR_HE01144'] + HH_AT_df_sampled['EUR_HE01146'])
    HH_AT_df_sampled['fish and seafood_total_spending'] = (HH_AT_df_sampled['EUR_HE01131'] + HH_AT_df_sampled['EUR_HE01132'] + HH_AT_df_sampled['EUR_HE01133'] + HH_AT_df_sampled['EUR_HE01134'] + HH_AT_df_sampled['EUR_HE01135'] + HH_AT_df_sampled['EUR_HE01136'] )
    HH_AT_df_sampled['Cheese_total_spending'] = HH_AT_df_sampled['EUR_HE01145']
    HH_AT_df_sampled['Dried vegetables_total_spending'] = HH_AT_df_sampled['EUR_HE01173']
    HH_AT_df_sampled['rice_total_spending'] = HH_AT_df_sampled['EUR_HE01111']
    HH_AT_df_sampled['bread_total_spending'] = HH_AT_df_sampled[ 'EUR_HE01113']
    HH_AT_df_sampled['egg_total_spending'] = HH_AT_df_sampled['EUR_HE01147']


    # Below is a short explanation of the code’s(below) logic:
    # 	1.	Calculate Price:
    # For each pair of quantity and spending columns, a new price column is created by dividing spending by quantity.
    # 	2.	Check Data Quality:
    # The code counts NaN/Inf values for spending, quantity, and price. It then prints summary statistics.
    # 	3.	Handle Non-finite Data:
    # 	•	If there are no finite price values, the user is prompted to provide a replacement value.
    # 	•	The replacement value is applied, and the change is shown via updated stats and a histogram.
    # 	4.	Outlier Capping:
    # If finite values exist, the code:
    # 	•	Plots a histogram of the finite price values.
    # 	•	Computes basic statistics (mean, median, mode, standard deviation).
    # 	•	Sets capping thresholds at the 1st and 99th percentiles.
    # 	•	Replaces non-finite values with the median, then caps all values within the defined thresholds.
    # 	•	Displays updated summary statistics and histograms after capping.
    # 
    # Finally, it records any columns that had no valid data. This modular structure helps avoid redundancy by using helper functions for histogram plotting and capping.

    # Lists to record columns that become empty after cleaning
    empty = []
    empty_code = []

    # Process each pair of quantity and spending columns
    for qty_col, spending_col in zip(total_quantity_cols, total_spending_cols):
        process_price_column(HH_AT_df_sampled, qty_col, spending_col, empty, empty_code, viz_var=False)

    print("Empty columns:", empty)
    print("Corresponding quantity codes:", empty_code)

    # re adjustment of quanity and spnding using the updated price vals.

    for idx, col in enumerate(total_quantity_cols):
        sp_item = total_spending_cols[idx]
        pr_item = prices_col[idx]

        # Condition 1: Both spending and quantity are non-zero (ensures consistency)
        mask1 = (HH_AT_df_sampled[col] != 0) & (HH_AT_df_sampled[sp_item] != 0)
        HH_AT_df_sampled.loc[mask1, col] = HH_AT_df_sampled.loc[mask1, sp_item] / HH_AT_df_sampled.loc[mask1, pr_item]

        # Condition 2: Quantity is zero but spending is non-zero → compute quantity
        mask2 = (HH_AT_df_sampled[col] == 0) & (HH_AT_df_sampled[sp_item] != 0)
        HH_AT_df_sampled.loc[mask2, col] = HH_AT_df_sampled.loc[mask2, sp_item] / HH_AT_df_sampled.loc[mask2, pr_item]

        # Condition 3: Quantity is non-zero but spending is zero → compute spending
        mask3 = (HH_AT_df_sampled[col] != 0) & (HH_AT_df_sampled[sp_item] == 0)
        HH_AT_df_sampled.loc[mask3, sp_item] = HH_AT_df_sampled.loc[mask3, pr_item] * HH_AT_df_sampled.loc[mask3, col]

    # ## conversions

    # ### quantity to protein content conversion

    for idx, col in enumerate(basic_protein_cols):
        c_f = list(conversion_factors.keys())[idx]
        HH_AT_df_sampled[pro_list[idx]] = HH_AT_df_sampled[col]*conversion_factors[c_f]


    HH_AT_df_sampled['total protein content'] = HH_AT_df_sampled[pro_list].sum(axis=1)

    HH_AT_df_sampled[pro_list].describe()

    HH_AT_df_sampled['total protein content'].describe()

    # ### filtering out rows based on total protein content

    total_rows = len(HH_AT_df_sampled)
    print(f"Total rows before removal: {total_rows}")

    HH_AT_df_sampled_filtered = HH_AT_df_sampled[HH_AT_df_sampled['total protein content'] > 10000].copy()

    filtered_rows = len(HH_AT_df_sampled_filtered)
    print(f"Total rows after removal: {filtered_rows}")
    removed_rows = total_rows - filtered_rows
    fraction_removed = removed_rows / total_rows

    print(f"Fraction of rows removed: {fraction_removed:.2%}")


    HH_AT_df_sampled = HH_AT_df_sampled_filtered


    # ### protein content to protein share

    var1 = 1.0/float(len(pro_share_list))

    # Loop through the protein list
    for col in pro_list:
        HH_AT_df_sampled[col+" share"] = np.where(
            HH_AT_df_sampled['total protein content'] == 0,  # Check if total_protein is zero
            var1,  #TODO: Assign 0.1 if total_protein is zero, setting zero is avoided so that sum of prorein shares add up to one, other wise it will make others deviate form one after a few iterations.
            HH_AT_df_sampled[col] / HH_AT_df_sampled['total protein content']  # Otherwise, compute the ratio
        )

    # ## env factors 

    HH_AT_df_sampled['culture2'] = (HH_AT_df_sampled['EUR_HE0942']-HH_AT_df_sampled['EUR_HE09423']) 
    HH_AT_df_sampled['income'] = HH_AT_df_sampled['EUR_HE00']
    HH_AT_df_sampled['cult1'] = HH_AT_df_sampled['culture2']
    HH_AT_df_sampled['cult'] = (HH_AT_df_sampled['cult1'] > 50).astype(int)

    HH_AT_df_sampled['pet'] = HH_AT_df_sampled['EUR_HE0934']
    HH_AT_df_sampled['fuel'] = HH_AT_df_sampled['EUR_HE0722'] 
    HH_AT_df_sampled['Package holidays'] = HH_AT_df_sampled['EUR_HE0960']
    HH_AT_df_sampled['redmeat'] = HH_AT_df_sampled['Beef and veal aggregated_total_spending'] + HH_AT_df_sampled['Pork aggregated_total_spending'] + HH_AT_df_sampled['Lamb and goat aggregated_total_spending'] 

    HH_AT_df_sampled['redmeatproteinshare'] = (HH_AT_df_sampled['Beef and veal aggregated protein share'] + HH_AT_df_sampled['Pork aggregated protein share'] + HH_AT_df_sampled['Lamb and goat aggregated protein share'])  #FIX ed here


    # ## emission features

    for idx, itm in enumerate(emission_list):
        itq = basic_protein_cols[idx]
        its = total_spending_cols[idx]
        if itq in emission_by_quant:
            HH_AT_df_sampled[itm] = HH_AT_df_sampled[itq]*emission_factors[itq]

        else: 
            HH_AT_df_sampled[itm] = HH_AT_df_sampled[its]*emission_factors[itq]



    # ## final df

    HH_AT_df_sampled['total spending on food'] = HH_AT_df_sampled[total_spending_cols].sum(axis = 1)

    HH_AT_df_sampled['total emission from food'] = HH_AT_df_sampled[emission_list].sum(axis = 1)

    HH_AT_df_sampled['denormalized total emission from food'] = HH_AT_df_sampled['total emission from food']*HH_AT_df_sampled['adults']

    HH_AT_df_sampled['sum of protein share (check)'] = HH_AT_df_sampled[pro_share_list].sum(axis = 1)

    # # adding clusteringcols
    # 

    #HH_AT_df_sampled.to_csv("preprocessedconsumptions2_before_cluster_ICM.csv", index=True)

    #HH_AT_df_sampled = pd.read_csv("preprocessedconsumptions2_before_cluster_ICM.csv", index_col=0)

    # ## feature prep


    HH_AT_df_sampled['consumption_rate'] = HH_AT_df_sampled['EUR_HE00']/HH_AT_df_sampled['EUR_HH099']


    HH_AT_df_sampled['dur_spend_r'] = HH_AT_df_sampled[dur_list].sum(axis=1)/ HH_AT_df_sampled['EUR_HE00']
    HH_AT_df_sampled['ndur_spend_r'] = HH_AT_df_sampled[ndur_list].sum(axis=1)/  HH_AT_df_sampled['EUR_HE00']
    HH_AT_df_sampled['serv_spend_r'] = HH_AT_df_sampled[serv_list].sum(axis=1)/  HH_AT_df_sampled['EUR_HE00']



    HH_AT_df_sampled['dur_ndur_ratio'] = np.where(
        HH_AT_df_sampled['ndur_spend_r'] > 0,
        HH_AT_df_sampled['dur_spend_r'] / HH_AT_df_sampled['ndur_spend_r'],
        np.nan     # uses a true NaN, keeps dtype float64
    )
    HH_AT_df_sampled['serv_ndur_ratio'] = np.where(
        HH_AT_df_sampled['ndur_spend_r'] > 0,
        HH_AT_df_sampled['serv_spend_r'] / HH_AT_df_sampled['ndur_spend_r'],
        np.nan
    )


    print(HH_AT_df_sampled[['dur_spend_r', 'ndur_spend_r', 'serv_spend_r', 'serv_ndur_ratio','dur_ndur_ratio']].describe())



    ########################################################################################
    HH_AT_df_sampled4=HH_AT_df_sampled.copy()

    HH_AT_df_sampled6=HH_AT_df_sampled.copy()

    missing_indx = set(HH_AT_df_sampled.index[HH_AT_df_sampled['consumption_rate'] > 4])
    #print(missing_indx)
    HH_AT_df_sampled = HH_AT_df_sampled[HH_AT_df_sampled['consumption_rate'] <= 4]



    missing_indx.update(set(HH_AT_df_sampled.index[HH_AT_df_sampled['dur_spend_r'] > 0.8]))
    #print(missing_indx)
    HH_AT_df_sampled = HH_AT_df_sampled[HH_AT_df_sampled['dur_spend_r'] <= 0.8]



    missing_indx.update(set(HH_AT_df_sampled.index[HH_AT_df_sampled['ndur_spend_r'] > 0.8]))
    #print(missing_indx)
    HH_AT_df_sampled = HH_AT_df_sampled[HH_AT_df_sampled['ndur_spend_r'] <= 0.8]



    missing_indx.update(set(HH_AT_df_sampled.index[HH_AT_df_sampled['serv_spend_r'] > 0.8]))
    #print(missing_indx)
    HH_AT_df_sampled = HH_AT_df_sampled[HH_AT_df_sampled['serv_spend_r'] <= 10]

    features = ['consumption_rate', 'dur_spend_r', 'ndur_spend_r', 'serv_spend_r']

    clus_col = features + ['total emission from food']
    # Create a separate DataFrame for clustering from the original data.
    #df_clustering = HH_AT_df_sampled[clus_col].copy() 
    assert HH_AT_df_sampled.index.is_monotonic_increasing

    #HH_AT_df_sampled, ind_rm = remove_top_percent_outliers_iqr(HH_AT_df_sampled, clus_col, pct = 2)

    ind_rm = []

    print(ind_rm)
    missing_indx.update(set(ind_rm))


    features = ['consumption_rate', 'dur_spend_r', 'ndur_spend_r', 'serv_spend_r']


    df_clustering = HH_AT_df_sampled[clus_col].copy()

    scaler1 = MinMaxScaler()
    df_clustering = pd.DataFrame(
        scaler1.fit_transform(df_clustering),
        columns=df_clustering.columns,
        index=df_clustering.index
    )

    # --- MinMaxScaler fitted earlier as scaler1 ---

    # Compute a and b for MinMax scaling
    a_minmax = 1 / (scaler1.data_max_ - scaler1.data_min_)
    b_minmax = -scaler1.data_min_ / (scaler1.data_max_ - scaler1.data_min_)

    # Build forward and inverse transform dictionaries
    forward_minmax_transform = {}
    inverse_minmax_transform = {}

    for col, a, b in zip(df_clustering.columns, a_minmax, b_minmax):
        forward_minmax_transform[col] = (a, b)         # scaled = a * x + b
        inverse_minmax_transform[col] = (1/a, -b/a)     # original = (scaled - b)/a

    # Optional: show dictionaries
    print("\nRaw dictionary formats built")
    #print("Forward MinMax Transform Dict:")
    #print(forward_minmax_transform)

    #print("\nInverse MinMax Transform Dict:")
    #print(inverse_minmax_transform)

    # Plot for inspection
    #plt.boxplot(df_clustering)
    #plt.title("MinMax Scaled Features")
    #plt.show()

    print(df_clustering.shape)

    print(HH_AT_df_sampled.shape)



    # ---------------------------------------------
    # 1. Data Preparation
    # ---------------------------------------------
    # Select the predictor features and target variable.

    X = df_clustering[features].values
    y = df_clustering['total emission from food'].values

    # Standardize the features.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # --- NEW: Get StandardScaler transforms ---
    means = scaler.mean_
    scales = scaler.scale_

    a_std = 1 / scales
    b_std = -means / scales

    inv_a_std = scales
    inv_b_std = means

    forward_standard_transform = {
        col: (a, b)
        for col, a, b in zip(features, a_std, b_std)
    }

    inverse_standard_transform = {
        col: (inv_a, inv_b)
        for col, inv_a, inv_b in zip(features, inv_a_std, inv_b_std)
    }

    print("\n[Info] StandardScaler forward and inverse transform dictionaries built.")

    best_k = 3


    print("best_k is set to:" , best_k)
    

    # Run final KMeans clustering with the best_k.
    final_kmeans = KMeans(n_clusters=best_k, random_state=MASTER_SEED, init='k-means++')
    cluster_labels = final_kmeans.fit_predict(X_scaled)

    # Assign the cluster labels to the DataFrame.
    df_clustering['kmeans_cluster'] = cluster_labels

    # Compute and print the size of each cluster.
    cluster_sizes = df_clustering['kmeans_cluster'].value_counts().sort_index()
    print("\nCluster Sizes:")
    print(cluster_sizes)

    # ---------------------------------------------
    # 5. Visualization of Final Clustering
    # ---------------------------------------------

    # A. PCA Scatter Plot with Clusters in Different Colors:
    # Reduce the standardized feature space to 2 dimensions for visualization.
    pca = PCA(n_components=2, random_state=MASTER_SEED)
    pca_results = pca.fit_transform(X_scaled)

    # Create a DataFrame for PCA results.
    df_pca = pd.DataFrame(pca_results, columns=['PCA1', 'PCA2'], index=df_clustering.index)
    df_pca['kmeans_cluster'] = df_clustering['kmeans_cluster']
    df_pca['total emission from food'] = df_clustering['total emission from food']

    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        data=df_pca, x='PCA1', y='PCA2', 
        hue='kmeans_cluster', 
        palette='viridis', 
        style='kmeans_cluster', 
        s=100,
        alpha=0.8
    )
    plt.title("PCA Scatter Plot of KMeans Clusters")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend(title="KMeans Cluster")
    plt.savefig(f"PCA-Scatter-Plot-of-KMeans-Clusters-{_}.png")

    # B. Boxplot: Target Variable ("total emission from food") by Cluster:
    plt.figure(figsize=(10, 6))
    sns.boxplot(
        x='kmeans_cluster', y='total emission from food', 
        data=df_clustering
    )
    plt.title("Boxplot of 'Total Emission from Food' by KMeans Cluster")
    plt.xlabel("KMeans Cluster")
    plt.ylabel("Total Emission from Food")
    plt.savefig(f"Boxplot-of-Total-Emission-from-Food-by-KMeans-Cluster-{_}.png")


    # Choose the variables to plot.
    cols_to_plot = ["total emission from food"]+features

    # Melt the DataFrame so that the values for these columns are in a single column.
    df_melt = df_clustering.melt(id_vars="kmeans_cluster", 
                                value_vars=cols_to_plot,
                                var_name="variable", 
                                value_name="value")

    plt.figure(figsize=(12, 6))
    sns.boxplot(x="kmeans_cluster", y="value", hue="variable", data=df_melt, palette="viridis")
    plt.title("Boxplot of variables by KMeans Cluster")
    plt.xlabel("KMeans Cluster")
    plt.ylabel("Value")
    plt.legend(title="Variable")
    plt.savefig(f"Boxplot-of-variables-by-KMeans-Cluster-{_}.png")

    print(HH_AT_df_sampled.shape)
    print(df_clustering.shape)

    HH_AT_df_sampled4['kmeans_cluster'] = df_clustering['kmeans_cluster']

    # Step 1: Find the missing rows (rows that didn't get cluster labels)
    missing_cluster_mask = HH_AT_df_sampled4['kmeans_cluster'].isna()
    missing_indices = HH_AT_df_sampled4.index[missing_cluster_mask]

    #print('missing_indices=', missing_indices)
    #print('missing_indx=', missing_indx)
    print(set(missing_indices) == set(missing_indx)) 

    # 
    # using already fitted final kmeans on the forward tranbfpermed nan rows of HH_AT_df_sampled4 (no change should happen 
    # in trh eoroginal frame) find and assign cluster number to those nan worws in least invasive way.


    # Lambdas for scaling
    minmax_forward = lambda value, a_b: a_b[0] * value + a_b[1]
    standard_forward = lambda value, a_b: a_b[0] * value + a_b[1]

    # Find missing rows
    missing_indices = HH_AT_df_sampled4.index[HH_AT_df_sampled4['kmeans_cluster'].isna()]
    print(f"Assigning clusters for {len(missing_indices)} missing rows...")

    for idx in missing_indices:
        try:
            # (1) Extract raw feature values
            row_raw = HH_AT_df_sampled4.loc[idx, clus_col]

            # (2) Apply MinMax scaling on clus_col
            row_minmax = row_raw.copy()
            for col in clus_col:
                if col in forward_minmax_transform:
                    a, b = forward_minmax_transform[col]
                    row_minmax[col] = minmax_forward(row_raw[col], (a, b))

            # (3) Apply Standard scaling on features only
            row_scaled = row_minmax.copy()
            for col in features:
                if col in forward_standard_transform:
                    a, b = forward_standard_transform[col]
                    row_scaled[col] = standard_forward(row_minmax[col], (a, b))

            # (4) Predict nearest cluster
            X_row = row_scaled[features].values.reshape(1, -1)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                cluster = final_kmeans.predict(X_row)[0]

            # (5) Assign back to HH_AT_df_sampled4
            HH_AT_df_sampled4.at[idx, 'kmeans_cluster'] = cluster

        except Exception as e:
            print(f" Index {idx} failed: {e}")

    # Final sanity check
    print(f" Remaining missing cluster values: {HH_AT_df_sampled4['kmeans_cluster'].isna().sum()}")

    print(HH_AT_df_sampled4.shape)

    # # df to file

    HH_AT_df_sampled4['kmeans_cluster'] = (
        HH_AT_df_sampled4['kmeans_cluster']
        .round()          # just in case there are 2.0, 3.0, etc.
        .astype(int)      # → 0, 1, 2, 3 …
    )

    list5point5 = list5 + ['kmeans_cluster', 'consumption_rate', 'dur_spend_r', 'ndur_spend_r', 'serv_spend_r']

    HH_AT_df_sampled_with_selected_col = HH_AT_df_sampled4[list5point5]

    HH_AT_df_sampled_with_selected_col.describe()


    print(datetime.now())
    HH_AT_df_sampled_with_selected_col.to_csv(f"preprocessedconsumptions{_}_MC.csv")



    #################################################################################################
    #################################################################################################
    #################################################################################################
    #################################################################################################



    def func1(val, val2):
        #return 3 * np.log(val + 10) + 2 + 0.0 * np.log(0.0001 * val2 + 1)
        return 3*(2*np.log(0.1*val+10) + 1*np.log(0.001*val2+1) -5) + 5


    def custom_g(x1: float, x2: float, scaling_param: float = 30.0) -> float:

        re1 = 1 - (1 + np.exp(scaling_param * (x2 - x1) / x1)) ** (-1 / scaling_param)
        return re1


    def custom_g1(x1, x2, scaling_param = 0.1):
        delta = x2 - x1
        alpha = 0.05 # controls the asymmetry — this replaces the 0.5 in your np.where
        smooth_factor = 1  # larger = sharper transition between the two regimes

        # Use a smooth transition between the two slopes
        weight = 1 / (1 + np.exp(-smooth_factor * delta))  # smooth approximation to 1(x2 > x1)

        slope = weight * scaling_param * alpha + (1 - weight) * scaling_param

        rew =  1 / (1 + np.exp(-slope * delta))
        return rew

    def cosine_similarity_on_adults(G: nx.Graph, i: Any, j: Any) -> float:
        features = ['adults']
        diff = sum((G.nodes[i][feat] - G.nodes[j][feat])**2 for feat in features)
        return 1.0 / (0.1 + diff)  # example


    def my_redmeat_sim(G: nx.Graph, i: Any, j: Any) -> float:
        r1 = G.nodes[i].get('redmeat', 0.0)
        r2 = G.nodes[j].get('redmeat', 0.0)
        return 1.0 / (1 + (r1 - r2)**2)




    #################################################################################################
    #################################################################################################
    #################################################################################################
    #################################################################################################


    # 1) Load your DataFrame


    # 2) Create & analyze your SocialNetwork
    analyzer = SocialNetworkAnalyzer(HH_AT_df_sampled_with_selected_col, "cult1", "income")

    G = analyzer.create_and_analyze_graph(func1, visualize=True)


    print(f"Created network with {G.number_of_nodes()} nodes & {G.number_of_edges()} edges.")




    # 3) Create AllAgents / ImposeDiet, or any needed objects
    all_agents_obj = AllAgents(conversion_factors, emission_factors, G, basic_protein_cols)



    del HH_AT_df_sampled_with_selected_col
    #################################################################################################
    #################################################################################################
    #################################################################################################
    #################################################################################################


    # 5) Define parameter ranges (an example)
    alpha_0_list = power_partition(0.0, 1.0, 7, p=3)  # e.g. 8 partitions between 0.0 & 1.0
    alpha_list   = [0.3]
    beta_list    = [5.0]#TODO: include standard values
    sit_list     = [0.005]
    x_list       = [0.0]

    # 6) Build parameter combinations
    param_combinations = []
    for alpha_0 in alpha_0_list:
        for alpha in alpha_list:
            for beta in beta_list:
                for sit in sit_list:
                    for x in x_list:
                        param_combinations.append((alpha_0, alpha, beta, sit, x))
    print("Loop about to start")
    print(f"Total parameter combinations: {len(param_combinations)}")

    def sim_func(params, base_seed=MASTER_SEED):
        # Derive unique seed using hash of parameters
        seed_offset = hash(params) % (2**32 - 1)
        sim_seed = (base_seed + seed_offset) % (2**32 - 1)
        random.seed(sim_seed)
        np.random.seed(sim_seed)
        alpha_0, alpha, beta, sit, x = params
        # Construct a fresh SimulationConfif
        fresh_agents = copy.deepcopy(all_agents_obj)
        conf = SimulationConfif(
        alpha_0, alpha, beta, sit,
        g_func=custom_g,
        hh_similarity_func=cosine_similarity_on_adults,
        redmeat_similarity_func=my_redmeat_sim,
        diet_imp=False,
        prnt = False
        )


        #select_nodes = NodeSelector()
        #staticnodes = select_nodes.get_nodes_by_quantiles(fresh_agents.network, x, [ 'Beef and veal aggregated protein share','Pork aggregated protein share','Lamb and goat aggregated protein share','Poultry aggregated protein share'] )
        run_sim_collect = RunSimulationCollectData()
        # choose from dtaticnodes dict

        #dietimposer object

        results =  run_sim_collect.run_simulation(
            objSimulationConfig=conf,
            items=pro_share_list,       # or your actual list
            objAllAgents=fresh_agents
        )#optional static_nodes

        del fresh_agents, conf, run_sim_collect
        return results

    # 7) Run the simulations in parallel
    results = Parallel(n_jobs=8)(
    delayed(sim_func)(p)
    for idx, p in enumerate(param_combinations)
    )

    # Filter out any None results, in case some parameter combos returned None
    results = [res for res in results if res is not None]

    results_df = pd.DataFrame(results)
    print("Simulations complete.")
    print(results_df.head())

    results_df['quantile'] = 'NA'
    results_df['use_random'] = 'NA'

    name = f"MC_Param_sensitivity_no_Lancet_{_}.csv"
    results_df.to_csv(name)
    HH_set_size[name] = G.number_of_nodes()

    print(datetime.now())





# Get all result files
csv_files = sorted(glob.glob("MC_Param_sensitivity_no_Lancet_*.csv"))

# Assumes HH_set_size has been defined earlier in the script
emission_data = []

for file in csv_files:

    
    HH_n = HH_set_size[file]
    df = pd.read_csv(file)

    # Filter only the parameters of interest
    df = df[(df['beta'] == 5.0) & (df['alpha'] == 0.3)].copy()

    # Normalize emission by actual HH size for that run
    df['avg_emission'] = df['dynamic_final_denormalized_total_emission_from_food_total'] / HH_n
    df['HH_n'] = HH_n
    df['run'] = file

    emission_data.append(df[['alpha_0', 'avg_emission', 'run']])

# Combine all into one DataFrame
df_all_runs = pd.concat(emission_data, ignore_index=True)
df_all_runs = df_all_runs.sort_values(by='alpha_0')

# Aggregate: compute mean and std for error bars
df_agg = df_all_runs.groupby('alpha_0')['avg_emission'].agg(['mean', 'std']).reset_index()

# Plot
plt.figure(figsize=(8, 5))
plt.errorbar(
    df_agg['alpha_0'], df_agg['mean'], yerr=df_agg['std'],
    fmt='o-', color='steelblue', capsize=4, label='Mean ± Std Dev'
)

plt.xlabel('α₀', fontsize=12)
plt.ylabel('Average emission from all the households (agents)', fontsize=12)
plt.title('Effect of α₀ on Mean Household (Food-related) Emissions', fontsize=14, weight='bold')
plt.grid(True, linestyle='--', alpha=0.6)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.legend()
plt.tight_layout()
plt.savefig('montecarlosensitivity.png')

