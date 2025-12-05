import pandas as pd
import numpy as np
from typing import Tuple, List
from scipy import stats
import matplotlib.pyplot as plt

# Define the conversion factors dictionary
conversion_factors = {
    'egg': 110.0,#per kg, in the file we converted no of egges to kg
    'bread': 88.5,
    'rice': 26.9,
    'vegetables without dried': 25.0, 
    'Dried vegetables': 217.5,  
    'fish and seafood': 230.0,  
    'milk without cheese': 35.0,  
    'Cheese': 225.0,             
    'Beef and veal': 200.0,
    'Pork': 160.0,
    'Lamb and goat': 200.0,
    'Poultry': 175.0,
    'Beef and veal aggregated': 200.0,
    'Pork aggregated': 160.0,
    'Lamb and goat aggregated': 200.0,
    'Poultry aggregated': 175.0
 }

protein_share_dict = {'HQ01147': 'egg',
 'HQ01113': 'bread',
 'HQ01111': 'rice',
 'HQ0117-': 'vegetables without dried',
 'HQ0117': 'vegetables',
 'HQ01173': 'Dried vegetables',
 'HQ0113': 'fish and seafood',
 'HQ0114-': 'milk without cheese',
 'HQ0114': 'milk',
 'HQ01145': 'Cheese',
 'HQ01121': 'Beef and veal',
 'HQ01122': 'Pork',
 'HQ01123': 'Lamb and goat',
 'HQ01124': 'Poultry',
 'HQ01125': 'Other meats',
 'HQ01126': 'Edible offal',
 'HQ01127': 'Dried meats',
 'HQ01128': 'Other meat preparations',
 'HQ01121+': 'Beef and veal aggregated',
 'HQ01122+': 'Pork aggregated',
 'HQ01123+': 'Lamb and goat aggregated',
 'HQ01124+': 'Poultry aggregated'
 }
new_items = {
    'HA04': 'id of household',
    'COUNTRY': 'country',
    'HA09': 'degree of urbanization',
    'HA10': 'sample weight',
    'EUR_HH032': 'imputed rent',
    'EUR_HH099': 'monetary income',
    'EUR_HE00': 'total consumption expenditure',
    'HB05': 'household size',
    'HB056': 'number of adults in the household (age 25-64)',
    'HB057': 'number of retired persons',
    'HB074': 'household type',
    'HC24' : 'occupation'
}
emission_factors = {'egg': 4.67,
 'bread': 1.57,
 'rice':  4.45,
 'vegetables without dried': 2.0, 
 'Dried vegetables': 1.79,
 'fish and seafood': 10.6,
 'milk without cheese': 2,
 'Cheese': 18.3,
 'Beef and veal aggregated': 35.4,
 'Pork aggregated': 10.1,
 'Lamb and goat aggregated': 35.2,
 'Poultry aggregated': 6.2}

final_vars = ['final_denormalized_total_emission_from_food_change_mean', 'final_total_spending_on_food_change_mean', 'final_redmeat_protein_share_change_mean']
colA = ['alpha_0', 'alpha', 'beta', 'sit', 'initial_total_emission_from_food_mean', 'initial_denormalized_total_emission_from_food_mean', 'initial_total_spending_on_food_mean', 'initial_redmeat_mean', 'initial_redmeat_protein_share_mean', 'final_total_emission_from_food_mean', 'final_denormalized_total_emission_from_food_mean', 'final_total_spending_on_food_mean', 'final_redmeat_mean', 'final_redmeat_protein_share_mean']
subset_cols = ['final_total_emission_from_food_var', 'final_total_emission_from_food_change_var', 'final_denormalized_total_emission_from_food_var', 'final_denormalized_total_emission_from_food_change_var', 'final_total_spending_on_food_var', 'final_total_spending_on_food_change_var', 'final_redmeat_var', 'final_redmeat_change_var', 'final_redmeat_protein_share_var', 'final_redmeat_protein_share_change_var']
basic_protein_cols = ['egg', 'bread', 'rice', 'vegetables without dried', 'Dried vegetables', 'fish and seafood', 'milk without cheese', 'Cheese', 'Beef and veal aggregated', 'Pork aggregated', 'Lamb and goat aggregated', 'Poultry aggregated']
emission_by_quant = ['egg', 'bread', 'rice', 'vegetables without dried', 'Dried vegetables', 'fish and seafood', 'milk without cheese', 'Cheese', 'Beef and veal aggregated', 'Pork aggregated', 'Lamb and goat aggregated', 'Poultry aggregated']
emission_list = ['egg_emission', 'bread_emission', 'rice_emission', 'vegetables without dried_emission', 'Dried vegetables_emission', 'fish and seafood_emission', 'milk without cheese_emission', 'Cheese_emission', 'Beef and veal aggregated_emission', 'Pork aggregated_emission', 'Lamb and goat aggregated_emission', 'Poultry aggregated_emission']
empty = []
empty_code = []
empty_code_price = []
env_features = ['Package holidays', 'redmeat', 'fuel', 'pet']
items = ['egg protein share', 'bread protein share', 'rice protein share', 'vegetables without dried protein share', 'Dried vegetables protein share', 'fish and seafood protein share', 'milk without cheese protein share', 'Cheese protein share', 'Beef and veal aggregated protein share', 'Pork aggregated protein share', 'Lamb and goat aggregated protein share', 'Poultry aggregated protein share']
list1 = ['egg', 'bread', 'rice', 'vegetables without dried', 'Dried vegetables', 'fish and seafood', 'milk without cheese', 'Cheese', 'Beef and veal aggregated', 'Pork aggregated', 'Lamb and goat aggregated', 'Poultry aggregated', 'total protein content']
list2 = ['egg', 'bread', 'rice', 'vegetables without dried', 'Dried vegetables', 'fish and seafood', 'milk without cheese', 'Cheese', 'Beef and veal aggregated', 'Pork aggregated', 'Lamb and goat aggregated', 'Poultry aggregated', 'egg protein', 'bread protein', 'rice protein', 'vegetables without dried protein', 'Dried vegetables protein', 'fish and seafood protein', 'milk without cheese protein', 'Cheese protein', 'Beef and veal aggregated protein', 'Pork aggregated protein', 'Lamb and goat aggregated protein', 'Poultry aggregated protein', 'egg protein share', 'bread protein share', 'rice protein share', 'vegetables without dried protein share', 'Dried vegetables protein share', 'fish and seafood protein share', 'milk without cheese protein share', 'Cheese protein share', 'Beef and veal aggregated protein share', 'Pork aggregated protein share', 'Lamb and goat aggregated protein share', 'Poultry aggregated protein share', 'total protein content', 'HC24', 'HA09']
list3 = ['egg protein share', 'bread protein share', 'rice protein share', 'vegetables without dried protein share', 'Dried vegetables protein share', 'fish and seafood protein share', 'milk without cheese protein share', 'Cheese protein share', 'Beef and veal aggregated protein share', 'Pork aggregated protein share', 'Lamb and goat aggregated protein share', 'Poultry aggregated protein share', 'total protein content']
list4 = ['egg', 'bread', 'rice', 'vegetables without dried', 'Dried vegetables', 'fish and seafood', 'milk without cheese', 'Cheese', 'Beef and veal aggregated', 'Pork aggregated', 'Lamb and goat aggregated', 'Poultry aggregated', 'egg protein', 'bread protein', 'rice protein', 'vegetables without dried protein', 'Dried vegetables protein', 'fish and seafood protein', 'milk without cheese protein', 'Cheese protein', 'Beef and veal aggregated protein', 'Pork aggregated protein', 'Lamb and goat aggregated protein', 'Poultry aggregated protein', 'egg protein share', 'bread protein share', 'rice protein share', 'vegetables without dried protein share', 'Dried vegetables protein share', 'fish and seafood protein share', 'milk without cheese protein share', 'Cheese protein share', 'Beef and veal aggregated protein share', 'Pork aggregated protein share', 'Lamb and goat aggregated protein share', 'Poultry aggregated protein share', 'total protein content', 'HC24', 'HA09', 'cult', 'cult1', 'income', 'Package holidays', 'redmeat', 'fuel', 'pet']
list5 = ['adults', 'egg', 'bread', 'rice', 'vegetables without dried', 'Dried vegetables', 'fish and seafood', 'milk without cheese', 'Cheese', 'Beef and veal aggregated', 'Pork aggregated', 'Lamb and goat aggregated', 'Poultry aggregated', 'egg protein', 'bread protein', 'rice protein', 'vegetables without dried protein', 'Dried vegetables protein', 'fish and seafood protein', 'milk without cheese protein', 'Cheese protein', 'Beef and veal aggregated protein', 'Pork aggregated protein', 'Lamb and goat aggregated protein', 'Poultry aggregated protein', 'egg protein share', 'bread protein share', 'rice protein share', 'vegetables without dried protein share', 'Dried vegetables protein share', 'fish and seafood protein share', 'milk without cheese protein share', 'Cheese protein share', 'Beef and veal aggregated protein share', 'Pork aggregated protein share', 'Lamb and goat aggregated protein share', 'Poultry aggregated protein share', 'total protein content', 'HC24', 'HA09', 'cult', 'cult1', 'income', 'Package holidays', 'redmeat', 'fuel', 'pet', 'egg price', 'bread price', 'rice price', 'vegetables without dried price', 'Dried vegetables price', 'fish and seafood price', 'milk without cheese price', 'Cheese price', 'Beef and veal aggregated price', 'Pork aggregated price', 'Lamb and goat aggregated price', 'Poultry aggregated price', 'egg_total_spending', 'bread_total_spending', 'rice_total_spending', 'vegetables without dried_total_spending', 'Dried vegetables_total_spending', 'fish and seafood_total_spending', 'milk without cheese_total_spending', 'Cheese_total_spending', 'Beef and veal aggregated_total_spending', 'Pork aggregated_total_spending', 'Lamb and goat aggregated_total_spending', 'Poultry aggregated_total_spending', 'egg_emission', 'bread_emission', 'rice_emission', 'vegetables without dried_emission', 'Dried vegetables_emission', 'fish and seafood_emission', 'milk without cheese_emission', 'Cheese_emission', 'Beef and veal aggregated_emission', 'Pork aggregated_emission', 'Lamb and goat aggregated_emission', 'Poultry aggregated_emission', 'total spending on food', 'total emission from food', 'sum of protein share (check)', 'denormalized total emission from food']
lst = ['egg_total_spending', 'bread_total_spending', 'rice_total_spending', 'vegetables without dried_total_spending', 'Dried vegetables_total_spending', 'fish and seafood_total_spending', 'milk without cheese_total_spending', 'Cheese_total_spending', 'Beef and veal aggregated_total_spending', 'Pork aggregated_total_spending', 'Lamb and goat aggregated_total_spending', 'Poultry aggregated_total_spending']
prices_col = ['egg price', 'bread price', 'rice price', 'vegetables without dried price', 'Dried vegetables price', 'fish and seafood price', 'milk without cheese price', 'Cheese price', 'Beef and veal aggregated price', 'Pork aggregated price', 'Lamb and goat aggregated price', 'Poultry aggregated price']
pro_list = ['egg protein', 'bread protein', 'rice protein', 'vegetables without dried protein', 'Dried vegetables protein', 'fish and seafood protein', 'milk without cheese protein', 'Cheese protein', 'Beef and veal aggregated protein', 'Pork aggregated protein', 'Lamb and goat aggregated protein', 'Poultry aggregated protein']
pro_share_list = ['egg protein share', 'bread protein share', 'rice protein share', 'vegetables without dried protein share', 'Dried vegetables protein share', 'fish and seafood protein share', 'milk without cheese protein share', 'Cheese protein share', 'Beef and veal aggregated protein share', 'Pork aggregated protein share', 'Lamb and goat aggregated protein share', 'Poultry aggregated protein share']
protein_cols = ['HQ01147', 'HQ01113', 'HQ01111', 'HQ0117-', 'HQ01173', 'HQ0113', 'HQ0114-', 'HQ01145', 'HQ01121+', 'HQ01122+', 'HQ01123+', 'HQ01124+']
redmeat_spending_cols = ['Beef and veal aggregated_total_spending', 'Pork aggregated_total_spending', 'Lamb and goat aggregated_total_spending']
some_list = ['redmeat', 'Pork aggregated protein share', 'Beef and veal aggregated protein share', 'total spending on food', 'total emission from food', 'denormalized total emission from food']
total_quantity_cols = ['egg', 'bread', 'rice', 'vegetables without dried', 'Dried vegetables', 'fish and seafood', 'milk without cheese', 'Cheese', 'Beef and veal aggregated', 'Pork aggregated', 'Lamb and goat aggregated', 'Poultry aggregated']
total_spending_cols = ['egg_total_spending', 'bread_total_spending', 'rice_total_spending', 'vegetables without dried_total_spending', 'Dried vegetables_total_spending', 'fish and seafood_total_spending', 'milk without cheese_total_spending', 'Cheese_total_spending', 'Beef and veal aggregated_total_spending', 'Pork aggregated_total_spending', 'Lamb and goat aggregated_total_spending', 'Poultry aggregated_total_spending']
comparison_list = ['total emission from food', 'denormalized total emission from food', 'total spending on food', 'redmeat', 'redmeat protein share'] + pro_share_list
non_numeric_items = ['occupation', 'HC24']
icv = []
all_quantity = ['HQ01111', 'HQ01112', 'HQ01113', 'HQ01114', 'HQ01115', 'HQ01116', 'HQ01117', 'HQ01118', 'HQ01121', 'HQ01122', 'HQ01123', 'HQ01124', 'HQ01125', 'HQ01126', 'HQ01127', 'HQ01128', 'HQ01131', 'HQ01132', 'HQ01133', 'HQ01134', 'HQ01135', 'HQ01136', 'HQ01141', 'HQ01142', 'HQ01143', 'HQ01144', 'HQ01145', 'HQ01146', 'HQ01147', 'HQ01151', 'HQ01152', 'HQ01153', 'HQ01154', 'HQ01155', 'HQ01161', 'HQ01162', 'HQ01163', 'HQ01164', 'HQ01171', 'HQ01172', 'HQ01173', 'HQ01174', 'HQ01175', 'HQ01176', 'HQ01181', 'HQ01182', 'HQ01183', 'HQ01184', 'HQ01185', 'HQ01186', 'HQ01191', 'HQ01192', 'HQ01193', 'HQ01194', 'HQ01199', 'HQ01211', 'HQ01212', 'HQ01213', 'HQ01221', 'HQ01222', 'HQ01223', 'HQ02111', 'HQ02112', 'HQ02121', 'HQ02122', 'HQ02123', 'HQ02124', 'HQ02131', 'HQ02132', 'HQ02133', 'HQ02134']
all_spendings1 = ['EUR_HE01111', 'EUR_HE01112', 'EUR_HE01113', 'EUR_HE01114', 'EUR_HE01115', 'EUR_HE01116', 'EUR_HE01117', 'EUR_HE01118', 'EUR_HE01121', 'EUR_HE01122', 'EUR_HE01123', 'EUR_HE01124', 'EUR_HE01125', 'EUR_HE01126', 'EUR_HE01127', 'EUR_HE01128', 'EUR_HE01131', 'EUR_HE01132', 'EUR_HE01133', 'EUR_HE01134', 'EUR_HE01135', 'EUR_HE01136', 'EUR_HE01141', 'EUR_HE01142', 'EUR_HE01143', 'EUR_HE01144', 'EUR_HE01145', 'EUR_HE01146', 'EUR_HE01147', 'EUR_HE01151', 'EUR_HE01152', 'EUR_HE01153', 'EUR_HE01154', 'EUR_HE01155', 'EUR_HE01161', 'EUR_HE01162', 'EUR_HE01163', 'EUR_HE01164', 'EUR_HE01171', 'EUR_HE01172', 'EUR_HE01173', 'EUR_HE01174', 'EUR_HE01175', 'EUR_HE01176', 'EUR_HE01181', 'EUR_HE01182', 'EUR_HE01183', 'EUR_HE01184', 'EUR_HE01185', 'EUR_HE01186', 'EUR_HE01191', 'EUR_HE01192', 'EUR_HE01193', 'EUR_HE01194', 'EUR_HE01199', 'EUR_HE01211', 'EUR_HE01212', 'EUR_HE01213', 'EUR_HE01221', 'EUR_HE01222', 'EUR_HE01223', 'EUR_HE02111', 'EUR_HE02112', 'EUR_HE02121', 'EUR_HE02122', 'EUR_HE02123', 'EUR_HE02124', 'EUR_HE02131', 'EUR_HE02132', 'EUR_HE02133', 'EUR_HE02134']
invalid_price = ['HQ01114', 'HQ01115', 'HQ01117', 'HQ01118', 'HQ01128', 'HQ01133', 'HQ01135', 'HQ01136', 'HQ01146', 'HQ01162', 'HQ01163', 'HQ01164', 'HQ01172', 'HQ01173', 'HQ01175', 'HQ01176', 'HQ01182', 'HQ01184', 'HQ01186', 'HQ01191', 'HQ01192', 'HQ01193', 'HQ01194', 'HQ01199', 'HQ02132']
p_price = ['egg protein_price', 'bread protein_price', 'rice protein_price', 'vegetables without dried protein_price', 'Dried vegetables protein_price', 'fish and seafood protein_price', 'milk without cheese protein_price', 'Cheese protein_price', 'Beef and veal aggregated protein_price', 'Pork aggregated protein_price', 'Lamb and goat aggregated protein_price', 'Poultry aggregated protein_price']
commodity_shares = ['egg protein share', 'bread protein share', 'rice protein share', 'vegetables without dried protein share', 'Dried vegetables protein share', 'fish and seafood protein share', 'milk without cheese protein share', 'Cheese protein share', 'Beef and veal aggregated protein share', 'Pork aggregated protein share', 'Lamb and goat aggregated protein share', 'Poultry aggregated protein share']
obj = ['egg protein share', 'bread protein share', 'rice protein share', 'vegetables without dried protein share', 'Dried vegetables protein share', 'fish and seafood protein share', 'milk without cheese protein share', 'Cheese protein share', 'Beef and veal aggregated protein share', 'Pork aggregated protein share', 'Lamb and goat aggregated protein share', 'Poultry aggregated protein share']
non_meat_protein_share = ['egg protein share',
 'bread protein share',
 'rice protein share',
 'vegetables without dried protein share',
 'Dried vegetables protein share',
 'fish and seafood protein share',
 'milk without cheese protein share',
 'Cheese protein share']

meat_protein_share = ['Beef and veal aggregated protein share',
 'Pork aggregated protein share',
 'Lamb and goat aggregated protein share',
 'Poultry aggregated protein share']

dur_list = ['EUR_HE03110', 'EUR_HE03121', 'EUR_HE03122', 'EUR_HE03123', 'EUR_HE03131', 'EUR_HE03132', 'EUR_HE03211', 'EUR_HE03212', 'EUR_HE03213', 'EUR_HE05111', 'EUR_HE05112', 'EUR_HE05113', 'EUR_HE05119', 'EUR_HE05121', 'EUR_HE05122', 'EUR_HE05123', 'EUR_HE05201', 
 'EUR_HE05202', 'EUR_HE05203', 'EUR_HE05209', 'EUR_HE05311', 'EUR_HE05312', 'EUR_HE05313', 'EUR_HE05314', 'EUR_HE05315', 'EUR_HE05319', 'EUR_HE05321', 'EUR_HE05322', 'EUR_HE05323', 'EUR_HE05324', 'EUR_HE05329', 'EUR_HE05401', 'EUR_HE05402', 'EUR_HE05403', 
 'EUR_HE05511', 'EUR_HE05521', 'EUR_HE05522', 'EUR_HE06132', 'EUR_HE06133', 'EUR_HE06139', 'EUR_HE07111', 'EUR_HE07112', 'EUR_HE07120', 'EUR_HE07130', 'EUR_HE07140', 'EUR_HE07211', 'EUR_HE07212', 'EUR_HE07213', 'EUR_HE08201', 'EUR_HE08202', 'EUR_HE08203', 
 'EUR_HE09111', 'EUR_HE09112', 'EUR_HE09113', 'EUR_HE09119', 'EUR_HE09121', 'EUR_HE09122', 'EUR_HE09123', 'EUR_HE09131', 'EUR_HE09132', 'EUR_HE09133', 'EUR_HE09134', 'EUR_HE09141', 'EUR_HE09142', 'EUR_HE09149', 'EUR_HE09150', 'EUR_HE09211', 'EUR_HE09212', 
 'EUR_HE09213', 'EUR_HE09214', 'EUR_HE09215', 'EUR_HE09221', 'EUR_HE09222', 'EUR_HE09311', 'EUR_HE09312', 'EUR_HE09321', 'EUR_HE09322', 'EUR_HE09331', 'EUR_HE09341', 'EUR_HE09342', 'EUR_HE09511', 'EUR_HE09512', 'EUR_HE09513', 'EUR_HE09514', 'EUR_HE09530', 
 'EUR_HE09541', 'EUR_HE09549', 'EUR_HE12121', 'EUR_HE12131', 'EUR_HE12311', 'EUR_HE12312', 'EUR_HE12321', 'EUR_HE12322']

ndur_list = ['EUR_HE01111', 'EUR_HE01112', 'EUR_HE01113', 'EUR_HE01114', 'EUR_HE01115', 'EUR_HE01116', 'EUR_HE01117', 'EUR_HE01118', 'EUR_HE01121', 'EUR_HE01122', 'EUR_HE01123', 'EUR_HE01124', 'EUR_HE01125', 'EUR_HE01126', 'EUR_HE01127', 'EUR_HE01128', 'EUR_HE01131', 
 'EUR_HE01132', 'EUR_HE01133', 'EUR_HE01134', 'EUR_HE01135', 'EUR_HE01136', 'EUR_HE01141', 'EUR_HE01142', 'EUR_HE01143', 'EUR_HE01144', 'EUR_HE01145', 'EUR_HE01146', 'EUR_HE01147', 'EUR_HE01151', 'EUR_HE01152', 'EUR_HE01153', 'EUR_HE01154', 'EUR_HE01155', 
 'EUR_HE01161', 'EUR_HE01162', 'EUR_HE01163', 'EUR_HE01164', 'EUR_HE01171', 'EUR_HE01172', 'EUR_HE01173', 'EUR_HE01174', 'EUR_HE01175', 'EUR_HE01176', 'EUR_HE01181', 'EUR_HE01182', 'EUR_HE01183', 'EUR_HE01184', 'EUR_HE01185', 'EUR_HE01186', 'EUR_HE01191', 
 'EUR_HE01192', 'EUR_HE01193', 'EUR_HE01194', 'EUR_HE01199', 'EUR_HE01211', 'EUR_HE01212', 'EUR_HE01213', 'EUR_HE01221', 'EUR_HE01222', 'EUR_HE01223', 'EUR_HE02111', 'EUR_HE02112', 'EUR_HE02121', 'EUR_HE02122', 'EUR_HE02123', 'EUR_HE02124', 'EUR_HE02131', 
 'EUR_HE02132', 'EUR_HE02133', 'EUR_HE02134', 'EUR_HE02201', 'EUR_HE02202', 'EUR_HE02203', 'EUR_HE05612', 'EUR_HE06110', 'EUR_HE06121', 'EUR_HE06129', 'EUR_HE06131', 'EUR_HE07221', 'EUR_HE07222', 'EUR_HE07223', 'EUR_HE07224', 'EUR_HE09332', 'EUR_HE09521', 
 'EUR_HE09522', 'EUR_HE12132']

serv_list = ['EUR_HE03141', 'EUR_HE03142', 'EUR_HE03220', 'EUR_HE05130', 'EUR_HE05204', 'EUR_HE05330', 'EUR_HE05404', 'EUR_HE05512', 'EUR_HE05523', 'EUR_HE05611', 'EUR_HE05621', 'EUR_HE05622', 'EUR_HE05623', 'EUR_HE05629', 'EUR_HE06211', 'EUR_HE06212', 'EUR_HE06220', 'EUR_HE06231', 
 'EUR_HE06232', 'EUR_HE06239', 'EUR_HE06300', 'EUR_HE07230', 'EUR_HE07241', 'EUR_HE07242', 'EUR_HE07243', 'EUR_HE07311', 'EUR_HE07312', 'EUR_HE07321', 'EUR_HE07322', 'EUR_HE07331', 'EUR_HE07332', 'EUR_HE07341', 'EUR_HE07342', 'EUR_HE07350', 'EUR_HE07361', 'EUR_HE07362', 
 'EUR_HE07369', 'EUR_HE08101', 'EUR_HE08109', 'EUR_HE08204', 'EUR_HE08301', 'EUR_HE08302', 'EUR_HE08303', 'EUR_HE08304', 'EUR_HE08305', 'EUR_HE09230', 'EUR_HE09323', 'EUR_HE09350', 'EUR_HE09411', 'EUR_HE09412', 'EUR_HE09421', 'EUR_HE09422', 'EUR_HE09423', 'EUR_HE09424', 
 'EUR_HE09425', 'EUR_HE09429', 'EUR_HE09601', 'EUR_HE09602', 'EUR_HE10101', 'EUR_HE10102', 'EUR_HE10200', 'EUR_HE10300', 'EUR_HE10400', 'EUR_HE11111', 'EUR_HE11112', 'EUR_HE11120', 'EUR_HE11201', 'EUR_HE11202', 'EUR_HE11203', 'EUR_HE12111', 'EUR_HE12112', 'EUR_HE12113', 
 'EUR_HE12122', 'EUR_HE12313', 'EUR_HE12323', 'EUR_HE12329', 'EUR_HE12401', 'EUR_HE12402', 'EUR_HE12403', 'EUR_HE12404', 'EUR_HE12520', 'EUR_HE12531', 'EUR_HE12532', 'EUR_HE12541', 'EUR_HE12542', 'EUR_HE12550', 'EUR_HE12621', 'EUR_HE12622', 'EUR_HE12701', 'EUR_HE12702', 'EUR_HE12703', 'EUR_HE12704']


cols_to_normalize = ['EUR_HE00', 'EUR_HE01', 'EUR_HE011', 'EUR_HE0111', 'EUR_HE01111', 'EUR_HE01112', 'EUR_HE01113', 'EUR_HE01114', 'EUR_HE01115', 'EUR_HE01116', 'EUR_HE01117', 'EUR_HE01118', 'EUR_HE0112', 'EUR_HE01121', 'EUR_HE01122', 'EUR_HE01123', 'EUR_HE01124', 'EUR_HE01125', 'EUR_HE01126', 
                     'EUR_HE01127', 'EUR_HE01128', 'EUR_HE0113', 'EUR_HE01131', 'EUR_HE01132', 'EUR_HE01133', 'EUR_HE01134', 'EUR_HE01135', 'EUR_HE01136', 'EUR_HE0114', 'EUR_HE01141', 'EUR_HE01142', 'EUR_HE01143', 'EUR_HE01144', 'EUR_HE01145', 'EUR_HE01146', 'EUR_HE01147', 'EUR_HE0115', 
                     'EUR_HE01151', 'EUR_HE01152', 'EUR_HE01153', 'EUR_HE01154', 'EUR_HE01155', 'EUR_HE0116', 'EUR_HE01161', 'EUR_HE01162', 'EUR_HE01163', 'EUR_HE01164', 'EUR_HE0117', 'EUR_HE01171', 'EUR_HE01172', 'EUR_HE01173', 'EUR_HE01174', 'EUR_HE01175', 'EUR_HE01176', 'EUR_HE0118', 'EUR_HE01181', 
                     'EUR_HE01182', 'EUR_HE01183', 'EUR_HE01184', 'EUR_HE01185', 'EUR_HE01186', 'EUR_HE0119', 'EUR_HE01191', 'EUR_HE01192', 'EUR_HE01193', 'EUR_HE01194', 'EUR_HE01199', 'EUR_HE012', 'EUR_HE0121', 'EUR_HE01211', 'EUR_HE01212', 'EUR_HE01213', 'EUR_HE0122', 'EUR_HE01221', 'EUR_HE01222', 
                     'EUR_HE01223', 'EUR_HE02', 'EUR_HE021', 'EUR_HE0211', 'EUR_HE02111', 'EUR_HE02112', 'EUR_HE0212', 'EUR_HE02121', 'EUR_HE02122', 'EUR_HE02123', 'EUR_HE02124', 'EUR_HE0213', 'EUR_HE02131', 'EUR_HE02132', 'EUR_HE02133', 'EUR_HE02134', 'EUR_HE022', 'EUR_HE0220', 'EUR_HE02201', 
                     'EUR_HE02202', 'EUR_HE02203', 'EUR_HE023', 'EUR_HE03', 'EUR_HE031', 'EUR_HE0311', 'EUR_HE03110', 'EUR_HE0312', 'EUR_HE03121', 'EUR_HE03122', 'EUR_HE03123', 'EUR_HE0313', 'EUR_HE03131', 'EUR_HE03132', 'EUR_HE0314', 'EUR_HE03141', 'EUR_HE03142', 'EUR_HE032', 'EUR_HE0321', 'EUR_HE03211', 
                     'EUR_HE03212', 'EUR_HE03213', 'EUR_HE0322', 'EUR_HE03220', 'EUR_HE04', 'EUR_HE041', 'EUR_HE0411', 'EUR_HE04110', 'EUR_HE0412', 'EUR_HE04121', 'EUR_HE04122', 'EUR_HE042', 'EUR_HE0421', 'EUR_HE04210', 'EUR_HE0422', 'EUR_HE04220', 'EUR_HE043', 'EUR_HE0431', 'EUR_HE04310', 'EUR_HE0432', 
                     'EUR_HE04321', 'EUR_HE04322', 'EUR_HE04323', 'EUR_HE04324', 'EUR_HE04325', 'EUR_HE04329', 'EUR_HE044', 'EUR_HE0441', 'EUR_HE04410', 'EUR_HE0442', 'EUR_HE04420', 'EUR_HE0443', 'EUR_HE04430', 'EUR_HE0444', 'EUR_HE04441', 'EUR_HE04442', 'EUR_HE04449', 'EUR_HE045', 'EUR_HE0451', 'EUR_HE04510', 
                     'EUR_HE0452', 'EUR_HE04521', 'EUR_HE04522', 'EUR_HE0453', 'EUR_HE04530', 'EUR_HE0454', 'EUR_HE04541', 'EUR_HE04549', 'EUR_HE0455', 'EUR_HE04550', 'EUR_HE05', 'EUR_HE051', 'EUR_HE0511', 'EUR_HE05111', 'EUR_HE05112', 'EUR_HE05113', 'EUR_HE05119', 'EUR_HE0512', 'EUR_HE05121', 'EUR_HE05122',
                     'EUR_HE05123', 'EUR_HE0513', 'EUR_HE05130', 'EUR_HE052', 'EUR_HE0520', 'EUR_HE05201', 'EUR_HE05202', 'EUR_HE05203', 'EUR_HE05204', 'EUR_HE05209', 'EUR_HE053', 'EUR_HE0531', 'EUR_HE05311', 'EUR_HE05312', 'EUR_HE05313', 'EUR_HE05314', 'EUR_HE05315', 'EUR_HE05319', 'EUR_HE0532', 'EUR_HE05321', 
                     'EUR_HE05322', 'EUR_HE05323', 'EUR_HE05324', 'EUR_HE05329', 'EUR_HE0533', 'EUR_HE05330', 'EUR_HE054', 'EUR_HE0540', 'EUR_HE05401', 'EUR_HE05402', 'EUR_HE05403', 'EUR_HE05404', 'EUR_HE055', 'EUR_HE0551', 'EUR_HE05511', 'EUR_HE05512', 'EUR_HE0552', 'EUR_HE05521', 'EUR_HE05522', 'EUR_HE05523', 
                     'EUR_HE056', 'EUR_HE0561', 'EUR_HE05611', 'EUR_HE05612', 'EUR_HE0562', 'EUR_HE05621', 'EUR_HE05622', 'EUR_HE05623', 'EUR_HE05629', 'EUR_HE06', 'EUR_HE061', 'EUR_HE0611', 'EUR_HE06110', 'EUR_HE0612', 'EUR_HE06121', 'EUR_HE06129', 'EUR_HE0613', 'EUR_HE06131', 'EUR_HE06132', 'EUR_HE06133', 
                     'EUR_HE06139', 'EUR_HE062', 'EUR_HE0621', 'EUR_HE06211', 'EUR_HE06212', 'EUR_HE0622', 'EUR_HE06220', 'EUR_HE0623', 'EUR_HE06231', 'EUR_HE06232', 'EUR_HE06239', 'EUR_HE063', 'EUR_HE0630', 'EUR_HE06300', 'EUR_HE07', 'EUR_HE071', 'EUR_HE0711', 'EUR_HE07111', 'EUR_HE07112', 'EUR_HE0712', 
                     'EUR_HE07120', 'EUR_HE0713', 'EUR_HE07130', 'EUR_HE0714', 'EUR_HE07140', 'EUR_HE072', 'EUR_HE0721', 'EUR_HE07211', 'EUR_HE07212', 'EUR_HE07213', 'EUR_HE0722', 'EUR_HE07221', 'EUR_HE07222', 'EUR_HE07223', 'EUR_HE07224', 'EUR_HE0723', 'EUR_HE07230', 'EUR_HE0724', 'EUR_HE07241', 
                     'EUR_HE07242', 'EUR_HE07243', 'EUR_HE073', 'EUR_HE0731', 'EUR_HE07311', 'EUR_HE07312', 'EUR_HE0732', 'EUR_HE07321', 'EUR_HE07322', 'EUR_HE0733', 'EUR_HE07331', 'EUR_HE07332', 'EUR_HE0734', 'EUR_HE07341', 'EUR_HE07342', 'EUR_HE0735', 'EUR_HE07350', 'EUR_HE0736', 'EUR_HE07361', 'EUR_HE07362', 
                     'EUR_HE07369', 'EUR_HE08', 'EUR_HE081', 'EUR_HE0810', 'EUR_HE08101', 'EUR_HE08109', 'EUR_HE082', 'EUR_HE0820', 'EUR_HE08201', 'EUR_HE08202', 'EUR_HE08203', 'EUR_HE08204', 'EUR_HE083', 'EUR_HE0830', 'EUR_HE08301', 'EUR_HE08302', 'EUR_HE08303', 'EUR_HE08304', 'EUR_HE08305', 'EUR_HE09', 
                     'EUR_HE091', 'EUR_HE0911', 'EUR_HE09111', 'EUR_HE09112', 'EUR_HE09113', 'EUR_HE09119', 'EUR_HE0912', 'EUR_HE09121', 'EUR_HE09122', 'EUR_HE09123', 'EUR_HE0913', 'EUR_HE09131', 'EUR_HE09132', 'EUR_HE09133', 'EUR_HE09134', 'EUR_HE0914', 'EUR_HE09141', 'EUR_HE09142', 'EUR_HE09149', 'EUR_HE0915', 
                     'EUR_HE09150', 'EUR_HE092', 'EUR_HE0921', 'EUR_HE09211', 'EUR_HE09212', 'EUR_HE09213', 'EUR_HE09214', 'EUR_HE09215', 'EUR_HE0922', 'EUR_HE09221', 'EUR_HE09222', 'EUR_HE0923', 'EUR_HE09230', 'EUR_HE093', 'EUR_HE0931', 'EUR_HE09311', 'EUR_HE09312', 'EUR_HE0932', 'EUR_HE09321', 'EUR_HE09322', 
                     'EUR_HE09323', 'EUR_HE0933', 'EUR_HE09331', 'EUR_HE09332', 'EUR_HE0934', 'EUR_HE09341', 'EUR_HE09342', 'EUR_HE0935', 'EUR_HE09350', 'EUR_HE094', 'EUR_HE0941', 'EUR_HE09411', 'EUR_HE09412', 'EUR_HE0942', 'EUR_HE09421', 'EUR_HE09422', 'EUR_HE09423', 'EUR_HE09424', 'EUR_HE09425', 'EUR_HE09429', 
                     'EUR_HE0943', 'EUR_HE095', 'EUR_HE0951', 'EUR_HE09511', 'EUR_HE09512', 'EUR_HE09513', 'EUR_HE09514', 'EUR_HE0952', 'EUR_HE09521', 'EUR_HE09522', 'EUR_HE0953', 'EUR_HE09530', 'EUR_HE0954', 'EUR_HE09541', 'EUR_HE09549', 'EUR_HE096', 'EUR_HE0960', 'EUR_HE09601', 'EUR_HE09602', 'EUR_HE10', 'EUR_HE101', 
                     'EUR_HE1010', 'EUR_HE10101', 'EUR_HE10102', 'EUR_HE102', 'EUR_HE1020', 'EUR_HE10200', 'EUR_HE103', 'EUR_HE1030', 'EUR_HE10300', 'EUR_HE104', 'EUR_HE1040', 'EUR_HE10400', 'EUR_HE105', 'EUR_HE1050', 'EUR_HE10500', 'EUR_HE11', 'EUR_HE111', 'EUR_HE1111', 'EUR_HE11111', 'EUR_HE11112', 'EUR_HE1112', 
                     'EUR_HE11120', 'EUR_HE112', 'EUR_HE1120', 'EUR_HE11201', 'EUR_HE11202', 'EUR_HE11203', 'EUR_HE12', 'EUR_HE121', 'EUR_HE1211', 'EUR_HE12111', 'EUR_HE12112', 'EUR_HE12113', 'EUR_HE1212', 'EUR_HE12121', 'EUR_HE12122', 'EUR_HE1213', 'EUR_HE12131', 'EUR_HE12132', 'EUR_HE122', 'EUR_HE123', 'EUR_HE1231', 
                     'EUR_HE12311', 'EUR_HE12312', 'EUR_HE12313', 'EUR_HE1232', 'EUR_HE12321', 'EUR_HE12322', 'EUR_HE12323', 'EUR_HE12329', 'EUR_HE124', 'EUR_HE1240', 'EUR_HE12401', 'EUR_HE12402', 'EUR_HE12403', 'EUR_HE12404', 'EUR_HE125', 'EUR_HE1252', 'EUR_HE12520', 'EUR_HE1253', 'EUR_HE12531', 'EUR_HE12532', 
                     'EUR_HE1254', 'EUR_HE12541', 'EUR_HE12542', 'EUR_HE1255', 'EUR_HE12550', 'EUR_HE126', 'EUR_HE1262', 'EUR_HE12621', 'EUR_HE12622', 'EUR_HE127', 'EUR_HE1270', 'EUR_HE12701', 'EUR_HE12702', 'EUR_HE12703', 'EUR_HE12704', 'HQ0111', 'HQ01111', 'HQ01112', 'HQ01113', 'HQ01114', 'HQ01115', 'HQ01116', 
                     'HQ01117', 'HQ01118', 'HQ0112', 'HQ01121', 'HQ01122', 'HQ01123', 'HQ01124', 'HQ01125', 'HQ01126', 'HQ01127', 'HQ01128', 'HQ0113', 'HQ01131', 'HQ01132', 'HQ01133', 'HQ01134', 'HQ01135', 'HQ01136', 'HQ0114', 'HQ01141', 'HQ01142', 'HQ01143', 'HQ01144', 'HQ01145', 'HQ01146', 'HQ01147', 'HQ0115', 
                     'HQ01151', 'HQ01152', 'HQ01153', 'HQ01154', 'HQ01155', 'HQ0116', 'HQ01161', 'HQ01162', 'HQ01163', 'HQ01164', 'HQ0117', 'HQ01171', 'HQ01172', 'HQ01173', 'HQ01174', 'HQ01175', 'HQ01176', 'HQ0118', 'HQ01181', 'HQ01182', 'HQ01183', 'HQ01184', 'HQ01185', 'HQ01186', 'HQ0119', 'HQ01191', 'HQ01192', 
                     'HQ01193', 'HQ01194', 'HQ01199', 'HQ012', 'HQ0121', 'HQ01211', 'HQ01212', 'HQ01213', 'HQ0122', 'HQ01221', 'HQ01222', 'HQ01223', 'HQ021', 'HQ0211', 'HQ02111', 'HQ02112', 'HQ0212', 'HQ02121', 'HQ02122', 'HQ02123', 'HQ02124', 'HQ0213', 'HQ02131', 'HQ02132', 'HQ02133', 'HQ02134', 'EUR_HH099']


list5point5 = ['adults', 'egg', 'bread', 'rice', 'vegetables without dried', 'Dried vegetables', 'fish and seafood', 'milk without cheese', 'Cheese', 'Beef and veal aggregated', 'Pork aggregated', 'Lamb and goat aggregated', 'Poultry aggregated', 'egg protein', 'bread protein', 'rice protein', 
               'vegetables without dried protein', 'Dried vegetables protein', 'fish and seafood protein', 'milk without cheese protein', 'Cheese protein', 'Beef and veal aggregated protein', 'Pork aggregated protein', 'Lamb and goat aggregated protein', 'Poultry aggregated protein', 
               'egg protein share', 'bread protein share', 'rice protein share', 'vegetables without dried protein share', 'Dried vegetables protein share', 'fish and seafood protein share', 'milk without cheese protein share', 'Cheese protein share', 'Beef and veal aggregated protein share', 
               'Pork aggregated protein share', 'Lamb and goat aggregated protein share', 'Poultry aggregated protein share', 'total protein content', 'HC24', 'HA09', 'cult', 'cult1', 'income', 'Package holidays', 'redmeat', 'fuel', 'pet', 'egg price', 'bread price', 'rice price', 'vegetables without dried price', 
               'Dried vegetables price', 'fish and seafood price', 'milk without cheese price', 'Cheese price', 'Beef and veal aggregated price', 'Pork aggregated price', 'Lamb and goat aggregated price', 'Poultry aggregated price', 'egg_total_spending', 'bread_total_spending', 'rice_total_spending', 
               'vegetables without dried_total_spending', 'Dried vegetables_total_spending', 'fish and seafood_total_spending', 'milk without cheese_total_spending', 'Cheese_total_spending', 'Beef and veal aggregated_total_spending', 'Pork aggregated_total_spending', 'Lamb and goat aggregated_total_spending', 
               'Poultry aggregated_total_spending', 'egg_emission', 'bread_emission', 'rice_emission', 'vegetables without dried_emission', 'Dried vegetables_emission', 'fish and seafood_emission', 'milk without cheese_emission', 'Cheese_emission', 'Beef and veal aggregated_emission', 'Pork aggregated_emission', 
               'Lamb and goat aggregated_emission', 'Poultry aggregated_emission', 'total spending on food', 'total emission from food', 'sum of protein share (check)', 'denormalized total emission from food', 'kmeans_cluster', 'consumption_rate', 'dur_spend_r', 'ndur_spend_r', 'serv_spend_r']

defrag_cols = {'Poultry aggregated price', 'milk without cheese_emission', 'vegetables without dried_emission', 'rice protein', 'HQ01121+', 'Dried vegetables_emission', 'egg_total_spending', 'HQ01124+', 'Cheese protein', 'Dried vegetables protein share', 'fish and seafood protein share', 
               'bread_total_spending', 'denormalized total emission from food', 'bread price', 'HQ01123+', 'Cheese_total_spending', 'Lamb and goat aggregated_total_spending', 'egg protein', 'Beef and veal aggregated protein', 'sum of protein share (check)', 'Poultry aggregated protein share', 
               'consumption_rate', 'total emission from food', 'milk without cheese price', 'Beef and veal aggregated_total_spending', 'Lamb and goat aggregated_emission', 'Poultry aggregated_emission', 'fish and seafood_emission', 'Pork aggregated_emission', 'milk without cheese protein share', 
               'bread_emission', 'income', 'fish and seafood_total_spending', 'basic_meat', 'ndur_spend_r', 'total protein content', 'Cheese_emission', 'vegetables without dried protein share', 'Beef and veal aggregated_emission', 'bread protein share', 'total spending on food', 'HQ01122+', 
               'fish and seafood price', 'redmeat', 'HQ0117-', 'serv_spend_r', 'rice protein share', 'cult', 'cult1', 'Pork aggregated protein', 'HQ0114-', 'kmeans_cluster', 'rice_emission', 'adults', 'Dried vegetables protein', 'milk without cheese_total_spending', 'Lamb and goat aggregated price', 
               'Dried vegetables_total_spending', 'vegetables without dried protein', 'egg_emission', 'milk without cheese protein', 'Pork aggregated protein share', 'fuel', 'vegetables without dried_total_spending', 'non_basic_meat', 'Pork aggregated_total_spending', 'Dried vegetables price', 
               'dur_spend_r', 'Package holidays', 'fish and seafood protein', 'Lamb and goat aggregated protein', 'rice_total_spending', 'basic_meat_spending', 'Pork aggregated price', 'non_basic_meat_spending', 'Cheese protein share', 'Cheese price', 'vegetables without dried price', 'egg protein share', 
               'Poultry aggregated protein', 'Poultry aggregated_total_spending', 'pet', 'bread protein', 'Lamb and goat aggregated protein share', 'Beef and veal aggregated price', 'egg price', 'rice price', 'Beef and veal aggregated protein share', 'dur_spend', 'ndur_spend', 'serv_spend', 
               'dur_ndur_ratio', 'serv_ndur_ratio', 'totalcon' , 'totalinc'}


attrdict = {'egg': {'protein_share': 'egg protein share',
  'protein_amount': 'egg protein',
  'quantity': 'egg',
  'spending': 'egg_total_spending',
  'emission': 'egg_emission',
  'price': 'egg price'},
 'bread': {'protein_share': 'bread protein share',
  'protein_amount': 'bread protein',
  'quantity': 'bread',
  'spending': 'bread_total_spending',
  'emission': 'bread_emission',
  'price': 'bread price'},
 'rice': {'protein_share': 'rice protein share',
  'protein_amount': 'rice protein',
  'quantity': 'rice',
  'spending': 'rice_total_spending',
  'emission': 'rice_emission',
  'price': 'rice price'},
 'vegetables without dried': {'protein_share': 'vegetables without dried protein share',
  'protein_amount': 'vegetables without dried protein',
  'quantity': 'vegetables without dried',
  'spending': 'vegetables without dried_total_spending',
  'emission': 'vegetables without dried_emission',
  'price': 'vegetables without dried price'},
 'Dried vegetables': {'protein_share': 'Dried vegetables protein share',
  'protein_amount': 'Dried vegetables protein',
  'quantity': 'Dried vegetables',
  'spending': 'Dried vegetables_total_spending',
  'emission': 'Dried vegetables_emission',
  'price': 'Dried vegetables price'},
 'fish and seafood': {'protein_share': 'fish and seafood protein share',
  'protein_amount': 'fish and seafood protein',
  'quantity': 'fish and seafood',
  'spending': 'fish and seafood_total_spending',
  'emission': 'fish and seafood_emission',
  'price': 'fish and seafood price'},
 'milk without cheese': {'protein_share': 'milk without cheese protein share',
  'protein_amount': 'milk without cheese protein',
  'quantity': 'milk without cheese',
  'spending': 'milk without cheese_total_spending',
  'emission': 'milk without cheese_emission',
  'price': 'milk without cheese price'},
 'Cheese': {'protein_share': 'Cheese protein share',
  'protein_amount': 'Cheese protein',
  'quantity': 'Cheese',
  'spending': 'Cheese_total_spending',
  'emission': 'Cheese_emission',
  'price': 'Cheese price'},
 'Beef and veal aggregated': {'protein_share': 'Beef and veal aggregated protein share',
  'protein_amount': 'Beef and veal aggregated protein',
  'quantity': 'Beef and veal aggregated',
  'spending': 'Beef and veal aggregated_total_spending',
  'emission': 'Beef and veal aggregated_emission',
  'price': 'Beef and veal aggregated price'},
 'Pork aggregated': {'protein_share': 'Pork aggregated protein share',
  'protein_amount': 'Pork aggregated protein',
  'quantity': 'Pork aggregated',
  'spending': 'Pork aggregated_total_spending',
  'emission': 'Pork aggregated_emission',
  'price': 'Pork aggregated price'},
 'Lamb and goat aggregated': {'protein_share': 'Lamb and goat aggregated protein share',
  'protein_amount': 'Lamb and goat aggregated protein',
  'quantity': 'Lamb and goat aggregated',
  'spending': 'Lamb and goat aggregated_total_spending',
  'emission': 'Lamb and goat aggregated_emission',
  'price': 'Lamb and goat aggregated price'},
 'Poultry aggregated': {'protein_share': 'Poultry aggregated protein share',
  'protein_amount': 'Poultry aggregated protein',
  'quantity': 'Poultry aggregated',
  'spending': 'Poultry aggregated_total_spending',
  'emission': 'Poultry aggregated_emission',
  'price': 'Poultry aggregated price'}}



def exclude_top_percent_outliers(df: pd.DataFrame, x: float =2) -> Tuple[pd.DataFrame, list] :
    """
    Remove rows from a DataFrame that contain values above the (100 - x)% 
    quantile in ANY numeric column.

    Parameters:
    -----------
    df : pd.DataFrame
        The DataFrame from which to remove outliers.
    x : float
        Percentage threshold. Rows exceeding the (100 - x)% quantile 
        in any numeric column are removed. Default is 5.

    Returns:
    --------
    pd.DataFrame
        A filtered DataFrame with outlier rows removed.
    """
  
    # 1) Identify numeric columns
    numeric_cols = df.select_dtypes(include=[np.number])

    # 2) Compute quantile thresholds (e.g., 95th percentile if x=5)
    thresholds = numeric_cols.quantile(1 - x/100)

    # 3) For each row, check if any column exceeds its threshold
    is_outlier_any_col = (numeric_cols > thresholds).any(axis=1)

    # 4) Filter out rows that exceed any threshold
    df_filtered = df[~is_outlier_any_col]

    return df_filtered, is_outlier_any_col





def remove_top_percent_outliers_iqr(df: pd.DataFrame, numeric_cols1: List[str], 
                                     multiplier: float = 1.5, pct: float = 1) -> Tuple[pd.DataFrame, List[int]]:
    """
    Identify outliers for all numeric columns using the IQR method. Then, 
    from the set of outlier rows, remove the top 'pct' percent (i.e., the worst 1% by default)
    based on the summed absolute deviation from acceptable bounds.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The DataFrame from which to remove outliers.
    multiplier : float, optional
        The multiplier for the IQR to define acceptable bounds (default is 1.5).
    pct : float, optional
        The percentage of the worst outliers to remove (default is 1, meaning top 1%).
    
    Returns:
    --------
    pd.DataFrame
        A filtered DataFrame with the worst outlier rows removed.
    """
    # 1) Identify numeric columns.
    numeric_cols = df[numeric_cols1]
    # 2) Compute Q1, Q3 and IQR for each numeric column.
    Q1 = numeric_cols.quantile(0.25)
    Q3 = numeric_cols.quantile(0.75)
    IQR = Q3 - Q1
    
    # 3) Define the acceptable range.
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR

    #print(lower_bound, upper_bound)
    
    # 4) Identify outlier rows where any numeric column value is outside the acceptable bounds.
    outlier_mask = ((numeric_cols < lower_bound) | (numeric_cols > upper_bound)).any(axis=1)
    #print(df.index[outlier_mask])
    # Separate non-outliers and outliers.
    df_non_outliers = df[~outlier_mask].copy()
    df_outliers = df[outlier_mask].copy()
    
    # 5) If there are no outliers, return the original DataFrame.
    num_outliers = len(df_outliers)
    if num_outliers == 0:
        return df.copy()
    
    # 6) Compute a deviation score for each outlier row.
    # For each numeric column, if the value is above the upper bound, compute (value - upper_bound).
    # If it is below the lower bound, compute (lower_bound - value). Sum the differences across columns.
    def compute_deviation(row):
        total_diff = 0
        for col in numeric_cols.columns:
            val = row[col]
            if val > upper_bound[col]:
                total_diff += (val - upper_bound[col])
            elif val < lower_bound[col]:
                total_diff += (lower_bound[col] - val)
        return total_diff

    df_outliers['deviation'] = df_outliers.apply(compute_deviation, axis=1)
    
    # 7) Determine the number of rows to remove: top pct percent (at least 1 row).
    num_to_remove = max(1, int(np.ceil(num_outliers * (pct / 100.0))))
    
    # 8) Sort the outliers by their deviation (largest first) and remove the worst ones.
    df_outliers_sorted = df_outliers.sort_values(by='deviation', ascending=False)
    indices_to_remove = df_outliers_sorted.head(num_to_remove).index
    
    df_outliers_remaining = df_outliers.drop(index=indices_to_remove)
    
    # 9) Combine the non-outlier rows with the remaining outlier rows.
    df_filtered = pd.concat([df_non_outliers, df_outliers_remaining])

    # Optionally, remove the temporary 'deviation' column.
    if 'deviation' in df_filtered.columns:
        df_filtered = df_filtered.drop(columns=['deviation'])
    
    return df_filtered, indices_to_remove




def plot_histogram(data, title):
    plt.hist(data, bins=80)
    plt.title(title)
    plt.show()

def cap_value(x, lower, upper):
    """
    Cap a numerical value within specified bounds.

    Parameters
    ----------
    x : float
        The value to be capped.
    lower : float
        The lower bound. Values below this will be set to this bound.
    upper : float
        The upper bound. Values above this will be set to this bound.

    Returns
    -------
    float
        The capped value, guaranteed to lie within [lower, upper].
    """
    return np.clip(x, lower, upper)

def process_price_column(df, qty_col, spending_col, empty, empty_code, viz_var = False):
    """
    Calculate, clean, and optionally visualize a price column derived from quantity and spending data.

    This function:
    1. Computes a new price column as spending divided by quantity.
    2. Reports counts of NaN and infinite values in spending, quantity, and price columns.
    3. If the price column contains no finite values, prompts the user for a replacement value and fills all non-finite entries.
    4. Otherwise, computes descriptive statistics (of the finite values), replaces remaining non-finite values with the median, 
    caps outliers beyond the 1st and 99th percentiles, and reports changes.
    5. Optionally plots histograms before and after capping.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the columns to process.
    qty_col : str
        Name of the column containing quantities.
    spending_col : str
        Name of the column containing total spending values.
    empty : list
        List to append the names of price columns that were entirely non-finite.
    empty_code : list
        List to append the quantity column codes corresponding to empty price columns.
    viz_var : bool, optional
        If True, display histograms of the price distribution before and after capping. Default is False.

    Returns
    -------
    None
        The function modifies `df` in place by adding and cleaning the new price column.
    """
    price_col = qty_col + ' price'
    # Create the price column
    df[price_col] = df[spending_col] / df[qty_col]
    
    # Count NaN and Inf values
    nan_spend, inf_spend = df[spending_col].isna().sum(), np.isinf(df[spending_col]).sum()
    nan_qty, inf_qty = df[qty_col].isna().sum(), np.isinf(df[qty_col]).sum()
    nan_price, inf_price = df[price_col].isna().sum(), np.isinf(df[price_col]).sum()
    
    #print('$' * 50)
    print(price_col)
    #print(f"{spending_col} (Total Spending) - NaN: {nan_spend}, ±Inf: {inf_spend}")
    #print(f"{qty_col} (Total Quantity) - NaN: {nan_qty}, ±Inf: {inf_qty}")
    #print(f"{price_col} (Price) - NaN: {nan_price}, ±Inf: {inf_price}")
    #print("Initial describe (may contain NaN/Inf):")
    #print(df[price_col].describe())

    original_col = df[price_col].copy()
    # Use only finite values
    sample = df[price_col][np.isfinite(df[price_col])]
    
    if sample.empty:
        print(f"{price_col}: the cleaned list is empty")
        empty.append(price_col)
        empty_code.append(qty_col)
        user_val = float(input(f'Enter {price_col}: '))
        df[price_col] = df[price_col].replace([np.nan, np.inf, -np.inf], user_val)
        print("After user replacement:")
        print(df[price_col].describe())
        changed = (original_col != df[price_col]).sum()
        print(f"Number of values changed: {changed} ({changed/len(original_col):.2%})")
        if viz_var:
            plot_histogram(df[price_col], f"Histogram (User Replaced): {price_col}")
    else:
        if viz_var:
            plot_histogram(sample, f"Histogram before capping: {price_col}")
        
        # Basic statistics
        mean_val = sample.mean()
        median_val = sample.median()
        mode_val = stats.mode(sample, keepdims=True).mode[0] if stats.mode(sample, keepdims=True).count[0] > 0 else None
        std_val = sample.std()
        #print(f"Mean: {mean_val}\nMedian: {median_val}\nMode: {mode_val}\nStandard Deviation: {std_val}")
        
        # Define thresholds: cap values below the 1st and above the 99th percentile
        lower_threshold = np.percentile(sample, 1)
        upper_threshold = np.percentile(sample, 99)
        #print(f"Lower threshold: {lower_threshold}\nUpper threshold: {upper_threshold}")
        
        # Replace non-finite values with the median, then apply capping
        df[price_col] = df[price_col].replace([np.nan, np.inf, -np.inf], median_val)
        df[price_col] = df[price_col].apply(lambda x: cap_value(x, lower_threshold, upper_threshold))
        #print("After capping top 1% outliers:")
        #print(df[price_col].describe())
        changed = (original_col != df[price_col]).sum()
        print(f"Number of values changed: {changed} ({changed/len(original_col):.2%})")
        if viz_var:
            plot_histogram(df[price_col], f"Histogram after capping: {price_col}")


def power_partition(a, b, N, p=3):
    """
    Partition the interval [a, b] into N subintervals using a power-law spacing.
    With p > 1, the subintervals are denser near a.

    Parameters:
        a (float): Start of the interval.
        b (float): End of the interval.
        N (int): Number of subintervals.
        p (float): Exponent to control density (default p=3).

    Returns:
        list: List of N+1 breakpoints.
    """
    return [a + (b - a) * ((i / N) ** p) for i in range(N + 1)]

