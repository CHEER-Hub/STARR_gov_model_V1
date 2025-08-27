#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 24 23:11:20 2024

@author: Jingya Wang
"""

import scipy.io
from main_s_gini_HPC_starr import main_s_gini_HPC
'''
main_s_gini_HPC_starr: Fixed efficient
main_s_gini_HPC_nodmg1: Fixed uniform
main_s_gini_HPC_nodmg2: Fixed population-based
main_s_gini_HPC: Event-triggered efficient
main_s_gini_HPC_d2: Event-triggered damage-based
'''
import numpy as np
import pickle
import os
import pandas as pd

# Load the data
dd = '/work/disasters/jw/STARR/Inputs_Data/' # TODO customize this directory
data_100_scenarios = scipy.io.loadmat(dd + 'Data_100_scenarios.mat')
scenarios = np.squeeze(data_100_scenarios['scenarios100'].reshape(-1,1))

FLAG_acq_option = 1
FLAG_ret_option = 1
FLAG_selfret_option = 0
FLAG_ins_option = 0
PARA_gov_benefitratio_forlow = 1
FLAG_load_gov_opt = 0
FLAG_DCM = 1

'''
This budget file is needed for fixed budget designs: Fixed efficient, Fixed uniform, and Fixed population-based
Otherwise, comment it out and also remove from the arguments of the function being called
'''
budgets = pd.read_csv('/work/disasters/jw/gov_v1/analysis/real_budget_early_20250710.csv', header = None)
budgets = budgets.to_numpy()
    
    
j = 0
# Iterate over scenarios
for i in scenarios[0:2]:
    print('Scenario', i)
    budget = budgets[j]
    print(budget)
    # Call the main function and obtain the solution
    SOLUTION = main_s_gini_HPC(i, budget, FLAG_acq_option, FLAG_ret_option, FLAG_selfret_option, FLAG_ins_option, PARA_gov_benefitratio_forlow, FLAG_load_gov_opt, FLAG_DCM, 'save')
    j += 1
    # Example solution dictionary
    '''    
        'scenario_ID'
        
        'FLAG_acq_option': 0: no acquisition; 1: regular acquisition; 2: regular acquisition and special acquisition for low-income group
        
        'FLAG_ret_option': 0: no retrofit; 1: regular retrofit; 2: regular retrofit and special retrofit for low-income group
        
        'FLAG_selfret_option': 0: no self-retrofit; 1: with self-retrofit
        
        'FLAG_ins_option': 0: no insurance; 1: with insurance
        
        'PARA_gov_benefitratio_forlow': a preset benefit ratio for low-income acquisition and retrofit; \
            e.g. if a ratio of 30-years losses to home value is larger than the preset benefit ratio, the household’s building will be acquired
            
        'FLAG_load_gov_opt': 0: set government factors in code; 1: load preset government factors
        
        'FLAG_DCM': 1: use DCM; 2: use utility model
        
    '''

    # Construct filename based on solution properties
    fileName = f'Outputs/Solution{SOLUTION["FLAG_acq_option"]}{SOLUTION["FLAG_ret_option"]}' \
               f'{SOLUTION["FLAG_selfret_option"]}{SOLUTION["FLAG_ins_option"]}' \
               f'{SOLUTION["PARA_gov_benefitratio_forlow"]}{SOLUTION["FLAG_load_gov_opt"]}{SOLUTION["FLAG_DCM"]}' \
               f'_Data_Y30_scen{i}_early_20250710.npy'
   
    
   # Print the full path to verify where it's going
    print("Saving to:", os.path.abspath(fileName))
    # Serialize the dictionary and save it to a .npy file
    with open(fileName, 'wb') as f:
        pickle.dump(SOLUTION, f)
    
    
    # ##### a note about how to load
    # # Load the dictionary from the .npy file
    # with open('data_dict.npy', 'rb') as f:
    #     SOLUTION = pickle.load(f)
    
    # # Print the loaded dictionary to verify
    # for key, value in SOLUTION.items():
    #     print(f"{key}: {value}")
    # #e.g.
    # track_price = SOLUTION['track_price']
