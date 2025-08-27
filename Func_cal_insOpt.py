#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 22 22:18:21 2024

@author: Jingya Wang
"""

import numpy as np
from Func_cal_insProfit import Func_cal_insProfit


def Func_cal_insOpt(year, temp_P_sum_avehurr_fldwd_LH, temp_L_sum_perhurr_fldwd_LH, temp_B_sum_perhurr_fldwd_LH, Pr, hurr_sim, n_scenarios, years, insurance_price_LB, g, phi, beta, precise_str):
    """ Calculate insurers’ profits, and optimal reinsurance strategies based on current insurance prices """

########
##  Initialize parameters and variables for insurers’ profits calculation
    P_sum_avehurr_fldwd_LH = temp_P_sum_avehurr_fldwd_LH.copy()  # premium 
    L_sum_perhurr_fldwd_LH = temp_L_sum_perhurr_fldwd_LH.copy()  # loss per hurricane, 1*97
    B_sum_perhurr_fldwd_LH = temp_B_sum_perhurr_fldwd_LH.copy()  # deductible per hurricane, 97*1

    profit_0 = P_sum_avehurr_fldwd_LH - (insurance_price_LB - 1) * np.dot(Pr.T, (L_sum_perhurr_fldwd_LH - B_sum_perhurr_fldwd_LH).T)
    profit = profit_0 * np.ones((n_scenarios, years))  # Initialize scenarios and years profit


########
##  Update insurers’ profits based on hurricane occurrence in different scenarios and years
    # for scenarios in range(2000):
    #     for varYear in range(30):
    #         temp = hurr_sim[scenarios, varYear]  # Retrieve hurricanes happened in current scenario and year.
    #         if temp > 1E8:  # 5 hurricanes happen
    #             h1 = int(temp % 100)
    #             h2 = int((temp % 10000 - h1) // 100)
    #             h3 = int((temp % 1E6 - h2 * 100 - h1) // 1E4)
    #             h4 = int((temp % 1E8 - h3 * 1E4 - h2 * 100 - h1) // 1E6)
    #             h5 = int((temp - h4 * 1E6 - h3 * 1E4 - h2 * 100 - h1) // 1E8)
    #             profit[scenarios, varYear] = profit_0 + B_sum_perhurr_fldwd_LH[h1-1] + B_sum_perhurr_fldwd_LH[h2-1] + B_sum_perhurr_fldwd_LH[h3-1] + B_sum_perhurr_fldwd_LH[h4-1] + B_sum_perhurr_fldwd_LH[h5-1] - L_sum_perhurr_fldwd_LH[h1-1] - L_sum_perhurr_fldwd_LH[h2-1] - L_sum_perhurr_fldwd_LH[h3-1] - L_sum_perhurr_fldwd_LH[h4-1] - L_sum_perhurr_fldwd_LH[h5-1]
    #         elif temp > 1E6:  # 4 hurricanes happen
    #             h1 = int(temp % 100)
    #             h2 = int((temp % 10000 - h1) // 100)
    #             h3 = int((temp % 1E6 - h2 * 100 - h1) // 1E4)
    #             h4 = int((temp - h3 * 1E4 - h2 * 100 - h1) // 1E6)
    #             profit[scenarios, varYear] = profit_0 + B_sum_perhurr_fldwd_LH[h1-1] + B_sum_perhurr_fldwd_LH[h2-1] + B_sum_perhurr_fldwd_LH[h3-1] + B_sum_perhurr_fldwd_LH[h4-1] - L_sum_perhurr_fldwd_LH[h1-1] - L_sum_perhurr_fldwd_LH[h2-1] - L_sum_perhurr_fldwd_LH[h3-1] - L_sum_perhurr_fldwd_LH[h4-1]
    #         elif temp > 1E4:  # 3 hurricanes happen
    #             h1 = int(temp % 100)
    #             h2 = int((temp % 10000 - h1) // 100)
    #             h3 = int((temp - h2 * 100 - h1) // 1E4)
    #             profit[scenarios, varYear] = profit_0 + B_sum_perhurr_fldwd_LH[h1-1] + B_sum_perhurr_fldwd_LH[h2-1] + B_sum_perhurr_fldwd_LH[h3-1] - L_sum_perhurr_fldwd_LH[h1-1] - L_sum_perhurr_fldwd_LH[h2-1] - L_sum_perhurr_fldwd_LH[h3-1]
    #         elif temp > 100:  # 2 hurricanes happen
    #             h1 = int(temp % 100)
    #             h2 = int((temp - h1) // 100)
    #             profit[scenarios, varYear] = profit_0 + B_sum_perhurr_fldwd_LH[h1-1] + B_sum_perhurr_fldwd_LH[h2-1] - L_sum_perhurr_fldwd_LH[h1-1] - L_sum_perhurr_fldwd_LH[h2-1]
    #         elif temp > 0:  # 1 hurricane happens
    #             h1 = int(temp)
    #             profit[scenarios, varYear] = profit_0 + B_sum_perhurr_fldwd_LH[h1-1] - L_sum_perhurr_fldwd_LH[h1-1]



    # # Create masks for each condition
    # mask_5 = hurr_sim > 1E8
    # mask_4 = (hurr_sim > 1E6) & (hurr_sim <= 1E8)
    # mask_3 = (hurr_sim > 1E4) & (hurr_sim <= 1E6)
    # mask_2 = (hurr_sim > 100) & (hurr_sim <= 1E4)
    # mask_1 = (hurr_sim > 0) & (hurr_sim <= 100)
    
    # # Vectorized function to calculate the hurricane indices
    # def extract_indices(hurr_array, num_hurricanes):
    #     hurr_array = hurr_array.astype(np.int64)  # Ensure the array is of integer type
    #     indices = np.zeros((hurr_array.size, num_hurricanes), dtype=np.int32)
    #     for k in range(num_hurricanes):
    #         indices[:, k] = hurr_array % 100
    #         hurr_array //= 100
    #     indices -= 1  # Adjust to zero-based indexing
    #     return indices
    
    # # Calculate profits
    # def calculate_profits(hurr_array, num_hurricanes, mask):
    #     indices = extract_indices(hurr_array, num_hurricanes)
    #     B_sums = np.sum(B_sum_perhurr_fldwd_LH[indices], axis=1)
    #     L_sums = np.sum(L_sum_perhurr_fldwd_LH[indices], axis=1)
    #     profit[mask] = profit_0 + B_sums - L_sums
    
    # # Apply the masks and process hurricanes
    # calculate_profits(hurr_sim[mask_5].flatten(), 5, mask_5)
    # calculate_profits(hurr_sim[mask_4].flatten(), 4, mask_4)
    # calculate_profits(hurr_sim[mask_3].flatten(), 3, mask_3)
    # calculate_profits(hurr_sim[mask_2].flatten(), 2, mask_2)
    # calculate_profits(hurr_sim[mask_1].flatten(), 1, mask_1)


    ### this is the fastest one
    def compute_profit_optimized(hurr_sim, profit_0, B_sum_perhurr_fldwd_LH, L_sum_perhurr_fldwd_LH):
        hurr_sim_flat = hurr_sim.flatten()
        
        # Create masks for different ranges
        masks = [
            hurr_sim_flat > 1E8,
            (hurr_sim_flat > 1E6) & (hurr_sim_flat <= 1E8),
            (hurr_sim_flat > 1E4) & (hurr_sim_flat <= 1E6),
            (hurr_sim_flat > 100) & (hurr_sim_flat <= 1E4),
            (hurr_sim_flat > 0) & (hurr_sim_flat <= 100)
        ]
    
        powers = [
            10**np.array([0, 2, 4, 6, 8]),
            10**np.array([0, 2, 4, 6]),
            10**np.array([0, 2, 4]),
            10**np.array([0, 2]),
            10**np.array([0])
        ]
        
        def extract_digits(arr, pwr):
            return ((arr[:, None] // pwr) % 100).astype(int)
    
        # Initialize profit_flat to store results
        profit_flat = np.full(hurr_sim_flat.shape, profit_0)
    
        # Process each mask
        for mask, power in zip(masks, powers):
            if np.any(mask):
                hurricanes = extract_digits(hurr_sim_flat[mask], power)
                B_sum = np.sum(B_sum_perhurr_fldwd_LH[hurricanes - 1], axis=1)
                L_sum = np.sum(L_sum_perhurr_fldwd_LH[hurricanes - 1], axis=1)
                profit_flat[mask] += B_sum - L_sum
        
        # Reshape the flattened result back to the original shape
        profit = profit_flat.reshape(hurr_sim.shape)
        
        return profit
    
    # Example usage:
    # Assuming hurr_sim, profit_0, B_sum_perhurr_fldwd_LH, and L_sum_perhurr_fldwd_LH are defined
    profit = compute_profit_optimized(hurr_sim, profit_0, B_sum_perhurr_fldwd_LH, L_sum_perhurr_fldwd_LH)



########
## •Set a series of pairs (A, M) for reinsurance optimization. Note that “precise” means 20*30 combinations of A and M, and “not precise” means 10*15 combinations of A and M
##•	Calculate optimal A and M for reinsurance and insurers’ profits by calling “Func_cal_insProfit.m”
##•	Record and return insurers’ profits, insolvent rates, reinsurance premiums, and optimal A and M
    FAST_profitFunc = lambda x: Func_cal_insProfit(year, P_sum_avehurr_fldwd_LH, L_sum_perhurr_fldwd_LH, x[0], x[1], Pr, hurr_sim, n_scenarios, years, profit, g, phi, beta)

    #if 1:  # if 1 means always true. assume can change to 0 here.
    if precise_str == 'precise':
        # precise: larger step
        A_steps = 20  # 30;20
        M_steps = 30  # 50;30
    else:
        # not precise: smaller step
        A_steps = 10  # 30;
        M_steps = 15  # 50;

    AAA = np.linspace(1/A_steps, 1, A_steps) * 2e9
    MMM = np.linspace(1/M_steps, 1, M_steps) * 8e9
    F_RSMTR = 0
    for i in range(len(AAA)):  # 10, 20
        for j in range(len(MMM)):  # 15, 30
            if AAA[i] < MMM[j]:  # A must be less than M; Otherwise, ignore.
                temp_F_RSMTR, temp_insolventRatio_RSMTR, temp_rsy_RSMTR, temp_insol_year = FAST_profitFunc([AAA[i], MMM[j]])
                if (i == 0 and j == 0) or temp_F_RSMTR > F_RSMTR:  # find maximum profit and record related information.
                    F_RSMTR = temp_F_RSMTR
                    insolventRatio_RSMTR = temp_insolventRatio_RSMTR
                    rsy_RSMTR = temp_rsy_RSMTR
                    insol_year = temp_insol_year
                    A_RSMTR = AAA[i]
                    M_RSMTR = MMM[j]
                    
        
        #count_loop = A_steps * M_steps  # 10*15 or 20*30
        
    
            
    # else: not needed
    #     decrease_flag = 0
    #     delta = [2e8, 2e8]
    #     center0 = [5e8, 5e9]
    #     center_x = center0
    #     center_y = FAST_profitFunc(center_x)
    #     best_x = center_x
    #     best_y = center_y
    #     count_loop = 0
    #     while decrease_flag < 2:
    #         count_loop += 1
    #         center_x, center_y, decrease_flag, best_x, best_y = Func_newRSM_stage1(center_x, center_y, delta, FAST_profitFunc, decrease_flag, best_x, best_y)

    #     A_RSMTR = best_x[0]
    #     M_RSMTR = best_x[1]
    #     F_RSMTR, insolventRatio_RSMTR, rsy_RSMTR = FAST_profitFunc(best_x)

    return F_RSMTR, A_RSMTR, M_RSMTR, insolventRatio_RSMTR, rsy_RSMTR, insol_year
