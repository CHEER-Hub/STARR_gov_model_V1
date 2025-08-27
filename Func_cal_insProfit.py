#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 22 22:28:28 2024

@author: Jingya Wang
"""

import numpy as np



def Func_cal_insProfit(year, P_sum_avehurr_fldwd_LH, L_sum_perhurr_fldwd_LH, A, M, Pr, hurr_sim, scenarios, years, profit, g, phi, beta):

    """Calculate insurers’ profits based on current insurance prices """

    ########
    ## •Initialize parameters and variables, where (i) “g=0.1” is the reinsurers’ risk aversion coefficient, (ii) “phi=0.1” is a predefined loading factor for reinsurance, (iii) “beta=0.95” is the reinsurer's liability portion, and (iv) “e_sy” is to record loss coverage by reinsurance by different scenarios (2,000) and years (30)
    ##•	Calculate loss coverage by reinsurance by hurricanes “q_superh”
    #g = 0.1  # Reinsurers’ risk aversion coefficient
    #phi = 0.1  # ϕ is a pre-defined loading factor
    #beta = 0.95  # reinsurer’s liability is a portion
    e_sy = np.zeros((scenarios, years))  # the losses covered by the reinsurer for each scenario and year

    # losses covered by reinsurers
    q_superh = np.minimum(np.maximum(L_sum_perhurr_fldwd_LH.T - A, 0), (M - A)) # C = max(A,B) returns an array with the largest elements taken from A or B. same with min


    #act_n_scenarios = len(act_scenarios)
    
    ########
    # Simulation loop
    
    ####10.5s per run
    # for scenarios in range(2000):
    #     for varYear in range(30):
    #         temp = hurr_sim[scenarios, varYear]
    #         if temp > 1E8:
    #             h1 = int(temp % 100)
    #             h2 = int((temp % 10000 - h1) // 100)
    #             h3 = int((temp % 1E6 - h2 * 100 - h1) // 1E4)
    #             h4 = int((temp % 1E8 - h3 * 1E4 - h2 * 100 - h1) // 1E6)
    #             h5 = int((temp - h4 * 1E6 - h3 * 1E4 - h2 * 100 - h1) // 1E8)
    #             e_sy[scenarios, varYear] = q_superh[h1-1] + q_superh[h2-1] + q_superh[h3-1] + q_superh[h4-1] + q_superh[h5-1]
    #         elif temp > 1E6:
    #             h1 = int(temp % 100)
    #             h2 = int((temp % 10000 - h1) // 100)
    #             h3 = int((temp % 1E6 - h2 * 100 - h1) // 1E4)
    #             h4 = int((temp - h3 * 1E4 - h2 * 100 - h1) // 1E6)
    #             e_sy[scenarios, varYear] = q_superh[h1-1] + q_superh[h2-1] + q_superh[h3-1] + q_superh[h4-1]
    #         elif temp > 1E4:
    #             h1 = int(temp % 100)
    #             h2 = int((temp % 10000 - h1) // 100)
    #             h3 = int((temp - h2 * 100 - h1) // 1E4)
    #             e_sy[scenarios, varYear] = q_superh[h1-1] + q_superh[h2-1] + q_superh[h3-1]
    #         elif temp > 100:
    #             h1 = int(temp % 100)
    #             h2 = int((temp - h1) // 100)
    #             e_sy[scenarios, varYear] = q_superh[h1-1] + q_superh[h2-1]
    #         elif temp > 0:
    #             h1 = int(temp)
    #             e_sy[scenarios, varYear] = q_superh[h1-1]

    #### 9.5s per run
    # # Assuming hurr_sim and q_superh are defined and available
    # e_sy = np.zeros((2000, 30))
    
    # # Reshape hurr_sim for easy manipulation
    # temp = hurr_sim.reshape(-1)
    
    # # Create masks for each condition
    # mask_1E8 = temp > 1E8
    # mask_1E6 = (temp > 1E6) & (temp <= 1E8)
    # mask_1E4 = (temp > 1E4) & (temp <= 1E6)
    # mask_100 = (temp > 100) & (temp <= 1E4)
    # mask_0 = (temp > 0) & (temp <= 100)
    
    # # Helper function to extract the digits
    # def extract_digits(temp, positions):
    #     return np.array([int(temp // 10**i % 100) for i in positions])
    
    # # Compute values for each condition
    # e_sy_val = np.zeros_like(temp)
    
    # # Process mask_1E8
    # temp_1E8 = temp[mask_1E8]
    # h1 = (temp_1E8 % 100).astype(int)
    # h2 = ((temp_1E8 % 10000 - h1) // 100).astype(int)
    # h3 = ((temp_1E8 % 1E6 - h2 * 100 - h1) // 1E4).astype(int)
    # h4 = ((temp_1E8 % 1E8 - h3 * 1E4 - h2 * 100 - h1) // 1E6).astype(int)
    # h5 = ((temp_1E8 - h4 * 1E6 - h3 * 1E4 - h2 * 100 - h1) // 1E8).astype(int)
    # e_sy_val[mask_1E8] = q_superh[h1-1] + q_superh[h2-1] + q_superh[h3-1] + q_superh[h4-1] + q_superh[h5-1]
    
    # # Process mask_1E6
    # temp_1E6 = temp[mask_1E6]
    # h1 = (temp_1E6 % 100).astype(int)
    # h2 = ((temp_1E6 % 10000 - h1) // 100).astype(int)
    # h3 = ((temp_1E6 % 1E6 - h2 * 100 - h1) // 1E4).astype(int)
    # h4 = ((temp_1E6 - h3 * 1E4 - h2 * 100 - h1) // 1E6).astype(int)
    # e_sy_val[mask_1E6] = q_superh[h1-1] + q_superh[h2-1] + q_superh[h3-1] + q_superh[h4-1]
    
    # # Process mask_1E4
    # temp_1E4 = temp[mask_1E4]
    # h1 = (temp_1E4 % 100).astype(int)
    # h2 = ((temp_1E4 % 10000 - h1) // 100).astype(int)
    # h3 = ((temp_1E4 - h2 * 100 - h1) // 1E4).astype(int)
    # e_sy_val[mask_1E4] = q_superh[h1-1] + q_superh[h2-1] + q_superh[h3-1]
    
    # # Process mask_100
    # temp_100 = temp[mask_100]
    # h1 = (temp_100 % 100).astype(int)
    # h2 = ((temp_100 - h1) // 100).astype(int)
    # e_sy_val[mask_100] = q_superh[h1-1] + q_superh[h2-1]
    
    # # Process mask_0
    # temp_0 = temp[mask_0]
    # h1 = (temp_0 % 100).astype(int)
    # e_sy_val[mask_0] = q_superh[h1-1]
    
    # # Reshape e_sy_val back to the original shape
    # e_sy = e_sy_val.reshape(2000, 30)
    
    ### 0.8s per run
    # # Assuming hurr_sim and q_superh are defined and available
    # e_sy = np.zeros((2000, 30))
    
    # # Flatten hurr_sim for easier manipulation
    # temp = hurr_sim.flatten()
    
    # # Precompute the indices of each digit position we need for each range of temp values
    # def extract_digits(temp, powers):
    #     digits = []
    #     for power in powers:
    #         digit = (temp // 10**power % 100).astype(int)
    #         digits.append(digit - 1)  # subtracting 1 to make it zero-based index
    #     return np.array(digits)
    
    # # Masks
    # mask_1E8 = temp > 1E8
    # mask_1E6 = (temp > 1E6) & (temp <= 1E8)
    # mask_1E4 = (temp > 1E4) & (temp <= 1E6)
    # mask_100 = (temp > 100) & (temp <= 1E4)
    # mask_0 = (temp > 0) & (temp <= 100)
    
    # # Compute e_sy values
    # e_sy_val = np.zeros_like(temp)
    
    # # Process each mask separately
    # for mask, powers in zip([mask_1E8, mask_1E6, mask_1E4, mask_100, mask_0],
    #                         [[0, 2, 4, 6, 8], [0, 2, 4, 6], [0, 2, 4], [0, 2], [0]]):
    #     temp_subset = temp[mask]
    #     if temp_subset.size > 0:
    #         digits = extract_digits(temp_subset, powers)
    #         e_sy_val[mask] = np.sum(q_superh[digits], axis=0)
    
    # # Reshape back to the original shape
    # e_sy = e_sy_val.reshape(2000, 30)
    
    
    # ### 0.58s per run; but sometmes takes 11s, not knowing why
    # # Function to extract digits
    # def extract_digits(arr, powers):
    #     return ((arr[:, None] // powers) % 100).astype(int)
    
    # # Define the ranges
    # scenarios_range = np.arange(2000)
    # varYear_range = np.arange(30)
    
    # # Precompute the powers of 10 for digit extraction
    # powers_1E8 = 10**np.array([0, 2, 4, 6, 8])
    # powers_1E6 = 10**np.array([0, 2, 4, 6])
    # powers_1E4 = 10**np.array([0, 2, 4])
    # powers_100 = 10**np.array([0, 2])
    
    # # Iterate through the scenarios and years
    # for scenarios in scenarios_range:
    #     temp = hurr_sim[scenarios]
    #     for varYear in varYear_range:
    #         temp_val = temp[varYear]
    #         if temp_val > 1E8:
    #             h = extract_digits(np.array([temp_val]), powers_1E8)
    #             e_sy[scenarios, varYear] = np.sum(q_superh[h[0] - 1])
    #         elif temp_val > 1E6:
    #             h = extract_digits(np.array([temp_val]), powers_1E6)
    #             e_sy[scenarios, varYear] = np.sum(q_superh[h[0] - 1])
    #         elif temp_val > 1E4:
    #             h = extract_digits(np.array([temp_val]), powers_1E4)
    #             e_sy[scenarios, varYear] = np.sum(q_superh[h[0] - 1])
    #         elif temp_val > 100:
    #             h = extract_digits(np.array([temp_val]), powers_100)
    #             e_sy[scenarios, varYear] = np.sum(q_superh[h[0] - 1])
    #         elif temp_val > 0:
    #             h1 = int(temp_val)
    #             e_sy[scenarios, varYear] = q_superh[h1 - 1]
    
    # # roughly 0.5s per run; more stable
    
    # Flatten the hurr_sim array for easier processing
    hurr_sim_flat = hurr_sim.flatten()
    
    # Create masks for different ranges
    mask_1E8 = hurr_sim_flat > 1E8
    mask_1E6 = (hurr_sim_flat > 1E6) & (hurr_sim_flat <= 1E8)
    mask_1E4 = (hurr_sim_flat > 1E4) & (hurr_sim_flat <= 1E6)
    mask_100 = (hurr_sim_flat > 100) & (hurr_sim_flat <= 1E4)
    mask_0 = (hurr_sim_flat > 0) & (hurr_sim_flat <= 100)
    
    # Extract digits using broadcasting
    def extract_digits(arr, powers):
        return ((arr[:, None] // powers) % 100).astype(int)
    
    # Precompute the powers of 10 for digit extraction
    powers_1E8 = 10**np.array([0, 2, 4, 6, 8])
    powers_1E6 = 10**np.array([0, 2, 4, 6])
    powers_1E4 = 10**np.array([0, 2, 4])
    powers_100 = 10**np.array([0, 2])
    
    # Initialize e_sy_flat to store results
    e_sy_flat = np.zeros_like(hurr_sim_flat)
    
    # Process each mask
    e_sy_flat[mask_1E8] = np.sum(q_superh[extract_digits(hurr_sim_flat[mask_1E8], powers_1E8) - 1], axis=1)
    e_sy_flat[mask_1E6] = np.sum(q_superh[extract_digits(hurr_sim_flat[mask_1E6], powers_1E6) - 1], axis=1)
    e_sy_flat[mask_1E4] = np.sum(q_superh[extract_digits(hurr_sim_flat[mask_1E4], powers_1E4) - 1], axis=1)
    e_sy_flat[mask_100] = np.sum(q_superh[extract_digits(hurr_sim_flat[mask_100], powers_100) - 1], axis=1)
    e_sy_flat[mask_0] = q_superh[(hurr_sim_flat[mask_0] % 100).astype(int) - 1]
    
    # Reshape the flattened result back to the original shape
    e_sy = e_sy_flat.reshape(hurr_sim.shape)



    # Calculate Eloss
    Eloss = Pr.T @ q_superh * beta #sum_h (e^h*p^h*beta); Pr: 97*1, q_h: 97*1 => 1*1

    # Reinsurance covered losses
    rsy_2nd = e_sy * Eloss / (M - A) #e_s,y/(M-A) * sum_h (e^h*p^h*beta); 2000*30

    # Calculate TT
    TT = beta * e_sy - rsy_2nd #reinsurer’s covered losses minus the reinstatement premium; use for calculating std.

    # Reinsurance premium
    b = (1 + phi) * Eloss + g * np.std(TT)
    rsy = b + rsy_2nd 

    # Calculate final profit
    F_sum_sy = profit + beta * e_sy - rsy
    rsy = np.mean(rsy)

    # Calculate insolvency
    C0 = P_sum_avehurr_fldwd_LH * 5  # initial cash position with 5 times of total premium collected
    miu_sy1 = np.zeros_like(F_sum_sy)
    temp_cap = 1
    miu_sy1[:, 0] = np.minimum(F_sum_sy[:, 0] + C0, C0 * temp_cap) #min(cash position of year 0 + profit of year 1, initial cash position)

    for i in range(1, miu_sy1.shape[1]):
        miu_sy1[:, i] = np.minimum(miu_sy1[:, i - 1] + F_sum_sy[:, i], C0 * temp_cap)  # miu records the cash postion at the end of year i

    count_new = 0
    for i in range(scenarios):
        for j in range(miu_sy1.shape[1]):
            if miu_sy1[i, j] < 0: # run out of money at the end of year j
                miu_sy1[i, j + 1:] = 0 # cash position for years after set to 0
                count_new += 1 # count number of scenarios involvents
                break

    insolventRatio = count_new / (scenarios * years)
    F_sum_sy = F_sum_sy * (miu_sy1 != 0) # update profit based on cash position != 0
    F_sum_sy = np.mean(F_sum_sy) # average over scenarios*years
    
    #if insolventRatio > 0.1/years:
     #   F_sum_sy = 0
    
    insol_year = np.sum(miu_sy1[:, year-1] < 0)/scenarios

    return F_sum_sy, insolventRatio, rsy, insol_year



