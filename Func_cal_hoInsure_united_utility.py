#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 19:13:52 2024

@author: Jingya Wang
"""
import numpy as np

def Func_cal_hoInsure_united(lambda_X, Pr, ho_perhurrLoss_fld_X, ho_perhurrLoss_wd_X,\
                              ho_aveLoss_fld_X, ho_aveLoss_wd_X, ho_prem_fld_X, ho_prem_wd_X,\
                              homevalue_X, ho_cumEXP_fld_X_year, ho_lastEXP_fld_X_year,\
                              insTail_fld_X, insTail_wd_X, index_L, insurance_price_LB, deductible, hoNum_LR, hoNum_SFHA, k_LR, k_SFHA, risk_theta, FLAG_poor):

    """ Calculate households’ insurance decisions, premiums collected by insurers, and market demand based on current insurance prices """
    
    ########
    ##•	Initialize parameters and variables. Set the fixed portion of operational cost “tao” to 0.35. Set the fixed deductible for both wind and flood insurance “deductible” to $2,500
    ##•	Calculate actual deductibles per hurricane for both wind and flood insurance “B_perhurr_wd_X” and “B_perhurr_fld_X”

    np.random.seed(99)
    hoNum_X = len(ho_prem_fld_X)  # num HO; 649012 or hoNum_SFHA
    tao = insurance_price_LB - 1  # operation factor
    #deductible = 2500  # fixed deductible

    temp = ho_perhurrLoss_fld_X - deductible  # num HO*97
    temp[temp > 0] = 0
    B_perhurr_fld_X = deductible + temp  # if loss > dec, B=dec; Otherwise, B=loss.

    temp = ho_perhurrLoss_wd_X - deductible
    temp[temp > 0] = 0
    B_perhurr_wd_X = deductible + temp
    
    # check the size of index_L
    if index_L ==[]:
        index_L = np.array([], dtype=int)
        
    ########
    ##•	Calculate actual values of losses minus deductibles for both wind and flood insurance “constx_wind” and “constx_flood”
    ##•	Set affordability parameters for low- and high-risk zones

    constx_flood = np.dot(ho_perhurrLoss_fld_X, np.squeeze(Pr)) - np.dot(B_perhurr_fld_X, np.squeeze(Pr))  # insurance pay = sum((loss - deductible) * hurricane i prob)
    constx_wind = np.dot(ho_perhurrLoss_wd_X, np.squeeze(Pr)) - np.dot(B_perhurr_wd_X, np.squeeze(Pr))  # wind, num HO*97 * 97*1 = num HO*1
    
    # Utility calculations
    Z_flood = (lambda_X + tao + 1) * constx_flood
    Z_wind = (lambda_X + tao + 1) * constx_wind
    
    Z_flood = Z_flood.reshape(-1, 1)
    Z_wind = Z_wind.reshape(-1, 1)
    # Utility function for spending (U(x)=e^(theta*x)) - risk averse
    U_ZB_flood_all = np.exp(risk_theta * (Z_flood + B_perhurr_fld_X)) @ Pr + np.exp(risk_theta * Z_flood) * (1 - np.sum(Pr))
    U_ZB_wind_all = np.exp(risk_theta * (Z_wind + B_perhurr_wd_X)) @ Pr + np.exp(risk_theta * Z_wind) * (1 - np.sum(Pr))
    
    Pr = Pr.reshape(1, -1)
    U_L_flood_all = np.exp(risk_theta * ho_perhurrLoss_fld_X) @ Pr.T + (1 - np.sum(Pr))
    U_L_wind_all = np.exp(risk_theta * ho_perhurrLoss_wd_X) @ Pr.T + (1 - np.sum(Pr))
    
    if index_L.size > 0: # only do it if index_l is not 0
        if hoNum_X == hoNum_LR:  # means calculation for low risk
            index_L = index_L[index_L <= hoNum_LR]  # 483128*1
        elif hoNum_X == hoNum_SFHA:  # means calculation for high risk
            index_L = index_L[index_L > hoNum_LR] - hoNum_LR
    if hoNum_X == hoNum_LR:  # means calculation for low risk
        coeff = k_LR # k_LR Affordability parameter
    elif hoNum_X == hoNum_SFHA:
        coeff = k_SFHA # k_SFHA Affordability parameter
        
    ########
    ##•	Calculate probabilities of households’ wind and flood insurance decisions through DCM
    ##•	Calculate wind and flood premiums if purchasing insurance
    ##•	Calculate households’ wind and flood insurance decisions based on DCM probabilities, minimum premium limit ($100), and affordability constraint (less than 2.5% or 5% of home values for low- or high-risk zones)
    ##•	Calculate final insurance decisions. We assume insurers either offer “wind insurance only” or “wind+flood insurance package”

    uninsured_fld_mark = np.zeros((hoNum_X, 3))  # 1st column: don't want to do; 2nd col: want to do but not qualified for 100; 3rd col: want to do but not qualified for budget
    uninsured_wd_mark = np.zeros((hoNum_X, 3))
    
    # Determine uninsured flood mark
    uninsured_fld_mark[:,0] = np.squeeze((U_ZB_flood_all > U_L_flood_all).astype(int))
    
    # Determine uninsured flood mark
    uninsured_wd_mark[:, 0] = np.squeeze((U_ZB_wind_all > U_L_wind_all).astype(int))
    
    priced_constx_flood = (lambda_X + tao + 1) * constx_flood  # num HO*1
    priced_constx_wind = (lambda_X + tao + 1) * constx_wind  # price(1+tau+lambda)*(Loss-deductible)    
    
    index_X_const1_fld = priced_constx_flood > 100
    uninsured_fld_mark[:, 1] = (1 - index_X_const1_fld)  # priced (L-B)>eta; num HO*1; uninsured written in col 2.
    
    index_X_const1_wd = priced_constx_wind > 100
    uninsured_wd_mark[:, 1] = (1 - index_X_const1_wd)

    index_X_const2_fld = priced_constx_flood < coeff * np.squeeze(homevalue_X)
    uninsured_fld_mark[:, 2] = (1 - index_X_const2_fld)  # priced (L-B)<k_LR/SFHA*HV_m; uninsured written in col 3.
    
    index_X_const2_wd = priced_constx_wind < coeff * np.squeeze(homevalue_X)
    uninsured_wd_mark[:, 2] = (1 - index_X_const2_wd)

    fixed_const_fld = np.squeeze((ho_aveLoss_fld_X > 0)) * index_X_const1_fld * index_X_const2_fld  # if average flood loss>0, >eta, <k*HV at the same time, then 1.
    fixed_const_wd = np.squeeze((ho_aveLoss_wd_X > 0)) * index_X_const1_wd * index_X_const2_wd   
    
    # Determine who buys insurance
    whobuy_flood_X = np.where(np.squeeze(U_ZB_flood_all <= U_L_flood_all), fixed_const_fld, 0)
    whobuy_wind_X = np.where(np.squeeze(U_ZB_wind_all <= U_L_wind_all), fixed_const_wd, 0)
    
    whofinallypay_wd_X = (np.squeeze((ho_aveLoss_fld_X == 0)) + whobuy_flood_X) * whobuy_wind_X  # ho_aveLoss_fld_X=0 => fixed_const_fld=0 => whobuy_flood_X=0
    whofinallypay_fld_X = whobuy_flood_X * whobuy_wind_X  # buy flood must buy wind; flood insurance is additional   
    
    ########
    ##•	Update insurance decisions based on low-income options. Options include: (i) without considering low-income households with “FLAG_poor=0”; (ii) mandatory insurance for low-income households with “FLAG_poor=1”; (iii) low-income households purchase insurance through DCM with “FLAG_poor=2”
    ##•	Update premiums for low-income households based on discounted price
    if index_L.size > 0:
        if FLAG_poor == 0:  # all low income HO no ins
            whofinallypay_fld_X[index_L-1] = 0
            whofinallypay_wd_X[index_L-1] = 0
        elif FLAG_poor == 1:  # all low income HO must ins
            whofinallypay_fld_X[index_L-1] = 1
            whofinallypay_wd_X[index_L-1] = 1
        elif FLAG_poor == 2:  # only keep index_L; re-set whofinallypay array; then all low income HO must ins
            whofinallypay_fld_X = np.zeros_like(whofinallypay_fld_X)
            whofinallypay_wd_X = np.zeros_like(whofinallypay_wd_X)
            whofinallypay_fld_X[index_L-1] = 1
            whofinallypay_wd_X[index_L-1] = 1

        priced_constx_flood[index_L-1] = 1.35 * constx_flood[index_L-1]  # for low income (HO index), no lambda loading factor. Only (1+tau)
        priced_constx_wind[index_L-1] = 1.35 * constx_wind[index_L-1]    
    
    ########
    ## •Calculate and return premiums, deductibles, insured losses, number of households insured, and market demand
    # Calculate the insurance premium for each HO
    P_ho_fldwd_X = priced_constx_wind * whofinallypay_wd_X + priced_constx_flood * whofinallypay_fld_X

    # Calculate the insurance deductible for each HO per hurricane
    B_buy_perhurr_fldwd_X = (B_perhurr_fld_X * whofinallypay_fld_X.reshape(-1,1) + B_perhurr_wd_X * whofinallypay_wd_X.reshape(-1,1))
    
    # Calculate the expected total premium for the insurer
    P_sum_avehurr_fldwd_X = np.dot(priced_constx_wind.T, whofinallypay_wd_X) + np.dot(priced_constx_flood.T, whofinallypay_fld_X)
    
    # Calculate the total deductible for the insurer per hurricane
    B_sum_buy_perhurr_fldwd_X = (np.dot(B_perhurr_fld_X.T, whofinallypay_fld_X) + np.dot(B_perhurr_wd_X.T, whofinallypay_wd_X)).T
    
    # Calculate the loss for each HO per hurricane
    L_ho_buy_perhurr_fldwd_X = ho_perhurrLoss_fld_X * whofinallypay_fld_X.reshape(-1,1) + ho_perhurrLoss_wd_X * whofinallypay_wd_X.reshape(-1,1)
    
    # Calculate the total loss for the insurer per hurricane
    L_sum_buy_perhurr_fldwd_X = np.sum(L_ho_buy_perhurr_fldwd_X, axis=0)
    
    # Calculate the expected total loss for the insurer
    L_sum_buy_avehurr_fldwd_X = np.dot(L_sum_buy_perhurr_fldwd_X, Pr)
    
    # Calculate the total loss for HOs who do not buy insurance
    L_sum_nobuy_perhurr_fldwd_X = np.sum(ho_perhurrLoss_fld_X, axis=0) + np.sum(ho_perhurrLoss_wd_X, axis=0) - L_sum_buy_perhurr_fldwd_X
    
    # Calculate the number of insured people for wind and flood
    num_insured_wd = np.floor(np.sum(whofinallypay_wd_X))  # pay wd; why need floor(), continuous var!
    num_insured_fld = np.floor(np.sum(whofinallypay_fld_X))  # pay wd >= pay wind
    
    # Calculate the number of insured and uninsured people
    num_insured = np.array([
        num_insured_fld, 
        num_insured_wd,
        np.sum(index_X_const1_fld) / hoNum_X, 
        np.sum(index_X_const1_wd) / hoNum_X, 
        np.sum(index_X_const2_fld) / hoNum_X, 
        np.sum(index_X_const2_wd) / hoNum_X,
        np.sum(uninsured_fld_mark) / hoNum_X, 
        np.sum(uninsured_wd_mark) / hoNum_X, 
        num_insured_fld / hoNum_X, 
        num_insured_wd / hoNum_X
    ])
    
    # Calculate expected insurance demand
    Q_sum_avehurr_fldwd_X = np.dot(constx_flood, whofinallypay_fld_X) + np.dot(constx_wind, whofinallypay_wd_X)  # 1*1; Q: expected insurance demand.
    
    # Clear variables
    del ho_perhurrLoss_fld_X, ho_perhurrLoss_wd_X
    
    # The first 7 are the outputs of no low income
    return (P_sum_avehurr_fldwd_X, L_sum_buy_avehurr_fldwd_X, L_sum_nobuy_perhurr_fldwd_X, 
            L_sum_buy_perhurr_fldwd_X, B_sum_buy_perhurr_fldwd_X, num_insured, Q_sum_avehurr_fldwd_X, 
            P_ho_fldwd_X, B_buy_perhurr_fldwd_X,num_insured_fld,num_insured_wd,L_ho_buy_perhurr_fldwd_X) # these for are for all income    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    