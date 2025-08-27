#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 23 17:37:17 2024

@author: Jingya Wang
"""

import numpy as np
from Func_cal_hoInsure_united import Func_cal_hoInsure_united
from Func_cal_insOpt import Func_cal_insOpt

def Func_imp_EQM_united(scenario, year, Elambda_perlvl_LR, Elambda_perlvl_SFHA, stru,
                        ho_aveLoss_fld_LR, ho_aveLoss_fld_SFHA, ho_perhurrLoss_fld_LR, ho_perhurrLoss_fld_SFHA,
                        ho_aveLoss_wd_LR, ho_aveLoss_wd_SFHA, ho_perhurrLoss_wd_LR, ho_perhurrLoss_wd_SFHA,
                        ho_cumEXP_fld_LR_SCENYEAR, ho_cumEXP_fld_SFHA_SCENYEAR, ho_lastEXP_fld_LR_SCENYEAR, ho_lastEXP_fld_SFHA_SCENYEAR,
                        index_L, n_scenarios, years, insurance_price_LB, deductible, hoNum_LR, hoNum_SFHA, k_LR, k_SFHA, g, phi, beta, ho_dcm, FLAG_DCM):
    """ 
    Implement equilibrium prices to different insurance markets and calculate households’ insurance decisions, premiums collected by insurers, 
    market demand, insurers’ profits, and optimal reinsurance strategies based on equilibrium prices
    """

    ########
    ## Initialize parameters and variables
    insTail_fld_LR = stru['stru']['insTail_fld_LR'][0,0]
    insTail_fld_SFHA = stru['stru']['insTail_fld_SFHA'][0,0]
    insTail_wd_LR = stru['stru']['insTail_wd_LR'][0,0]
    insTail_wd_SFHA = stru['stru']['insTail_wd_SFHA'][0,0]

    Pr = stru['stru']['Pr'][0,0]
    hurr_sim = stru['stru']['hurr_sim'][0,0]
    
    ho_prem_fld_LR = stru['stru']['ho_prem_fld_LR'][0,0]
    ho_prem_wd_LR = stru['stru']['ho_prem_wd_LR'][0,0]
    homevalue_LR = stru['stru']['homevalue_LR'][0,0]
    ho_prem_fld_SFHA = stru['stru']['ho_prem_fld_SFHA'][0,0]
    ho_prem_wd_SFHA = stru['stru']['ho_prem_wd_SFHA'][0,0]
    homevalue_SFHA = stru['stru']['homevalue_SFHA'][0,0]
    
    
    Elambda_LR_1firm = Elambda_perlvl_LR[0]
    Elambda_SFHA_1firm = Elambda_perlvl_SFHA[0]
    Elambda_LR_4firm = Elambda_perlvl_LR[3]
    Elambda_SFHA_4firm = Elambda_perlvl_SFHA[3]

    optimized_F_EQM_eachInsurer = np.zeros(4)
    optimized_A_EQM_eachInsurer = np.zeros(4)
    optimized_M_EQM_eachInsurer = np.zeros(4)
    optimized_insolvent_EQM_eachInsurer = np.zeros(4)
    insol_year = np.zeros(4)
    real_L_allInsurers = np.zeros(4)
    real_P_allInsurers = np.zeros(4)
    real_B_allInsurers = np.zeros(4)
    real_reIns_P_eachInsurer = np.zeros(4)
    
    household = hoNum_LR + hoNum_SFHA
    
    risk_theta_LR = None # risk aversion, not used in DCM, will imported if utility function is used
    risk_theta_SFHA = None
    if FLAG_DCM == 2: # utility function is used
        risk_theta_LR = stru['stru']['risk_theta_LR'][0,0]
        risk_theta_SFHA = stru['stru']['risk_theta_SFHA'][0,0]    
    
    ########
    ## Calculate households’ uninsured losses for current scenario and year assuming no insurance in the market

    temp = hurr_sim[scenario-1, year-1]
    hs = np.mod(np.floor([temp/1e8, temp/1e6, temp/1e4, temp/1e2, temp]), 100)
    hs = hs[hs != 0]
    hs = hs.astype(int) - 1

    L_gini_LR = np.sum(ho_perhurrLoss_fld_LR[:, hs], axis=1) + np.sum(ho_perhurrLoss_wd_LR[:, hs], axis=1)
    L_gini_SFHA = np.sum(ho_perhurrLoss_fld_SFHA[:, hs], axis=1) + np.sum(ho_perhurrLoss_wd_SFHA[:, hs], axis=1)
    Luni_gini_noPB = np.concatenate((L_gini_LR, L_gini_SFHA))
    
    L_gini_fld_LR = np.sum(ho_perhurrLoss_fld_LR[:, hs], axis=1)
    L_gini_fld_SFHA = np.sum(ho_perhurrLoss_fld_SFHA[:, hs], axis=1)
    Luni_gini_noPB_fld = np.concatenate((L_gini_fld_LR, L_gini_fld_SFHA))
    
    L_gini_wd_LR = np.sum(ho_perhurrLoss_wd_LR[:, hs], axis=1)
    L_gini_wd_SFHA = np.sum(ho_perhurrLoss_wd_SFHA[:, hs], axis=1)
    Luni_gini_noPB_wd = np.concatenate((L_gini_wd_LR, L_gini_wd_SFHA))
    
    
    
    ########
    ## For market with only 1 insurer, calculate premiums, deductibles, uninsured losses for households, 
    ## as well as premiums collected, total insured losses, total deductibles, reinsurance premiums, profits, A, M, and insolvent rates for insurers by calling functions “Func_cal_hoInsure_united.m” and “Func_cal_insOpt.py”
    x, y, z = Elambda_LR_1firm, [], 0
    P_sum_avehurr_fldwd_LR_1firm_all, _,L_sum_nobuy_perhurr_fldwd_LR_1firm_all,L_sum_buy_perhurr_fldwd_LR_1firm_all,B_sum_buy_perhurr_fldwd_LR_1firm_all,temp_num_insured_LR_1firm_all,_,\
        P_ho_fldwd_LR_1firm_all, B_buy_perhurr_fldwd_LR_1firm_all,num_insured_fld_LR_1firm_all,num_insured_wd_LR_1firm_all,L_ho_buy_perhurr_fldwd_LR, whofinallypay_wd_lr, whofinallypay_fld_lr,_,_\
            = Func_cal_hoInsure_united(
        x, Pr, ho_perhurrLoss_fld_LR, ho_perhurrLoss_wd_LR, ho_aveLoss_fld_LR, ho_aveLoss_wd_LR, ho_prem_fld_LR, ho_prem_wd_LR, homevalue_LR, ho_cumEXP_fld_LR_SCENYEAR, ho_lastEXP_fld_LR_SCENYEAR, insTail_fld_LR, insTail_wd_LR, y, risk_theta_LR, insurance_price_LB, deductible, hoNum_LR, hoNum_SFHA, k_LR, k_SFHA, ho_dcm, z)

    x, y, z = Elambda_SFHA_1firm, [], 0
    P_sum_avehurr_fldwd_SFHA_1firm_all, _,L_sum_nobuy_perhurr_fldwd_SFHA_1firm_all,L_sum_buy_perhurr_fldwd_SFHA_1firm_all,B_sum_buy_perhurr_fldwd_SFHA_1firm_all,temp_num_insured_SFHA_1firm_all,_,\
        P_ho_fldwd_SFHA_1firm_all, B_buy_perhurr_fldwd_SFHA_1firm_all,num_insured_fld_SFHA_1firm_all,num_insured_wd_SFHA_1firm_all,L_ho_buy_perhurr_fldwd_SFHA, whofinallypay_wd_hr, whofinallypay_fld_hr,_,_ \
            = Func_cal_hoInsure_united(
        x, Pr, ho_perhurrLoss_fld_SFHA, ho_perhurrLoss_wd_SFHA, ho_aveLoss_fld_SFHA, ho_aveLoss_wd_SFHA, ho_prem_fld_SFHA, ho_prem_wd_SFHA, homevalue_SFHA, ho_cumEXP_fld_SFHA_SCENYEAR, ho_lastEXP_fld_SFHA_SCENYEAR, insTail_fld_SFHA, insTail_wd_SFHA, y, risk_theta_SFHA, insurance_price_LB, deductible, hoNum_LR, hoNum_SFHA, k_LR, k_SFHA, ho_dcm, z)

    F_eachInsurer_1firm_all, A_eachInsurer_1firm_all, M_eachInsurer_1firm_all, insolvent_eachInsurer_1firm_all, rsy_eachInsurer_1firm_all, insol_year_1firm = Func_cal_insOpt(
        year, (P_sum_avehurr_fldwd_LR_1firm_all + P_sum_avehurr_fldwd_SFHA_1firm_all) / 1, (L_sum_buy_perhurr_fldwd_LR_1firm_all + L_sum_buy_perhurr_fldwd_SFHA_1firm_all) / 1, (B_sum_buy_perhurr_fldwd_LR_1firm_all + B_sum_buy_perhurr_fldwd_SFHA_1firm_all) / 1, Pr, hurr_sim, n_scenarios, years, insurance_price_LB, g, phi, beta, 'not precise')

    x = '1firm_all'
    i = 0
    # Calculate and assign values
    if len(hs) > 0:
        real_L_allInsurers[i] = np.sum(L_sum_buy_perhurr_fldwd_LR_1firm_all[hs]) + np.sum(L_sum_buy_perhurr_fldwd_SFHA_1firm_all[hs])
        real_B_allInsurers[i] = np.sum(B_sum_buy_perhurr_fldwd_LR_1firm_all[hs]) + np.sum(B_sum_buy_perhurr_fldwd_SFHA_1firm_all[hs])
    else:
        real_L_allInsurers[i] = 0  # If hs is empty, assign 0
        real_B_allInsurers[i] = 0  # If hs is empty, assign 0
    
    real_P_allInsurers[i] = P_sum_avehurr_fldwd_LR_1firm_all + P_sum_avehurr_fldwd_SFHA_1firm_all
    real_reIns_P_eachInsurer[i] = rsy_eachInsurer_1firm_all
    
    optimized_F_EQM_eachInsurer[i] = F_eachInsurer_1firm_all
    optimized_A_EQM_eachInsurer[i] = A_eachInsurer_1firm_all
    optimized_M_EQM_eachInsurer[i] = M_eachInsurer_1firm_all
    optimized_insolvent_EQM_eachInsurer[i] = insolvent_eachInsurer_1firm_all
    insol_year[i] = insol_year_1firm


    P_gini_LR = P_ho_fldwd_LR_1firm_all
    P_gini_SFHA = P_ho_fldwd_SFHA_1firm_all
    
    # check is hs is empty
    if hs.size == 0:
        B_gini_LR = np.zeros(B_buy_perhurr_fldwd_LR_1firm_all.shape[0])
        B_gini_SFHA = np.zeros(B_buy_perhurr_fldwd_SFHA_1firm_all.shape[0])
    else:
        # Summing specific columns
        B_gini_LR = np.sum(B_buy_perhurr_fldwd_LR_1firm_all[:, hs], axis=1)
        B_gini_SFHA = np.sum(B_buy_perhurr_fldwd_SFHA_1firm_all[:, hs], axis=1)
    
    # Concatenating results
    P_gini_1firm = np.concatenate((P_gini_LR, P_gini_SFHA))
    B_gini_1firm = np.concatenate((B_gini_LR, B_gini_SFHA))

    # Check if hs is empty
    if hs.size == 0:
        # If hs is empty, summing along axis 1 would result in an error, so handle it differently
        L_ho_sum = np.zeros_like(Luni_gini_noPB)  # Create an array of zeros with the same shape as Luni_gini_noPB
    else:
        # Sum the specified columns along the second axis (axis=1)
        sum_LR = np.sum(L_ho_buy_perhurr_fldwd_LR[:, hs], axis=1)
        sum_SFHA = np.sum(L_ho_buy_perhurr_fldwd_SFHA[:, hs], axis=1)
        
        # Concatenate the results vertically
        L_ho_sum = np.concatenate((sum_LR, sum_SFHA))
        
        # Subtract the concatenated result from Luni_gini_noPB
    Luni_gini_1firm = Luni_gini_noPB - L_ho_sum

    # Clear the variables from memory
    P_ho_fldwd_LR_1firm_all = None
    P_ho_fldwd_SFHA_1firm_all = None
    B_buy_perhurr_fldwd_LR_1firm_all = None
    B_buy_perhurr_fldwd_SFHA_1firm_all = None
    

    ########
    ## For market with 4 insurers with only considering middle- and high-income groups, calculate premiums, deductibles, uninsured losses for households, 
    ## as well as premiums collected, total insured losses, total deductibles, reinsurance premiums, profits, A, M, and insolvent rates for insurers by calling functions “Func_cal_hoInsure_united.py” and “Func_cal_insOpt.py”
    
    # For 4 firm, rich
    index_L = np.squeeze(index_L)
    x, y, z = Elambda_LR_4firm, index_L, 0
    P_sum_avehurr_fldwd_LR_4firm_rich, _, L_sum_nobuy_perhurr_fldwd_LR_4firm_rich, \
    L_sum_buy_perhurr_fldwd_LR_4firm_rich, B_sum_buy_perhurr_fldwd_LR_4firm_rich, \
    temp_num_insured_LR_4firm_rich, _, P_ho_fldwd_LR_4firm_rich, \
    B_buy_perhurr_fldwd_LR_4firm_rich, num_insured_fld_LR_4firm_rich, \
    num_insured_wd_LR_4firm_rich, L_ho_buy_perhurr_fldwd_LR, whofinallypay_wd_lr_rich, whofinallypay_fld_lr_rich,uninsured_fld_mark_lr_rich, uninsured_wd_mark_lr_rich = Func_cal_hoInsure_united(
        x, Pr, ho_perhurrLoss_fld_LR, ho_perhurrLoss_wd_LR, ho_aveLoss_fld_LR,
        ho_aveLoss_wd_LR, ho_prem_fld_LR, ho_prem_wd_LR, homevalue_LR, 
        ho_cumEXP_fld_LR_SCENYEAR, ho_lastEXP_fld_LR_SCENYEAR, insTail_fld_LR, 
        insTail_wd_LR, y, risk_theta_LR, insurance_price_LB, deductible, hoNum_LR, hoNum_SFHA, k_LR, k_SFHA, ho_dcm, z
    )
    
    x, y, z = Elambda_SFHA_4firm, index_L, 0
    P_sum_avehurr_fldwd_SFHA_4firm_rich, _, L_sum_nobuy_perhurr_fldwd_SFHA_4firm_rich, \
    L_sum_buy_perhurr_fldwd_SFHA_4firm_rich, B_sum_buy_perhurr_fldwd_SFHA_4firm_rich, \
    temp_num_insured_SFHA_4firm_rich, _, P_ho_fldwd_SFHA_4firm_rich, \
    B_buy_perhurr_fldwd_SFHA_4firm_rich, num_insured_fld_SFHA_4firm_rich, \
    num_insured_wd_SFHA_4firm_rich, L_ho_buy_perhurr_fldwd_SFHA, whofinallypay_wd_hr_rich, whofinallypay_fld_hr_rich, uninsured_fld_mark_hr_rich, uninsured_wd_mark_hr_rich = Func_cal_hoInsure_united(
        x, Pr, ho_perhurrLoss_fld_SFHA, ho_perhurrLoss_wd_SFHA, ho_aveLoss_fld_SFHA,
        ho_aveLoss_wd_SFHA, ho_prem_fld_SFHA, ho_prem_wd_SFHA, homevalue_SFHA, 
        ho_cumEXP_fld_SFHA_SCENYEAR, ho_lastEXP_fld_SFHA_SCENYEAR, insTail_fld_SFHA, 
        insTail_wd_SFHA, y, risk_theta_SFHA, insurance_price_LB, deductible, hoNum_LR, hoNum_SFHA, k_LR, k_SFHA, ho_dcm, z
    )
    
    F_eachInsurer_4firm_rich, A_eachInsurer_4firm_rich, M_eachInsurer_4firm_rich, \
    insolvent_eachInsurer_4firm_rich, rsy_eachInsurer_4firm_rich, insol_year_4firm_rich = Func_cal_insOpt(
        year, (P_sum_avehurr_fldwd_LR_4firm_rich + P_sum_avehurr_fldwd_SFHA_4firm_rich) / 4,
        (L_sum_buy_perhurr_fldwd_LR_4firm_rich + L_sum_buy_perhurr_fldwd_SFHA_4firm_rich) / 4, 
        (B_sum_buy_perhurr_fldwd_LR_4firm_rich + B_sum_buy_perhurr_fldwd_SFHA_4firm_rich) / 4, 
        Pr, hurr_sim, n_scenarios, years, insurance_price_LB, g, phi, beta, 'not precise'
    )
    
    x = '4firm_rich'
    i = 1  # In MATLAB, indexing starts from 1
    
    # Calculating real_L_allInsurers, real_B_allInsurers, real_P_allInsurers, real_reIns_P_eachInsurer
    if hs.size == 0:
        real_L_allInsurers[i] = 0
        real_B_allInsurers[i] = 0
    else:
        real_L_allInsurers[i] = np.sum(L_sum_buy_perhurr_fldwd_LR_4firm_rich[hs]) + np.sum(L_sum_buy_perhurr_fldwd_SFHA_4firm_rich[hs])
        real_B_allInsurers[i] = np.sum(B_sum_buy_perhurr_fldwd_LR_4firm_rich[hs]) + np.sum(B_sum_buy_perhurr_fldwd_SFHA_4firm_rich[hs])
    
    real_P_allInsurers[i] = P_sum_avehurr_fldwd_LR_4firm_rich + P_sum_avehurr_fldwd_SFHA_4firm_rich
    real_reIns_P_eachInsurer[i] = rsy_eachInsurer_4firm_rich
    
    # Calculating optimized_F_EQM_eachInsurer, optimized_A_EQM_eachInsurer, optimized_M_EQM_eachInsurer, optimized_insolvent_EQM_eachInsurer
    optimized_F_EQM_eachInsurer[i] = F_eachInsurer_4firm_rich
    optimized_A_EQM_eachInsurer[i] = A_eachInsurer_4firm_rich
    optimized_M_EQM_eachInsurer[i] = M_eachInsurer_4firm_rich
    optimized_insolvent_EQM_eachInsurer[i] = insolvent_eachInsurer_4firm_rich
    insol_year[i] = insol_year_4firm_rich
    
    # Concatenate arrays only if hs is not empty
    if hs.size > 0:
        temp_4firm_rich = np.concatenate((np.sum(L_ho_buy_perhurr_fldwd_LR[:, hs], axis=1), np.sum(L_ho_buy_perhurr_fldwd_SFHA[:, hs], axis=1)))
    else:
        temp_4firm_rich = np.zeros(len(L_ho_buy_perhurr_fldwd_LR) + len(L_ho_buy_perhurr_fldwd_SFHA))


    
    
    
    ########
    ## For market with 4 insurers with only considering low-income groups, calculate premiums, deductibles, uninsured losses for households, 
    ## as well as premiums collected, total insured losses, total deductibles, reinsurance premiums, profits, A, M, and insolvent rates for insurers by calling functions “Func_cal_hoInsure_united.py” and “Func_cal_insOpt.py”
    # 4 firm, poor
    # Create the full range from 1 to 931902 (MATLAB style)
    full_range = np.arange(1, household + 1)  # This creates an array from 1 to 931902
    
    # Use numpy's setdiff1d to find the difference
    index_MH = np.setdiff1d(full_range, index_L)
    x, y, z = Elambda_LR_4firm, index_MH, 0
    P_sum_avehurr_fldwd_LR_4firm_poor, _, L_sum_nobuy_perhurr_fldwd_LR_4firm_poor, \
    L_sum_buy_perhurr_fldwd_LR_4firm_poor, B_sum_buy_perhurr_fldwd_LR_4firm_poor, \
    temp_num_insured_LR_4firm_poor, _, P_ho_fldwd_LR_4firm_poor, \
    B_buy_perhurr_fldwd_LR_4firm_poor, num_insured_fld_LR_4firm_poor, \
    num_insured_wd_LR_4firm_poor, L_ho_buy_perhurr_fldwd_LR, whofinallypay_wd_lr_poor, whofinallypay_fld_lr_poor, uninsured_fld_mark_lr_poor, uninsured_wd_mark_lr_poor = Func_cal_hoInsure_united(
        x, Pr, ho_perhurrLoss_fld_LR, ho_perhurrLoss_wd_LR, ho_aveLoss_fld_LR,
        ho_aveLoss_wd_LR, ho_prem_fld_LR, ho_prem_wd_LR, homevalue_LR, 
        ho_cumEXP_fld_LR_SCENYEAR, ho_lastEXP_fld_LR_SCENYEAR, insTail_fld_LR, 
        insTail_wd_LR, y, risk_theta_LR, insurance_price_LB, deductible, hoNum_LR, hoNum_SFHA, k_LR, k_SFHA, ho_dcm, z
    )
    
    x, y, z = Elambda_SFHA_4firm, index_MH, 0
    P_sum_avehurr_fldwd_SFHA_4firm_poor, _, L_sum_nobuy_perhurr_fldwd_SFHA_4firm_poor, \
    L_sum_buy_perhurr_fldwd_SFHA_4firm_poor, B_sum_buy_perhurr_fldwd_SFHA_4firm_poor, \
    temp_num_insured_SFHA_4firm_poor, _, P_ho_fldwd_SFHA_4firm_poor, \
    B_buy_perhurr_fldwd_SFHA_4firm_poor, num_insured_fld_SFHA_4firm_poor, \
    num_insured_wd_SFHA_4firm_poor, L_ho_buy_perhurr_fldwd_SFHA, whofinallypay_wd_hr_poor, whofinallypay_fld_hr_poor, uninsured_fld_mark_hr_poor, uninsured_wd_mark_hr_poor = Func_cal_hoInsure_united(
        x, Pr, ho_perhurrLoss_fld_SFHA, ho_perhurrLoss_wd_SFHA, ho_aveLoss_fld_SFHA,
        ho_aveLoss_wd_SFHA, ho_prem_fld_SFHA, ho_prem_wd_SFHA, homevalue_SFHA, 
        ho_cumEXP_fld_SFHA_SCENYEAR, ho_lastEXP_fld_SFHA_SCENYEAR, insTail_fld_SFHA, 
        insTail_wd_SFHA, y, risk_theta_SFHA, insurance_price_LB, deductible, hoNum_LR, hoNum_SFHA, k_LR, k_SFHA, ho_dcm, z
    )
    
    (F_eachInsurer_4firm_poor, A_eachInsurer_4firm_poor, M_eachInsurer_4firm_poor, 
    insolvent_eachInsurer_4firm_poor, rsy_eachInsurer_4firm_poor, insol_year_4firm_poor) = Func_cal_insOpt(
        year, (P_sum_avehurr_fldwd_LR_4firm_poor + P_sum_avehurr_fldwd_SFHA_4firm_poor) / 4,
        (L_sum_buy_perhurr_fldwd_LR_4firm_poor + L_sum_buy_perhurr_fldwd_SFHA_4firm_poor) / 4, 
        (B_sum_buy_perhurr_fldwd_LR_4firm_poor + B_sum_buy_perhurr_fldwd_SFHA_4firm_poor) / 4, 
        Pr, hurr_sim, n_scenarios, years, insurance_price_LB, g, phi, beta, 'not precise'
    )
    
    x = '4firm_poor'
    i = 2
    
    # Calculating real_L_allInsurers, real_B_allInsurers, real_P_allInsurers, real_reIns_P_eachInsurer
    if hs.size == 0:
        real_L_allInsurers[i] = 0
        real_B_allInsurers[i] = 0
    else:
        real_L_allInsurers[i] = np.sum(L_sum_buy_perhurr_fldwd_LR_4firm_poor[hs]) + np.sum(L_sum_buy_perhurr_fldwd_SFHA_4firm_poor[hs])
        real_B_allInsurers[i] = np.sum(B_sum_buy_perhurr_fldwd_LR_4firm_poor[hs]) + np.sum(B_sum_buy_perhurr_fldwd_SFHA_4firm_poor[hs])
    
    real_P_allInsurers[i] = P_sum_avehurr_fldwd_LR_4firm_poor + P_sum_avehurr_fldwd_SFHA_4firm_poor
    real_reIns_P_eachInsurer[i] = rsy_eachInsurer_4firm_poor
    
    # Calculating optimized_F_EQM_eachInsurer, optimized_A_EQM_eachInsurer, optimized_M_EQM_eachInsurer, optimized_insolvent_EQM_eachInsurer
    optimized_F_EQM_eachInsurer[i] = F_eachInsurer_4firm_poor
    optimized_A_EQM_eachInsurer[i] = A_eachInsurer_4firm_poor
    optimized_M_EQM_eachInsurer[i] = M_eachInsurer_4firm_poor
    optimized_insolvent_EQM_eachInsurer[i] = insolvent_eachInsurer_4firm_poor
    insol_year[i] = insol_year_4firm_poor
    
    # Concatenate arrays only if hs is not empty
    if hs.size > 0:
        temp_4firm_poor = np.concatenate((np.sum(L_ho_buy_perhurr_fldwd_LR[:, hs], axis=1), np.sum(L_ho_buy_perhurr_fldwd_SFHA[:, hs], axis=1)))
    else:
        temp_4firm_poor = np.zeros(len(L_ho_buy_perhurr_fldwd_LR) + len(L_ho_buy_perhurr_fldwd_SFHA))
                
        
        
    
    ########
    ## For market with 4 insurers, re-calculate premiums, deductibles, uninsured losses for households, 
    ## as well as premiums collected, total insured losses, total deductibles, reinsurance premiums, profits, A, M, and insolvent rates for insurers based on results from “4 insurers low-income households only” and “4 insurers middle- and high-income households only”
    
    # Calculating values for 4 firm, all
    (F_eachInsurer_4firm_all, A_eachInsurer_4firm_all, M_eachInsurer_4firm_all, 
     insolvent_eachInsurer_4firm_all, rsy_eachInsurer_4firm_all, insol_year_4firm_all) = Func_cal_insOpt(
        year, 
        (P_sum_avehurr_fldwd_LR_4firm_rich + P_sum_avehurr_fldwd_SFHA_4firm_rich + P_sum_avehurr_fldwd_LR_4firm_poor + P_sum_avehurr_fldwd_SFHA_4firm_poor) / 4, 
        (L_sum_buy_perhurr_fldwd_LR_4firm_rich + L_sum_buy_perhurr_fldwd_SFHA_4firm_rich + L_sum_buy_perhurr_fldwd_LR_4firm_poor + L_sum_buy_perhurr_fldwd_SFHA_4firm_poor) / 4, 
        (B_sum_buy_perhurr_fldwd_LR_4firm_rich + B_sum_buy_perhurr_fldwd_SFHA_4firm_rich + B_sum_buy_perhurr_fldwd_LR_4firm_poor + B_sum_buy_perhurr_fldwd_SFHA_4firm_poor) / 4, 
        Pr, hurr_sim, n_scenarios, year, insurance_price_LB, g, phi, beta, 'not precise')
    
    # Summing up the results
    P_sum_avehurr_fldwd_LR_4firm_all = P_sum_avehurr_fldwd_LR_4firm_rich + P_sum_avehurr_fldwd_LR_4firm_poor
    L_sum_buy_perhurr_fldwd_LR_4firm_all = L_sum_buy_perhurr_fldwd_LR_4firm_rich + L_sum_buy_perhurr_fldwd_LR_4firm_poor
    B_sum_buy_perhurr_fldwd_LR_4firm_all = B_sum_buy_perhurr_fldwd_LR_4firm_rich + B_sum_buy_perhurr_fldwd_LR_4firm_poor
    P_ho_fldwd_LR_4firm_all = P_ho_fldwd_LR_4firm_rich + P_ho_fldwd_LR_4firm_poor
    B_buy_perhurr_fldwd_LR_4firm_all = B_buy_perhurr_fldwd_LR_4firm_rich + B_buy_perhurr_fldwd_LR_4firm_poor
    num_insured_fld_LR_4firm_all = num_insured_fld_LR_4firm_rich + num_insured_fld_LR_4firm_poor
    num_insured_wd_LR_4firm_all = num_insured_wd_LR_4firm_rich + num_insured_wd_LR_4firm_poor
    pay_wd_lr = whofinallypay_wd_lr_poor+ whofinallypay_wd_lr_rich
    pay_fld_lr = whofinallypay_fld_lr_poor+ whofinallypay_fld_lr_rich
    
    P_sum_avehurr_fldwd_SFHA_4firm_all = P_sum_avehurr_fldwd_SFHA_4firm_rich + P_sum_avehurr_fldwd_SFHA_4firm_poor
    L_sum_buy_perhurr_fldwd_SFHA_4firm_all = L_sum_buy_perhurr_fldwd_SFHA_4firm_rich + L_sum_buy_perhurr_fldwd_SFHA_4firm_poor
    B_sum_buy_perhurr_fldwd_SFHA_4firm_all = B_sum_buy_perhurr_fldwd_SFHA_4firm_rich + B_sum_buy_perhurr_fldwd_SFHA_4firm_poor
    P_ho_fldwd_SFHA_4firm_all = P_ho_fldwd_SFHA_4firm_rich + P_ho_fldwd_SFHA_4firm_poor
    B_buy_perhurr_fldwd_SFHA_4firm_all = B_buy_perhurr_fldwd_SFHA_4firm_rich + B_buy_perhurr_fldwd_SFHA_4firm_poor
    num_insured_fld_SFHA_4firm_all = num_insured_fld_SFHA_4firm_rich + num_insured_fld_SFHA_4firm_poor
    num_insured_wd_SFHA_4firm_all = num_insured_wd_SFHA_4firm_rich + num_insured_wd_SFHA_4firm_poor
    pay_wd_hr = whofinallypay_wd_hr_poor+ whofinallypay_wd_hr_rich
    pay_fld_hr = whofinallypay_fld_hr_poor+ whofinallypay_fld_hr_rich
    
    # Calculating and assigning the values to the respective arrays
    x = '4firm_all'
    i = 3
    if hs.size == 0:
        real_L_allInsurers[i] = 0
        real_B_allInsurers[i] = 0
    else:
        real_L_allInsurers[i] = np.sum(L_sum_buy_perhurr_fldwd_LR_4firm_all[hs]) + np.sum(L_sum_buy_perhurr_fldwd_SFHA_4firm_all[hs])
        real_B_allInsurers[i] = np.sum(B_sum_buy_perhurr_fldwd_LR_4firm_all[hs]) + np.sum(B_sum_buy_perhurr_fldwd_SFHA_4firm_all[hs])
    
    real_P_allInsurers[i] = P_sum_avehurr_fldwd_LR_4firm_all + P_sum_avehurr_fldwd_SFHA_4firm_all
    real_reIns_P_eachInsurer[i] = rsy_eachInsurer_4firm_all
    
    optimized_F_EQM_eachInsurer[i] = F_eachInsurer_4firm_all
    optimized_A_EQM_eachInsurer[i] = A_eachInsurer_4firm_all
    optimized_M_EQM_eachInsurer[i] = M_eachInsurer_4firm_all
    optimized_insolvent_EQM_eachInsurer[i] = insolvent_eachInsurer_4firm_all
    insol_year[i] = insol_year_4firm_all
    
    # Calculating P_gini and B_gini
    P_gini_LR = P_ho_fldwd_LR_4firm_all
    P_gini_SFHA = P_ho_fldwd_SFHA_4firm_all
    
    if hs.size == 0:
        B_gini_LR = np.zeros(B_buy_perhurr_fldwd_LR_4firm_all.shape[0])
        B_gini_SFHA = np.zeros(B_buy_perhurr_fldwd_SFHA_4firm_all.shape[0])
    else:
        B_gini_LR = np.sum(B_buy_perhurr_fldwd_LR_4firm_all[:, hs], axis=1)
        B_gini_SFHA = np.sum(B_buy_perhurr_fldwd_SFHA_4firm_all[:, hs], axis=1)
        
    P_gini_4firms = np.concatenate((P_gini_LR, P_gini_SFHA))
    B_gini_4firms = np.concatenate((B_gini_LR, B_gini_SFHA))
    
    # Calculating Luni_gini_4firms
    Luni_gini_4firms = Luni_gini_noPB - temp_4firm_poor - temp_4firm_rich
    
    # Clearing unnecessary variables
    P_ho_fldwd_LR_4firm_all = None
    P_ho_fldwd_SFHA_4firm_all = None
    B_buy_perhurr_fldwd_LR_4firm_all = None
    B_buy_perhurr_fldwd_SFHA_4firm_all = None


    
    
    
    ########
    ## For low-income special market (price: $1.35; mandatory insurance), calculate premiums, deductibles, and uninsured losses for households by calling function “Func_cal_hoInsure_united.m”
    
    # 1.35, all poor buy, forced
    [x, y, z] = Elambda_LR_4firm, index_L, 2
    _, _, _, _, _, _, _, P_ho_fldwd_LR_135_poor_forced, B_buy_perhurr_fldwd_LR_135_poor_forced, num_insured_fld_LR_135_poor_forced, num_insured_wd_LR_135_poor_forced, _ ,_,_,_,_= \
        Func_cal_hoInsure_united(x, Pr, ho_perhurrLoss_fld_LR, ho_perhurrLoss_wd_LR, ho_aveLoss_fld_LR, ho_aveLoss_wd_LR, ho_prem_fld_LR, ho_prem_wd_LR, homevalue_LR, ho_cumEXP_fld_LR_SCENYEAR, ho_lastEXP_fld_LR_SCENYEAR, insTail_fld_LR, insTail_wd_LR, y, risk_theta_LR, insurance_price_LB, deductible, hoNum_LR, hoNum_SFHA, k_LR, k_SFHA, ho_dcm, z)
    [x, y, z] = Elambda_SFHA_4firm, index_L, 2
    _, _, _, _, _, _, _, P_ho_fldwd_SFHA_135_poor_forced, B_buy_perhurr_fldwd_SFHA_135_poor_forced, num_insured_fld_SFHA_135_poor_forced, num_insured_wd_SFHA_135_poor_forced, _,_,_,_,_ = \
        Func_cal_hoInsure_united(x, Pr, ho_perhurrLoss_fld_SFHA, ho_perhurrLoss_wd_SFHA, ho_aveLoss_fld_SFHA, ho_aveLoss_wd_SFHA, ho_prem_fld_SFHA, ho_prem_wd_SFHA, homevalue_SFHA, ho_cumEXP_fld_SFHA_SCENYEAR, ho_lastEXP_fld_SFHA_SCENYEAR, insTail_fld_SFHA, insTail_wd_SFHA, y, risk_theta_SFHA, insurance_price_LB, deductible, hoNum_LR, hoNum_SFHA, k_LR, k_SFHA, ho_dcm, z)
    
    
    ########
    ## For low-income special market (price: $1.35; mandatory insurance) and 4 insurers market for middle- and high-income households, re-calculate premiums, deductibles, 
    ## and uninsured losses for households based on results from “$1.35 mandatory insurance for low-income households” and “4 insurers middle- and high-income households only”
    # 1.35 forced for all poor + DCM for rich 4 firm
    P_ho_fldwd_LR_135_poor_forced_all = P_ho_fldwd_LR_4firm_rich + P_ho_fldwd_LR_135_poor_forced
    B_buy_perhurr_fldwd_LR_135_poor_forced_all = B_buy_perhurr_fldwd_LR_4firm_rich + B_buy_perhurr_fldwd_LR_135_poor_forced
    num_insured_fld_LR_135_poor_forced_all = num_insured_fld_LR_4firm_rich + num_insured_fld_LR_135_poor_forced
    num_insured_wd_LR_135_poor_forced_all = num_insured_wd_LR_4firm_rich + num_insured_wd_LR_135_poor_forced
    
    P_ho_fldwd_SFHA_135_poor_forced_all = P_ho_fldwd_SFHA_4firm_rich + P_ho_fldwd_SFHA_135_poor_forced
    B_buy_perhurr_fldwd_SFHA_135_poor_forced_all = B_buy_perhurr_fldwd_SFHA_4firm_rich + B_buy_perhurr_fldwd_SFHA_135_poor_forced
    num_insured_fld_SFHA_135_poor_forced_all = num_insured_fld_SFHA_4firm_rich + num_insured_fld_SFHA_135_poor_forced
    num_insured_wd_SFHA_135_poor_forced_all = num_insured_wd_SFHA_4firm_rich + num_insured_wd_SFHA_135_poor_forced
    
    P_gini_LR = P_ho_fldwd_LR_135_poor_forced_all
    P_gini_SFHA = P_ho_fldwd_SFHA_135_poor_forced_all
    
    if hs.size == 0:
        B_gini_LR = np.zeros(B_buy_perhurr_fldwd_LR_135_poor_forced_all.shape[0])
        B_gini_SFHA = np.zeros(B_buy_perhurr_fldwd_SFHA_135_poor_forced_all.shape[0])
    else:
        B_gini_LR = np.sum(B_buy_perhurr_fldwd_LR_135_poor_forced_all[:, hs], axis=1)
        B_gini_SFHA = np.sum(B_buy_perhurr_fldwd_SFHA_135_poor_forced_all[:, hs], axis=1)
        

    P_gini_135_forced = np.concatenate((P_gini_LR, P_gini_SFHA))
    B_gini_135_forced = np.concatenate((B_gini_LR, B_gini_SFHA))
    
    # Initialize Luni_gini_135_forced
    if hs.size == 0:
        sum_L_ho_buy_LR = np.zeros(L_ho_buy_perhurr_fldwd_LR.shape[0])
        sum_L_ho_buy_SFHA = np.zeros(L_ho_buy_perhurr_fldwd_SFHA.shape[0])
    else:
        sum_L_ho_buy_LR = np.sum(L_ho_buy_perhurr_fldwd_LR[:, hs], axis=1)
        sum_L_ho_buy_SFHA = np.sum(L_ho_buy_perhurr_fldwd_SFHA[:, hs], axis=1)
    
    Luni_gini_135_forced = Luni_gini_noPB - np.concatenate((sum_L_ho_buy_LR, sum_L_ho_buy_SFHA)) - temp_4firm_rich
    
        
    # Clear specific variables by setting them to None
    P_ho_fldwd_LR_135_poor_forced_all = None
    P_ho_fldwd_SFHA_135_poor_forced_all = None
    B_buy_perhurr_fldwd_LR_135_poor_forced_all = None
    B_buy_perhurr_fldwd_SFHA_135_poor_forced_all = None

    
    ########
    ## For low-income special market (price: $1.35; DCM insurance) and 4 insurers market for middle- and high-income households, calculate premiums, deductibles, 
    ## and uninsured losses for households by calling function “Func_cal_hoInsure_united.m” and based on results from “4 insurers middle- and high-income households only”
    # 1.35, poor, DCM, discrete choice model
    x, y, z = 0, index_MH, 0
    _, _, _, _, _, num_check_poor_LR_135_DCM, _, P_ho_fldwd_LR_135_poor_DCM, B_buy_perhurr_fldwd_LR_135_poor_DCM, num_insured_fld_LR_135_poor_DCM, num_insured_wd_LR_135_poor_DCM, L_ho_buy_perhurr_fldwd_LR,_,_ ,_,_= Func_cal_hoInsure_united(x, Pr, ho_perhurrLoss_fld_LR, ho_perhurrLoss_wd_LR, ho_aveLoss_fld_LR, ho_aveLoss_wd_LR, ho_prem_fld_LR, ho_prem_wd_LR, homevalue_LR, ho_cumEXP_fld_LR_SCENYEAR, ho_lastEXP_fld_LR_SCENYEAR, insTail_fld_LR, insTail_wd_LR, y, risk_theta_LR, insurance_price_LB, deductible, hoNum_LR, hoNum_SFHA, k_LR, k_SFHA, ho_dcm, z)
    _, _, _, _, _, num_check_poor_SFHA_135_DCM, _, P_ho_fldwd_SFHA_135_poor_DCM, B_buy_perhurr_fldwd_SFHA_135_poor_DCM, num_insured_fld_SFHA_135_poor_DCM, num_insured_wd_SFHA_135_poor_DCM, L_ho_buy_perhurr_fldwd_SFHA,_,_,_,_ = Func_cal_hoInsure_united(x, Pr, ho_perhurrLoss_fld_SFHA, ho_perhurrLoss_wd_SFHA, ho_aveLoss_fld_SFHA, ho_aveLoss_wd_SFHA, ho_prem_fld_SFHA, ho_prem_wd_SFHA, homevalue_SFHA, ho_cumEXP_fld_SFHA_SCENYEAR, ho_lastEXP_fld_SFHA_SCENYEAR, insTail_fld_SFHA, insTail_wd_SFHA, y, risk_theta_SFHA, insurance_price_LB, deductible, hoNum_LR, hoNum_SFHA, k_LR, k_SFHA, ho_dcm, z)
     
    P_ho_fldwd_LR_135_poor_DCM_all = P_ho_fldwd_LR_4firm_rich + P_ho_fldwd_LR_135_poor_DCM
    B_buy_perhurr_fldwd_LR_135_poor_DCM_all = B_buy_perhurr_fldwd_LR_4firm_rich + B_buy_perhurr_fldwd_LR_135_poor_DCM
    P_ho_fldwd_SFHA_135_poor_DCM_all = P_ho_fldwd_SFHA_4firm_rich + P_ho_fldwd_SFHA_135_poor_DCM
    B_buy_perhurr_fldwd_SFHA_135_poor_DCM_all = B_buy_perhurr_fldwd_SFHA_4firm_rich + B_buy_perhurr_fldwd_SFHA_135_poor_DCM
    
    num_insured_fld_LR_135_poor_DCM_all = num_insured_fld_LR_4firm_rich + num_insured_fld_LR_135_poor_DCM
    num_insured_wd_LR_135_poor_DCM_all = num_insured_wd_LR_4firm_rich + num_insured_wd_LR_135_poor_DCM
    num_insured_fld_SFHA_135_poor_DCM_all = num_insured_fld_SFHA_4firm_rich + num_insured_fld_SFHA_135_poor_DCM
    num_insured_wd_SFHA_135_poor_DCM_all = num_insured_wd_SFHA_4firm_rich + num_insured_wd_SFHA_135_poor_DCM
    
    P_gini_LR = P_ho_fldwd_LR_135_poor_DCM_all
    P_gini_SFHA = P_ho_fldwd_SFHA_135_poor_DCM_all
    
    if hs.size == 0:
        B_gini_LR = np.zeros(B_buy_perhurr_fldwd_LR_135_poor_DCM_all.shape[0])
        B_gini_SFHA = np.zeros(B_buy_perhurr_fldwd_SFHA_135_poor_DCM_all.shape[0])       
    else: 
        B_gini_LR = np.sum(B_buy_perhurr_fldwd_LR_135_poor_DCM_all[:, hs], axis=1)
        B_gini_SFHA = np.sum(B_buy_perhurr_fldwd_SFHA_135_poor_DCM_all[:, hs], axis=1)

    P_gini_135_DCM = np.concatenate((P_gini_LR, P_gini_SFHA))
    B_gini_135_DCM = np.concatenate((B_gini_LR, B_gini_SFHA))

    # Calculate Luni_gini_135_DCM
    if hs.size == 0:
        Luni_gini_135_DCM = Luni_gini_noPB - temp_4firm_rich
    else:
        # Sum the specified columns along the rows (axis=1)
        sum_LR = np.sum(L_ho_buy_perhurr_fldwd_LR[:, hs], axis=1)
        sum_SFHA = np.sum(L_ho_buy_perhurr_fldwd_SFHA[:, hs], axis=1)

        # Concatenate the results vertically
        L_ho_sum = np.concatenate((sum_LR, sum_SFHA))
        Luni_gini_135_DCM = Luni_gini_noPB - L_ho_sum - temp_4firm_rich
    
    # Clear specified variables
    P_ho_fldwd_LR_135_poor_DCM_all = None
    P_ho_fldwd_SFHA_135_poor_DCM_all = None
    B_buy_perhurr_fldwd_LR_135_poor_DCM_all = None
    B_buy_perhurr_fldwd_SFHA_135_poor_DCM_all = None
    
        
    
    
    
    ########
    ## For low-income special market (price: $1.95; DCM insurance) and 4 insurers market for middle- and high-income households, calculate premiums, deductibles,  
    ## and uninsured losses for households by calling function “Func_cal_hoInsure_united.m” and based on results from “4 insurers middle- and high-income households only”
    
    # 1.95, poor, DCM, discrete choice model
    x, y, z = 0.6, index_MH, 0
    _, _, _, _, _, _, _, P_ho_fldwd_LR_135_poor_DCM, B_buy_perhurr_fldwd_LR_135_poor_DCM, num_insured_fld_LR_135_poor_DCM, num_insured_wd_LR_135_poor_DCM, L_ho_buy_perhurr_fldwd_LR,_,_,_,_ = Func_cal_hoInsure_united(x, Pr, ho_perhurrLoss_fld_LR, ho_perhurrLoss_wd_LR, ho_aveLoss_fld_LR, ho_aveLoss_wd_LR, ho_prem_fld_LR, ho_prem_wd_LR, homevalue_LR, ho_cumEXP_fld_LR_SCENYEAR, ho_lastEXP_fld_LR_SCENYEAR, insTail_fld_LR, insTail_wd_LR, y, risk_theta_LR, insurance_price_LB, deductible, hoNum_LR, hoNum_SFHA, k_LR, k_SFHA, ho_dcm, z)
    _, _, _, _, _, _, _, P_ho_fldwd_SFHA_135_poor_DCM, B_buy_perhurr_fldwd_SFHA_135_poor_DCM, num_insured_fld_SFHA_135_poor_DCM, num_insured_wd_SFHA_135_poor_DCM, L_ho_buy_perhurr_fldwd_SFHA,_,_,_,_ = Func_cal_hoInsure_united(x, Pr, ho_perhurrLoss_fld_SFHA, ho_perhurrLoss_wd_SFHA, ho_aveLoss_fld_SFHA, ho_aveLoss_wd_SFHA, ho_prem_fld_SFHA, ho_prem_wd_SFHA, homevalue_SFHA, ho_cumEXP_fld_SFHA_SCENYEAR, ho_lastEXP_fld_SFHA_SCENYEAR, insTail_fld_SFHA, insTail_wd_SFHA, y, risk_theta_SFHA, insurance_price_LB, deductible, hoNum_LR, hoNum_SFHA, k_LR, k_SFHA, ho_dcm, z)
     
    P_ho_fldwd_LR_195_poor_DCM_all = P_ho_fldwd_LR_4firm_rich + P_ho_fldwd_LR_135_poor_DCM
    B_buy_perhurr_fldwd_LR_195_poor_DCM_all = B_buy_perhurr_fldwd_LR_4firm_rich + B_buy_perhurr_fldwd_LR_135_poor_DCM
    P_ho_fldwd_SFHA_195_poor_DCM_all = P_ho_fldwd_SFHA_4firm_rich + P_ho_fldwd_SFHA_135_poor_DCM
    B_buy_perhurr_fldwd_SFHA_195_poor_DCM_all = B_buy_perhurr_fldwd_SFHA_4firm_rich + B_buy_perhurr_fldwd_SFHA_135_poor_DCM
    
    num_insured_fld_LR_195_poor_DCM_all = num_insured_fld_LR_4firm_rich + num_insured_fld_LR_135_poor_DCM
    num_insured_wd_LR_195_poor_DCM_all = num_insured_wd_LR_4firm_rich + num_insured_wd_LR_135_poor_DCM
    num_insured_fld_SFHA_195_poor_DCM_all = num_insured_fld_SFHA_4firm_rich + num_insured_fld_SFHA_135_poor_DCM
    num_insured_wd_SFHA_195_poor_DCM_all = num_insured_wd_SFHA_4firm_rich + num_insured_wd_SFHA_135_poor_DCM
    
    P_gini_LR = P_ho_fldwd_LR_195_poor_DCM_all
    P_gini_SFHA = P_ho_fldwd_SFHA_195_poor_DCM_all
    
    if hs.size == 0:
        B_gini_LR = np.zeros(B_buy_perhurr_fldwd_LR_195_poor_DCM_all.shape[0])
        B_gini_SFHA = np.zeros(B_buy_perhurr_fldwd_SFHA_195_poor_DCM_all.shape[0])
    else:
        B_gini_LR = np.sum(B_buy_perhurr_fldwd_LR_195_poor_DCM_all[:, hs], axis=1)
        B_gini_SFHA = np.sum(B_buy_perhurr_fldwd_SFHA_195_poor_DCM_all[:, hs], axis=1)
    P_gini_195_DCM = np.concatenate((P_gini_LR, P_gini_SFHA))
    B_gini_195_DCM = np.concatenate((B_gini_LR, B_gini_SFHA))
    
    # Check if hs is empty
    if hs.size == 0:
        # If hs is empty, summing along axis 0 would result in an error, so handle it differently
        L_ho_sum = np.zeros_like(Luni_gini_noPB)  # Create an array of zeros with the same shape as Luni_gini_noPB
    else:
        # Calculate sum along axis 0 for L_ho_buy_perhurr_fldwd_LR and L_ho_buy_perhurr_fldwd_SFHA
        # Sum the specified columns along the rows (axis=1)
        sum_LR = np.sum(L_ho_buy_perhurr_fldwd_LR[:, hs], axis=1)
        sum_SFHA = np.sum(L_ho_buy_perhurr_fldwd_SFHA[:, hs], axis=1)
        
        # Concatenate the results vertically
        L_ho_sum = np.concatenate((sum_LR, sum_SFHA))
    
    # Calculate Luni_gini_195_DCM
    Luni_gini_195_DCM = Luni_gini_noPB - L_ho_sum - temp_4firm_rich

    
    ########
    ## Record and return numbers of insured households in low- and high-risk zones for wind and flood insurance by all insurance market types
    # Concatenating the arrays for different scenarios
    num_insured_fld_LR = [
        num_insured_fld_LR_1firm_all,
        num_insured_fld_LR_4firm_all,
        num_insured_fld_LR_135_poor_forced_all,
        num_insured_fld_LR_135_poor_DCM_all,
        num_insured_fld_LR_195_poor_DCM_all
    ]
    
    num_insured_wd_LR = [
        num_insured_wd_LR_1firm_all,
        num_insured_wd_LR_4firm_all,
        num_insured_wd_LR_135_poor_forced_all,
        num_insured_wd_LR_135_poor_DCM_all,
        num_insured_wd_LR_195_poor_DCM_all
    ]
    
    num_insured_fld_SFHA = [
        num_insured_fld_SFHA_1firm_all,
        num_insured_fld_SFHA_4firm_all,
        num_insured_fld_SFHA_135_poor_forced_all,
        num_insured_fld_SFHA_135_poor_DCM_all,
        num_insured_fld_SFHA_195_poor_DCM_all
    ]
    
    num_insured_wd_SFHA = [
        num_insured_wd_SFHA_1firm_all,
        num_insured_wd_SFHA_4firm_all,
        num_insured_wd_SFHA_135_poor_forced_all,
        num_insured_wd_SFHA_135_poor_DCM_all,
        num_insured_wd_SFHA_195_poor_DCM_all
    ]
     
    return optimized_F_EQM_eachInsurer, optimized_A_EQM_eachInsurer, optimized_M_EQM_eachInsurer, optimized_insolvent_EQM_eachInsurer, insol_year,\
        real_P_allInsurers,real_L_allInsurers,real_B_allInsurers,real_reIns_P_eachInsurer,\
            Luni_gini_noPB,Luni_gini_noPB_wd, Luni_gini_noPB_fld, P_gini_1firm,B_gini_1firm,Luni_gini_1firm,\
                P_gini_4firms,B_gini_4firms,Luni_gini_4firms,\
                    P_gini_135_forced,B_gini_135_forced,Luni_gini_135_forced,\
                        P_gini_135_DCM,B_gini_135_DCM,Luni_gini_135_DCM,\
                            P_gini_195_DCM,B_gini_195_DCM,Luni_gini_195_DCM,\
                                num_insured_fld_LR,num_insured_wd_LR,num_insured_fld_SFHA,num_insured_wd_SFHA,num_check_poor_LR_135_DCM,num_check_poor_SFHA_135_DCM, pay_wd_lr, pay_fld_lr, pay_wd_hr, pay_fld_hr

