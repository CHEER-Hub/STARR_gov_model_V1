#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 21 16:17:41 2024

@author: Jingya Wang
"""
import numpy as np

#%%
def Func_supRand(prob):
    """
    Generate decisions based on a given probability distribution.
    
    Parameters:
    prob (np.ndarray): Array of probabilities for each decision.
    
    Returns:
    np.ndarray: Array of decisions (0 or 1) based on the given probabilities.
    """
    experiment_num = 1  # Number of experiments to run
    rec_decision = np.zeros((len(prob), experiment_num))

    for i in range(experiment_num):
        rec_decision[:, i] = np.random.rand(len(prob)) < prob

    # If there is only one experiment, return the result directly
    if experiment_num == 1:
        decision = rec_decision[:, 0]
    else:
        # Calculate the mean decision for each row
        mean_decision = np.mean(rec_decision, axis=1)

        # Get the index that would sort the mean decisions
        sorted_indices = np.argsort(mean_decision)

        # Select the median experiment based on the sorted indices
        median_experiment_index = sorted_indices[len(sorted_indices) // 2]

        # Extract the decision corresponding to the median experiment
        decision = rec_decision[:, median_experiment_index]

    return decision

#%%
def Func_cal_ret_voluntary(ho_acqRecord_LR, ho_acqRecord_SFHA, priceRet_alpha, priceRet_beta, J, annual_threshold, stru, L_au3D_fld, L_au3D_wd,
                           L_au4D_fld, L_au4D_wd, ho_aveLoss_fld_LR, ho_aveLoss_fld_SFHA, ho_aveLoss_wd_LR, ho_aveLoss_wd_SFHA,
                           ho_perhurrLoss_fld_LR, ho_perhurrLoss_fld_SFHA, ho_perhurrLoss_wd_LR, ho_perhurrLoss_wd_SFHA,
                           ho_cumEXP_fld_LR_SCENYEAR, ho_cumEXP_fld_SFHA_SCENYEAR, R_LR_new_12in1_withGrant, R_SFHA_new_12in1_withGrant,
                           hoNum_LR, hoNum_SFHA, zoneNum_LR, zoneNum_SFHA, years, fld_dcm_coeff, wd_dcm_coeff):
    """ Calculate self-retrofit decisions """
    
    ########
    ## Initialize parameters and variables
    priceRet_alpha = 0  # self ret, no grant!
    
    exclude_granted_LR = 1 - R_LR_new_12in1_withGrant[:, 8]  # R_LR_new_12in1_withGrant[:,9] ret decision with grant

    exclude_granted_SFHA = 1 - R_SFHA_new_12in1_withGrant[:, 8]
    
    priceRet_beta = 0
    J = 0

    np.random.seed(2)

    distance_LR = np.squeeze(stru['stru']['distance_LR'][0,0]) # distance to coast
    distance_SFHA = np.squeeze(stru['stru']['distance_SFHA'][0,0])
    cindex = np.squeeze(stru['stru']['cindex'][0,0]) # resistance
    #hoNum_LR = np.squeeze(stru['stru']['hoNum_LR'][0,0])
    #hoNum_SFHA = np.squeeze(stru['stru']['hoNum_SFHA'][0,0]) # employment status 0/1
    employment_LR = np.squeeze(stru['stru']['employment_LR'][0,0])
    employment_SFHA = np.squeeze(stru['stru']['employment_SFHA'][0,0])
    inv_LR_dynamic = np.squeeze(stru['stru']['inv_LR'][0,0]) # inventory
    inv_SFHA_dynamic = np.squeeze(stru['stru']['inv_SFHA'][0,0])
    retrofitcost_LR = np.squeeze(stru['stru']['retrofitcost_LR'][0,0])
    retrofitcost_SFHA = np.squeeze(stru['stru']['retrofitcost_SFHA'][0,0]) # 8 types of retrofit with price
    ho_areaID_LR = np.squeeze(stru['stru']['ho_areaID_LR'][0,0])
    ho_areaID_SFHA = np.squeeze(stru['stru']['ho_areaID_SFHA'][0,0])
    
    ho_aveLoss_fld_LR = ho_aveLoss_fld_LR.reshape(-1,1)
    ho_aveLoss_fld_SFHA = ho_aveLoss_fld_SFHA.reshape(-1,1)
    temp_ho_aveLoss_fldwd_LR = ho_aveLoss_fld_LR + ho_aveLoss_wd_LR
    temp_ho_aveLoss_fldwd_SFHA = ho_aveLoss_fld_SFHA + ho_aveLoss_wd_SFHA

    ho_aveLoss_fld_LR_dynamic = ho_aveLoss_fld_LR
    ho_aveLoss_fld_SFHA_dynamic = ho_aveLoss_fld_SFHA
    ho_aveLoss_wd_LR_dynamic = ho_aveLoss_wd_LR
    ho_aveLoss_wd_SFHA_dynamic = ho_aveLoss_wd_SFHA
    
    ########
    ##•	Calculate zone level retrofit loading factors (“area_priceRet_loadingFactor_LR” and “area_priceRet_loadingFactor_SFHA”) and zone level retrofit prices (“ho_priceRet_LR” and “ho_priceRet_SFHA”)
    ##•	Since this is self-retrofit, zone level retrofit prices are set to 0
    # Define the size of arrays
    area_priceRet_loadingFactor_LR = np.zeros(zoneNum_LR)
    ho_priceRet_LR = np.zeros(hoNum_LR)
    area_priceRet_loadingFactor_SFHA = np.zeros(zoneNum_SFHA)
    ho_priceRet_SFHA = np.zeros(hoNum_SFHA)
    
    # Calculate loading factors for LR areas
    for i in range(zoneNum_LR):
        temp = (ho_areaID_LR == (i+1)) & (ho_acqRecord_LR == 0)
        area_priceRet_loadingFactor_LR[i] = np.mean(temp_ho_aveLoss_fldwd_LR[temp]) # sum P^h L^h * X (num) / sum X (num) = mean
    area_priceRet_loadingFactor_LR[np.isnan(area_priceRet_loadingFactor_LR)] = 0 # area no building set to 0; 0/0=NaN
    area_priceRet_loadingFactor_LR /= 1e4
    
    # Assign price retrofit values for LR areas
    for i in range(zoneNum_LR):
        temp = ho_areaID_LR == (i+1)
        ho_priceRet_LR[temp] = priceRet_alpha + priceRet_beta * area_priceRet_loadingFactor_LR[i] # c_base+c_prop*loading factor
    
    # Calculate loading factors for SFHA areas
    for i in range(zoneNum_SFHA):
        temp = (ho_areaID_SFHA == (i + zoneNum_LR + 1)) & (ho_acqRecord_SFHA == 0)
        area_priceRet_loadingFactor_SFHA[i] = np.mean(temp_ho_aveLoss_fldwd_SFHA[temp])
    
    area_priceRet_loadingFactor_SFHA[np.isnan(area_priceRet_loadingFactor_SFHA)] = 0
    area_priceRet_loadingFactor_SFHA /= 1e4
    
    # Assign price retrofit values for SFHA areas
    for i in range(zoneNum_SFHA):
        temp = ho_areaID_SFHA == (i + zoneNum_LR + 1)
        ho_priceRet_SFHA[temp] = priceRet_alpha + priceRet_beta * area_priceRet_loadingFactor_SFHA[i]
        

        
    ########
    ## Calculate households’ flood-related self-retrofit decisions through DCM. There are 3 flood-related retrofits: 
    ##(i) elevating house appliances (“R_ap_fld_LR” and “R_ap_fld_SFHA”); (ii) installing water resistant (“R_in_fld_LR” and “R_in_fld_SFHA”); 
    ##(iii) elevating the entire house (“R_el_fld_LR” and “R_el_fld_SFHA”)
    
    logical_inv_LR_dynamic_9 = inv_LR_dynamic[:, 8] == 1  # position 9 == 1
    p_apin_fld_LR = np.exp(fld_dcm_coeff[0] * ho_priceRet_LR + fld_dcm_coeff[1] * (ho_cumEXP_fld_LR_SCENYEAR > 1) + fld_dcm_coeff[2] * distance_LR)  # DCM; calculate prob distribution
    prob_apin_fld_LR = p_apin_fld_LR / (1 + p_apin_fld_LR)  # average 0.4807; regularize to range (0,1)
    R_apin_fld_LR = Func_supRand(prob_apin_fld_LR) * logical_inv_LR_dynamic_9  # decision of retrofit "ap" "in"
    R_ap_fld_LR = Func_supRand(fld_dcm_coeff[3] * np.ones(hoNum_LR)) * R_apin_fld_LR  # HO has 0.2 prob to do "ap" when they decide to retrofit "ap" "in"
    R_in_fld_LR = (R_ap_fld_LR == 0) * R_apin_fld_LR  # 0.8 "in"
    p_el_fld_LR = np.exp(fld_dcm_coeff[4] * ho_priceRet_LR + fld_dcm_coeff[5] * J * fld_dcm_coeff[6] + fld_dcm_coeff[7] * (ho_cumEXP_fld_LR_SCENYEAR > 1) + fld_dcm_coeff[8] * distance_LR)  # DCM
    prob_el_fld_LR = p_el_fld_LR / (1 + p_el_fld_LR)  # average 0.8727
    R_el_fld_LR = Func_supRand(prob_el_fld_LR) * logical_inv_LR_dynamic_9  # decision "el"
    
    # SFHA
    logical_inv_SFHA_dynamic_9 = inv_SFHA_dynamic[:, 8] == 1
    p_apin_fld_SFHA = np.exp(fld_dcm_coeff[0] * ho_priceRet_SFHA + fld_dcm_coeff[1] * (ho_cumEXP_fld_SFHA_SCENYEAR > 1) + fld_dcm_coeff[2] * distance_SFHA)  # DCM
    prob_apin_fld_SFHA = p_apin_fld_SFHA / (1 + p_apin_fld_SFHA)  # average 0.6571; distance closer
    R_apin_fld_SFHA = Func_supRand(prob_apin_fld_SFHA) * logical_inv_SFHA_dynamic_9  # "ap" "in"
    R_ap_fld_SFHA = Func_supRand(fld_dcm_coeff[3] * np.ones(hoNum_SFHA)) * R_apin_fld_SFHA  # 0.2 "ap"
    R_in_fld_SFHA = (R_ap_fld_SFHA == 0) * R_apin_fld_SFHA  # 0.8 "in"
    p_el_fld_SFHA = np.exp(fld_dcm_coeff[4] * ho_priceRet_SFHA + fld_dcm_coeff[5] * J * fld_dcm_coeff[6] + fld_dcm_coeff[7] * (ho_cumEXP_fld_SFHA_SCENYEAR > 1) + fld_dcm_coeff[8] * distance_SFHA)  # DCM
    prob_el_fld_SFHA = p_el_fld_SFHA / (1 + p_el_fld_SFHA)  # average 0.9873
    R_el_fld_SFHA = Func_supRand(prob_el_fld_SFHA) * logical_inv_SFHA_dynamic_9  # "el"
    
    logical_ho_aveLoss_fld_LR_dynamic_positive = ho_aveLoss_fld_LR_dynamic > 0  # retrofit if average flood loss > 0
    R_ap_fld_LR = R_ap_fld_LR * np.squeeze(logical_ho_aveLoss_fld_LR_dynamic_positive)
    R_in_fld_LR = R_in_fld_LR * np.squeeze(logical_ho_aveLoss_fld_LR_dynamic_positive)
    R_el_fld_LR = R_el_fld_LR * np.squeeze(logical_ho_aveLoss_fld_LR_dynamic_positive)
    logical_ho_aveLoss_fld_SFHA_dynamic_positive = ho_aveLoss_fld_SFHA_dynamic > 0
    R_ap_fld_SFHA = R_ap_fld_SFHA * np.squeeze(logical_ho_aveLoss_fld_SFHA_dynamic_positive)
    R_in_fld_SFHA = R_in_fld_SFHA * np.squeeze(logical_ho_aveLoss_fld_SFHA_dynamic_positive)
    R_el_fld_SFHA = R_el_fld_SFHA * np.squeeze(logical_ho_aveLoss_fld_SFHA_dynamic_positive)
    
    ########
    ## Calculate households’ wind-related self-retrofit decisions through DCM. There are 5 wind-related retrofits: 
    ##(i) reinforcing roof with high wind load shingles (“R_roof_sh_LR” and “R_roof_sh_SFHA”); (ii) reinforcing roof with adhesive foam (“R_roof_ad_LR” and “R_roof_ad_SFHA”); 
    ##(iii) strengthening openings with shutters (“R_openings_sh_LR” and “R_openings_sh_SFHA”); (iv) strengthening openings with impact‐resistant windows (“R_openings_ir_LR” and “R_openings_ir_SFHA”); 
    ##(v) strengthening roof‐to‐wall connection using straps (“R_rtw_st_LR” and “R_rtw_st_SFHA”)
    # Wind DCM for LR (Low Risk)
    
    # Roof LR 'sh' 'ad'
    p_roof_LR = np.exp(wd_dcm_coeff[0] * ho_priceRet_LR + wd_dcm_coeff[1] * J * wd_dcm_coeff[2] + wd_dcm_coeff[3] * (ho_cumEXP_fld_LR_SCENYEAR > 1) + wd_dcm_coeff[4] * distance_LR + wd_dcm_coeff[5] * employment_LR) # DCM
    prob_roof_LR = p_roof_LR / (1 + p_roof_LR)
    R_roof_LR = Func_supRand(prob_roof_LR) * (inv_LR_dynamic[:,4] == 1)
    R_roof_sh_LR = R_roof_LR * (inv_LR_dynamic[:,3] == 1)
    R_roof_ad_LR = Func_supRand(wd_dcm_coeff[6] * np.ones(hoNum_LR)) * (R_roof_sh_LR == 0) * R_roof_LR
    
    # Openings LR
    p_openings_LR = np.exp(wd_dcm_coeff[7] + wd_dcm_coeff[8] * ho_priceRet_LR + wd_dcm_coeff[9] * J * wd_dcm_coeff[2] + wd_dcm_coeff[10] * distance_LR) # DCM
    prob_openings_LR = p_openings_LR / (1 + p_openings_LR)
    R_openings_LR = Func_supRand(prob_openings_LR) * (inv_LR_dynamic[:,6] == 1) * (inv_LR_dynamic[:,4] == 2)
    R_openings_sh_LR = Func_supRand(wd_dcm_coeff[11] * np.ones(hoNum_LR)) * R_openings_LR
    R_openings_ir_LR = R_openings_LR * (R_openings_sh_LR == 0)
    
    # Roof-to-wall LR
    p_rtw_st_LR = np.exp(wd_dcm_coeff[12] * ho_priceRet_LR + wd_dcm_coeff[13] * J * wd_dcm_coeff[2] + wd_dcm_coeff[14] * (ho_cumEXP_fld_LR_SCENYEAR > 1) + wd_dcm_coeff[15] * distance_LR) # DCM
    prob_rtw_st_LR = p_rtw_st_LR / (1 + p_rtw_st_LR)
    R_rtw_st_LR = Func_supRand(prob_rtw_st_LR) * (inv_LR_dynamic[:,5] == 1) * (inv_LR_dynamic[:,4] == 2) * (inv_LR_dynamic[:,6] > 1)
    
    # Roof SFHA
    p_roof_SFHA = np.exp(wd_dcm_coeff[0] * ho_priceRet_SFHA + wd_dcm_coeff[1] * J * wd_dcm_coeff[2] + wd_dcm_coeff[3] * (ho_cumEXP_fld_SFHA_SCENYEAR > 1) + wd_dcm_coeff[4] * distance_SFHA + wd_dcm_coeff[5] * employment_SFHA)
    prob_roof_SFHA = p_roof_SFHA / (1 + p_roof_SFHA)
    R_roof_SFHA = Func_supRand(prob_roof_SFHA) * (inv_SFHA_dynamic[:,4] == 1)
    R_roof_sh_SFHA = R_roof_SFHA * (inv_SFHA_dynamic[:,3] == 1)
    R_roof_ad_SFHA = Func_supRand(wd_dcm_coeff[6] * np.ones(hoNum_SFHA)) * (R_roof_sh_SFHA == 0) * R_roof_SFHA
    
    # Openings SFHA
    p_openings_SFHA = np.exp(wd_dcm_coeff[7] + wd_dcm_coeff[8] * ho_priceRet_SFHA + wd_dcm_coeff[9] * J * wd_dcm_coeff[2] + wd_dcm_coeff[10] * distance_SFHA)
    prob_openings_SFHA = p_openings_SFHA / (1 + p_openings_SFHA)
    R_openings_SFHA = Func_supRand(prob_openings_SFHA) * (inv_SFHA_dynamic[:,6] == 1) * (inv_SFHA_dynamic[:,4] == 2)
    R_openings_sh_SFHA = Func_supRand(wd_dcm_coeff[11] * np.ones(hoNum_SFHA)) * R_openings_SFHA
    R_openings_ir_SFHA = R_openings_SFHA * (R_openings_sh_SFHA == 0)
    
    # Roof-to-wall SFHA
    p_rtw_st_SFHA = np.exp(wd_dcm_coeff[12] * ho_priceRet_SFHA + wd_dcm_coeff[13] * J * wd_dcm_coeff[2] + wd_dcm_coeff[14] * (ho_cumEXP_fld_SFHA_SCENYEAR > 1) + wd_dcm_coeff[15] * distance_SFHA)
    prob_rtw_st_SFHA = p_rtw_st_SFHA / (1 + p_rtw_st_SFHA)
    R_rtw_st_SFHA = Func_supRand(prob_rtw_st_SFHA) * (inv_SFHA_dynamic[:,5] == 1) * (inv_SFHA_dynamic[:,4] == 2) * (inv_SFHA_dynamic[:,6] > 1)

    ########
    ## Randomly pick 1 retrofit for households, if they have multiple retrofit plans, meaning households try to do more than 1 type (out of 8) of retrofits.
    ## Update households’ retrofit decisions
    # Randomly choose only one retrofit for LR (Low Risk)
    randRetrofit_LR = np.column_stack([R_ap_fld_LR, R_in_fld_LR, R_el_fld_LR, R_roof_sh_LR, R_roof_ad_LR, R_openings_sh_LR, R_openings_ir_LR, R_rtw_st_LR])
    sum_randRetrofit_LR = np.sum(randRetrofit_LR, axis=1)
    rand_LR8 = np.random.random((8, hoNum_LR)).T
    for i in range(hoNum_LR):
        if sum_randRetrofit_LR[i] > 1:
            temp = randRetrofit_LR[i, :] * rand_LR8[i, :]
            loc = np.argmax(temp)
            randRetrofit_LR[i, :] = 0
            randRetrofit_LR[i, loc] = 1
            sum_randRetrofit_LR[i] = 1
    
    R_ap_fld_LR = randRetrofit_LR[:, 0]
    R_in_fld_LR = randRetrofit_LR[:, 1]
    R_el_fld_LR = randRetrofit_LR[:, 2]
    R_roof_sh_LR = randRetrofit_LR[:, 3]
    R_roof_ad_LR = randRetrofit_LR[:, 4]
    R_openings_sh_LR = randRetrofit_LR[:, 5]
    R_openings_ir_LR = randRetrofit_LR[:, 6]
    R_rtw_st_LR = randRetrofit_LR[:, 7]
    R_binary_LR_old = sum_randRetrofit_LR
    
    randRetrofit_SFHA = np.column_stack((R_ap_fld_SFHA, R_in_fld_SFHA, R_el_fld_SFHA, R_roof_sh_SFHA, R_roof_ad_SFHA, R_openings_sh_SFHA, R_openings_ir_SFHA, R_rtw_st_SFHA))
    sum_randRetrofit_SFHA = np.sum(randRetrofit_SFHA, axis=1)
    rand_SFHA8 = np.random.random((8, hoNum_SFHA)).T
    for i in range(hoNum_SFHA):
        if sum_randRetrofit_SFHA[i] > 1:
            temp = randRetrofit_SFHA[i, :] * rand_SFHA8[i, :]
            loc = np.argmax(temp)
            randRetrofit_SFHA[i, :] = 0
            randRetrofit_SFHA[i, loc] = 1
            sum_randRetrofit_SFHA[i] = 1
    
    R_ap_fld_SFHA = randRetrofit_SFHA[:, 0]
    R_in_fld_SFHA = randRetrofit_SFHA[:, 1]
    R_el_fld_SFHA = randRetrofit_SFHA[:, 2]
    R_roof_sh_SFHA = randRetrofit_SFHA[:, 3]
    R_roof_ad_SFHA = randRetrofit_SFHA[:, 4]
    R_openings_sh_SFHA = randRetrofit_SFHA[:, 5]
    R_openings_ir_SFHA = randRetrofit_SFHA[:, 6]
    R_rtw_st_SFHA = randRetrofit_SFHA[:, 7]
    R_binary_SFHA_old = sum_randRetrofit_SFHA
    
    ########
    ## •	Update costs for retrofits based on retrofit decisions
    # Calculate retrofit costs for LR (Low Risk)
    cost_ap_fld_LR = R_ap_fld_LR * retrofitcost_LR[:, 5]  # "ap" 6
    cost_in_fld_LR = R_in_fld_LR * retrofitcost_LR[:, 6]  # "in" 7
    cost_el_fld_LR = R_el_fld_LR * retrofitcost_LR[:, 7]  # "el" 8
    cost_ap_fld_SFHA = R_ap_fld_SFHA * retrofitcost_SFHA[:, 5]
    cost_in_fld_SFHA = R_in_fld_SFHA * retrofitcost_SFHA[:, 6]
    cost_el_fld_SFHA = R_el_fld_SFHA * retrofitcost_SFHA[:, 7]
    
    cost_roof_sh_LR = R_roof_sh_LR * retrofitcost_LR[:, 0]  # "sh" 1
    cost_roof_ad_LR = R_roof_ad_LR * retrofitcost_LR[:, 1]  # "ad" 2
    cost_roof_sh_SFHA = R_roof_sh_SFHA * retrofitcost_SFHA[:, 0]
    cost_roof_ad_SFHA = R_roof_ad_SFHA * retrofitcost_SFHA[:, 1]
    
    cost_openings_sh_LR = R_openings_sh_LR * retrofitcost_LR[:, 2]  # "sh" (opening) 3
    cost_openings_ir_LR = R_openings_ir_LR * retrofitcost_LR[:, 3]  # "ir" 4
    cost_openings_sh_SFHA = R_openings_sh_SFHA * retrofitcost_SFHA[:, 2]
    cost_openings_ir_SFHA = R_openings_ir_SFHA * retrofitcost_SFHA[:, 3]
    
    cost_rtw_st_LR = R_rtw_st_LR * retrofitcost_LR[:, 4]  # "rtw" 5
    cost_rtw_st_SFHA = R_rtw_st_SFHA * retrofitcost_SFHA[:, 4]
    
    ########
    ## Update resistance levels and resistance details in building inventory based on retrofit decisions. 
    ## As retrofits strengthening buildings, the resistance levels and details should be updated based on different retrofit options
    temp_inv_LR_dynamic3 = inv_LR_dynamic[:, 2].copy()  # c index
    for i in range(hoNum_LR):
        if R_binary_LR_old[i]:  # do retrofit
            integrated_temp_LR = [
                inv_LR_dynamic[i, 3] + inv_LR_dynamic[i, 3] * R_roof_sh_LR[i],
                inv_LR_dynamic[i, 4] + inv_LR_dynamic[i, 4] * R_roof_ad_LR[i] + inv_LR_dynamic[i, 4] * R_roof_sh_LR[i],
                inv_LR_dynamic[i, 5] + inv_LR_dynamic[i, 5] * R_rtw_st_LR[i],
                inv_LR_dynamic[i, 6] + 2 * R_openings_sh_LR[i] * inv_LR_dynamic[i, 6] + R_openings_ir_LR[i] * inv_LR_dynamic[i, 6],
                inv_LR_dynamic[i, 7],
                inv_LR_dynamic[i, 8] + inv_LR_dynamic[i, 8] * R_ap_fld_LR[i] + inv_LR_dynamic[i, 8] * R_in_fld_LR[i] * 2 + inv_LR_dynamic[i, 8] * R_el_fld_LR[i] * 3
            ]  # 1*6; [4,5,6,7,8,9] columns in inv
            # sh: pos4+1, pos5+1; ad: pos5+1; rtw: pos6+1; sh (open): pos7+2;
            # ir: pos7+1; ap: pos9+1; in: pos9+2; el: pos9+3
            temp = cindex[:, 1:7] - np.tile(integrated_temp_LR, (192, 1))  # 192*6
            temp = np.sum(np.abs(temp), axis=1)  # Y = abs(X) returns the absolute value of each element in array X.
            # sum dim 2, 192*1
            temp_inv_LR_dynamic3[i] = cindex[temp == 0, 0]  # locate index c. based on current c level, find the index.
            # update c
    
    temp_inv_SFHA_dynamic3 = inv_SFHA_dynamic[:, 2].copy()
    for i in range(hoNum_SFHA):
        if R_binary_SFHA_old[i]:
            integrated_temp_SFHA = [
                inv_SFHA_dynamic[i, 3] + inv_SFHA_dynamic[i, 3] * R_roof_sh_SFHA[i],
                inv_SFHA_dynamic[i, 4] + inv_SFHA_dynamic[i, 4] * R_roof_ad_SFHA[i] + inv_SFHA_dynamic[i, 4] * R_roof_sh_SFHA[i],
                inv_SFHA_dynamic[i, 5] + inv_SFHA_dynamic[i, 5] * R_rtw_st_SFHA[i],
                inv_SFHA_dynamic[i, 6] + 2 * R_openings_sh_SFHA[i] * inv_SFHA_dynamic[i, 6] + R_openings_ir_SFHA[i] * inv_SFHA_dynamic[i, 6],
                inv_SFHA_dynamic[i, 7],
                inv_SFHA_dynamic[i, 8] + inv_SFHA_dynamic[i, 8] * R_ap_fld_SFHA[i] + inv_SFHA_dynamic[i, 8] * R_in_fld_SFHA[i] * 2 + inv_SFHA_dynamic[i, 8] * R_el_fld_SFHA[i] * 3
            ]
            temp = cindex[:, 1:7] - np.tile(integrated_temp_SFHA, (192, 1))
            temp = np.sum(np.abs(temp), axis=1)
            temp_inv_SFHA_dynamic3[i] = cindex[temp == 0, 0]
            
    ########
    ## Update households’ expected losses based on retrofit decisions
    homevalue_LR = np.squeeze(stru['stru']['homevalue_LR'][0,0])
    homevalue_SFHA = np.squeeze(stru['stru']['homevalue_SFHA'][0,0])
    temp_updated_aveLoss_fldwd_LR = ho_aveLoss_fld_LR_dynamic + ho_aveLoss_wd_LR_dynamic
    for i in range(hoNum_LR):

        if R_binary_LR_old[i]:  # do retrofit
            temp_updated_aveLoss_fldwd_LR[i] = L_au3D_fld[inv_LR_dynamic[i, 0]-1, inv_LR_dynamic[i, 1]-1, temp_inv_LR_dynamic3[i]-1] * homevalue_LR[i] \
                                                + L_au3D_wd[inv_LR_dynamic[i, 0]-1, inv_LR_dynamic[i, 1]-1, temp_inv_LR_dynamic3[i]-1] * homevalue_LR[i]
    
    temp_updated_aveLoss_fldwd_SFHA = ho_aveLoss_fld_SFHA_dynamic + ho_aveLoss_wd_SFHA_dynamic
    for i in range(hoNum_SFHA):

        if R_binary_SFHA_old[i]:
            temp_updated_aveLoss_fldwd_SFHA[i] = L_au3D_fld[inv_SFHA_dynamic[i, 0]-1 + zoneNum_LR, inv_SFHA_dynamic[i, 1]-1, temp_inv_SFHA_dynamic3[i]-1] * homevalue_SFHA[i] \
                                                  + L_au3D_wd[inv_SFHA_dynamic[i, 0]-1 + zoneNum_LR, inv_SFHA_dynamic[i, 1]-1, temp_inv_SFHA_dynamic3[i]-1] * homevalue_SFHA[i]

    ########
    ## •	Calculate households’ benefits (loss reduction) after retrofitting
    ##•	Calculate total retrofit costs (sum all 8 types of retrofits)
    temp_reduced_aveLoss_fldwd_LR = np.squeeze(ho_aveLoss_fld_LR_dynamic + ho_aveLoss_wd_LR_dynamic - temp_updated_aveLoss_fldwd_LR)  # loss benefit
    temp_reduced_aveLoss_fldwd_SFHA = np.squeeze(ho_aveLoss_fld_SFHA_dynamic + ho_aveLoss_wd_SFHA_dynamic - temp_updated_aveLoss_fldwd_SFHA)
    
    cost_binaryR_LR = cost_ap_fld_LR + cost_in_fld_LR + cost_el_fld_LR + \
        cost_roof_sh_LR + cost_roof_ad_LR + \
        cost_openings_sh_LR + cost_openings_ir_LR + \
        cost_rtw_st_LR
    
    cost_binaryR_SFHA = cost_ap_fld_SFHA + cost_in_fld_SFHA + cost_el_fld_SFHA + \
        cost_roof_sh_SFHA + cost_roof_ad_SFHA + \
        cost_openings_sh_SFHA + cost_openings_ir_SFHA + \
        cost_rtw_st_SFHA

    ########
    ## •Update households’ benefits by considering self-retrofit costs
    ##•	Update households’ self-retrofit decisions. Do retrofit if their benefits are greater than “annual_threshold”; otherwise, decline retrofit
    ##•	Update benefits and retrofit costs based updated retrofit decisions.
    temp_benefit_fldwd_LR_withGrant = temp_reduced_aveLoss_fldwd_LR - cost_binaryR_LR
    temp_benefit_fldwd_SFHA_withGrant = temp_reduced_aveLoss_fldwd_SFHA - cost_binaryR_SFHA
    
    logical_withGrant_LR = (temp_benefit_fldwd_LR_withGrant >= annual_threshold)
    logical_withGrant_SFHA = (temp_benefit_fldwd_SFHA_withGrant >= annual_threshold)
    
    R_ap_fld_LR_withGrant = R_ap_fld_LR * logical_withGrant_LR
    R_in_fld_LR_withGrant = R_in_fld_LR * logical_withGrant_LR
    R_el_fld_LR_withGrant = R_el_fld_LR * logical_withGrant_LR
    R_roof_sh_LR_withGrant = R_roof_sh_LR * logical_withGrant_LR
    R_roof_ad_LR_withGrant = R_roof_ad_LR * logical_withGrant_LR
    R_openings_sh_LR_withGrant = R_openings_sh_LR * logical_withGrant_LR
    R_openings_ir_LR_withGrant = R_openings_ir_LR * logical_withGrant_LR
    R_rtw_st_LR_withGrant = R_rtw_st_LR * logical_withGrant_LR
    temp_reduced_aveLoss_fldwd_LR = temp_reduced_aveLoss_fldwd_LR * logical_withGrant_LR
    cost_binaryR_LR = cost_binaryR_LR * logical_withGrant_LR
    
    R_ap_fld_SFHA_withGrant = R_ap_fld_SFHA * logical_withGrant_SFHA
    R_in_fld_SFHA_withGrant = R_in_fld_SFHA * logical_withGrant_SFHA
    R_el_fld_SFHA_withGrant = R_el_fld_SFHA * logical_withGrant_SFHA
    R_roof_sh_SFHA_withGrant = R_roof_sh_SFHA * logical_withGrant_SFHA
    R_roof_ad_SFHA_withGrant = R_roof_ad_SFHA * logical_withGrant_SFHA
    R_openings_sh_SFHA_withGrant = R_openings_sh_SFHA * logical_withGrant_SFHA
    R_openings_ir_SFHA_withGrant = R_openings_ir_SFHA * logical_withGrant_SFHA
    R_rtw_st_SFHA_withGrant = R_rtw_st_SFHA * logical_withGrant_SFHA
    temp_reduced_aveLoss_fldwd_SFHA = temp_reduced_aveLoss_fldwd_SFHA * logical_withGrant_SFHA
    cost_binaryR_SFHA = cost_binaryR_SFHA * logical_withGrant_SFHA
    
    R_binary_LR_new_withGrant = R_binary_LR_old * logical_withGrant_LR
    R_binary_SFHA_new_withGrant = R_binary_SFHA_old * logical_withGrant_SFHA
    


    ########
    ## •Record self-retrofit decisions using variables “R_LR_new_12in1_withGrant” and “R_SFHA_new_12in1_withGrant”
    ##•	Exclude households who have accepted retrofit offers this year
    ##•	Return self-retrofit decisions
    R_LR_new_12in1_withGrant = np.column_stack([
    R_ap_fld_LR_withGrant, R_in_fld_LR_withGrant, R_el_fld_LR_withGrant,
    R_roof_sh_LR_withGrant, R_roof_ad_LR_withGrant,
    R_openings_sh_LR_withGrant, R_openings_ir_LR_withGrant, R_rtw_st_LR_withGrant,
    R_binary_LR_new_withGrant,
    temp_reduced_aveLoss_fldwd_LR * R_binary_LR_new_withGrant * years, 
    cost_binaryR_LR * R_binary_LR_new_withGrant * years, 
    np.zeros_like(R_binary_LR_new_withGrant) * years
    ])
    
    R_SFHA_new_12in1_withGrant = np.column_stack([
        R_ap_fld_SFHA_withGrant, R_in_fld_SFHA_withGrant, R_el_fld_SFHA_withGrant,
        R_roof_sh_SFHA_withGrant, R_roof_ad_SFHA_withGrant,
        R_openings_sh_SFHA_withGrant, R_openings_ir_SFHA_withGrant, R_rtw_st_SFHA_withGrant,
        R_binary_SFHA_new_withGrant,
        temp_reduced_aveLoss_fldwd_SFHA * R_binary_SFHA_new_withGrant * years, 
        cost_binaryR_SFHA * R_binary_SFHA_new_withGrant * years, 
        np.zeros_like(R_binary_SFHA_new_withGrant) * years
    ])
    

    # Exclude granted retrofit decisions
    R_LR_new_12in1_withGrant = R_LR_new_12in1_withGrant * exclude_granted_LR.reshape(-1,1)
    R_SFHA_new_12in1_withGrant = R_SFHA_new_12in1_withGrant * exclude_granted_SFHA.reshape(-1,1)
    
        
    
    ########
    ## •Initialize parameters and variables for implementing
    ##•	Retrieve 8 types of households’ self-retrofit decisions
    acquisition_flag_LR = (ho_acqRecord_LR == 0)
    acquisition_flag_SFHA = (ho_acqRecord_SFHA == 0)
    
    # Updating retrofit values
    R_LR_new_12in1_withGrant = R_LR_new_12in1_withGrant * acquisition_flag_LR[:, np.newaxis]  # Element-wise multiplication
    R_SFHA_new_12in1_withGrant = R_SFHA_new_12in1_withGrant * acquisition_flag_SFHA[:, np.newaxis]
    
    # Summing the retrofit values
    sum_R_LR_12in1_voluntary = np.sum(R_LR_new_12in1_withGrant, axis=0)  # Sum along the first dimension
    sum_R_SFHA_12in1_voluntary = np.sum(R_SFHA_new_12in1_withGrant, axis=0)
    
    # Updating average loss values
    ho_aveLoss_fld_LR = (np.squeeze(ho_aveLoss_fld_LR) * acquisition_flag_LR).reshape(-1,1)
    ho_aveLoss_fld_SFHA = (np.squeeze(ho_aveLoss_fld_SFHA) * acquisition_flag_SFHA).reshape(-1,1)
    ho_perhurrLoss_fld_LR = np.squeeze(ho_perhurrLoss_fld_LR )* acquisition_flag_LR[:, np.newaxis]
    ho_perhurrLoss_fld_SFHA = np.squeeze(ho_perhurrLoss_fld_SFHA) * acquisition_flag_SFHA[:, np.newaxis]

    
    # The part involving grants has already been filtered out, so this is the final list of people who will perform the retrofit.
    R_1to9_LR = R_LR_new_12in1_withGrant[:, :9] > 0  # num HO*9
    R_roof_sh_LR = R_1to9_LR[:, 3]  # Column indices in Python start from 0
    R_roof_ad_LR = R_1to9_LR[:, 4]
    R_rtw_st_LR = R_1to9_LR[:, 7]
    R_openings_sh_LR = R_1to9_LR[:, 5]
    R_openings_ir_LR = R_1to9_LR[:, 6]
    R_ap_fld_LR = R_1to9_LR[:, 0]
    R_in_fld_LR = R_1to9_LR[:, 1]
    R_el_fld_LR = R_1to9_LR[:, 2]
    R_binary_LR_new = R_1to9_LR[:, 8]
    
    R_1to9_SFHA = R_SFHA_new_12in1_withGrant[:, :9] > 0
    R_roof_sh_SFHA = R_1to9_SFHA[:, 3]
    R_roof_ad_SFHA = R_1to9_SFHA[:, 4]
    R_rtw_st_SFHA = R_1to9_SFHA[:, 7]
    R_openings_sh_SFHA = R_1to9_SFHA[:, 5]
    R_openings_ir_SFHA = R_1to9_SFHA[:, 6]
    R_ap_fld_SFHA = R_1to9_SFHA[:, 0]
    R_in_fld_SFHA = R_1to9_SFHA[:, 1]
    R_el_fld_SFHA = R_1to9_SFHA[:, 2]
    R_binary_SFHA_new = R_1to9_SFHA[:, 8]


    ########
    ## Update resistance levels and resistance details in building inventory based on households’ self-retrofit decisions
    
    inv_LR_dynamic[:, 3] += inv_LR_dynamic[:, 3] * R_roof_sh_LR  # 4th column in MATLAB is 3rd index in Python
    inv_LR_dynamic[:, 4] += inv_LR_dynamic[:, 4] * R_roof_ad_LR + inv_LR_dynamic[:, 4] * R_roof_sh_LR  # 5th column
    inv_LR_dynamic[:, 5] += inv_LR_dynamic[:, 5] * R_rtw_st_LR  # 6th column
    inv_LR_dynamic[:, 6] += inv_LR_dynamic[:, 6] * R_openings_sh_LR * 2 + inv_LR_dynamic[:, 6] * R_openings_ir_LR  # 7th column
    inv_LR_dynamic[:, 8] += (inv_LR_dynamic[:, 8] * R_ap_fld_LR +
                             inv_LR_dynamic[:, 8] * R_in_fld_LR * 2 +
                             inv_LR_dynamic[:, 8] * R_el_fld_LR * 3)  # 9th column
    
    # Update inv_SFHA_dynamic for high-risk households
    inv_SFHA_dynamic[:, 3] += inv_SFHA_dynamic[:, 3] * R_roof_sh_SFHA  # 4th column in MATLAB is 3rd index in Python
    inv_SFHA_dynamic[:, 4] += inv_SFHA_dynamic[:, 4] * R_roof_ad_SFHA + inv_SFHA_dynamic[:, 4] * R_roof_sh_SFHA  # 5th column
    inv_SFHA_dynamic[:, 5] += inv_SFHA_dynamic[:, 5] * R_rtw_st_SFHA  # 6th column
    inv_SFHA_dynamic[:, 6] += inv_SFHA_dynamic[:, 6] * R_openings_sh_SFHA * 2 + inv_SFHA_dynamic[:, 6] * R_openings_ir_SFHA  # 7th column
    inv_SFHA_dynamic[:, 8] += (inv_SFHA_dynamic[:, 8] * R_ap_fld_SFHA +
                               inv_SFHA_dynamic[:, 8] * R_in_fld_SFHA * 2 +
                               inv_SFHA_dynamic[:, 8] * R_el_fld_SFHA * 3)  # 9th column
    
    # Update resistance index after retrofits, updating the 3rd column
    temp_inv_LR_dynamic3 = temp_inv_LR_dynamic3 * R_binary_LR_new
    temp_inv_SFHA_dynamic3 = temp_inv_SFHA_dynamic3 * R_binary_SFHA_new
    
    inv_LR_dynamic[temp_inv_LR_dynamic3 != 0, 2] = temp_inv_LR_dynamic3[temp_inv_LR_dynamic3 != 0]
    inv_SFHA_dynamic[temp_inv_SFHA_dynamic3 != 0, 2] = temp_inv_SFHA_dynamic3[temp_inv_SFHA_dynamic3 != 0]

    
    ########
    ##•	Update households’ expected losses and losses per hurricane based on households’ self-retrofit decisions
    updated_ho_aveLoss_fld_LR_dynamic = ho_aveLoss_fld_LR.copy()
    updated_ho_aveLoss_wd_LR_dynamic = ho_aveLoss_wd_LR.copy()
    updated_ho_perhurrLoss_fld_LR_dynamic = ho_perhurrLoss_fld_LR.copy()
    updated_ho_perhurrLoss_wd_LR_dynamic = ho_perhurrLoss_wd_LR.copy()
    temp_collect_LR = []

    #homevalue_LR = np.squeeze(stru['stru']['homevalue_LR'][0,0])
    #homevalue_SFHA = np.squeeze(stru['stru']['homevalue_SFHA'][0,0])
    
    for i in range(hoNum_LR):
        if R_binary_LR_new[i] > 0:
            updated_ho_aveLoss_fld_LR_dynamic[i] = L_au3D_fld[inv_LR_dynamic[i, 0]-1, inv_LR_dynamic[i, 1]-1, inv_LR_dynamic[i, 2]-1] * homevalue_LR[i]
                                                
            updated_ho_perhurrLoss_fld_LR_dynamic[i,:] = np.squeeze(L_au4D_fld[inv_LR_dynamic[i,0]-1, inv_LR_dynamic[i,1]-1, inv_LR_dynamic[i,2]-1,:]) * homevalue_LR[i]

            updated_ho_aveLoss_wd_LR_dynamic[i] = L_au3D_wd[inv_LR_dynamic[i, 0]-1, inv_LR_dynamic[i, 1]-1, inv_LR_dynamic[i, 2]-1] * homevalue_LR[i]
            
            updated_ho_perhurrLoss_wd_LR_dynamic[i, :] = np.squeeze(L_au4D_wd[inv_LR_dynamic[i, 0]-1, inv_LR_dynamic[i, 1]-1, inv_LR_dynamic[i, 2]-1, :]) * homevalue_LR[i]
            
            temp_collect_LR.append(updated_ho_aveLoss_fld_LR_dynamic[i] + updated_ho_aveLoss_wd_LR_dynamic[i] - ho_aveLoss_fld_LR[i] - ho_aveLoss_wd_LR[i])

    # Updating loss values for SFHA structures
    updated_ho_aveLoss_fld_SFHA_dynamic = ho_aveLoss_fld_SFHA.copy()
    updated_ho_aveLoss_wd_SFHA_dynamic = ho_aveLoss_wd_SFHA.copy()
    updated_ho_perhurrLoss_fld_SFHA_dynamic = ho_perhurrLoss_fld_SFHA.copy()
    updated_ho_perhurrLoss_wd_SFHA_dynamic = ho_perhurrLoss_wd_SFHA.copy()
    temp_collect_SFHA = []
    
    for i in range(hoNum_SFHA):
        if R_binary_SFHA_new[i] > 0:
            updated_ho_aveLoss_fld_SFHA_dynamic[i] = L_au3D_fld[inv_SFHA_dynamic[i, 0]-1 + zoneNum_LR, inv_SFHA_dynamic[i, 1]-1, inv_SFHA_dynamic[i, 2]-1] * homevalue_SFHA[i]
            
            updated_ho_perhurrLoss_fld_SFHA_dynamic[i, :] = np.squeeze(L_au4D_fld[inv_SFHA_dynamic[i, 0]-1 + zoneNum_LR, inv_SFHA_dynamic[i, 1]-1, inv_SFHA_dynamic[i, 2]-1, :]) * homevalue_SFHA[i]
            
            updated_ho_aveLoss_wd_SFHA_dynamic[i] = L_au3D_wd[inv_SFHA_dynamic[i, 0]-1 + zoneNum_LR, inv_SFHA_dynamic[i, 1]-1, inv_SFHA_dynamic[i, 2]-1] * homevalue_SFHA[i]
            
            updated_ho_perhurrLoss_wd_SFHA_dynamic[i, :] = np.squeeze(L_au4D_wd[inv_SFHA_dynamic[i, 0]-1 + zoneNum_LR, inv_SFHA_dynamic[i, 1]-1, inv_SFHA_dynamic[i, 2]-1, :]) * homevalue_SFHA[i]
            
            temp_collect_SFHA.append(updated_ho_aveLoss_fld_SFHA_dynamic[i] + updated_ho_aveLoss_wd_SFHA_dynamic[i] - ho_aveLoss_fld_SFHA[i] - ho_aveLoss_wd_SFHA[i])
    
    # Update data
    ho_aveLoss_fld_LR = updated_ho_aveLoss_fld_LR_dynamic.copy()
    ho_aveLoss_fld_SFHA = updated_ho_aveLoss_fld_SFHA_dynamic.copy()
    ho_perhurrLoss_fld_LR = updated_ho_perhurrLoss_fld_LR_dynamic.copy()
    ho_perhurrLoss_fld_SFHA = updated_ho_perhurrLoss_fld_SFHA_dynamic.copy()
    ho_aveLoss_wd_LR = updated_ho_aveLoss_wd_LR_dynamic.copy()
    ho_aveLoss_wd_SFHA = updated_ho_aveLoss_wd_SFHA_dynamic.copy()
    ho_perhurrLoss_wd_LR = updated_ho_perhurrLoss_wd_LR_dynamic.copy()
    ho_perhurrLoss_wd_SFHA = updated_ho_perhurrLoss_wd_SFHA_dynamic.copy()
    stru['stru']['inv_LR'][0,0] = inv_LR_dynamic
    stru['stru']['inv_SFHA'][0,0] = inv_SFHA_dynamic
    
    return ho_aveLoss_fld_LR,ho_aveLoss_fld_SFHA,ho_perhurrLoss_fld_LR,ho_perhurrLoss_fld_SFHA,\
    ho_aveLoss_wd_LR,ho_aveLoss_wd_SFHA,ho_perhurrLoss_wd_LR,ho_perhurrLoss_wd_SFHA,\
    stru,R_LR_new_12in1_withGrant,R_SFHA_new_12in1_withGrant,\
    sum_R_LR_12in1_voluntary,sum_R_SFHA_12in1_voluntary

    
        
        
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

    




