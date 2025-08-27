#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 18 13:04:49 2024

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
def Func_cal_retrofit(ho_acqRecord_LR, ho_acqRecord_SFHA, priceRet_alpha, priceRet_beta, J, annual_threshold, stru, L_au3D_fld, L_au3D_wd, \
                      ho_aveLoss_fld_LR, ho_aveLoss_fld_SFHA, ho_aveLoss_wd_LR, ho_aveLoss_wd_SFHA, ho_cumEXP_fld_LR_SCENYEAR, ho_cumEXP_fld_SFHA_SCENYEAR, real_index_L,\
                        hoNum_LR, hoNum_SFHA, zoneNum_LR, zoneNum_SFHA, years, fld_dcm_coeff, wd_dcm_coeff):
    """ Calculate retrofit decisions"""
    
    lowAllIn_ho_acqRecord_LR = ho_acqRecord_LR.copy()
    lowAllIn_ho_acqRecord_SFHA = ho_acqRecord_SFHA.copy()
    #if real_index_L:
    #    lowAllIn_ho_acqRecord_LR[real_index_L[real_index_L <= hoNum_LR] - 1] = 1
    #    lowAllIn_ho_acqRecord_SFHA[real_index_L[real_index_L > hoNum_LR] - hoNum_LR - 1] = 1

    np.random.seed(2)
    
    # The variables used in the retrofit models include alternative specific
    # constants of revealed preference variables, retrofit grant percentage,
    # maximum grant amount, number of hurricanes previously experienced,
    # straightline distance to the coastline (km), and unemployment status
    # (employed, unemployed, or retired).
    # Esther Chiew et al. in review

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
    
    temp_ho_aveLoss_fldwd_LR = ho_aveLoss_fld_LR + ho_aveLoss_wd_LR # total loss = wind + flood
    temp_ho_aveLoss_fldwd_SFHA = ho_aveLoss_fld_SFHA + ho_aveLoss_wd_SFHA

    ho_aveLoss_fld_LR_dynamic = ho_aveLoss_fld_LR.copy()
    ho_aveLoss_fld_SFHA_dynamic = ho_aveLoss_fld_SFHA.copy()
    ho_aveLoss_wd_LR_dynamic = ho_aveLoss_wd_LR.copy()
    ho_aveLoss_wd_SFHA_dynamic = ho_aveLoss_wd_SFHA.copy()
   
    ########
    ## Calculate zone level retrofit loading factors (“area_priceRet_loadingFactor_LR” and “area_priceRet_loadingFactor_SFHA”) and zone level retrofit prices (“ho_priceRet_LR” and “ho_priceRet_SFHA”)
    area_priceRet_loadingFactor_LR = np.zeros(zoneNum_LR)
    for i in range(zoneNum_LR):
        temp = (ho_areaID_LR == (i+1)) & (ho_acqRecord_LR == 0)
        area_priceRet_loadingFactor_LR[i] = np.mean(temp_ho_aveLoss_fldwd_LR[temp]) # sum P^h L^h * X (num) / sum X (num) = mean
    area_priceRet_loadingFactor_LR[np.isnan(area_priceRet_loadingFactor_LR)] = 0 # area no building set to 0; 0/0=NaN
    area_priceRet_loadingFactor_LR /= 1e4

    ho_priceRet_LR = np.zeros(hoNum_LR)
    for i in range(zoneNum_LR):
        temp = ho_areaID_LR == (i+1)
        ho_priceRet_LR[temp] = priceRet_alpha + priceRet_beta * area_priceRet_loadingFactor_LR[i] # c_base+c_prop*loading factor

    area_priceRet_loadingFactor_SFHA = np.zeros(zoneNum_SFHA)
    for i in range(zoneNum_SFHA):
        temp = (ho_areaID_SFHA == (i + zoneNum_LR + 1)) & (ho_acqRecord_SFHA == 0)
        area_priceRet_loadingFactor_SFHA[i] = np.mean(temp_ho_aveLoss_fldwd_SFHA[temp])
    area_priceRet_loadingFactor_SFHA[np.isnan(area_priceRet_loadingFactor_SFHA)] = 0
    area_priceRet_loadingFactor_SFHA /= 1e4

    ho_priceRet_SFHA = np.zeros(hoNum_SFHA)
    for i in range(zoneNum_SFHA):
        temp = ho_areaID_SFHA == (i + 1 + zoneNum_LR)
        ho_priceRet_SFHA[temp] = priceRet_alpha + priceRet_beta * area_priceRet_loadingFactor_SFHA[i]
    
    #########
    # inv_LR, SFHA, 64w, 28w*9
    # 1st column: location ID
    # 2nd column: building type
    # 3rd column: resistance type
    # Maximum is 192, the last 6 digits actually combine to make 192=2*2*2*3*2*4. Why so many?
    # 4th column: roof cover resistance level
    # 5th column: roof sheathing resistance level
    # 6th column: roof to wall connection resistance level
    # 7th column: opening resistance level
    # 8th column: wall resistance level
    # 9th column: flood resistance level
    
    # Ximc is X_LR, 503*1536 and L_SFHA1006*1536, is this split into 3D?
    # cindex 192*7 is the index of resistance types and resistance levels of each component.
    # This is the standard, lookup table.
    # 1st column: resistance type is 1 to 192, below are integer levels.
    # 2nd column: roof cover resistance level
    # 3rd column: roof sheathing resistance level
    # 4th column: roof to wall connection resistance level
    # 5th column: opening resistance level
    # 6th column: wall resistance level
    # 7th column: flood resistance level
    
    # Retrofit costs are arranged like this, wind-related from 1-5, flood-related from 6-8.
    
    # 4-D L_au4D_fld, loss, 1509*8*192*97
    # It should be location ID, resistance type, 6*retrofit
    
    # For fld: ap in el, and for wd: roof_sh roof_ad openings_sh openings_ir rtw, total 8 types.
    # Resistance levels have 6 types, corresponding to columns 4, 5, 6, 7, 9 in inv_LR.
    #######


    ########
    ## Calculate households’ flood-related retrofit decisions through DCM. There are 3 flood-related retrofits: 
    ## (i) elevating house appliances (“R_ap_fld_LR” and “R_ap_fld_SFHA”); 
    ##(ii) installing water resistant (“R_in_fld_LR” and “R_in_fld_SFHA”); 
    ##(iii) elevating the entire house (“R_el_fld_LR” and “R_el_fld_SFHA”)
    # "ap" "in" "el" (ap: elevating house appliances; in: installing water resistant; el: elevating the entire house)
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
    ## Calculate households’ wind-related retrofit decisions through DCM. There are 5 wind-related retrofits: 
    ##(i) reinforcing roof with high wind load shingles (“R_roof_sh_LR” and “R_roof_sh_SFHA”); 
    ##(ii) reinforcing roof with adhesive foam (“R_roof_ad_LR” and “R_roof_ad_SFHA”); 
    ##(iii) strengthening openings with shutters (“R_openings_sh_LR” and “R_openings_sh_SFHA”); 
    ##(iv) strengthening openings with impact‐resistant windows (“R_openings_ir_LR” and “R_openings_ir_SFHA”); 
    ##(v) strengthening roof‐to‐wall connection using straps (“R_rtw_st_LR” and “R_rtw_st_SFHA”)
    # Roof LR
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
    ## Update costs for retrofits based on retrofit decisions
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
    ## Calculate households’ benefits (loss reduction) after retrofitting
    ##Calculate total retrofit costs (sum all 8 types of retrofits)
    ##Calculate retrofit grants for households based on zone level retrofit prices and maximum retrofit payment (“J”)
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
    
    grant_LR_old = np.minimum(ho_priceRet_LR * cost_binaryR_LR * R_binary_LR_old, J * R_binary_LR_old)
    grant_SFHA_old = np.minimum(ho_priceRet_SFHA * cost_binaryR_SFHA * R_binary_SFHA_old, J * R_binary_SFHA_old)

    ########
    ## Update households’ benefits by considering grants and retrofit costs
    ## Update households’ retrofit decisions. Accept retrofit if their benefits are greater than “annual_threshold”; otherwise, refuse retrofit
    ## Update benefits, retrofit costs, and retrofit grants based updated retrofit decisions.
    temp_benefit_fldwd_LR_withGrant = temp_reduced_aveLoss_fldwd_LR - cost_binaryR_LR + grant_LR_old  # benefit=total loss benefit - total cost + grant
    temp_benefit_fldwd_SFHA_withGrant = temp_reduced_aveLoss_fldwd_SFHA - cost_binaryR_SFHA + grant_SFHA_old
    
    # Update retrofit decisions - only those above threshold will retrofit
    logical_withGrant_LR = (temp_benefit_fldwd_LR_withGrant >= annual_threshold)  # benefit >= -300
    R_ap_fld_LR_withGrant = R_ap_fld_LR * logical_withGrant_LR  # if benefit exists (>=-300), do retrofit. Otherwise, don't do.
    R_in_fld_LR_withGrant = R_in_fld_LR * logical_withGrant_LR
    R_el_fld_LR_withGrant = R_el_fld_LR * logical_withGrant_LR
    R_roof_sh_LR_withGrant = R_roof_sh_LR * logical_withGrant_LR
    R_roof_ad_LR_withGrant = R_roof_ad_LR * logical_withGrant_LR
    R_openings_sh_LR_withGrant = R_openings_sh_LR * logical_withGrant_LR
    R_openings_ir_LR_withGrant = R_openings_ir_LR * logical_withGrant_LR
    R_rtw_st_LR_withGrant = R_rtw_st_LR * logical_withGrant_LR
    temp_reduced_aveLoss_fldwd_LR = temp_reduced_aveLoss_fldwd_LR * logical_withGrant_LR  # if no benefit, loss benefit=0
    cost_binaryR_LR = cost_binaryR_LR * logical_withGrant_LR  # if no benefit, cost=0
    grant_LR_old = grant_LR_old * logical_withGrant_LR  # if no benefit, grant=0
    
    logical_withGrant_SFHA = (temp_benefit_fldwd_SFHA_withGrant >= annual_threshold)
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
    grant_SFHA_old = grant_SFHA_old * logical_withGrant_SFHA
    
    R_binary_LR_new_withGrant = R_binary_LR_old * logical_withGrant_LR
    R_binary_SFHA_new_withGrant = R_binary_SFHA_old * logical_withGrant_SFHA
    
    ########
    ## Record and return retrofit decisions using variables “R_LR_new_12in1_withGrant” and “R_SFHA_new_12in1_withGrant”
    R_LR_new_12in1_withGrant = np.column_stack([
    R_ap_fld_LR_withGrant,
    R_in_fld_LR_withGrant,
    R_el_fld_LR_withGrant,
    R_roof_sh_LR_withGrant,
    R_roof_ad_LR_withGrant,
    R_openings_sh_LR_withGrant,
    R_openings_ir_LR_withGrant,
    R_rtw_st_LR_withGrant,
    R_binary_LR_new_withGrant,
    temp_reduced_aveLoss_fldwd_LR * R_binary_LR_new_withGrant * years,
    cost_binaryR_LR * R_binary_LR_new_withGrant * years,
    grant_LR_old * R_binary_LR_new_withGrant * years
    ])
    
    R_SFHA_new_12in1_withGrant = np.column_stack([
        R_ap_fld_SFHA_withGrant,
        R_in_fld_SFHA_withGrant,
        R_el_fld_SFHA_withGrant,
        R_roof_sh_SFHA_withGrant,
        R_roof_ad_SFHA_withGrant,
        R_openings_sh_SFHA_withGrant,
        R_openings_ir_SFHA_withGrant,
        R_rtw_st_SFHA_withGrant,
        R_binary_SFHA_new_withGrant,
        temp_reduced_aveLoss_fldwd_SFHA * R_binary_SFHA_new_withGrant * years,
        cost_binaryR_SFHA * R_binary_SFHA_new_withGrant * years,
        grant_SFHA_old * R_binary_SFHA_new_withGrant * years
    ])
    
    # Exclude low-income homeowners
    R_LR_new_12in1_withGrant *= (lowAllIn_ho_acqRecord_LR == 0).reshape(-1, 1)
    R_SFHA_new_12in1_withGrant *= (lowAllIn_ho_acqRecord_SFHA == 0).reshape(-1, 1)
    
    ho_priceRet_LR *= R_binary_LR_new_withGrant
    ho_priceRet_SFHA *= R_binary_SFHA_new_withGrant
    
    
    return (R_LR_new_12in1_withGrant,R_SFHA_new_12in1_withGrant,
    temp_inv_LR_dynamic3,temp_inv_SFHA_dynamic3,
    area_priceRet_loadingFactor_LR,area_priceRet_loadingFactor_SFHA,ho_priceRet_LR,ho_priceRet_SFHA)



    
    
    
    
    