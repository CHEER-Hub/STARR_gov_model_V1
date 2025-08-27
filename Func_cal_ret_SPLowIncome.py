#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 20 21:41:25 2024

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
def Func_cal_ret_SPforLowIncome(ho_acqRecord_LR, ho_acqRecord_SFHA,
                               priceRet_alpha, priceRet_beta, J, annual_threshold,
                               stru, L_au3D_fld, L_au3D_wd,
                               L_au4D_fld, L_au4D_wd,
                               ho_aveLoss_fld_LR, ho_aveLoss_fld_SFHA,
                               ho_aveLoss_wd_LR, ho_aveLoss_wd_SFHA,
                               ho_perhurrLoss_fld_LR, ho_perhurrLoss_fld_SFHA,
                               ho_perhurrLoss_wd_LR, ho_perhurrLoss_wd_SFHA,
                               ho_cumEXP_fld_LR_SCENYEAR, ho_cumEXP_fld_SFHA_SCENYEAR,
                               R_LR_new_12in1_withGrant, R_SFHA_new_12in1_withGrant,
                               real_index_L, PARA_gov_benefitratio_forlow,
                               hoNum_LR, hoNum_SFHA, zoneNum_LR, zoneNum_SFHA):
    """ Calculate special retrofit decisions for low-income group """

    ########
    ## •Record low-income households without doing acquisition
    ##•	Exclude households who have accepted retrofit offers this year

    index_L_LR = real_index_L[real_index_L <= hoNum_LR - 1] -1  # low income HO in low risk
    index_L_SFHA = real_index_L[real_index_L > hoNum_LR] - hoNum_LR -1 # low income HO in high risk
    
    temp = []
    
    for i in range(len(index_L_SFHA)):
        if ho_acqRecord_SFHA[index_L_SFHA[i]] > 0:
            temp.append(i)
    index_L_SFHA = np.delete(index_L_SFHA, temp)

    exclude_granted_LR = 1 - R_LR_new_12in1_withGrant[:, 8]  # R_LR_new_12in1_withGrant[:,9] ret decision with grant

    exclude_granted_SFHA = 1 - R_SFHA_new_12in1_withGrant[:, 8]
    
    ########
    ## Initialize parameters and variables
    np.random.seed(2)
    cindex = np.squeeze(stru['stru']['cindex'][0,0]) # resistance
    #hoNum_LR = np.squeeze(stru['stru']['hoNum_LR'][0,0])
    #hoNum_SFHA = np.squeeze(stru['stru']['hoNum_SFHA'][0,0]) # employment status 0/1

    inv_LR_dynamic = np.squeeze(stru['stru']['inv_LR'][0,0]) # inventory
    inv_SFHA_dynamic = np.squeeze(stru['stru']['inv_SFHA'][0,0])

    retrofitcost_LR = np.squeeze(stru['stru']['retrofitcost_LR'][0,0])
    retrofitcost_SFHA = np.squeeze(stru['stru']['retrofitcost_SFHA'][0,0]) # 8 types of retrofit with price

    ho_areaID_LR = np.squeeze(stru['stru']['ho_areaID_LR'][0,0])
    ho_areaID_SFHA = np.squeeze(stru['stru']['ho_areaID_SFHA'][0,0])

    temp_ho_aveLoss_fldwd_LR = ho_aveLoss_fld_LR + ho_aveLoss_wd_LR
    temp_ho_aveLoss_fldwd_SFHA = ho_aveLoss_fld_SFHA + ho_aveLoss_wd_SFHA

    ho_aveLoss_fld_LR_dynamic = ho_aveLoss_fld_LR
    ho_aveLoss_fld_SFHA_dynamic = ho_aveLoss_fld_SFHA
    ho_aveLoss_wd_LR_dynamic = ho_aveLoss_wd_LR
    ho_aveLoss_wd_SFHA_dynamic = ho_aveLoss_wd_SFHA
    
    ########
    ## Calculate zone level retrofit loading factors (“area_priceRet_loadingFactor_LR” and “area_priceRet_loadingFactor_SFHA”) 
    ## and zone level retrofit prices (“ho_priceRet_LR” and “ho_priceRet_SFHA”)

    # Define the size of arrays
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

    ########
    ## •Calculate low-income households’ flood-related retrofit decisions. Low-income households are able to choose any of 3 flood-related retrofits without DCM
    ## •Update flood-related retrofits based on households’ expected flood losses. Low-income households must have positive expected flood losses to be eligible for flood-related retrofits

    R_apin_fld_LR = np.zeros(hoNum_LR)
    R_ap_fld_LR = np.zeros(hoNum_LR)
    R_in_fld_LR = np.zeros(hoNum_LR)
    R_el_fld_LR = np.zeros(hoNum_LR)
    logical_inv_LR_dynamic_9 = inv_LR_dynamic[:, 8] == 1  # Adjusted index to 8 (9-1)
    R_apin_fld_LR[index_L_LR] = logical_inv_LR_dynamic_9[index_L_LR]  # Adjusted indices to start from 0
    
    R_ap_fld_LR[index_L_LR] = Func_supRand(0.2 * np.ones(len(index_L_LR))) * R_apin_fld_LR[index_L_LR]
    R_in_fld_LR[index_L_LR] = (R_ap_fld_LR[index_L_LR] == 0) * R_apin_fld_LR[index_L_LR]
    R_el_fld_LR[index_L_LR] = logical_inv_LR_dynamic_9[index_L_LR]
    
    R_apin_fld_SFHA = np.zeros(hoNum_SFHA)
    R_ap_fld_SFHA = np.zeros(hoNum_SFHA)
    R_in_fld_SFHA = np.zeros(hoNum_SFHA)
    R_el_fld_SFHA = np.zeros(hoNum_SFHA)
    logical_inv_SFHA_dynamic_9 = inv_SFHA_dynamic[:, 8] == 1  # Adjusted index to 8 (9-1)
    R_apin_fld_SFHA[index_L_SFHA] = logical_inv_SFHA_dynamic_9[index_L_SFHA]  # Adjusted indices to start from 0
    
    R_ap_fld_SFHA[index_L_SFHA] = Func_supRand(0.2 * np.ones(len(index_L_SFHA))) * R_apin_fld_SFHA[index_L_SFHA]
    R_in_fld_SFHA[index_L_SFHA] = (R_ap_fld_SFHA[index_L_SFHA] == 0) * R_apin_fld_SFHA[index_L_SFHA]
    R_el_fld_SFHA[index_L_SFHA] = logical_inv_SFHA_dynamic_9[index_L_SFHA]
    
    # Filtering by dynamic positive flood loss
    logical_ho_aveLoss_fld_LR_dynamic_positive = np.squeeze(ho_aveLoss_fld_LR_dynamic > 0)
    R_ap_fld_LR *= logical_ho_aveLoss_fld_LR_dynamic_positive
    R_in_fld_LR *= logical_ho_aveLoss_fld_LR_dynamic_positive
    R_el_fld_LR *= logical_ho_aveLoss_fld_LR_dynamic_positive
    
    logical_ho_aveLoss_fld_SFHA_dynamic_positive = np.squeeze(ho_aveLoss_fld_SFHA_dynamic > 0)
    R_ap_fld_SFHA *= logical_ho_aveLoss_fld_SFHA_dynamic_positive
    R_in_fld_SFHA *= logical_ho_aveLoss_fld_SFHA_dynamic_positive
    R_el_fld_SFHA *= logical_ho_aveLoss_fld_SFHA_dynamic_positive

    ########
    ## Calculate low-income households’ wind-related retrofit decisions. Low-income households are able to choose any of 5 wind-related retrofits without DCM

    # Define the size of arrays
    R_roof_LR = np.zeros(hoNum_LR)
    R_roof_sh_LR = np.zeros(hoNum_LR)
    R_roof_ad_LR = np.zeros(hoNum_LR)
    R_openings_LR = np.zeros(hoNum_LR)
    R_openings_sh_LR = np.zeros(hoNum_LR)
    R_openings_ir_LR = np.zeros(hoNum_LR)
    R_rtw_st_LR = np.zeros(hoNum_LR)
    
    # Compute values for LR areas
    R_roof_LR[index_L_LR] = inv_LR_dynamic[index_L_LR, 4] == 1
    R_roof_sh_LR[index_L_LR] = R_roof_LR[index_L_LR] * (inv_LR_dynamic[index_L_LR, 3] == 1)
    R_roof_ad_LR[index_L_LR] = (R_roof_sh_LR[index_L_LR] == 0) * R_roof_LR[index_L_LR]
    R_openings_LR[index_L_LR] = (inv_LR_dynamic[index_L_LR, 6] == 1) * (inv_LR_dynamic[index_L_LR, 4] == 2)
    R_openings_sh_LR[index_L_LR] = R_openings_LR[index_L_LR]
    R_openings_ir_LR[index_L_LR] = R_openings_LR[index_L_LR] * (R_openings_sh_LR[index_L_LR] == 0)
    R_rtw_st_LR[index_L_LR] = (inv_LR_dynamic[index_L_LR, 5] == 1) * (inv_LR_dynamic[index_L_LR, 4] == 2) * (inv_LR_dynamic[index_L_LR, 6] > 1)
    
    # Similar computations for SFHA areas
    R_roof_SFHA = np.zeros(hoNum_SFHA)
    R_roof_sh_SFHA = np.zeros(hoNum_SFHA)
    R_roof_ad_SFHA = np.zeros(hoNum_SFHA)
    R_openings_SFHA = np.zeros(hoNum_SFHA)
    R_openings_sh_SFHA = np.zeros(hoNum_SFHA)
    R_openings_ir_SFHA = np.zeros(hoNum_SFHA)
    R_rtw_st_SFHA = np.zeros(hoNum_SFHA)
    
    R_roof_SFHA[index_L_SFHA] = inv_SFHA_dynamic[index_L_SFHA, 4] == 1
    R_roof_sh_SFHA[index_L_SFHA] = R_roof_SFHA[index_L_SFHA] * (inv_SFHA_dynamic[index_L_SFHA, 3] == 1)
    R_roof_ad_SFHA[index_L_SFHA] = (R_roof_sh_SFHA[index_L_SFHA] == 0) * R_roof_SFHA[index_L_SFHA]
    R_openings_SFHA[index_L_SFHA] = (inv_SFHA_dynamic[index_L_SFHA, 6] == 1) * (inv_SFHA_dynamic[index_L_SFHA, 4] == 2)
    R_openings_sh_SFHA[index_L_SFHA] = R_openings_SFHA[index_L_SFHA]
    R_openings_ir_SFHA[index_L_SFHA] = R_openings_SFHA[index_L_SFHA] * (R_openings_sh_SFHA[index_L_SFHA] == 0)
    R_rtw_st_SFHA[index_L_SFHA] = (inv_SFHA_dynamic[index_L_SFHA, 5] == 1) * (inv_SFHA_dynamic[index_L_SFHA, 4] == 2) * (inv_SFHA_dynamic[index_L_SFHA, 6] > 1)
    
        
    ########
    ## •Randomly pick 1 retrofit for households, if they have multiple retrofit plans, meaning households try to do more than 1 type (out of 8) of retrofits.
    ##•	Update households’ retrofit decisions

    # Assume all variables like R_ap_fld_LR, hoNum_LR, etc. are already defined
    
    # Randomly select one retrofit for LR
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
    # Calculate retrofit costs for LR
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
    # Initialize the updated average loss arrays
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
    ## •Calculate households’ benefits (loss reduction) after retrofitting
    ##•	Calculate total retrofit costs (sum all 8 types of retrofits)
    ##•	Calculate retrofit grants for households based on zone level retrofit prices and maximum retrofit payment (“J”)

    # Calculate benefit_fld after retrofits
    temp_reduced_aveLoss_fldwd_LR = np.squeeze(ho_aveLoss_fld_LR_dynamic + ho_aveLoss_wd_LR_dynamic - temp_updated_aveLoss_fldwd_LR)  # loss benefit
    temp_reduced_aveLoss_fldwd_SFHA = np.squeeze(ho_aveLoss_fld_SFHA_dynamic + ho_aveLoss_wd_SFHA_dynamic - temp_updated_aveLoss_fldwd_SFHA)
    
    # Handle potential data errors where loss reduction is negative
    temp_reduced_aveLoss_fldwd_LR[temp_reduced_aveLoss_fldwd_LR < 0] = 0
    temp_reduced_aveLoss_fldwd_SFHA[temp_reduced_aveLoss_fldwd_SFHA < 0] = 0
    
    # Calculate retrofit costs for LR and SFHA households
    cost_binaryR_LR = (cost_ap_fld_LR + cost_in_fld_LR + cost_el_fld_LR +
                       cost_roof_sh_LR + cost_roof_ad_LR +
                       cost_openings_sh_LR + cost_openings_ir_LR +
                       cost_rtw_st_LR)
    
    cost_binaryR_SFHA = (cost_ap_fld_SFHA + cost_in_fld_SFHA + cost_el_fld_SFHA +
                         cost_roof_sh_SFHA + cost_roof_ad_SFHA +
                         cost_openings_sh_SFHA + cost_openings_ir_SFHA +
                         cost_rtw_st_SFHA)
    
    # Calculate the grant for retrofits
    grant_LR_old = np.minimum(ho_priceRet_LR * cost_binaryR_LR * R_binary_LR_old, J * R_binary_LR_old)
    grant_SFHA_old = np.minimum(ho_priceRet_SFHA * cost_binaryR_SFHA * R_binary_SFHA_old, J * R_binary_SFHA_old)
    
    ########
    ## •Update households’ benefits by considering grants and retrofit costs
    ##•	Update households’ retrofit decisions. Accept retrofit if their benefits are greater than “annual_threshold”; otherwise, refuse retrofit
    ##•	Update benefits, retrofit costs, and retrofit grants based updated retrofit decisions

    # Update benefit calculations after retrofits
    temp_benefit_fldwd_LR_withGrant = temp_reduced_aveLoss_fldwd_LR - cost_binaryR_LR + grant_LR_old
    temp_benefit_fldwd_SFHA_withGrant = temp_reduced_aveLoss_fldwd_SFHA - cost_binaryR_SFHA + grant_SFHA_old
    
    # Determine which households should retrofit based on the annual threshold
    logical_withGrant_LR = (temp_benefit_fldwd_LR_withGrant >= annual_threshold)
    logical_withGrant_SFHA = (temp_benefit_fldwd_SFHA_withGrant >= annual_threshold)
    
    # Update retrofit decisions and related costs/benefits for LR households
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
    grant_LR_old = grant_LR_old * logical_withGrant_LR
    
    # Update retrofit decisions and related costs/benefits for SFHA households
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
    
    # Update binary retrofit decisions
    R_binary_LR_new_withGrant = R_binary_LR_old * logical_withGrant_LR
    R_binary_SFHA_new_withGrant = R_binary_SFHA_old * logical_withGrant_SFHA
    
    ########
    ## Record retrofit decisions using variables “R_LR_new_12in1_withGrant” and “R_SFHA_new_12in1_withGrant”    

    # Combine retrofit decision arrays for LR households with respective benefits, costs, and grants
    R_LR_new_12in1_withGrant = np.column_stack((
        R_ap_fld_LR_withGrant,
        R_in_fld_LR_withGrant,
        R_el_fld_LR_withGrant,
        R_roof_sh_LR_withGrant,
        R_roof_ad_LR_withGrant,
        R_openings_sh_LR_withGrant,
        R_openings_ir_LR_withGrant,
        R_rtw_st_LR_withGrant,
        R_binary_LR_new_withGrant,
        temp_reduced_aveLoss_fldwd_LR * R_binary_LR_new_withGrant * 30,
        cost_binaryR_LR * R_binary_LR_new_withGrant * 30,
        grant_LR_old * R_binary_LR_new_withGrant * 30
    ))
    
    # Combine retrofit decision arrays for SFHA households with respective benefits, costs, and grants
    R_SFHA_new_12in1_withGrant = np.column_stack((
        R_ap_fld_SFHA_withGrant,
        R_in_fld_SFHA_withGrant,
        R_el_fld_SFHA_withGrant,
        R_roof_sh_SFHA_withGrant,
        R_roof_ad_SFHA_withGrant,
        R_openings_sh_SFHA_withGrant,
        R_openings_ir_SFHA_withGrant,
        R_rtw_st_SFHA_withGrant,
        R_binary_SFHA_new_withGrant,
        temp_reduced_aveLoss_fldwd_SFHA * R_binary_SFHA_new_withGrant * 30,
        cost_binaryR_SFHA * R_binary_SFHA_new_withGrant * 30,
        grant_SFHA_old * R_binary_SFHA_new_withGrant * 30
    ))
    
    # Optional debug statements to display the sum of retrofit decisions before applying the exclusions
    # print('Efficiency before LR:', np.sum(R_LR_new_12in1_withGrant[:, :8], axis=0))
    # print('Efficiency before SFHA:', np.sum(R_SFHA_new_12in1_withGrant[:, :8], axis=0))
    
    # Apply exclusions for granted LR households
    R_LR_new_12in1_withGrant = R_LR_new_12in1_withGrant * exclude_granted_LR.reshape(-1,1)
    
    # Apply exclusions for granted SFHA households (optional if not needed)
    R_SFHA_new_12in1_withGrant = R_SFHA_new_12in1_withGrant * exclude_granted_SFHA.reshape(-1,1)

    ########
    ## Update and return low-income households’ retrofit decisions based on the preset benefit ratio. 
    ## (accept: if the ratio of benefits to retrofit grants is larger than preset benefit ratio; decline: otherwise)
    # Iterate through index_L_LR and apply conditions
    for i in index_L_LR:
        if R_LR_new_12in1_withGrant[i, 8] > 0:
            if R_LR_new_12in1_withGrant[i, 9] / R_LR_new_12in1_withGrant[i, 11] < PARA_gov_benefitratio_forlow:
                R_LR_new_12in1_withGrant[i, :] = 0
    
    # Iterate through index_L_SFHA and apply conditions
    for i in index_L_SFHA:
        if R_SFHA_new_12in1_withGrant[i, 8] > 0:
            if R_SFHA_new_12in1_withGrant[i, 9] / R_SFHA_new_12in1_withGrant[i, 11] < PARA_gov_benefitratio_forlow:
                R_SFHA_new_12in1_withGrant[i, :] = 0


    ########
    ## • Initialize parameters and variables for implementing
    ## • Retrieve 8 types of low-income households’ retrofit decisions
    # Implementation start
    # Acquisition flags
    acquisition_flag_LR = (ho_acqRecord_LR == 0)
    acquisition_flag_SFHA = (ho_acqRecord_SFHA == 0)
    
    # Updating retrofit values
    R_LR_new_12in1_withGrant = R_LR_new_12in1_withGrant * acquisition_flag_LR[:, np.newaxis]  # Element-wise multiplication
    R_SFHA_new_12in1_withGrant = R_SFHA_new_12in1_withGrant * acquisition_flag_SFHA[:, np.newaxis]
    
    # Summing the retrofit values
    sum_R_LR_12in1_voluntary = np.sum(R_LR_new_12in1_withGrant, axis=0)  # Sum along the first dimension
    sum_R_SFHA_12in1_voluntary = np.sum(R_SFHA_new_12in1_withGrant, axis=0)
    
    # Updating average loss values
    ho_aveLoss_fld_LR = np.squeeze(ho_aveLoss_fld_LR) * acquisition_flag_LR
    ho_aveLoss_fld_SFHA = np.squeeze(ho_aveLoss_fld_SFHA) * acquisition_flag_SFHA
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
    ## Update resistance levels and resistance details in building inventory based on low-income households’ retrofit decisions
    # Update inv_LR_dynamic for low-risk households
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

    # This also needs verification: is there a difference between going from A to B, A to C, B to C, and A directly to C?
    # Update resistance index after retrofits, updating the third column
    # This is comparing the last six digits and then only changing the resistance type of the third position of inv_LR inventory
    # Apply the binary mask
    temp_inv_LR_dynamic3 = temp_inv_LR_dynamic3 * R_binary_LR_new  # if do ret, keep new c level index
    temp_inv_SFHA_dynamic3 = temp_inv_SFHA_dynamic3 * R_binary_SFHA_new
    
    # Update inv_LR_dynamic
    indices_LR = np.nonzero(temp_inv_LR_dynamic3)[0]
    inv_LR_dynamic[indices_LR, 2] = temp_inv_LR_dynamic3[indices_LR]  # update new c level for pos3 of inventory
    
    # Update inv_SFHA_dynamic
    indices_SFHA = np.nonzero(temp_inv_SFHA_dynamic3)[0]
    inv_SFHA_dynamic[indices_SFHA, 2] = temp_inv_SFHA_dynamic3[indices_SFHA]
    
    ########
    ## • Update low-income households’ expected losses and losses per hurricane based on low-income households’ retrofit decisions
    # Updating loss values for LR structures
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
            updated_ho_aveLoss_fld_SFHA_dynamic[i] = L_au3D_fld[inv_SFHA_dynamic[i, 0]-1 + 503, inv_SFHA_dynamic[i, 1]-1, inv_SFHA_dynamic[i, 2]-1] * homevalue_SFHA[i]
            
            updated_ho_perhurrLoss_fld_SFHA_dynamic[i, :] = np.squeeze(L_au4D_fld[inv_SFHA_dynamic[i, 0]-1 + 503, inv_SFHA_dynamic[i, 1]-1, inv_SFHA_dynamic[i, 2]-1, :]) * homevalue_SFHA[i]
            
            updated_ho_aveLoss_wd_SFHA_dynamic[i] = L_au3D_wd[inv_SFHA_dynamic[i, 0]-1 + 503, inv_SFHA_dynamic[i, 1]-1, inv_SFHA_dynamic[i, 2]-1] * homevalue_SFHA[i]
            
            updated_ho_perhurrLoss_wd_SFHA_dynamic[i, :] = np.squeeze(L_au4D_wd[inv_SFHA_dynamic[i, 0]-1 + 503, inv_SFHA_dynamic[i, 1]-1, inv_SFHA_dynamic[i, 2]-1, :]) * homevalue_SFHA[i]
            
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

    return (ho_aveLoss_fld_LR,ho_aveLoss_fld_SFHA,ho_perhurrLoss_fld_LR,ho_perhurrLoss_fld_SFHA,
    ho_aveLoss_wd_LR,ho_aveLoss_wd_SFHA,ho_perhurrLoss_wd_LR,ho_perhurrLoss_wd_SFHA,
    stru,sum_R_LR_12in1_voluntary,sum_R_SFHA_12in1_voluntary,
    R_LR_new_12in1_withGrant,R_SFHA_new_12in1_withGrant)
    
    


