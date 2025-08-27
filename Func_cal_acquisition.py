#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 16 22:31:50 2024

@author: Jingya Wang
"""

import numpy as np
from scipy.stats import norm


def Func_cal_acquisition(year, priceAcq_alpha, priceAcq_beta, aftershock, budget, loss_merged, county_zoneID,
                         ho_acqRecord_SFHA, stru, ho_aveLoss_fld_SFHA, ho_aveLoss_wd_SFHA,
                         ho_lastEXP_fld_SFHA_SCENYEAR, real_index_L, 
                         hoNum_LR, zones, zoneNum_LR, zoneNum_SFHA, years, acq_param_a, acq_param_b, act_zones, county_flag):
    """ Calculate acquisition decisions"""
    
    ########
    ## 	Initialize parameters and variables
    A_LR_2in1 = np.zeros((hoNum_LR, 2))
    area_priceAcq_loadingFactor_LR = np.zeros(zoneNum_LR) # 503 is the # of low risk zones TODO: find a variable name for this
    ho_priceAcq_LR = np.zeros((hoNum_LR, 2))
    real_index_L = real_index_L[real_index_L > hoNum_LR] - hoNum_LR -1
    lowAllIn_ho_acqRecord_SFHA = ho_acqRecord_SFHA.copy() # preserve previous records
    #lowAllIn_ho_acqRecord_SFHA[real_index_L] = 1 # don't consider low income first

    np.random.seed(1)
    
    ho_lastEXP_fld_SFHA_SCENYEARCopy = ho_lastEXP_fld_SFHA_SCENYEAR.copy()
    ho_lastEXP_fld_SFHA_SCENYEARCopy[ho_lastEXP_fld_SFHA_SCENYEARCopy != 1] = 0

    acqTail_SFHA = np.squeeze(stru['stru']['acqTail_SFHA'][0,0])
    homevalue_SFHA = np.squeeze(stru['stru']['homevalue_SFHA'][0,0])
    hoNum_SFHA = np.squeeze(stru['stru']['hoNum_SFHA'][0,0])
    ho_areaID_SFHA = np.squeeze(stru['stru']['ho_areaID_SFHA'][0,0])
    temp_ho_aveLoss_fldwd_SFHA = ho_aveLoss_fld_SFHA + ho_aveLoss_wd_SFHA
    
    ########
    ## Calculate zone level acquisition loading factors (“area_priceAcq_loadingFactor_SFHA”) and 
    ## zone level acquisition prices (“ho_priceAcq_SFHA”)
    area_priceAcq_loadingFactor_SFHA = np.zeros(zoneNum_SFHA)
    for i in range(zoneNum_SFHA):
        temp = np.where((ho_areaID_SFHA == (i+1) + zoneNum_LR) & (ho_acqRecord_SFHA == 0))[0] #satisfy both conditions; price for home owners low risk [504,1509]
        area_priceAcq_loadingFactor_SFHA[i] = years * np.sum(temp_ho_aveLoss_fldwd_SFHA[temp]) / np.sum(homevalue_SFHA[temp])

    area_priceAcq_loadingFactor_SFHA[np.isnan(area_priceAcq_loadingFactor_SFHA)] = 0 # some area no HO, 0/0=NaN; set to 0
    
    ho_priceAcq_SFHA = np.zeros(hoNum_SFHA)
    for i in range(zoneNum_SFHA):
        temp = ho_areaID_SFHA == (i+1) + zoneNum_LR
        ho_priceAcq_SFHA[temp] = priceAcq_alpha + priceAcq_beta * area_priceAcq_loadingFactor_SFHA[i]  #equation of price acq.
        
    ########
    ## Rank zones based on ratios of total losses to total home values in zones and list households in those ranked zones
    acqIndex = np.zeros(zones)
    for i in range(zoneNum_LR+1, zones+1):
        temp = np.where((ho_areaID_SFHA == i) & (ho_acqRecord_SFHA == 0))[0]
        acqIndex[i-1] = np.sum(temp_ho_aveLoss_fldwd_SFHA[temp]) * years / np.sum(homevalue_SFHA[temp])
    acqIndex[np.isnan(acqIndex)] = 0
    #acqIndex_descend = np.sort(acqIndex)[::-1]
    acqIndex_descend_areaID = np.argsort(acqIndex)[::-1] + 1
    
    ho_areaPosition_SFHA = np.zeros(hoNum_SFHA)
    for i in range(zoneNum_LR+1, zones+1):
        temp = np.where(acqIndex_descend_areaID == i)[0] + 1
        ho_areaPosition_SFHA[ho_areaID_SFHA == i] = temp
        
    ########
    ##	If households experience hurricane flood last year, the acquisition price should be discounted with “aftershock”
    ho_aftershock_SFHA = (ho_lastEXP_fld_SFHA_SCENYEARCopy == 0) + (ho_lastEXP_fld_SFHA_SCENYEARCopy == 1) * aftershock
    
    ######
    ## Calculate probabilities of households’ acquisition decisions through DCM
    ho_priceAcq_SFHA = ho_priceAcq_SFHA * ho_aftershock_SFHA
    prob_acq_SFHA = norm.cdf(acq_param_a * ho_priceAcq_SFHA + acqTail_SFHA + acq_param_b * (ho_lastEXP_fld_SFHA_SCENYEARCopy == 0))
    ho_rand_SFHA = np.random.random(len(prob_acq_SFHA)) 
    
    ########
    ## Start at the beginning of ranked zones, calculate in-zone households’ decisions of accepting acquisition or not through pre-calculated probabilities \
    ##and cumulate costs for acquisition if accepted. When all households in a zone are calculated, move to another zone. 
    ##Keep computing until government is run out of budget
    
    if county_flag == False: # allocate over zones, regardless of counties
        
        candidate_hoID_SFHA = np.zeros(hoNum_SFHA)
        cost = 0
        break_flag = False
        for j_position in range(1, act_zones + 1): # TODO: why 708? They said this is feasible zones
        
            hoID_areaj_temp = np.where(ho_areaPosition_SFHA == j_position)[0]
            hoID_areaj = np.zeros(len(hoID_areaj_temp), dtype =int)
            ind = np.argsort(np.random.random((1,len(hoID_areaj_temp))))
            for i in range(len(hoID_areaj_temp)):
                
                hoID_areaj[i] = int(hoID_areaj_temp[ind.T[i]])
            #hoID_areaj = hoID_areaj.T
            #hoID_areaj = np.random.permutation(hoID_areaj) # randomly order them
            
            if len(hoID_areaj) > 0:
                accept = (ho_rand_SFHA[hoID_areaj] < prob_acq_SFHA[hoID_areaj]) & (lowAllIn_ho_acqRecord_SFHA[hoID_areaj] == 0)
        
                for k in range(len(accept)):
                    if accept[k] > 0: # 1: accept; 0: not accept
                        cost += ho_priceAcq_SFHA[hoID_areaj[k]] * homevalue_SFHA[hoID_areaj[k]] #cumulate cost
        
                        if cost > budget * 2: # cost exceeds double budget
                            cost -= ho_priceAcq_SFHA[hoID_areaj[k]] * homevalue_SFHA[hoID_areaj[k]] # exclude current home acquisition
                            break_flag = True
                            break
                        else:
                            candidate_hoID_SFHA[hoID_areaj[k]] = year # hoID_areaj(k)=HO index; set candidate acq year to current year
        
                if break_flag:
                    break # end since budget is run out of
    else: # allocate over counties first
        # Step 1: Prepare mapping from county_ID to list of zone_IDs
        zoneIDs_by_county = county_zoneID.groupby('county_ID')['zone_ID'].apply(list).to_dict()
        
        # Step 2: Prepare fast lookup of county budgets
        county_budget_dict = dict(zip(loss_merged['county_ID'], loss_merged['budget']))
        
        # Step 3: Initialize output
        candidate_hoID_SFHA = np.zeros(ho_areaPosition_SFHA.shape[0], dtype=int)
        
        # Step 4: Loop through counties
        for county_id, budget in county_budget_dict.items():
            if county_id not in zoneIDs_by_county:
                continue
        
            cost = 0
            break_flag = False
            zone_ids = zoneIDs_by_county[county_id]
            np.random.shuffle(zone_ids) # shuffle zones within a county
        
            for zone_id in zone_ids:
                # Convert zone_ID to zone_position used in ho_areaPosition_SFHA
                # If zone_ID already equals zone_position, skip this part
                zone_position = zone_id
        
                # Get all households in this zone
                hoID_areaj_temp = np.where(ho_areaPosition_SFHA == zone_position)[0]
        
                # Randomly shuffle them
                if len(hoID_areaj_temp) == 0:
                    continue
                ind = np.argsort(np.random.random(len(hoID_areaj_temp)))
                hoID_areaj = hoID_areaj_temp[ind]
        
                # Check who accepts acquisition
                accept = (ho_rand_SFHA[hoID_areaj] < prob_acq_SFHA[hoID_areaj]) & \
                         (lowAllIn_ho_acqRecord_SFHA[hoID_areaj] == 0)
        
                for k in range(len(accept)):
                    if accept[k]:
                        idx = hoID_areaj[k]
                        cost_k = ho_priceAcq_SFHA[idx] * homevalue_SFHA[idx]
                        cost += cost_k
        
                        if cost > budget * 2:
                            cost -= cost_k  # rollback
                            break_flag = True
                            break
        
                        # Mark household as acquired in this year
                        candidate_hoID_SFHA[idx] = year
        
                if break_flag:
                    break  # stop county-level allocation if over budget

    
    ########
    ## Record and return acquisition decisions using variables “A_LR_2in1” and “A_SFHA_2in1”
    A_SFHA_2in1 = np.zeros((len(ho_aveLoss_fld_SFHA), 2))
    A_SFHA_2in1[:, 0] = np.squeeze((ho_aveLoss_fld_SFHA + ho_aveLoss_wd_SFHA)) * (candidate_hoID_SFHA == year) * years
    A_SFHA_2in1[:, 1] = homevalue_SFHA * ho_priceAcq_SFHA * (candidate_hoID_SFHA == year)
    
    ho_priceAcq_SFHA_noLastEXP = ho_priceAcq_SFHA * (candidate_hoID_SFHA == year) * (ho_lastEXP_fld_SFHA_SCENYEARCopy == 0)
    ho_priceAcq_SFHA_withLastEXP = ho_priceAcq_SFHA * (candidate_hoID_SFHA == year) * (ho_lastEXP_fld_SFHA_SCENYEARCopy == 1)
    
    return A_LR_2in1, A_SFHA_2in1, area_priceAcq_loadingFactor_LR, area_priceAcq_loadingFactor_SFHA, ho_priceAcq_LR, ho_priceAcq_SFHA_noLastEXP, ho_priceAcq_SFHA_withLastEXP    