#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 19 21:35:35 2024

@author: Jingya Wang
"""

import numpy as np

def Func_cal_ranking(year, budget, stru, A_LR_2in1, A_SFHA_2in1,
                     R_LR_new_12in1_withGrant, R_SFHA_new_12in1_withGrant,
                     ho_acqRecord_LR, ho_acqRecord_SFHA, priceAcq_alpha,
                     priceAcq_beta, priceRet_alpha, priceRet_beta,
                     area_priceAcq_loadingFactor_LR, area_priceAcq_loadingFactor_SFHA,
                     area_priceRet_loadingFactor_LR, area_priceRet_loadingFactor_SFHA,
                     ho_priceAcq_LR, ho_priceAcq_SFHA_noLastEXP, ho_priceAcq_SFHA_withLastEXP,
                     ho_priceRet_LR, ho_priceRet_SFHA, index_L_allcounties,
                     index_M_allcounties, index_H_allcounties,
                     zones, zoneNum_LR, hoNum_LR, hoNum_SFHA):
    """ Calculate government grants allocation"""

    np.random.seed(4)

    #hoNum_LR = stru['hoNum_LR']
    #hoNum_SFHA = stru['hoNum_SFHA']
    X_LR = stru['stru']['X_LR'][0,0]  # 503*1536
    X_SFHA = stru['stru']['X_SFHA'][0,0]  # 1006*1536

    # make copies of everything whose values might change
    A_LR_2in1Copy = A_LR_2in1.copy()
    A_SFHA_2in1Copy = A_SFHA_2in1.copy()
    R_LR_new_12in1_withGrantCopy = R_LR_new_12in1_withGrant.copy()
    R_SFHA_new_12in1_withGrantCopy = R_SFHA_new_12in1_withGrant.copy()
    ho_acqRecord_LRCopy = ho_acqRecord_LR.copy()
    ho_acqRecord_SFHACopy = ho_acqRecord_SFHA.copy()
    
    
    
    A_LR = A_LR_2in1Copy.copy()
    A_SFHA = A_SFHA_2in1Copy.copy()
    R_LR = R_LR_new_12in1_withGrantCopy[:, [-3, -1]]
    R_SFHA = R_SFHA_new_12in1_withGrantCopy[:, [-3, -1]]

    ho_areaID_LR = np.squeeze(stru['stru']['ho_areaID_LR'][0,0])
    ho_areaID_SFHA = np.squeeze(stru['stru']['ho_areaID_SFHA'][0,0])

    ind_areaARLH_unsorted = np.zeros((2 * zones, 7))
    
    ########
    ## Calculate zone level acquisition and retrofit information. Variable “ind_areaARLH_unsorted” records corresponding information, including 
    ##(i) rate of returns, (ii) costs, (iii) in low- or high-risk zone, (iv) doing acquisition or retrofit, 
    ##(v) number of households doing acquisition/retrofit in the current zone, (vi) number of households in the current zone, (vii) zone ID
    for i in range(zones):
        if i < zoneNum_LR: # low-risk zones

            logical_A_LR = (ho_areaID_LR == (i+1)) & (A_LR[:, 0] > 0) # acquisition savings; >0 means do acq
            if np.sum(logical_A_LR) > 0: # at least one building in this area with positive acquisition saving
                double_A_LR = logical_A_LR == 1
                ind_areaARLH_unsorted[i * 2, :] = [
                    np.sum(A_LR[double_A_LR, 0]) / np.sum(A_LR[double_A_LR, 1]),
                    np.sum(A_LR[double_A_LR, 1]),
                    -1,
                    -1,
                    np.sum(double_A_LR),
                    np.sum(X_LR[i, :]),
                    i+1
                ]
            logical_R_LR = (ho_areaID_LR == (i+1)) & (R_LR[:, 0] > 0) # loss benefit; >0 means do ret
            if np.sum(logical_R_LR) > 0: # at least one building in this area with positive retrofit saving
                double_R_LR = logical_R_LR == 1
                ind_areaARLH_unsorted[i * 2 + 1, :] = [
                    np.sum(R_LR[double_R_LR, 0]) / np.sum(R_LR[double_R_LR, 1]),
                    np.sum(R_LR[double_R_LR, 1]),
                    -1,
                    1,
                    np.sum(logical_R_LR),
                    np.sum(X_LR[i, :]),
                    i+1
                ]
        else: # high-risk zones
            logical_A_SFHA = (ho_areaID_SFHA == (i+1)) & (A_SFHA[:, 0] > 0)
            if np.sum(logical_A_SFHA) > 0:
                double_A_SFHA = logical_A_SFHA == 1
                ind_areaARLH_unsorted[i * 2, :] = [
                    np.sum(A_SFHA[double_A_SFHA, 0]) / np.sum(A_SFHA[double_A_SFHA, 1]),
                    np.sum(A_SFHA[double_A_SFHA, 1]),
                    1,
                    -1,
                    np.sum(logical_A_SFHA),
                    np.sum(X_SFHA[i - zoneNum_LR, :]),
                    i+1
                ]
            logical_R_SFHA = (ho_areaID_SFHA == (i+1)) & (R_SFHA[:, 0] > 0)
            if np.sum(logical_R_SFHA) > 0:
                double_R_SFHA = logical_R_SFHA == 1
                ind_areaARLH_unsorted[i * 2 + 1, :] = [
                    np.sum(R_SFHA[double_R_SFHA, 0]) / np.sum(R_SFHA[double_R_SFHA, 1]),
                    np.sum(R_SFHA[double_R_SFHA, 1]),
                    1,
                    1,
                    np.sum(logical_R_SFHA),
                    np.sum(X_SFHA[i - zoneNum_LR, :]),
                    i+1
                ]

    ind_areaARLH_unsorted = ind_areaARLH_unsorted[~np.isnan(ind_areaARLH_unsorted[:, 0]), :]
    
    #######
    ##•	Increase retrofit rate of returns a bit when acquisition rate of returns is similar to retrofit rate of returns
    ##•	Rank zones based on rate of returns

    temp = np.where(ind_areaARLH_unsorted[:, 3] == 1)[0]
    temp2 = ind_areaARLH_unsorted.copy()
    temp2[temp, 0] += 0.01

    # Sort by return rate in descending order
    sorted_indices = np.argsort(temp2[:, 0])[::-1]
    
    # Apply the sorting to ind_areaARLH_unsorted
    ind_areaARLH_descend = ind_areaARLH_unsorted[sorted_indices]
    
    # Remove rows where the last column (area ID) is zero (indicating no HO or no action)
    ind_areaARLH_descend = ind_areaARLH_descend[ind_areaARLH_descend[:, -1] != 0]
    
    # Store the results in a dictionary
    RES = {'ind_areaARLH_descend': ind_areaARLH_descend.copy()}
    
    ########
    ## Based on the ranked order, government offers acquisition/retrofit grants for the selected zone and cumulates cost spent. Keep offering until running out of budget
    temp = 0
    num_granted = 0
    for i in range(ind_areaARLH_descend.shape[0]):
        if temp + ind_areaARLH_descend[i, 1] < budget: # cumulative expense based on sorted return rate.
            temp += ind_areaARLH_descend[i, 1]
            num_granted += 1
        else:
            num_granted += 1 # use for one more incomplete area
            break
        
    ########
    ## Although the last area cannot be fully covered, a portion can be randomly selected until no new houses can be chosen
    ## The last area always needs special handling
    ## Reallocate grants to ensure that zones can only offer either acquisition or retrofit grants
    while len(np.unique(ind_areaARLH_descend[:num_granted, 6])) != num_granted:
        temp_del = []
        for i in range(num_granted):
            if ind_areaARLH_descend[i, 3] == 1 and len(np.where(ind_areaARLH_descend[:num_granted, 6] == ind_areaARLH_descend[i, 6])[0]) > 1:
                temp_del.append(i)

        ind_areaARLH_descend = np.delete(ind_areaARLH_descend, temp_del, axis=0)

        temp = 0
        num_granted = 0
        for i in range(ind_areaARLH_descend.shape[0]):
            if temp + ind_areaARLH_descend[i, 1] < budget:
                temp += ind_areaARLH_descend[i, 1]
                num_granted += 1
            else:
                num_granted += 1
                break
    
    ########
    ## •Set acquisition and retrofit decisions to 0 if budget cannot cover any complete zone (complete zone: government is able to offer enough grants to cover all households in that zone who are willing to accept acquisition or retrofit offers; incomplete zone: otherwise)
    ##•	Record acquisition and retrofit allocation information for each household (if the household lives in a zone that offers acquisition/retrofit, set corresponding indicator to 1)
    ind_areaARLH_descend_granted = ind_areaARLH_descend[:num_granted, :] # record granted area and info
    lastArea_budget = budget - temp # money for last area (not enough for whole area)

    if num_granted == 0: # budget can't afford any area; num_granted can't be 0! always add 1 before finish the for loop!
        A_LR_2in1Copy = np.zeros_like(A_LR_2in1Copy)
        A_SFHA_2in1Copy = np.zeros_like(A_SFHA_2in1Copy)
        R_LR_new_12in1_withGrantCopy = np.zeros_like(R_LR_new_12in1_withGrantCopy)
        R_SFHA_new_12in1_withGrantCopy = np.zeros_like(R_SFHA_new_12in1_withGrantCopy)
        price_Acq_Ret_aveChosen = [-1, -1, -1, -1, -1]
        acquired_areaID_SFHA = []
        RES['ind_areaARLH_descend_granted'] = []
    else:
        ho_A_LR_granted = np.zeros(hoNum_LR)
        ho_R_LR_granted = np.zeros(hoNum_LR)
        ho_A_SFHA_granted = np.zeros(hoNum_SFHA)
        ho_R_SFHA_granted = np.zeros(hoNum_SFHA)

    for i in range(num_granted - 1): # excluding the lst zone

        if ind_areaARLH_descend_granted[i, 3] == -1: # acq
            ho_A_LR_granted += (ho_areaID_LR == ind_areaARLH_descend_granted[i, 6]) # HO in area i have acq offer. set to 1
            ho_A_SFHA_granted += (ho_areaID_SFHA == ind_areaARLH_descend_granted[i, 6])
        else:
            ho_R_LR_granted += (ho_areaID_LR == ind_areaARLH_descend_granted[i, 6]) # HO in area i have ret offer. set to 1.
            ho_R_SFHA_granted += (ho_areaID_SFHA == ind_areaARLH_descend_granted[i, 6])

    ## For the last incomplete zone, assign remaining budget to partial households in that zone who are willing to do acquisition/retrofit

    if ind_areaARLH_descend_granted[-1, 2] == -1 and ind_areaARLH_descend_granted[-1, 3] == -1:  # LR, acquisition
        print('LR acq')
        temp = np.where(ho_areaID_LR == ind_areaARLH_descend_granted[-1, 6])[0]  # HO index in last area
        temp_remained = []
        for i in range(len(temp)):  # num of HO in last area
            if A_LR[temp[i], 0] != 0:  # acq savings > 0, means do acq.
                temp_remained.append(i)
        temp = temp[temp_remained]  # only focus on HO index do acq; temp stores HO index
        temp_cost = 0
        temp_remained = []
        for i in range(len(temp)):
            if temp_cost + A_LR[temp[i], 1] <= lastArea_budget:
                temp_remained.append(temp[i])
                temp_cost += A_LR[temp[i], 1]
            else:
                break
        temp_array = np.zeros((hoNum_LR, 1))  # num HO*1
        temp_array[temp_remained] = 1  # only focus on HO index that selected to do acq.
        ho_A_LR_granted += np.squeeze(temp_array)  # all selected HO to do acq. and ret.
        add_lastArea_to_list = A_LR[temp_remained, :]
    
    elif ind_areaARLH_descend_granted[-1, 2] == 1 and ind_areaARLH_descend_granted[-1, 3] == -1:  # SFHA, acquisition
        print('SFHA, acq')
        temp = np.where(ho_areaID_SFHA == ind_areaARLH_descend_granted[-1, 6])[0]
        temp_remained = []
        for i in range(len(temp)):
            if A_SFHA[temp[i], 0] != 0:
                temp_remained.append(i)
        temp = temp[temp_remained]
        temp_cost = 0
        temp_remained = []
        for i in range(len(temp)):
            if temp_cost + A_SFHA[temp[i], 1] <= lastArea_budget:
                temp_remained.append(temp[i])
                temp_cost += A_SFHA[temp[i], 1]
            else:
                break
        temp_array = np.zeros((hoNum_SFHA, 1))
        temp_array[temp_remained] = 1
        ho_A_SFHA_granted += np.squeeze(temp_array)
        add_lastArea_to_list = A_SFHA[temp_remained, :]
    
    elif ind_areaARLH_descend_granted[-1, 2] == -1 and ind_areaARLH_descend_granted[-1, 3] == 1:  # LR, retrofit
        print('LR ret')
        temp = np.where(ho_areaID_LR == ind_areaARLH_descend_granted[-1, 6])[0]
        temp_remained = []
        for i in range(len(temp)):
            if R_LR[temp[i], 1] != 0:
                temp_remained.append(i)
        temp = temp[temp_remained]
        temp_cost = 0
        temp_remained = []
        for i in range(len(temp)):
            if temp_cost + R_LR[temp[i], 1] <= lastArea_budget:
                temp_remained.append(temp[i])
                temp_cost += R_LR[temp[i], 1]
            else:
                break
        temp_array = np.zeros((hoNum_LR, 1))
        temp_array[temp_remained] = 1
        ho_R_LR_granted += np.squeeze(temp_array)
        add_lastArea_to_list = R_LR[temp_remained, :]
    
    elif ind_areaARLH_descend_granted[-1, 2] == 1 and ind_areaARLH_descend_granted[-1, 3] == 1:  # SFHA, retrofit
        print('SFHA ret')
        temp = np.where(ho_areaID_SFHA == ind_areaARLH_descend_granted[-1, 6])[0]
        temp_remained = []
        for i in range(len(temp)):
            if R_SFHA[temp[i], 1] != 0:
                temp_remained.append(i)
        temp = temp[temp_remained]
        temp_cost = 0
        temp_remained = []
        for i in range(len(temp)):
            if temp_cost + R_SFHA[temp[i], 1] <= lastArea_budget:
                temp_remained.append(temp[i])
                temp_cost += R_SFHA[temp[i], 1]
            else:
                break
        temp_array = np.zeros((hoNum_SFHA, 1))
        temp_array[temp_remained] = 1
        ho_R_SFHA_granted += np.squeeze(temp_array)
        add_lastArea_to_list = R_SFHA[temp_remained, :]

      ########
      ## •	Update zone level acquisition and retrofit information for the last incomplete zone
      ## •	Update and return households’ final acquisition and retrofit decisions
    # Calculate area-wise metrics
    ind_areaARLH_descend_granted[-1, :] = np.array([
        np.sum(add_lastArea_to_list[:, 0]) / np.sum(add_lastArea_to_list[:, 1]),
        np.sum(add_lastArea_to_list[:, 1]),
        *ind_areaARLH_descend_granted[-1, [2, 3]],
        add_lastArea_to_list.shape[0],
        *ind_areaARLH_descend_granted[-1, [5, 6]]
    ])
    
    if np.isnan(ind_areaARLH_descend_granted[-1, 0]):  # Check for NaN
        ind_areaARLH_descend_granted = np.delete(ind_areaARLH_descend_granted, -1, axis=0)

    
    # Update A_LR and A_SFHA based on who actually gets the grant
    A_LR_2in1Copy *= ho_A_LR_granted.reshape(-1,1)
    A_SFHA_2in1Copy *= ho_A_SFHA_granted.reshape(-1,1)
    
    # Update ho_acqRecord_LRCopy and ho_acqRecord_SFHACopy with the current year
    ho_acqRecord_LRCopy[A_LR_2in1Copy[:, 0] != 0] = year
    ho_acqRecord_SFHACopy[A_SFHA_2in1Copy[:, 0] != 0] = year
    
    # Update R_LR_new_12in1_withGrantCopy and R_SFHA_new_12in1_withGrantCopy
    R_LR_new_12in1_withGrantCopy *= (ho_R_LR_granted.reshape(-1,1)) * ((ho_acqRecord_LRCopy == 0).reshape(-1,1))
    R_SFHA_new_12in1_withGrantCopy *= (ho_R_SFHA_granted.reshape(-1,1)) * ((ho_acqRecord_SFHACopy == 0).reshape(-1,1))

    
    #######
    ## Calculate average acquisition and retrofit prices for households who are chosen to receive grants

    # Cost for LMH
    L_LR_index = index_L_allcounties[index_L_allcounties <= hoNum_LR] - 1
    M_LR_index = index_M_allcounties[index_M_allcounties <= hoNum_LR] - 1
    H_LR_index = index_H_allcounties[index_H_allcounties <= hoNum_LR] - 1
    L_SFHA_index = index_L_allcounties[index_L_allcounties > hoNum_LR] - hoNum_LR - 1
    M_SFHA_index = index_M_allcounties[index_M_allcounties > hoNum_LR] - hoNum_LR - 1
    H_SFHA_index = index_H_allcounties[index_H_allcounties > hoNum_LR] - hoNum_LR - 1
    
    A_grant = [A_LR_2in1Copy[L_LR_index, 1], A_LR_2in1Copy[M_LR_index, 1], A_LR_2in1Copy[H_LR_index, 1],
               A_SFHA_2in1Copy[L_SFHA_index, 1], A_SFHA_2in1Copy[M_SFHA_index, 1], A_SFHA_2in1Copy[H_SFHA_index, 1]]
    
    R_grant = [R_LR_new_12in1_withGrantCopy[L_LR_index, 11], R_LR_new_12in1_withGrantCopy[M_LR_index, 11],
               R_LR_new_12in1_withGrantCopy[H_LR_index, 11], R_SFHA_new_12in1_withGrantCopy[L_SFHA_index, 11],
               R_SFHA_new_12in1_withGrantCopy[M_SFHA_index, 11], R_SFHA_new_12in1_withGrantCopy[H_SFHA_index, 11]]
    
    # RES attribute
    RES['ind_areaARLH_descend_granted'] = ind_areaARLH_descend_granted
    
    # Calculate average costs
    priceAcq_aveChosenHo_LR = 0
    priceAcq_aveChosenHo_SFHA_noLastEXP = 0
    priceAcq_aveChosenHo_SFHA_withLastEXP = 0
    priceRet_aveChosenHo_LR = 0
    priceRet_aveChosenHo_SFHA = 0
    
    temp = ho_priceAcq_LR * ho_A_LR_granted.reshape(-1,1)
    if np.sum(temp) > 0:
        priceAcq_aveChosenHo_LR = np.mean((ho_priceAcq_LR.reshape(-1,1))[temp > 0])
    
    temp = ho_priceAcq_SFHA_noLastEXP * ho_A_SFHA_granted
    if np.sum(temp) > 0:
        priceAcq_aveChosenHo_SFHA_noLastEXP = np.mean((ho_priceAcq_SFHA_noLastEXP)[temp > 0])
    
    temp = ho_priceAcq_SFHA_withLastEXP * ho_A_SFHA_granted
    if np.sum(temp) > 0:
        priceAcq_aveChosenHo_SFHA_withLastEXP = np.mean(ho_priceAcq_SFHA_withLastEXP[temp > 0])
    
    temp = ho_priceRet_LR * ho_R_LR_granted * (ho_acqRecord_LRCopy == 0)
    if np.sum(temp) > 0:
        priceRet_aveChosenHo_LR = np.mean((ho_priceRet_LR)[temp > 0])
    
    temp = ho_priceRet_SFHA * (ho_R_SFHA_granted) * (ho_acqRecord_SFHACopy == 0)
    if np.sum(temp) > 0:
        priceRet_aveChosenHo_SFHA = np.mean((ho_priceRet_SFHA)[temp > 0])
    
    price_Acq_Ret_aveChosen = [priceAcq_aveChosenHo_LR, priceAcq_aveChosenHo_SFHA_noLastEXP,
                                priceAcq_aveChosenHo_SFHA_withLastEXP, priceRet_aveChosenHo_LR,
                                priceRet_aveChosenHo_SFHA]
    
    granted_areaID = ind_areaARLH_descend_granted[:, [2, 3, 6]]
    temp2 = granted_areaID[granted_areaID[:, 0] == 1]  # SFHA
    temp2a = temp2[temp2[:, 1] == -1][:, 2]  # SFHA Acq
    acquired_areaID_SFHA = temp2a
    

    return (A_LR_2in1Copy,A_SFHA_2in1Copy,
    R_LR_new_12in1_withGrantCopy,R_SFHA_new_12in1_withGrantCopy,
    price_Acq_Ret_aveChosen,acquired_areaID_SFHA,
    RES,A_grant,R_grant)
    
        
    
    
    
    
    
    
    
    
    