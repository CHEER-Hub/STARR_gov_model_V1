#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 19 21:35:35 2024

@author: Jingya Wang
"""
import pandas as pd
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
                     zones, zoneNum_LR, hoNum_LR, hoNum_SFHA, county_zoneID, loss_merged):
    """ Calculate government grants allocation"""

    np.random.seed(4)
    X_LR = stru['stru']['X_LR'][0,0]
    X_SFHA = stru['stru']['X_SFHA'][0,0]
    ho_areaID_LR = np.squeeze(stru['stru']['ho_areaID_LR'][0,0])
    ho_areaID_SFHA = np.squeeze(stru['stru']['ho_areaID_SFHA'][0,0])
    hoNum_LR = len(ho_areaID_LR)
    hoNum_SFHA = len(ho_areaID_SFHA)

    A_LR_2in1Copy = A_LR_2in1.copy()
    A_SFHA_2in1Copy = A_SFHA_2in1.copy()
    R_LR_new_12in1_withGrantCopy = R_LR_new_12in1_withGrant.copy()
    R_SFHA_new_12in1_withGrantCopy = R_SFHA_new_12in1_withGrant.copy()
    ho_acqRecord_LRCopy = ho_acqRecord_LR.copy()
    ho_acqRecord_SFHACopy = ho_acqRecord_SFHA.copy()

    ho_A_LR_granted = np.zeros(hoNum_LR)
    ho_R_LR_granted = np.zeros(hoNum_LR)
    ho_A_SFHA_granted = np.zeros(hoNum_SFHA)
    ho_R_SFHA_granted = np.zeros(hoNum_SFHA)

    all_granted_rows = []

    for county_id in loss_merged['county_ID'].astype(int):

        county_budget = loss_merged.loc[loss_merged['county_ID'] == county_id, 'budget'].values[0]
        zone_ids = county_zoneID.loc[county_zoneID['county_ID'] == county_id, 'zone_ID'].values

        ind_areaARLH_unsorted = np.zeros((2 * len(zone_ids), 7))
        row_idx = 0
        for z in zone_ids:
            z_idx = int(z) - 1
            if z_idx < zoneNum_LR:
                mask_A = (ho_areaID_LR == z) & (A_LR_2in1[:, 0] > 0)
                if np.any(mask_A):
                    ind_areaARLH_unsorted[row_idx] = [
                        np.sum(A_LR_2in1[mask_A, 0]) / np.sum(A_LR_2in1[mask_A, 1]),
                        np.sum(A_LR_2in1[mask_A, 1]), -1, -1,
                        np.sum(mask_A), np.sum(X_LR[z_idx, :]), z
                    ]
                    row_idx += 1
                mask_R = (ho_areaID_LR == z) & (R_LR_new_12in1_withGrant[:, -3] > 0)
                if np.any(mask_R):
                    ind_areaARLH_unsorted[row_idx] = [
                        np.sum(R_LR_new_12in1_withGrant[mask_R, -3]) / np.sum(R_LR_new_12in1_withGrant[mask_R, -1]),
                        np.sum(R_LR_new_12in1_withGrant[mask_R, -1]), -1, 1,
                        np.sum(mask_R), np.sum(X_LR[z_idx, :]), z
                    ]
                    row_idx += 1
            else:
                mask_A = (ho_areaID_SFHA == z) & (A_SFHA_2in1[:, 0] > 0)
                if np.any(mask_A):
                    ind_areaARLH_unsorted[row_idx] = [
                        np.sum(A_SFHA_2in1[mask_A, 0]) / np.sum(A_SFHA_2in1[mask_A, 1]),
                        np.sum(A_SFHA_2in1[mask_A, 1]), 1, -1,
                        np.sum(mask_A), np.sum(X_SFHA[z_idx - zoneNum_LR, :]), z
                    ]
                    row_idx += 1
                mask_R = (ho_areaID_SFHA == z) & (R_SFHA_new_12in1_withGrant[:, -3] > 0)
                if np.any(mask_R):
                    ind_areaARLH_unsorted[row_idx] = [
                        np.sum(R_SFHA_new_12in1_withGrant[mask_R, -3]) / np.sum(R_SFHA_new_12in1_withGrant[mask_R, -1]),
                        np.sum(R_SFHA_new_12in1_withGrant[mask_R, -1]), 1, 1,
                        np.sum(mask_R), np.sum(X_SFHA[z_idx - zoneNum_LR, :]), z
                    ]
                    row_idx += 1

        ind_areaARLH_unsorted = ind_areaARLH_unsorted[:row_idx]
        ind_areaARLH_unsorted = ind_areaARLH_unsorted[~np.isnan(ind_areaARLH_unsorted[:, 0])]

        temp2 = ind_areaARLH_unsorted.copy()
        temp2[temp2[:, 3] == 1, 0] += 0.01
        ind_areaARLH_descend = temp2[np.argsort(temp2[:, 0])[::-1]]

        used_zone_ids = set()
        for row in ind_areaARLH_descend:
            z = int(row[6])
            if z in used_zone_ids:
                continue
            used_zone_ids.add(z)

            if row[1] <= county_budget:
                all_granted_rows.append(np.append(row, county_id))
                county_budget -= row[1]

                is_SFHA = (row[2] == 1)
                is_retrofit = (row[3] == 1)
                if is_SFHA:
                    if is_retrofit:
                        ho_ids = np.where((ho_areaID_SFHA == z) & (R_SFHA_new_12in1_withGrant[:, -3] > 0))[0]
                        ho_R_SFHA_granted[ho_ids] = 1
                    else:
                        ho_ids = np.where((ho_areaID_SFHA == z) & (A_SFHA_2in1[:, 0] > 0))[0]
                        ho_A_SFHA_granted[ho_ids] = 1
                else:
                    if is_retrofit:
                        ho_ids = np.where((ho_areaID_LR == z) & (R_LR_new_12in1_withGrant[:, -3] > 0))[0]
                        ho_R_LR_granted[ho_ids] = 1
                    else:
                        ho_ids = np.where((ho_areaID_LR == z) & (A_LR_2in1[:, 0] > 0))[0]
                        ho_A_LR_granted[ho_ids] = 1
            else:
                remaining_budget = county_budget
                is_SFHA = (row[2] == 1)
                is_retrofit = (row[3] == 1)
                if is_SFHA:
                    if is_retrofit:
                        ho_ids = np.where((ho_areaID_SFHA == z) & (R_SFHA_new_12in1_withGrant[:, -3] > 0))[0]
                        costs = R_SFHA_new_12in1_withGrant[ho_ids, -1]
                    else:
                        ho_ids = np.where((ho_areaID_SFHA == z) & (A_SFHA_2in1[:, 0] > 0))[0]
                        costs = A_SFHA_2in1[ho_ids, 1]
                else:
                    if is_retrofit:
                        ho_ids = np.where((ho_areaID_LR == z) & (R_LR_new_12in1_withGrant[:, -3] > 0))[0]
                        costs = R_LR_new_12in1_withGrant[ho_ids, -1]
                    else:
                        ho_ids = np.where((ho_areaID_LR == z) & (A_LR_2in1[:, 0] > 0))[0]
                        costs = A_LR_2in1[ho_ids, 1]

                sorted_idx = np.argsort(costs)
                total = 0
                selected = []
                for idx in sorted_idx:
                    if total + costs[idx] <= remaining_budget:
                        total += costs[idx]
                        selected.append(idx)
                    else:
                        break

                if is_SFHA:
                    if is_retrofit:
                        ho_R_SFHA_granted[ho_ids[selected]] = 1
                    else:
                        ho_A_SFHA_granted[ho_ids[selected]] = 1
                else:
                    if is_retrofit:
                        ho_R_LR_granted[ho_ids[selected]] = 1
                    else:
                        ho_A_LR_granted[ho_ids[selected]] = 1

                updated_row = row.copy()
                updated_row[1] = total
                updated_row[4] = len(selected)
                all_granted_rows.append(np.append(updated_row, county_id))
                county_budget -= total

    ind_areaARLH_descend_granted_all = np.array(all_granted_rows)
    ind_areaARLH_descend_granted_all = ind_areaARLH_descend_granted_all[ind_areaARLH_descend_granted_all[:, 1] > 0]

    RES = {'ind_areaARLH_descend_granted': ind_areaARLH_descend_granted_all}

    A_LR_2in1Copy *= ho_A_LR_granted.reshape(-1,1)
    A_SFHA_2in1Copy *= ho_A_SFHA_granted.reshape(-1,1)
    ho_acqRecord_LRCopy[A_LR_2in1Copy[:, 0] != 0] = year
    ho_acqRecord_SFHACopy[A_SFHA_2in1Copy[:, 0] != 0] = year
    R_LR_new_12in1_withGrantCopy *= ho_R_LR_granted.reshape(-1,1) * (ho_acqRecord_LRCopy == 0).reshape(-1,1)
    R_SFHA_new_12in1_withGrantCopy *= ho_R_SFHA_granted.reshape(-1,1) * (ho_acqRecord_SFHACopy == 0).reshape(-1,1)
    
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
    
    granted_areaID = ind_areaARLH_descend_granted_all[:, [2, 3, 6]]
    temp2 = granted_areaID[granted_areaID[:, 0] == 1]  # SFHA
    temp2a = temp2[temp2[:, 1] == -1][:, 2]  # SFHA Acq
    acquired_areaID_SFHA = temp2a

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

    return (A_LR_2in1Copy,A_SFHA_2in1Copy,
    R_LR_new_12in1_withGrantCopy,R_SFHA_new_12in1_withGrantCopy,
    price_Acq_Ret_aveChosen,acquired_areaID_SFHA,
    RES,A_grant,R_grant)
    
        
    
    
    
    
    
    
    
    
    