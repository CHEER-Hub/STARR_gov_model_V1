#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 16 22:03:31 2024

@author: Jingya Wang
"""
import numpy as np

def Func_cal_exp_fld(scenario, hurr_sim, stru, hoNum_LR, hoNum_SFHA, zones, hurr, years):
    """Calculate hurricane flood experience"""
    
    ########
    ## Identify flood losses by zones and by hurricanes with variables “ho_perhurrLoss_fld_LR_Binary” and “ho_perhurrLoss_fld_SFHA_Binary” 
    ##(0: no flood loss in zone under current hurricane event; 1: suffering flood loss in zone under current hurricane event) 
    ho_areaID_LR = np.squeeze(stru['stru']['ho_areaID_LR'][0,0])  # area ID
    ho_areaID_SFHA = np.squeeze(stru['stru']['ho_areaID_SFHA'][0,0])
    ho_perhurrLoss_fld_LR_Binary = np.zeros((hoNum_LR, hurr))  # 649102*30
    ho_perhurrLoss_fld_SFHA_Binary = np.zeros((hoNum_SFHA, hurr))  # 282890*30
    hurr_fld = stru['stru']['hurr_fld'][0,0]

    for i in range(zones):  # area amount 1509 (include empty)
        for j in range(hurr):  # hurricane
            if hurr_fld[i, j] > 0:
                ho_perhurrLoss_fld_LR_Binary[ho_areaID_LR == i+1, j] = 1  # identify flood loss; 1: flood loss, 0: no flood loss
                ho_perhurrLoss_fld_SFHA_Binary[ho_areaID_SFHA == i+1, j] = 1

    ho_lastEXP_fld_LR_SCEN30YEAR = np.zeros((hoNum_LR, years))
    ho_lastEXP_fld_SFHA_SCEN30YEAR = np.zeros((hoNum_SFHA, years))
    ho_lastEXP_fld_LR_SCEN30YEAR[:, 0] = np.squeeze(stru['stru']['ho_lastEXP_fld_LR_year1'][0,0])  # initial last hurricane experienced for the first year
    ho_lastEXP_fld_SFHA_SCEN30YEAR[:, 0] = np.squeeze(stru['stru']['ho_lastEXP_fld_SFHA_year1'][0,0])
    ho_cumEXP_fld_LR_SCEN30YEAR = np.zeros((hoNum_LR, years))
    ho_cumEXP_fld_SFHA_SCEN30YEAR = np.zeros((hoNum_SFHA, years))
    ho_cumEXP_fld_LR_SCEN30YEAR[:, 0] = np.squeeze(stru['stru']['ho_cumEXP_fld_LR_year1'][0,0])  # initial cumulated hurricane experienced for the first year
    ho_cumEXP_fld_SFHA_SCEN30YEAR[:, 0] = np.squeeze(stru['stru']['ho_cumEXP_fld_SFHA_year1'][0,0])
    

    #########
    ##•	Calculate and record the last hurricane flood experience (“ho_lastEXP_fld_LR_SCEN30YEAR” and “ho_lastEXP_fld_SFHA_SCEN30YEAR”) \
    ## and cumulated numbers of hurricanes experienced (“ho_cumEXP_fld_LR_SCEN30YEAR” and “ho_cumEXP_fld_SFHA_SCEN30YEAR”) by households and by years
    for i in range(years-1): # from year 1 to year 29
        temp = hurr_sim[scenario-1, i]  # hurricane event for current scenario at year i
        hs = np.mod(np.floor([temp/1e8, temp/1e6, temp/1e4, temp/1e2, temp]), 100)

        # Filter out elements that are zero
        hs = hs[hs != 0]
        hs = [int(x-1) for x in hs]
        
        if len(hs) > 1:
            temp_LR = np.sum(ho_perhurrLoss_fld_LR_Binary[:, hs], axis=1)  # sum col
            temp_LR = np.squeeze(temp_LR)            
            temp_SFHA = np.sum(ho_perhurrLoss_fld_SFHA_Binary[:, hs], axis=1)  # sum hurricane experienced flood loss
            temp_SFHA = np.squeeze(temp_SFHA)
            ho_cumEXP_fld_LR_SCEN30YEAR[:, i+1] = ho_cumEXP_fld_LR_SCEN30YEAR[:, i] + temp_LR # cumulated hurriances flood loss for LR
            ho_cumEXP_fld_SFHA_SCEN30YEAR[:, i+1] = ho_cumEXP_fld_SFHA_SCEN30YEAR[:, i] + temp_SFHA # cumulated flood loss for SFHA
            temp_LR = temp_LR > 0  # 1: last year experience hurricane; 0: otherwise
            temp_SFHA = temp_SFHA > 0
            ho_lastEXP_fld_LR_SCEN30YEAR[:, i+1] = ho_lastEXP_fld_LR_SCEN30YEAR[:, i] * (1 - temp_LR) + 1 # how long since last experienced hurricane
            ho_lastEXP_fld_SFHA_SCEN30YEAR[:, i+1] = ho_lastEXP_fld_SFHA_SCEN30YEAR[:, i] * (1 - temp_SFHA) + 1
            
        elif len(hs) == 1:
            temp_LR = ho_perhurrLoss_fld_LR_Binary[:, hs]  # sum col
            temp_LR = np.squeeze(temp_LR)
            
            temp_SFHA = ho_perhurrLoss_fld_SFHA_Binary[:, hs]  # sum hurricane experienced flood loss
            temp_SFHA = np.squeeze(temp_SFHA)
            
            ho_cumEXP_fld_LR_SCEN30YEAR[:, i+1] = ho_cumEXP_fld_LR_SCEN30YEAR[:, i] + temp_LR # cumulated hurriances flood loss for LR
            

            ho_cumEXP_fld_SFHA_SCEN30YEAR[:, i+1] = ho_cumEXP_fld_SFHA_SCEN30YEAR[:, i] + temp_SFHA # cumulated flood loss for SFHA

            
            temp_LR = temp_LR > 0  # 1: last year experience hurricane; 0: otherwise
            temp_SFHA = temp_SFHA > 0
            ho_lastEXP_fld_LR_SCEN30YEAR[:, i+1] = ho_lastEXP_fld_LR_SCEN30YEAR[:, i] * (1 - temp_LR) + 1 # how long since last experienced hurricane
            ho_lastEXP_fld_SFHA_SCEN30YEAR[:, i+1] = ho_lastEXP_fld_SFHA_SCEN30YEAR[:, i] * (1 - temp_SFHA) + 1
            
        else: # no hurriances experienced
            ho_lastEXP_fld_LR_SCEN30YEAR[:, i+1] = ho_lastEXP_fld_LR_SCEN30YEAR[:, i] + 1
            ho_lastEXP_fld_SFHA_SCEN30YEAR[:, i+1] = ho_lastEXP_fld_SFHA_SCEN30YEAR[:, i] + 1
            ho_cumEXP_fld_LR_SCEN30YEAR[:, i+1] = ho_cumEXP_fld_LR_SCEN30YEAR[:, i]
            ho_cumEXP_fld_SFHA_SCEN30YEAR[:, i+1] = ho_cumEXP_fld_SFHA_SCEN30YEAR[:, i]
    
    return ho_lastEXP_fld_LR_SCEN30YEAR, ho_lastEXP_fld_SFHA_SCEN30YEAR, ho_cumEXP_fld_LR_SCEN30YEAR, ho_cumEXP_fld_SFHA_SCEN30YEAR



