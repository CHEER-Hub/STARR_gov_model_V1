import numpy as np

def Func_cal_acq_SPforLowIncome(year, priceAcq_alpha, priceAcq_beta, aftershock, budget,
                                ho_acqRecord_SFHA, stru,
                                ho_aveLoss_fld_SFHA, ho_aveLoss_wd_SFHA,
                                ho_lastEXP_fld_SFHA_SCENYEAR, real_index_L, PARA_gov_benefitratio_forlow,
                                hoNum_LR, zoneNum_LR, zoneNum_SFHA, years):
    """ Calculate special acquisition decisions for low-income group """
    
    # Initialize outputs
    A_LR_2in1 = np.zeros((hoNum_LR, 2))
    area_priceAcq_loadingFactor_LR = np.zeros(zoneNum_LR)
    ho_priceAcq_LR = np.zeros((hoNum_LR, 2))
    
    ho_acqRecord_SFHACopy = ho_acqRecord_SFHA.copy()
    ho_lastEXP_fld_SFHA_SCENYEARCopy = ho_lastEXP_fld_SFHA_SCENYEAR.copy()
    real_index_LCopy = real_index_L.copy()
    
    
    
    # Seed the random number generator
    np.random.seed(1)
    
    # Preprocess ho_lastEXP_fld_SFHA_SCENYEAR
    ho_lastEXP_fld_SFHA_SCENYEARCopy[ho_lastEXP_fld_SFHA_SCENYEARCopy != 1] = 0
    
    # Extract values from structure
    homevalue_SFHA = np.squeeze(stru['stru']['homevalue_SFHA'][0,0])
    hoNum_SFHA = np.squeeze(stru['stru']['hoNum_SFHA'][0,0])
    ho_areaID_SFHA = np.squeeze(stru['stru']['ho_areaID_SFHA'][0,0])
    temp_ho_aveLoss_fldwd_SFHA = np.squeeze(ho_aveLoss_fld_SFHA + ho_aveLoss_wd_SFHA)
    
    ########
    ## Calculate zone level acquisition loading factors (“area_priceAcq_loadingFactor_SFHA”) and zone level acquisition prices (“ho_priceAcq_SFHA”)
    
    # Calculate area_priceAcq_loadingFactor_SFHA
    area_priceAcq_loadingFactor_SFHA = np.zeros(zoneNum_SFHA)
    for i in range(zoneNum_SFHA):
        temp = np.where((ho_areaID_SFHA == (i+1) + zoneNum_LR) & (ho_acqRecord_SFHACopy == 0))[0] #satisfy both conditions; price for home owners low risk [504,1509]
        area_priceAcq_loadingFactor_SFHA[i] = years * np.sum(temp_ho_aveLoss_fldwd_SFHA[temp]) / np.sum(homevalue_SFHA[temp])

    area_priceAcq_loadingFactor_SFHA[np.isnan(area_priceAcq_loadingFactor_SFHA)] = 0 # some area no HO, 0/0=NaN; set to 0
    
    ho_priceAcq_SFHA = np.zeros(hoNum_SFHA)
    for i in range(zoneNum_SFHA):
        temp = ho_areaID_SFHA == (i+1) + zoneNum_LR
        ho_priceAcq_SFHA[temp] = priceAcq_alpha + priceAcq_beta * area_priceAcq_loadingFactor_SFHA[i]  #equation of price acq.
    
    ########
    ## Calculate low-income households’ acquisition decisions. (accept: if the ratio of 30-years total losses to acquisition grants is larger than preset benefit ratio; decline: otherwise)
    
    # Update real_index_L
    real_index_LCopy = real_index_LCopy[real_index_LCopy > hoNum_LR] - hoNum_LR -1
    
    # Process low-income homeowners
    for i in real_index_LCopy:
        i = int(i)  # Ensure the index is an integer
        if ho_acqRecord_SFHACopy[i] == 0:
            total_loss = (ho_aveLoss_fld_SFHA[i] + ho_aveLoss_wd_SFHA[i]) * years
            benefit_ratio = homevalue_SFHA[i] * ho_priceAcq_SFHA[i]
            if benefit_ratio != 0 and total_loss / benefit_ratio >= PARA_gov_benefitratio_forlow:
                ho_acqRecord_SFHACopy[i] = -1
    
    # Candidate homeowner IDs
    candidate_hoID_SFHA = ho_acqRecord_SFHACopy.copy()
    
    ########
    ## 	Record and return low-income households acquisition decisions using variables “A_LR_2in1” and “A_SFHA_2in1”
    
    # Calculate A_SFHA_2in1
    A_SFHA_2in1 = np.zeros((hoNum_SFHA, 2))
    A_SFHA_2in1[:, 0] = np.squeeze((ho_aveLoss_fld_SFHA + ho_aveLoss_wd_SFHA)) * (candidate_hoID_SFHA == -1) * years
    A_SFHA_2in1[:, 1] = homevalue_SFHA * ho_priceAcq_SFHA * (candidate_hoID_SFHA == -1)
    
    # Calculate ho_priceAcq_SFHA_noLastEXP and ho_priceAcq_SFHA_withLastEXP
    ho_priceAcq_SFHA_noLastEXP = ho_priceAcq_SFHA * (candidate_hoID_SFHA == -1) * (ho_lastEXP_fld_SFHA_SCENYEARCopy == 0)
    ho_priceAcq_SFHA_withLastEXP = ho_priceAcq_SFHA * (candidate_hoID_SFHA == -1) * (ho_lastEXP_fld_SFHA_SCENYEARCopy == 1)
    
    return A_LR_2in1, A_SFHA_2in1, area_priceAcq_loadingFactor_LR, area_priceAcq_loadingFactor_SFHA, ho_priceAcq_LR, ho_priceAcq_SFHA_noLastEXP, ho_priceAcq_SFHA_withLastEXP
