#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 20 17:50:21 2024

@author: Jingya Wang
"""

import numpy as np


def Func_imp_retrofit(ho_acqRecord_LR, ho_acqRecord_SFHA, R_LR_new_12in1_withGrant, R_SFHA_new_12in1_withGrant,
                      temp_inv_LR_dynamic3, temp_inv_SFHA_dynamic3, stru, L_au3D_fld, L_au3D_wd,
                      L_au4D_fld, L_au4D_wd, ho_aveLoss_fld_LR, ho_aveLoss_fld_SFHA, ho_perhurrLoss_fld_LR, ho_perhurrLoss_fld_SFHA,
                      ho_aveLoss_wd_LR, ho_aveLoss_wd_SFHA, ho_perhurrLoss_wd_LR, ho_perhurrLoss_wd_SFHA,
                      zoneNum_LR, hoNum_LR, hoNum_SFHA):
    """ Update building inventory and losses information """
    
    #hoNum_LR = stru['stru']['hoNum_LR'][0,0]
    #hoNum_SFHA = stru['hoNum_SFHA']
    inv_LR_dynamic = stru['stru']['inv_LR'][0,0]
    inv_SFHA_dynamic = stru['stru']['inv_SFHA'][0,0]
    
    acquisition_flag_LR = (ho_acqRecord_LR == 0)
    acquisition_flag_SFHA = (ho_acqRecord_SFHA == 0)
    
    
    R_LR_new_12in1_withGrantCopy = R_LR_new_12in1_withGrant.copy()
    R_SFHA_new_12in1_withGrantCopy = R_SFHA_new_12in1_withGrant.copy()
    
    
    R_LR_new_12in1_withGrantCopy *= acquisition_flag_LR.reshape(-1,1)
    R_SFHA_new_12in1_withGrantCopy *= acquisition_flag_SFHA.reshape(-1,1)
    
    ho_aveLoss_fld_LR *= acquisition_flag_LR.reshape(-1,1)
    ho_aveLoss_fld_SFHA *= acquisition_flag_SFHA.reshape(-1,1)
    ho_perhurrLoss_fld_LR *= acquisition_flag_LR.reshape(-1,1)
    ho_perhurrLoss_fld_SFHA *= acquisition_flag_SFHA.reshape(-1,1)
    
    #######
    ## Retrieve 8 types of final retrofit decisions
    
    # Record 8 types of retrofit decisions and the final retrofit decision in an array, num HO * 9
    R_1to9_LR = R_LR_new_12in1_withGrantCopy[:, :9] > 0  # Convert values to boolean
    
    # Extract specific columns
    R_roof_sh_LR = R_1to9_LR[:, 3]  # Roof shingle retrofit decisions
    R_roof_ad_LR = R_1to9_LR[:, 4]  # Roof adhesive retrofit decisions
    R_rtw_st_LR = R_1to9_LR[:, 7]   # Roof-to-wall strap retrofit decisions
    R_openings_sh_LR = R_1to9_LR[:, 5]  # Openings shingle retrofit decisions
    R_openings_ir_LR = R_1to9_LR[:, 6]  # Openings impact retrofit decisions
    R_ap_fld_LR = R_1to9_LR[:, 0]       # Above-ground pool flood retrofit decisions
    R_in_fld_LR = R_1to9_LR[:, 1]       # In-ground pool flood retrofit decisions
    R_el_fld_LR = R_1to9_LR[:, 2]       # Elevated flood retrofit decisions
    R_binary_LR_new = R_1to9_LR[:, 8]   # Binary new retrofit decisions
    
    # Record 8 types of retrofit decisions and the final retrofit decision in an array, num HO * 9
    R_1to9_SFHA = R_SFHA_new_12in1_withGrantCopy[:, :9] > 0  # Convert values to boolean
    
    # Extract specific columns
    R_roof_sh_SFHA = R_1to9_SFHA[:, 3]  # Roof shingle retrofit decisions
    R_roof_ad_SFHA = R_1to9_SFHA[:, 4]  # Roof adhesive retrofit decisions
    R_rtw_st_SFHA = R_1to9_SFHA[:, 7]   # Roof-to-wall strap retrofit decisions
    R_openings_sh_SFHA = R_1to9_SFHA[:, 5]  # Openings shingle retrofit decisions
    R_openings_ir_SFHA = R_1to9_SFHA[:, 6]  # Openings impact retrofit decisions
    R_ap_fld_SFHA = R_1to9_SFHA[:, 0]       # Above-ground pool flood retrofit decisions
    R_in_fld_SFHA = R_1to9_SFHA[:, 1]       # In-ground pool flood retrofit decisions
    R_el_fld_SFHA = R_1to9_SFHA[:, 2]       # Elevated flood retrofit decisions
    R_binary_SFHA_new = R_1to9_SFHA[:, 8]   # Binary new retrofit decisions
        
    ########
    ## Update resistance levels and resistance details in building inventory based on final retrofit decisions
    
    # Update inv_LR_dynamic
    inv_LR_dynamic[:, 3] = inv_LR_dynamic[:, 3] + inv_LR_dynamic[:, 3] * R_roof_sh_LR
    inv_LR_dynamic[:, 4] = inv_LR_dynamic[:, 4] + inv_LR_dynamic[:, 4] * R_roof_ad_LR + inv_LR_dynamic[:, 4] * R_roof_sh_LR
    inv_LR_dynamic[:, 5] = inv_LR_dynamic[:, 5] + inv_LR_dynamic[:, 5] * R_rtw_st_LR
    inv_LR_dynamic[:, 6] = inv_LR_dynamic[:, 6] + inv_LR_dynamic[:, 6] * R_openings_sh_LR * 2 + inv_LR_dynamic[:, 6] * R_openings_ir_LR
    inv_LR_dynamic[:, 8] = inv_LR_dynamic[:, 8] + inv_LR_dynamic[:, 8] * R_ap_fld_LR + inv_LR_dynamic[:, 8] * R_in_fld_LR * 2 + inv_LR_dynamic[:, 8] * R_el_fld_LR * 3
    
    # Update inv_SFHA_dynamic
    inv_SFHA_dynamic[:, 3] = inv_SFHA_dynamic[:, 3] + inv_SFHA_dynamic[:, 3] * R_roof_sh_SFHA
    inv_SFHA_dynamic[:, 4] = inv_SFHA_dynamic[:, 4] + inv_SFHA_dynamic[:, 4] * R_roof_ad_SFHA + inv_SFHA_dynamic[:, 4] * R_roof_sh_SFHA
    inv_SFHA_dynamic[:, 5] = inv_SFHA_dynamic[:, 5] + inv_SFHA_dynamic[:, 5] * R_rtw_st_SFHA
    inv_SFHA_dynamic[:, 6] = inv_SFHA_dynamic[:, 6] + inv_SFHA_dynamic[:, 6] * R_openings_sh_SFHA * 2 + inv_SFHA_dynamic[:, 6] * R_openings_ir_SFHA
    inv_SFHA_dynamic[:, 8] = inv_SFHA_dynamic[:, 8] + inv_SFHA_dynamic[:, 8] * R_ap_fld_SFHA + inv_SFHA_dynamic[:, 8] * R_in_fld_SFHA * 2 + inv_SFHA_dynamic[:, 8] * R_el_fld_SFHA * 3
        
    # Apply the binary mask
    temp_inv_LR_dynamic3 = temp_inv_LR_dynamic3 * R_binary_LR_new
    temp_inv_SFHA_dynamic3 = temp_inv_SFHA_dynamic3 * R_binary_SFHA_new
    
    # Update the inventory indices where temp_inv_LR_dynamic3 and temp_inv_SFHA_dynamic3 are not zero
    inv_LR_dynamic[temp_inv_LR_dynamic3 != 0, 2] = temp_inv_LR_dynamic3[temp_inv_LR_dynamic3 != 0]
    inv_SFHA_dynamic[temp_inv_SFHA_dynamic3 != 0, 2] = temp_inv_SFHA_dynamic3[temp_inv_SFHA_dynamic3 != 0]
    
    ########
    ## Update households’ expected losses and losses per hurricane based on final retrofit decisions
    
    updated_ho_aveLoss_fld_LR_dynamic = ho_aveLoss_fld_LR.copy()
    updated_ho_aveLoss_wd_LR_dynamic = ho_aveLoss_wd_LR.copy()
    updated_ho_perhurrLoss_fld_LR_dynamic = ho_perhurrLoss_fld_LR.copy()
    updated_ho_perhurrLoss_wd_LR_dynamic = ho_perhurrLoss_wd_LR.copy()
    
    homevalue_LR = np.squeeze(stru['stru']['homevalue_LR'][0,0])
    homevalue_SFHA = np.squeeze(stru['stru']['homevalue_SFHA'][0,0])
    
    for i in range(hoNum_LR):
        if R_binary_LR_new[i] > 0:
            updated_ho_aveLoss_fld_LR_dynamic[i] = L_au3D_fld[inv_LR_dynamic[i, 0]-1, inv_LR_dynamic[i, 1]-1, inv_LR_dynamic[i, 2]-1] * homevalue_LR[i]
            updated_ho_perhurrLoss_fld_LR_dynamic[i, :] = np.squeeze(L_au4D_fld[inv_LR_dynamic[i, 0] - 1, inv_LR_dynamic[i, 1] - 1, inv_LR_dynamic[i, 2] - 1, :] * homevalue_LR[i])
            updated_ho_aveLoss_wd_LR_dynamic[i] = L_au3D_wd[inv_LR_dynamic[i, 0]-1, inv_LR_dynamic[i, 1]-1, inv_LR_dynamic[i, 2]-1] * homevalue_LR[i]
            updated_ho_perhurrLoss_wd_LR_dynamic[i, :] = np.squeeze(L_au4D_wd[inv_LR_dynamic[i, 0]-1, inv_LR_dynamic[i, 1]-1, inv_LR_dynamic[i, 2]-1, :] * homevalue_LR[i])
    
    updated_ho_aveLoss_fld_SFHA_dynamic = ho_aveLoss_fld_SFHA.copy()
    updated_ho_aveLoss_wd_SFHA_dynamic = ho_aveLoss_wd_SFHA.copy()
    updated_ho_perhurrLoss_fld_SFHA_dynamic = ho_perhurrLoss_fld_SFHA.copy()
    updated_ho_perhurrLoss_wd_SFHA_dynamic = ho_perhurrLoss_wd_SFHA.copy()
    
    for i in range(hoNum_SFHA):
        if R_binary_SFHA_new[i] > 0:
            updated_ho_aveLoss_fld_SFHA_dynamic[i] = L_au3D_fld[inv_SFHA_dynamic[i, 0]-1 + zoneNum_LR, inv_SFHA_dynamic[i, 1]-1, inv_SFHA_dynamic[i, 2]-1] * homevalue_SFHA[i]
            updated_ho_perhurrLoss_fld_SFHA_dynamic[i, :] = np.squeeze(L_au4D_fld[inv_SFHA_dynamic[i, 0]-1 + zoneNum_LR, inv_SFHA_dynamic[i, 1]-1, inv_SFHA_dynamic[i, 2]-1, :] * homevalue_SFHA[i])
            updated_ho_aveLoss_wd_SFHA_dynamic[i] = L_au3D_wd[inv_SFHA_dynamic[i, 0]-1 + zoneNum_LR, inv_SFHA_dynamic[i, 1]-1, inv_SFHA_dynamic[i, 2]-1] * homevalue_SFHA[i]
            updated_ho_perhurrLoss_wd_SFHA_dynamic[i, :] = np.squeeze(L_au4D_wd[inv_SFHA_dynamic[i, 0]-1 + zoneNum_LR, inv_SFHA_dynamic[i, 1]-1, inv_SFHA_dynamic[i, 2]-1, :] * homevalue_SFHA[i])
    
    ########
    ## Return updated losses and resistance information
    
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
    
    return (ho_aveLoss_fld_LR, ho_aveLoss_fld_SFHA, ho_perhurrLoss_fld_LR, ho_perhurrLoss_fld_SFHA,
            ho_aveLoss_wd_LR, ho_aveLoss_wd_SFHA, ho_perhurrLoss_wd_LR, ho_perhurrLoss_wd_SFHA, stru)



