#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 17 16:30:49 2024

@author: Jingya Wang
"""

import numpy as np

def Func_imp_acquisition(year, ho_acqRecord_LR, ho_acqRecord_SFHA, A_LR_2in1, A_SFHA_2in1,
                         ho_aveLoss_fld_LR, ho_aveLoss_fld_SFHA, ho_perhurrLoss_fld_LR, ho_perhurrLoss_fld_SFHA,
                         ho_aveLoss_wd_LR, ho_aveLoss_wd_SFHA, ho_perhurrLoss_wd_LR, ho_perhurrLoss_wd_SFHA):
    """Update building inventory and losses information """

    ho_acqRecord_LRCopy = ho_acqRecord_LR.copy()
    ho_acqRecord_SFHACopy = ho_acqRecord_SFHA.copy()
    ho_aveLoss_fld_LRCopy = ho_aveLoss_fld_LR.copy()
    ho_aveLoss_fld_SFHACopy = ho_aveLoss_fld_SFHA.copy()
    ho_perhurrLoss_fld_LRCopy = ho_perhurrLoss_fld_LR.copy()
    ho_perhurrLoss_fld_SFHACopy = ho_perhurrLoss_fld_SFHA.copy()
    ho_aveLoss_wd_LRCopy = ho_aveLoss_wd_LR.copy()
    ho_aveLoss_wd_SFHACopy = ho_aveLoss_wd_SFHA.copy()
    ho_perhurrLoss_wd_LRCopy = ho_perhurrLoss_wd_LR.copy()
    ho_perhurrLoss_wd_SFHACopy = ho_perhurrLoss_wd_SFHA.copy()
    
    

    ########
    ## Update losses for acquired buildings to 0 

    ho_acqRecord_LRCopy = ho_acqRecord_LRCopy + year * (A_LR_2in1[:, 0] != 0)  # Update acquisition record with year

    print(sum(ho_acqRecord_SFHACopy))
    ho_acqRecord_SFHACopy = ho_acqRecord_SFHACopy + year * (A_SFHA_2in1[:, 0] != 0)
    print(sum(A_SFHA_2in1[:, 0] != 0))
    print(sum(ho_acqRecord_SFHACopy))

    # Set losses to 0 for homes that have been acquired
    ho_aveLoss_fld_LRCopy[ho_acqRecord_LRCopy != 0, :] = 0
    ho_aveLoss_fld_SFHACopy[ho_acqRecord_SFHACopy != 0, :] = 0
    ho_perhurrLoss_fld_LR[ho_acqRecord_LRCopy != 0, :] = 0
    ho_perhurrLoss_fld_SFHA[ho_acqRecord_SFHACopy != 0, :] = 0
    ho_aveLoss_wd_LRCopy[ho_acqRecord_LRCopy != 0, :] = 0
    ho_aveLoss_wd_SFHACopy[ho_acqRecord_SFHACopy != 0, :] = 0
    ho_perhurrLoss_wd_LRCopy[ho_acqRecord_LRCopy != 0, :] = 0
    ho_perhurrLoss_wd_SFHACopy[ho_acqRecord_SFHACopy != 0, :] = 0

    return (ho_acqRecord_LRCopy, ho_acqRecord_SFHACopy, ho_aveLoss_fld_LRCopy, ho_aveLoss_fld_SFHACopy,
            ho_perhurrLoss_fld_LR, ho_perhurrLoss_fld_SFHA, ho_aveLoss_wd_LRCopy, ho_aveLoss_wd_SFHACopy,
            ho_perhurrLoss_wd_LRCopy, ho_perhurrLoss_wd_SFHACopy)
