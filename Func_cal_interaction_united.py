#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 21 22:18:18 2024

@author: Jingya Wang
"""
## TODO: check the parallel

import numpy as np
#from Func_cal_hoInsure_united import Func_cal_hoInsure_united
#from Func_cal_insOpt import Func_cal_insOpt
from Func_cal_insProfit import Func_cal_insProfit



def Func_cal_interaction_united(year, lambda_LR, lambda_SFHA, stru, \
                                ho_aveLoss_fld_LR, ho_aveLoss_fld_SFHA, ho_perhurrLoss_fld_LR, ho_perhurrLoss_fld_SFHA,\
                                    ho_aveLoss_wd_LR, ho_aveLoss_wd_SFHA, ho_perhurrLoss_wd_LR, ho_perhurrLoss_wd_SFHA,\
                                        ho_cumEXP_fld_LR_SCENYEAR, ho_cumEXP_fld_SFHA_SCENYEAR, ho_lastEXP_fld_LR_SCENYEAR, ho_lastEXP_fld_SFHA_SCENYEAR,\
                                             hurr, real_index_L, n_scenarios, years, insurance_price_LB, deductible, hoNum_LR, hoNum_SFHA, k_LR, k_SFHA, g, phi, beta, cpus, ho_dcm, FLAG_DCM):
    if FLAG_DCM == 1:
        from Func_cal_hoInsure_united import Func_cal_hoInsure_united
    else:
        from Func_cal_hoInsure_united_utility import Func_cal_hoInsure_united
    
    
    
    """ Calculate households’ insurance decisions, premiums collected by insurers, market demand, insurers’ profits, 
    and optimal reinsurance strategies based on preset insurance prices """


    ########
    ## Initialize parameters and variables. Variables include: (i) premium collected “P_sum_avehurr_fldwd_LR”, “P_sum_avehurr_fldwd_SFHA”, and “P_sum_avehurr_duo_fldwd_LH”; 
    ##(ii) expected insured losses and insured losses per hurricane after households’ insurance decisions “L_sum_buy_avehurr_fldwd_LR”, “L_sum_buy_avehurr_fldwd_SFHA”, “L_sum_buy_perhurr_fldwd_LR”, and “L_sum_buy_perhurr_fldwd_SFHA”; 
    ##(iii) deductibles per hurricane after insurance decisions “B_sum_buy_perhurr_fldwd_LR” and “B_sum_buy_perhurr_fldwd_SFHA”; (iv) market demand “Q_sum_avehurr_fldwd_LR” and “Q_sum_avehurr_fldwd_SFHA”
    insTail_fld_LR = stru['stru']['insTail_fld_LR'][0,0]
    insTail_fld_SFHA = stru['stru']['insTail_fld_SFHA'][0,0]
    insTail_wd_LR = stru['stru']['insTail_wd_LR'][0,0]
    insTail_wd_SFHA = stru['stru']['insTail_wd_SFHA'][0,0]

    len_lambda_LR = len(lambda_LR)
    len_lambda_SFHA = len(lambda_SFHA)
    Pr = stru['stru']['Pr'][0,0]
    hurr_sim = stru['stru']['hurr_sim'][0,0]
    
    ho_prem_fld_LR = stru['stru']['ho_prem_fld_LR'][0,0]
    ho_prem_wd_LR = stru['stru']['ho_prem_wd_LR'][0,0]
    homevalue_LR = stru['stru']['homevalue_LR'][0,0]
    ho_prem_fld_SFHA = stru['stru']['ho_prem_fld_SFHA'][0,0]
    ho_prem_wd_SFHA = stru['stru']['ho_prem_wd_SFHA'][0,0]
    homevalue_SFHA = stru['stru']['homevalue_SFHA'][0,0]
    
    P_sum_avehurr_fldwd_LR = np.zeros(len_lambda_LR)
    L_sum_buy_avehurr_fldwd_LR = np.zeros(len_lambda_LR)
    L_sum_buy_perhurr_fldwd_LR = np.zeros((len_lambda_LR, hurr))
    B_sum_buy_perhurr_fldwd_LR = np.zeros((len_lambda_LR, hurr))
    P_sum_avehurr_fldwd_SFHA = np.zeros(len_lambda_SFHA)
    L_sum_buy_avehurr_fldwd_SFHA = np.zeros(len_lambda_SFHA)
    L_sum_buy_perhurr_fldwd_SFHA = np.zeros((len_lambda_SFHA, hurr))
    B_sum_buy_perhurr_fldwd_SFHA = np.zeros((len_lambda_SFHA, hurr))
    P_sum_avehurr_duo_fldwd_LH = np.zeros((len_lambda_LR, len_lambda_SFHA))
    Q_sum_avehurr_fldwd_LR = np.zeros(len_lambda_LR)
    Q_sum_avehurr_fldwd_SFHA = np.zeros(len_lambda_SFHA)
    
    risk_theta_LR = None # risk aversion, not used in DCM, will imported if utility function is used
    risk_theta_SFHA = None
    if FLAG_DCM == 2: # utility function is used
        risk_theta_LR = stru['stru']['risk_theta_LR'][0,0]
        risk_theta_SFHA = stru['stru']['risk_theta_SFHA'][0,0]


    import time
    start_time = time.time()
    ########
    ## •Calculate households’ insurance decisions, premiums collected by insurers, expected insured losses and insured losses per hurricane, and market demand based on preset insurance prices by calling function “Func_cal_hoInsure_united.m”
    ##•	Pair premium collected for low- and high-risk zones to one variable “P_sum_avehurr_duo_fldwd_LH
    for i in range(len_lambda_LR):
        (P_sum_avehurr_fldwd_LR[i], L_sum_buy_avehurr_fldwd_LR[i], _, L_sum_buy_perhurr_fldwd_LR[i, :], B_sum_buy_perhurr_fldwd_LR[i, :], _, Q_sum_avehurr_fldwd_LR[i], _, _, _, _,_,_,_,uninsured_fld_mark, uninsured_wd_mark)\
            = Func_cal_hoInsure_united(\
            lambda_LR[i], Pr, \
            ho_perhurrLoss_fld_LR, ho_perhurrLoss_wd_LR,\
            ho_aveLoss_fld_LR, ho_aveLoss_wd_LR,\
            ho_prem_fld_LR, ho_prem_wd_LR,\
            homevalue_LR, ho_cumEXP_fld_LR_SCENYEAR, \
            ho_lastEXP_fld_LR_SCENYEAR, insTail_fld_LR, insTail_wd_LR, \
            real_index_L, risk_theta_LR, insurance_price_LB, deductible, hoNum_LR, hoNum_SFHA, k_LR, k_SFHA, ho_dcm, 0)
    
    for i in range(len_lambda_SFHA):
        (P_sum_avehurr_fldwd_SFHA[i], L_sum_buy_avehurr_fldwd_SFHA[i], _, L_sum_buy_perhurr_fldwd_SFHA[i, :], B_sum_buy_perhurr_fldwd_SFHA[i, :], _, Q_sum_avehurr_fldwd_SFHA[i], _, _, _, _,_,_,_, uninsured_fld_mark, uninsured_wd_mark) \
            = Func_cal_hoInsure_united(
            lambda_SFHA[i], Pr,
            ho_perhurrLoss_fld_SFHA, ho_perhurrLoss_wd_SFHA,
            ho_aveLoss_fld_SFHA, ho_aveLoss_wd_SFHA,
            ho_prem_fld_SFHA, ho_prem_wd_SFHA,
            homevalue_SFHA, ho_cumEXP_fld_SFHA_SCENYEAR,
            ho_lastEXP_fld_SFHA_SCENYEAR, insTail_fld_SFHA, insTail_wd_SFHA,
            real_index_L, risk_theta_SFHA, insurance_price_LB, deductible, hoNum_LR, hoNum_SFHA, k_LR, k_SFHA, ho_dcm, 0)
    
    for i in range(len_lambda_LR):
        for j in range(len_lambda_SFHA):
            P_sum_avehurr_duo_fldwd_LH[i, j] = P_sum_avehurr_fldwd_LR[i] + P_sum_avehurr_fldwd_SFHA[j]

            
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f'{elapsed_time:.1f} s')

        
    
    
    ########
    ## Initialize variables. Variables include: (i) insurers’ profits based on preset low- and high-risk prices “FX_duo”; 
    ##(ii) reinsurance attachment point based on preset low- and high-risk prices “AX_duo”; (iii) reinsurance maximum payout based on preset low- and high-risk prices “MX_duo”; 
    ##(iv) insolvent rates based on preset low- and high-risk prices “insolvent_duo”; (v) reinsurance premiums based on preset low- and high-risk prices “rsy_duo
    
    
    
    
    ########
    ## Set cluster parameters for parallel computing “parfor”
    # Set up environment and number of workers
    
    
    ##### non-parallel version
    # import time
    # start_time = time.time()
    # len_lambda_LR = 2
    # len_lambda_SFHA = 3
    # for i in range(len_lambda_LR):
    #     if not parfor_flag:
    #         for j in range(len_lambda_SFHA):
    #             FX_duo[i, j], AX_duo[i, j], MX_duo[i, j], insolvent_duo[i, j], rsy_duo[i, j] \
    #                 = Func_cal_insOpt(year, P_sum_avehurr_duo_fldwd_LH[i, j], \
    #                 L_sum_buy_perhurr_fldwd_LR[i, :] + L_sum_buy_perhurr_fldwd_SFHA[j, :],B_sum_buy_perhurr_fldwd_LR[i, :] + B_sum_buy_perhurr_fldwd_SFHA[j, :], Pr, hurr_sim, scenarios, years, 'not precise')
    #     else:
    #         for j in range(len_lambda_SFHA):
    #             FX_duo[i, j], AX_duo[i, j], MX_duo[i, j], insolvent_duo[i, j], rsy_duo[i, j] \
    #                 = Func_cal_insOpt(year, P_sum_avehurr_duo_fldwd_LH[i, j], \
    #                                   L_sum_buy_perhurr_fldwd_LR[i, :] + L_sum_buy_perhurr_fldwd_SFHA[j, :],\
    #                                       B_sum_buy_perhurr_fldwd_LR[i, :] + B_sum_buy_perhurr_fldwd_SFHA[j, :], Pr, hurr_sim, scenarios, years, 'not precise')
    # end_time = time.time()
    # elapsed_time = end_time - start_time
    # print(f'{elapsed_time:.1f} s')
    
    from Func_cal_insOpt import Func_cal_insOpt
    from Func_cal_insProfit import Func_cal_insProfit
    
    start_time = time.time()
    from joblib import Parallel, delayed


    def parallel_func(i, j):
        return (i, j) + Func_cal_insOpt(year, P_sum_avehurr_duo_fldwd_LH[i, j], L_sum_buy_perhurr_fldwd_LR[i, :] + L_sum_buy_perhurr_fldwd_SFHA[j, :], \
                                        B_sum_buy_perhurr_fldwd_LR[i, :] + B_sum_buy_perhurr_fldwd_SFHA[j, :], Pr, hurr_sim, n_scenarios, years, insurance_price_LB, g, phi, beta,'not precise')
    results = Parallel(n_jobs=cpus)(delayed(parallel_func)(i, j) for i in range(len_lambda_LR) for j in range(len_lambda_SFHA))
    
    FX_duo, AX_duo, MX_duo, insolvent_duo, rsy_duo, insol_rate_duo = (np.zeros((len_lambda_LR, len_lambda_SFHA)) for _ in range(6))
    
    for i, j, fx, ax, mx, insolvent, rsy, insol in results:
        FX_duo[i, j] = fx
        AX_duo[i, j] = ax
        MX_duo[i, j] = mx
        insolvent_duo[i, j] = insolvent
        rsy_duo[i, j] = rsy
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f'{elapsed_time:.1f} s')


    # from joblib import Parallel, delayed
    
    # def parallel_func(i, j, year, P_sum_avehurr_duo_fldwd_LH, L_sum_buy_perhurr_fldwd_LR, L_sum_buy_perhurr_fldwd_SFHA, 
    #                   B_sum_buy_perhurr_fldwd_LR, B_sum_buy_perhurr_fldwd_SFHA, Pr, hurr_sim, scenarios, years):
    #     combined_L = L_sum_buy_perhurr_fldwd_LR[i, :] + L_sum_buy_perhurr_fldwd_SFHA[j, :]
    #     combined_B = B_sum_buy_perhurr_fldwd_LR[i, :] + B_sum_buy_perhurr_fldwd_SFHA[j, :]
    #     fx, ax, mx, insolvent, rsy = Func_cal_insOpt(year, P_sum_avehurr_duo_fldwd_LH[i, j], combined_L, combined_B, Pr, hurr_sim, scenarios, years, 'not precise')
    #     return (i, j, fx, ax, mx, insolvent, rsy)
    
    # # Precompute any constants outside the loop if possible
    # precomputed_data = (year, P_sum_avehurr_duo_fldwd_LH, L_sum_buy_perhurr_fldwd_LR, L_sum_buy_perhurr_fldwd_SFHA, 
    #                     B_sum_buy_perhurr_fldwd_LR, B_sum_buy_perhurr_fldwd_SFHA, Pr, hurr_sim, scenarios, years)
    
    # results = Parallel(n_jobs=20)(
    #     delayed(parallel_func)(i, j, *precomputed_data) for i in range(len_lambda_LR) for j in range(len_lambda_SFHA)
    # )
    
    # FX_duo, AX_duo, MX_duo, insolvent_duo, rsy_duo = (
    #     np.zeros((len_lambda_LR, len_lambda_SFHA)) for _ in range(5)
    # )
    
    # for i, j, fx, ax, mx, insolvent, rsy in results:
    #     FX_duo[i, j] = fx
    #     AX_duo[i, j] = ax
    #     MX_duo[i, j] = mx
    #     insolvent_duo[i, j] = insolvent
    #     rsy_duo[i, j] = rsy

    
    
    return (P_sum_avehurr_duo_fldwd_LH, L_sum_buy_avehurr_fldwd_LR, L_sum_buy_avehurr_fldwd_SFHA, 
            AX_duo, MX_duo, FX_duo, insolvent_duo, rsy_duo, Q_sum_avehurr_fldwd_LR, Q_sum_avehurr_fldwd_SFHA)
    
    





