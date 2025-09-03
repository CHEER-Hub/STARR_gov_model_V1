    
    
def main_s_gini_HPC(scenario,FLAG_acq_option,FLAG_ret_option, FLAG_selfret_option,FLAG_ins_option,PARA_gov_benefitratio_forlow,FLAG_load_gov_opt, FLAG_DCM, save_str):
    
    import scipy.io
    #from main_s_gini_HPC import main_s_gini_HPC
    import numpy as np
    import pickle
    
    
    # # Load the data
    dd = '/work/disasters/jw/gov_v1/Inputs_Data/' # TODO customize this directory
    
    import numpy as np
    import time
    import scipy.io
    import pandas as pd
    
    
    ## import other functions
    from Func_cal_acq_SPforLowIncome import Func_cal_acq_SPforLowIncome
    from Func_cal_acquisition import Func_cal_acquisition
    from Func_cal_equilibrium import Func_cal_equilibrium
    from Func_cal_exp_fld import Func_cal_exp_fld
    from Func_cal_insOpt import Func_cal_insOpt
    from Func_cal_insProfit import Func_cal_insProfit
    from Func_cal_hoInsure_united import Func_cal_hoInsure_united
    from Func_cal_interaction_united import Func_cal_interaction_united
    #from Func_cal_ranking import Func_cal_ranking
    from Func_cal_ret_SPLowIncome import Func_cal_ret_SPforLowIncome
    from Func_cal_retrofit import Func_cal_retrofit
    from Func_imp_acquisition import Func_imp_acquisition
    from Func_imp_EQM_united import Func_imp_EQM_united
    from Func_imp_retrofit import Func_imp_retrofit
    from Func_cal_ret_voluntary import Func_cal_ret_voluntary
    from decode_hurr_losses import decode_hurr_losses
    
    if FLAG_DCM == 2:
        from Func_cal_hoInsure_united_utility import Func_cal_hoInsure_united
    
    
    # set the start time of the whole running
    global_t0 = time.time()
    
    ##TODO customize directory
    #dd = ''
    # user-specified parameters
    params = pd.read_csv(dd + 'parameters_STARR_gov.csv', index_col=0, header=None)
    params = params[1].to_dict()

    # census data
    census = pd.read_csv(dd + 'census.csv', header = None)
    # Rename census columns for clarity
    census.columns = ['county_ID', 'population']
    census_df = census.copy()
    # records of counties funding received
    county_records = pd.read_csv(dd + 'county_records.csv')
    total_census = float(params['population'])
    
    # county ID vs zone ID
    ho_zone_county = pd.read_csv(dd + 'household_zones_counties.csv')
    county_zoneID = pd.read_csv(dd + 'county_zoneID.csv')
    

    #pdd threholds
    pdd_threshold = float(params['pdd_threshold'])
    county_threshold = float(params['county_threshold'])
    
    # past 11-year average, used for Y1
    hmgp_y1 = sum(county_records['Projects Amount'])
    
    # maximum buget
    max_budget = float(params['max_budget'])
    
    # budget is X% of damage
    dmg_perc = float(params['dmg_perc'])

    
    hoNum_LR = int(params['hoNum_LR'])
    hoNum_SFHA = int(params['hoNum_SFHA'])
    household = hoNum_LR + hoNum_SFHA
    zoneNum_LR = int(params['zoneNum_LR'])
    zoneNum_SFHA = int(params['zoneNum_SFHA'])
    zones = zoneNum_LR + zoneNum_SFHA
    acq_param_a = params['acq_param_a']
    acq_param_b = params['acq_param_b']
    hurr = int(params['hurr'])
    years = int(params['year'])
    act_years = int(params['act_years'])
    act_zones = int(params['act_zones'])
    
    n_scenarios = int(params['n_scenarios'])
    #act_n_scenarios = int(params['act_n_scenarios'])
    
    cpus = int(params['cpus'])
    
    
    #priceRet_alpha = 1
    
    #%%
    ########
    ##•	Preset variables and parameters: (i) government factors (“gov_optCenters”), (ii) number of insurers (“clevel”), (iii) budget limit (“budget”), (iv) variables to record acquisition, retrofit, and insurance results (e.g. “ARlist_granted”, “R_LR_new_12in1_withGrant”, and “rec_ho_P_gini_1firm”)
    ##•	Load input files: (i) “Data_allcounties_newhanover_hoID.mat”, (ii) “Data_stru.mat”, (iii) “Data_L_au3D4D.mat”, (iv) “Data_loss_fld_afterratio.mat”, (v) “Data_loss_wd.mat”
    
    import warnings
    warnings.filterwarnings('ignore')


    gov_optCenters = np.repeat([[params['gov_optCenters_1'], params['gov_optCenters_2'], params['gov_optCenters_3']/params['gov_optCenters_4'], params['gov_optCenters_5'], params['gov_optCenters_6']] + [params['gov_optCenters_7']]*int(params['gov_optCenters_8'])], years, axis=0)
    gov_centers = gov_optCenters[:, :5]
    clevel = int(params['clevel'])
    budget = params['budget']
    ho_acqRecord_LR = np.zeros(hoNum_LR)
    ho_acqRecord_SFHA = np.zeros(hoNum_SFHA)
    
    # coefficients for DCM models of retrofit
    fld_dcm_coeff = [params['fld_DCM_1'], params['fld_DCM_2'], params['fld_DCM_3'], params['fld_DCM_4'], params['fld_DCM_5'], params['fld_DCM_6'], params['fld_DCM_7'], params['fld_DCM_8'], params['fld_DCM_9']]
    wd_dcm_coeff = [params['wd_DCM_1'], params['wd_DCM_2'], params['wd_DCM_3'], params['wd_DCM_4'], params['wd_DCM_5'], params['wd_DCM_6'], params['wd_DCM_7'], params['wd_DCM_8'], params['wd_DCM_9'], \
                    params['wd_DCM_10'], params['wd_DCM_11'], params['wd_DCM_12'], params['wd_DCM_13'], params['wd_DCM_14'], params['wd_DCM_15'], params['wd_DCM_16']]
    
    # loss data
    Data_allcounties_newhanover_hoID = scipy.io.loadmat(dd + 'Data_allcounties_newhanover_hoID.mat')
    
    # indices of counties
    real_index_L = Data_allcounties_newhanover_hoID['index_L_allcounties']
    
    # running start time
    global_t0 = time.time()
    t = time.localtime()
    
    np.random.seed(1)
    
    print('@_load ')

    stru = scipy.io.loadmat(dd + 'Data_stru_all.mat')

    # time series of hurricanes  
    hurr_sim = stru['stru']['hurr_sim'][0,0]

    # this is for dummy hurricane scenarios
    #hurr_sim_dummy = pd.read_csv(dd + 'hurr_sim_dummy.csv', header=None)
    #hurr_sim_dummy = hurr_sim_dummy.to_numpy()
    
    # zone IDs for households
    ho_areaID_LR = np.squeeze(stru['stru']['ho_areaID_LR'][0,0])  # area ID
    ho_areaID_SFHA = np.squeeze(stru['stru']['ho_areaID_SFHA'][0,0])
    
    # lookup tables for loss and cost data
    L_au3D4D = scipy.io.loadmat(dd + 'Data_L_au3D4D.mat')
    L_au3D_fld = L_au3D4D['L_au3D_fld']
    L_au3D_wd = L_au3D4D['L_au3D_wd']
    L_au4D_fld = L_au3D4D['L_au4D_fld']
    L_au4D_wd = L_au3D4D['L_au4D_wd']
    
    
    # loss data after being discounted
    loss_fld_afterratio = scipy.io.loadmat(dd + 'Data_loss_fld_afterratio.mat')
    ho_aveLoss_fld_LR = loss_fld_afterratio['ho_aveLoss_fld_LR']
    ho_aveLoss_fld_SFHA = loss_fld_afterratio['ho_aveLoss_fld_SFHA']
    ho_perhurrLoss_fld_LR = loss_fld_afterratio['ho_perhurrLoss_fld_LR']
    ho_perhurrLoss_fld_SFHA = loss_fld_afterratio['ho_perhurrLoss_fld_SFHA']
    
    loss_wd = scipy.io.loadmat(dd + 'Data_loss_wd.mat')
    ho_aveLoss_wd_LR = loss_wd['ho_aveLoss_wd_LR']
    ho_aveLoss_wd_SFHA = loss_wd['ho_aveLoss_wd_SFHA']
    ho_perhurrLoss_wd_LR = loss_wd['ho_perhurrLoss_wd_LR']
    ho_perhurrLoss_wd_SFHA = loss_wd['ho_perhurrLoss_wd_SFHA']
    
    # a function returnning an array with all NaNs
    # not using np.zeros, distinguish real value "0" from calculations with the zeros in the preset empty arrays
    def nans(shape, dtype=float):
        a = np.empty(shape, dtype)
        a[:] = np.nan
        return a
    
    
    ## initialize a dictonary ANS to save output files
    ANS = {
        'PDD_mets': np.zeros(act_years),
        'ARlist_granted': [],
        'track_price': [],
        'finalPrice': [],
        'dynamic_invLoss0': np.zeros((4, years)),
        'ARnum': np.zeros((4, years)),  # Assuming 30 years
        'retNum_LR': np.zeros((9, years)),  # Assuming 9 categories of retrofits
        'retNum_SFHA': np.zeros((9, years)),
        'ARspend': np.zeros((4, years)),
        'ARspend_LMH': [None] * years,
        'ARbenefit': np.zeros((4, years)),
        'price_Acq_Ret_aveChosen': np.zeros((5, years)),
        #'ARlistAll': [None] * years,
        'dynamic_invLoss0p5': np.zeros((4, years)),
        'dynamic_invLoss1': np.zeros((4, years)),
        'SPacq': np.zeros((2, years)),
        'SPacq_num': np.zeros((2, years)),
        'ho_acqRecord_LR': np.zeros_like(ho_acqRecord_LR),
        'ho_acqRecord_SFHA': np.zeros_like(ho_acqRecord_SFHA),
        'dynamic_invLoss1p5': np.zeros((4, years)),
        'SPforLowIncome_sum_R_LR_12in1': np.zeros((12, years)),
        'SPforLowIncome_sum_R_SFHA_12in1': np.zeros((12, years)),
        'SPret': np.zeros((2, years)),
        'SPret_num': np.zeros((2, years)),
        'dynamic_invLoss2': np.zeros((4, years)),
        'sum_R_LR_12in1_voluntary': np.zeros((12, years)),
        'sum_R_SFHA_12in1_voluntary': np.zeros((12, years)),
        'dynamic_invLoss': np.zeros((4, years)),
        'dynamic_invLoss3': np.zeros((4, years)),
        'Eprice_perlvl_LR': np.zeros((clevel, years)),
        'Eprice_perlvl_SFHA': np.zeros((clevel, years)),
        'Elambda_perlvl_LR': np.zeros((clevel, years)),
        'Elambda_perlvl_SFHA': np.zeros((clevel, years)),
        'Edemand_perlvl_LR': np.zeros((clevel, years)),
        'Edemand_perlvl_SFHA': np.zeros((clevel, years)),
        'optimized_F_EQM_eachInsurer': np.zeros((clevel, years)),
        'optimized_A_EQM_eachInsurer': np.zeros((clevel, years)),
        'optimized_M_EQM_eachInsurer': np.zeros((clevel, years)),
        'optimized_insolvent_EQM_eachInsurer': np.zeros((clevel, years)),
        'insol_year': np.zeros((clevel, years)),
        'real_L_allInsurers': np.zeros((clevel, years)),
        'real_B_allInsurers': np.zeros((clevel, years)),
        'real_P_allInsurers': np.zeros((clevel, years)),
        'real_reIns_P_eachInsurer': np.zeros((clevel, years)),
        'rec_IMP_EQM_CGE_num_insured_fld_LR': [],
        'rec_IMP_EQM_CGE_num_insured_wd_LR':[],
        'rec_IMP_EQM_CGE_num_insured_fld_SFHA': [],
        'rec_IMP_EQM_CGE_num_insured_wd_SFHA': []
    }

    paywd = nans([household, act_years])
    payfld = nans([household, act_years])
    acq_spend = nans([household, act_years])
    retrofit_record = nans([household, act_years])
    retrofit_cost = nans([household, act_years])
    retrofit_grants = nans([household, act_years])
    ## initialize results of insurance parts
    # rec_ho_P_gini_1firm = nans([household, act_years])
    # rec_ho_B_gini_1firm = nans([household, act_years])
    # rec_ho_Luni_gini_1firm = nans([household, act_years])
    rec_ho_P_gini_4firms = nans([household, act_years])
    rec_ho_B_gini_4firms = nans([household, act_years])
    rec_ho_Luni_gini_4firms = nans([household, act_years])
    # rec_ho_P_gini_135_forced = nans([household, act_years])
    # rec_ho_B_gini_135_forced = nans([household, act_years])
    # rec_ho_Luni_gini_135_forced = nans([household, act_years])
    # rec_ho_P_gini_135_DCM = nans([household, act_years])
    # rec_ho_B_gini_135_DCM = nans([household, act_years])
    # rec_ho_Luni_gini_135_DCM = nans([household, act_years])
    # rec_ho_P_gini_195_DCM = nans([household, act_years])
    # rec_ho_B_gini_195_DCM = nans([household, act_years])
    # rec_ho_Luni_gini_195_DCM = nans([household, act_years])
    rec_ho_Luni_gini_noPB = nans([household, act_years])
    rec_ho_Luni_gini_noPB_fld = nans([household, act_years])
    rec_ho_Luni_gini_noPB_wd = nans([household, act_years])
    rec_IMP_EQM_CGE_num_insured_fld_LR, rec_IMP_EQM_CGE_num_insured_wd_LR, rec_IMP_EQM_CGE_num_insured_fld_SFHA, rec_IMP_EQM_CGE_num_insured_wd_SFHA = [], [], [], []
    
    #TODO seem not needed
    #R_LR_new_12in1_withGrant = np.zeros((hoNum_LR, 12))
    #R_SFHA_new_12in1_withGrant = np.zeros((hoNum_SFHA, 12))
    #rec_ho_invLoss = nans([household, act_years])
    rec_ho_invLoss_fld = nans([household, act_years])
    rec_ho_invLoss_wd = nans([household, act_years])
    
    rec_ho_invLoss_fld_20 = nans([household, act_years])
    
    
    #%%
    ########
    ## •	Call function “Func_cal_exp_fld.py” to calculate hurricane flood experience
    """ FLOOD EXPERIENCE """
    
    ho_lastEXP_fld_LR_SCEN30YEAR, ho_lastEXP_fld_SFHA_SCEN30YEAR, ho_cumEXP_fld_LR_SCEN30YEAR, ho_cumEXP_fld_SFHA_SCEN30YEAR = Func_cal_exp_fld(scenario, hurr_sim, stru, hoNum_LR, hoNum_SFHA, zones, hurr, years)
    print(f'{time.process_time() - np.array(t)[0]:.1f} s\n--\n')
    
    
    
    #%%
    pdd_mets = np.zeros(act_years)
    for year in range(1,act_years+1): 
        print("Running Year", year)

        dynamic_invLoss0_values = [
        np.sum(ho_aveLoss_fld_LR),
        np.sum(ho_aveLoss_fld_SFHA),
        np.sum(ho_aveLoss_wd_LR),
        np.sum(ho_aveLoss_wd_SFHA)]
        
        ANS['dynamic_invLoss0'][:, year-1] = dynamic_invLoss0_values
        
        t_SCENYEAR = time.time()
        ho_lastEXP_fld_LR_SCENYEAR = ho_lastEXP_fld_LR_SCEN30YEAR[:, year-2]
        ho_lastEXP_fld_SFHA_SCENYEAR = ho_lastEXP_fld_SFHA_SCEN30YEAR[:, year-2]
        ho_cumEXP_fld_LR_SCENYEAR = ho_cumEXP_fld_LR_SCEN30YEAR[:, year-2]
        ho_cumEXP_fld_SFHA_SCENYEAR = ho_cumEXP_fld_SFHA_SCEN30YEAR[:, year-2]
        priceAcq_alpha = gov_centers[year-2, 0]
        priceAcq_beta = gov_centers[year-2, 1]
        aftershock = gov_centers[year-2, 2]
        priceRet_alpha = gov_centers[year-2, 3]
        priceRet_beta = gov_centers[year-2, 4]
        J = params['J']
        
        #%%
        ########
        ## •Calculate losses for this year
        ## •Determine whether PDD is met
        """ LOSS """
        
        # get hurricane IDs for this year this scenario
        if year == 1:
            hurr_codes = hurr_sim[scenario-1, year-1]
        else:
            hurr_codes = hurr_sim[scenario-1, year-2] #TODO: testing year
        
        print('simulating Hurricane', hurr_codes)

        # get corresponding losses for each household
        ho_fld_LR = decode_hurr_losses(hurr_codes, ho_perhurrLoss_fld_LR)
        ho_fld_SFHA = decode_hurr_losses(hurr_codes, ho_perhurrLoss_fld_SFHA)
        ho_wd_LR = decode_hurr_losses(hurr_codes, ho_perhurrLoss_wd_LR)
        ho_wd_SFHA = decode_hurr_losses(hurr_codes, ho_perhurrLoss_wd_SFHA) 
        
        ho_fld = np.concatenate((ho_fld_LR, ho_fld_SFHA))
        ho_wd = np.concatenate((ho_wd_LR, ho_wd_SFHA))
        
        ho_loss = ho_fld + ho_wd
        
        #### check if state-wide PDD is met
        state_census = total_census
        # state-wide damage per capita
        state_dmg = (np.sum(ho_fld_LR) + np.sum(ho_fld_SFHA) + np.sum(ho_wd_LR) + np.sum(ho_wd_SFHA))
        print(state_dmg)
        state_dmg_capita = state_dmg/state_census
        # check threshold
        if state_dmg_capita > pdd_threshold: # larger than the threshold: PDD state-wide met
            print('PDD met')
            county_flag = False
            ANS['PDD_mets'][year-1] = 1
            if year == 1: # use past years average
                budget = hmgp_y1
            else:
                budget =  min(max_budget, dmg_perc * state_dmg)
            loss_merged = None
                
        else: # PDD state-wide not met
            print('PDD not met')
            county_flag = True

            # Create a new DataFrame combining losses and county IDs
            df = ho_zone_county.copy()
            df['loss'] = ho_loss

            # Group by county and sum the losses
            loss_by_county = df.groupby('county_ID')['loss'].sum()         
            loss_df = loss_by_county.reset_index()  # now has columns ['county_ID', 'loss']
            
            # Merge on county_ID
            loss_merged = pd.merge(loss_by_county, census_df, on='county_ID', how='inner')
            # Compute loss per capita
            loss_merged['loss_per_capita'] = loss_merged['loss'] / loss_merged['population']
            # Compare and create boolean indicator
            loss_merged['threshold_met'] = loss_merged['loss_per_capita'] >= county_threshold

            # Start with zeros
            loss_merged['budget'] = 0

            # Get a boolean mask for counties where threshold is met
            met_mask = loss_merged['threshold_met']  
            
            if year == 1: # use past years average
                # Create a Series from county_records for easy matching
                county_records_series = county_records.set_index('county_ID')['Projects Amount']
                # Match by county_ID using .map
                loss_merged.loc[met_mask, 'county_ID'].map(county_records_series)
                
            else:

                # Use a percentage of the loss_by_county
                loss_merged.loc[met_mask, 'budget'] = np.minimum(
                    loss_merged.loc[met_mask, 'county_ID'].map(loss_by_county) * dmg_perc,
                    max_budget)
            loss_merged = loss_merged[['county_ID', 'budget']].copy()


        if loss_merged is not None:
            print(np.sum(loss_merged['budget']))
        else:
            print(budget)

        #%%
        ########
        ## •Initialize variables and parameters for acquisition calculation
        ##•	Calculate acquisition decisions based on the preset acquisition option by calling “Func_cal_acquisition.py”
        """ ACQUISITION """
        if FLAG_acq_option == 0: # no acq
            A_LR_2in1 = np.zeros(2)
            A_SFHA_2in1 = np.zeros(2)
            ho_priceAcq_LR = np.zeros(2)
            ho_priceAcq_SFHA_noLastEXP = np.zeros(2)
            ho_priceAcq_SFHA_withLastEXP = np.zeros(2)
            area_priceAcq_loadingFactor_LR = np.zeros(2)
            area_priceAcq_loadingFactor_SFHA = np.zeros(2)
        else:
            t = time.localtime()
            print('@_cal_acquisiton ')
            if FLAG_acq_option == 1: # normal acq, including low-income groups, no need to consider low income separately
                temp_index_L = np.array([])
            else: # normal acq excluding low-income, need to do acq for low-income separately
                temp_index_L = real_index_L
            A_LR_2in1, A_SFHA_2in1, area_priceAcq_loadingFactor_LR, area_priceAcq_loadingFactor_SFHA, ho_priceAcq_LR, ho_priceAcq_SFHA_noLastEXP, ho_priceAcq_SFHA_withLastEXP \
                = Func_cal_acquisition(year, priceAcq_alpha, priceAcq_beta, aftershock, budget, loss_merged, county_zoneID,
                             ho_acqRecord_SFHA, stru, ho_aveLoss_fld_SFHA, ho_aveLoss_wd_SFHA,
                             ho_lastEXP_fld_SFHA_SCENYEAR, temp_index_L, 
                             hoNum_LR, zones, zoneNum_LR, zoneNum_SFHA, years, acq_param_a, acq_param_b, act_zones, county_flag)
    
            print(f'{time.process_time() - np.array(t)[0]:.1f} s')
    
        #%%
        ########
        ## •Initialize variables and parameters for retrofit calculation
        ##•	Calculate retrofit decisions based on the preset retrofit option by calling “Func_cal_retrofit.m”
        """ RETROFIT """
        
        annual_threshold = params['annual_threshold']
        if FLAG_ret_option == 0: # no ret
            R_LR_new_12in1_withGrant = np.zeros((hoNum_LR, 12))
            R_SFHA_new_12in1_withGrant = np.zeros((hoNum_SFHA, 12))
            area_priceRet_loadingFactor_LR = np.zeros(zoneNum_LR)
            area_priceRet_loadingFactor_SFHA = np.zeros(zoneNum_SFHA)
            ho_priceRet_LR = np.zeros((hoNum_LR, 12))
            ho_priceRet_SFHA = np.zeros((hoNum_SFHA, 12))
        else:
            t = time.time()
            print('@_cal_retrofit ')
            if FLAG_ret_option == 1: # normal ret, including low-income groups
                temp_index_L = []
            else: # normal ret, excluding low-income groups
                temp_index_L = real_index_L
            R_LR_new_12in1_withGrant, R_SFHA_new_12in1_withGrant, temp_inv_LR_dynamic3, temp_inv_SFHA_dynamic3, area_priceRet_loadingFactor_LR, area_priceRet_loadingFactor_SFHA, ho_priceRet_LR, ho_priceRet_SFHA \
                = Func_cal_retrofit(ho_acqRecord_LR, ho_acqRecord_SFHA, priceRet_alpha, priceRet_beta, J, annual_threshold, stru, L_au3D_fld, L_au3D_wd, ho_aveLoss_fld_LR, ho_aveLoss_fld_SFHA, ho_aveLoss_wd_LR, \
                                    ho_aveLoss_wd_SFHA, ho_cumEXP_fld_LR_SCENYEAR, ho_cumEXP_fld_SFHA_SCENYEAR, temp_index_L, hoNum_LR, hoNum_SFHA, zoneNum_LR, zoneNum_SFHA, years, fld_dcm_coeff, wd_dcm_coeff)
            elapsed_time = time.time() - t
            print(f'{elapsed_time:.1f} s')
            
       #%%
        ########
        ##• Calculate government grants allocation based on benefit rates by calling “Func_cal_ranking.m”
        ##•	Update acquisition and retrofit decisions
        """ GRANT ALLOCATION """
            
        if FLAG_acq_option + FLAG_ret_option != 0:
            start_time = time.time()
            print('@_cal_ranking ')
            
            if county_flag == True:
                from Func_cal_ranking_county import Func_cal_ranking as rank_county
                A_LR_2in1, A_SFHA_2in1, R_LR_new_12in1_withGrant, R_SFHA_new_12in1_withGrant, price_Acq_Ret_aveChosen, acquired_areaID_SFHA, RESranking, A_grant, R_grant = \
                    rank_county(year, budget, stru, A_LR_2in1, A_SFHA_2in1, R_LR_new_12in1_withGrant, R_SFHA_new_12in1_withGrant, ho_acqRecord_LR, ho_acqRecord_SFHA, priceAcq_alpha, priceAcq_beta, priceRet_alpha, priceRet_beta, area_priceAcq_loadingFactor_LR, area_priceAcq_loadingFactor_SFHA, area_priceRet_loadingFactor_LR, area_priceRet_loadingFactor_SFHA, \
                                ho_priceAcq_LR, ho_priceAcq_SFHA_noLastEXP, ho_priceAcq_SFHA_withLastEXP, ho_priceRet_LR, ho_priceRet_SFHA, Data_allcounties_newhanover_hoID['index_L_allcounties'], Data_allcounties_newhanover_hoID['index_M_allcounties'], Data_allcounties_newhanover_hoID['index_H_allcounties'], zones, zoneNum_LR, hoNum_LR, hoNum_SFHA, county_zoneID, loss_merged)
            else:
                from Func_cal_ranking import Func_cal_ranking as rank_zone

                A_LR_2in1, A_SFHA_2in1, R_LR_new_12in1_withGrant, R_SFHA_new_12in1_withGrant, price_Acq_Ret_aveChosen, acquired_areaID_SFHA, RESranking, A_grant, R_grant = \
                    rank_zone(year, budget, stru, A_LR_2in1, A_SFHA_2in1, R_LR_new_12in1_withGrant, R_SFHA_new_12in1_withGrant, ho_acqRecord_LR, ho_acqRecord_SFHA, priceAcq_alpha, priceAcq_beta, priceRet_alpha, priceRet_beta, area_priceAcq_loadingFactor_LR, area_priceAcq_loadingFactor_SFHA, area_priceRet_loadingFactor_LR, area_priceRet_loadingFactor_SFHA, \
                                ho_priceAcq_LR, ho_priceAcq_SFHA_noLastEXP, ho_priceAcq_SFHA_withLastEXP, ho_priceRet_LR, ho_priceRet_SFHA, Data_allcounties_newhanover_hoID['index_L_allcounties'], Data_allcounties_newhanover_hoID['index_M_allcounties'], Data_allcounties_newhanover_hoID['index_H_allcounties'], zones, zoneNum_LR, hoNum_LR, hoNum_SFHA)
            
            print(sum(ho_acqRecord_SFHA))
            end_time = time.time()
            elapsed_time = end_time - start_time
    
            # Print elapsed time formatted to one decimal place
            print(f'{elapsed_time:.1f} s')
            
            ANS['ARlist_granted'].append(np.column_stack((RESranking['ind_areaARLH_descend_granted'], np.full(len(RESranking['ind_areaARLH_descend_granted']), year))))
        
            # Calculate the number of grants for acquisitions and retrofits
            ANS['ARnum'][:, year-1] = [
                np.sum(A_LR_2in1[:, 0] > 0),
                np.sum(A_SFHA_2in1[:, 0] > 0),
                np.sum(R_LR_new_12in1_withGrant[:, 8] > 0),
                np.sum(R_SFHA_new_12in1_withGrant[:, 8] > 0)
            ]
            
            # Calculate the number of retrofits per category
            ANS['retNum_LR'][:, year-1] = np.sum(R_LR_new_12in1_withGrant[:, :9], axis=0)
            ANS['retNum_SFHA'][:, year-1] = np.sum(R_SFHA_new_12in1_withGrant[:, :9], axis=0)
            
            # Calculate the spending for acquisitions and retrofits
            ANS['ARspend'][:, year-1] = [
                np.sum(A_LR_2in1[:, 1]),
                np.sum(A_SFHA_2in1[:, 1]),
                np.sum(R_LR_new_12in1_withGrant[:, 11]),
                np.sum(R_SFHA_new_12in1_withGrant[:, 11])
            ]
            
            # Store the grants per category
            ANS['ARspend_LMH'][year-1] = [A_grant, R_grant]
            
            # Calculate the benefits for acquisitions and retrofits
            ANS['ARbenefit'][:, year-1] = [
                np.sum(A_LR_2in1[:, 0]),
                np.sum(A_SFHA_2in1[:, 0]),
                np.sum(R_LR_new_12in1_withGrant[:, 9]),
                np.sum(R_SFHA_new_12in1_withGrant[:, 9])
            ]
            
            # Store the average chosen prices for acquisitions and retrofits
            ANS['price_Acq_Ret_aveChosen'][:, year-1] = price_Acq_Ret_aveChosen
            
            # Store the descending sorted list of areas
            #ANS['ARlistAll'][year-1] = RESranking['ind_areaARLH_descend']
            
            acq_spend[: (hoNum_LR + hoNum_SFHA), year -1] = np.concatenate((A_LR_2in1[:,1], A_SFHA_2in1[:, 1]))
            
            retrofit_record[: (hoNum_LR + hoNum_SFHA), year-1] = np.concatenate((R_LR_new_12in1_withGrant[:,8] * year, R_SFHA_new_12in1_withGrant[:,8] * year))
            retrofit_cost[: (hoNum_LR + hoNum_SFHA), year-1] = np.concatenate((R_LR_new_12in1_withGrant[:,10], R_SFHA_new_12in1_withGrant[:,10]))
            retrofit_grants[: (hoNum_LR + hoNum_SFHA), year-1] = np.concatenate((R_LR_new_12in1_withGrant[:,11], R_SFHA_new_12in1_withGrant[:,11]))            
            #%%
            ########
            ## •	Update building inventory and losses information based on acquisition decisions by calling “Func_imp_acquisition.py”
            """ UPDATE AFTER ACQUISITION """
            if FLAG_acq_option != 0:
                start_time = time.time()
                print('@_imp_acquisition ') # imp: implement
                ho_acqRecord_LR, ho_acqRecord_SFHA, ho_aveLoss_fld_LR, ho_aveLoss_fld_SFHA, ho_perhurrLoss_fld_LR, ho_perhurrLoss_fld_SFHA, ho_aveLoss_wd_LR, ho_aveLoss_wd_SFHA, ho_perhurrLoss_wd_LR, ho_perhurrLoss_wd_SFHA \
                    = Func_imp_acquisition(year, ho_acqRecord_LR, ho_acqRecord_SFHA, A_LR_2in1, A_SFHA_2in1, ho_aveLoss_fld_LR, ho_aveLoss_fld_SFHA, ho_perhurrLoss_fld_LR, ho_perhurrLoss_fld_SFHA, ho_aveLoss_wd_LR, ho_aveLoss_wd_SFHA, ho_perhurrLoss_wd_LR, ho_perhurrLoss_wd_SFHA) # call "Func_imp_acquisition"
                end_time = time.time()
                elapsed_time = end_time - start_time
                print(f'{elapsed_time:.1f} s')

    
           #%%
            ########
            ## •	Update building inventory and losses information based on retrofit decisions by calling “Func_imp_retrofit.py”
            """ UPDTAE AFTER RETROFIT """
            if FLAG_ret_option != 0:
                start_time = time.time()
                print('@_imp_retrofit ')
                ho_aveLoss_fld_LR, ho_aveLoss_fld_SFHA, ho_perhurrLoss_fld_LR, ho_perhurrLoss_fld_SFHA, ho_aveLoss_wd_LR, ho_aveLoss_wd_SFHA, ho_perhurrLoss_wd_LR, ho_perhurrLoss_wd_SFHA, stru = Func_imp_retrofit(ho_acqRecord_LR, ho_acqRecord_SFHA, R_LR_new_12in1_withGrant, R_SFHA_new_12in1_withGrant, temp_inv_LR_dynamic3, temp_inv_SFHA_dynamic3, stru, L_au3D_fld, L_au3D_wd, L_au4D_fld, L_au4D_wd, ho_aveLoss_fld_LR, ho_aveLoss_fld_SFHA, ho_perhurrLoss_fld_LR, ho_perhurrLoss_fld_SFHA, ho_aveLoss_wd_LR, ho_aveLoss_wd_SFHA, ho_perhurrLoss_wd_LR, ho_perhurrLoss_wd_SFHA, zoneNum_LR, hoNum_LR, hoNum_SFHA) # call "Func_imp_retrofit"
                end_time = time.time()
                elapsed_time = end_time - start_time
                print(f'{elapsed_time:.1f} s')
            ########
            ## •	If there is no acquisition and no retrofit allowed, set results for acquisition and retrofit to 0
            else:
                ANS['ARlist_granted'] = []
                ANS['ARnum'][:, year-1] = [0, 0, 0, 0]
                ANS['retNum_LR'][:, year-1] = np.zeros(9)
                ANS['retNum_SFHA'][:, year-1] = np.zeros(9)
                ANS['ARspend'][:, year-1] = [0, 0, 0, 0]
                ANS['ARbenefit'][:, year-1] = [0, 0, 0, 0]
                ANS['price_Acq_Ret_aveChosen'][:, year-1] = np.zeros(5)
                #ANS['ARlistAll'][year-1] = 0
        
        #%%
        ########
        ## •If there is a special acquisition option for low-income group, calculate low-income acquisition decisions by calling “Func_cal_acq_SPforLowIncome.m”
        ##•	Update building inventory and losses information based on low-income acquisition decisions by calling “Func_imp_acquisition.m”
        """ LOW-INCOME for ACQUISITION """
    
        if FLAG_acq_option == 2:
            start_time = time.time()
            print('@SP_acq for the low ')
            
            # Call the acquisition function for low-income homeowners
            A_LR_2in1_SPforLowIncome,A_SFHA_2in1_SPforLowIncome,area_priceAcq_loadingFactor_LR,area_priceAcq_loadingFactor_SFHA,ho_priceAcq_LR,ho_priceAcq_SFHA_noLastEXP,ho_priceAcq_SFHA_withLastEXP \
                = Func_cal_acq_SPforLowIncome(
                year, priceAcq_alpha, priceAcq_beta, aftershock, budget,
                ho_acqRecord_SFHA, stru, ho_aveLoss_fld_SFHA, ho_aveLoss_wd_SFHA,
                ho_lastEXP_fld_SFHA_SCENYEAR, real_index_L, PARA_gov_benefitratio_forlow, hoNum_LR, zoneNum_LR, zoneNum_SFHA, years)
       
            # call imp_acq.py
            ho_acqRecord_LR, ho_acqRecord_SFHA, ho_aveLoss_fld_LR, ho_aveLoss_fld_SFHA, ho_perhurrLoss_fld_LR, ho_perhurrLoss_fld_SFHA, ho_aveLoss_wd_LR, ho_aveLoss_wd_SFHA, ho_perhurrLoss_wd_LR, ho_perhurrLoss_wd_SFHA\
                = Func_imp_acquisition(year, ho_acqRecord_LR, ho_acqRecord_SFHA, A_LR_2in1_SPforLowIncome,A_SFHA_2in1_SPforLowIncome, ho_aveLoss_fld_LR, ho_aveLoss_fld_SFHA, ho_perhurrLoss_fld_LR, ho_perhurrLoss_fld_SFHA, ho_aveLoss_wd_LR, ho_aveLoss_wd_SFHA, ho_perhurrLoss_wd_LR, ho_perhurrLoss_wd_SFHA) # call "Func_imp_acquisition"
    
            ANS['SPacq'][:, year-1] = (sum(A_LR_2in1_SPforLowIncome, 1) + sum(A_SFHA_2in1_SPforLowIncome, 1))
            ANS['SPacq_num'][:, year-1] = [len(np.where(A_LR_2in1_SPforLowIncome[:, 0] > 0)[0]), len(np.where(A_SFHA_2in1_SPforLowIncome[:, 0] > 0)[0])]
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f'{elapsed_time:.1f} s')
        
        else:
            ANS['SPacq'][:, year-1] = [0, 0]
            ANS['SPacq_num'][:, year-1] = [0, 0]
        
        ANS['ho_acqRecord_LR'] = ho_acqRecord_LR
        ANS['ho_acqRecord_SFHA'] = ho_acqRecord_SFHA
        
        #%%
        ########
        ## •If there is a special retrofit option for low-income group, calculate low-income retrofit decisions, and update building inventory 
        ## and losses information based on low-income retrofit decisions by calling “Func_cal_ret_SPforLowIncome.m”
        """ LOW-INCOME for RETROFIT """
        if FLAG_ret_option == 2:
            start_time = time.time()
            print('@SP_ret for the low ')
            
            ho_aveLoss_fld_LR, ho_aveLoss_fld_SFHA, ho_perhurrLoss_fld_LR, ho_perhurrLoss_fld_SFHA, \
            ho_aveLoss_wd_LR, ho_aveLoss_wd_SFHA, ho_perhurrLoss_wd_LR, ho_perhurrLoss_wd_SFHA, \
            stru, SPforLowIncome_sum_R_LR_12in1, SPforLowIncome_sum_R_SFHA_12in1, \
            SPforLowIncome_R_LR_new_12in1_withGrant, SPforLowIncome_R_SFHA_new_12in1_withGrant  \
                = Func_cal_ret_SPforLowIncome(
            ho_acqRecord_LR, ho_acqRecord_SFHA, priceRet_alpha, priceRet_beta, J, annual_threshold,
            stru, L_au3D_fld, L_au3D_wd, L_au4D_fld, L_au4D_wd, ho_aveLoss_fld_LR, ho_aveLoss_fld_SFHA,
            ho_aveLoss_wd_LR, ho_aveLoss_wd_SFHA, ho_perhurrLoss_fld_LR, ho_perhurrLoss_fld_SFHA,
            ho_perhurrLoss_wd_LR, ho_perhurrLoss_wd_SFHA, ho_cumEXP_fld_LR_SCENYEAR, ho_cumEXP_fld_SFHA_SCENYEAR,
            R_LR_new_12in1_withGrant, R_SFHA_new_12in1_withGrant, real_index_L, PARA_gov_benefitratio_forlow, hoNum_LR, hoNum_SFHA, zoneNum_LR, zoneNum_SFHA) # call "Func_cal_ret_SPforLowIncome"
            
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f'{elapsed_time:.1f} s')
            
            ANS['SPforLowIncome_sum_R_LR_12in1'][:, year-1] = SPforLowIncome_sum_R_LR_12in1.T # 12*1 sum along HO
            ANS['SPforLowIncome_sum_R_SFHA_12in1'][:, year-1] = SPforLowIncome_sum_R_SFHA_12in1.T
            ANS['SPret'][:, year-1] = SPforLowIncome_sum_R_LR_12in1[[9, 11]].T + SPforLowIncome_sum_R_SFHA_12in1[[9, 11]].T # loss benefit, cost, grant
            ANS['SPret_num'][:, year-1] = [SPforLowIncome_sum_R_LR_12in1[9], SPforLowIncome_sum_R_SFHA_12in1[9]] # total num of HO do ret low income
            index_L_LR = real_index_L[real_index_L <= hoNum_LR] - 1
            index_L_SFHA = real_index_L[real_index_L > hoNum_LR] - hoNum_LR - 1
            R_LR_new_12in1_withGrant[index_L_LR, :] = SPforLowIncome_R_LR_new_12in1_withGrant[index_L_LR, :] # retrofit information for all HO
            R_SFHA_new_12in1_withGrant[index_L_SFHA, :] = SPforLowIncome_R_SFHA_new_12in1_withGrant[index_L_SFHA, :]
        
        else:
            ANS['SPforLowIncome_sum_R_LR_12in1'][:, year-1] = np.zeros(12)
            ANS['SPforLowIncome_sum_R_SFHA_12in1'][:, year-1] = np.zeros(12)
            ANS['SPret'][:, year-1] = np.zeros(2)
            ANS['SPret_num'][:, year-1] = np.zeros(2)
    
    
        #%%
        ########
        ## If self-retrofit is allowed, calculate self-retrofit decisions, and update building inventory and losses information 
        ## based on self-retrofit decisions by calling “Func_cal_ret_voluntary.m”
        """ SELF-RETROFIT """
        if FLAG_selfret_option:
            start_time = time.time()
            print('@_cal_ret_voluntary ')
            ho_aveLoss_fld_LR, ho_aveLoss_fld_SFHA, ho_perhurrLoss_fld_LR, ho_perhurrLoss_fld_SFHA, ho_aveLoss_wd_LR, ho_aveLoss_wd_SFHA, ho_perhurrLoss_wd_LR, ho_perhurrLoss_wd_SFHA, stru, R_LR_new_12in1_withGrant, R_SFHA_new_12in1_withGrant, sum_R_LR_12in1_voluntary, sum_R_SFHA_12in1_voluntary\
                = Func_cal_ret_voluntary(ho_acqRecord_LR, ho_acqRecord_SFHA, priceRet_alpha, priceRet_beta, J, annual_threshold, stru, L_au3D_fld, L_au3D_wd, L_au4D_fld, L_au4D_wd, ho_aveLoss_fld_LR, ho_aveLoss_fld_SFHA, ho_aveLoss_wd_LR, ho_aveLoss_wd_SFHA, ho_perhurrLoss_fld_LR, ho_perhurrLoss_fld_SFHA, \
                                         ho_perhurrLoss_wd_LR, ho_perhurrLoss_wd_SFHA, ho_cumEXP_fld_LR_SCENYEAR, ho_cumEXP_fld_SFHA_SCENYEAR, R_LR_new_12in1_withGrant, R_SFHA_new_12in1_withGrant, hoNum_LR, hoNum_SFHA, zoneNum_LR, zoneNum_SFHA, years, fld_dcm_coeff, wd_dcm_coeff) # call "Func_cal_ret_voluntary"
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f'{elapsed_time:.1f} s')
            ANS['sum_R_LR_12in1_voluntary'][:, year-1] = sum_R_LR_12in1_voluntary.T # 12*1 sum along HO
            ANS['sum_R_SFHA_12in1_voluntary'][:, year-1] = sum_R_SFHA_12in1_voluntary.T
        
        else:
            ANS['sum_R_LR_12in1_voluntary'][:, year-1] = np.zeros(12)
            ANS['sum_R_SFHA_12in1_voluntary'][:, year-1] = np.zeros(12)
        
        ANS['dynamic_invLoss'][:, year-1] = [
            float(np.sum(ho_aveLoss_fld_LR)),
            float(np.sum(ho_aveLoss_fld_SFHA)),
            float(np.sum(ho_aveLoss_wd_LR)),
            float(np.sum(ho_aveLoss_wd_SFHA))
        ]# update loss information
        
        # Combine average flood and wind losses for LR and SFHA
        combined_aveLoss_LR = ho_aveLoss_fld_LR + ho_aveLoss_wd_LR
        combined_aveLoss_SFHA = ho_aveLoss_fld_SFHA + ho_aveLoss_wd_SFHA
        
        # Concatenate the combined losses and assign to the corresponding year column in rec_ho_invLoss
        rec_ho_invLoss_fld[: (hoNum_LR + hoNum_SFHA), year-1] = np.squeeze(np.concatenate((ho_aveLoss_fld_LR, ho_aveLoss_fld_SFHA)))
        rec_ho_invLoss_wd[: (hoNum_LR + hoNum_SFHA), year-1] = np.squeeze(np.concatenate((ho_aveLoss_wd_LR, ho_aveLoss_wd_SFHA)))
    
        
        if year == act_years:
            rec_ho_invLoss_fld_20 = np.concatenate((ho_perhurrLoss_fld_LR, ho_perhurrLoss_fld_SFHA))
        
    
    
        #%%
        ########
        ## •Set a series of insurance prices (“insPrice_LR” and “insPrice_SFHA”) for low- and high-risk zones
        ##•	Calculate households’ insurance decisions as well as insurance premiums collected by insurers, market demand, optimal strategy of purchasing reinsurance (determine attachment point A and the maximum payout M), and profits of insurers by calling “Func_cal_interaction_united.m”
        """ INSURANCE """
        if FLAG_ins_option:
            insPrice_LR = np.arange(4, 15, 0.4) # search grids
            insPrice_SFHA = np.arange(1.3, 5.4, 0.1) # search grids
        
            insurance_price_LB = params['insurance_price_LB']
            insurance_price_UB = params['insurance_price_UB']
            lambda_LR = insPrice_LR - insurance_price_LB
            lambda_SFHA = insPrice_SFHA - insurance_price_LB
            
            deductible = params['deductible']
            k_LR = params['k_LR']
            k_SFHA = params['k_SFHA']
            g= params['g']
            phi = params['phi']
            beta = params['beta']
            
            ho_dcm = [params['ho_DCM_fld_1'], params['ho_DCM_fld_2'], params['ho_DCM_limit']]
            
            start_time = time.time()
            print('@_cal_interaction ')
            P_sum_avehurr_duo_fldwd_LH, L_sum_avehurr_fldwd_LR, L_sum_avehurr_fldwd_SFHA, AX_duo, MX_duo, FX_duo, insolvent_duo, rsy_duo, Q_sum_avehurr_fldwd_LR, Q_sum_avehurr_fldwd_SFHA \
                = Func_cal_interaction_united(year, lambda_LR, lambda_SFHA, stru, ho_aveLoss_fld_LR, ho_aveLoss_fld_SFHA, ho_perhurrLoss_fld_LR, ho_perhurrLoss_fld_SFHA, ho_aveLoss_wd_LR, ho_aveLoss_wd_SFHA, ho_perhurrLoss_wd_LR, ho_perhurrLoss_wd_SFHA, \
                                              ho_cumEXP_fld_LR_SCENYEAR, ho_cumEXP_fld_SFHA_SCENYEAR, ho_lastEXP_fld_LR_SCENYEAR, ho_lastEXP_fld_SFHA_SCENYEAR, hurr, real_index_L, n_scenarios, years, insurance_price_LB, deductible, hoNum_LR, hoNum_SFHA, k_LR, k_SFHA, g, phi, beta, cpus, ho_dcm, FLAG_DCM) # call "Func_cal_interaction_united"
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f'{elapsed_time:.1f} s')
    
    #%%    
            ########
            ## •Calculate equilibrium prices (Cournot-Nash game) for low- and high-risk zones by calling “Func_cal_equilibrium.m”
            start_time = time.time()
            print('@_cal_equilibrium ')
            Eprice_perlvl_LR, Eprice_perlvl_SFHA, Elambda_perlvl_LR, Elambda_perlvl_SFHA, Edemand_perlvl_LR, Edemand_perlvl_SFHA, track_price = Func_cal_equilibrium(P_sum_avehurr_duo_fldwd_LH, Q_sum_avehurr_fldwd_LR, Q_sum_avehurr_fldwd_SFHA, FX_duo, lambda_LR, lambda_SFHA, clevel, insurance_price_LB, insurance_price_UB) # EQM price
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f'{elapsed_time:.1f} s')
            ANS['Eprice_perlvl_LR'][:, year-1] = Eprice_perlvl_LR
            ANS['Eprice_perlvl_SFHA'][:, year-1] = Eprice_perlvl_SFHA
            ANS['Elambda_perlvl_LR'][:, year-1] = Elambda_perlvl_LR
            ANS['Elambda_perlvl_SFHA'][:, year-1] = Elambda_perlvl_SFHA
            ANS['Edemand_perlvl_LR'][:, year-1] = Edemand_perlvl_LR
            ANS['Edemand_perlvl_SFHA'][:, year-1] = Edemand_perlvl_SFHA
            
            print(Eprice_perlvl_LR)
            print(Eprice_perlvl_SFHA)
            
            # Concatenate the result with ANS.track_price vertically
            if not isinstance(ANS['track_price'], np.ndarray):
                ANS['track_price'] = np.array(ANS['track_price'])
            
            # Ensure track_price is a numpy array
            if not isinstance(track_price, np.ndarray):
                track_price = np.array(track_price)
            
            # Create an array of the same number of rows as track_price with the year value
            year_column = np.full((track_price.shape[0], 1), year)
            
            # Concatenate track_price and the year_column horizontally
            track_price_with_year = np.hstack((track_price, year_column))
            
            # Check if ANS.track_price is empty
            if ANS['track_price'].size == 0:
                ANS['track_price'] = track_price_with_year
            else:
                # Concatenate the result with ANS.track_price vertically
                ANS['track_price'] = np.vstack((ANS['track_price'], track_price_with_year))
    
    
    #%%        
            ########
            ## Calculate households’ insurance decisions based on equilibrium prices, and report premiums, deductibles, and uninsured losses by calling “Func_imp_EQM_united.m”. 
            ## Also report optimal profits for insurers, optimal strategies for purchasing insurance, and insolvent rates. There are 7 different types of results: 
                ## (i) no insurance; (ii) market with only 1 insurer; (ii) market with 4 insurers; 
                ##(iv) mandatory insurance for low-income group with price $1.35 and DCM for middle- and high-income groups with equilibrium prices; 
                ##(v) DCM insurance for low-income group with price $1.35 and DCM for middle- and high-income groups with equilibrium prices; 
                ##(vi) mandatory insurance for low-income group with price $1.95 and DCM for middle- and high-income groups with equilibrium prices; 
                ##(vii) DCM insurance for low-income group with price $1.95 and DCM for middle- and high-income groups with equilibrium prices
            start_time = time.time()
            print('@_imp_EQM ')
            optimized_F_EQM_eachInsurer, optimized_A_EQM_eachInsurer, optimized_M_EQM_eachInsurer, optimized_insolvent_EQM_eachInsurer, insol_year, real_P_allInsurers, real_L_allInsurers, real_B_allInsurers, real_reIns_P_eachInsurer,\
                Luni_gini_noPB, Luni_gini_noPB_fld, Luni_gini_noPB_wd, P_gini_1firm, B_gini_1firm, Luni_gini_1firm, P_gini_4firms, B_gini_4firms, Luni_gini_4firms, P_gini_135_forced, B_gini_135_forced, Luni_gini_135_forced, P_gini_135_DCM, B_gini_135_DCM, Luni_gini_135_DCM, \
                    P_gini_195_DCM, B_gini_195_DCM, Luni_gini_195_DCM, num_insured_fld_LR, num_insured_wd_LR, num_insured_fld_SFHA, num_insured_wd_SFHA, num_check_poor_LR_135_DCM, num_check_poor_SFHA_135_DCM, pay_wd_lr, pay_fld_lr, pay_wd_hr, pay_fld_hr\
                = Func_imp_EQM_united(scenario, year, Elambda_perlvl_LR, Elambda_perlvl_SFHA, stru, ho_aveLoss_fld_LR, ho_aveLoss_fld_SFHA, ho_perhurrLoss_fld_LR, ho_perhurrLoss_fld_SFHA, ho_aveLoss_wd_LR, ho_aveLoss_wd_SFHA, ho_perhurrLoss_wd_LR, ho_perhurrLoss_wd_SFHA, \
                                      ho_cumEXP_fld_LR_SCENYEAR, ho_cumEXP_fld_SFHA_SCENYEAR, ho_lastEXP_fld_LR_SCENYEAR, ho_lastEXP_fld_SFHA_SCENYEAR, real_index_L, n_scenarios, years, insurance_price_LB, deductible, hoNum_LR, hoNum_SFHA, k_LR, k_SFHA, g, phi, beta, ho_dcm, FLAG_DCM) # call "Func_imp_EQM_united"
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f'{elapsed_time:.1f} s')
            ANS['optimized_F_EQM_eachInsurer'][:, year-1] = optimized_F_EQM_eachInsurer # optimal profits per insurer for insurance markets by years
            ANS['optimized_A_EQM_eachInsurer'][:, year-1] = optimized_A_EQM_eachInsurer # optimal reinsurance attachment point A
            ANS['optimized_M_EQM_eachInsurer'][:, year-1] = optimized_M_EQM_eachInsurer # optimal reinsurance maximum payout M
            print(optimized_insolvent_EQM_eachInsurer)
            ANS['optimized_insolvent_EQM_eachInsurer'][:, year-1] = optimized_insolvent_EQM_eachInsurer # insolvent rates
            ANS['insol_year'][:, year-1] = insol_year # insolvency rate by year
            print(insol_year)
            ANS['real_L_allInsurers'][:, year-1] = real_L_allInsurers # total insured losses
            ANS['real_B_allInsurers'][:, year-1] = real_B_allInsurers # total deductibles under equilibrium prices
            ANS['real_P_allInsurers'][:, year-1] = real_P_allInsurers # total premiums collected under equilibrium prices
            ANS['real_reIns_P_eachInsurer'][:, year-1] = real_reIns_P_eachInsurer # reinsurance premiumss
            print('A/1e8 M/1e8 F/1e8 insolvent*100:')

            paywd[: (hoNum_LR + hoNum_SFHA), year-1] = np.concatenate((pay_wd_lr, pay_wd_hr))
            payfld[: (hoNum_LR + hoNum_SFHA), year-1] = np.concatenate((pay_fld_lr, pay_fld_hr))
            # rec_ho_P_gini_1firm[: (hoNum_LR + hoNum_SFHA), year-1] = P_gini_1firm
            # rec_ho_B_gini_1firm[: (hoNum_LR + hoNum_SFHA), year-1] = B_gini_1firm
            # rec_ho_Luni_gini_1firm[: (hoNum_LR + hoNum_SFHA), year-1] = Luni_gini_1firm
        
            rec_ho_P_gini_4firms[: (hoNum_LR + hoNum_SFHA), year-1] = P_gini_4firms # premiums by households and years
            rec_ho_B_gini_4firms[: (hoNum_LR + hoNum_SFHA), year-1] = B_gini_4firms # deductibles by household and years
            rec_ho_Luni_gini_4firms[: (hoNum_LR + hoNum_SFHA), year-1] = Luni_gini_4firms # uninsured losses by households and years
        
            # rec_ho_P_gini_135_forced[: (hoNum_LR + hoNum_SFHA), year-1] = P_gini_135_forced
            # rec_ho_B_gini_135_forced[: (hoNum_LR + hoNum_SFHA), year-1] = B_gini_135_forced
            # rec_ho_Luni_gini_135_forced[: (hoNum_LR + hoNum_SFHA), year-1] = Luni_gini_135_forced
        
            # rec_ho_P_gini_135_DCM[: (hoNum_LR + hoNum_SFHA), year-1] = P_gini_135_DCM
            # rec_ho_B_gini_135_DCM[: (hoNum_LR + hoNum_SFHA), year-1] = B_gini_135_DCM
            # rec_ho_Luni_gini_135_DCM[: (hoNum_LR + hoNum_SFHA), year-1] = Luni_gini_135_DCM
        
            # rec_ho_P_gini_195_DCM[: (hoNum_LR + hoNum_SFHA), year-1] = P_gini_195_DCM
            # rec_ho_B_gini_195_DCM[: (hoNum_LR + hoNum_SFHA), year-1] = B_gini_195_DCM
            # rec_ho_Luni_gini_195_DCM[: (hoNum_LR + hoNum_SFHA), year-1] = Luni_gini_195_DCM
        
            rec_ho_Luni_gini_noPB[: (hoNum_LR + hoNum_SFHA), year-1] = Luni_gini_noPB # original loss: without insurance markets
            rec_ho_Luni_gini_noPB_fld[: (hoNum_LR + hoNum_SFHA), year-1] = Luni_gini_noPB_fld
            rec_ho_Luni_gini_noPB_wd[: (hoNum_LR + hoNum_SFHA), year-1] = Luni_gini_noPB_wd
            
            # records of number of households purchasing flood/wind insurance in low- and high-risk regions for different insurance markets by years
            rec_IMP_EQM_CGE_num_insured_fld_LR = list(zip(rec_IMP_EQM_CGE_num_insured_fld_LR, num_insured_fld_LR))
            rec_IMP_EQM_CGE_num_insured_wd_LR = list(zip(rec_IMP_EQM_CGE_num_insured_wd_LR, num_insured_wd_LR))
            rec_IMP_EQM_CGE_num_insured_fld_SFHA = list(zip(rec_IMP_EQM_CGE_num_insured_fld_SFHA, num_insured_fld_SFHA))
            rec_IMP_EQM_CGE_num_insured_wd_SFHA = list(zip(rec_IMP_EQM_CGE_num_insured_wd_SFHA, num_insured_wd_SFHA))
    
            
            ANS['lambda_LR'] = lambda_LR
            ANS['lambda_SFHA'] = lambda_SFHA
    
    
        else:
        
            # Calculate hs
            temp = hurr_sim[scenario-1, year-1]
            hs = np.mod(np.floor([temp/1e8, temp/1e6, temp/1e4, temp/1e2, temp]), 100)
            hs = hs[hs != 0]
            hs = hs.astype(int) - 1
        
            L_gini_LR = np.sum(ho_perhurrLoss_fld_LR[:, hs], axis=1) + np.sum(ho_perhurrLoss_wd_LR[:, hs], axis=1)
            L_gini_SFHA = np.sum(ho_perhurrLoss_fld_SFHA[:, hs], axis=1) + np.sum(ho_perhurrLoss_wd_SFHA[:, hs], axis=1)
            Luni_gini_noPB = np.concatenate((L_gini_LR, L_gini_SFHA))
            
            # Update rec_ho_Luni_gini_noPB for the given year
            rec_ho_Luni_gini_noPB[: (hoNum_LR + hoNum_SFHA), year-1] = Luni_gini_noPB
    
        scenario_endTime = time.time()
        # Calculate the elapsed time in seconds
        elapsed_time = scenario_endTime - t_SCENYEAR
    
    #if not FLAG_ins_option:
        # rec_ho_P_gini_1firm, rec_ho_B_gini_1firm, rec_ho_P_gini_4firms, rec_ho_B_gini_4firms, rec_ho_P_gini_135_forced, rec_ho_B_gini_135_forced, rec_ho_P_gini_135_DCM, rec_ho_B_gini_135_DCM, rec_ho_P_gini_195_DCM, rec_ho_B_gini_195_DCM = [np.zeros((931902, 20)) for _ in range(10)]
        # rec_ho_Luni_gini_1firm, rec_ho_Luni_gini_4firms, rec_ho_Luni_gini_135_forced, rec_ho_Luni_gini_135_DCM, rec_ho_Luni_gini_195_DCM = rec_ho_Luni_gini_noPB
    
    
    #%%
    # Calculate the overall running time in seconds
    running_time_seconds = time.time() - global_t0
    
    # Display overall running time in seconds
    print(f"Overall running time {running_time_seconds:.2f} s")
    
    # Convert running time to minutes
    running_time = int(running_time_seconds // 60)
    
    print(f"Running time in minutes: {running_time} min")
    
    ##these are not needed
    # ANS['priceAcq_alpha'] = priceAcq_alpha
    # ANS['priceAcq_beta'] = priceAcq_beta
    # ANS['priceRet_alpha'] = priceRet_alpha
    # ANS['priceRet_beta'] = priceRet_beta
    # ANS['J'] = J
    # ANS['budget'] = budget
    # ANS['running_time'] = running_time
    # ANS['year'] = year
    # ANS['hurr_sim'] = stru['stru']['hurr_sim'][0,0]
    #ANS['scenario'] = scenario
    ANS['clevel'] = clevel
    ANS['rec_IMP_EQM_CGE_num_insured_fld_LR'] = rec_IMP_EQM_CGE_num_insured_fld_LR
    ANS['rec_IMP_EQM_CGE_num_insured_wd_LR'] = rec_IMP_EQM_CGE_num_insured_wd_LR
    ANS['rec_IMP_EQM_CGE_num_insured_fld_SFHA'] = rec_IMP_EQM_CGE_num_insured_fld_SFHA
    ANS['rec_IMP_EQM_CGE_num_insured_wd_SFHA'] = rec_IMP_EQM_CGE_num_insured_wd_SFHA
    
    SOLUTION = {}
    
    SOLUTION['FLAG_acq_option'] = FLAG_acq_option
    SOLUTION['FLAG_ret_option'] = FLAG_ret_option
    SOLUTION['FLAG_selfret_option'] = FLAG_selfret_option
    SOLUTION['FLAG_ins_option'] = FLAG_ins_option
    SOLUTION['PARA_gov_benefitratio_forlow'] = 10 * PARA_gov_benefitratio_forlow
    SOLUTION['FLAG_load_gov_opt'] = FLAG_load_gov_opt
    SOLUTION['gov_optCenters'] = gov_optCenters
    SOLUTION['FLAG_DCM'] = FLAG_DCM
    
    SOLUTION.update(ANS)
    
    SOLUTION['acq_spend'] = acq_spend
    SOLUTION['retrofit_record'] = retrofit_record
    SOLUTION['retrofit_cost'] = retrofit_cost
    SOLUTION['retrofit_grants'] = retrofit_grants
    
    SOLUTION['paywd'] = paywd
    SOLUTION['payfld'] = payfld
    SOLUTION['rec_ho_P_gini_4firms'] = rec_ho_P_gini_4firms
    SOLUTION['rec_ho_B_gini_4firms'] = rec_ho_B_gini_4firms
    SOLUTION['rec_ho_Luni_gini_4firms'] = rec_ho_Luni_gini_4firms
    
    
    SOLUTION['rec_ho_Luni_gini_noPB'] = rec_ho_Luni_gini_noPB
    
    SOLUTION['running_time'] = running_time
    SOLUTION['rec_ho_invLoss_wd'] = rec_ho_invLoss_wd
    SOLUTION['rec_ho_invLoss_fld'] = rec_ho_invLoss_fld
    SOLUTION['rec_ho_invLoss_fld_20'] = rec_ho_invLoss_fld_20
    
    
    return SOLUTION
