#modified version from lib_compute_resilience_and_risk_financing.py
import matplotlib
# matplotlib.use('AGG')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas_helper import get_list_of_index_names, broadcast_simple, concat_categories
from scipy.interpolate import interp1d
from lib_gather_data import social_to_tx_and_gsp

from lib_country_dir import *

import seaborn as sns

sns_pal = sns.color_palette('Set1', n_colors=8, desat=.5)
q_colors = [sns_pal[0],sns_pal[1],sns_pal[2],sns_pal[3],sns_pal[5]]

def get_weighted_mean(q1,q2,q3,q4,q5,key,weight_key='pcwgt'):

    return [np.average(q1[key], weights=q1[weight_key]),
            np.average(q2[key], weights=q2[weight_key]),
            np.average(q3[key], weights=q3[weight_key]),
            np.average(q4[key], weights=q4[weight_key]),
            np.average(q5[key], weights=q5[weight_key])]

def get_weighted_median(q1,q2,q3,q4,q5,key):
    
    q1.sort_values(key, inplace=True)
    q2.sort_values(key, inplace=True)
    q3.sort_values(key, inplace=True)
    q4.sort_values(key, inplace=True)
    q5.sort_values(key, inplace=True)

    cumsum = q1.pcwgt.cumsum()
    cutoff = q1.pcwgt.sum() / 2.0
    median_q1 = round(q1[key][cumsum >= cutoff].iloc[0],3)

    cumsum = q2.pcwgt.cumsum()
    cutoff = q2.pcwgt.sum() / 2.0
    median_q2 = round(q2[key][cumsum >= cutoff].iloc[0],3)

    cumsum = q3.pcwgt.cumsum()
    cutoff = q3.pcwgt.sum() / 2.0
    median_q3 = round(q3[key][cumsum >= cutoff].iloc[0],3)

    cumsum = q4.pcwgt.cumsum()
    cutoff = q4.pcwgt.sum() / 2.0
    median_q4 = round(q4[key][cumsum >= cutoff].iloc[0],3)

    cumsum = q5.pcwgt.cumsum()
    cutoff = q5.pcwgt.sum() / 2.0
    median_q5 = round(q5[key][cumsum >= cutoff].iloc[0],3)

    return [median_q1,median_q2,median_q3,median_q4,median_q5]

def apply_policies(pol_str,macro,cat_info,hazard_ratios):
    
    print('CAT_INFO columns:\n',cat_info.columns)
    print('MACRO columns:\n',macro.columns)
    print('HAZARD_RATIOS columns:\n',hazard_ratios.columns)
    
    # POLICY: Reduce vulnerability of the poor by 5% of their current exposure
    if pol_str == '_exp095':
        print('--> POLICY('+pol_str+'): Reducing vulnerability of the poor by 5%!')

        cat_info.loc[cat_info.ispoor==1,'v']*=0.95 
    
    # POLICY: Reduce vulnerability of the rich by 5% of their current exposure    
    elif pol_str == '_exr095':
        print('--> POLICY('+pol_str+'): Reducing vulnerability of the rich by 5%!')

        cat_info.loc[cat_info.ispoor==0,'v']*=0.95 
        
    # POLICY: Increase income of the poor by 10%
    elif pol_str == '_pcinc_p_110':
        print('--> POLICY('+pol_str+'): Increase income of the poor by 10%')
        
        cat_info.loc[cat_info.ispoor==1,'c'] *= 1.10
        cat_info.loc[cat_info.ispoor==1,'pcinc'] *= 1.10
        cat_info.loc[cat_info.ispoor==1,'pcinc_ae'] *= 1.10

        cat_info['social'] = cat_info['social']/cat_info['pcinc']

    # POLICY: Increase social transfers to poor BY one third
    elif pol_str == '_soc133':
        print('--> POLICY('+pol_str+'): Increase social transfers to poor BY one third')

        # Cost of this policy = sum(social_topup), per person
        cat_info['social_topup'] = 0
        cat_info.loc[cat_info.ispoor==1,'social_topup'] = 0.333*cat_info.loc[cat_info.ispoor==1,['social','c']].prod(axis=1)

        cat_info.loc[cat_info.ispoor==1,'c']*=(1.0+0.333*cat_info.loc[cat_info.ispoor==1,'social'])
        cat_info.loc[cat_info.ispoor==1,'pcinc']*=(1.0+0.333*cat_info.loc[cat_info.ispoor==1,'social'])
        cat_info.loc[cat_info.ispoor==1,'pcinc_ae']*=(1.0+0.333*cat_info.loc[cat_info.ispoor==1,'social'])

        cat_info['social'] = (cat_info['social_topup']+cat_info['pcsoc'])/cat_info['pcinc']

    # POLICY: Decrease reconstruction time by 1/3
    elif pol_str == '_rec067':
        print('--> POLICY('+pol_str+'): Decrease reconstruction time by 1/3')
        macro['T_rebuild_K'] *= 0.666667


    # POLICY: Increase access to early warnings to 100%
    elif pol_str == '_ew100':
        print('--> POLICY('+pol_str+'): Increase access to early warnings to 100%')
        cat_info['shew'] = 1.0

    # POLICY: Decrease vulnerability of poor by 30%
    elif pol_str == '_vul070':
        print('--> POLICY('+pol_str+'): Decrease vulnerability of poor by 30%')

        cat_info.loc[cat_info.ispoor==1,'v']*=0.70

    # POLICY: Decrease vulnerability of rich by 30%
    elif pol_str == '_vul070r':
        print('--> POLICY('+pol_str+'): Decrease vulnerability of poor by 30%')
        
        cat_info.loc[cat_info.ispoor==0,'v']*=0.70

    elif pol_str != '': 
        print('What is this? --> ',pol_str)
        assert(False)

    return macro,cat_info,hazard_ratios

def compute_with_hazard_ratios(myCountry,pol_str,fname,macro,cat_info,economy,event_level,income_cats,default_rp,verbose_replace=True):

    #cat_info = cat_info[cat_info.c>0]
    hazard_ratios = pd.read_csv(fname, index_col=event_level+[income_cats])

    macro,cat_info,hazard_ratios = apply_policies(pol_str,macro,cat_info,hazard_ratios)

    #compute
    return process_input(myCountry,pol_str,macro,cat_info,hazard_ratios,economy,event_level,default_rp,verbose_replace=True)

def process_input(myCountry,pol_str,macro,cat_info,hazard_ratios,economy,event_level,default_rp,verbose_replace=True):
    flag1=False
    flag2=False

    #assert(False)
    if type(hazard_ratios)==pd.DataFrame:
        
        hazard_ratios = hazard_ratios.reset_index().set_index(economy).dropna()
        if 'Unnamed: 0' in hazard_ratios.columns: hazard_ratios = hazard_ratios.drop('Unnamed: 0',axis=1)
        
        #These lines remove countries in macro not in cat_info
        if myCountry == 'SL': hazard_ratios = hazard_ratios.dropna()
        else: hazard_ratios = hazard_ratios.fillna(0)
            
        common_places = [c for c in macro.index if c in cat_info.index and c in hazard_ratios.index]
        print(common_places)

        hazard_ratios = hazard_ratios.reset_index().set_index(event_level+['hhid'])

        # This drops 1 province from macro
        macro = macro.ix[common_places]

        # Nothing drops from cat_info
        cat_info = cat_info.ix[common_places]

        # Nothing drops from hazard_ratios
        hazard_ratios.to_csv('~/Desktop/my_file.csv')
        print(hazard_ratios)
        hazard_ratios = hazard_ratios.ix[common_places]

        if hazard_ratios.empty:
            hazard_ratios=None
			
    if hazard_ratios is None:
        hazard_ratios = pd.Series(1,index=pd.MultiIndex.from_product([macro.index,'default_hazard'],names=[economy, 'hazard']))
		
    #if hazard data has no hazard, it is broadcasted to default hazard
    if 'hazard' not in get_list_of_index_names(hazard_ratios):
        print('Should not be here: hazard not in \'hazard_ratios\'')
        hazard_ratios = broadcast_simple(hazard_ratios, pd.Index(['default_hazard'], name='hazard'))     
		
    #if hazard data has no rp, it is broadcasted to default rp
    if 'rp' not in get_list_of_index_names(hazard_ratios):
        print('Should not be here: RP not in \'hazard_ratios\'')
        hazard_ratios_event = broadcast_simple(hazard_ratios, pd.Index([default_rp], name='rp'))

    # Interpolates data to a more granular grid for return periods that includes all protection values that are potentially not the same in hazard_ratios.
    else:
        hazard_ratios_event = interpolate_rps(hazard_ratios, macro.protection,option=default_rp)
        hazard_ratios_event.to_csv("hazard_ratios_event.csv")

    # PSA input: original value of c
    avg_c = round(np.average(macro['gdp_pc_pp_prov'],weights=macro['pop'])/get_to_USD(myCountry),2)
    print('\nMean consumption (PSA): ',avg_c,' USD.\nMean GDP pc ',round(np.average(macro['gdp_pc_pp_prov'],weights=macro['pop'])/get_to_USD(myCountry),2),' USD.\n')

    cat_info['protection']=broadcast_simple(macro['protection'],cat_info.index)	

    ##add finance to diversification and taxation
    cat_info['social'] = unpack_social(macro,cat_info)

    ##cat_info['social']+= 0.1* cat_info['axfin']
    macro['tau_tax'], cat_info['gamma_SP'] = social_to_tx_and_gsp(economy,cat_info)
            
    #RECompute consumption from k and new gamma_SP and tau_tax
    cat_info['c'] = macro['avg_prod_k']*(1.-macro['tau_tax'])*cat_info['k']/(1.-cat_info['social'])
    # ^ this is per individual

    print('all weights ',cat_info['pcwgt'].sum())

    #plt.cla()
    #ax = plt.gca()
    #ci_heights, ci_bins = np.histogram(cat_info.c.clip(upper=50000),bins=50, weights=cat_info.pcwgt)
    #plt.gca().bar(ci_bins[:-1], ci_heights, width=ci_bins[1]-ci_bins[0], facecolor=q_colors[1], label='C2',alpha=0.4)
    #plt.legend()
    #fig = ax.get_figure()
    #fig.savefig('/Users/brian/Desktop/my_plots/'+myCountry+pol_str+'_consumption_init.pdf',format='pdf')

    print('Re-recalc mean cons (pc)',round(np.average((cat_info['c']*cat_info['pcwgt']).sum(level=economy)/macro['pop'],weights=macro['pop']),2),'(local curr).\n')    

    #rebuilding exponentially to 95% of initial stock in reconst_duration
    recons_rate = np.log(1/0.05) / macro['T_rebuild_K']  
    
    #Calculation of macroeconomic resilience
    macro['macro_multiplier'] =(macro['avg_prod_k']+recons_rate)/(macro['rho']+recons_rate)  #Gamma in the technical paper

    ####FORMATTING
    #gets the event level index
    event_level_index = hazard_ratios_event.reset_index().set_index(event_level).index #index composed on countries, hazards and rps.

    #Broadcast macro to event level 
    macro_event = broadcast_simple(macro,event_level_index)	

    #updates columns in macro with columns in hazard_ratios_event
    cols = [c for c in macro_event if c in hazard_ratios_event] #columns that are both in macro_event and hazard_ratios_event
    if not cols==[]:
        if verbose_replace:
            flag1=True
            print('Replaced in macro: '+', '.join(cols))
            macro_event[cols] =  hazard_ratios_event[cols]
    
    #Broadcast categories to event level
    cats_event = broadcast_simple(cat_info,  event_level_index)

    #updates columns in cats with columns in hazard_ratios_event	
    # applies mh ratios to relevant columns
    cols_c = [c for c in cats_event if c in hazard_ratios_event] #columns that are both in cats_event and hazard_ratios_event    
    if not cols_c==[]:
        hrb = broadcast_simple(hazard_ratios_event[cols_c], cat_info.index).reset_index().set_index(get_list_of_index_names(cats_event)) #explicitly broadcasts hazard ratios to contain income categories
        cats_event[cols_c] = hrb
        cats_event.to_csv("cats_event.csv")
        if verbose_replace:
            flag2=True
            print('Replaced in cats: '+', '.join(cols_c))
    if (flag1 and flag2):
        print('Replaced in both: '+', '.join(np.intersect1d(cols,cols_c)))

    return macro_event, cats_event, hazard_ratios_event 

def compute_dK(pol_str,macro_event, cats_event,event_level,affected_cats):

    cats_event_ia=concat_categories(cats_event,cats_event,index= affected_cats)
    cats_event.to_csv("cats_event.csv")
    
    #counts affected and non affected
    print('From here: \'hhwgt\' = nAffected and nNotAffected: households') 

    cats_event['fa'] = cats_event.fa.fillna(1E-8)
    
    # Instead, clipping in maps file
    #cats_event = cats_event.reset_index().set_index('hhid')
    #cats_event.loc[(cats_event.Division == 'Rotuma') & (cats_event.hazard == 'typhoon'),'fa'] *= 0.01
    #cats_event = cats_event.reset_index().set_index(['Division','hazard','rp','hhid'])

    # print(cats_event)

    for aWGT in ['hhwgt','pcwgt','pcwgt_ae']:
        myNaf = cats_event[aWGT]*cats_event.fa
        myNna = cats_event[aWGT]*(1-cats_event.fa)
        cats_event_ia[aWGT] = concat_categories(myNaf,myNna, index= affected_cats)    
        print('From here: \'weight\' = nAffected and nNotAffected: individuals') 
        
    #de_index so can access cats as columns and index is still event
    cats_event_ia = cats_event_ia.reset_index(['hhid', 'affected_cat']).sort_index()

    #actual vulnerability
    cats_event_ia['v_shew']=cats_event_ia['v']*(1-macro_event['pi']*cats_event_ia['shew']) 

    #capital losses and total capital losses
    cats_event_ia['dk']  = cats_event_ia[['k','v_shew']].prod(axis=1, skipna=False) #capital potentially be damaged 

    cats_event_ia.ix[(cats_event_ia.affected_cat=='na'), 'dk']=0

    #'provincial' losses
    # dk_event is WHEN the event happens--doesn't yet include RP/probability
    macro_event['dk_event']   =  cats_event_ia[['dk','pcwgt']].prod(axis=1,skipna=False).sum(level=event_level)
 
    #immediate consumption losses: direct capital losses plus losses through event-scale depression of transfers
    cats_event_ia['dc'] = (1-macro_event['tau_tax'])*cats_event_ia['dk']  +  cats_event_ia['gamma_SP']*macro_event['tau_tax'] *macro_event['dk_event'] 

    # This term is the impact on income from labor
    # cats_event_ia['dc_1'] = (1-macro_event['tau_tax'])*cats_event_ia['dk']

    # This term is the impact on national transfers
    # cats_event_ia['dc_2'] = cats_event_ia['gamma_SP']*macro_event['tau_tax'] *macro_event['dk_event'] 

    # NPV consumption losses accounting for reconstruction and productivity of capital (pre-response)
    cats_event_ia['dc_npv_pre'] = cats_event_ia['dc']*macro_event['macro_multiplier']

    return 	macro_event, cats_event_ia


def calculate_response(pol_str,macro_event,cats_event_ia,event_level,helped_cats,default_rp,option_CB,optionFee='tax',optionT='data', optionPDS='unif_poor', optionB='data',loss_measure='dk',fraction_inside=1, share_insured=.25):

    cats_event_iah = concat_categories(cats_event_ia,cats_event_ia, index= helped_cats).reset_index(helped_cats.name).sort_index().dropna()

    # Baseline case (no insurance):
    cats_event_iah['help_received'] = 0
    cats_event_iah['help_fee'] =0

    macro_event, cats_event_iah = compute_response(pol_str,macro_event, cats_event_iah, event_level,default_rp,option_CB,optionT=optionT, 
                                                   optionPDS=optionPDS, optionB=optionB, optionFee=optionFee, fraction_inside=fraction_inside, loss_measure = loss_measure)
        
    cats_event_iah.drop('protection',axis=1, inplace=True)	      

    return macro_event, cats_event_iah
	
def compute_response(pol_str, macro_event, cats_event_iah, event_level, default_rp, option_CB,optionT='data', optionPDS='unif_poor', optionB='data', optionFee='tax', fraction_inside=1, loss_measure='dk'):    

    print('NB: when summing over cats_event_iah, be aware that each hh appears 4X in the file: {a,na}x{helped,not_helped}')
    
    """Computes aid received,  aid fee, and other stuff, from losses and PDS options on targeting, financing, and dimensioning of the help.
    Returns copies of macro_event and cats_event_iah updated with stuff"""
    macro_event    = macro_event.copy()
    cats_event_iah = cats_event_iah.copy()

    macro_event['fa'] = (cats_event_iah.loc[(cats_event_iah.affected_cat=='a'),'pcwgt'].sum(level=event_level)/(cats_event_iah['pcwgt'].sum(level=event_level))).fillna(1E-8)

    #print('Check fa: ',macro_event['fa'].sum(level=event_level))
    # Need to check whether everything is right here:
    #print(cats_event_iah['pcwgt'].sum(level=event_level))
    #print(cats_event_iah['hhwgt'].sum(level=event_level))
    #print(cats_event_iah['pcwgt'].sum(level=['hazard','rp']))
    #print(cats_event_iah['hhwgt'].sum(level=['hazard','rp']))

    # Edit: no factor of 2 in denominator because we're only summing affected households here
    #macro_event['fa'] = agg_to_event_level(cats_event_iah,'fa',event_level)/2 # because cats_event_ia is duplicated in cats_event_iah, cats_event_iah.n.sum(level=event_level) is 2 instead of 1, here /2 is to correct it.

    ####targeting errors
    if optionT=='perfect':
        macro_event['error_incl'] = 0
        macro_event['error_excl'] = 0    
    elif optionT=='prop_nonpoor_lms':
        macro_event['error_incl'] = 0
        macro_event['error_excl'] = 1-25/80  #25% of pop chosen among top 80 DO receive the aid
    elif optionT=='data':
        macro_event['error_incl']=(1)/2*macro_event['fa']/(1-macro_event['fa'])
        macro_event['error_excl']=(1)/2
    elif optionT=='x33':
        macro_event['error_incl']= .33*macro_event['fa']/(1-macro_event['fa'])
        macro_event['error_excl']= .33
    elif optionT=='incl':
        macro_event['error_incl']= .33*macro_event['fa']/(1-macro_event['fa'])
        macro_event['error_excl']= 0
    elif optionT=='excl':
        macro_event['error_incl']= 0
        macro_event['error_excl']= 0.33

    else:
        print('unrecognized targeting error option '+optionT)
        return None
        
    cats_event_iah.to_csv("cats_event_iah.csv")
    
    #counting (mind self multiplication of n)
    for aWGT in ['hhwgt','pcwgt','pcwgt_ae']:
        cats_event_iah.ix[(cats_event_iah.helped_cat=='helped')    & (cats_event_iah.affected_cat=='a') ,aWGT]*=(1-macro_event['error_excl'])
        cats_event_iah.ix[(cats_event_iah.helped_cat=='not_helped')& (cats_event_iah.affected_cat=='a') ,aWGT]*=(  macro_event['error_excl'])
        cats_event_iah.ix[(cats_event_iah.helped_cat=='helped')    & (cats_event_iah.affected_cat=='na'),aWGT]*=(  macro_event['error_incl'])  
        cats_event_iah.ix[(cats_event_iah.helped_cat=='not_helped')& (cats_event_iah.affected_cat=='na'),aWGT]*=(1-macro_event['error_incl'])
    ###!!!! n is one again from here.
    #print(cats_event_iah.pcwgt.sum(level=event_level))
	

    # MAXIMUM NATIONAL SPENDING ON SCALE UP
    macro_event['max_increased_spending'] = 0.05

    # max_aid is per cap, and it is constant for all disasters & provinces
    # --> If this much were distributed to everyone in the country, it would be 5% of GDP
    # --> All of these '.mean()' are because I'm bad at Pandas. The arrays are all the same value, and we want to pull out a single instance here
    macro_event['max_aid'] = macro_event['max_increased_spending'].mean()*macro_event[['gdp_pc_pp_prov','pop']].prod(axis=1).sum(level=['hazard','rp']).mean()/macro_event['pop'].sum(level=['hazard','rp']).mean()

    if optionFee == 'insurance_premium':
        temp = cats_event_iah.copy()
	
    if optionPDS=='no':
        macro_event['aid'] = 0
        macro_event['need']=0
        cats_event_iah['help_received']=0
        optionB='no'
        
    elif optionPDS=='unif_poor':

        cats_event_iah['help_received'] = 0        

        # For this policy:
        # --> help_received = 0.8*average losses of lowest quintile (households)
        cats_event_iah.loc[(cats_event_iah.helped_cat=='helped')&(cats_event_iah.affected_cat=='a'),'help_received'] = macro_event['shareable']*cats_event_iah.loc[(cats_event_iah.affected_cat=='a')&(cats_event_iah.quintile==1),[loss_measure,'pcwgt']].prod(axis=1).sum(level=event_level)/cats_event_iah.loc[(cats_event_iah.affected_cat=='a')&(cats_event_iah.quintile==1),'pcwgt'].sum(level=event_level)

        # These should be zero here, but just to make sure...
        cats_event_iah.ix[(cats_event_iah.helped_cat=='not_helped'),'help_received']=0
        cats_event_iah.ix[(cats_event_iah.affected_cat=='na'),'help_received']=0
        # Could test with these:
        #assert(cats_event_iah.ix[(cats_event_iah.helped_cat=='not_helped'),'help_received'].sum()==0)
        #assert(cats_event_iah.ix[(cats_event_iah.affected_cat=='na'),'help_received'].sum()==0)


    elif optionPDS=='unif_poor_only':
        cats_event_iah.ix[(cats_event_iah.helped_cat=='helped')&(cats_event_iah.affected_cat=='a')&(cats_event_iah.quintile==1),'help_received']=macro_event['shareable']*cats_event_iah.loc[(cats_event_iah.affected_cat=='a')&(cats_event_iah.quintile==1),[loss_measure,'pcwgt']].prod(axis=1).sum(level=event_level)/cats_event_iah.loc[(cats_event_iah.affected_cat=='a')&(cats_event_iah.quintile==1),'pcwgt'].sum(level=event_level)

        # These should be zero here, but just to make sure...
        cats_event_iah.ix[(cats_event_iah.helped_cat=='not_helped')|(cats_event_iah.quintile > 1),'help_received']=0
        cats_event_iah.ix[(cats_event_iah.affected_cat=='na')|(cats_event_iah.quintile > 1),'help_received']=0

        print('Calculating loss measure\n')

    elif optionPDS=='prop':
        if not 'has_received_help_from_PDS_cat' in cats_event_iah.columns:
            print(optionPDS)
            cats_event_iah.ix[(cats_event_iah.helped_cat=='helped')& (cats_event_iah.affected_cat=='a'),'help_received']= macro_event['shareable']*cats_event_iah.ix[(cats_event_iah.helped_cat=='helped')& (cats_event_iah.affected_cat=='a'),loss_measure]
            cats_event_iah.ix[(cats_event_iah.helped_cat=='helped')& (cats_event_iah.affected_cat=='na'),'help_received']= macro_event['shareable']*cats_event_iah.ix[(cats_event_iah.helped_cat=='helped')& (cats_event_iah.affected_cat=='a'),loss_measure]
            cats_event_iah.ix[cats_event_iah.helped_cat=='not_helped','help_received']=0		

        else:
            cats_event_iah.ix[(cats_event_iah.helped_cat=='helped')& (cats_event_iah.affected_cat=='a')  & (cats_event_iah.has_received_help_from_PDS_cat=='helped'),'help_received']= macro_event['shareable']*cats_event_iah.ix[(cats_event_iah.helped_cat=='helped')& (cats_event_iah.affected_cat=='a')  & (cats_event_iah.has_received_help_from_PDS_cat=='helped'),loss_measure]
            cats_event_iah.ix[(cats_event_iah.helped_cat=='helped')& (cats_event_iah.affected_cat=='na')  & (cats_event_iah.has_received_help_from_PDS_cat=='helped'),'help_received']= macro_event['shareable']*cats_event_iah.ix[(cats_event_iah.helped_cat=='helped')& (cats_event_iah.affected_cat=='a')  & (cats_event_iah.has_received_help_from_PDS_cat=='helped'),loss_measure]			
            cats_event_iah.ix[(cats_event_iah.helped_cat=='helped')& (cats_event_iah.affected_cat=='a')  & (cats_event_iah.has_received_help_from_PDS_cat=='not_helped'),'help_received']= macro_event['shareable']*cats_event_iah.ix[(cats_event_iah.helped_cat=='helped')& (cats_event_iah.affected_cat=='a')  & (cats_event_iah.has_received_help_from_PDS_cat=='helped'),loss_measure]
            cats_event_iah.ix[(cats_event_iah.helped_cat=='helped')& (cats_event_iah.affected_cat=='na')  & (cats_event_iah.has_received_help_from_PDS_cat=='not_helped'),'help_received']= macro_event['shareable']*cats_event_iah.ix[(cats_event_iah.helped_cat=='helped')& (cats_event_iah.affected_cat=='a')  & (cats_event_iah.has_received_help_from_PDS_cat=='helped'),loss_measure]			
            cats_event_iah.ix[cats_event_iah.helped_cat=='not_helped','help_received']=0           
		
    # What is the function of need?
    # --> Defining it as the cost of disaster assistance distributed among all people in each province
    # --> 'need' is household, not per person!!
    macro_event['need'] = cats_event_iah[['help_received','pcwgt']].prod(axis=1).sum(level=event_level)/(cats_event_iah['pcwgt'].sum(level=event_level))
    macro_event['need_tot'] = cats_event_iah[['help_received','pcwgt']].prod(axis=1).sum(level=event_level)

    #actual aid reduced by capacity
    if optionPDS == 'no':
        macro_event['my_help_fee'] = 0
    elif optionB=='data' or optionB=='unif_poor':

        # See discussion above. This is the cost 
        print('No upper limit on help_received coded at this point...if we did exceed 5% of GDP, the help_fee would just be capped')
        macro_event['my_help_fee'] = macro_event['need'].clip(upper=macro_event['max_aid'])

    else:
        assert(False)
        
    #elif optionB=='max01':
    #    macro_event['max_aid'] = 0.01*macro_event['gdp_pc_pp_nat']
    #    macro_event['aid'] = (macro_event['need']).clip(upper=macro_event['max_aid']) 
    #elif optionB=='max05':
    #    macro_event['max_aid'] = 0.05*macro_event['gdp_pc_pp_nat']
    #    macro_event['aid'] = (macro_event['need']).clip(upper=macro_event['max_aid'])
    #elif optionB=='unlimited':
    #    macro_event['aid'] = macro_event['need']
    #elif optionB=='one_per_affected':
    #    d = cats_event_iah.ix[(cats_event_iah.affected_cat=='a')]        
    #    d['un']=1
    #    macro_event['need'] = agg_to_event_level(d,'un',event_level)
    #    macro_event['aid'] = macro_event['need']
    #elif optionB=='one_per_helped':
    #    d = cats_event_iah.ix[(cats_event_iah.helped_cat=='helped')]        
    #    d['un']=1
    #    macro_event['need'] = agg_to_event_level(d,'un',event_level)
    #    macro_event['aid'] = macro_event['need']
    #elif optionB=='one':
    #    macro_event['aid'] = 1
    #elif optionB=='no':
    #    pass	
    
    #NO!!!!!
    #if optionPDS=='unif_poor':
        # NO. we have already calculated help_received.
        #macro_event['unif_aid'] = macro_event['aid']

    #elif optionPDS=='unif_poor_only':
    #    macro_event['unif_aid'] = macro_event['aid']/(cats_event_iah.ix[(cats_event_iah.helped_cat=='helped')&(cats_event_iah.quintile==1),'pcwgt'].sum(level=event_level)) 
    #    cats_event_iah.ix[(cats_event_iah.helped_cat=='helped')&(cats_event_iah.quintile==1),'help_received'] = macro_event['unif_aid']
    #    cats_event_iah.ix[(cats_event_iah.helped_cat=='not_helped')|(cats_event_iah.quintile==1),'help_received']=0
    
    #if optionPDS=='prop':
    #    cats_event_iah['help_received'] = macro_event['aid']/macro_event['need']*cats_event_iah['help_received'] 		
		
    if optionFee=='tax':

        # Original code:
        #cats_event_iah['help_fee'] = fraction_inside*macro_event['aid']*cats_event_iah['k']/agg_to_event_level(cats_event_iah,'k',event_level)

        # I *think* this is only transfers inside province 
        # not going to multiply by weight here, because then it would look like some households are paying hundreds of times more...save that for the histogram
        # but we do need to multiply 'my_help_fee' by weight, 
        # --> If 1 hh had all the capital, its help_fee would be my_help_fee, which is the per capita value

        cats_event_iah['help_fee'] = 0
        cats_event_iah.loc[cats_event_iah.pcwgt != 0,'help_fee'] = (cats_event_iah.loc[cats_event_iah.pcwgt != 0,['help_received','pcwgt']].prod(axis=1).sum(level=event_level) * 
                                                                     # ^ total expenditure
                                                                     (cats_event_iah.loc[cats_event_iah.pcwgt != 0,['k','pcwgt']].prod(axis=1) /
                                                                      cats_event_iah.loc[cats_event_iah.pcwgt != 0,['k','pcwgt']].prod(axis=1).sum(level=event_level)) /
                                                                     # ^ weighted average of capital
                                                                     cats_event_iah.loc[cats_event_iah.pcwgt != 0,'pcwgt']) 
                                                                     # ^ help_fee is per individual!

        #print(macro_event['need_tot'])
        #print(cats_event_iah['k'].sum(level=event_level))
        #print(cats_event_iah[['help_fee','weight']].prod(axis=1).sum(level=event_level))
        #assert(False)

    elif optionFee=='insurance_premium':
        print(optionFee)
        cats_event_iah = cats_event_iah.reset_index().set_index([economy,'hazard','helped_cat',  'affected_cat',     'hhid','rp']) 

#        cats_event_iah = cats_event_iah.reset_index().set_index(['province','hazard','helped_cat',  'affected_cat',     'hhid','has_received_help_from_PDS_cat','rp'])
        averaged,proba_serie = average_over_rp(cats_event_iah['help_received'],default_rp,cats_event_iah['protection'].squeeze())
#        proba_serie = proba_serie.reset_index().set_index(['province','hazard','helped_cat',  'affected_cat',     'hhid','has_received_help_from_PDS_cat','rp']) 
        proba_serie = proba_serie.reset_index().set_index([economy,'hazard','helped_cat',  'affected_cat',     'hhid','rp']) 
        cats_event_iah['help_received'] = broadcast_simple(averaged,cats_event_iah.index)
#        cats_event_iah.help_received = cats_event_iah.help_received/proba_serie.prob
        cats_event_iah = cats_event_iah.reset_index().set_index(event_level)
#        aa = cats_event_iah.loc[('Ampara',slice(None),[5,10]),['help_received','help_fee','helped_cat','affected_cat']]
#        aa1 = aa[aa.helped_cat=='helped']
#        aa2 = aa[aa.helped_cat=='not_helped']
        cats_event_iah.ix[cats_event_iah.helped_cat=='helped','help_received_ins'] = cats_event_iah.ix[cats_event_iah.helped_cat=='helped','help_received']
        cats_event_iah.ix[cats_event_iah.helped_cat=='not_helped','help_received_ins'] = cats_event_iah.ix[cats_event_iah.helped_cat=='helped','help_received']
        
        ###
        cats_event_iah['help_fee'] = agg_to_event_level(cats_event_iah,'help_received',event_level)/(cats_event_iah.n.sum(level=event_level))*cats_event_iah['help_received_ins']/agg_to_event_level(cats_event_iah,'help_received_ins',event_level)
        print('Calculation of help_fee is definitely wrong!')
        assert(False)
        ###

        cats_event_iah.ix[cats_event_iah.affected_cat=='na','help_received'] = 0
        cats_event_iah.ix[cats_event_iah.helped_cat=='not_helped','help_received'] = 0
#        cats_event_iah.drop('help_received_ins',axis=1,inplace=True)
#       print(cats_event_iah[['help_fee','help_received']])
#       print(temp[['help_fee','help_received']])
#        cats_event_iah[['help_received','help_fee']]+=temp[['help_received','help_fee']]
    return macro_event, cats_event_iah


def compute_dW(myCountry,pol_str,macro_event,cats_event_iah,event_level,option_CB,return_stats=True,return_iah=True):

    # check that each of these is per individual:
    cats_event_iah['dc_npv_post'] = cats_event_iah['dc_npv_pre'] -  cats_event_iah['help_received']  + cats_event_iah['help_fee']*option_CB 
    cats_event_iah['dw'] = calc_delta_welfare(cats_event_iah, macro_event) 
    #cats_event_iah.to_csv('~/Desktop/my_iah.csv')

    plt.cla()
    ax = plt.gca()
    c0_bins = None
    cum_heights = None

    for aQuint in range(1,6):
        mean = np.average(cats_event_iah.loc[(cats_event_iah.affected_cat == 'a') & (cats_event_iah.quintile == aQuint),'dw'],
                          weights=cats_event_iah.loc[(cats_event_iah.affected_cat == 'a') & (cats_event_iah.quintile == aQuint),'pcwgt'])
        print(aQuint,mean)
        
        ci_heights, ci_bins = np.histogram(cats_event_iah.loc[(cats_event_iah.affected_cat == 'a') & (cats_event_iah.quintile == aQuint),'dw'].clip(upper=0.0005),bins=50, 
                                           weights=cats_event_iah.loc[(cats_event_iah.affected_cat == 'a') & (cats_event_iah.quintile == aQuint),'pcwgt'])
        if c0_bins == None: c0_bins = ci_bins
        if cum_heights == None: cum_heights = [0. for aBin in ci_heights]

        ax.bar(c0_bins[:-1], ci_heights, width=c0_bins[1]-c0_bins[0],bottom=cum_heights, facecolor=q_colors[aQuint-1], label='Q'+str(aQuint)+r' ($\mu$ = '+str(round(mean*1E4,2))+'E-4)',alpha=0.4)
        cum_heights += ci_heights
    
    plt.legend()
    plt.ylabel('Population')
    plt.xlabel('dw')
    fig = ax.get_figure()
    # fig.savefig('/../check_plots/'+myCountry+pol_str+'_dw.pdf',format='pdf')
    

    #aggregates dK and delta_W at df level
    # --> dK, dW are averages per individual
    dK      = cats_event_iah[['dk','pcwgt']].prod(axis=1).sum(level=event_level)/cats_event_iah['pcwgt'].sum(level=event_level)
    delta_W = cats_event_iah[['dw','pcwgt']].prod(axis=1).sum(level=event_level)/cats_event_iah['pcwgt'].sum(level=event_level)

    ###########
    #OUTPUT
    df_out = pd.DataFrame(index=macro_event.index)

    # dktot is already summed with RP -- just add them normally to get losses
    df_out['dK'] = dK
    df_out['dKtot']=dK*cats_event_iah['pcwgt'].sum(level=event_level)#macro_event['pop']

    df_out['delta_W']    =delta_W
    df_out['delta_W_tot']=delta_W*cats_event_iah['pcwgt'].sum(level=event_level)#macro_event['pop'] 

    df_out['average_aid_cost_pc'] = macro_event['my_help_fee']
    
    if return_stats:
        if not 'has_received_help_from_PDS_cat' in cats_event_iah.columns:
            stats = np.setdiff1d(cats_event_iah.columns,event_level+['helped_cat',  'affected_cat',     'hhid'])
        else:
            stats = np.setdiff1d(cats_event_iah.columns,event_level+['helped_cat',  'affected_cat',     'hhid','has_received_help_from_PDS_cat'])		
		
        df_stats = agg_to_event_level(cats_event_iah, stats,event_level)
        # if verbose_replace:
        print('stats are '+','.join(stats))
        df_out[df_stats.columns]=df_stats 
		    
    if return_iah:
        return df_out,cats_event_iah
    else: 
        return df_out
    
	
def process_output(pol_str,out,macro_event,economy,default_rp,return_iah=True,is_local_welfare=True):

    #unpacks if needed
    if return_iah:
        dkdw_event,cats_event_iah  = out


    else:
        dkdw_event = out

    ##AGGREGATES LOSSES
    #Averages over return periods to get dk_{hazard} and dW_{hazard}
#    print(dkdw_event.shape)
#    print(macro_event.shape)
    dkdw_h = average_over_rp1(dkdw_event,default_rp,macro_event['protection']).set_index(macro_event.index)
    macro_event[dkdw_h.columns]=dkdw_h

    #computes socio economic capacity and risk at economy level
    macro = calc_risk_and_resilience_from_k_w(macro_event,cats_event_iah,economy, is_local_welfare)

    ###OUTPUTS
    if return_iah:
        return macro, cats_event_iah
    else:
        return macro
	
def unpack_social(m,cat):
    """Compute social from gamma_SP, taux tax and k and avg_prod_k"""
    c  = cat.c
    gs = cat.gamma_SP

    social = gs*m.gdp_pc_pp_nat*m.tau_tax/(c+1.0e-10) #gdp*tax should give the total social protection. gs=each one's social protection/(total social protection). social is defined as t(which is social protection)/c_i(consumption)

    return social
	
def interpolate_rps(fa_ratios,protection_list,option):
    ###INPUT CHECKING
    default_rp=option
    if fa_ratios is None:
        return None
    
    if default_rp in fa_ratios.index:
        return fa_ratios
    
    flag_stack= False
    if 'rp' in get_list_of_index_names(fa_ratios):
        fa_ratios = fa_ratios.unstack('rp')
        flag_stack = True
 
    if type(protection_list) in [pd.Series, pd.DataFrame]:
        protection_list=protection_list.squeeze().unique().tolist()
        
    #in case of a Multicolumn dataframe, perform this function on each one of the higher level columns
    if type(fa_ratios.columns)==pd.MultiIndex:
        keys = fa_ratios.columns.get_level_values(0).unique()
        return pd.concat({col:interpolate_rps(fa_ratios[col],protection_list,option) for col in  keys}, axis=1).stack('rp')


    ### ACTUAL FUNCTION    
    #figures out all the return periods to be included
    all_rps = list(set(protection_list+fa_ratios.columns.tolist()))
    
    fa_ratios_rps = fa_ratios.copy()
    
    #extrapolates linear towards the 0 return period exposure  (this creates negative exposure that is tackled after interp) (mind the 0 rp when computing probas)
    if len(fa_ratios_rps.columns)==1:
        fa_ratios_rps[0] = fa_ratios_rps.squeeze()
    else:
        fa_ratios_rps[0]=0 #the line below was creating issues when there were nans in fa_ratios_rps (ie when different hazards had different rp)
        # fa_ratios_rps[0]=fa_ratios_rps.iloc[:,0]- fa_ratios_rps.columns[0]*(
        # fa_ratios_rps.iloc[:,1]-fa_ratios_rps.iloc[:,0])/(
        # fa_ratios_rps.columns[1]-fa_ratios_rps.columns[0])
        
    fa_ratios_rps = fa_ratios_rps.reindex_axis(sorted(fa_ratios_rps.columns), axis=1)
    fa_ratios_rps = fa_ratios_rps.interpolate(axis=1,limit_direction="both",downcast="infer")
    
    #add new, interpolated values for fa_ratios, assuming constant exposure on the right
    x = fa_ratios_rps.columns.values
    y = fa_ratios_rps.values
    fa_ratios_rps= pd.concat(
        [pd.DataFrame(interp1d(x,y,bounds_error=False)(all_rps),index=fa_ratios_rps.index, columns=all_rps)]
        ,axis=1).sort_index(axis=1).clip(lower=0).fillna(method='pad',axis=1)
    fa_ratios_rps.columns.name='rp'

    if flag_stack:
        fa_ratios_rps = fa_ratios_rps.stack('rp')
    
    return fa_ratios_rps    

def agg_to_economy_level (df, seriesname,economy):
    """ aggregates seriesname in df (string of list of string) to economy (country) level using n in df as weight
    does NOT normalize weights to 1."""
    return (df[seriesname].T*df['pcwgt']).T.sum()#level=economy)
	
def agg_to_event_level (df, seriesname,event_level):
    """ aggregates seriesname in df (string of list of string) to event level (country, hazard, rp) across income_cat and affected_cat using n in df as weight
    does NOT normalize weights to 1."""
    return (df[seriesname].T*df['pcwgt']).T.sum(level=event_level)
    
def calc_delta_welfare(micro, macro):
    """welfare cost from consumption before (c) 
    an after (dc_npv_post) event. Line by line"""
    #computes welfare losses per category
    dwA = welf1(micro['c']/macro['rho'], macro['income_elast'], micro['c_5']/macro['rho'])
    dwB = -1*welf1(micro['c']/macro['rho']-micro['dc_npv_post'], macro['income_elast'],micro['c_5']/macro['rho'])

    dw= (welf1(micro['c']/macro['rho'], macro['income_elast'], micro['c_5']/macro['rho'])
                 - welf1(micro['c']/macro['rho']-micro['dc_npv_post'], macro['income_elast'],micro['c_5']/macro['rho']))
    
    return dw
	
def welf1(c,elast,comp):
    """"Welfare function"""
    y=(c**(1-elast)-1)/(1-elast)
    row1 = c<comp
    row2 = c<=0
    y[row1]=(comp**(1-elast)-1)/(1-elast) + comp**(-elast)*(c-comp)
#    y[row2]=(comp**(1-elast)-1)/(1-elast) + comp**(-elast)*(0-comp)
    return y
	
def welf(c,elast):
    y=(c**(1-elast)-1)/(1-elast)
    return y
	
def average_over_rp(df,default_rp,protection=None):        
    """Aggregation of the outputs over return periods"""    
    if protection is None:
        protection=pd.Series(0,index=df.index)        

    #just drops rp index if df contains default_rp
    if default_rp in df.index.get_level_values('rp'):
        print('default_rp detected, dropping rp')
        return (df.T/protection).T.reset_index('rp',drop=True)
           
    df=df.copy().reset_index('rp')
    protection=protection.copy().reset_index('rp',drop=True)
    
    #computes frequency of each return period
    return_periods=np.unique(df['rp'].dropna())

    proba = pd.Series(np.diff(np.append(1/return_periods,0)[::-1])[::-1],index=return_periods) #removes 0 from the rps 

    #matches return periods and their frequency
    proba_serie=df['rp'].replace(proba).rename('prob')
    proba_serie1 = pd.concat([df.rp,proba_serie],axis=1)
#    print(proba_serie.shape)
#    print(df.rp.shape)
#    print(protection)
    #removes events below the protection level
    proba_serie[protection>df.rp] =0

    #handles cases with multi index and single index (works around pandas limitation)
    idxlevels = list(range(df.index.nlevels))
    if idxlevels==[0]:
        idxlevels =0
#    print(idxlevels)
#    print(get_list_of_index_names(df))
#    print(df.head(10))
    #average weighted by proba
    averaged = df.mul(proba_serie,axis=0).sum(level=idxlevels).drop('rp',axis=1) # frequency times each variables in the columns including rp.
    return averaged,proba_serie1 #here drop rp.
	
	
def average_over_rp1(df,default_rp,protection=None):        
    """Aggregation of the outputs over return periods"""    
    if protection is None:
        protection=pd.Series(0,index=df.index)        

    #just drops rp index if df contains default_rp
    if default_rp in df.index.get_level_values('rp'):
        print('default_rp detected, dropping rp')
        return (df.T/protection).T.reset_index('rp',drop=True)
           
    df=df.copy().reset_index('rp')
    protection=protection.copy().reset_index('rp',drop=True)
    
    #computes frequency of each return period
    return_periods=np.unique(df['rp'].dropna())

    proba = pd.Series(np.diff(np.append(1/return_periods,0)[::-1])[::-1],index=return_periods) #removes 0 from the rps 

    #matches return periods and their frequency
    proba_serie=df['rp'].replace(proba)
#    print(proba_serie.shape)
#    print(df.rp.shape)
#    print(protection)
    #removes events below the protection level
    proba_serie[protection>df.rp] =0

    #handles cases with multi index and single index (works around pandas limitation)
    idxlevels = list(range(df.index.nlevels))
    if idxlevels==[0]:
        idxlevels =0

    #average weighted by proba
    averaged = df.mul(proba_serie,axis=0)#.sum(level=idxlevels) # frequency times each variables in the columns including rp.
    return averaged.drop('rp',axis=1) #here drop rp.

def calc_risk_and_resilience_from_k_w(df, cats_event_iah,economy,is_local_welfare=True): 
    """Computes risk and resilience from dk, dw and protection. Line by line: multiple return periods or hazard is transparent to this function"""
    df=df.copy()    
    ############################
    #Expressing welfare losses in currency 
    #discount rate
    rho = df['rho']
    h=1e-4

    # flag: gdp_pc_pp is per household. making sure all other things are per household
    if is_local_welfare:
        wprime =(welf(df['gdp_pc_pp_prov']/rho+h,df['income_elast'])-welf(df['gdp_pc_pp_prov']/rho-h,df['income_elast']))/(2*h)
    else:
        nat_GDP_pc = np.average(df['gdp_pc_pp_nat'])
        wprime =(welf(nat_GDP_pc/rho+h,df['income_elast'])-welf(nat_GDP_pc/rho-h,df['income_elast']))/(2*h)
        
    dWref   = wprime*df['dK']
    
    #expected welfare loss (per family and total)
    df['wprime'] = wprime
    df['dWref'] = dWref
    df['dWpc_currency'] = df['delta_W']/wprime 
    df['dWtot_currency']=df['dWpc_currency']*cats_event_iah['pcwgt'].sum(level=[economy,'hazard','rp'])#*df['pop']
    
    #Risk to welfare as percentage of local GDP
    df['risk']= df['dWpc_currency']/(df['gdp_pc_pp_prov'])
    
    ############
    #SOCIO-ECONOMIC CAPACITY)
    df['resilience'] =dWref/(df['delta_W'] )

    ############
    #RISK TO ASSETS
    df['risk_to_assets']  =df.resilience* df.risk
    
    return df
