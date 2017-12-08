#modified version from lib_compute_resilience_and_risk_financing.py
import matplotlib
matplotlib.use('AGG')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas_helper import get_list_of_index_names, broadcast_simple, concat_categories
from scipy.interpolate import interp1d
from lib_gather_data import social_to_tx_and_gsp
import math
from lib_country_dir import *
import seaborn as sns

pd.set_option('display.width', 200)

sns_pal = sns.color_palette('Set1', n_colors=8, desat=.5)
q_colors = [sns_pal[0],sns_pal[1],sns_pal[2],sns_pal[3],sns_pal[5]]

const_nom_reco_rate, const_rho, const_ie = None, None, None
const_pds_rate = None

debug = '~/Desktop/BANK/debug/'

def get_weighted_mean(q1,q2,q3,q4,q5,key,weight_key='pcwgt'):
    
    if q1.shape[0] > 0:
        my_ret = [np.average(q1[key], weights=q1[weight_key])]
    else: my_ret = [0]
    
    if q2.shape[0] > 0:
        my_ret.append(np.average(q2[key], weights=q2[weight_key]))
    else: my_ret.append(0)

    if q3.shape[0] > 0:
        my_ret.append(np.average(q3[key], weights=q3[weight_key]))
    else: my_ret.append(0)

    if q4.shape[0] > 0:
        my_ret.append(np.average(q4[key], weights=q4[weight_key]))
    else: my_ret.append(0)

    if q5.shape[0] > 0:
        my_ret.append(np.average(q5[key], weights=q5[weight_key]))
    else: my_ret.append(0)    

    return my_ret

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

    elif pol_str == '_soc133':
        # POLICY: Increase social transfers to poor TO one third
        print('--> POLICY('+pol_str+'): Increase social transfers to poor TO one third of their income')
        cat_info.loc[(cat_info.ispoor==1)&(cat_info.social<0.334),'social'] = 0.334

        ########################################################
        # POLICY: Increase social transfers to poor BY one third
        #print('--> POLICY('+pol_str+'): Increase social transfers to poor BY one third')

        # Cost of this policy = sum(social_topup), per person
        #cat_info['social_topup'] = 0
        #cat_info.loc[cat_info.ispoor==1,'social_topup'] = 0.333*cat_info.loc[cat_info.ispoor==1,['social','c']].prod(axis=1)

        #cat_info.loc[cat_info.ispoor==1,'c']*=(1.0+0.333*cat_info.loc[cat_info.ispoor==1,'social'])
        #cat_info.loc[cat_info.ispoor==1,'pcinc']*=(1.0+0.333*cat_info.loc[cat_info.ispoor==1,'social'])
        #cat_info.loc[cat_info.ispoor==1,'pcinc_ae']*=(1.0+0.333*cat_info.loc[cat_info.ispoor==1,'social'])

        #cat_info['social'] = (cat_info['social_topup']+cat_info['pcsoc'])/cat_info['pcinc']
        ########################################################

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

    elif pol_str == '_noPT': pass

    elif pol_str != '':
        print('What is this? --> ',pol_str)
        assert(False)

    return macro,cat_info,hazard_ratios

def compute_with_hazard_ratios(myCountry,pol_str,fname,macro,cat_info,economy,event_level,income_cats,default_rp,rm_overlap,verbose_replace=True):

    #cat_info = cat_info[cat_info.c>0]
    hazard_ratios = pd.read_csv(fname, index_col=event_level+[income_cats])

    macro,cat_info,hazard_ratios = apply_policies(pol_str,macro,cat_info,hazard_ratios)

    #compute
    return process_input(myCountry,pol_str,macro,cat_info,hazard_ratios,economy,event_level,default_rp,rm_overlap,verbose_replace=True)

def process_input(myCountry,pol_str,macro,cat_info,hazard_ratios,economy,event_level,default_rp,rm_overlap,verbose_replace=True):
    flag1=False
    flag2=False

    if type(hazard_ratios)==pd.DataFrame:
        
        hazard_ratios = hazard_ratios.reset_index().set_index(economy).dropna()
        if 'Unnamed: 0' in hazard_ratios.columns: hazard_ratios = hazard_ratios.drop('Unnamed: 0',axis=1)
        
        #These lines remove countries in macro not in cat_info
        if myCountry == 'SL': hazard_ratios = hazard_ratios.dropna()
        else: hazard_ratios = hazard_ratios.fillna(0)
            
        common_places = [c for c in macro.index if c in cat_info.index and c in hazard_ratios.index]
        #print(common_places)

        hazard_ratios = hazard_ratios.reset_index().set_index(event_level+['hhid'])

        # This drops 1 province from macro
        macro = macro.ix[common_places]

        # Nothing drops from cat_info
        cat_info = cat_info.ix[common_places]

        # Nothing drops from hazard_ratios
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
        hazard_ratios_event = pd.DataFrame()
        for haz in hazard_ratios.reset_index().hazard.unique():
            hazard_ratios_event = hazard_ratios_event.append(interpolate_rps(hazard_ratios.reset_index().ix[hazard_ratios.reset_index().hazard==haz,:].set_index(hazard_ratios.index.names), macro.protection,option=default_rp))
            
        hazard_ratios_event = same_rps_all_hazards(hazard_ratios_event)

    # Now that we have the same set of return periods, remove overlap of losses between PCRAFI and SSBN
    if myCountry == 'FJ' and rm_overlap == True:
        hazard_ratios_event = hazard_ratios_event.reset_index().set_index(['Division','rp','hhid'])
        hazard_ratios_event.loc[hazard_ratios_event.hazard=='flood_fluv_undef','fa'] -= 0.4*hazard_ratios_event.loc[hazard_ratios_event.hazard=='TC','fa']*(hazard_ratios_event.loc[hazard_ratios_event.hazard=='flood_fluv_undef','fa']/(hazard_ratios_event.loc[hazard_ratios_event.hazard=='flood_fluv_undef','fa']+hazard_ratios_event.loc[hazard_ratios_event.hazard=='flood_pluv','fa']))
        hazard_ratios_event.loc[hazard_ratios_event.hazard=='flood_pluv','fa'] -= 0.4*hazard_ratios_event.loc[hazard_ratios_event.hazard=='TC','fa']*(hazard_ratios_event.loc[hazard_ratios_event.hazard=='flood_pluv','fa']/(hazard_ratios_event.loc[hazard_ratios_event.hazard=='flood_fluv_undef','fa']+hazard_ratios_event.loc[hazard_ratios_event.hazard=='flood_pluv','fa']))
        hazard_ratios_event['fa'] = hazard_ratios_event['fa'].clip(lower=0.0)
        
        hazard_ratios_event = hazard_ratios_event.reset_index().set_index(['Division','hazard','rp','hhid'])

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


    ####FORMATTING
    #gets the event level index
    event_level_index = hazard_ratios_event.reset_index().set_index(event_level).index #index composed on countries, hazards and rps.

    #Broadcast macro to event level 
    macro_event = broadcast_simple(macro,event_level_index)	
    #rebuilding exponentially to 95% of initial stock in reconst_duration
    global const_nom_reco_rate
    const_nom_reco_rate = float(np.log(1/0.05) / macro_event['T_rebuild_K'].mean())
    
    global const_pds_rate
    const_pds_rate = const_nom_reco_rate*7.
    
    global const_rho
    const_rho = float(macro_event['rho'].mean())

    global const_ie
    const_ie = float(macro_event['income_elast'].mean())

    #Calculation of macroeconomic resilience
    #macro_event['macro_multiplier'] =(hazard_ratios_event['dy_over_dk'].mean(level=event_level)+const_nom_reco_rate)/(const_rho+const_nom_reco_rate)  #Gamma in the technical paper
    
    #updates columns in macro with columns in hazard_ratios_event
    cols = [c for c in macro_event if c in hazard_ratios_event] #columns that are both in macro_event and hazard_ratios_event
    if not cols==[]:
        if verbose_replace:
            flag1=True
            print('Replaced in macro: '+', '.join(cols))
            macro_event[cols] =  hazard_ratios_event[cols]
    
    #Broadcast categories to event level
    cats_event = broadcast_simple(cat_info,  event_level_index)

    # Need to broadcast 'hh_share' to cats_event
    hazard_ratios_event = hazard_ratios_event.reset_index().set_index(event_level+['hhid'])    
    cats_event = cats_event.reset_index().set_index(event_level+['hhid'])
    cats_event['hh_share'] = hazard_ratios_event['hh_share']
    cats_event['hh_share'] = cats_event['hh_share'].fillna(1.)

    ###############
    # Don't know what this does, except empty the overlapping columns.
    # --> checked before & after adn saw that no columns actually change
    #updates columns in cats with columns in hazard_ratios_event	
    # applies mh ratios to relevant columns
    #cols_c = [c for c in cats_event if c in hazard_ratios_event] #columns that are both in cats_event and hazard_ratios_event    
    cols_c = [c for c in ['fa']] #columns that are both in cats_event and hazard_ratios_event 
    if not cols_c==[]:
        hrb = broadcast_simple(hazard_ratios_event[cols_c], cat_info.index).reset_index().set_index(get_list_of_index_names(cats_event)) 
        # ^ explicitly broadcasts hazard ratios to contain income categories
        cats_event[cols_c] = hrb

    return macro_event, cats_event, hazard_ratios_event 

def compute_dK(pol_str,macro_event,cats_event,event_level,affected_cats,myC,share_public_assets=True):

    #counts affected and non affected
    cats_event_ia=concat_categories(cats_event,cats_event,index= affected_cats)

    # Make sure there are no NaN in fa 
    # --> should be ~0 for provinces with no exposure to a particular disaster
    cats_event['fa'].fillna(value=1.E-8,inplace=True)

    # From here, [hhwgt, pcwgt, and pcwgt_ae] are merged with fa
    # --> print('From here: weights (pc and hh) = nAffected and nNotAffected hh/ind') 
    for aWGT in ['hhwgt','pcwgt','pcwgt_ae']:
        myNaf = cats_event[aWGT]*cats_event.fa
        myNna = cats_event[aWGT]*(1.-cats_event.fa)
        cats_event_ia[aWGT] = concat_categories(myNaf,myNna, index=affected_cats)

    # de_index so can access cats as columns and index is still event
    cats_event_ia = cats_event_ia.reset_index(['hhid', 'affected_cat']).sort_index()
    #cats_event_ia.loc[cats_event_ia.affected_cat== 'a','pcwgt'] = cats_event_ia.loc[cats_event_ia.affected_cat== 'a'].fillna(0.)
    #cats_event_ia.loc[cats_event_ia.affected_cat=='na','pcwgt'] = cats_event_ia.loc[(cats_event_ia.affected_cat=='na'),'pcwgt'].fillna(value=cats_event_ia.loc[cats_event_ia.affected_cat=='na',['hhwgt','hhsize']].prod(axis=1))

    # set vulnerability to zero for non-affected households
    # --> v for [private, public] assets
    cats_event_ia.loc[cats_event_ia.affected_cat=='na',['v']] = 0

    # 'Actual' vulnerability includes migitating effect of early warning systems
    # --> still 0 for non-affected hh
    cats_event_ia['v_shew']=cats_event_ia['v']*(1-macro_event['pi']*cats_event_ia['shew'])    

    ############################
    # Calculate capital losses (public, private, & other) 
    # --> Each household's capital losses is the sum of their private losses and public infrastructure losses
    # --> 'hh_share' recovers fraction that is private property
    
    cats_event_ia['dk_private'] = cats_event_ia[['k','v_shew','hh_share']].prod(axis=1, skipna=False).fillna(0)
    cats_event_ia['dk_public']  = (cats_event_ia[['k','v_shew']].prod(axis=1, skipna=False)*(1-cats_event_ia['hh_share'])).fillna(0)
    cats_event_ia['dk_other']   = 0. 
    #cats_event_ia['dk_public']  = cats_event_ia[['k','public_loss_v']].prod(axis=1, skipna=False)
    # ^ this was FJ; it's buggy -> results in dk > k0
    
    ############################
    # Definition of dk0
    cats_event_ia['dk0'] = cats_event_ia['dk_private'] + cats_event_ia['dk_public'] + cats_event_ia['dk_other'] 

    # Independent of who pays for reconstruction, the total event losses are given by sum of priv, pub, & other:
    macro_event['dk_event'] = ((cats_event_ia['dk0'])*cats_event_ia['pcwgt']).sum(level=event_level)
    # ^ dk_event is WHEN the event happens--doesn't yet include RP/probability

    # --> DEPRECATED: assign losses to each hh
    if not share_public_assets: 
        print('\n** share_public_assets = False --> Infra & public asset costs are assigned to *each hh*\n')
        cats_event_ia['dc'] = (1-macro_event['tau_tax'])*cats_event_ia['dk0'] + cats_event_ia['gamma_SP']*macro_event[['tau_tax','dk_event']].prod(axis=1)
        # ^ 2 terms are dk (hh losses) less tax burden AND reduction to transfers due to losses
        public_costs = None

    # --> distribute losses to every hh
    # TODO: this is going to be the tax rate for all hh, but needs to be broadcast to hh outside of region
    else:
        print('\n** share_public_assets = True --> Sharing infra & public asset costs among all households *nationally*\n')

        # Create a new dataframe for calculation of public fees borne by affected province & all others 
        cats_event_ia = cats_event_ia.reset_index().set_index(event_level+['hhid','affected_cat'])
        rebuild_fees = pd.DataFrame(cats_event_ia[['k','dk_private','dk_public','dk_other','pcwgt']],index=cats_event_ia.index)
       
        cats_event_ia = cats_event_ia.reset_index().set_index(event_level)
        rebuild_fees = rebuild_fees.reset_index().set_index(event_level)

        # Total value of public & private asset losses, when an event hits a single province              
        rebuild_fees['dk_private_tot'] = rebuild_fees[['pcwgt','dk_private']].prod(axis=1).sum(level=event_level)
        rebuild_fees['dk_public_tot']  = rebuild_fees[['pcwgt', 'dk_public']].prod(axis=1).sum(level=event_level)
        rebuild_fees['dk_other_tot']   = rebuild_fees[['pcwgt',  'dk_other']].prod(axis=1).sum(level=event_level)
        rebuild_fees['dk_tot']         = rebuild_fees[['dk_private_tot','dk_public_tot','dk_other_tot']].sum(axis=1)

        #######################################################
        # Now we have dk (private, public, other)
        # Need to calculate each hh's liability for dk_public_tot

        #######################################################
        # Create a temporary dataframe that sums over provinces
        rebuild_fees_tmp = pd.DataFrame(index=cats_event_ia.sum(level=['hazard','rp']).index)
        
        # tot_k_BE is all the assets in the country BEFORE EVENT *** BE = BEFORE EVENT ***
        rebuild_fees_tmp['tot_k_BE'] = cats_event_ia[['pcwgt','k']].prod(axis=1,skipna=False).sum(level=['hazard','rp'])
        # Can't calculate PE, because the _tmp df is already summed over provinces
        # ^ BE is the same for all events in all provinces; PE is not

        # Merge _tmp into original df
        rebuild_fees = pd.merge(rebuild_fees.reset_index(),rebuild_fees_tmp.reset_index(),on=['hazard','rp']).reset_index().set_index(event_level)

        # tot_k_PE is all the assets in the country POST EVENT *** PE = POST EVENT ***
        # --> delta(tot_k_BE , tot_k_PE) will include both public and private and other (?) assets
        # --> that's appropriate because we're imagining tax assessors come after the disaster, and the tax is on income
        rebuild_fees['tot_k_PE'] = rebuild_fees['tot_k_BE'] - (rebuild_fees['dk_private_tot']+rebuild_fees['dk_public_tot']+rebuild_fees['dk_other_tot'])     
        #######################################################
        
        # Prepare 2 dfs for working together again
        cats_event_ia = cats_event_ia.reset_index().set_index(event_level+['hhid','affected_cat'])
        rebuild_fees = rebuild_fees.reset_index().set_index(event_level+['hhid','affected_cat'])

        # Determine the fraction of all capital in the country in each hh (includes weighting here)
        # NB: note the difference between BE and PE here
        rebuild_fees['frac_k_BE'] = cats_event_ia[['pcwgt','k']].prod(axis=1,skipna=False)/rebuild_fees['tot_k_BE']
        rebuild_fees['frac_k_PE'] = cats_event_ia['pcwgt']*(cats_event_ia['k']-cats_event_ia['dk0'])/rebuild_fees['tot_k_PE']
        print('frac_k_BE and _PE are based on K. Need to add social income to this for it to be I.')

        # This is the fraction of damages for which each hh in affected prov will pay
        rebuild_fees['pc_fee_BE'] = (rebuild_fees[['dk_public_tot','frac_k_BE']].prod(axis=1)/rebuild_fees['pcwgt']).fillna(0)
        rebuild_fees['pc_fee_PE'] = (rebuild_fees[['dk_public_tot','frac_k_PE']].prod(axis=1)/rebuild_fees['pcwgt']).fillna(0)
        # --> this is where it goes sideways, unless we have a special approach tho...
        # --> dk_public_tot is for a specific province/hazard/rp, and it needs to be distributed among everyone, nationally
        # --> but it only goes to the hh in the province

        # Now calculate the fee paid by each hh
        # --> still assessing both before and after disaster
        rebuild_fees['hh_fee_BE'] = rebuild_fees[['pc_fee_BE','pcwgt']].prod(axis=1)       
        rebuild_fees['hh_fee_PE'] = rebuild_fees[['pc_fee_PE','pcwgt']].prod(axis=1)  
        
        # Transfer per capita fees back to cats_event_ia 
        cats_event_ia[['pc_fee_BE','pc_fee_PE']] = rebuild_fees[['pc_fee_BE','pc_fee_PE']]
        rebuild_fees['dk0'] = cats_event_ia['dk0']
        cats_event_ia = cats_event_ia.reset_index().set_index(event_level)

        # Sanity Check: we know this works if hh_fee = 'dk_public_hh'*'frac_k'
        #print(rebuild_fees['dk_public_tot'].head(1))
        #print(rebuild_fees[['hh_fee_BE','frac_k_BE']].sum(level=event_level).head(17))
        #print(rebuild_fees[['hh_fee_PE','frac_k_PE']].sum(level=event_level).head(17))
        
        public_costs = distribute_public_costs(macro_event,rebuild_fees,event_level,'dk_public')
                
        ############################
        # Choose whether to assess tax on k (BE='before event') or (PE='post event')
        # --> the total k in non-aff provinces doesn't change, either way
        # --> the fraction of assets in the country does, because of destruction in the aff province. 

        # Uncomment these 2 lines for tax assessment before disaster
        #public_costs = public_costs.rename(columns={'transfer_pub_BE':'transfer_pub','pc_fee_BE':'pc_fee'})
        #cats_event_ia = cats_event_ia.rename(columns={'pc_fee_PE':'pc_fee'})

        # Uncomment these 2 lines for tax assessment after disaster
        public_costs = public_costs.rename(columns={'transfer_pub_PE':'transfer_pub'})
        cats_event_ia = cats_event_ia.rename(columns={'pc_fee_PE':'pc_fee'})

        ############################
        # We can already calculate di0, dc0 for hh in province
        #
        # Define di0 for all households in province where disaster occurred
        cats_event_ia['di0'] = (cats_event_ia['dk0']*macro_event['avg_prod_k'].mean()*(1-macro_event['tau_tax'].mean())
                                + cats_event_ia['pcsoc']*(rebuild_fees['dk_tot']/rebuild_fees['tot_k_BE']).mean(level=event_level))

        # Leaving out all terms without time-dependence
        # EG: + cats_event_ia['pc_fee'] + cats_event_ia['pds']

        # Define dc0 for all households in province where disaster occurred
        cats_event_ia['dc0'] = cats_event_ia['di0'] + const_nom_reco_rate*cats_event_ia['dk_private']

        # Get indexing right, so there are not multiple entries with same index:
        cats_event_ia = cats_event_ia.reset_index().set_index(event_level+['hhid','affected_cat'])

        # Tweak dc0 for hh where (c - dc0) < pov_line:
        cats_event_ia['hh_reco_rate'] = const_nom_reco_rate

        # Classify these hh responses for studying dw
        cats_event_ia['welf_class'] = 0
        # 1 = quick (<3yrs) reco, while avoiding poverty

        if 'sub_line' in cats_event_ia.columns: cats_event_ia['c_min'] = cats_event_ia.sub_line
        elif get_subsistence_line(myC): cats_event_ia['c_min'] = get_subsistence_line(myC)
        else: cats_event_ia['c_min'] = cats_event_ia.c_5
        print('Using hh response: avoid subsistence '+str(round(float(cats_event_ia['c_min'].mean()),2)))

        # Case 1: hh can afford to reconstruct more quickly than 3 years
        # --> income losses do not push hh into poverty 
        # --> standard reco costs do not push into poverty
        # *** HH response: reconstruct as quickly as possible (cap of 0.5 yrs/6 months) while keeping consumption above poverty
        tmp = cats_event_ia.loc[(cats_event_ia.c>cats_event_ia.pov_line)                      # HHs initially above poverty line 
                                &((cats_event_ia.c-cats_event_ia.dc0)>cats_event_ia.pov_line) # AND nominal reconstruction costs don't push them below poverty line 
                                &(cats_event_ia.dk_private!=0)].copy()                        # AND they did lose some assets
        tmp['welf_class']   = 1
        tmp['dc_old']       = tmp['dc0']
        tmp['hh_reco_rate'] = ((tmp['c']-tmp['pov_line']-tmp['di0'])/tmp['dk_private']).clip(upper=6.*const_nom_reco_rate)
        tmp['dc0']          = tmp['di0'] + tmp[['hh_reco_rate','dk_private']].prod(axis=1)
        cats_event_ia.loc[(cats_event_ia.c>=cats_event_ia.pov_line)
                          &((cats_event_ia.c-cats_event_ia.dc0)>cats_event_ia.pov_line)
                          &(cats_event_ia.dk_private!=0),['dc0','hh_reco_rate','welf_class']] = tmp[['dc0','hh_reco_rate','welf_class']]
        print('C0: '+str(round(float(100*cats_event_ia.loc[(cats_event_ia.hh_reco_rate!=const_nom_reco_rate)].shape[0])/float(cats_event_ia.shape[0]),2))+'% of hh have t_reco != 3yr')        

        # Case 2: initial consumption is above or below poverty line
        # --> income losses push below poverty_line, 
        # --> hh can still afford to avoid subsistence
        # *** HH response: keep consumption above subsistence
        tmp = cats_event_ia.loc[((cats_event_ia.c>=cats_event_ia.pov_line)|(cats_event_ia.c<cats_event_ia.pov_line)) # HHs initially above or below poverty line (doesn't matter)
                                #&((cats_event_ia.c-cats_event_ia.di0)<cats_event_ia.pov_line)                         # AND income losses (even before reco) do push them below poverty line
                                &((cats_event_ia.c-cats_event_ia.di0)>=cats_event_ia.c_min)                           # AND they can afford (c-di0) at subsistence (or XX% of poverty line)
                                &(cats_event_ia.welf_class==0)                                                        # AND they're not in any other class
                                &(cats_event_ia.dk_private!=0)].copy()                                                # AND they did lose some assets
        tmp['welf_class']   = 2
        tmp['dc_old']       = tmp['dc0']
        tmp['hh_reco_rate'] = ((tmp['c']-tmp['c_min']-tmp['di0'])/tmp['dk_private']).clip(upper=3.*const_nom_reco_rate)
        tmp['dc0']          = tmp['di0'] + tmp[['hh_reco_rate','dk_private']].prod(axis=1)
        tmp.head(20000).to_csv(debug+'my_tmp_newpov.csv')
        cats_event_ia.loc[((cats_event_ia.c>=cats_event_ia.pov_line)|(cats_event_ia.c<cats_event_ia.pov_line))
                          #&((cats_event_ia.c-cats_event_ia.di0)<cats_event_ia.pov_line)
                          &((cats_event_ia.c-cats_event_ia.di0)>=cats_event_ia.c_min)
                          &(cats_event_ia.welf_class==0)                            
                          &(cats_event_ia.dk_private!=0),['dc0','hh_reco_rate','welf_class']] = tmp[['dc0','hh_reco_rate','welf_class']]
        print('C2: '+str(round(float(100*cats_event_ia.loc[(cats_event_ia.hh_reco_rate!=const_nom_reco_rate)].shape[0])/float(cats_event_ia.shape[0]),2))+'% of hh have t_reco != 3yr')
        
        # Case 3: Post-disaster income is below subsistence (or c_5):
        # HH response: do not reconstruct
        tmp = cats_event_ia.loc[((cats_event_ia.c-cats_event_ia.di0)>0.0)                   # (C-di) does not bankrupt *this can't really evaluate to false, I think...*
                                &((cats_event_ia.c-cats_event_ia.di0)<=cats_event_ia.c_min) # AND (c-di) is below subsistence 
                                &(cats_event_ia.welf_class==0)                              # AND they're not in any other class
                                &(cats_event_ia.dk_private!=0)].copy()                      # AND they did lose some assets
        tmp['welf_class']   = 3
        tmp['dc_old']       = tmp['dc0']
        tmp['hh_reco_rate'] = 0.         # No Reconstruction ((tmp['c']-tmp['di0']-1.)/tmp['dk_private']).clip(upper=const_nom_reco_rate)
        tmp['dc0']          = tmp['di0'] # No Reconstruction tmp['di0'] + tmp[['hh_reco_rate','dk_private']].prod(axis=1)
        tmp.head(20000).to_csv(debug+'my_tmp_pov.csv')
        cats_event_ia.loc[((cats_event_ia.c-cats_event_ia.di0)>0.0)
                          &((cats_event_ia.c-cats_event_ia.di0)<=cats_event_ia.c_min)
                          &(cats_event_ia.welf_class==0) 
                          &(cats_event_ia.dk_private!=0),['dc0','hh_reco_rate','welf_class']] = tmp[['dc0','hh_reco_rate','welf_class']]
        print('C3: '+str(round(float(100*cats_event_ia.loc[(cats_event_ia.hh_reco_rate!=const_nom_reco_rate)].shape[0])/float(cats_event_ia.shape[0]),2))+'% of hh have t_reco != 3yr')

        # make plot here: (income vs. length of reco)

        if cats_event_ia.loc[(cats_event_ia.dc0 > cats_event_ia.c)].shape[0] != 0:
            cats_event_ia.loc[(cats_event_ia.dc0 > cats_event_ia.c)].to_csv(debug+'excess.csv')
            hh_extinction = str(round(float(cats_event_ia.loc[(cats_event_ia.dc0 > cats_event_ia.c)].shape[0]/cats_event_ia.shape[0])*100.,2))
            print('\n\n***ISSUE: '+hh_extinction+'% of (hh x event) combos face dc0 > i0. Could mean extinction!\n--> SOLUTION: capping dw at 20xGDPpc \n\n')
        
        # NOTES
        # --> when affected by a disaster, hh lose their private assets...
        # --> *AND* the fraction of the public assets for which they're responsible
        # So far, the losses don't include pc_fee (public reco) or PDS
        # --> pc_fee is only for people (aff & non-aff) in the affected province
        # Question: if we charge the pc_fee on post-disaster assets, a & na versions of each hh pay different amounts. need to make sure the math adds up 
        ############################
        cats_event_ia = cats_event_ia.reset_index().set_index(event_level).drop(['c_min'],axis=1)
        
    return macro_event, cats_event_ia, public_costs

def distribute_public_costs(macro_event,rebuild_fees,event_level,transfer_type):
    
    ############################        
    # Make another output file --> public_costs.csv
    # --> this contains the cost to each province/region of each disaster (hazard x rp) in another province
    public_costs = pd.DataFrame(index=macro_event.index)

    # Total capital in each province
    public_costs['tot_k_recipient_BE'] = rebuild_fees[['pcwgt','k']].prod(axis=1).sum(level=event_level) 
    public_costs['tot_k_recipient_PE'] = (rebuild_fees['pcwgt']*(rebuild_fees['k']-rebuild_fees['dk0'])).sum(level=event_level)
        
    # Total public losses from each event        
    public_costs['dk_public_recipient']    = rebuild_fees[['pcwgt',transfer_type]].prod(axis=1).sum(level=event_level).fillna(0)
    
    # Cost to province where disaster occurred of rebuilding public assets
    rebuild_fees[transfer_type+'_hh'] = rebuild_fees[['pcwgt',transfer_type]].prod(axis=1)
    # ^ gives total public losses suffered by all people represented by each hh
    public_costs['int_cost_BE'] = (rebuild_fees[[transfer_type+'_hh','frac_k_BE']].sum(level=event_level)).prod(axis=1)
        
    # Total cost to ALL provinces where disaster did not occur rebuilding public assets
    public_costs['ext_cost_BE'] = public_costs['dk_public_recipient'] - public_costs['int_cost_BE']
    public_costs['tmp'] = 1

    # Grab a list of names of all regions/provinces
    prov_k = pd.DataFrame(index=rebuild_fees.sum(level=event_level[0]).index)
    prov_k.index.names = ['contributer']
    prov_k['frac_k_BE'] = rebuild_fees['frac_k_BE'].sum(level=event_level).mean(level=event_level[0])/rebuild_fees['frac_k_BE'].sum(level=event_level).mean(level=event_level[0],skipna=True).sum()
    prov_k['tot_k_contributer'] = rebuild_fees[['pcwgt','k']].prod(axis=1).sum(level=event_level).mean(level=event_level[0])
    # Can't define frac_k_PE here: _tmp does not operate at event level
    prov_k['tmp'] = 1
    prov_k = prov_k.reset_index()
        
    public_costs = pd.merge(public_costs.reset_index(),prov_k.reset_index(),on=['tmp']).reset_index().set_index(event_level+['contributer']).sort_index()
    public_costs = public_costs.drop(['index','level_0','tmp'],axis=1)
    # ^ broadcast prov index to public_costs (2nd provincial index)
    
    public_costs = public_costs.reset_index()
    public_costs['prov_assets_PE'] = 0
    public_costs.loc[(public_costs.contributer==public_costs[event_level[0]]),'prov_assets_PE'] = public_costs.loc[(public_costs.contributer==public_costs[event_level[0]]),'tot_k_recipient_PE']
    public_costs.loc[(public_costs.contributer!=public_costs[event_level[0]]),'prov_assets_PE'] = public_costs.loc[(public_costs.contributer!=public_costs[event_level[0]]),'tot_k_contributer']
    
    public_costs = public_costs.reset_index().set_index(event_level).drop('index',axis=1)
    
    public_costs['frac_k_PE'] = public_costs['prov_assets_PE']/public_costs['prov_assets_PE'].sum(level=event_level)
    
    public_costs['transfer_pub_BE'] = public_costs[['dk_public_recipient','frac_k_BE']].prod(axis=1)
    public_costs['transfer_pub_PE'] = public_costs[['dk_public_recipient','frac_k_PE']].prod(axis=1)
    public_costs['PE_to_BE'] = public_costs['transfer_pub_PE']/public_costs['transfer_pub_BE']
    
    #public_costs = public_costs.drop([i for i in public_costs.columns if i not in ['contributer','dk_public_recipient','frac_k_BE','frac_k_PE',
    #                                                                               'transfer_pub_BE','transfer_pub_PE','PE_to_BE']],axis=1)
    
    public_costs['dw'] = None
    public_costs.to_csv(debug+'public_costs.csv')

    return public_costs

def calc_dw_outside_affected_province(macro_event, cat_info, public_costs_pub, public_costs_pds, event_level, is_contemporaneous=False, is_local_welfare=False, is_revised_dw=True):

    public_costs = pd.DataFrame(index=public_costs_pub.index)
    public_costs[['contributer','transfer_pub','tot_k_recipient_BE','tot_k_recipient_PE','tot_k_contributer']] \
        = public_costs_pub[['contributer','transfer_pub','tot_k_recipient_BE','tot_k_recipient_PE','tot_k_contributer']]
    public_costs['transfer_pds'] = public_costs_pds['transfer_pds']

    ############################
    # So we have cost of each disaster in each province to every other province
    # - need to calc welfare impact of these transfers
    public_costs = public_costs.reset_index()
    cat_info = cat_info.reset_index().set_index(['hhid'])

    for iP in public_costs.contributer.unique():
        print('Running revised, non-contemp dw for hh outside '+iP) 
        
        tmp_df = cat_info.loc[(cat_info[event_level[0]]==iP)].copy()

        tmp_df['hh_frac_k'] = tmp_df[['pcwgt','k']].prod(axis=1)/(tmp_df[['pcwgt','k']].prod(axis=1)).sum()
        tmp_df['pc_frac_k'] = tmp_df['hh_frac_k']/tmp_df['pcwgt']
        # ^ this grabs a single instance of each hh in each contributing (non-aff) province
        # --> 'k' and 'c' are not distributed between {a,na} (use mean), but pcwgt is (use sum)
        # --> 'pc_frac_k' used to determine what they'll pay when a disaster happens elsewhere
        # Only using capital--not income...
        tmp_t_reco     = float(macro_event['T_rebuild_K'].mean())
        c_mean         = float(cat_info[['pcwgt','c']].prod(axis=1).sum()/cat_info['pcwgt'].sum())
        h = 1.E-4

        wprime = c_mean**(-const_ie)
        # ^ these *could* vary by province/event, but don't (for now), so I'll use them outside the pandas dfs.
            
        for iRecip in public_costs[event_level[0]].unique():
            for iHaz in public_costs.hazard.unique():
                for iRP in public_costs.rp.unique():

                    # Calculate wprime
                    tmp_wp = None
                    if is_local_welfare:
                        tmp_gdp = macro_event.loc[(macro_event[event_level[0]]==iP),'gdp_pc_pp_prov'].mean()
                        tmp_wp =(welf(tmp_gdp/const_rho+h,const_ie)-welf(tmp_gdp/const_rho-h,const_ie))/(2*h)
                    else: 
                        tmp_wp =(welf(macro_event['gdp_pc_pp_nat'].mean()/const_rho+h,const_ie)-welf(macro_event['gdp_pc_pp_nat'].mean()/const_rho-h,const_ie))/(2*h)

                    tmp_cost_pub = float(public_costs.loc[((public_costs[event_level[0]]==iRecip)&(public_costs.contributer == iP)
                                                           &(public_costs.hazard == iHaz)&(public_costs.rp==iRP)),'transfer_pub'])
                    tmp_cost_pds = float(public_costs.loc[((public_costs[event_level[0]]==iRecip)&(public_costs.contributer == iP)
                                                           &(public_costs.hazard == iHaz)&(public_costs.rp==iRP)),'transfer_pds'])
                    # ^ this identifies the amount that a province (iP, above) will contribute to another province when a disaster occurs

                    if is_contemporaneous or not is_revised_dw: 
                        tmp_df['dc_per_cap'] = (tmp_cost_pub+tmp_cost_pds)*tmp_df['pc_frac_k']

                        if not is_revised_dw:
                            tmp_df['dw'] = tmp_df['pcwgt']*(welf1(tmp_df['c']/const_rho, const_ie, tmp_df['c_5']/const_rho)
                                                            - welf1((tmp_df['c']-tmp_df['dc_per_cap'])/const_rho, const_ie,tmp_df['c_5']/const_rho))/tmp_wp
                            # ^ returns NPV

                        else:
                            tmp_df['dw'] += tmp_df['pcwgt']*(welf1(tmp_df['c'], const_ie, tmp_df['c_5']) - welf1((tmp_df['c']-tmp_df['dc_per_cap']), const_ie,tmp_df['c_5']))/wprime
                       
                    else:                        
                        # Here, we calculate the impact of transfers for public assets & PDS
                        tmp_df['dw']     = 0.
                        for iT in [tmp_cost_pub,tmp_cost_pds]:
                            tmp_df['dw'] += tmp_df['pcwgt']*(welf1(tmp_df['c'], const_ie, tmp_df['c_5']) - welf1((tmp_df['c']-iT*tmp_df['pc_frac_k']), const_ie,tmp_df['c_5']))
                            
                        frac_dk_natl = float((public_costs.loc[((public_costs[event_level[0]]==iRecip)&(public_costs.hazard==iHaz)&(public_costs.rp==iRP)
                                                                &(public_costs.contributer == iP)),'tot_k_recipient_BE']-
                                              public_costs.loc[((public_costs[event_level[0]]==iRecip)&(public_costs.hazard==iHaz)&(public_costs.rp==iRP)
                                                                &(public_costs.contributer == iP)),'tot_k_recipient_PE'])/
                                             (public_costs.loc[(public_costs[event_level[0]]==iRecip)&(public_costs.hazard == iHaz)&(public_costs.rp==iRP),'tot_k_contributer']).sum())
                        
                        # Also need to add impacts of social transfer decreases
                        # Set-up to be able to calculate integral
                        tmp_df['const'] = -1.*tmp_df['c']**(1.-const_ie)/(1.-const_ie)
                        tmp_df['integ'] = 0.

                        x_min, x_max, n_steps = 0.5,tmp_t_reco+0.5,12.
                        int_dt,step_dt = np.linspace(x_min,x_max,num=n_steps,endpoint=True,retstep=True)
                        # ^ make sure that, if T_recon changes, so does x_max!

                        # Calculate integral
                        for i_dt in int_dt:
                            # need total asset losses --> tax base reduction
                            tmp_df['integ'] += step_dt*((1.-frac_dk_natl*tmp_df['social']*math.e**(-i_dt*const_nom_reco_rate))**(1-const_ie)-1)*math.e**(-i_dt*const_rho)
                            # ^ const_nom_reco_rate doesn't get replaced by hh-dep values here, because this is aggregate reco
                            
                        # put it all together, including w_prime:
                        tmp_df['dw_soc'] = tmp_df[['pcwgt','const','integ']].prod(axis=1)

                    public_costs.loc[((public_costs[event_level[0]]==iRecip)&(public_costs.contributer==iP)&(public_costs.hazard==iHaz)&(public_costs.rp==iRP)),'dw'] = \
                        abs(tmp_df['dw'].sum()/wprime)
                    public_costs.loc[((public_costs[event_level[0]]==iRecip)&(public_costs.contributer==iP)&(public_costs.hazard==iHaz)&(public_costs.rp==iRP)),'dw_soc'] = \
                        abs(tmp_df['dw_soc'].sum()/wprime)

    public_costs = public_costs.reset_index().set_index(event_level).drop('index',axis=1)

    # NPV consumption losses accounting for reconstruction and productivity of capital (pre-response)
    #cat_info['dc_npv_pre'] = cat_info['dc0']*macro_event['macro_multiplier']
    
    return public_costs


# We already have the time-dependent parts of dk, di, dc
# 'calculate_response' is a separate function because it bifurcates each hh again between {helped X not helped}
def calculate_response(myCountry,pol_str,macro_event,cats_event_ia,public_costs,event_level,helped_cats,default_rp,option_CB,
                       optionFee='tax',optionT='data', optionPDS='unif_poor', optionB='data',loss_measure='dk_private',fraction_inside=1, share_insured=.25):

    cats_event_iah = broadcast_simple(cats_event_ia, helped_cats).reset_index().set_index(event_level)

    # Baseline case (no insurance):
    cats_event_iah['help_received'] = 0
    cats_event_iah['help_fee'] =0

    macro_event, cats_event_iah, public_costs = compute_response(myCountry, pol_str,macro_event, cats_event_iah, public_costs, event_level,default_rp,option_CB,optionT=optionT, 
                                                                 optionPDS=optionPDS, optionB=optionB, optionFee=optionFee, fraction_inside=fraction_inside, loss_measure = loss_measure)
    
    cats_event_iah.drop(['n','protection'],axis=1, inplace=True)	      
    print('T3:'+str(round(float(100*cats_event_iah.loc[(cats_event_iah['hh_reco_rate']!=const_nom_reco_rate)].shape[0])/float(cats_event_iah.shape[0]),2))+'% of hh have t_reco != 3yr')
   
    return macro_event, cats_event_iah, public_costs
	
def compute_response(myCountry, pol_str, macro_event, cats_event_iah,public_costs, event_level, default_rp, option_CB,optionT='data', 
                     optionPDS='unif_poor', optionB='data', optionFee='tax', fraction_inside=1, loss_measure='dk_private'):    

    print('NB: when summing over cats_event_iah, be aware that each hh appears 4X in the file: {a,na}x{helped,not_helped}')
    # This function computes aid received, aid fee, and other stuff, 
    # --> including losses and PDS options on targeting, financing, and dimensioning of the help.
    # --> Returns copies of macro_event and cats_event_iah updated with stuff

    macro_event    = macro_event.copy()
    cats_event_iah = cats_event_iah.copy()

    macro_event['fa'] = (cats_event_iah.loc[(cats_event_iah.affected_cat=='a'),'pcwgt'].sum(level=event_level)/(cats_event_iah['pcwgt'].sum(level=event_level))).fillna(1E-8)
    # No factor of 2 in denominator affected households are counted twice in both the num & denom
    # --> at this point, each appears twice (helped & not_helped)
    # --> the weights haven't been adjusted to include targetng errors

    ####targeting errors
    if optionPDS == 'fiji_SPS' or optionPDS == 'fiji_SPP':
        macro_event['error_incl'] = 1.0
        macro_event['error_excl'] = 0.0
    elif optionT=='perfect':
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
            
    #counting (mind self multiplication of n)
    df_index = cats_event_iah.index.names    

    cats_event_iah = pd.merge(cats_event_iah.reset_index(),macro_event.reset_index()[[i for i in macro_event.index.names]+['error_excl','error_incl']],on=[i for i in macro_event.index.names])

    for aWGT in ['hhwgt','pcwgt','pcwgt_ae']:
        cats_event_iah.loc[(cats_event_iah.helped_cat=='helped')    & (cats_event_iah.affected_cat=='a') ,aWGT]*=(1-cats_event_iah['error_excl'])
        cats_event_iah.loc[(cats_event_iah.helped_cat=='not_helped')& (cats_event_iah.affected_cat=='a') ,aWGT]*=(  cats_event_iah['error_excl'])
        cats_event_iah.loc[(cats_event_iah.helped_cat=='helped')    & (cats_event_iah.affected_cat=='na'),aWGT]*=(  cats_event_iah['error_incl'])  
        cats_event_iah.loc[(cats_event_iah.helped_cat=='not_helped')& (cats_event_iah.affected_cat=='na'),aWGT]*=(1-cats_event_iah['error_incl'])

    cats_event_iah = cats_event_iah.reset_index().set_index(df_index).drop([icol for icol in ['index','error_excl','error_incl'] if icol in cats_event_iah.columns],axis=1)
    cats_event_iah = cats_event_iah.drop(['index'],axis=1)
    
    # MAXIMUM NATIONAL SPENDING ON SCALE UP
    macro_event['max_increased_spending'] = 0.05

    # max_aid is per cap, and it is constant for all disasters & provinces
    # --> If this much were distributed to everyone in the country, it would be 5% of GDP
    macro_event['max_aid'] = macro_event['max_increased_spending'].mean()*macro_event[['gdp_pc_pp_prov','pop']].prod(axis=1).sum(level=['hazard','rp']).mean()/macro_event['pop'].sum(level=['hazard','rp']).mean()

    if optionFee == 'insurance_premium':
        temp = cats_event_iah.copy()
	
    if optionPDS=='no':
        macro_event['aid'] = 0
        cats_event_iah['help_received']=0
        optionB='no'

    if optionPDS == 'fiji_SPP':
        
        sp_payout = pd.DataFrame(index=macro_event.sum(level='rp').index)
        sp_payout = sp_payout.reset_index()

        sp_payout['monthly_allow'] = 177#FJD
        
        sp_payout['frac_core'] = 0.0
        #sp_payout.loc[sp_payout.rp >= 20,'frac_core'] = 1.0
        sp_payout.loc[sp_payout.rp >= 10,'frac_core'] = 1.0        

        sp_payout['frac_add'] = 0.0
        #sp_payout.loc[(sp_payout.rp >=  5)&(sp_payout.rp < 10),'frac_add'] = 0.25
        sp_payout.loc[(sp_payout.rp >= 10)&(sp_payout.rp < 20),'frac_add'] = 0.50
        sp_payout.loc[(sp_payout.rp >= 20)&(sp_payout.rp < 40),'frac_add'] = 0.75
        sp_payout.loc[(sp_payout.rp >= 40),'frac_add'] = 1.00

        sp_payout['multiplier'] = 1.0
        sp_payout.loc[(sp_payout.rp >=  40)&(sp_payout.rp <  50),'multiplier'] = 2.0
        sp_payout.loc[(sp_payout.rp >=  50)&(sp_payout.rp < 100),'multiplier'] = 3.0
        sp_payout.loc[(sp_payout.rp >= 100),'multiplier'] = 4.0

        #sp_payout['payout_core'] = sp_payout[['monthly_allow','frac_core','multiplier']].prod(axis=1)
        #sp_payout['payout_add']  = sp_payout[['monthly_allow', 'frac_add','multiplier']].prod(axis=1)
        # ^ comment out these lines when uncommenting below
        sp_payout['payout'] = sp_payout[['monthly_allow','multiplier']].prod(axis=1)
        # ^ this line is for when we randomly choose people in each group

        sp_payout.to_csv('../output_country/FJ/SPP_details.csv')
 
        cats_event_iah = pd.merge(cats_event_iah.reset_index(),sp_payout[['rp','payout','frac_core','frac_add']].reset_index(),on=['rp'])
        cats_event_iah = cats_event_iah.reset_index().set_index(['Division','hazard','hhid','rp','affected_cat','helped_cat'])
        cats_event_iah = cats_event_iah.drop([i for i in ['level_0'] if i in cats_event_iah.columns],axis=1)

        # Generate random numbers to determine payouts
        cats_event_iah['SP_lottery'] = np.random.uniform(0.0,1.0,cats_event_iah.shape[0])

        # Calculate payouts for core: 100% payout for RP >= 20
        cats_event_iah.loc[(cats_event_iah.SPP_core == True)&(cats_event_iah.SP_lottery<cats_event_iah.frac_core),'help_received'] = cats_event_iah.loc[(cats_event_iah.SPP_core == True)&(cats_event_iah.SP_lottery<cats_event_iah.frac_core),'payout']/cats_event_iah.loc[(cats_event_iah.SPP_core == True)&(cats_event_iah.SP_lottery<cats_event_iah.frac_core),'hhsize']
        
        # Calculate payouts for additional: variable payout based on lottery
        cats_event_iah.loc[(cats_event_iah.SPP_add==True)&(cats_event_iah.SP_lottery<cats_event_iah.frac_add),'SP_lottery_win'] = True
        cats_event_iah['SP_lottery_win'] = cats_event_iah['SP_lottery_win'].fillna(False)
        
        cats_event_iah.loc[(cats_event_iah.SP_lottery_win==True),'help_received'] = cats_event_iah.loc[(cats_event_iah.SP_lottery_win==True),'payout']/cats_event_iah.loc[(cats_event_iah.SP_lottery<cats_event_iah.frac_core),'hhsize']

        cats_event_iah = cats_event_iah.reset_index().set_index(['Division','hazard','rp','hhid']).sortlevel()
        # ^ Take helped_cat and affected_cat out of index. Need to slice on helped_cat, and the rest of the code doesn't want hhtypes in index
  
        cats_event_iah.loc[(cats_event_iah.helped_cat=='not_helped'),'help_received'] = 0
        cats_event_iah = cats_event_iah.drop([i for i in ['level_0','SPP_core','SPP_add','payout','frac_core','frac_add','SP_lottery','SP_lottery_win'] if i in cats_event_iah.columns],axis=1)

        my_out = cats_event_iah[['help_received','pcwgt']].prod(axis=1).sum(level=['hazard','rp'])
        my_out.to_csv('../output_country/FJ/SPplus_expenditure.csv')
        my_out,_ = average_over_rp(my_out.sum(level=['rp']),default_rp)
        my_out.sum().to_csv('../output_country/FJ/SPplus_expenditure_annual.csv')

    if optionPDS=='fiji_SPS':

        sp_payout = macro_event['dk_event'].copy()
        sp_payout = sp_payout.sum(level=['hazard','rp'])
        sp_payout = sp_payout.reset_index().set_index(['hazard'])

        sp_200yr = sp_payout.loc[sp_payout.rp==200,'dk_event']
        
        sp_payout = pd.concat([sp_payout,sp_200yr],axis=1,join='inner')
        sp_payout.columns = ['rp','dk_event','benchmark_losses']

        sp_payout['f_benchmark'] = (sp_payout['dk_event']/sp_payout['benchmark_losses']).clip(lower=0.0,upper=1.0)
        sp_payout.loc[(sp_payout.rp < 10),'f_benchmark'] = 0.0
        sp_payout = sp_payout.drop(['dk_event','benchmark_losses'],axis=1)
        sp_payout = sp_payout.reset_index().set_index(['hazard','rp'])
        sp_payout.to_csv('../output_country/FJ/SPS_details.csv')

        cats_event_iah = pd.merge(cats_event_iah.reset_index(),sp_payout.reset_index(),on=['hazard','rp'])
        cats_event_iah = cats_event_iah.reset_index().set_index(['Division','hazard','rp','hhid']).sortlevel()

        # paying out per cap
        cats_event_iah['help_received'] = 0

        print('\nTotal N Households:',cats_event_iah['hhwgt'].sum(level=['hazard','rp']).mean())
        print('\nSPS enrollment:',cats_event_iah.loc[(cats_event_iah.SP_SPS==True),['nOlds','hhwgt']].prod(axis=1).sum(level=['hazard','rp']).mean())
        print('\nCPP enrollment:',cats_event_iah.loc[(cats_event_iah.SP_CPP==True),'hhwgt'].sum(level=['hazard','rp']).mean())
        print('\nFAP enrollment:',cats_event_iah.loc[(cats_event_iah.SP_FAP==True),'hhwgt'].sum(level=['hazard','rp']).mean())
        print('\nPBS enrollment:',cats_event_iah.loc[(cats_event_iah.SP_PBS==True),'hhwgt'].sum(level=['hazard','rp']).mean())

        print('\nMax SPS expenditure =',(cats_event_iah.loc[(cats_event_iah.SP_SPS==True),['hhwgt','nOlds']].prod(axis=1).sum()*300+
                                         cats_event_iah.loc[(cats_event_iah.SP_CPP==True),'hhwgt'].sum()*300+
                                         cats_event_iah.loc[(cats_event_iah.SP_PBS==True),'hhwgt'].sum()*600)/(17*4),'\n')
        
        cats_event_iah.loc[(cats_event_iah.SP_SPS==True),'help_received']+=300*(cats_event_iah.loc[(cats_event_iah.SP_SPS==True),['hhwgt','nOlds','f_benchmark']].prod(axis=1)/
                                                                                cats_event_iah.loc[(cats_event_iah.SP_SPS==True),'pcwgt']).fillna(0)
        cats_event_iah.loc[(cats_event_iah.SP_CPP==True),'help_received']+=300*(cats_event_iah.loc[(cats_event_iah.SP_CPP==True),['hhwgt','f_benchmark']].prod(axis=1)/
                                                                                cats_event_iah.loc[(cats_event_iah.SP_CPP==True),'pcwgt']).fillna(0)
        cats_event_iah.loc[(cats_event_iah.SP_PBS==True),'help_received']+=600*(cats_event_iah.loc[(cats_event_iah.SP_PBS==True),['hhwgt','f_benchmark']].prod(axis=1)/
                                                                                cats_event_iah.loc[(cats_event_iah.SP_PBS==True),'pcwgt']).fillna(0)
        cats_event_iah.loc[(cats_event_iah.helped_cat=='not_helped'),'help_received'] = 0
        my_out = cats_event_iah[['help_received','pcwgt']].prod(axis=1).sum(level=['hazard','rp'])
        my_out.to_csv('../output_country/FJ/SPS_expenditure.csv')
        my_out,_ = average_over_rp(my_out.sum(level=['rp']),default_rp)
        my_out.sum().to_csv('../output_country/FJ/SPS_expenditure_annual.csv')

        cats_event_iah = cats_event_iah.drop(['f_benchmark'],axis=1)

    elif optionPDS=='unif_poor':

        cats_event_iah['help_received'] = 0        

        # For this policy:
        # --> help_received = 0.8*average losses of lowest quintile (households)
        cats_event_iah.loc[(cats_event_iah.helped_cat=='helped')&(cats_event_iah.affected_cat=='a'),'help_received'] = macro_event['shareable']*cats_event_iah.loc[(cats_event_iah.affected_cat=='a')&(cats_event_iah.quintile==1),[loss_measure,'pcwgt']].prod(axis=1).sum(level=event_level)/cats_event_iah.loc[(cats_event_iah.affected_cat=='a')&(cats_event_iah.quintile==1),'pcwgt'].sum(level=event_level)

        # These should be zero here, but just to make sure...
        # Is this right? Does help_received = 0 for na_hh when there is a targeting error?
        cats_event_iah.ix[(cats_event_iah.helped_cat=='not_helped'),'help_received']=0
        cats_event_iah.ix[(cats_event_iah.affected_cat=='na'),'help_received']=0
        # Could test with these:
        #assert(cats_event_iah.ix[(cats_event_iah.helped_cat=='not_helped'),'help_received'].sum()==0)
        #assert(cats_event_iah.ix[(cats_event_iah.affected_cat=='na'),'help_received'].sum()==0)


    elif optionPDS=='unif_poor_only':
        cats_event_iah.ix[(cats_event_iah.helped_cat=='helped')&(cats_event_iah.affected_cat=='a')&(cats_event_iah.quintile==1),'help_received']=macro_event['shareable']*cats_event_iah.loc[(cats_event_iah.affected_cat=='a')&(cats_event_iah.quintile==1),[loss_measure,'pcwgt']].prod(axis=1).sum(level=event_level)/cats_event_iah.loc[(cats_event_iah.affected_cat=='a')&(cats_event_iah.quintile==1),'pcwgt'].sum(level=event_level)

        # These should be zero here, but just to make sure...
        cats_event_iah.ix[(cats_event_iah.helped_cat=='not_helped')|(cats_event_iah.quintile > 1),'help_received']=0
        #cats_event_iah.ix[(cats_event_iah.affected_cat=='na')|(cats_event_iah.quintile > 1),'help_received']=0
        # ^ not necessarily zero, if there is targeting error...
        print('Calculating loss measure\n')

    elif optionPDS=='prop':
        if not 'has_received_help_from_PDS_cat' in cats_event_iah.columns:
            cats_event_iah.ix[(cats_event_iah.helped_cat=='helped')& (cats_event_iah.affected_cat=='a'),'help_received']= macro_event['shareable']*cats_event_iah.ix[(cats_event_iah.helped_cat=='helped')& (cats_event_iah.affected_cat=='a'),loss_measure]
            cats_event_iah.ix[(cats_event_iah.helped_cat=='helped')& (cats_event_iah.affected_cat=='na'),'help_received']= macro_event['shareable']*cats_event_iah.ix[(cats_event_iah.helped_cat=='helped')& (cats_event_iah.affected_cat=='a'),loss_measure]
            cats_event_iah.ix[cats_event_iah.helped_cat=='not_helped','help_received']=0		

        else:
            cats_event_iah.ix[(cats_event_iah.helped_cat=='helped')& (cats_event_iah.affected_cat=='a')  & (cats_event_iah.has_received_help_from_PDS_cat=='helped'),'help_received']= macro_event['shareable']*cats_event_iah.ix[(cats_event_iah.helped_cat=='helped')& (cats_event_iah.affected_cat=='a')  & (cats_event_iah.has_received_help_from_PDS_cat=='helped'),loss_measure]
            cats_event_iah.ix[(cats_event_iah.helped_cat=='helped')& (cats_event_iah.affected_cat=='na')  & (cats_event_iah.has_received_help_from_PDS_cat=='helped'),'help_received']= macro_event['shareable']*cats_event_iah.ix[(cats_event_iah.helped_cat=='helped')& (cats_event_iah.affected_cat=='a')  & (cats_event_iah.has_received_help_from_PDS_cat=='helped'),loss_measure]			
            cats_event_iah.ix[(cats_event_iah.helped_cat=='helped')& (cats_event_iah.affected_cat=='a')  & (cats_event_iah.has_received_help_from_PDS_cat=='not_helped'),'help_received']= macro_event['shareable']*cats_event_iah.ix[(cats_event_iah.helped_cat=='helped')& (cats_event_iah.affected_cat=='a')  & (cats_event_iah.has_received_help_from_PDS_cat=='helped'),loss_measure]
            cats_event_iah.ix[(cats_event_iah.helped_cat=='helped')& (cats_event_iah.affected_cat=='na')  & (cats_event_iah.has_received_help_from_PDS_cat=='not_helped'),'help_received']= macro_event['shareable']*cats_event_iah.ix[(cats_event_iah.helped_cat=='helped')& (cats_event_iah.affected_cat=='a')  & (cats_event_iah.has_received_help_from_PDS_cat=='helped'),loss_measure]			
            cats_event_iah.ix[cats_event_iah.helped_cat=='not_helped','help_received']=0
		
    #actual aid reduced by capacity
    # ^ Not implemented for now
		
    #######################################
    # Above code calculated the benefits
    # now need to determine the costs

    # Already did all the work for this in determining dk_public
    public_costs = public_costs.drop([i for i in public_costs.columns if i not in ['contributer','frac_k_BE','frac_k_PE']], axis=1)
    public_costs['pds_cost'] = cats_event_iah[['help_received','pcwgt']].prod(axis=1).sum(level=event_level)
    public_costs['pds_cost'] = public_costs['pds_cost'].fillna(0)

    public_costs['transfer_pds_BE'] = public_costs[['pds_cost','frac_k_BE']].prod(axis=1)
    public_costs['transfer_pds_PE'] = public_costs[['pds_cost','frac_k_PE']].prod(axis=1)

    # Uncomment this line for tax assessment before disaster
    #public_costs = public_costs.rename(columns={'transfer_pds_BE':'transfer_pds','pc_fee_BE':'pc_fee'})
    # Uncomment this line for tax assessment after disaster
    public_costs = public_costs.rename(columns={'transfer_pds_PE':'transfer_pds'})

    # this will do for hh in the affected province
    if optionFee=='tax':

        # Original code:
        #cats_event_iah['help_fee'] = fraction_inside*macro_event['aid']*cats_event_iah['k']/agg_to_event_level(cats_event_iah,'k',event_level)
        # ^ this only manages transfers within each province 

        cats_event_iah['help_fee'] = 0
        if optionPDS == 'fiji_SPS' or optionPDS == 'fiji_SPP':
            
            cats_event_iah = pd.merge(cats_event_iah.reset_index(),(cats_event_iah[['help_received','pcwgt']].prod(axis=1).sum(level=['hazard','rp'])).reset_index(),on=['hazard','rp'])
            cats_event_iah = cats_event_iah.reset_index().set_index(event_level)
            cats_event_iah = cats_event_iah.rename(columns={0:'totex'}).drop(['index','level_0'],axis=1)
            ## ^ total expenditure

            cats_event_iah = pd.merge(cats_event_iah.reset_index(),(cats_event_iah[['k','pcwgt']].prod(axis=1).sum(level=['hazard','rp'])).reset_index(),on=['hazard','rp'])
            cats_event_iah = cats_event_iah.reset_index().set_index(event_level)
            cats_event_iah = cats_event_iah.rename(columns={0:'totk'})          
            
            cats_event_iah['help_fee'] = (cats_event_iah[['totex','k','pcwgt']].prod(axis=1)/cats_event_iah['totk'])/cats_event_iah['pcwgt']
            # ^ could eliminate the two instances of 'pcwgt', but I'm leaving them in to make clear to future me how this was constructed

            cats_event_iah = cats_event_iah.reset_index().set_index(event_level)
            cats_event_iah = cats_event_iah.drop(['index','totex','totk','nOlds','SP_CPP','SP_FAP','SP_FNPF','SP_SPS','SP_PBS'],axis=1)

            # Something like this should evaluate to true
            #assert((cats_event_iah[['pcwgt','help_received']].prod(axis=1).sum(level=['hazard','rp'])).ix['TC'] == 
            #       (cats_event_iah[['pcwgt','help_fee']].prod(axis=1).sum(level=['hazard','rp'])).ix['TC'])

        else:
            cats_event_iah.loc[cats_event_iah.pcwgt != 0,'help_fee'] = (cats_event_iah.loc[cats_event_iah.pcwgt != 0,['help_received','pcwgt']].prod(axis=1).sum(level=event_level) * 
                                                                        # ^ total expenditure
                                                                        (cats_event_iah.loc[cats_event_iah.pcwgt != 0,['k','pcwgt']].prod(axis=1) /
                                                                         cats_event_iah.loc[cats_event_iah.pcwgt != 0,['k','pcwgt']].prod(axis=1).sum(level=event_level)) /
                                                                        # ^ weighted average of capital
                                                                        cats_event_iah.loc[cats_event_iah.pcwgt != 0,'pcwgt']) 
                                                                        # ^ help_fee is per individual!

    elif optionFee=='insurance_premium':
        print('Have not implemented optionFee==insurance_premium')
        assert(False)

    return macro_event, cats_event_iah, public_costs


def calc_dw_inside_affected_province(myCountry,pol_str,macro_event,cats_event_iah,event_level,option_CB,return_stats=True,return_iah=True,is_revised_dw=True):

    cats_event_iah = cats_event_iah.reset_index().set_index(event_level+['hhid','affected_cat','helped_cat'])

    # These terms contribute to dc in the affected province:
    # 1) dk0 -> di0 -> dc0
    # 2) dc_reco (private only)
    # 3) PDS receipts
    # 4) New tax burden
    # 5)*Soc. transfer reductions
    # 6)*Public asset reco fees
    # 7)*PDS fees

    # These terms contribute to dc outside the affected province: 
    # 5)*Soc. transfer reductions
    # 6)*Public asset reco fees
    # 7)*PDS fees
  
    # *These are calculated in calc_dw_outside_affected_province():

    #cats_event_iah['dc_npv_post'] = cats_event_iah['dc_npv_pre']-cats_event_iah['help_received']+cats_event_iah['help_fee']*option_CB
    if is_revised_dw:
        print('changing dc to include help_received and help_fee, since instantaneous loss is used instead of npv for dw')
        print('how does timing affect the appropriateness of this?')
        cats_event_iah['dc_post_pds'] = cats_event_iah['dc0']-cats_event_iah['help_received']+cats_event_iah['help_fee']*option_CB        
    
    cats_event_iah['dw'] = calc_delta_welfare(cats_event_iah,macro_event,is_revised_dw)
    cats_event_iah = cats_event_iah.reset_index().set_index(event_level)

    ###########
    #OUTPUT
    df_out = pd.DataFrame(index=macro_event.index)
    
    #aggregates dK and delta_W at df level
    # --> dK, dW are averages per individual
    df_out['dK']        = cats_event_iah[['dk0'    ,'pcwgt']].prod(axis=1).sum(level=event_level)/cats_event_iah['pcwgt'].sum(level=event_level)
    #df_out['dK_public'] = cats_event_iah[['dk_public','pcwgt']].prod(axis=1).sum(level=event_level)/cats_event_iah['pcwgt'].sum(level=event_level)
    df_out['delta_W']   = cats_event_iah[['dw'     ,'pcwgt']].prod(axis=1).sum(level=event_level)/cats_event_iah['pcwgt'].sum(level=event_level)

    # dktot is already summed with RP -- just add them normally to get losses
    df_out['dKtot']       =      df_out['dK']*cats_event_iah['pcwgt'].sum(level=event_level)#macro_event['pop']
    df_out['delta_W_tot'] = df_out['delta_W']*cats_event_iah['pcwgt'].sum(level=event_level)#macro_event['pop'] 
    # ^ dK and dK_tot include both public and private losses

    df_out['average_aid_cost_pc'] = (cats_event_iah[['pcwgt','help_fee']].prod(axis=1).sum(level=event_level))/cats_event_iah['pcwgt'].sum(level=event_level)
    
    if return_stats:
        if not 'has_received_help_from_PDS_cat' in cats_event_iah.columns:
            stats = np.setdiff1d(cats_event_iah.columns,event_level+['helped_cat','affected_cat','hhid']+[i for i in ['province'] if i in cats_event_iah.columns])
        else:
            stats = np.setdiff1d(cats_event_iah.columns,event_level+['helped_cat','affected_cat','hhid','has_received_help_from_PDS_cat']+[i for i in ['province'] if i in cats_event_iah.columns])
		
        print('stats are '+','.join(stats))
        df_stats = agg_to_event_level(cats_event_iah,stats,event_level)
        df_out[df_stats.columns]=df_stats 
		    
    if return_iah:
        return df_out,cats_event_iah
    else: 
        return df_out

	
def process_output(pol_str,out,macro_event,economy,default_rp,return_iah=True,is_local_welfare=False,is_revised_dw=True):

    #unpacks if needed
    if return_iah:
        dkdw_event,cats_event_iah  = out

    else:
        dkdw_event = out

    ##AGGREGATES LOSSES
    #Averages over return periods to get dk_{hazard} and dW_{hazard}
    dkdw_h = average_over_rp1(dkdw_event,default_rp,macro_event['protection']).set_index(macro_event.index)
    macro_event[dkdw_h.columns]=dkdw_h

    #computes socio economic capacity and risk at economy level
    macro = calc_risk_and_resilience_from_k_w(macro_event,cats_event_iah,economy,is_local_welfare,is_revised_dw)

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
    
def same_rps_all_hazards(fa_ratios):
    ''' inspired by interpolate_rps but made to make sure all hazards have the same return periods (not that the protection rps are included by hazard)'''
    flag_stack= False
    if 'rp' in get_list_of_index_names(fa_ratios):
        fa_ratios = fa_ratios.unstack('rp')
        flag_stack = True
        
    #in case of a Multicolumn dataframe, perform this function on each one of the higher level columns
    if type(fa_ratios.columns)==pd.MultiIndex:
        keys = fa_ratios.columns.get_level_values(0).unique()
        return pd.concat({col:same_rps_all_hazards(fa_ratios[col]) for col in  keys}, axis=1).stack('rp')

    ### ACTUAL FUNCTION    
    #figures out all the return periods to be included
    all_rps = fa_ratios.columns.tolist()
    
    fa_ratios_rps = fa_ratios.copy()
    
    fa_ratios_rps = fa_ratios_rps.reindex_axis(sorted(fa_ratios_rps.columns), axis=1)
    # fa_ratios_rps = fa_ratios_rps.interpolate(axis=1,limit_direction="both",downcast="infer")
    fa_ratios_rps = fa_ratios_rps.interpolate(axis=1,limit_direction="both")
    if flag_stack:
        fa_ratios_rps = fa_ratios_rps.stack('rp')
    
    return fa_ratios_rps    


	
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
        fa_ratios_rps[0]=fa_ratios_rps.iloc[:,0]- fa_ratios_rps.columns[0]*(
        fa_ratios_rps.iloc[:,1]-fa_ratios_rps.iloc[:,0])/(
        fa_ratios_rps.columns[1]-fa_ratios_rps.columns[0])
        
    
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

def calc_delta_welfare(micro, macro,is_revised_dw,study=False):
    # welfare cost from consumption before (c) and after (dc_npv_post) event. Line by line

    #####################################
    # If running in 'study' mode, just load the file from my desktop
    # ^ grab one hh in the poorest quintile, and one in the wealthiest
    temp, c_mean = None, None
    if study == True:
        temp = pd.read_csv('~/Desktop/my_temp.csv',index_col=['Division','hazard','rp'])

        c_mean = float(temp[['pcwgt','c']].prod(axis=1).sum()/temp['pcwgt'].sum())

        temp['c_mean'] = c_mean
        temp = pd.concat([temp.loc[(temp.quintile==1)].head(2),temp.loc[(temp.quintile==5)].head(2)])

        #temp['dc']/=1.E5
        # ^ uncomment here if we want to make sure that (dw/wprime) converges to dc for small losses among wealthy

    else:
        mac_ix = macro.index.names
        mic_ix = micro.index.names

        #temp = pd.merge(micro.reset_index(),macro.reset_index(),on=[i for i in mac_ix]).reset_index().set_index([i for i in mic_ix])
        # ^ is merge of cats_event with macro_event actually necessary?
        temp = micro.copy()
        
        # Upper limit for per cap dw
        c_mean = temp[['pcwgt','c']].prod(axis=1).sum()/temp['pcwgt'].sum()
        temp['dw_limit'] = abs(20.*c_mean)

    ######################################
    # For comparison: this is the legacy definition of dw
    #temp['w'] = welf1(temp['c']/const_rho, const_ie, temp['c_5']/const_rho)
    #temp['dw_dep'] = (welf1(temp['c']/const_rho, const_ie, temp['c_5']/const_rho)
    #              - welf1(temp['c']/const_rho-temp['dc_npv_post'], const_ie,temp['c_5']/const_rho))
    #temp['wprime'] =(welf(temp['gdp_pc_pp_prov']/const_rho+h,const_ie)-welf(temp['gdp_pc_pp_prov']/const_rho-h,const_ie))/(2*h) <-- Missing a factor of (1-eta) in denom here?
    #temp['dw_curr'] = temp['dw_dep']/temp['wprime']

    #if is_revised_dw == False: 
    #    print('using legacy calculation of dw')
    #    temp = temp.reset_index().set_index([i for i in mic_ix])
    #    return temp['dw_dep']

    ########################################
    # Returns the revised ('rev') definition of dw
    print('using revised calculation of dw')

    # New defintion of w'
    temp['wprime'] = c_mean**(-const_ie)

    # Set-up to be able to calculate integral
    temp['const'] = -1.*(temp['c']**(1.-const_ie))/(1.-const_ie)
    temp['integ'] = 0.0

    my_out_x, my_out_yA, my_out_yNA, my_out_yzero = [], [], [], []
    x_min, n_steps = 0.,5E2
    x_max = (np.log(1/0.05)/float(temp['hh_reco_rate'].min()))
    if x_max > 25: x_max = 25
    print('\nIntegrating well-being losses over '+str(x_max)+' years after disaster') 
    # ^ make sure that, if T_recon changes, so does x_max!

    int_dt,step_dt = np.linspace(x_min,x_max,num=n_steps,endpoint=True,retstep=True)

    # Calculate integral
    for i_dt in int_dt:
        temp['integ'] += step_dt*((1.-(temp['dc0']*math.e**(-i_dt*temp['hh_reco_rate'])-temp['help_received']*const_pds_rate*math.e**(-i_dt*const_pds_rate))/temp['c'])**(1-const_ie)-1.)*math.e**(-i_dt*const_rho)
        # NB: dc0 has both public and private capital in it...need to allow for the possibility of independent reconstruction times
        # NOTE: if consumption goes negative, the integral can't be calculated...death!
        # --> if consumption increases, dw will come out negative. that's just making money off the disaster.
        
    ################################
    # 'revised' calculation of dw
    temp['dw_curr_no_clip'] = temp[['const','integ']].prod(axis=1)/temp['wprime']
    temp.loc[(temp.dc0 >= temp.c),'dw_curr_no_clip'] = temp.loc[(temp.dc0 >= temp.c),['dw_limit']].prod(axis=1)
    temp['dw'] = (temp[['const','integ']].prod(axis=1)).clip(upper=temp[['dw_limit','wprime']].prod(axis=1))
    # ^ calculate dw for most hh, including ceiling on dw
    temp.loc[(temp.dc0 >= temp.c),'dw'] = temp.loc[(temp.dc0 >= temp.c),['dw_limit','wprime']].prod(axis=1)
    # ^ assign dw = upper limit for all hh where dc0 > c
    
    # two alternative definitions of w'
    temp['dw_curr'] = temp['dw']/temp['wprime']

    print('Applying upper limit for dw = ',temp['dw_limit'].mean())
    temp['dw_tot'] = temp[['dw_curr','pcwgt']].prod(axis=1)
    temp.loc[(temp.dw_curr==temp.dw_limit)].to_csv(debug+'my_late_excess.csv')

    print('\nTotal well-being losses before deathclip:',round(float(temp[['dw_curr_no_clip','pcwgt']].prod(axis=1).sum())/1.E6,3))

    tmp_out = pd.DataFrame(index=temp.sum(level=[i for i in mac_ix]).index)

    tmp_out['dk_tot'] = temp[['dk0','pcwgt']].prod(axis=1).sum(level=[i for i in mac_ix])/1.E6
    tmp_out['dw_tot'] = temp[['dw_curr','pcwgt']].prod(axis=1).sum(level=[i for i in mac_ix])/1.E6
    tmp_out['res_tot'] = tmp_out['dk_tot']/tmp_out['dw_tot']

    tmp_out['dk_lim'] = temp.loc[(temp.dw_curr==temp.dw_limit),['dk0','pcwgt']].prod(axis=1).sum(level=[i for i in mac_ix])/1.E6    
    tmp_out['dw_lim'] = temp.loc[(temp.dw_curr==temp.dw_limit),['dw_curr','pcwgt']].prod(axis=1).sum(level=[i for i in mac_ix])/1.E6
    tmp_out['res_lim'] = tmp_out['dk_lim']/tmp_out['dw_lim']

    tmp_out['dk_sub'] = temp.loc[(temp.dw_curr!=temp.dw_limit),['dk0','pcwgt']].prod(axis=1).sum(level=[i for i in mac_ix])/1.E6
    tmp_out['dw_sub'] = temp.loc[(temp.dw_curr!=temp.dw_limit),['dw_curr','pcwgt']].prod(axis=1).sum(level=[i for i in mac_ix])/1.E6
    tmp_out['res_sub'] = tmp_out['dk_sub']/tmp_out['dw_sub']

    tmp_out['ratio_dw_lim_tot']  = tmp_out['dw_lim']/tmp_out['dw_tot']
    tmp_out.to_csv(debug+'my_summary.csv')
    print('Wrote out summary stats for dw')
    temp[['pcwgt','k','c','dk0','di0','dc0','help_received','dw_limit','wprime','const','integ','dw','dw_curr','dw_curr_no_clip']].head(10000).to_csv(debug+'my_temp.csv')

    return temp['dw']
	
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

def calc_risk_and_resilience_from_k_w(df, cats_event_iah,economy,is_local_welfare,is_revised_dw): 
    """Computes risk and resilience from dk, dw and protection. Line by line: multiple return periods or hazard is transparent to this function"""
    df=df.copy()    
    ############################
    #Expressing welfare losses in currency 
    #discount rate
    h=1e-4

    if is_revised_dw:
        #if is_local_welfare or not is_local_welfare:
        # ^ no dependence on this flag, for now
        c_mean = cats_event_iah[['pcwgt','c']].prod(axis=1).sum()/cats_event_iah['pcwgt'].sum()
        wprime = c_mean**(-const_ie)

    if not is_revised_dw:
        print('Getting wprime (legacy)')
        if is_local_welfare:
            wprime =(welf(df['gdp_pc_pp_prov']/const_rho+h,df['income_elast'])-welf(df['gdp_pc_pp_prov']/const_rho-h,df['income_elast']))/(2*h*(1-const_ie))
            print('Adding (1-eta) to denominator of legacy wprime...')
        else:
            wprime =(welf(df['gdp_pc_pp_nat']/const_rho+h,df['income_elast'])-welf(df['gdp_pc_pp_nat']/const_rho-h,df['income_elast']))/(2*h*(1-const_ie))
            print('Adding (1-eta) to denominator of legacy wprime...') 
    
    dWref = wprime*df['dK']
    #dWref = wprime*(df['dK']-df['dK_public'])
    # ^ doesn't add in dW from transfers from other provinces...

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
    df['risk_to_assets']  =df.resilience*df.risk
    
    return df
