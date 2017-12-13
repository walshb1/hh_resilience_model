# This script provides data input for the resilience indicator multihazard model for the Philippines. 
# Restructured from the global model and developed by Jinqiang Chen and Brian Walsh

# Magic
from IPython import get_ipython
get_ipython().magic('reset -f')
get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')

# Import packages for data analysis
from lib_asset_info import *
from lib_country_dir import *
from lib_gather_data import *
from lib_sea_level_rise import *
from replace_with_warning import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import isnull
import os, time
import warnings
import sys
warnings.filterwarnings('always',category=UserWarning)

if len(sys.argv) < 2:
    print('Need to list country. Try PH or FJ')
    assert(False)
else: myCountry = sys.argv[1]

# Set up directories/tell code where to look for inputs & where to save outputs
intermediate = set_directories(myCountry)

# Options and parameters
economy       = get_economic_unit(myCountry)

# Levels of index at which one event happens
event_level   = [economy, 'hazard', 'rp']

#Country dictionaries
#STATE/PROVINCE NAMES
df = get_places(myCountry,economy)
prov_code,region_code = get_places_dict(myCountry)

###Define parameters
df['pi']                     = reduction_vul           # how much early warning reduces vulnerability
df['rho']                    = discount_rate           # discount rate
df['shareable']              = asset_loss_covered      # target of asset losses to be covered by scale up
df['avg_prod_k']             = get_avg_prod(myCountry) # average productivity of capital, value from the global resilience model
df['T_rebuild_K']            = reconstruction_time     # Reconstruction time
df['income_elast']           = inc_elast               # income elasticity
df['max_increased_spending'] = max_support             # 5% of GDP in post-disaster support maximum, if everything is ready

# Protected from events with RP < 'protection'
if myCountry == 'PH':  df['protection'] = 1
if myCountry == 'FJ':  df['protection'] = 1
if myCountry == 'SL':  df['protection'] = 1

# Secondary dataframe, if necessary
# For PH: this is GDP per cap info:
df2 = get_df2(myCountry)

inc_sf = None
if myCountry =='FJ': inc_sf = (4.632E9/0.48) # GDP in USD (2016, WDI) -> FJD
cat_info = load_survey_data(myCountry,inc_sf)

print('Survey population:',cat_info.pcwgt.sum())

if myCountry == 'PH':
    get_hhid_FIES(cat_info)
    cat_info = cat_info.rename(columns={'w_prov':'province','w_regn':'region'}).reset_index()
    cat_info['province'].replace(prov_code,inplace=True)     
    cat_info['region'].replace(region_code,inplace=True)
    cat_info = cat_info.reset_index().set_index(economy).drop(['index','level_0'],axis=1)

    # There's no region info in df--put that in...
    df = df.reset_index().set_index('province')
    cat_info = cat_info.reset_index().set_index('province')
    df['region'] = cat_info[~cat_info.index.duplicated(keep='first')].region

    try:
        df.reset_index()[['province','region']].to_csv('../inputs/PH/prov_to_reg_dict.csv',header=True)        
        print('Updated PH regional-provincial dict')
    except: 
        print('Print cound not update regional-provincial dict')

    df = df.reset_index().set_index(economy)

    df['psa_pop'] = df.sum(level=economy)
    df = df.mean(level=economy)

    # There's no region info in df2--put that in...
    df2 = df2.reset_index().set_index('province')
    df2['region'] = cat_info[~cat_info.index.duplicated(keep='first')].region
    df2 = df2.reset_index().set_index(economy)    

    df2['shewp'] = df2[['shewp','pop']].prod(axis=1).sum(level=economy)/df2['pop'].sum(level=economy)    
    df2['shewr'] = df2[['shewr','pop']].prod(axis=1).sum(level=economy)/df2['pop'].sum(level=economy)   
    df2['gdp_pc_pp'] = df2[['gdp_pc_pp','pop']].prod(axis=1).sum(level=economy)/df2['pop'].sum(level=economy)

    df2['pop'] = df2['pop'].sum(level=economy)
    df2['gdp_pp'] = df2['gdp_pp'].sum(level=economy)
    df2 = df2.mean(level=economy)

    cat_info = cat_info.reset_index().set_index(economy)

if myCountry == 'SL':
    df = df.reset_index()
    df['district'].replace(prov_code,inplace=True)
    df = df.reset_index().set_index(economy).drop(['index'],axis=1)

    cat_info = cat_info.reset_index()
    cat_info['district'].replace(prov_code,inplace=True) #replace district code with its name
    cat_info = cat_info.reset_index().set_index(economy).drop(['index'],axis=1)

# Define per capita income (in local currency)
df['gdp_pc_prov'] = cat_info[['pcinc','pcwgt']].prod(axis=1).sum(level=economy)/cat_info['pcwgt'].sum(level=economy)
df['gdp_pc_nat'] = cat_info[['pcinc','pcwgt']].prod(axis=1).sum()/cat_info['pcwgt'].sum()
# ^ this is per capita income

df['pop'] = cat_info.pcwgt.sum(level=economy)

if myCountry == 'PH':
    df['pct_diff'] = 100.*(df['psa_pop']-df['pop'])/df['pop']

#Vulnerability
print('Getting vulnerabilities')
vul_curve = get_vul_curve(myCountry,'wall')
for thecat in vul_curve.desc.unique():

    if myCountry == 'PH': cat_info.ix[cat_info.walls.values == thecat,'v'] = vul_curve.loc[vul_curve.desc.values == thecat].v.values
    if myCountry == 'FJ': cat_info.ix[cat_info.Constructionofouterwalls.values == thecat,'v'] = vul_curve.loc[vul_curve.desc.values == thecat].v.values
    # Fiji doesn't have info on roofing, but it does have info on the condition of outer walls. Include that as a multiplier?
    if myCountry == 'SL': cat_info.ix[cat_info.walls.values == thecat,'v'] = vul_curve.loc[vul_curve.desc.values == thecat].v.values

print('Getting roof info')
# Get roofing data (but Fiji doesn't have this info)
if myCountry != 'FJ':
    vul_curve = get_vul_curve(myCountry,'roof')
    for thecat in vul_curve.desc.unique():
        cat_info.ix[cat_info.roof.values == thecat,'v'] += vul_curve.loc[vul_curve.desc.values == thecat].v.values
    cat_info.v = cat_info.v/2

if myCountry == 'PH':
    cat_info.ix[cat_info.v==0.1,'v'] *= np.random.uniform(.8,2,cat_info.ix[cat_info.v==0.1].shape[0])
    cat_info.ix[cat_info.v==0.25,'v'] *= np.random.uniform(.8,1.2,cat_info.ix[cat_info.v==0.25].shape[0])  
    cat_info.ix[(10*(cat_info.v-.4)).round()==0,'v'] *= np.random.uniform(.8,1.2,cat_info.ix[(10*(cat_info.v-.4)).round()==0].shape[0])
    cat_info.ix[cat_info.v==0.55,'v'] *= np.random.uniform(.8,1.2,cat_info.ix[cat_info.v==0.55].shape[0])
    cat_info.ix[cat_info.v==0.70,'v'] *= np.random.uniform(.8,1.2,cat_info.ix[cat_info.v==0.70].shape[0]) 
    cat_info.drop(['walls','roof'],axis=1,inplace=True)

    plot_simple_hist(cat_info,['v'],[''],'/Users/brian/Desktop/Dropbox/Bank/unbreakable_writeup/Figures/vulnerabilities_log.pdf',uclip=1,nBins=25,xlab='Vulnerability ($v_h$)',logy=True)
    plot_simple_hist(cat_info,['v'],[''],'/Users/brian/Desktop/Dropbox/Bank/unbreakable_writeup/Figures/vulnerabilities.pdf',uclip=1,nBins=25,xlab='Vulnerability ($v_h$)',logy=False)
    
if myCountry == 'FJ':
    cat_info.ix[cat_info.v==0.1,'v'] *= np.random.uniform(.8,2,cat_info.ix[cat_info.v==0.1].shape[0])
    cat_info.ix[cat_info.v==0.4,'v'] *= np.random.uniform(.8,1.2,cat_info.ix[cat_info.v==0.4].shape[0])
    cat_info.ix[cat_info.v==0.7,'v'] *= np.random.uniform(.8,1.2,cat_info.ix[cat_info.v==0.7].shape[0]) 
    cat_info.drop(['Constructionofouterwalls','Conditionofouterwalls'],axis=1,inplace=True)

print('Setting c to pcinc') 
cat_info['c'] = cat_info['pcinc']    
# --> What's the difference between income & consumption/disbursements?
# --> totdis = 'total family disbursements'    
# --> totex = 'total family expenditures'
# --> pcinc_s seems to be what they use to calculate poverty...
# --> can be converted to pcinc_ppp11 by dividing by (365*21.1782)

# Cash receipts, abroad & domestic, other gifts
cat_info['social'] = cat_info['pcsoc']/cat_info['pcinc']
cat_info.ix[cat_info.social>=1,'social'] = 0.99
# --> All of this is selected & defined in lib_country_dir
# --> Excluding international remittances ('cash_abroad')

scale_fac = 1.

print('Getting pov line')
if myCountry == 'PH':    
    cat_info['pov_line'] = cat_info.loc[(cat_info.ispoor == 1),'pcinc'].max() # <-- Individual
elif myCountry == 'FJ':
    pass
    # Poverty line set (and scaled) in scale_hh function in lib_country_dir

elif myCountry == 'SL':
    print('Need SL poverty info!!')
    cat_info['pov_line'] = 100000.

print('Total population:',cat_info.pcwgt.sum())
print('Total n households:',cat_info.hhwgt.sum())
print('--> Individuals in poverty:', cat_info.loc[(cat_info.pcinc_ae <= cat_info.pov_line),'pcwgt'].sum())
print('-----> Families in poverty:', cat_info.loc[(cat_info.pcinc_ae <= cat_info.pov_line), 'hhwgt'].sum())

if myCountry == 'FJ':
    print('-----------> Rural poverty:', cat_info.loc[(cat_info.Sector=='Rural')&(cat_info.pcinc_ae <= cat_info.pov_line),'pcwgt'].sum()/cat_info.loc[cat_info.Sector=='Rural','pcwgt'].sum())
    print('-----------> Urban poverty:', cat_info.loc[(cat_info.Sector=='Urban')&(cat_info.pcinc_ae <= cat_info.pov_line),'pcwgt'].sum()/cat_info.loc[cat_info.Sector=='Urban','pcwgt'].sum())
    print('-----> Rural pov (flagged):',round(100.*cat_info.loc[(cat_info.Sector=='Rural')&(cat_info.ispoor==1),'pcwgt'].sum()/cat_info.loc[cat_info.Sector=='Rural','pcwgt'].sum(),1),'%')
    print('-----> Urban pov (flagged):',round(100.*cat_info.loc[(cat_info.Sector=='Urban')&(cat_info.ispoor==1),'pcwgt'].sum()/cat_info.loc[cat_info.Sector=='Urban','pcwgt'].sum(),1),'%')

# Change the name: district to code, and create an multi-level index 
cat_info = cat_info.rename(columns={'district':'code','HHID':'hhid'})

# Assing weighted household consumption to quintiles within each province
print('Finding quintiles')
listofquintiles=np.arange(0.20, 1.01, 0.20)
cat_info = cat_info.reset_index().groupby(economy,sort=True).apply(lambda x:match_percentiles(x,perc_with_spline(reshape_data(x.c),reshape_data(x.pcwgt),listofquintiles),'quintile'))
# 'c_5_nat' is the upper consumption limit for the lowest 5% throughout country
percentiles_05 = np.arange(0.05, 1.01, 0.05) #create a list of deciles 
my_c5 = match_percentiles(cat_info,perc_with_spline(reshape_data(cat_info.c),reshape_data(cat_info.pcwgt),percentiles_05),'pctle_05_nat')
cat_info['c_5_nat'] = cat_info.ix[cat_info.pctle_05_nat==1,'c'].max()

cat_info = cat_info.reset_index().groupby(economy,sort=True).apply(lambda x:match_percentiles(x,perc_with_spline(reshape_data(x.c),reshape_data(x.pcwgt),percentiles_05),'pctle_05'))
if 'level_0' in cat_info.columns:
    cat_info = cat_info.drop(['level_0','index'],axis=1)
cat_info_c_5 = cat_info.reset_index().groupby(economy,sort=True).apply(lambda x:x.ix[x.pctle_05==1,'c'].max())
cat_info = cat_info.reset_index().set_index([economy,'hhid']) #change the name: district to code, and create an multi-level index 
cat_info['c_5'] = broadcast_simple(cat_info_c_5,cat_info.index)
cat_info['c_5'] = cat_info.c_5.fillna(cat_info.c_5.mean(level=economy).min())
# ^ this is a line to prevent c_5 from being left empty due to paucity of hh from a given province (for Rotuma, FJ)

cat_info.drop([icol for icol in ['level_0','index','pctle_05','pctle_05_nat'] if icol in cat_info.columns],axis=1,inplace=True)

#cat_info_c_5 = cat_info.reset_index().groupby(economy,sort=True).apply(lambda x:x.ix[x.quintile==1,'c'].max())
#cat_info = cat_info.reset_index().set_index([economy,'hhid']) #change the name: district to code, and create an multi-level index 
#cat_info['c_5'] = broadcast_simple(cat_info_c_5,cat_info.index)

# Population of household as fraction of population of provinces
cat_info['n'] = cat_info.hhwgt/cat_info.hhwgt.sum(level=economy)

# population of household as fraction of total population
cat_info['n_national'] = cat_info.hhwgt/cat_info.hhwgt.sum() 

# These sum to 1 per province & nationally, respectively
#print('province normalization:',cat_info.n.sum(level=economy)) 
print('normalization:',cat_info.n_national.sum())

print('Get the tax used for domestic social transfer')
# --> tau_tax = 0.075391
df['tau_tax'] = cat_info[['social','c','pcwgt']].prod(axis=1, skipna=False).sum()/cat_info[['c','pcwgt']].prod(axis=1, skipna=False).sum()

print('Get the share of Social Protection')
cat_info['gamma_SP'] = cat_info[['social','c']].prod(axis=1,skipna=False)*cat_info['pcwgt'].sum()/cat_info[['social','c','pcwgt']].prod(axis=1, skipna=False).sum()
cat_info.drop('n_national',axis=1,inplace=True)

# Intl remittances: subtract from 'c'
print('Calculate capital')
cat_info['k'] = (1-cat_info['social'])*cat_info['c']/((1-df['tau_tax'])*df['avg_prod_k']) #calculate the capital
cat_info.ix[cat_info.k<0,'k'] = 0.0

if myCountry == 'FJ':
    #replace division codes with names
    df = df.reset_index()
    df['Division'].replace(prov_code,inplace=True)
    df = df.reset_index().set_index(['Division']).drop(['index'],axis=1)

    cat_info = cat_info.reset_index()
    cat_info['Division'].replace(prov_code,inplace=True) # replace division code with its name
    cat_info = cat_info.reset_index().set_index(['Division','hhid']).drop(['index'],axis=1)

# Getting rid of Prov_code 98, 99 here
#print('Check total population:',cat_info.pcwgt.sum())
cat_info.dropna(inplace=True,how='all')
#print('Check total population (after dropna):',cat_info.pcwgt.sum())

# Assign access to early warning based on 'ispoor' flag
if myCountry == 'PH':
    # --> doesn't seem to match up with the quintiles we assigned
    cat_info['shew'] = broadcast_simple(df2['shewr'],cat_info.index)
    cat_info.ix[cat_info.ispoor == 1,'shew'] = broadcast_simple(df2['shewp'],cat_info.index)
elif myCountry == 'FJ' or myCountry == 'SL': 
    cat_info['shew'] = 0
    # can't find relevant info for Fiji and Sri Lanka

# Exposure
cat_info['fa'] = 0
cat_info.fillna(0,inplace=True)

# Cleanup dfs for writing out
cat_info_col = [economy,'province','hhid','region','pcwgt','pcwgt_ae','hhwgt','code','np','score','v','c','pcsoc','social','c_5','n','hhsize','hhsize_ae',
                'gamma_SP','k','shew','fa','quintile','ispoor','pcinc','pcinc_ae','pov_line','SP_FAP','SP_CPP','SP_SPS','nOlds',
                'SP_PBS','SP_FNPF','SPP_core','SPP_add']
cat_info = cat_info.drop([iXX for iXX in cat_info.columns.values.tolist() if (iXX in cat_info.columns and iXX not in cat_info_col)],axis=1)
cat_info_index = cat_info.drop([iXX for iXX in cat_info.columns.values.tolist() if iXX not in [economy,'hhid']],axis=1)

#########################
# HAZARD INFO
#
# This is the GAR
#hazard_ratios = pd.read_csv(inputs+'/PHL_frac_value_destroyed_gar_completed_edit.csv').set_index([economy, 'hazard', 'rp'])

# PHILIPPINES:
# This is the AIR dataset:
# df_haz is already in pesos
# --> Need to think about public assets
#df_haz = get_AIR_data(inputs+'/Risk_Profile_Master_With_Population.xlsx','Loss_Results','all','Agg')

df_haz,df_tikina = get_hazard_df(myCountry,economy,rm_overlap=True)
if myCountry == 'FJ': _ = get_SLR_hazard(myCountry,df_tikina)

# Edit & Shuffle provinces
if myCountry == 'PH':
    AIR_prov_rename = {'Shariff Kabunsuan':'Maguindanao',
                       'Davao Oriental':'Davao',
                       'Davao del Norte':'Davao',
                       'Metropolitan Manila':'Manila',
                       'Dinagat Islands':'Surigao del Norte'}
    df_haz['province'].replace(AIR_prov_rename,inplace=True) 

    # Add NCR 2-4 to AIR dataset
    df_NCR = pd.DataFrame(df_haz.loc[(df_haz.province == 'Manila')])
    df_NCR['province'] = 'NCR-2nd Dist.'
    df_haz = df_haz.append(df_NCR)

    df_NCR['province'] = 'NCR-3rd Dist.'
    df_haz = df_haz.append(df_NCR)
    
    df_NCR['province'] = 'NCR-4th Dist.'
    df_haz = df_haz.append(df_NCR)

    # In AIR, we only have 'Metropolitan Manila'
    # Distribute losses among Manila & NCR 2-4 according to assets
    cat_info = cat_info.reset_index()
    k_NCR = cat_info.loc[((cat_info.province == 'Manila') | (cat_info.province == 'NCR-2nd Dist.') 
                          | (cat_info.province == 'NCR-3rd Dist.') | (cat_info.province == 'NCR-4th Dist.')), ['k','pcwgt']].prod(axis=1).sum()

    for k_type in ['value_destroyed_prv','value_destroyed_pub']:
        df_haz.loc[df_haz.province ==        'Manila',k_type] *= cat_info.loc[cat_info.province ==        'Manila', ['k','pcwgt']].prod(axis=1).sum()/k_NCR
        df_haz.loc[df_haz.province == 'NCR-2nd Dist.',k_type] *= cat_info.loc[cat_info.province == 'NCR-2nd Dist.', ['k','pcwgt']].prod(axis=1).sum()/k_NCR
        df_haz.loc[df_haz.province == 'NCR-3rd Dist.',k_type] *= cat_info.loc[cat_info.province == 'NCR-3rd Dist.', ['k','pcwgt']].prod(axis=1).sum()/k_NCR
        df_haz.loc[df_haz.province == 'NCR-4th Dist.',k_type] *= cat_info.loc[cat_info.province == 'NCR-4th Dist.', ['k','pcwgt']].prod(axis=1).sum()/k_NCR
        
    # Add region info to df_haz:
    df_haz = df_haz.reset_index().set_index('province')
    cat_info = cat_info.reset_index().set_index('province')
    df_haz['region'] = cat_info[~cat_info.index.duplicated(keep='first')].region

    df_haz = df_haz.reset_index().set_index(economy)
    cat_info = cat_info.reset_index().set_index(economy)

    # Sum over the provinces that we're merging
    # Losses are absolute value, so they are additive
    df_haz = df_haz.reset_index().set_index([economy,'hazard','rp']).sum(level=[economy,'hazard','rp']).drop(['index'],axis=1)

    df_haz['value_destroyed'] = df_haz[['value_destroyed_prv','value_destroyed_pub']].sum(axis=1)
    df_haz['hh_share'] = df_haz['value_destroyed_prv']/df_haz['value_destroyed']
    
elif myCountry == 'FJ':
    df_haz = df_haz.reset_index().set_index([economy,'hazard','rp']).sum(level=[economy,'hazard','rp'])
    # All the magic happens inside get_hazard_df()

# Turn losses into fraction
cat_info = cat_info.reset_index().set_index([economy])

hazard_ratios = cat_info[['k','pcwgt']].prod(axis=1).sum(level=economy).to_frame(name='HIES_capital')
hazard_ratios = hazard_ratios.join(df_haz,how='outer')

if myCountry == 'PH':
    hazard_ratios['frac_destroyed'] = hazard_ratios['value_destroyed']/hazard_ratios['HIES_capital']
    hazard_ratios = hazard_ratios.drop(['HIES_capital','value_destroyed','value_destroyed_prv','value_destroyed_pub'],axis=1)

elif myCountry == 'FJ':

    # PLOT
    plot_df = hazard_ratios.loc[:,['Exp_Value','HIES_capital','frac_bld_res','frac_agr']]
    plot_df['RES_AGR_Exp_Value'] = hazard_ratios[['Exp_Value','frac_bld_res']].prod(axis=1) + hazard_ratios[['Exp_Value','frac_agr']].prod(axis=1)
    plot_df['RES_Exp_Value'] = hazard_ratios[['Exp_Value','frac_bld_res']].prod(axis=1)

    plot_df = plot_df.mean(level='Division')

    ax = plot_df.plot.scatter('HIES_capital','Exp_Value')
    fit_line_1 = np.polyfit(plot_df['HIES_capital'],plot_df['Exp_Value'],1)

    plot_df.plot.scatter('HIES_capital','RES_AGR_Exp_Value',ax=ax)
    fit_line_2 = np.polyfit(plot_df['HIES_capital'],plot_df['RES_AGR_Exp_Value'],1)

    plot_df.plot.scatter('HIES_capital','RES_Exp_Value',ax=ax)
    fit_line_3 = np.polyfit(plot_df['HIES_capital'],plot_df['RES_Exp_Value'],1)

    ax.plot()
    
    my_linspace_x = np.array(np.linspace(plt.gca().get_xlim()[0],plt.gca().get_xlim()[1],10))
    my_linspace_y1 = fit_line_1[0]*my_linspace_x+fit_line_1[1]
    my_linspace_y2 = fit_line_2[0]*my_linspace_x+fit_line_2[1]
    my_linspace_y3 = fit_line_3[0]*my_linspace_x+fit_line_3[1]
    
    plt.plot(my_linspace_x,my_linspace_y1)
    plt.plot(my_linspace_x,my_linspace_y2)
    plt.plot(my_linspace_x,my_linspace_y3)

    plt.annotate('All Assets '+str(round(100.*fit_line_1[0],2))+'%',[0.,8.E9])
    plt.annotate('Res+Ag Assets '+str(round(100.*fit_line_2[0],2))+'%',[0.,7.E9])
    plt.annotate('Res Assets '+str(round(100.*fit_line_3[0],2))+'%',[0.,6.E9])
    
    fig = plt.gcf()
    fig.savefig(intermediate+'/new_HIES_vs_PCRAFI_household_assets.pdf',format='pdf')

    # This is *the* line
    # --> fa is losses/exposed_value
    #hazard_ratios['frac_destroyed'] = hazard_ratios['fa'] 

elif myCountry == 'SL':
    hazard_ratios['frac_destroyed'] = hazard_ratios['fa']

# Have frac destroyed, need fa...
# Frac value destroyed = SUM_i(k*v*fa)

hazard_ratios = pd.merge(hazard_ratios.reset_index(),cat_info.reset_index(),on=economy,how='outer')

if myCountry == 'FJ':
    
    hazard_ratios = hazard_ratios.set_index(['Division','hazard','rp'])
    
    #print('Scaling up by factor of ',round(hazard_ratios['Exp_Value'].sum(level='Division').mean()/hazard_ratios['HIES_capital'].sum(level='Division').mean(),2))
    #cat_info['k'] *= hazard_ratios['Exp_Value'].sum(level='Division').mean()/hazard_ratios['HIES_capital'].sum(level='Division').mean()
    #df['avg_prod_k'] *= hazard_ratios['HIES_capital'].mean()/hazard_ratios['Exp_Value'].mean()

    print('Avg prod K:',df['avg_prod_k'].mean())

if 'hh_share' not in hazard_ratios.columns: hazard_ratios['hh_share'] = None
hazard_ratios = hazard_ratios.reset_index().set_index(event_level+['hhid'])[['frac_destroyed','v','k','pcwgt','hh_share']]

# Transfer fa in excess of 95% to vulnerability
fa_threshold = 0.95
hazard_ratios['fa'] = (hazard_ratios['frac_destroyed']/hazard_ratios['v']).fillna(1E-8)

hazard_ratios.loc[hazard_ratios.fa>fa_threshold,'v'] = (hazard_ratios.loc[hazard_ratios.fa>fa_threshold,['v','fa']].prod(axis=1)/fa_threshold).clip(0.99)
hazard_ratios['fa'] = hazard_ratios['fa'].clip(lower=0.0000001,upper=fa_threshold)

cat_info = cat_info.reset_index().set_index([economy,'hhid'])
#cat_info['v'] = hazard_ratios.reset_index().set_index([economy,'hhid'])['v'].mean(level=[economy,'hhid']).clip(upper=0.99)
# ^ I think this is throwing off the losses!! Average vulnerability isn't going to cut it
# --> Use hazard-specific vulnerability for each hh (in hazard_ratios instead of in cats_event)

# This function collects info on the value and vulnerability of public assets
cat_info, hazard_ratios = get_asset_infos(myCountry,cat_info,hazard_ratios,df_haz)

df.to_csv(intermediate+'/macro.csv',encoding='utf-8', header=True,index=True)

cat_info = cat_info.drop([icol for icol in ['level_0','index'] if icol in cat_info.columns],axis=1)
#cat_info = cat_info.drop([i for i in ['province'] if i != economy],axis=1)
cat_info.to_csv(intermediate+'/cat_info.csv',encoding='utf-8', header=True,index=True)

hazard_ratios= hazard_ratios.drop(['frac_destroyed'],axis=1).drop(["flood_fluv_def"],level="hazard")
hazard_ratios.to_csv(intermediate+'/hazard_ratios.csv',encoding='utf-8', header=True)

# If we have 2 sets of data on k, gdp, look at them now:
try:
    summary_df = pd.DataFrame({'macro_stats':df2[['gdp_pc_pp','pop']].prod(axis=1).sum(level=economy),
                               'hies':df['avg_prod_k'].mean()*cat_info[['k','pcwgt']].prod(axis=1).sum(level=economy)})
    summary_df['macro_to_hh_ratio'] = summary_df['macro_stats'].divide(summary_df['hies'])
    print(summary_df)

    totals = summary_df[['macro_stats','hies']].sum().squeeze()
    ratio = totals[0]/totals[1]

    print(totals, ratio)

except:
    print('Dont have 2 datasets for GDP. HH survey data:')
#    print(df['avg_prod_k'].mean()*cat_info[['k','pcwgt']].prod(axis=1).sum(level=economy))
    
    
if myCountry == 'FJ':
    # Compare assets from survey to assets from AIR-PCRAFI

    df_haz = df_haz.reset_index()
    my_df = ((df[['gdp_pc_prov','pop']].prod(axis=1))/df['avg_prod_k']).to_frame(name='HIES')
    my_df['PCRAFI'] = df_haz.ix[(df_haz.rp==1)&(df_haz.hazard=='TC'),['Division','Exp_Value']].set_index('Division')
    
    my_df['HIES']/=1.E9
    my_df['PCRAFI']/=1.E9
    
    ax = my_df.plot.scatter('PCRAFI','HIES')
    fit_line = np.polyfit(my_df['PCRAFI'],my_df['HIES'],1)
    ax.plot()
    
    plt.xlim(0.,8.)
    plt.ylim(0.,5.)
    
    my_linspace_x = np.array(np.linspace(plt.gca().get_xlim()[0],plt.gca().get_xlim()[1],10))
    my_linspace_y = fit_line[0]*my_linspace_x+fit_line[1]
    
    plt.plot(my_linspace_x,my_linspace_y)
    plt.annotate(str(round(100.*my_linspace_x[1]/my_linspace_y[1],1))+'%',[1.,4.])
    
    # 147 km main distribution line
    # 7000 km tranmission
    
    fig = plt.gcf()
    fig.savefig('HIES_vs_PCRAFI_assets.pdf',format='pdf')
    
