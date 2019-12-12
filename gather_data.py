# This script provides data input for the resilience indicator multihazard model for the Philippines, Fiji, Sri Lanka, and Malawi. 

# Restructured from the global model and developed by Jinqiang Chen and Brian Walsh

# Compiler/Python interface (Magic)
from IPython import get_ipython
get_ipython().magic('reset -f')
get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')

# Import packages for data analysis
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import isnull
import os, time
import warnings
import sys
import pickle

# Import local libraries
from libraries.lib_asset_info import *
from libraries.lib_country_dir import *
from libraries.lib_gather_data import *
from libraries.lib_sea_level_rise import *
from libraries.replace_with_warning import *
from libraries.lib_agents import optimize_reco
from libraries.lib_urban_plots import run_urban_plots
from libraries.lib_sp_analysis import run_sp_analysis
from libraries.lib_gather_data import *
from libraries.lib_make_regional_inputs_summary_table import *
from libraries.lib_drought import get_agricultural_vulnerability_to_drought, get_ag_value
#
from special_events.cyclone_idai_mw import get_idai_loss
#
warnings.filterwarnings('always',category=UserWarning)

#####################################
# Which country are we running over?

# --> This variable <sys.argv> is the first optional argument on the command line
# --> If you can't pass an argument, set <myCountry> in the "else:" loop equal to your country:
#
# --> BO = Bolivia
# or  FJ = Fiji 
# or  MW = Malawi
# or  PH = Philippines
# or  SL = Sri Lanka
#
if len(sys.argv) >= 2: myCountry = sys.argv[1]
else:
    myCountry = 'BO'
    print('Setting country to BO. Currently implemented: MW, PH, FJ, SL, BO')

#####################################
# Primary library used is lib_country_dir.py

# Set up directories/tell code where to look for inputs & where to save outputs
intermediate = set_directories(myCountry)

# Administrative unit (eg region or province)
economy = get_economic_unit(myCountry)

# Levels of index at which one event happens
event_level = [economy, 'hazard', 'rp']

#Country dictionaries
# df = state/province names
df = get_places(myCountry)
prov_code,region_code = get_places_dict(myCountry)


###Define parameters, all coming from lib_country_dir
df['avg_prod_k']             = get_avg_prod(myCountry) # average productivity of capital, value from the global resilience model
df['shareable']              = nominal_asset_loss_covered_by_PDS # target of asset losses to be covered by scale up
df['T_rebuild_K']            = reconstruction_time     # Reconstruction time
df['income_elast']           = inc_elast               # income elasticity
df['max_increased_spending'] = max_support             # 5% of GDP in post-disaster support maximum, if everything is ready
df['pi']                     = reduction_vul           # how much early warning reduces vulnerability
df['rho']                    = 0.3*df['avg_prod_k']    # discount rate
# ^ We have been using a constant discount rate = 0.06
# --> BUT: this breaks the assumption that hh are in steady-state equilibrium before the hazard


##########################
# Countries will be 'protected' from events with RP < 'protection'
# --> means that asset losses (dK) will be set to zero for these events
df['protection'] = 1
if myCountry == 'SL': df['protection'] = 5


##########################
# Big function loads standardized hh survey info
cat_info = load_survey_data(myCountry)
run_urban_plots(myCountry,cat_info.copy())
print('Survey population:',cat_info.pcwgt.sum())


cat_info = cat_info.reset_index().set_index([economy,'hhid'])
try: cat_info = cat_info.drop('index',axis=1)
except: pass
# Now we have a dataframe called <cat_info> with the household info.
# Index = [economy (='region';'district'; country-dependent), hhid]



########################################
# Calculate regional averages from household info
df['gdp_pc_prov'] = cat_info[['pcinc','pcwgt']].prod(axis=1).sum(level=economy)/cat_info['pcwgt'].sum(level=economy)
# ^ per capita income (in local currency), regional average
df['gdp_pc_nat'] = cat_info[['pcinc','pcwgt']].prod(axis=1).sum()/cat_info['pcwgt'].sum()
# ^ this is per capita income (local currency), national average

df['pop'] = cat_info.pcwgt.sum(level=economy)
# ^ regional population

try: df['pct_diff'] = 100.*(df['psa_pop']-df['pop'])/df['pop']
except: pass
# ^ (Philippines specific) compare PSA population to FIES population


########################################
########################################
# Asset vulnerability
print('Getting vulnerabilities')

vul_curve = get_vul_curve(myCountry,'wall')
# From diff: vul_curve = get_vul_curve(myCountry,'wall').set_index('desc').to_dict()
# below commented out
for thecat in vul_curve.desc.unique():  
    cat_info.loc[cat_info['walls'] == thecat,'v'] = float(vul_curve.loc[vul_curve.desc.values == thecat,'v'])
    # Fiji doesn't have info on roofing, but it does have info on the *condition* of outer walls. Include that as a multiplier?
    
# Get roofing data (but Fiji doesn't have this info)

try:
    print('Getting roof info')
    vul_curve = get_vul_curve(myCountry,'roof')
    for thecat in vul_curve.desc.unique():
        cat_info.loc[cat_info.roof.values == thecat,'v'] += vul_curve.loc[vul_curve.desc.values == thecat].v.values
    cat_info.v = cat_info.v/2
except: pass

# Get vulnerability of agricultural income to drought.
# --> includes fraction of income from ag, so v=0 for hh with no ag income
try: cat_info['v_ag'] = get_agricultural_vulnerability_to_drought(myCountry,cat_info)
except: cat_info['v_ag'] = -1.

########################################
########################################
# Random stuff--needs to be sorted
# --> What's the difference between income & consumption/disbursements?
# --> totdis = 'total family disbursements'    
# --> totex = 'total family expenditures'
# --> pcinc_s seems to be what they use to calculate poverty...
# --> can be converted to pcinc_ppp11 by dividing by (365*21.1782)

#cat_info = cat_info.reset_index().set_index(['hhid',economy])

# Save a sp_receipts_by_region.csv that has summary statistics on social payments
try: run_sp_analysis(myCountry,cat_info.copy())
except: pass

#cat_info = cat_info.reset_index('hhid')

# Cash receipts, abroad & domestic, other gifts
cat_info['social'] = (cat_info['pcsoc']/cat_info['c']).fillna(0)
# --> All of this is selected & defined in lib_country_dir
# --> Excluding international remittances ('cash_abroad')

print('Getting pov line')
cat_info = cat_info.reset_index().set_index('hhid')
if 'pov_line' not in cat_info.columns:
    try:
        cat_info.loc[cat_info.Sector=='Urban','pov_line'] = get_poverty_line(myCountry,'Urban')
        cat_info.loc[cat_info.Sector=='Rural','pov_line'] = get_poverty_line(myCountry,'Rural')
        cat_info['sub_line'] = get_subsistence_line(myCountry)
    except:
        try: cat_info['pov_line'] = get_poverty_line(myCountry,by_district=False)
        except: cat_info['pov_line'] = 0
if 'sub_line' not in cat_info.columns:
    try: cat_info['sub_line'] = get_subsistence_line(myCountry)
    except: cat_info['sub_line'] = 0
cat_info[['sub_line','pov_line']]

cat_info = cat_info.reset_index().set_index(event_level[0])

# Print some summary statistics from the survey data.
print(cat_info.describe().T)
print('Total population:',int(cat_info.pcwgt.sum()))
print('Total population (AE):',int(cat_info.aewgt.sum()))
print('Total n households:',int(cat_info.hhwgt.sum()))
print('Average income - (adults) ',cat_info[['pcinc','pcwgt']].prod(axis=1).sum()/cat_info[['pcwgt']].sum())
try: print('\nAverage income (Adults-eq)',cat_info[['aeinc','aewgt']].prod(axis=1).sum()/cat_info[['aewgt']].sum())
except: pass

try:
    print('--> Individuals in poverty (inc):', float(round(cat_info.loc[(cat_info.pcinc <= cat_info.pov_line),'pcwgt'].sum()/1.E6,3)),'million')
    print('-----> Households in poverty (inc):', float(round(cat_info.loc[(cat_info.pcinc <= cat_info.pov_line),'hhwgt'].sum()/1.E6,3)),'million')
    print('-->          AE in poverty (inc):', float(round(cat_info.loc[(cat_info.aeinc <= cat_info.pov_line),'aewgt'].sum()/1.E6,3)),'million')
except: pass

try:
    print('-----> Children in poverty (inc):', float(round(cat_info.loc[(cat_info.pcinc <= cat_info.pov_line),['N_children','hhwgt']].prod(axis=1).sum()/1.E6,3)),'million')
    print('------> Individuals in poverty (exclusive):', float(round(cat_info.loc[cat_info.eval('pcinc<=pov_line & pcinc>sub_line'),'pcwgt'].sum()/1E6,3)),'million')
    print('---------> Families in poverty (exclusive):', float(round(cat_info.loc[cat_info.eval('pcinc<=pov_line & pcinc>sub_line'),'hhwgt'].sum()/1E6,3)),'million')
    print('--> Individuals in subsistence (exclusive):', float(round(cat_info.loc[cat_info.eval('pcinc<=sub_line'),'pcwgt'].sum()/1E6,3)),'million')
    print('-----> Families in subsistence (exclusive):', float(round(cat_info.loc[cat_info.eval('pcinc<=sub_line'),'hhwgt'].sum()/1E6,3)),'million')
except: print('No subsistence info...')

print('\n--> Number in poverty (flagged poor):',float(round(cat_info.loc[(cat_info.ispoor==1),'pcwgt'].sum()/1E6,3)),'million')
print('--> Poverty rate (flagged poor):',round(100.*cat_info.loc[(cat_info.ispoor==1),'pcwgt'].sum()/cat_info['pcwgt'].sum(),1),'%\n\n\n')

# Save poverty_rate.csv, summary stats on poverty rate
pd.DataFrame({'population':cat_info['pcwgt'].sum(level=economy),
              'nPoor':cat_info.loc[cat_info.ispoor==1,'pcwgt'].sum(level=economy),
              'n_pov':cat_info.loc[cat_info.eval('pcinc<=pov_line & pcinc>sub_line'),'pcwgt'].sum(level=economy), # exclusive of subsistence
              'n_sub':cat_info.loc[cat_info.eval('pcinc<=sub_line'),'pcwgt'].sum(level=economy),
              'pctPoor':100.*cat_info.loc[cat_info.ispoor==1,'pcwgt'].sum(level=economy)/cat_info['pcwgt'].sum(level=economy)}).to_csv('../output_country/'+myCountry+'/poverty_rate.csv')
# Could also look at urban/rural if we have that divide
try:
    print('\n--> Rural poverty (flagged poor):',float(round(cat_info.loc[(cat_info.ispoor==1)&(cat_info.isrural),'pcwgt'].sum()/1E6,3)),'million')
    print('\n--> Urban poverty (flagged poor):',float(round(cat_info.loc[(cat_info.ispoor==1)&~(cat_info.isrural),'pcwgt'].sum()/1E6,3)),'million')
except: print('Sad fish')

# Standardize--hhid should be lowercase
cat_info = cat_info.rename(columns={'HHID':'hhid'})


#########################
# Calculate K from C

# Change the name: district to code, and create an multi-level index
if (myCountry == 'SL') or (myCountry =='BO'):
    cat_info = cat_info.rename(columns={'district':'code','HHID':'hhid'})

# tau_tax = total value of social as fraction of total C
# gamma_SP = Fraction of social that goes to each hh
print('Get the tax used for domestic social transfer and the share of Social Protection')
df['tau_tax'], cat_info['gamma_SP'] = social_to_tx_and_gsp(economy,cat_info)


# Calculate K from C - pretax income (without sp) over avg social income
# Only count pre-tax income that goes towards sp
print('Calculating capital from income')
# Capital is consumption over average productivity of capital over a multiplier which is net social payments/taxes for each household
cat_info['k'] = ((cat_info['c']/df['avg_prod_k'].mean())*((1-cat_info['social'])/(1-df['tau_tax'].mean()))).clip(lower=0.)
print('Derived capital from income')

cat_info.eval('k*pcwgt').sum(level=economy).to_csv('../inputs/'+myCountry+'/total_capital.csv')


#########################
# Replace any codes with names
if myCountry == 'FJ' or myCountry == 'RO' or myCountry == 'SL':
    df = df.reset_index()
    if myCountry == 'FJ' or myCountry == 'SL':
        df[economy] = df[economy].replace(prov_code)

    df = df.reset_index().set_index([economy])
    try: df = df.drop(['index'],axis=1)
    except: pass

    cat_info = cat_info.reset_index()
    if myCountry == 'FJ' or myCountry == 'SL':
        cat_info[economy].replace(prov_code,inplace=True) # replace division code with its name

    if myCountry == 'RO':
        cat_info[economy].replace(region_code,inplace=True) # replace division code with its name

    cat_info = cat_info.reset_index().set_index([economy,'hhid'])
    try: cat_info = cat_info.drop(['index'],axis=1)
    except: pass

elif myCountry == 'BO':
    df = df.reset_index()
    # 2015 data
    # df[economy] = df[economy].astype(int).replace(prov_code)
    cat_info = cat_info.reset_index()
    # cat_info[economy].replace(prov_code,inplace=True) # replace division code with its name
    cat_info = cat_info.reset_index().set_index([economy,'hhid']).drop(['index'],axis=1)
    

########################################
# Calculate regional averages from household info
df['gdp_pc_prov'] = cat_info[['pcinc','pcwgt']].prod(axis=1).sum(level=economy)/cat_info['pcwgt'].sum(level=economy)
# ^ per capita income (in local currency), regional average

df['gdp_pc_nat'] = cat_info[['pcinc','pcwgt']].prod(axis=1).sum()/cat_info['pcwgt'].sum()
# ^ this is per capita income (local currency), national average

#########################
print('Save out regional poverty rates to regional_poverty_rate.csv')
print(cat_info.head())

(100*cat_info.loc[cat_info.eval('c<pov_line'),'pcwgt'].sum(level=economy)
 /cat_info['pcwgt'].sum(level=economy)).to_frame(name='poverty_rate').to_csv('../inputs/'+myCountry+'/regional_poverty_rate.csv')

# Shouldn't be losing anything here
cat_info = cat_info.loc[cat_info['pcwgt'] != 0]
print('Check total population:',cat_info.pcwgt.sum())
if myCountry == 'RO':
    cat_info.to_csv('~/Desktop/tmp/RO_drops.csv')
# cat_info.dropna(inplace=True,how='any')
# Get rid of househouseholds with 0 consumption
if myCountry == 'BO':
    cat_info.drop(cat_info[cat_info['c'] ==0].index, inplace = True)
print('Check total population (after dropna):',cat_info.pcwgt.sum())

# Drop partially empty columns
if myCountry == 'BO':
    cat_info = cat_info.dropna(axis = 1)
    # Save total populations to file to compute fa from population affected
    cat_info.pcwgt.sum(level = 0).to_frame().rename({'pcwgt':'population'}, axis = 1).to_csv(os.path.join('../inputs/',myCountry,'population_by_state.csv'))

# Exposure
print('check:',cat_info.shape[0],'=?',cat_info.dropna().shape[0])

#cat_info.to_csv('~/Desktop/tmp/check.csv')
cat_info =cat_info.dropna()

# Cleanup dfs for writing out
cat_info_col = [economy,'province','hhid','region','pcwgt','aewgt','hhwgt','np','score','v','v_ag','c','pcinc_ag_gross',
                'pcsoc','social','c_5','hhsize','ethnicity','hhsize_ae','gamma_SP','k','quintile','ispoor','ismiddleclass','isrural','issub',
                'pcinc','aeinc','pcexp','pov_line','SP_FAP','SP_CPP','SP_SPS','nOlds','has_ew',
                'SP_PBS','SP_FNPF','SPP_core','SPP_add','axfin','pcsamurdhi','gsp_samurdhi','frac_remittance','N_children']
cat_info = cat_info.drop([i for i in cat_info.columns if (i in cat_info.columns and i not in cat_info_col)],axis=1)
cat_info_index = cat_info.drop([i for i in cat_info.columns if i not in [economy,'hhid']],axis=1)


#########################
# HAZARD INFO
special_event=None
#special_event = 'Idai'

# SL FLAG: get_hazard_df returns two of the same flooding data, and doesn't use the landslide data that is analyzed within the function.
df_haz,df_tikina = get_hazard_df(myCountry,economy,agg_or_occ='Agg',rm_overlap=True,special_event=special_event)
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
    df_haz['hh_share'] = (df_haz['value_destroyed_prv']/df_haz['value_destroyed']).fillna(1.)
    
    df_haz.to_csv('~/Desktop/hh_share.csv')
    #assert(False)
    # Weird things can happen for rp=2000 (negative losses), but they're < 10E-5, so we don't worry much about them
    #df_haz.loc[df_haz.hh_share>1.].to_csv('~/Desktop/hh_share.csv')

elif myCountry == 'FJ':
    df_haz = df_haz.reset_index().set_index([economy,'hazard','rp']).sum(level=[economy,'hazard','rp'])
    # All the magic happens inside get_hazard_df()

else: 
    print('\n\nSetting hh_share to 1!\n\n')
    df_haz['hh_share'] = 1.

# Turn losses into fraction
cat_info = cat_info.reset_index().set_index([economy])

# Available capital by economy:
# HIES stands for Household Income and Expenditure Survey
hazard_ratios = cat_info[['k','pcwgt']].prod(axis=1).sum(level=economy).to_frame(name='HIES_capital')
# Join on economy with hazards
hazard_ratios = hazard_ratios.join(df_haz,how='outer')


# Implemented only for Philippines, others return none.
hazard_ratios['grdp_to_assets'] = get_subnational_gdp_macro(myCountry,hazard_ratios,float(df['avg_prod_k'].mean()))


# fa is the exposure, or probability of a household being affected.
if myCountry == 'PH':
    hazard_ratios['frac_destroyed'] = hazard_ratios['value_destroyed']/hazard_ratios['grdp_to_assets']
    hazard_ratios = hazard_ratios.drop(['HIES_capital', 'value_destroyed','value_destroyed_prv','value_destroyed_pub'],axis=1)
    
elif (myCountry == 'FJ' 
      or myCountry == 'SL' 
      or myCountry == 'RO'
      or myCountry == 'BO'): pass
# For FJ:
# --> fa is losses/(exposed_value*v)
#hazard_ratios['frac_destroyed'] = hazard_ratios['fa'] 

# For SL and RO, 'fa' is fa, not frac_destroyed
# hazard_ratios['frac_destroyed'] = hazard_ratios.pop('fa')

# Have frac destroyed, need fa...
# Frac value destroyed = SUM_i(k*v*fa)

# Merge hazard_ratios with cat_info
hazard_ratios = pd.merge(hazard_ratios.reset_index(),cat_info.reset_index(),on=economy,how='outer')

# Reduce vulnerability by reduction_vul if hh has access to early warning:
hazard_ratios.loc[(hazard_ratios.hazard!='EQ')
                  &(hazard_ratios.hazard!='DR'),'v'] *= (1-reduction_vul*hazard_ratios.loc[(hazard_ratios.hazard!='EQ')
                                                                                           &(hazard_ratios.hazard!='DR'),'has_ew'])

# Add some randomness, but at different levels for different assumptions
# FLAG: This does not change the vulnerability by the same random factor across different intensities/rps of events[]

hazard_ratios.loc[hazard_ratios['v']<=0.1,'v'] *= np.random.uniform(.8,2,hazard_ratios.loc[hazard_ratios['v']<=0.1].shape[0])
hazard_ratios.loc[hazard_ratios['v'] >0.1,'v'] *= np.random.uniform(.8,1.2,hazard_ratios.loc[hazard_ratios['v'] >0.1].shape[0])

# Calculate frac_destroyed for SL, since we don't have that in this case
if myCountry == 'SL': hazard_ratios['frac_destroyed'] = hazard_ratios[['v','fa']].prod(axis=1)

# Calculate frac_destroyed for RO
if myCountry == 'RO': 
    # This is slightly tricky...
    # For RO, we're using different hazard inputs for EQ and PF, and that's why they're treated differently below
    # NB: these inputs come from the library lib_collect_hazard_data_RO (this script doesn't get called)
    hazard_ratios.loc[hazard_ratios.hazard=='EQ','frac_destroyed'] = hazard_ratios.loc[hazard_ratios.hazard=='EQ','fa'].copy()
    # ^ EQ hazard is based on "caploss", which is total losses expressed as fraction of total capital stock (currently using gross, but could be net?) 
    hazard_ratios.loc[hazard_ratios.hazard=='PF','frac_destroyed'] = hazard_ratios.loc[hazard_ratios.hazard=='PF',['v','fa']].prod(axis=1)
    # ^ PF hazard is based on "popaff", which is the affected population, and "affected" could be anything. So we're applying the vulnerability curve to this hazard.
    
if myCountry == 'BO': hazard_ratios['frac_destroyed'] = hazard_ratios[['v','fa']].prod(axis=1)


if 'hh_share' not in hazard_ratios.columns: hazard_ratios['hh_share'] = None
hazard_ratios = hazard_ratios.reset_index().set_index(event_level+['hhid'])[[i for i in ['frac_destroyed','v','v_ag','k','pcinc_ag_gross',
                                                                                         'pcwgt','hh_share','grdp_to_assets','fa'] if i in hazard_ratios.columns]]

try: hazard_ratios = hazard_ratios.drop('index',axis=1)
except: pass


###########################################
# 2 things going on here:
# 1) Pull v out of frac_destroyed
# 2) Transfer fa in excess of 95% to vulnerability
fa_threshold = 0.95
v_threshold = 0.95

# # look up hazard ratios for one particular houshold.
# idx = pd.IndexSlice
# hazard_ratios.loc[idx['Ampara', 'PF', :, '521471']]

# Calculate avg vulnerability at event level
# --> v_mean is weighted by capital & pc_weight
v_mean = (hazard_ratios[['pcwgt','k','v']].prod(axis=1).sum(level=event_level)/hazard_ratios[['pcwgt','k']].prod(axis=1).sum(level=event_level)).to_frame(name='v_mean')
try: v_mean['v_ag_mean'] = (hazard_ratios[['pcwgt','pcinc_ag_gross','v_ag']].prod(axis=1).sum(level=event_level)
                            /hazard_ratios[['pcwgt','pcinc_ag_gross']].prod(axis=1).sum(level=event_level))
except: pass

#
hazard_ratios = pd.merge(hazard_ratios.reset_index(),v_mean.reset_index(),on=event_level).reset_index().set_index(event_level+['hhid']).sort_index().drop('index',axis=1)
hazard_ratios_drought = None
#
if 'DR' in get_all_hazards(myCountry,hazard_ratios):
    #
    # separate drought from other hazards
    hazard_ratios_drought = hazard_ratios.loc[(slice(None),['DR'],slice(None),slice(None)),:].copy()
    hazard_ratios = hazard_ratios.loc[hazard_ratios.index.get_level_values('hazard')!='DR',:].drop('pcinc_ag_gross',axis=1)


if myCountry == 'MW':

    print(hazard_ratios.head())

    try: hazard_ratios['fa'] = hazard_ratios.eval('frac_destroyed/v_mean').fillna(0)
    except: pass

    if hazard_ratios_drought is not None:
        hazard_ratios_drought['fa_ag'] = hazard_ratios_drought.eval('frac_destroyed/v_ag_mean').fillna(0)
    #
    hazard_ratios.loc[hazard_ratios.fa>fa_threshold,'v'] = (hazard_ratios.loc[hazard_ratios.fa>fa_threshold,['v','fa']].prod(axis=1)/fa_threshold).clip(upper=0.95)
    hazard_ratios['fa'] = hazard_ratios['fa'].clip(lower=0,upper=fa_threshold)
    hazard_ratios = hazard_ratios.fillna(0)
    
    hazard_renorm = pd.DataFrame({'total_k':hazard_ratios[['k','pcwgt']].prod(axis=1),
                                  'exp_loss':hazard_ratios[['k','pcwgt','fa','v']].prod(axis=1)},index=hazard_ratios.index.copy()).sum(level=event_level)
    #
    if hazard_ratios_drought is not None:
        mz_frac_of_ag = get_ag_value(myCountry,fom='mz_frac_ag',yr=2016)
        #
        # Sanity check: what is total value of agricultural production in FAOSTAT vs. HIES?
        total_ag_fao = get_ag_value(myCountry,fom='ag_value',yr=2016)
        total_ag_ihs = 1E-6*float(cat_info[['pcinc_ag_gross','pcwgt']].prod(axis=1).sum())
        print(total_ag_fao,'mil. MWK = total value of ag in FAO')
        print(total_ag_ihs,'mil. MWK = total value of ag in IHS')
        print('IHS = '+str(round(1E2*(total_ag_ihs/total_ag_fao),1))+'% of FAO')
        #
        # Get expected losses from GAR (maize)
        ag_exposure_gross = float(pd.read_excel('../inputs/MW/GAR/GAR_PML_curve_MW.xlsx',sheet_name='total_exposed_val').loc['Malawi','Gross value, maize'].squeeze())
        print(ag_exposure_gross*730.,' value of maize in GAR')
        print(mz_frac_of_ag*total_ag_fao,' value of maize in FAO')
        #
        # total losses:
        print(hazard_ratios_drought.head())
        hazard_renorm_ag = pd.DataFrame({'total_ag_income':hazard_ratios_drought[['pcinc_ag_gross','pcwgt']].prod(axis=1),
                                         'exp_loss':hazard_ratios_drought[['pcinc_ag_gross','pcwgt','fa_ag','v_ag']].prod(axis=1)},
                                        index=hazard_ratios_drought.index.copy()).sum(level=event_level)
        #
        hazard_renorm_ag_aal,_ = average_over_rp(hazard_renorm_ag)
        hazard_renorm_ag_aal = hazard_renorm_ag_aal.sum(level='hazard')
        #
    if not special_event: 

        #########################
        # Calibrate AAL in MW  
        # - EQ (and FF? what's going on here?) 
        hazard_renorm_aal,_ = average_over_rp(hazard_renorm)
        hazard_renorm_aal = hazard_renorm_aal.sum(level='hazard')
        #
        GAR_eq_aal = 8.2*(1/get_currency(myCountry)[2])*1.E6
        #
        eq_scale_factor = GAR_eq_aal/float(hazard_renorm_aal.loc['EQ']['exp_loss'])
        #
        hazard_ratios['fa'] *= eq_scale_factor
        #
        # - DR/Drought 
        #drought_renorm_aal,_ = average_over_rp(hazard_renorm_drought)

    if special_event and special_event.lower() == 'idai':

        # check total losses:
        hazard_renorm_total_loss = (get_idai_loss().sum(axis=1).squeeze()*730.).to_frame(name='actual_losses')

        # numerator: hazard_renorm_total_loss
        # denominator: hazard_renorm
        hazard_renorm = pd.merge(hazard_renorm.reset_index(),hazard_renorm_total_loss.reset_index(),on='district',how='outer').fillna(0).set_index('district')
        hazard_renorm['scale_factor'] = hazard_renorm.eval('actual_losses/exp_loss')

        hazard_ratios = pd.merge(hazard_ratios.reset_index(),hazard_renorm.reset_index(),on=event_level).set_index(event_level+['hhid'])
        hazard_ratios['v'] *= hazard_ratios['scale_factor']

        # v can be greater than 1 here...if v > 0.99, transfer to fa        
        hazard_ratios.loc[hazard_ratios.v>v_threshold,'fa'] = (hazard_ratios.loc[hazard_ratios.v>v_threshold,['v','fa']].prod(axis=1)/v_threshold)#.clip(upper=fa_threshold)
        hazard_ratios['v'] = hazard_ratios['v'].clip(upper=v_threshold)

        v_mean = (hazard_ratios[['pcwgt','k','v']].prod(axis=1).sum(level=event_level)/hazard_ratios[['pcwgt','k']].prod(axis=1).sum(level=event_level)).to_frame(name='v_mean')
        hazard_ratios['frac_destroyed'] = hazard_ratios.eval('fa*v_mean')

        

if myCountry != 'SL' and myCountry != 'BO' and not special_event:
    # Normally, we pull fa out of frac_destroyed.
    # --> for SL, I think we have fa (not frac_destroyed) from HIES
    hazard_ratios['fa'] = (hazard_ratios['frac_destroyed']/hazard_ratios['v_mean']).fillna(0)

    hazard_ratios.loc[hazard_ratios.fa>fa_threshold,'v'] = (hazard_ratios.loc[hazard_ratios.fa>fa_threshold,['v','fa']].prod(axis=1)/fa_threshold).clip(upper=0.95)
    hazard_ratios['fa'] = hazard_ratios['fa'].clip(lower=0,upper=fa_threshold)
    hazard_ratios = hazard_ratios.fillna(0)

hazard_ratios = hazard_ratios.append(hazard_ratios_drought).fillna(0)
hazard_ratios[[_ for _ in ['fa','v_mean','fa_ag','v_ag_mean'] if _ in hazard_ratios.columns]].mean(level=event_level).to_csv('tmp/fa_v.csv')

# check
#hazard_renorm = pd.DataFrame({'total_k':hazard_ratios[['k','pcwgt']].prod(axis=1),
#                              'exp_loss':hazard_ratios[['k','pcwgt','fa','v']].prod(axis=1)},index=hazard_ratios.index.copy()).sum(level=event_level)
#hazard_renorm.to_csv('~/Desktop/tmp/out.csv')
#assert(False)


#####################################
# Get optimal reconstruction rate
_pi = float(df['avg_prod_k'].mean())
_rho = float(df['rho'].mean())

print('Running hh_reco_rate optimization')
hazard_ratios['hh_reco_rate'] = 0

if True:
    v_to_reco_rate = {}
    try:
        v_to_reco_rate = pickle.load(open('../optimization_libs/'+myCountry+('_'+special_event if special_event != None else '')+'_v_to_reco_rate.p','rb'))
        #pickle.dump(v_to_reco_rate, open('../optimization_libs/'+myCountry+'_v_to_reco_rate_proto2.p', 'wb'),protocol=2)
    except: print('Was not able to load v to hh_reco_rate library from ../optimization_libs/'+myCountry+'_v_to_reco_rate.p')

    hazard_ratios.loc[hazard_ratios.index.duplicated(keep=False)].to_csv('~/Desktop/tmp/dupes.csv')
    assert(hazard_ratios.loc[hazard_ratios.index.duplicated(keep=False)].shape[0]==0)

    hazard_ratios['hh_reco_rate'] = hazard_ratios.apply(lambda x:optimize_reco(v_to_reco_rate,_pi,_rho,x['v']),axis=1)
    try: 
        pickle.dump(v_to_reco_rate,open('../optimization_libs/'+myCountry+('_'+special_event if special_event != None else '')+'_v_to_reco_rate.p','wb'))
        print('gotcha')
    except: print('didnt getcha')

#except:
#    for _n, _i in enumerate(hazard_ratios.index):
#        
#        if round(_n/len(hazard_ratios.index)*100,3)%10 == 0:
#            print(round(_n/len(hazard_ratios.index)*100,2),'% through optimization')
#
#        _v = round(hazard_ratios.loc[_i,'v'].squeeze(),2)
#
#        #if _v not in v_to_reco_rate:
#        #    v_to_reco_rate[_v] = optimize_reco(_pi,_rho,_v)
#        #hazard_ratios.loc[_i,'hh_reco_rate'] = v_to_reco_rate[_v]
#
#        hazard_ratios.loc[_i,'hh_reco_rate'] = optimize_reco(_pi,_rho,_v)
#
#    try: pickle.dump(hazard_ratios[['_v','hh_reco_rate']].to_dict(),open('../optimization_libs/'+myCountry+'_v_to_reco_rate.p','wb'))
#    except: pass

# Set hh_reco_rate = 0 for drought
hazard_ratios.loc[hazard_ratios.index.get_level_values('hazard') == 'DR','hh_reco_rate'] = 0
# no drought recovery. lasts forever. eep.

cat_info = cat_info.reset_index().set_index([economy,'hhid'])

#cat_info['v'] = hazard_ratios.reset_index().set_index([economy,'hhid'])['v'].mean(level=[economy,'hhid']).clip(upper=0.99)
# ^ I think this is throwing off the losses!! Average vulnerability isn't going to cut it
# --> Use hazard-specific vulnerability for each hh (in hazard_ratios instead of in cats_event)

# This function collects info on the value and vulnerability of public assets
cat_info, hazard_ratios = get_asset_infos(myCountry,cat_info,hazard_ratios,df_haz)

df.to_csv(intermediate+'/macro'+('_'+special_event if special_event is not None else '')+'.csv',encoding='utf-8', header=True,index=True)

cat_info = cat_info.drop([icol for icol in ['level_0','index'] if icol in cat_info.columns],axis=1)

try: make_regional_inputs_summary_table(myCountry,cat_info.copy()) # this is for PH
except: pass

cat_info.to_csv(intermediate+'/cat_info'+('_'+special_event if special_event is not None else '')+'.csv',encoding='utf-8', header=True,index=True)


# If we have 2 sets of data on k, gdp, look at them now:
summary_df = pd.DataFrame({'FIES':df['avg_prod_k'].mean()*cat_info[['k','pcwgt']].prod(axis=1).sum(level=economy)/1E9})
try: summary_df['GRDP'] = df['avg_prod_k'].mean()*hazard_ratios['grdp_to_assets'].mean(level=economy)*1.E-9
except: pass
summary_df.loc['Total'] = summary_df.sum()

try: 
    summary_df['Ratio'] = 100.*summary_df.eval('FIES/GRDP')

    totals = summary_df[['FIES','GRDP']].sum().squeeze()
    ratio = totals[0]/totals[1]
    print(totals, ratio)
except: print('Dont have 2 datasets for GDP. Just using hh survey data.')
summary_df.round(1).to_latex('latex/'+myCountry+'/grdp_table.tex')
summary_df.to_csv(intermediate+'/gdp.csv')


##############
# Write out hazard ratios
hazard_ratios= hazard_ratios.drop(['frac_destroyed','grdp_to_assets'],axis=1).drop(["flood_fluv_def"],level="hazard")
hazard_ratios.dropna().to_csv(intermediate+'/hazard_ratios'+('_'+special_event if special_event is not None else '')+'.csv',encoding='utf-8', header=True)
