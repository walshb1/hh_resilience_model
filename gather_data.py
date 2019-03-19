# This script provides data input for the resilience indicator multihazard model for the Philippines, Fiji, Sri Lanka, and (eventually) Malawi.
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
from libraries.lib_sp_analysis import run_sp_analysis
warnings.filterwarnings('always',category=UserWarning)

#####################################
# Which country are we running over?
# --> This variable <sys.argv> is the first optional argument on the command line
# --> If you're not using command line and/or don't know how to pass an argument, set <myCountry> in the "else:" loop equal to your country:
#
# -- 1) BO = Bolivia
# or 2) FJ = Fiji
# or 3) MW = Malawi
# or 4) PH = Philippines
# or 5) SL = Sri Lanka
#
if len(sys.argv) >= 2: myCountry = sys.argv[1]
else:
    myCountry = 'BO'
    print('Setting country to BO. Currently implemented: MW, PH, FJ, SL, BO')
#####################################

# Primary library ued is lib_country_dir.py

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
print('Survey population:',cat_info.pcwgt.sum())

#  below is messy--should be in <load_survey_data>
if myCountry == 'PH':

    # Standardize province info
    get_hhid_FIES(cat_info)
    cat_info = cat_info.rename(columns={'w_prov':'province','w_regn':'region'}).reset_index()
    cat_info['province'].replace(prov_code,inplace=True)
    cat_info['region'].replace(region_code,inplace=True)
    cat_info = cat_info.reset_index().set_index(economy).drop(['index','level_0'],axis=1)

    # There's no region info in df--put that in...
    df = df.reset_index().set_index('province')
    cat_info = cat_info.reset_index().set_index('province')
    df['region'] = cat_info[~cat_info.index.duplicated(keep='first')].region

    try: df.reset_index()[['province','region']].to_csv('../inputs/PH/prov_to_reg_dict.csv',header=True)
    except: print('Could not update regional-provincial dict')

    # Manipulate PSA (non-FIES) dataframe
    df = df.reset_index().set_index(economy)
    df['psa_pop'] = df.sum(level=economy)
    df = df.mean(level=economy)

cat_info = cat_info.reset_index().set_index([economy,'hhid'])

try: cat_info = cat_info.drop('index',axis=1)
except: pass
# Now we have a dataframe called <cat_info> with the household info.
# Index = [economy (='region';'district'; country-dependent), hhid]



########################################
# Calculate regional averages from household info
# per capita income (in local currency), regional average
df['gdp_pc_prov'] = cat_info[['pcinc','pcwgt']].prod(axis=1).sum(level=economy)/cat_info['pcwgt'].sum(level=economy)
# this is per capita income (local currency), national average
df['gdp_pc_nat'] = cat_info[['pcinc','pcwgt']].prod(axis=1).sum()/cat_info['pcwgt'].sum()

# regional population
df['pop'] = cat_info.pcwgt.sum(level=economy)

# (Philippines specific) compare PSA population to FIES population
try: df['pct_diff'] = 100.*(df['psa_pop']-df['pop'])/df['pop']
except: pass

########################################
########################################
# Asset vulnerability
print('Getting vulnerabilities')

vul_curve = get_vul_curve(myCountry,'wall').set_index('desc').to_dict()

cat_info['v'] = cat_info.walls.map(vul_curve['v'])

# vul_curve = get_vul_curve(myCountry,'wall')
# for thecat in vul_curve.desc.unique():
#     hh_private_asset_vulnerability = float(vul_curve.loc[vul_curve.desc.values == thecat,'v'])
#     cat_info.loc[cat_info['walls'] == thecat,'v'] = hh_private_asset_vulnerability
#     # Fiji doesn't have info on roofing, but it does have info on the *condition* of outer walls. Include that as a multiplier?
#

# Get roofing data (but Fiji doesn't have this info)
# Set home vulnerability to mean of the two vulnerabilities.
try:
    print('Getting roof info')
    vul_curve = get_vul_curve(myCountry,'roof')
    for thecat in vul_curve.desc.unique():
        cat_info.loc[cat_info.roof.values == thecat,'v'] += vul_curve.loc[vul_curve.desc.values == thecat].v.values
    cat_info.v = cat_info.v/2
except: pass

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
    print('\n--> Rural poverty (flagged poor):',float(round(cat_info.loc[(cat_info.ispoor==1)&(cat_info.urban=='RURAL'),'pcwgt'].sum()/1E6,3)),'million')
    print('\n--> Urban poverty (flagged poor):',float(round(cat_info.loc[(cat_info.ispoor==1)&(cat_info.urban=='URBAN'),'pcwgt'].sum()/1E6,3)),'million')
except: pass


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



#########################
# Replace any codes with names
if myCountry == 'FJ' or myCountry == 'RO' or myCountry == 'SL':
    #flag

    df = df.reset_index()
    if myCountry == 'FJ' or myCountry == 'SL':
        df[economy] = df[economy].replace(prov_code)
    if myCountry == 'RO':
        df[economy] = df[economy].astype('int').replace(region_code)

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
    df[economy] = df[economy].astype(int).replace(prov_code)
    cat_info = cat_info.reset_index()
    cat_info[economy].replace(prov_code,inplace=True) # replace division code with its name
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
cat_info.dropna(inplace=True,how='any')
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
cat_info_col = [economy,'province','hhid','region','pcwgt','aewgt','hhwgt','np','score','v','c','pcsoc','social','c_5','hhsize','ethnicity',
                'hhsize_ae','gamma_SP','k','quintile','ispoor','pcinc','aeinc','pcexp','pov_line','SP_FAP','SP_CPP','SP_SPS','nOlds',
                'has_ew','SP_PBS','SP_FNPF','SPP_core','SPP_add','axfin','pcsamurdhi','gsp_samurdhi','frac_remittance','N_children']
cat_info = cat_info.drop([i for i in cat_info.columns if (i in cat_info.columns and i not in cat_info_col)],axis=1)
cat_info_index = cat_info.drop([i for i in cat_info.columns if i not in [economy,'hhid']],axis=1)


#########################
# HAZARD INFO

# SL FLAG: get_hazard_df returns two of the same flooding data, and doesn't use the landslide data that is analyzed within the function.
df_haz,df_tikina = get_hazard_df(myCountry,economy,agg_or_occ='Agg',rm_overlap=True)
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

elif myCountry == 'FJ': pass
# --> fa is losses/(exposed_value*v)
#hazard_ratios['frac_destroyed'] = hazard_ratios['fa']

elif myCountry == 'SL': pass
# For SL, 'fa' is fa, not frac_destroyed
# hazard_ratios['frac_destroyed'] = hazard_ratios.pop('fa')

elif myCountry == 'BO':
    pass


# Have frac destroyed, need fa...
# Frac value destroyed = SUM_i(k*v*fa)

# Merge hazard_ratios with cat_info

hazard_ratios = pd.merge(hazard_ratios.reset_index(),cat_info.reset_index(),on=economy,how='outer')

# Reduce vulnerability by reduction_vul if hh has access to early warning:
hazard_ratios.loc[hazard_ratios.hazard!='EQ','v'] *= (1-reduction_vul*hazard_ratios.loc[hazard_ratios.hazard!='EQ','has_ew'])
# Add some randomness, but at different levels for different assumptions
# FLAG: This does not change the vulnerability by the same random factor across different intensities/rps of events[]
hazard_ratios.loc[hazard_ratios['v']<=0.1,'v'] *= np.random.uniform(.8,2,hazard_ratios.loc[hazard_ratios['v']<=0.1].shape[0])
hazard_ratios.loc[hazard_ratios['v'] >0.1,'v'] *= np.random.uniform(.8,1.2,hazard_ratios.loc[hazard_ratios['v'] >0.1].shape[0])

# Calculate frac_destroyed for SL, since we don't have that in this case
# frac_destroyed=exposure*vulnerability
if myCountry == 'SL': hazard_ratios['frac_destroyed'] = hazard_ratios[['v','fa']].prod(axis=1)
# We could calculate frac_destroyed for BO, but following SL for now.
if myCountry == 'BO': hazard_ratios['frac_destroyed'] = hazard_ratios[['v','fa']].prod(axis=1)

# cleanup
if 'hh_share' not in hazard_ratios.columns: hazard_ratios['hh_share'] = None
hazard_ratios = hazard_ratios.reset_index().set_index(event_level+['hhid'])[[i for i in ['frac_destroyed','v','k','pcwgt','hh_share','grdp_to_assets','fa'] if i in hazard_ratios.columns]]
hazard_ratios = hazard_ratios.drop([i for i in ['index'] if i in hazard_ratios.columns])



###########################################
# 2 things going on here:
# 1) Pull v out of frac_destroyed
# 2) Transfer fa in excess of 95% to vulnerability
fa_threshold = 0.95

# # look up hazard ratios for one particular houshold.
# idx = pd.IndexSlice
# hazard_ratios.loc[idx['Ampara', 'PF', :, '521471']]

# Calculate avg vulnerability at event level
# --> v_mean is weighted by capital & pc_weight
v_mean = (hazard_ratios[['pcwgt','k','v']].prod(axis=1).sum(level=event_level)/hazard_ratios[['pcwgt','k']].prod(axis=1).sum(level=event_level)).to_frame(name='v_mean')
#v_mean.name = 'v_mean'
hazard_ratios = pd.merge(hazard_ratios.reset_index(),v_mean.reset_index(),on=event_level).reset_index().set_index(event_level+['hhid']).sort_index().drop('index',axis=1)


if myCountry != 'SL' and myCountry != 'BO':

    # Normally, we pull fa out of frac_destroyed.
    # --> for SL, I think we have fa (not frac_destroyed) from HIES
    hazard_ratios['fa'] = (hazard_ratios['frac_destroyed']/hazard_ratios['v_mean']).fillna(1E-8)

    hazard_ratios.loc[hazard_ratios.fa>fa_threshold,'v'] = (hazard_ratios.loc[hazard_ratios.fa>fa_threshold,['v','fa']].prod(axis=1)/fa_threshold).clip(upper=0.95)
    hazard_ratios['fa'] = hazard_ratios['fa'].clip(lower=1E-8,upper=fa_threshold)


if myCountry == 'MW':

    hazard_renorm = pd.DataFrame({'total_k':hazard_ratios[['k','pcwgt']].prod(axis=1),
                                  'exp_loss':hazard_ratios[['k','pcwgt','fa','v']].prod(axis=1)},index=hazard_ratios.index.copy()).sum(level=event_level)
    #
    hazard_renorm_aal,_ = average_over_rp(hazard_renorm)
    hazard_renorm_aal = hazard_renorm_aal.sum(level='hazard')
    #
    GAR_eq_aal = 8.2*(1/get_currency(myCountry)[2])*1.E6
    #
    eq_scale_factor = GAR_eq_aal/float(hazard_renorm_aal.loc['EQ']['exp_loss'])
    #
    hazard_ratios['fa'] *= eq_scale_factor

hazard_ratios[['fa','v']].mean(level=event_level).to_csv('tmp/fa_v.csv')


# Get optimal reconstruction rate
_pi = float(df['avg_prod_k'].mean())
_rho = float(df['rho'].mean())

print('Running hh_reco_rate optimization')
hazard_ratios['hh_reco_rate'] = 0

v_to_reco_rate = {}
try:
    v_to_reco_rate = pickle.load(open('../optimization_libs/'+myCountry+'_v_to_reco_rate.p','rb'))
    #pickle.dump(v_to_reco_rate, open('../optimization_libs/'+myCountry+'_v_to_reco_rate_proto2.p', 'wb'),protocol=2)
except: print('Was not able to load v to hh_reco_rate library from ../optimization_libs/'+myCountry+'_v_to_reco_rate.p')


#hazard_ratios.loc[hazard_ratios.index.duplicated(keep=False)].to_csv('~/Desktop/tmp/dupes.csv')
assert(hazard_ratios.loc[hazard_ratios.index.duplicated(keep=False)].shape[0]==0)

hazard_ratios['hh_reco_rate'] = hazard_ratios.apply(lambda x:optimize_reco(v_to_reco_rate,_pi,_rho,x['v']),axis=1)
try:
    pickle.dump(v_to_reco_rate,open('../optimization_libs/'+myCountry+'_v_to_reco_rate.p','wb'))
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

if myCountry == 'PH':
    _path = '/Users/brian/Desktop/Dropbox/Bank/unbreakable_writeup/Figures/'
    _ = hazard_ratios.reset_index().copy()

    plot_simple_hist(_.loc[(_.hazard=='PF')&(_.rp==10)],['v'],[''],_path+'vulnerabilities_log.pdf',uclip=1,nBins=25,xlab='Asset vulnerability ($v_h$)',logy=True)
    plot_simple_hist(_.loc[(_.hazard=='PF')&(_.rp==10)],['v'],[''],_path+'vulnerabilities.pdf',uclip=1,nBins=25,xlab='Asset vulnerability ($v_h$)',logy=False)

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

print(summary_df.round(1))
summary_df.round(1).to_latex('latex/'+myCountry+'/grdp_table.tex')
summary_df.to_csv(intermediate+'/gdp.csv')

hazard_ratios= hazard_ratios.drop(['frac_destroyed','grdp_to_assets'],axis=1).drop(["flood_fluv_def"],level="hazard")
hazard_ratios.to_csv(intermediate+'/hazard_ratios.csv',encoding='utf-8', header=True)


# Compare assets from survey to assets from AIR-PCRAFI
if myCountry == 'FJ':

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

    fig = plt.gcf()
    fig.savefig('HIES_vs_PCRAFI_assets.pdf',format='pdf')
