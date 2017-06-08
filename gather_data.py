# This script provides data input for the resilience indicator multihazard model for Philippines. 
# Restructured from the global model and developed by Jinqiang Chen and Brian Walsh

# Magic
from IPython import get_ipython
get_ipython().magic('reset -f')
get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')

# Import packages for data analysis
from lib_gather_data import *
from replace_with_warning import *
import numpy as np
import pandas as pd
from pandas import isnull
import os, time
import warnings
import sys
warnings.filterwarnings('always',category=UserWarning)

if len(sys.argv) < 2:
    print('Need to list country. Try PH')
    assert(False)
else: myCountry = sys.argv[1]

# Define directories
model         = os.getcwd()                             # get current directory
inputs        = model+'/../inputs/'+myCountry+'/'       # get inputs data directory
intermediate  = model+'/../intermediate/'+myCountry+'/' # get outputs data directory

# If the depository directories don't exist, create one:
if not os.path.exists(inputs): 
    print('You need to put the country survey files in a directory titled ','/inputs/'+myCountry+'/')
    assert(False)
if not os.path.exists(intermediate):
    os.makedirs(intermediate)

#Options and parameters
economy       = 'province'
event_level   = [economy, 'hazard', 'rp']                            # levels of index at which one event happens
affected_cats = pd.Index(['a', 'na'], name='affected_cat')	     # categories for social protection
helped_cats   = pd.Index(['helped','not_helped'], name='helped_cat')
#hazard_cats  = pd.Index(['flooding','storm_surge'], name='hazard')

reconstruction_time = 3.00 #time needed for reconstruction
reduction_vul       = 0.20 # how much early warning reduces vulnerability
inc_elast           = 1.50 # income elasticity
discount_rate       = 0.06 # discount rate
asset_loss_covered  = 0.80 # becomes 'shareable' 
max_support         = 0.05 # fraction of GDP

# GDP per cap info:
# should be national, or provincial?
# --> gdp_pc_pp = 8900 #GDP per capita in USD
# --> fx = 127.3 #foreign exchange ratio
PSA = pd.read_excel(inputs+'PSA_compiled.xlsx',skiprows=1)[['province','gdp_pc_pp','pop','shewp','shewr']].set_index('province')
PSA['gdp'] = PSA['gdp_pc_pp']*PSA['pop']

#Country dictionaries
#STATE/PROVINCE NAMES
df = pd.read_excel(inputs+'population_2015.xlsx',sheetname='population').set_index(economy)
prov_code = pd.read_excel(inputs+'FIES_provinces.xlsx')[['province_code','province_AIR']].set_index('province_code').squeeze() 

###Define parameters
df['pi']                     = reduction_vul       # how much early warning reduces vulnerability
df['rho']                    = discount_rate       # discount rate
df['pop']                    = df['population']    # Provincial population
df['shareable']              = asset_loss_covered  # target of asset losses to be covered by scale up
df['protection']             = 1                   # Protected from events with RP < 'protection'
df['avg_prod_k']             = 0.337960802589002   # average productivity of capital, value from the global resilience model
df['T_rebuild_K']            = reconstruction_time # Reconstruction time
df['income_elast']           = inc_elast           # income elasticity
df['max_increased_spending'] = max_support         # 5% of GDP in post-disaster support maximum, if everything is ready
df.drop(['population'],axis=1,inplace=True)

df['gdp_pc_pp']   = PSA['gdp']/df['pop'] #in Pesos
df['avg_hh_size'] = df['pop']/PSA['pop'] # nPeople/nHH

cat_info = pd.read_csv(inputs+'fies2015.csv',usecols=['w_regn','w_prov','w_mun','w_bgy','w_ea','w_shsn','w_hcn','walls','roof','totex','cash_abroad','cash_domestic','regft','hhwgt','fsize','poorhh'])
get_hhid_FIES(cat_info)

cat_info = cat_info.rename(columns={'w_prov':'province'})
cat_info = cat_info.reset_index().set_index([cat_info.province.replace(prov_code)]) #replace district code with its name
cat_info = cat_info.drop('province',axis=1)

# --> trying to get rid of provinces 97 & 98 here
#cat_info = cat_info.ix[cat_info.index.drop(['97','98'],level='province',errors='ignore')]

#Vulnerability
vul_curve = pd.read_excel(inputs+'vulnerability_curves_FIES.xlsx',sheetname='wall')[['desc','v']]
for thecat in vul_curve.desc.unique():
    cat_info.ix[cat_info.walls.values == thecat,'v'] = vul_curve.loc[vul_curve.desc.values == thecat].v.values

vul_curve = pd.read_excel(inputs+'vulnerability_curves_FIES.xlsx',sheetname='roof')[['desc','v']]
for thecat in vul_curve.desc.unique():
    cat_info.ix[cat_info.roof.values == thecat,'v'] += vul_curve.loc[vul_curve.desc.values == thecat].v.values
cat_info.v = cat_info.v/2

cat_info.ix[cat_info.v==0.1,'v'] *= np.random.uniform(.8,2,30187)
cat_info.ix[cat_info.v==0.25,'v'] *= np.random.uniform(.8,1.2,5895)  
cat_info.ix[(10*(cat_info.v-.4)).round()==0,'v'] *= np.random.uniform(.8,1.2,5130)
cat_info.ix[cat_info.v==0.55,'v'] *= np.random.uniform(.8,1.2,153)
cat_info.ix[cat_info.v==0.70,'v'] *= np.random.uniform(.8,1.2,179) 
cat_info.drop(['walls','roof'],axis=1,inplace=True)

# Calculate income per household
# --> What's the difference between income & consumption/disbursements?
# --> totdis = 'total family disbursements'; totex = 'total family expenditures'
# --> (SL model) cat_info['c'] = cat_info[['emp','agri','other_agri','non_agri','other_inc','income_local']].sum(1)
cat_info['c'] = cat_info['totex']

# Cash receipts, abroad & domestic, other gifts
# --> Excluding international remittances ('cash_abroad')
# --> what about 'net_receipt'?
cat_info['social'] = cat_info[['cash_domestic','regft']].sum(axis=1)/cat_info['totex']
cat_info.ix[cat_info.social>1,'social']=1
cat_info.drop(['cash_abroad','cash_domestic','regft'],axis=1,inplace=True)

# Weight = household_weight * family_size
cat_info['weight'] = cat_info[['hhwgt','fsize']].prod(axis=1)
print('Total population:',cat_info.weight.sum())
print('Total n households:',cat_info.hhwgt.sum())

# Change the name: district to code, and create an multi-level index 
cat_info = cat_info.rename(columns={'district':'code'})

# Assing weighted household consumption to quintiles within each province
listofquintiles=np.arange(0.20, 1.01, 0.20) 
cat_info = cat_info.reset_index().groupby('province',sort=True).apply(lambda x:match_quintiles(x,perc_with_spline(reshape_data(x.c),reshape_data(x.hhwgt),listofquintiles)))

# 'c_5' is the upper consumption limit for the lowest quintile
cat_info_c_5 = cat_info.reset_index().groupby('province',sort=True).apply(lambda x:x.ix[x.quintile==1,'c'].max())
cat_info = cat_info.reset_index().set_index(['province','hhid']) #change the name: district to code, and create an multi-level index 
cat_info['c_5'] = broadcast_simple(cat_info_c_5,cat_info.index)

# Population of household as fraction of population of province
cat_info['n'] = cat_info.weight/cat_info.weight.sum(level=economy)

# population of household as fraction of total population
cat_info['n_national'] = cat_info.weight/cat_info.weight.sum() 

# These sum to 1 per province & nationally, respectively
#print('province normalization:',cat_info.n.sum(level=economy)) 
print('normalization:',cat_info.n_national.sum())

# Get the tax used for domestic social transfer
# --> tau_tax = 0.075391
df['tau_tax'] = 1/((cat_info[['c','n_national']].prod(axis=1, skipna=False).sum())/(cat_info[['social','c','n_national']].prod(axis=1, skipna=False).sum())+1)

# Get the share of Social Protection
cat_info['gamma_SP'] = cat_info[['social','c']].prod(axis=1,skipna=False)/cat_info[['social','c','n_national']].prod(axis=1, skipna=False).sum()
cat_info.drop('n_national',axis=1,inplace=True)

cat_info['k'] = (1-cat_info['social'])*cat_info['c']/((1-df['tau_tax'])*df['avg_prod_k']) #calculate the capital
cat_info.ix[cat_info.k<0,'k'] = 0.0

# Getting rid of Prov_code 98, 99 here
cat_info.dropna(inplace=True)

# Assign access to early warning based on 'poorhh' flag
# --> doesn't seem to match up with the quintiles we assigned
cat_info['shew'] = broadcast_simple(PSA['shewr'],cat_info.index)
cat_info.ix[cat_info.poorhh == 1,'shew'] = broadcast_simple(PSA['shewp'],cat_info.index)

# Exposure
cat_info['fa'] = 0
cat_info.fillna(0,inplace=True)

# Cleanup dfs for writing out
cat_info = cat_info.drop([iXX for iXX in cat_info.columns.values.tolist() if iXX not in ['province','hhid','weight','code','np','flooding','score','v','c','social','c_5','n','gamma_SP','k','shew','fa','quintile','hhwgt']],axis=1)
cat_info_index = cat_info.drop([iXX for iXX in cat_info.columns.values.tolist() if iXX not in ['province','hhid']],axis=1)

#########################
# HAZARD INFO
#
# This is the GAR
#hazard_ratios = pd.read_csv(inputs+'/PHL_frac_value_destroyed_gar_completed_edit.csv').set_index(['province', 'hazard', 'rp'])

# This is the AIR dataset:
# df_AIR is already in pesos
# --> Need to think about public assets
#df_AIR = get_AIR_data(inputs+'/Risk_Profile_Master_With_Population.xlsx','Loss_Results','all','Agg')
df_AIR = get_AIR_data(inputs+'/Risk_Profile_Master_With_Population.xlsx','Loss_Results','Private','Agg').reset_index()
df_AIR.columns = ['province','hazard','rp','value_destroyed']

# Edit & Shuffle provinces
AIR_prov_rename = {'Shariff Kabunsuan':'Maguindanao',
                   'Davao Oriental':'Davao',
                   'Davao del Norte':'Davao',
                   'Metropolitan Manila':'Manila',
                   'Dinagat Islands':'Surigao del Norte'}
df_AIR['province'].replace(AIR_prov_rename,inplace=True) 

# Add NCR 2-4 to AIR dataset
df_NCR = pd.DataFrame(df_AIR.loc[(df_AIR.province == 'Manila')])
df_NCR['province'] = 'NCR-2nd Dist.'
df_AIR = df_AIR.append(df_NCR)

df_NCR['province'] = 'NCR-3rd Dist.'
df_AIR = df_AIR.append(df_NCR)

df_NCR['province'] = 'NCR-4th Dist.'
df_AIR = df_AIR.append(df_NCR)

# In AIR, we only have 'Metropolitan Manila'
# Distribute losses among Manila & NCR 2-4 according to assets
cat_info = cat_info.reset_index()
k_NCR = cat_info.loc[((cat_info.province == 'Manila') | (cat_info.province == 'NCR-2nd Dist.') 
                      | (cat_info.province == 'NCR-3rd Dist.') | (cat_info.province == 'NCR-4th Dist.')), ['k','hhwgt']].prod(axis=1).sum()

df_AIR.loc[df_AIR.province ==        'Manila','value_destroyed'] *= cat_info.loc[cat_info.province ==        'Manila', ['k','hhwgt']].prod(axis=1).sum()/k_NCR
df_AIR.loc[df_AIR.province == 'NCR-2nd Dist.','value_destroyed'] *= cat_info.loc[cat_info.province == 'NCR-2nd Dist.', ['k','hhwgt']].prod(axis=1).sum()/k_NCR
df_AIR.loc[df_AIR.province == 'NCR-3rd Dist.','value_destroyed'] *= cat_info.loc[cat_info.province == 'NCR-3rd Dist.', ['k','hhwgt']].prod(axis=1).sum()/k_NCR
df_AIR.loc[df_AIR.province == 'NCR-4th Dist.','value_destroyed'] *= cat_info.loc[cat_info.province == 'NCR-4th Dist.', ['k','hhwgt']].prod(axis=1).sum()/k_NCR

# Sum over the provinces that we're merging
# Losses are absolute value, so they are additive
df_AIR = df_AIR.reset_index().set_index(['province','hazard','rp']).sum(level=['province','hazard','rp']).drop(['index'],axis=1)

# Turn losses into fraction
cat_info = cat_info.reset_index().set_index(['province'])

hazard_ratios = cat_info[['k','hhwgt']].prod(axis=1).sum(level='province').to_frame(name='provincial_capital')
hazard_ratios = hazard_ratios.join(df_AIR,how='outer')

hazard_ratios['frac_destroyed'] = hazard_ratios['value_destroyed']/hazard_ratios['provincial_capital']
hazard_ratios = hazard_ratios.drop(['provincial_capital','value_destroyed'],axis=1)

# Have frac destroyed, need fa...
# Frac value destroyed = SUM_i(k*v*fa)

hazard_ratios = pd.merge(hazard_ratios.reset_index(),cat_info.reset_index(),on='province',how='outer').set_index(event_level+['hhid'])[['frac_destroyed','v']]

hazard_ratios['fa'] = (hazard_ratios['frac_destroyed']/hazard_ratios['v']).clip(upper=0.9)
hazard_ratios= hazard_ratios.drop(['frac_destroyed','v'],axis=1)

# Transfer fa in excess of 90% to vulnerability
#fa_threshold = 0.90
#excess=hazard_ratios[hazard_ratios>fa_threshold].max(level=['province'])
#for c in excess.index:
#    
#    r = (excess/fa_threshold)[c]
#    #print(c,r, fa_guessed_gar[fa_guessed_gar>fa_threshold].ix[c])
#    hazard_ratios.update(hazard_ratios.ix[[c]]/r)  # i don't care.
#    
#    cat_info.ix[c,'v'] *= r
#    #vr.ix[c] *= r
#    #v.ix[c] *=r#
#cat_info['v'] = cat_info.v.clip(upper=.99)

df.to_csv(intermediate+'/macro.csv',encoding='utf-8', header=True,index=True)

cat_info = cat_info.drop(['index'],axis=1)
cat_info.to_csv(intermediate+'/cat_info.csv',encoding='utf-8', header=True,index=True)

hazard_ratios.to_csv(intermediate+'/hazard_ratios.csv',encoding='utf-8', header=True)
