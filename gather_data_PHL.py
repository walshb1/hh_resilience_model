#This script provides data input for the resilience indicator multihazard model for Sri Lanka. Restructured from the global model and developed by Jinqiang Chen.
from IPython import get_ipython
get_ipython().magic('reset -f')
get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')

#Import package for data analysis
from lib_gather_data import *
from replace_with_warning import *
import numpy as np
import pandas as pd
from pandas import isnull
import os, time
import warnings
warnings.filterwarnings("always",category=UserWarning)


#Options and parameters
economy="province" #province, deparmtent
event_level = [economy, "hazard", "rp"]	#levels of index at which one event happens
affected_cats = pd.Index(["a", "na"]            ,name="affected_cat")	#categories for social protection
helped_cats   = pd.Index(["helped","not_helped"],name="helped_cat")
hazard_cats = pd.Index(["flooding","storm_surge"],name="hazard")

reconstruction_time=3.0 #time needed for reconstruction
reduction_vul=0.2 # how much early warning reduces vulnerability
inc_elast=1.5 #income elasticity
discount_rate=0.06 #discount rate
asset_loss_covered=0.8 
max_support=0.05
gdp_pc_pp = 8900 #GDP per capita in USD
fx = 127.3 #foreign exchange ratio

#define directory
model             = os.getcwd() #get current directory
inputs     = model+'/PH_inputs/' #get inputs data directory
intermediate = model+'/PH_intermediate/' #get outputs data directory
if not os.path.exists(intermediate): #if the depository directory doesn't exist, create one
    os.makedirs(intermediate)

#Country dictionaries
#STATE/PROVINCE NAMES
df = pd.read_excel(inputs+"population_2015.xlsx",sheetname='population').set_index(economy)

prov_code = pd.read_excel(inputs+"FIES_provinces.xlsx")[['province_code','province_AIR']].set_index('province_code').squeeze() 

###Define parameters
df["T_rebuild_K"] = reconstruction_time #Reconstruction time
df["pi"] = reduction_vul	# how much early warning reduces vulnerability
df["income_elast"] = inc_elast	#income elasticity
df["rho"] = discount_rate	#discount rate
df["shareable"] = asset_loss_covered  #target of asset losses to be covered by scale up
df["max_increased_spending"] = max_support # 5% of GDP in post-disaster support maximum, if everything is ready
df["avg_prod_k"] = 0.337960802589002 #average productivity of capital, value from the global resilience model
df["gdp_pc_pp"] = gdp_pc_pp*fx #in Sri Lanka Rupee
df["pop"] = df['population'] #average population
df["protection"] = 5
df.drop(['population'],axis=1,inplace=True)

cat_info = pd.read_csv(inputs+"fies2015.csv")
get_hhid_FIES(cat_info)

cat_info = cat_info.rename(columns={'w_prov':'province'})
cat_info = cat_info.reset_index().set_index([cat_info.province.replace(prov_code)]) #replace district code with its name
cat_info = cat_info.drop('province',axis=1)

# --> trying to get rid of provinces 97 & 98 here
#cat_info = cat_info.ix[cat_info.index.drop(['97','98'],level='province',errors='ignore')]

print('OK1')

try:
    pd.read_csv(inputs+"pmt_2012_hh_model1_score.csv")
    pmt = pd.read_csv(inputs+"pmt_2012_hh_model1_score.csv").set_index('hhid')
    cat_info[['score','rpccons']] = pmt[['score','rpccons']]
except: pass


#Vulnerability
vul_curve = pd.read_excel(inputs+"vulnerability_curves_FIES.xlsx",sheetname='wall')[["desc","v"]]
for thecat in vul_curve.desc.unique():
    cat_info.ix[cat_info.walls.values == thecat,'v'] = vul_curve.loc[vul_curve.desc.values == thecat].v.values
vul_curve = pd.read_excel(inputs+"vulnerability_curves_FIES.xlsx",sheetname='roof')[["desc","v"]]
for thecat in vul_curve.desc.unique():
    cat_info.ix[cat_info.roof.values == thecat,'v'] += vul_curve.loc[vul_curve.desc.values == thecat].v.values
cat_info.v = cat_info.v/2

cat_info.ix[cat_info.v==0.1,'v'] *= np.random.uniform(.8,2,30187)
cat_info.ix[cat_info.v==0.25,'v'] *= np.random.uniform(.8,1.2,5895)  
cat_info.ix[(10*(cat_info.v-.4)).round()==0,'v'] *= np.random.uniform(.8,1.2,5130)
cat_info.ix[cat_info.v==0.55,'v'] *= np.random.uniform(.8,1.2,153)
cat_info.ix[cat_info.v==0.70,'v'] *= np.random.uniform(.8,1.2,179) 
cat_info.drop(['walls','roof'],axis=1,inplace=True)

#cat_info['c'] = cat_info[['emp','agri','other_agri','non_agri','other_inc','income_local']].sum(1) #calculate income per household

#--> is this income or consumption/disbursements?
cat_info['c'] = cat_info['totex']# totdis = 'total family disbursements'; totex = 'total family expenditures'

# cash receipts, abroad & domestic, other gifts
# --> what about net_receipt?
cat_info['social'] = cat_info[['cash_abroad','cash_domestic','regft']].sum(axis=1)/cat_info['totex']
cat_info.ix[cat_info.social>1,'social']=1
cat_info.drop(['cash_abroad','cash_domestic','regft'],axis=1,inplace=True)

#cat_info.drop(['emp','agri','other_agri','non_agri','other_inc','income_forign','income_local'],axis=1,inplace=True)
#cat_info = cat_info[cat_info.c>0]
#cat_info = cat_info[cat_info.c!=0]
#cat_info.c.dropna(inplace=True)

cat_info['weight'] = cat_info[['hhwgt','fsize']].prod(axis=1)
print(cat_info.weight.sum())

# listofdeciles=np.arange(0.05, 1.01, 0.05) #create a list of deciles 
# cat_info = match_deciles(cat_info,perc_with_spline(reshape_data(cat_info.c),reshape_data(cat_info.weight),listofdeciles))
# cat_info['c_5_nat'] = cat_info.ix[cat_info.decile==1,'c'].max()
# cat_info.drop('decile',axis=1,inplace=True)
# listofdeciles=np.arange(0.2, 1.1, 0.2) #create a list of deciles 
# cat_info = match_deciles(cat_info,perc_with_spline(reshape_data(cat_info.c),reshape_data(cat_info.weight),listofdeciles))
# cat_info = cat_info.rename(columns={'decile':'decile_nat'}).reset_index()

cat_info = cat_info.rename(columns={'district':'code'})#change the name: district to code, and create an multi-level index 

listofdeciles=np.arange(0.05, 1.01, 0.05) #create a list of deciles 
cat_info = cat_info.reset_index().groupby('province',sort=True).apply(lambda x:match_deciles(x,perc_with_spline(reshape_data(x.c),reshape_data(x.weight),listofdeciles)))
cat_info_c_5 = cat_info.reset_index().groupby('province',sort=True).apply(lambda x:x.ix[x.decile==1,'c'].max())
cat_info = cat_info.reset_index().set_index(['province','hhid']) #change the name: district to code, and create an multi-level index 
cat_info['c_5'] = broadcast_simple(cat_info_c_5,cat_info.index)

cat_info.drop('decile',axis=1,inplace=True)

cat_info = cat_info.reset_index().set_index(['province','hhid'])

# listofdeciles=np.arange(0.2, 1.1, 0.2) #create a list of deciles 
# cat_info = cat_info.groupby('district',sort=True).apply(lambda x:match_deciles_score(x,perc_with_spline(reshape_data(x.score),reshape_data(x.weight),listofdeciles)))
# cat_info = cat_info.set_index(['district','hhid']) #change the name: district to code, and create an multi-level index 

# cat_info = cat_info.reset_index()
# listofdeciles=np.arange(0.2, 1.1, 0.2) #create a list of deciles 
# cat_info = cat_info.groupby('district',sort=True).apply(lambda x:match_deciles(x,perc_with_spline(reshape_data(x.c),reshape_data(x.weight),listofdeciles)))
# cat_info = cat_info.reset_index().set_index('district',append=True) #change the name: district to code, and create an multi-level index 

cat_info['n'] = cat_info.weight/cat_info.weight.sum(level=economy) #get the fraction of a household in each district
cat_info["n_national"] = cat_info.weight/cat_info.weight.sum()
print(cat_info.n_national.sum())
#cat_info.drop('weight',axis=1,inplace=True)

#get the tax used for domestic social transfer
df["tau_tax"] = 1/((cat_info[['c','n_national']].prod(axis=1, skipna=False).sum())/(cat_info[['social','c','n_national']].prod(axis=1, skipna=False).sum())+1) 

cat_info["gamma_SP"] = cat_info[["social","c"]].prod(axis=1,skipna=False)/cat_info[["social","c","n_national"]].prod(axis=1, skipna=False).sum()
cat_info.drop("n_national",axis=1,inplace=True)

#print(cat_info.head(5))
print(df.head(5))


cat_info["k"] = (1-cat_info["social"])*cat_info["c"]/((1-df["tau_tax"])*df["avg_prod_k"]) #calculate the capital
cat_info.ix[cat_info.k<0,"k"] = 0.0

# --> need to pull in regional data for this
cat_info["shew"] = 0.64 #access to early warning system from global model


#Exposure
cat_info['fa'] = 0
# --> where is FIES flooding data?
#cat_info.ix[cat_info.flooding == 1,"fa"] = 1
cat_info.fillna(0,inplace=True)


#hazard
cat_info = cat_info.reset_index().set_index(['province','hhid'])
hazard_ratios = pd.read_csv(inputs+'/PHL_frac_value_destroyed_gar_completed_edit.csv').set_index(['province', 'hazard', 'rp'])

#hazard_ratios = pd.read_excel(inputs+"hazards_data.xlsx",sheetname='hazard').set_index(["province", "hazard", "rp"])
#hazard_ratios['shew'] = 0.64

#print(hazard_ratios.index.names,'\n\n')
#hazard_ratios.index.names = ['province', 'hazard', 'rp']
#print(cat_info.index.names,'\n\n')
#cat_info.index.name = 'province'
#print(hazard_ratios.head(10),'\n\n')
#print(cat_info.head(10),'\n\n')

#result = left.join(right, how='inner')
#result = pd.merge(left.reset_index(), right.reset_index(),
#   ....:                   on=['key'], how='inner').set_index(['key','X','Y'])

cat_info = cat_info.drop([iXX for iXX in cat_info.columns.values.tolist() if iXX not in ['province','hhid','index','weight','code','np','flooding','score','v','c','social','c_5','n','gamma_SP','k','shew','fa']],axis=1)
cat_info_index = cat_info.drop([iXX for iXX in cat_info.columns.values.tolist() if iXX not in ['province','hhid']],axis=1)

hazard_ratios_s = pd.merge(hazard_ratios.reset_index(),cat_info_index.reset_index(),on=['province'],how='inner').set_index(['province','hazard','rp','hhid'])
#hazard_ratios_s = broadcast_simple(hazard_ratios.squeeze(),index=cat_info.head(500).index)

print(hazard_ratios_s.index.names,'\n\n')

df.to_csv(intermediate+"/macro.csv",encoding="utf-8", header=True,index=True)
cat_info.to_csv(intermediate+"/cat_info.csv",encoding="utf-8", header=True,index=True)
hazard_ratios_s.to_csv(intermediate+"/hazard_ratios.csv",encoding="utf-8", header=True)
