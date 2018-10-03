##################################
#Import packages for data analysis
import matplotlib
matplotlib.use('AGG')

from libraries.lib_compute_resilience_and_risk import get_weighted_mean
from libraries.lib_poverty_tables_and_maps import run_poverty_duration_plot, run_poverty_tables_and_maps, map_recovery_time
from libraries.replace_with_warning import *
from libraries.lib_country_dir import *
from libraries.lib_common_plotting_functions import *
from libraries.maps_lib import *

from libraries.lib_plot_income_and_consumption_distributions import plot_income_and_consumption_distributions
from libraries.lib_plot_impact_by_quintile import plot_impact_by_quintile, plot_relative_losses
from libraries.lib_average_over_rp import *

from scipy.stats import norm
import matplotlib.mlab as mlab

from multiprocessing import Pool
from itertools import repeat,product

from pandas import isnull
import pandas as pd
import numpy as np
import os, time
import sys

font = {'family' : 'sans serif',
    'size'   : 15}
plt.rc('font', **font)
mpl.rcParams['xtick.labelsize'] = 12
mpl.rcParams['ytick.labelsize'] = 12
mpl.rcParams['legend.facecolor'] = 'white'

import warnings
warnings.filterwarnings('always',category=UserWarning)

myCountry = 'SL'
if len(sys.argv) >= 2: myCountry = sys.argv[1]
print('Running '+myCountry)

##################################
economy = get_economic_unit(myCountry)
event_level = [economy, 'hazard', 'rp']

haz_dict = {'SS':'Storm surge',
            'PF':'Precipitation flood',
            'HU':'Hurricane',
            'EQ':'Earthquake',
            'DR':'Drought',
            'FF':'Fluvial flood'}

to_usd = get_currency(myCountry)[2]


##################################
# Set directories (where to look for files)
out_files = os.getcwd()+'/../output_country/'+myCountry+'/'

##################################
# Set policy params

base_str = 'no'

path = os.getcwd()+'/../output_country/'+myCountry+'/'
pattern = 'sp_costs_*.csv'

pds_sims = []
for f in glob.glob(path+pattern):
    if f != base_str:
        pds_sims.append(f.replace(path,'').replace('sp_costs_','').replace('.csv',''))


drm_pov_sign = -1 # toggle subtraction or addition of dK to affected people's incomes
all_policies = []#['_exp095','_exr095','_ew100','_vul070','_vul070r','_rec067']



##################################
# Load base and PDS files
iah_base = pd.read_csv(out_files+'iah_tax_'+base_str+'_.csv', index_col=[economy,'hazard','rp','hhid'])
df_base = pd.read_csv(out_files+'results_tax_'+base_str+'_.csv', index_col=[economy,'hazard','rp'])

# DW costs of risk sharing in noPDS scenario
public_costs = pd.read_csv(out_files+'public_costs_tax_'+base_str+'_.csv').set_index([economy,'hazard','rp'])
public_costs['dw_tot_curr'] = public_costs[['dw_pub','dw_soc']].sum(axis=1)/df_base.wprime.mean()
public_costs_sum = public_costs.loc[public_costs['contributer']!=public_costs.index.get_level_values(event_level[0]),['dw_tot_curr']].sum(level=[economy,'hazard','rp'])
#

iah = pd.read_csv(out_files+'iah_tax_unif_poor_.csv', index_col=[economy,'hazard','rp','hhid','affected_cat','helped_cat']).sort_index()
df = pd.read_csv(out_files+'results_tax_unif_poor_.csv', index_col=[economy,'hazard','rp'])
macro = pd.read_csv(out_files+'macro_tax_unif_poor_.csv', index_col=[economy,'hazard','rp'])

# PDS
pds_effects_out = pd.DataFrame(index=pd.read_csv(out_files+'sp_costs_no.csv').set_index([economy,'hazard','rp']).index)
for _pds in ['no']+pds_sims:

    ####
    hh_summary = pd.read_csv(out_files+'my_summary_'+_pds+'.csv').set_index([economy,'hazard','rp'])

    ####
    pds_effects = pd.read_csv(out_files+'sp_costs_'+_pds+'.csv').set_index([economy,'hazard','rp'])

    # DW costs of risk sharing in each SP scenario
    public_costs_pds = pd.read_csv(out_files+'public_costs_tax_'+_pds+'_.csv').set_index([economy,'hazard','rp'])
    public_costs_pds['dw_tot_curr'] = 1E-6*public_costs_pds[['dw_pub','dw_soc']].sum(axis=1)/df.wprime.mean()
    public_costs_pds_sum = public_costs_pds.loc[public_costs_pds['contributer']!=public_costs_pds.index.get_level_values(event_level[0]),['dw_tot_curr']].sum(level=[economy,'hazard','rp'])
    #
    pds_effects_out['dw_'+_pds] = hh_summary['dw_tot']+public_costs_pds_sum['dw_tot_curr']
    #
    if _pds != 'no': 
        pds_effects_out['dw_DELTA_'+_pds] = pds_effects_out['dw_no'] - pds_effects_out['dw_'+_pds]
        pds_effects_out['ROI_event_'+_pds] = pds_effects_out['dw_DELTA_'+_pds]/(1E-6*pds_effects['event_cost'])

pds_effects_out.to_csv(out_files+'pds_effects.csv')

if False:
    for iPol in all_policies:
        iah_pol = pd.read_csv(out_files+'iah_tax_'+pds1_str+'_'+iPol+'.csv', index_col=[economy,'hazard','rp','hhid'])
        df_pol  = pd.read_csv(out_files+'results_tax_'+pds1_str+'_'+iPol+'.csv', index_col=[economy,'hazard','rp'])

        iah['dk0'+iPol] = iah_pol[['dk0','pcwgt']].prod(axis=1)
        iah['dw'+iPol] = iah_pol[['dw','pcwgt']].prod(axis=1)/df_pol.wprime.mean()

        print(iPol,'added to iah (these policies are run *with* PDS)')
        
        del iah_pol
        del df_pol
        gc.collect()

##################################
# SAVE OUT SOME RESULTS FILES
df_prov = df[['dKtot','dWtot_currency']].copy()
df_prov['gdp'] = df[['pop','gdp_pc_prov']].prod(axis=1).copy()
results_df = macro.reset_index().set_index([economy,'hazard'])
results_df = results_df.loc[results_df.rp==100,'dk_event'].sum(level='hazard')
results_df = results_df.rename(columns={'dk_event':'dk_event_100'})
results_df = pd.concat([results_df,df_prov.reset_index().set_index([economy,'hazard']).sum(level='hazard')['dKtot']],axis=1,join='inner')
results_df.columns = ['dk_event_100','AAL']
#results_df.to_csv(out_files+'results_table_new.csv')
print('Writing '+out_files+'results_table_new.csv')

##################################
# Manipulate iah 
# --> use AE, in case that's different from per cap
iah['c_initial']    = (iah[['c','pcwgt']].prod(axis=1)/iah['aewgt']).fillna(0)
# ^ hh consumption, as reported in HIES

iah['di_pre_reco']  = (iah[['di0','pcwgt']].prod(axis=1)/iah['aewgt']).fillna(0)
iah['dc_pre_reco']  = (iah[['dc0','pcwgt']].prod(axis=1)/iah['aewgt']).fillna(0)
# ^ hh income loss (di & dc) immediately after disaster

iah['dc_post_reco'] = (iah[['dc_post_reco','pcwgt']].prod(axis=1)/iah['aewgt']).fillna(0)
# ^ hh consumption loss (dc) after 10 years of reconstruction

iah['pds_nrh']      = iah.eval('(pc_fee+help_fee-help_received)*(pcwgt/aewgt)').fillna(0)
# ^ Net post-disaster support

iah['i_pre_reco']   = (iah['c_initial'] + drm_pov_sign*iah['di_pre_reco'])
iah['c_pre_reco']   = (iah['c_initial'] + drm_pov_sign*iah['dc_pre_reco'])
iah['c_post_reco']  = (iah['c_initial'] + drm_pov_sign*iah['dc_post_reco'])
# ^ income & consumption before & after reconstruction

##################################
# Create additional dfs

# Clone index of iah with just one entry/hhid
iah_res = pd.DataFrame(index=(iah.sum(level=[economy,'hazard','rp','hhid'])).index)

## Translate from iah by summing over hh categories [(a,na)x(helped,not_helped)]
# These are special--pcwgt has been distributed among [(a,na)x(helped,not_helped)] categories
iah_res['pcwgt'] = iah['pcwgt'].sum(level=[economy,'hazard','rp','hhid'])
iah_res['aewgt'] = iah['aewgt'].sum(level=[economy,'hazard','rp','hhid'])
iah_res['hhwgt'] = iah['hhwgt'].sum(level=[economy,'hazard','rp','hhid'])

#These are the same across [(a,na)x(helped,not_helped)] categories 
iah_res['k']         = iah['k'].mean(level=[economy,'hazard','rp','hhid'])
iah_res['c']         = iah['c'].mean(level=[economy,'hazard','rp','hhid'])
#iah_res['aeinc']      = iah['aeinc'].mean(level=[economy,'hazard','rp','hhid'])
#iah_res['hhsize_ae'] = iah['hhsize_ae'].mean(level=[economy,'hazard','rp','hhid'])
iah_res['quintile']  = iah['quintile'].mean(level=[economy,'hazard','rp','hhid'])
iah_res['pov_line']  = iah['pov_line'].mean(level=[economy,'hazard','rp','hhid'])

# Get subsistence line
if get_subsistence_line(myCountry) != None: 
    iah['sub_line'] = get_subsistence_line(myCountry)
    iah_res['sub_line'] = get_subsistence_line(myCountry)

# These need to be averaged across [(a,na)x(helped,not_helped)] categories (weighted by pcwgt)
# ^ values still reported per capita
iah_res['dk0']           = iah[[  'dk0','pcwgt']].prod(axis=1).sum(level=[economy,'hazard','rp','hhid'])/iah_res['pcwgt']
iah_res['dc0']           = iah[[  'dc0','pcwgt']].prod(axis=1).sum(level=[economy,'hazard','rp','hhid'])/iah_res['pcwgt']
iah_res['help_received'] = iah[['help_received','pcwgt']].prod(axis=1).sum(level=[economy,'hazard','rp','hhid'])/iah_res['pcwgt']
iah_res['pc_fee']        = iah[['pc_fee','pcwgt']].prod(axis=1).sum(level=[economy,'hazard','rp','hhid'])/iah_res['pcwgt']
iah_res['dc_npv_pre']    = iah[['dc_npv_pre','pcwgt']].prod(axis=1).sum(level=[economy,'hazard','rp','hhid'])/iah_res['pcwgt']

# These are the other policies (scorecard)
# NB: already weighted by pcwgt from their respective files
for iPol in all_policies:
    print('dk0'+iPol)
    iah_res['dk0'+iPol] = iah['dk0'+iPol].sum(level=[economy,'hazard','rp','hhid'])/iah_res['pcwgt']
    iah_res['dw'+iPol] = iah['dw'+iPol].sum(level=[economy,'hazard','rp','hhid'])/iah_res['pcwgt']

# Note that we're pulling dw in from iah_base and  here
iah_res['dw']     = (iah_base[['dw','pcwgt']].prod(axis=1).sum(level=[economy,'hazard','rp','hhid'])/iah_res['pcwgt'])/df.wprime.mean()
iah_res['pds_dw'] = (iah[['dw','pcwgt']].prod(axis=1).sum(level=[economy,'hazard','rp','hhid'])/iah_res['pcwgt'])/df.wprime.mean()
try: iah_res['pds2_dw'] = (iah_SP2[['dw','pcwgt']].prod(axis=1).sum(level=[economy,'hazard','rp','hhid'])/iah_res['pcwgt'])/df_SP2.wprime.mean()
except: pass
try: iah_res['pds3_dw'] = (iah_SP3[['dw','pcwgt']].prod(axis=1).sum(level=[economy,'hazard','rp','hhid'])/iah_res['pcwgt'])/df_SP3.wprime.mean()
except: pass

# Huge file
del iah_base

iah_res['c_initial']   = iah[['c_initial'  ,'aewgt']].prod(axis=1).sum(level=[economy,'hazard','rp','hhid'])/iah_res['aewgt'] # c per AE
iah_res['di_pre_reco'] = iah[['di_pre_reco','aewgt']].prod(axis=1).sum(level=[economy,'hazard','rp','hhid'])/iah_res['aewgt'] # di per AE
iah_res['dc_pre_reco'] = iah[['dc_pre_reco','aewgt']].prod(axis=1).sum(level=[economy,'hazard','rp','hhid'])/iah_res['aewgt'] # dc per AE
iah_res['pds_nrh']     = iah[['pds_nrh'    ,'aewgt']].prod(axis=1).sum(level=[economy,'hazard','rp','hhid'])/iah_res['aewgt'] # nrh per AE
iah_res['i_pre_reco']  = iah[['i_pre_reco' ,'aewgt']].prod(axis=1).sum(level=[economy,'hazard','rp','hhid'])/iah_res['aewgt'] # i pre-reco per AE
iah_res['c_pre_reco']  = iah[['c_pre_reco' ,'aewgt']].prod(axis=1).sum(level=[economy,'hazard','rp','hhid'])/iah_res['aewgt'] # c pre-reco per AE
iah_res['c_post_reco'] = iah[['c_post_reco','aewgt']].prod(axis=1).sum(level=[economy,'hazard','rp','hhid'])/iah_res['aewgt'] # c post-reco per AE
#iah_res['c_final_pds'] = iah[['c_final_pds','aewgt']].prod(axis=1).sum(level=[economy,'hazard','rp','hhid'])/iah_res['aewgt'] # c per AE

# Calc people who fell into poverty on the regional level for each disaster
iah_res['delta_pov_pre_reco']  = iah.loc[(iah.c_initial > iah.pov_line)&(iah.c_pre_reco <= iah.pov_line),'pcwgt'].sum(level=[economy,'hazard','rp','hhid'])
iah_res['delta_pov_post_reco'] = iah.loc[(iah.c_initial > iah.pov_line)&(iah.c_post_reco <= iah.pov_line),'pcwgt'].sum(level=[economy,'hazard','rp','hhid'])

iah_res = iah_res.reset_index()
iah_res['delta_pov_pre_reco'] = iah_res.groupby([economy,'hazard','rp'])['delta_pov_pre_reco'].transform('sum')
iah_res['delta_pov_post_reco'] = iah_res.groupby([economy,'hazard','rp'])['delta_pov_post_reco'].transform('sum')
iah_res = iah_res.reset_index().set_index([economy,'hazard','rp','hhid']).drop('index',axis=1)

iah = iah.reset_index()

# Save out iah by economic unit
iah_out = pd.DataFrame(index=iah_res.sum(level=[economy,'hazard','rp']).index)
for iPol in ['']+all_policies:
    iah_out['Asset risk'+iPol] = iah_res[['dk0'+iPol,'pcwgt']].prod(axis=1).sum(level=[economy,'hazard','rp'])
    iah_out['Well-being risk'+iPol] = iah_res[['dw'+iPol,'pcwgt']].prod(axis=1).sum(level=[economy,'hazard','rp']) 

# Add public well-being costs to this output & update resilience
iah_out['Well-being risk'] += public_costs_sum['dw_tot_curr']
iah_out['SE capacity']  = iah_out['Asset risk']/iah_out['Well-being risk']
#iah_out['pds_dw']      = iah_res[['pds_dw','pcwgt']].prod(axis=1).sum(level=[economy,'hazard','rp'])
#iah_out['pc_fee']      = iah_res[['pc_fee','pcwgt']].prod(axis=1).sum(level=[economy,'hazard','rp'])
#iah_out['pds2_dw'] = iah_res[['pds2_dw','pcwgt']].prod(axis=1).sum(level=[economy,'hazard','rp'])
#iah_out['pds3_dw'] = iah_res[['pds3_dw','pcwgt']].prod(axis=1).sum(level=[economy,'hazard','rp'])

# Save out risk to assets & welfare & resilience by economy/hazard/rp
iah_out.to_csv(out_files+'risk_by_economy_hazard_rp.csv')





# Average over RP & update resilience
iah_out,_ = average_over_rp(iah_out)
iah_out['SE capacity']  = iah_out['Asset risk']/iah_out['Well-being risk']

# Save out risk to assets & welfare & resilience by economy/hazard
iah_out.to_csv(out_files+'risk_by_economy_hazard.csv')


# Write LATEX tables with risk to assets & welfare & resilience by economy/hazard
# Local currency
_ = (iah_out['Asset risk']/1.E6).round(1).unstack().copy()
_['Total'] = _.sum(axis=1)
_.loc['Total'] = _.sum()
_.sort_values('Total',ascending=False).round(0).to_latex('latex/'+myCountry+'/risk_by_economy_hazard.tex')
# USD
_ = (iah_out['Asset risk']/1.E6).round(1).unstack().copy()*to_usd*1.E3
_['Total'] = _.sum(axis=1)
_.loc['Total'] = _.sum(axis=0)
_.sort_values('Total',ascending=False).round(1).to_latex('latex/'+myCountry+'/risk_by_economy_hazard_usd.tex')


# Sum over hazards
iah_out = iah_out.sum(level=economy)
iah_out[['Asset risk','Well-being risk']]/=1.E6 # iah_out is thousands [1E3]

iah_out.loc['Total'] = [float(iah_out['Asset risk'].sum()),
                        float(iah_out['Well-being risk'].sum()),
                        float(iah_out['Asset risk'].sum()/iah_out['Well-being risk'].sum())]
iah_out['SE capacity']  = 100.*iah_out['Asset risk']/iah_out['Well-being risk']

# Write out risk to assets & welfare & resilience by economy
# Local currency
iah_out.to_csv(out_files+'risk_by_economy.csv')
iah_out[['Asset risk',
         'SE capacity',
         'Well-being risk']].sort_values(['Well-being risk'],ascending=False).round(0).to_latex('latex/'+myCountry+'/risk_by_economy.tex')
#USD
iah_out = iah_out.drop('Total')
iah_out[['Asset risk','Well-being risk']]*=to_usd*1E3 # iah_out is millions of USD now
iah_out.loc['Total'] = [float(iah_out['Asset risk'].sum()),
                        float(iah_out['Well-being risk'].sum()),
                        float(iah_out['Asset risk'].sum()/iah_out['Well-being risk'].sum())]
iah_out['SE capacity']  = 100.*iah_out['Asset risk']/iah_out['Well-being risk']

iah_out[['Asset risk','SE capacity','Well-being risk']].sort_values(['Well-being risk'],ascending=False).round(1).to_latex('latex/'+myCountry+'/risk_by_economy_usd.tex')
print('Wrote latex! Sums:\n',iah_out.loc['Total',['Asset risk','Well-being risk']])

# Save out iah by economic unit, *only for poorest quintile*
iah_out_q1 = pd.DataFrame(index=iah_res.sum(level=[economy,'hazard','rp']).index)
for iPol in ['']+all_policies:
    iah_out_q1['Asset risk'+iPol] = iah_res.loc[(iah_res.quintile==1),['dk0'+iPol,'pcwgt']].prod(axis=1).sum(level=[economy,'hazard','rp'])*1E-3
    iah_out_q1['Well-being risk'+iPol] = iah_res.loc[(iah_res.quintile==1),['dw'+iPol,'pcwgt']].prod(axis=1).sum(level=[economy,'hazard','rp'])*1E-3

iah_out_q1.to_csv(out_files+'risk_q1_by_economy_hazard_rp.csv')
iah_out_q1,_ = average_over_rp(iah_out_q1,'default_rp')
iah_out_q1['SE capacity']  = 100*iah_out_q1['Asset risk']/iah_out_q1['Well-being risk']
iah_out_q1.to_csv(out_files+'risk_q1_by_economy_hazard.csv')




iah_out_q1 = iah_out_q1.sum(level=economy)
iah_out_q1.loc['Total'] = [float(iah_out_q1['Asset risk'].sum()),
                           float(iah_out_q1['Well-being risk'].sum()),
                           float(iah_out_q1['Asset risk'].sum()/iah_out_q1['Well-being risk'].sum())]
iah_out_q1['SE capacity']  = iah_out_q1['Asset risk']/iah_out_q1['Well-being risk']
iah_out_q1.to_csv(out_files+'risk_q1_by_economy.csv')


iah_out_q1['% total RA'] = (100.*iah_out_q1['Asset risk']*to_usd/iah_out['Asset risk']).round(1)
iah_out_q1['% total RW'] = (100.*iah_out_q1['Well-being risk']*to_usd/iah_out['Well-being risk']).round(1)
iah_out_q1['SE capacity']*=100.
iah_out_q1['SE capacity']=iah_out_q1['SE capacity'].round(1)

iah_out_q1[['Asset risk','% total RA','SE capacity','Well-being risk','% total RW']].sort_values(['Well-being risk'],ascending=False).astype('int').to_latex('latex/'+myCountry+'/risk_q1_by_economy.tex')

iah_out_q1['Asset risk']*=to_usd
iah_out_q1['Well-being risk']*=to_usd
iah_out_q1[['Asset risk','% total RA','SE capacity','Well-being risk','% total RW']].sort_values(['Well-being risk'],ascending=False).astype('int').to_latex('latex/'+myCountry+'/risk_q1_by_economy_usd.tex')

print('Wrote latex! Q1 sums: ',iah_out_q1.sum())
assert(False)

iah_out_q1['pop_q1']  = iah_res.loc[iah_res.quintile==1,'pcwgt'].sum(level=event_level).mean(level=event_level[0])/1.E3
#iah_out_q1['grdp_q1'] = iah_res.loc[iah_res.quintile==1,['pcwgt','c']].prod(axis=1).sum(level=event_level).mean(level=event_level[0])

_ = iah_out_q1.drop('Total',axis=0)[['pop_q1','Asset risk','Well-being risk']].copy()

#_[['Asset risk','Well-being risk']]*=to_usd
_['Asset risk pc'] = iah_out_q1['Asset risk']*1.E3/iah_out_q1['pop_q1']
_['Well-being risk pc'] = iah_out_q1['Well-being risk']*1.E3/iah_out_q1['pop_q1']
_.loc['Total'] = [_['pop_q1'].sum(),
                  _['Asset risk'].sum(),
                  _['Well-being risk'].sum(),
                  _['Asset risk'].sum()*1.E3/_['pop_q1'].sum(),
                  _['Well-being risk'].sum()*1.E3/_['pop_q1'].sum()]

_['pop_q1'] = _['pop_q1'].astype('int')

_[['pop_q1','Asset risk pc','Well-being risk pc']].round(2).sort_values('Well-being risk pc',ascending=False).to_latex('latex/'+myCountry+'/risk_pc_q1_by_economy.tex')
_[['pop_q1','Asset risk pc','Well-being risk pc']].round(2).sort_values('Well-being risk pc',ascending=False).to_csv('latex/'+myCountry+'/risk_pc_q1_by_economy.csv')

_ = _.drop('Total')
_['Asset risk pc'] *= to_usd
_['Well-being risk pc'] *= to_usd
_.loc['Total'] = [_['pop_q1'].sum(),
                  _['Asset risk'].sum(),
                  _['Well-being risk'].sum(),
                  _['Asset risk'].sum()*1.E3/_['pop_q1'].sum(),
                  _['Well-being risk'].sum()*1.E3/_['pop_q1'].sum()]
_[['pop_q1','Asset risk pc','Well-being risk pc']].round(2).sort_values('Well-being risk pc',ascending=False).to_latex('latex/'+myCountry+'/risk_pc_q1_by_economy_usd.tex')

# Save out iah
iah_out = pd.DataFrame(index=iah_res.sum(level=['hazard','rp']).index)
for iPol in ['']+all_policies:
    iah_out['dk0'+iPol] = iah_res[['dk0'+iPol,'pcwgt']].prod(axis=1).sum(level=['hazard','rp'])
    iah_out['dw'+iPol] = iah_res[['dw'+iPol,'pcwgt']].prod(axis=1).sum(level=['hazard','rp'])
iah_out['pds_dw'] = iah_res[['pds_dw','pcwgt']].prod(axis=1).sum(level=['hazard','rp'])
iah_out['pc_fee'] = iah_res[['pc_fee','pcwgt']].prod(axis=1).sum(level=['hazard','rp'])
try: iah_out['pds2_dw'] = iah_res[['pds2_dw','pcwgt']].prod(axis=1).sum(level=['hazard','rp'])
except: pass
try: iah_out['pds3_dw'] = iah_res[['pds3_dw','pcwgt']].prod(axis=1).sum(level=['hazard','rp'])
except: pass

iah_out.to_csv(out_files+'risk_by_hazard_rp.csv')
print(iah_out.head(10))
iah_out,_ = average_over_rp(iah_out,'default_rp')
iah_out.to_csv(out_files+'risk_total.csv')




# Clone index of iah at national level
iah_ntl = pd.DataFrame(index=(iah_res.sum(level=['hazard','rp'])).index)
#
iah_ntl['pop'] = iah_res.pcwgt.sum(level=['hazard','rp'])
iah_ntl['pov_pc_i'] = iah_res.loc[(iah_res.c_initial <= iah_res.pov_line),'pcwgt'].sum(level=['hazard','rp'])
iah_ntl['sub_pc_i'] = iah_res.loc[(iah_res.c_initial <= iah_res.sub_line),'pcwgt'].sum(level=['hazard','rp'])
#
iah_ntl['c_pov_pc_f'] = iah_res.loc[(iah_res.c_pre_reco <= iah_res.pov_line),'pcwgt'].sum(level=['hazard','rp'])
iah_ntl['i_pov_pc_f'] = iah_res.loc[(iah_res.i_pre_reco <= iah_res.pov_line),'pcwgt'].sum(level=['hazard','rp'])
#
iah_ntl['c_sub_pc_f'] = iah_res.loc[(iah_res.c_pre_reco <= iah_res.sub_line),'pcwgt'].sum(level=['hazard','rp'])
iah_ntl['i_sub_pc_f'] = iah_res.loc[(iah_res.i_pre_reco <= iah_res.sub_line),'pcwgt'].sum(level=['hazard','rp'])
#
iah_ntl['c_pov_pc_change'] = iah_ntl['c_pov_pc_f'] - iah_ntl['pov_pc_i']
iah_ntl['i_pov_pc_change'] = iah_ntl['i_pov_pc_f'] - iah_ntl['pov_pc_i']
#
iah_ntl['c_sub_pc_change'] = iah_ntl['c_sub_pc_f'] - iah_ntl['sub_pc_i']
iah_ntl['i_sub_pc_change'] = iah_ntl['i_sub_pc_f'] - iah_ntl['sub_pc_i']
#
print('\n\nInitial poverty incidence:\n',iah_ntl['pov_pc_i'].mean())
print('--> In case of SL: THIS IS NOT RIGHT! Maybe because of the 3 provinces that dropped off?')

# Print out plots for iah_res
iah_res = iah_res.reset_index()
iah_ntl = iah_ntl.reset_index()

#########################
# Save out
iah_ntl.to_csv(out_files+'poverty_ntl_by_haz.csv')
iah_ntl = iah_ntl.reset_index().set_index(['hazard','rp']).sort_index()
iah_ntl_haz,_ = average_over_rp(iah_ntl,'default_rp')
iah_ntl_haz.sum(level='hazard').to_csv(out_files+'poverty_haz_sum.csv')

iah_ntl = iah_ntl.reset_index().set_index('rp').sum(level='rp').sort_index()
iah_ntl.to_csv(out_files+'poverty_ntl.csv')
iah_sum,_ = average_over_rp(iah_ntl,'default_rp')
iah_sum.sum().to_csv(out_files+'poverty_sum.csv')
##########################

myHaz = None
if myCountry == 'FJ': myHaz = [['Ba','Lau','Tailevu'],get_all_hazards(myCountry,iah_res),[1,10,100,500,1000]]
#elif myCountry == 'PH': myHaz = [['V - Bicol','II - Cagayan Valley','NCR','IVA - CALABARZON','ARMM','CAR'],['HU','EQ'],[10,25,50,100,250,500]]
#elif myCountry == 'PH': myHaz = [['ompong'],['HU','PF','SS'],[10,50,100,200,500]]
elif myCountry == 'PH': myHaz = [['I - Ilocos','II - Cagayan Valley','CAR'],['HU'],[25,100]]
elif myCountry == 'SL': myHaz = [['Colombo','Rathnapura','Kalutara','Mannar'],get_all_hazards(myCountry,iah_res),get_all_rps(myCountry,iah_res)]
elif myCountry == 'MW': myHaz = [['Lilongwe','Chitipa'],get_all_hazards(myCountry,iah_res),get_all_rps(myCountry,iah_res)]

##################################################################
# This code generates output on poverty dimensions
# ^ this is by household, so we use iah
if True:
    run_poverty_duration_plot(myCountry)
    run_poverty_tables_and_maps(myCountry,iah.reset_index().set_index(event_level),event_level)
    map_recovery_time('PH')

##################################################################
# This code generates the histograms showing income before & after disaster (in local_curr)
# ^ this is at household level, so we'll use iah

#for _A in myHaz[0]:
#    for _B in myHaz[1]:
#        for _C in myHaz[2]:
#            plot_income_and_consumption_distributions('SL',iah,_A,_B,_C)

if True:            
    with Pool(processes=3,maxtasksperchild=1) as pool:
        print('LAUNCHING',len(list(product(myHaz[0],myHaz[1],myHaz[2]))),'THREADS')
        try: pool.starmap(plot_income_and_consumption_distributions,list(product([myCountry],[iah],myHaz[0],myHaz[1],myHaz[2])))
        except: pass

##################################################################
# This code generates the histograms showing income before & after disaster (in USD)
# ^ this is at household level, so we'll use iah
if True:            
    with Pool(processes=2,maxtasksperchild=1) as pool:
        print('LAUNCHING',len(list(product(myHaz[0],myHaz[1],myHaz[2]))),'THREADS')
        try: pool.starmap(plot_income_and_consumption_distributions,list(product([myCountry],[iah],myHaz[0],myHaz[1],myHaz[2],['USD'])))
        except: pass

##################################################################
# This code generates the histograms including [k,dk,dc,dw,&pds]
# ^ this is by province/region, so it will use iah_res
if True:
    with Pool(processes=2,maxtasksperchild=1) as pool:
        print('LAUNCHING',len(list(product(myHaz[0],myHaz[1],myHaz[2]))),'THREADS')
        pool.starmap(plot_impact_by_quintile,list(product([myCountry],myHaz[0],myHaz[1],myHaz[2],[iah_res])))  


##################################################################
# This code generates the histograms 
# ^ this is only for affected hosueholds, so it will use iah
if True:
    with Pool(processes=2,maxtasksperchild=1) as pool:
        print('LAUNCHING',len(list(product(myHaz[0],myHaz[1],myHaz[2]))),'THREADS')
        pool.starmap(plot_relative_losses,list(product([myCountry],myHaz[0],myHaz[1],myHaz[2],[iah])))  
