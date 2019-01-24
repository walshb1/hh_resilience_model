##################################
# Import packages for data analysis
import matplotlib
matplotlib.use('AGG')

from libraries.maps_lib import *
from libraries.lib_country_dir import *
from libraries.lib_scaleout import get_pmt
from libraries.lib_pds_dict import pds_dict
from libraries.lib_average_over_rp import *
from libraries.replace_with_warning import *
#from libraries.pandas_helper import broadcast_simple
from libraries.lib_common_plotting_functions import *
#from libraries.maps_lib import make_map_from_svg, purge
from libraries.lib_compute_resilience_and_risk import get_weighted_mean
from libraries.lib_get_expectation_values_df import get_expectation_values_df
from libraries.lib_plot_impact_by_quintile import plot_impact_by_quintile, plot_relative_losses
from libraries.lib_plot_income_and_consumption_distributions import plot_income_and_consumption_distributions
from libraries.lib_poverty_tables_and_maps import run_poverty_duration_plot, run_poverty_tables_and_maps, map_recovery_time

from scipy.stats import norm
import matplotlib.mlab as mlab

from multiprocessing import Pool
from itertools import repeat,product

from pandas import isnull
import pandas as pd
import numpy as np
import os,time,gc
import sys

font = {'family' : 'sans serif',
    'size'   : 15}
plt.rc('font', **font)
mpl.rcParams['xtick.labelsize'] = 10
mpl.rcParams['ytick.labelsize'] = 10
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

places_to_drop = None
#if myCountry == 'MW': places_to_drop = ['Blantyre City','Lilongwe City','Mzuzu City','Zomba City']

##################################
# Set directories (where to look for files)
out_files = os.getcwd()+'/../output_country/'+myCountry+'/'


##################################
# Set policy params
drm_pov_sign = -1 # toggle subtraction or addition of dK to affected people's incomes

my_PDS = 'unif_poor'#'scaleout_samurdhi'#'scaleout_samurdhi_universal'#'samurdhi_scaleup'#'samurdhi_scaleup00'#
base_str = 'no'

path = os.getcwd()+'/../output_country/'+myCountry+'/'
pattern = 'sp_costs_*.csv'

all_pds_options = []
for f in glob.glob(path+pattern):
    if f != base_str:
        all_pds_options.append(f.replace(path,'').replace('sp_costs_','').replace('.csv',''))
print(all_pds_options)

if my_PDS not in all_pds_options: my_PDS = 'no'

pds_options = []
if my_PDS != base_str:
    pds_options = [_pds for _pds in ['unif_poor','samurdhi_scaleup','scaleout_samurdhi_universal'] if (_pds != base_str) and (_pds in all_pds_options)]
#pds_options = [all_pds_options[0]]

policy_options = []#['_exp095','_exr095','_ew100','_vul070','_vul070r','_rec067']

##################################
# Load main, hh-level results file
macro = pd.read_csv(out_files+'macro_tax_'+base_str+'_.csv', index_col=[economy,'hazard','rp'])
df = pd.read_csv(out_files+'results_tax_'+base_str+'_.csv', index_col=[economy,'hazard','rp'])
_wprime = df.wprime.mean()

iah = pd.DataFrame(index=pd.read_csv('../intermediate/'+myCountry+'/hazard_ratios.csv', index_col=[economy,'hazard','rp','hhid']).sort_index().index)

affected_cats = pd.Index(['a', 'na'], name='affected_cat')           
helped_cats   = pd.Index(['helped','not_helped'], name='helped_cat')
iah = broadcast_simple(iah,affected_cats)
iah = broadcast_simple(iah,helped_cats)

_b = pd.read_csv(out_files+'iah_tax_'+base_str+'_.csv', index_col=[economy,'hazard','rp','hhid','affected_cat','helped_cat'])
iah = iah.join(_b, how='outer')
iah = iah.fillna(0)

# Get my columns right
iah = iah.rename(columns={'c':'c_initial',
                          'di0':'di_pre_reco',
                          'pcwgt':'pcwgt_'+base_str,
                          'hhwgt':'hhwgt_'+base_str,
                          'aewgt':'aewgt_'+base_str,
                          'hhwgt':'hhwgt_'+base_str,
                          'dw':'dw_'+base_str,
                          'help_received':'help_received_'+base_str})
iah = iah.drop([_c for _c in ['dc0','N_children','frac_remittance','gsp_samurdhi','pcsamurdhi'] if _c in iah.columns],axis=1)
# ^ drop 'dc0' so that it can't be mistakenly used
# --> dc0 is net of reco costs, but not savings or PDS. dc_pre_reco is the right variable to use

##################################
# Manipulate iah
if get_subsistence_line(myCountry) != None: iah['sub_line'] = get_subsistence_line(myCountry)

iah['i_pre_reco']   = iah.eval('c_initial+@drm_pov_sign*di_pre_reco')
try: iah['c_pre_reco']   = iah.eval('c_initial+@drm_pov_sign*dc_pre_reco')
except: 
    iah['c_pre_reco']   = iah.eval('c_initial+@drm_pov_sign*di_pre_reco')
    print('\n Setting i_pre_reco where c_pre_reco should be. Check why it is not in the output\n')
iah['c_post_reco']  = iah.eval('c_initial+@drm_pov_sign*dc_post_reco')
# ^ income & consumption before & after reconstruction


##################################
# Add all the PDS info (hh- and event-level) that I'm going to need
for iPDS in ['no']+pds_options:

    if iPDS == base_str:
        iah['dw_'+iPDS] /= _wprime
        _eval_str = '(pc_fee+help_fee-help_received_'+base_str+')'
        iah['net_help_received_'+iPDS] = iah.eval(_eval_str)
        
        if base_str == 'no' and iah['help_fee'].sum() != 0: assert(False)
        # ^ Sanity check

        iah = iah.drop(['pc_fee','help_fee'],axis=1)
        # ^ Post-disaster support: net received help

    else: 
        _iah_pds = pd.read_csv(out_files+'iah_tax_'+iPDS+'_.csv', index_col=[economy,'hazard','rp','hhid','affected_cat','helped_cat']).sort_index()
        iah['pcwgt_'+iPDS] = _iah_pds['pcwgt'].copy()
        iah['dw_'+iPDS] = _iah_pds['dw']/_wprime
        iah['help_received_'+iPDS] = _iah_pds['help_received'].copy()
        iah['net_help_received_'+iPDS] = _iah_pds.eval('(pc_fee+help_fee-help_received)')

        print(iPDS,'added to iah (PDS option)')

    ## DW costs of risk sharing in each scenario:
    #_pc = pd.read_csv(out_files+'public_costs_tax_'+iPDS+'_.csv').set_index([economy,'hazard','rp'])
    #_pc['dw_tot_curr'] = _pc.eval('dw_pub+dw_soc')/_wprime
    #public_costs_sum['total_dw_curr_'+iPDS] = _pc.loc[_pc['contributer']!=_pc.index.get_level_values(event_level[0]),['dw_tot_curr']].sum(level=[economy,'hazard','rp'])

    gc.collect()


# Decide whether to use AE, in case that's different from per cap
use_aewgt = False
if use_aewgt:
    for pcwgt_to_aewgt in ['c_initial', 'i_pre_reco', 'di_pre_reco', 'c_pre_reco', 'dc_pre_reco', 'dc_post_reco']:
        iah[pcwgt_to_aewgt] *= iah.eval('aewgt_'+base_str+'/pcwgt_'+base_str).fillna(0)
        iah = iah.drop('pcwgt_'+base_str)
else: iah = iah.drop('aewgt_'+base_str,axis=1)


##################################
# PDS
pds_effects_out = pd.DataFrame(index=pd.read_csv(out_files+'sp_costs_no.csv').set_index([economy,'hazard','rp']).index)
# ^ empty dataframe with event-level index

for _pds in ['no']+all_pds_options:

    try:
        ####
        hh_summary = pd.read_csv(out_files+'my_summary_'+_pds+'.csv').set_index([economy,'hazard','rp'])
        ####
        pds_effects = pd.read_csv(out_files+'sp_costs_'+_pds+'.csv').set_index([economy,'hazard','rp'])
        
        # DW costs of risk sharing in each SP scenario
        public_costs_pds = pd.read_csv(out_files+'public_costs_tax_'+_pds+'_.csv').set_index([economy,'hazard','rp'])
        public_costs_pds['dw_tot_curr'] = 1E-6*public_costs_pds[['dw_pub','dw_soc']].sum(axis=1)/_wprime
        public_costs_pds_sum = public_costs_pds.loc[public_costs_pds['contributer']!=public_costs_pds.index.get_level_values(event_level[0]),['dw_tot_curr']].sum(level=[economy,'hazard','rp'])
        #
        pds_effects_out['dw_'+_pds] = hh_summary['dw_tot']+public_costs_pds_sum['dw_tot_curr']
        #
        if _pds != 'no': 
            pds_effects_out['dw_DELTA_'+_pds] = pds_effects_out['dw_no'] - pds_effects_out['dw_'+_pds]
            pds_effects_out['ROI_event_'+_pds] = pds_effects_out['dw_DELTA_'+_pds]/(1E-6*pds_effects['event_cost'])
    except: pass

pds_effects_out.to_csv(out_files+'pds_effects.csv')



##################################
# SAVE OUT FILE with expected Asset losses (DK) for RP = 100 and AAL
if False:
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
# Create additional dfs
#
#
# 1) Get dataframe with expectation values: one entry per household, averaged over (a/na & helped/not_helped)
iah_avg = get_expectation_values_df(myCountry,economy,iah,pds_options,base_str=base_str,use_aewgt=use_aewgt)

# 2) Save out iah by economic unit
iah_out = pd.DataFrame(index=iah_avg.sum(level=[economy,'hazard','rp']).index)
for iPDS in ['no']+pds_options:
    iah_out['Asset risk'] = iah_avg[['dk0','pcwgt_'+base_str]].prod(axis=1).sum(level=[economy,'hazard','rp'])
    iah_out['Well-being risk'+(' '+iPDS).replace(' '+base_str,'')] = iah_avg[['dw_'+iPDS,'pcwgt_'+iPDS]].prod(axis=1).sum(level=[economy,'hazard','rp']) 

    # Add public well-being costs to this output & update resilience
    _pc = pd.read_csv(out_files+'public_costs_tax_'+_pds+'_.csv').set_index([economy,'hazard','rp'])
    _pc['dw_tot_curr'] = _pc.eval('(dw_pub+dw_soc)/@_wprime')
    _pc_sum = _pc.loc[_pc['contributer']!=_pc.index.get_level_values(event_level[0]),['dw_tot_curr']].sum(level=[economy,'hazard','rp'])

    iah_out['Well-being risk'+(' '+iPDS).replace(' no','')] += _pc_sum['dw_tot_curr']
    iah_out['SE capacity'+(' '+iPDS).replace(' no','')]  = iah_out['Asset risk']/iah_out['Well-being risk'+(' '+iPDS).replace(' no','')]

iah_out.to_csv(out_files+'risk_by_economy_hazard_rp.csv')



# 3) Average over RP & update resilience
iah_out,_ = average_over_rp(iah_out)
iah_out['SE capacity']  = iah_out['Asset risk']/iah_out['Well-being risk']
iah_out.to_csv(out_files+'risk_by_economy_hazard.csv')
if myCountry == 'MW':
    iah_out = iah_out.reset_index()
    for _hack in [['Blantyre City','Blantyre'],
                  ['Lilongwe City','Lilongwe'],
                  ['Mzuzu City','Mzimba'],
                  ['Zomba City','Zomba']]:
        iah_out.loc[iah_out[event_level[0]]==_hack[0],event_level[0]]=_hack[1]
    iah_out = iah_out.reset_index().set_index(event_level[:-1])
if False:
    svg_file = get_svg_file(myCountry)
    for _h in get_all_hazards(myCountry,iah_out):

        make_map_from_svg(
            iah_out.loc[(slice(None),(_h)),'Asset risk'].sum(level=economy)*1E-6*to_usd,
            svg_file,
            outname=myCountry+'_expected_losses_'+_h,
            color_maper=plt.cm.get_cmap('Reds'), 
            label='Annual asset losses to '+haz_dict[_h].lower()+'s [mil. USD]',
            new_title='',
            do_qualitative=False,
            res=2000,
            drop_spots=places_to_drop)

    purge('img/','map_of_*.png')
    purge('img/','legend_of_*.png')
    purge('img/','map_of_*.svg')
    purge('img/','legend_of_*.svg')
# ^ Save out risk to assets & welfare & resilience by economy/hazard/PDS


# 4) Write tables with just NO PDS
no_pds = iah_out[['Asset risk','Well-being risk']].copy()

# Write LATEX tables with risk to assets & welfare & resilience by economy/hazard
# Local currency
_ = (no_pds['Asset risk']/1.E6).round(1).unstack().copy()
_['Total'] = _.sum(axis=1)
_.loc['Total'] = _.sum()
_.sort_values('Total',ascending=False).round(0).to_latex('latex/'+myCountry+'/risk_by_economy_hazard.tex')
# USD
_ = (no_pds['Asset risk']/1.E6).round(1).unstack().copy()*to_usd
_['Total'] = _.sum(axis=1)
_.loc['Total'] = _.sum(axis=0)
_.sort_values('Total',ascending=False).round(1).to_latex('latex/'+myCountry+'/risk_by_economy_hazard_usd.tex')


# Sum over hazards
no_pds = no_pds.sum(level=economy)
no_pds[['Asset risk','Well-being risk']]/=1.E9 # no_pds is thousands [1E3]
no_pds['SE capacity']  = 100.*no_pds['Asset risk']/no_pds['Well-being risk']

no_pds.loc['Total'] = [float(no_pds['Asset risk'].sum()),
                        float(no_pds['Well-being risk'].sum()),
                        float(no_pds['Asset risk'].sum()/no_pds['Well-being risk'].sum())]
no_pds['SE capacity']  = 100.*no_pds['Asset risk']/no_pds['Well-being risk']

# Write out risk to assets & welfare & resilience by economy
# Local currency
no_pds.to_csv(out_files+'risk_by_economy.csv')
no_pds[['Asset risk',
         'SE capacity',
         'Well-being risk']].sort_values(['Well-being risk'],ascending=False).round(0).to_latex('latex/'+myCountry+'/risk_by_economy.tex')
#USD
no_pds = no_pds.drop('Total')
no_pds[['Asset risk','Well-being risk']]*=to_usd*1E3 # no_pds is millions of USD now
no_pds.loc['Total'] = [float(no_pds['Asset risk'].sum()),
                        float(no_pds['Well-being risk'].sum()),
                        float(no_pds['Asset risk'].sum()/no_pds['Well-being risk'].sum())]
no_pds['SE capacity']  = 100.*no_pds['Asset risk']/no_pds['Well-being risk']

no_pds[['Asset risk','SE capacity','Well-being risk']].sort_values(['Well-being risk'],ascending=False).round(1).to_latex('latex/'+myCountry+'/risk_by_economy_usd.tex')
print('Wrote latex! Sums:\n',no_pds.loc['Total',['Asset risk','Well-being risk']])

####################################
# Save out iah by economic unit, *only for poorest quintile*
no_pds_q1 = pd.DataFrame(index=iah.sum(level=[economy,'hazard','rp']).index)

no_pds_q1['Asset risk'] = iah.loc[(iah.quintile==1),['dk0','pcwgt_no']].prod(axis=1).sum(level=[economy,'hazard','rp'])*1E-6
no_pds_q1['Well-being risk'] = iah.loc[(iah.quintile==1),['dw_no','pcwgt_no']].prod(axis=1).sum(level=[economy,'hazard','rp'])*1E-6
no_pds_q1['TOTAL Asset risk'] = iah[['dk0','pcwgt_no']].prod(axis=1).sum(level=[economy,'hazard','rp'])*1E-6
no_pds_q1['TOTAL Well-being risk'] = iah[['dw_no','pcwgt_no']].prod(axis=1).sum(level=[economy,'hazard','rp'])*1E-6
# needs external DWs!!!

no_pds_q1.to_csv(out_files+'risk_q1_by_economy_hazard_rp.csv')
no_pds_q1,_ = average_over_rp(no_pds_q1,'default_rp')
no_pds_q1['SE capacity']  = 100*no_pds_q1['Asset risk']/no_pds_q1['Well-being risk']
no_pds_q1.to_csv(out_files+'risk_q1_by_economy_hazard.csv')


no_pds_q1 = no_pds_q1.sum(level=economy)
no_pds_q1.loc['Total',['Asset risk','Well-being risk',
                       'TOTAL Asset risk','TOTAL Well-being risk','SE capacity']] = [float(no_pds_q1['Asset risk'].sum()),
                                                                                     float(no_pds_q1['Well-being risk'].sum()),
                                                                                     float(no_pds_q1['TOTAL Asset risk'].sum()),
                                                                                     float(no_pds_q1['TOTAL Well-being risk'].sum()),
                                                                                     100.*float(no_pds_q1['Asset risk'].sum()/no_pds_q1['Well-being risk'].sum())]
no_pds_q1['SE capacity']  = 100.*no_pds_q1['Asset risk']/no_pds_q1['Well-being risk']
no_pds_q1.to_csv(out_files+'risk_q1_by_economy.csv')

no_pds_q1['% total RA'] = 100.*no_pds_q1['Asset risk']/no_pds_q1['TOTAL Asset risk']
no_pds_q1['% total RW'] = 100.*no_pds_q1['Well-being risk']/no_pds_q1['TOTAL Well-being risk']

no_pds_q1.loc['Total',['% total RA','% total RW']] = 100*no_pds_q1.loc['Total','Asset risk']/no_pds_q1['TOTAL Asset risk'].sum()
no_pds_q1.loc['Total',['% total RA','% total RW']] = 100*no_pds_q1.loc['Total','Well-being risk']/no_pds_q1['TOTAL Well-being risk'].sum()

print(no_pds_q1.head(30))
no_pds_q1 = no_pds_q1.fillna(0)

no_pds_q1[['Asset risk','% total RA','SE capacity',
           'Well-being risk','% total RW']].sort_values(['Well-being risk'],ascending=False).round(1).to_latex('latex/'+myCountry+'/risk_q1_by_economy.tex')

no_pds_q1['Asset risk']*=to_usd
no_pds_q1['Well-being risk']*=to_usd
no_pds_q1[['Asset risk','% total RA','SE capacity',
           'Well-being risk','% total RW']].sort_values(['Well-being risk'],ascending=False).round(1).to_latex('latex/'+myCountry+'/risk_q1_by_economy_usd.tex')
print('Wrote latex! Q1 sums: ',no_pds_q1.sum())



no_pds_q1['pop_q1']  = iah.loc[iah.quintile==1,'pcwgt_no'].sum(level=event_level).mean(level=event_level[0])/1.E3
#iah_out_q1['grdp_q1'] = iah.loc[iah.quintile==1,['pcwgt_no','c_initial']].prod(axis=1).sum(level=event_level).mean(level=event_level[0])

_ = no_pds_q1.drop('Total',axis=0)[['pop_q1','Asset risk','Well-being risk']].copy()

#_[['Asset risk','Well-being risk']]*=to_usd
_['Asset risk pc'] = no_pds_q1['Asset risk']*1.E3/no_pds_q1['pop_q1']
_['Well-being risk pc'] = no_pds_q1['Well-being risk']*1.E3/no_pds_q1['pop_q1']
_.loc['Total'] = [_['pop_q1'].sum(),
                  _['Asset risk'].sum(),
                  _['Well-being risk'].sum(),
                  _['Asset risk'].sum()*1.E3/_['pop_q1'].sum(),
                  _['Well-being risk'].sum()*1.E3/_['pop_q1'].sum()]

_['pop_q1'] = _['pop_q1'].astype('int')

_[['pop_q1','Asset risk pc','Well-being risk pc']].round(2).sort_values('Well-being risk pc',ascending=False).to_latex('latex/'+myCountry+'/risk_pc_q1_by_economy_usd.tex')
_[['pop_q1','Asset risk pc','Well-being risk pc']].round(2).sort_values('Well-being risk pc',ascending=False).to_csv('latex/'+myCountry+'/risk_pc_q1_by_economy_usd.csv')

_ = _.drop('Total')
_['Asset risk pc'] /= to_usd
_['Well-being risk pc'] /= to_usd
_.loc['Total'] = [_['pop_q1'].sum(),
                  _['Asset risk'].sum(),
                  _['Well-being risk'].sum(),
                  _['Asset risk'].sum()*1.E3/_['pop_q1'].sum(),
                  _['Well-being risk'].sum()*1.E3/_['pop_q1'].sum()]
_[['pop_q1','Asset risk pc','Well-being risk pc']].round(2).sort_values('Well-being risk pc',ascending=False).to_latex('latex/'+myCountry+'/risk_pc_q1_by_economy.tex')

# Save out iah
iah_out = pd.DataFrame(index=iah_avg.sum(level=['hazard','rp']).index)
iah_out['dk0'] = iah_avg[['dk0','pcwgt_no']].prod(axis=1).sum(level=['hazard','rp'])
for iPDS in ['no']+pds_options:
    iah_out['dw_'+iPDS] = iah_avg[['dw_'+iPDS,'pcwgt_'+iPDS]].prod(axis=1).sum(level=['hazard','rp'])
    iah_out['net_help_received_'+iPDS] = iah_avg[['net_help_received_'+iPDS,'pcwgt_'+iPDS]].prod(axis=1).sum(level=['hazard','rp'])

iah_out.to_csv(out_files+'risk_by_hazard_rp.csv')
print(iah_out.head(10))
iah_out,_ = average_over_rp(iah_out,'default_rp')
iah_out.to_csv(out_files+'risk_total.csv')


####################################################
# Clone index of iah at national level
# --> could use iah here, or could use expectation value (iah_avg).
# This affects the results!
# --> that's because, for poverty line (binary), it matters whether you use the whole household, or its fractional pieces

iah = iah.drop([_c for _c in ['welf_class','province','pcexp','axfin','optimal_hh_reco_rate'] if _c in iah.columns],axis=1)

# --> leaning toward iah
myiah = iah.copy(deep=True)
iah = None
#iah_avg = None

##############

iah_ntl = pd.DataFrame(index=(myiah.sum(level=['hazard','rp'])).index)
#
iah_ntl['pop'] = myiah.pcwgt_no.sum(level=['hazard','rp'])
iah_ntl['pov_pc_i'] = myiah.loc[(myiah.c_initial <= myiah.pov_line),'pcwgt_no'].sum(level=['hazard','rp'])
iah_ntl['sub_pc_i'] = myiah.loc[(myiah.c_initial <= myiah.sub_line),'pcwgt_no'].sum(level=['hazard','rp'])
#
iah_ntl['c_pov_pc_f'] = myiah.loc[myiah.eval('c_pre_reco<=pov_line'),'pcwgt_no'].sum(level=['hazard','rp'])
iah_ntl['i_pov_pc_f'] = myiah.loc[myiah.eval('i_pre_reco<=pov_line'),'pcwgt_no'].sum(level=['hazard','rp'])
#
iah_ntl['c_sub_pc_f'] = myiah.loc[myiah.eval('c_pre_reco<=sub_line'),'pcwgt_no'].sum(level=['hazard','rp'])
iah_ntl['i_sub_pc_f'] = myiah.loc[myiah.eval('i_pre_reco<=sub_line'),'pcwgt_no'].sum(level=['hazard','rp'])
#
iah_ntl['c_pov_pc_change'] = iah_ntl['c_pov_pc_f'] - iah_ntl['pov_pc_i']
iah_ntl['i_pov_pc_change'] = iah_ntl['i_pov_pc_f'] - iah_ntl['pov_pc_i']
#
iah_ntl['c_sub_pc_change'] = iah_ntl['c_sub_pc_f'] - iah_ntl['sub_pc_i']
iah_ntl['i_sub_pc_change'] = iah_ntl['i_sub_pc_f'] - iah_ntl['sub_pc_i']
#
print('\n\nInitial poverty incidence:\n',iah_ntl['pov_pc_i'].mean())
print('--> In case of SL: THIS IS NOT RIGHT! Maybe because of the 3 provinces that dropped off?')
print('\nFinal income poverty incidence:\n',iah_ntl['i_pov_pc_change'].mean())
print('Final consumption poverty incidence:\n',iah_ntl['c_pov_pc_change'].mean())

# Print out plots for myiah
myiah = myiah.reset_index()
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
if myCountry == 'FJ': myHaz = [['Ba','Lau','Tailevu'],get_all_hazards(myCountry,myiah),[1,10,100,500,1000]]
#elif myCountry == 'PH': myHaz = [['V - Bicol','II - Cagayan Valley','NCR','IVA - CALABARZON','ARMM','CAR'],['HU','EQ'],[10,25,50,100,250,500]]
#elif myCountry == 'PH': myHaz = [['ompong'],['HU','PF','SS'],[10,50,100,200,500]]
#elif myCountry == 'PH': myHaz = [['I - Ilocos','II - Cagayan Valley','CAR'],['HU','EQ'],[25,100]]
elif myCountry == 'PH': myHaz = [['VIII - Eastern Visayas'],['HU'],[100]]
elif myCountry == 'SL': myHaz = [['Rathnapura','Colombo'],get_all_hazards(myCountry,myiah),get_all_rps(myCountry,myiah)]
elif myCountry == 'MW': myHaz = [['Lilongwe','Chitipa'],get_all_hazards(myCountry,myiah),get_all_rps(myCountry,myiah)]


##################################################################
if myCountry == 'SL':

    listofquintiles=np.arange(0.20, 1.01, 0.20)
    quint_labels = ['Poorest\nquintile','Second','Third','Fourth','Wealthiest\nquintile']

    myiah['hhid'] = myiah['hhid'].astype('str')
    myiah = myiah.set_index('hhid')
    pmt,_ = get_pmt(myiah)
    myiah['PMT'] = pmt

    for _loc in myHaz[0]:
        for _haz in myHaz[1]:
            for _rp in myHaz[2]:

                plt.cla()
                _ = myiah.loc[(myiah[economy]==_loc)&(myiah['hazard']==_haz)&(myiah['rp']==_rp)].copy()

                _ = _.reset_index().groupby(economy,sort=True).apply(lambda x:match_percentiles(x,perc_with_spline(reshape_data(x.PMT),reshape_data(x.pcwgt_no),listofquintiles),'quintile','PMT'))

                _ = _.sort_values('PMT',ascending=True)

                _['pcwgt_cum_'+base_str] = _['pcwgt_'+base_str].cumsum()
                _['pcwgt_cum_'+my_PDS] = _['pcwgt_'+my_PDS].cumsum()

                _['dk0_cum'] = _[['pcwgt_'+base_str,'dk0']].prod(axis=1).cumsum()

                _['cost_cum_'+my_PDS] = _[['pcwgt_'+my_PDS,'help_received_'+my_PDS]].prod(axis=1).cumsum()
                # ^ cumulative cost
                _['cost_frac_'+my_PDS] = _[['pcwgt_'+my_PDS,'help_received_'+my_PDS]].prod(axis=1).cumsum()/_[['pcwgt_'+my_PDS,'help_received_'+my_PDS]].prod(axis=1).sum()
                # ^ cumulative cost as fraction of total

                # GET WELFARE COSTS
                _['dw_cum_'+base_str] = _[['pcwgt_'+base_str,'dw_'+base_str]].prod(axis=1).cumsum()
                # Include public costs in baseline (dw_cum)
                ext_costs_base = pd.read_csv(out_files+'public_costs_tax_'+base_str+'_.csv').set_index([economy,'hazard','rp'])

                ext_costs_base['dw_pub_curr'] = ext_costs_base['dw_pub']/_wprime
                ext_costs_base['dw_soc_curr'] = ext_costs_base['dw_soc']/_wprime
                ext_costs_base['dw_tot_curr'] = ext_costs_base[['dw_pub','dw_soc']].sum(axis=1)/_wprime

                ext_costs_base_sum = ext_costs_base.loc[ext_costs_base['contributer']!=ext_costs_base.index.get_level_values(event_level[0]),
                                                        ['dw_pub_curr','dw_soc_curr','dw_tot_curr']].sum(level=[economy,'hazard','rp']).reset_index()

                ext_costs_base_pub = float(ext_costs_base_sum.loc[(ext_costs_base_sum[economy]==_loc)&ext_costs_base_sum.eval('(hazard==@_haz)&(rp==@_rp)'),'dw_pub_curr'])
                ext_costs_base_soc = float(ext_costs_base_sum.loc[(ext_costs_base_sum[economy]==_loc)&ext_costs_base_sum.eval('(hazard==@_haz)&(rp==@_rp)'),'dw_soc_curr'])            
                ext_costs_base_sum = float(ext_costs_base_sum.loc[(ext_costs_base_sum[economy]==_loc)&ext_costs_base_sum.eval('(hazard==@_haz)&(rp==@_rp)'),'dw_tot_curr'])


                _['dw_cum_'+my_PDS] = _[['pcwgt_'+my_PDS,'dw_'+my_PDS]].prod(axis=1).cumsum()
                # ^ cumulative DW, with my_PDS implemented

                # Include public costs in pds_dw_cum
                ext_costs_pds = pd.read_csv(out_files+'public_costs_tax_'+my_PDS+'_.csv').set_index([economy,'hazard','rp'])

                ext_costs_pds['dw_pub_curr'] = ext_costs_pds['dw_pub']/_wprime
                ext_costs_pds['dw_soc_curr'] = ext_costs_pds['dw_soc']/_wprime
                ext_costs_pds['dw_tot_curr'] = ext_costs_pds[['dw_pub','dw_soc']].sum(axis=1)/_wprime

                ext_costs_pds_sum = ext_costs_pds.loc[(ext_costs_pds['contributer']!=ext_costs_pds.index.get_level_values(event_level[0])),
                                                      ['dw_pub_curr','dw_soc_curr','dw_tot_curr']].sum(level=[economy,'hazard','rp']).reset_index()

                ext_costs_pds_pub = float(ext_costs_pds_sum.loc[(ext_costs_pds_sum[economy]==_loc)&ext_costs_pds_sum.eval('(hazard==@_haz)&(rp==@_rp)'),'dw_pub_curr'])
                ext_costs_pds_soc = float(ext_costs_pds_sum.loc[(ext_costs_pds_sum[economy]==_loc)&ext_costs_pds_sum.eval('(hazard==@_haz)&(rp==@_rp)'),'dw_soc_curr'])
                ext_costs_pds_sum = float(ext_costs_pds_sum.loc[(ext_costs_pds_sum[economy]==_loc)&ext_costs_pds_sum.eval('(hazard==@_haz)&(rp==@_rp)'),'dw_tot_curr'])

                _['dw_cum_'+my_PDS] += (ext_costs_pds_pub + ext_costs_pds_soc)*_['cost_frac_'+my_PDS]
                _['delta_dw_cum_'+my_PDS] = _['dw_cum_'+base_str]-_['dw_cum_'+my_PDS]


                ### PMT-ranked population coverage [%]
                plt.plot(100.*_['pcwgt_cum_'+base_str]/_['pcwgt_'+base_str].sum(),
                         100.*_['dk0_cum']/_[['pcwgt_'+base_str,'dk0']].prod(axis=1).sum())
                plt.annotate('Total asset losses\n$'+str(round(1E-6*to_usd*_.iloc[-1]['dk0_cum'],1))+' mil.',xy=(0.1,0.85),xycoords='axes fraction',color=greys_pal[7],fontsize=10)
                if False:
                    plt.plot(100.*_['pcwgt_cum_'+base_str]/_['pcwgt_'+base_str].sum(),
                             100.*_['dk0_cum']/_[['pcwgt_'+base_str,'dk0']].prod(axis=1).sum())

                plt.xlabel('PMT-ranked population coverage [%]',labelpad=8,fontsize=10)
                plt.ylabel('Cumulative asset losses [%]',labelpad=8,fontsize=10)
                plt.xlim(0);plt.ylim(-0.1)
                plt.gca().xaxis.set_ticks([20,40,60,80,100])
                sns.despine()
                plt.grid(False)

                plt.gcf().savefig('../output_plots/SL/PMT/pcwgt_vs_dk0_'+_loc+'_'+_haz+'_'+str(_rp)+'.pdf',format='pdf',bbox_inches='tight')
                plt.cla()


                #####################################
                ### PMT threshold vs dk (normalized)
                plt.plot(_['PMT'],100.*_['dk0_cum']/_[['pcwgt_'+base_str,'dk0']].prod(axis=1).sum(),linewidth=1.8,zorder=99,color=q_colors[1])

                for _q in [1,2,3,4,5]:
                    _q_x = _.loc[_['quintile']==_q,'PMT'].max()
                    _q_y = 100.*_.loc[_['quintile']<=_q,['pcwgt_'+base_str,'dk0']].prod(axis=1).sum()/_[['pcwgt_'+base_str,'dk0']].prod(axis=1).sum()
                    if _q == 1: _q_yprime = _q_y/20

                    plt.plot([_q_x,_q_x],[0,_q_y],color=greys_pal[4],ls=':',linewidth=1.5,zorder=91)

                    _usd = ' mil.'
                    plt.annotate((quint_labels[_q-1]+'\n$'+str(round(1E-6*to_usd*_.loc[_['quintile']==_q,['pcwgt_'+base_str,'dk0']].prod(axis=1).sum(),1))+_usd),
                                 xy=(_q_x,_q_y+_q_yprime),color=greys_pal[6],ha='right',va='bottom',style='italic',fontsize=8,zorder=91)

                if False:
                    plt.scatter(_['PMT'],100.*_['dk0_cum']/_[['pcwgt_'+base_str,'dk0']].prod(axis=1).sum(),alpha=0.08,s=6,zorder=10,color=q_colors[1])

                plt.xlabel('Household income [PMT]',labelpad=8,fontsize=10)
                plt.ylabel('Cumulative asset losses [%]',labelpad=8,fontsize=10)
                plt.annotate('Total asset losses\n$'+str(round(1E-6*to_usd*_.iloc[-1]['dk0_cum'],1))+' mil.',xy=(0.1,0.85),xycoords='axes fraction',color=greys_pal[7],fontsize=10)
                plt.xlim(825);plt.ylim(-0.1)

                sns.despine()
                plt.grid(False)

                plt.gcf().savefig('../output_plots/SL/PMT/pmt_vs_dk_norm_'+_loc+'_'+_haz+'_'+str(_rp)+'.pdf',format='pdf',bbox_inches='tight')

                #####################################
                ### PMT threshold vs dk & dw
                plt.cla()
                plt.plot(_['PMT'],_['dk0_cum']*to_usd*1E-6,color=q_colors[1],linewidth=1.8,zorder=99)
                plt.plot(_['PMT'],_['dw_cum_'+base_str]*to_usd*1E-6,color=q_colors[3],linewidth=1.8,zorder=99)

                _y1 = 1.08; _y2 = 1.03
                if _['dk0_cum'].max() < _['dw_cum_'+base_str].max(): _y1 = 1.03; _y2 = 1.08

                plt.annotate('Total asset losses = $'+str(round(_['dk0_cum'].max()*to_usd*1E-6,1))+' million',xy=(0.02,_y1),
                             xycoords='axes fraction',color=q_colors[1],ha='left',va='top',fontsize=10,annotation_clip=False)
                plt.annotate('Total welfare losses = \$'+str(round(_['dw_cum_'+base_str].max()*to_usd*1E-6,1))+' million (+\$'+str(round(ext_costs_base_sum*to_usd*1E-6,1))+')',
                             xy=(0.02,_y2),xycoords='axes fraction',color=q_colors[3],ha='left',va='top',fontsize=10,annotation_clip=False)
                #plt.annotate('National welfare losses\n  $'+str(round(ext_costs_base_sum*to_usd*1E-6,1))+' million',xy=(0.02,0.77),
                #             xycoords='axes fraction',color=q_colors[3],ha='left',va='top',fontsize=10)            

                for _q in [1,2,3,4,5]:
                    _q_x = _.loc[_['quintile']==_q,'PMT'].max()
                    _q_y = max(_.loc[_['quintile']<=_q,['pcwgt_'+base_str,'dk0']].prod(axis=1).sum()*to_usd*1E-6,
                               _.loc[_['quintile']<=_q,['pcwgt_'+base_str,'dw_'+base_str]].prod(axis=1).sum()*to_usd*1E-6)
                    if _q == 1: _q_yprime = _q_y/25

                    plt.plot([_q_x,_q_x],[0,_q_y],color=greys_pal[4],ls=':',linewidth=1.5,zorder=91)
                    plt.annotate(quint_labels[_q-1].replace('\n',' '),xy=(_q_x,_q_y+7*_q_yprime),color=greys_pal[6],ha='right',va='bottom',style='italic',fontsize=8,zorder=91,annotation_clip=False)


                    # This figures out label ordering (are cumulative asset or cum welfare lossers higher?)
                    _cumk = round(_.loc[_['quintile']<=_q,['pcwgt_'+base_str,'dk0']].prod(axis=1).sum()*to_usd*1E-6,1)
                    _cumw = round(_.loc[_['quintile']<=_q,['pcwgt_'+base_str,'dw_'+base_str]].prod(axis=1).sum()*to_usd*1E-6,1)               
                    if _cumk >= _cumw: 
                        _yprime_k = 4*_q_yprime
                        _yprime_w = 1*_q_yprime
                    else: 
                        _yprime_k = 1*_q_yprime
                        _yprime_w = 4*_q_yprime


                    _qk = round(_.loc[_['quintile']==_q,['pcwgt_'+base_str,'dk0']].prod(axis=1).sum()*to_usd*1E-6,1)
                    _qw = round(_.loc[_['quintile']==_q,['pcwgt_'+base_str,'dw_'+base_str]].prod(axis=1).sum()*to_usd*1E-6,1)

                    plt.annotate('$'+str(_qk)+' mil.',xy=(_q_x,_q_y+_yprime_k),color=q_colors[1],ha='right',va='bottom',style='italic',fontsize=8,zorder=91,annotation_clip=False)
                    plt.annotate('$'+str(_qw)+' mil.',xy=(_q_x,_q_y+_yprime_w),color=q_colors[3],ha='right',va='bottom',style='italic',fontsize=8,zorder=91,annotation_clip=False)

                plt.xlabel('Household income [PMT]',labelpad=8,fontsize=10)
                plt.ylabel('Cumulative losses [mil. USD]',labelpad=8,fontsize=10)
                plt.xlim(825);plt.ylim(-0.1)

                plt.title(' '+str(_rp)+'-year '+haz_dict[_haz].lower()+' in '+_loc,loc='left',color=greys_pal[7],pad=30,fontsize=15)

                sns.despine()
                plt.grid(False)

                plt.gcf().savefig('../output_plots/SL/PMT/pmt_vs_dk0_'+_loc+'_'+_haz+'_'+str(_rp)+'.pdf',format='pdf',bbox_inches='tight')
                plt.close('all')


                #####################################
                ### Cost vs benefit of PMT

                _['dw_cum_'+base_str] += (ext_costs_base_pub+ext_costs_base_soc)*_['cost_frac_'+my_PDS]
                #*_[['pcwgt_'+base_str,'dk0']].prod(axis=1).cumsum()/_[['pcwgt_'+base_str,'dk0']].prod(axis=1).sum()
                # ^ include national costs in baseline dw
                _['delta_dw_cum_'+my_PDS] = _['dw_cum_'+base_str]-_['dw_cum_'+my_PDS]
                # redefine this because above changed

                plt.cla()
                plt.plot(_['PMT'],_['cost_cum_'+my_PDS]*to_usd*1E-6,color=q_colors[1],linewidth=1.8,zorder=99)
                plt.plot(_['PMT'],_['delta_dw_cum_'+my_PDS]*to_usd*1E-6,color=q_colors[3],linewidth=1.8,zorder=99)

                plt.annotate('PDS cost =\n$'+str(round(_['cost_cum_'+my_PDS].max()*to_usd*1E-6,2))+' mil.',
                             xy=(_['PMT'].max(),_['cost_cum_'+my_PDS].max()*to_usd*1E-6),color=q_colors[1],weight='bold',ha='left',va='top',fontsize=10,annotation_clip=False)
                plt.annotate('Avoided welfare\nlosses = $'+str(round(_.iloc[-1]['delta_dw_cum_'+my_PDS]*to_usd*1E-6,2))+' mil.',
                             xy=(_['PMT'].max(),_.iloc[-1]['delta_dw_cum_'+my_PDS]*to_usd*1E-6),color=q_colors[3],weight='bold',ha='left',va='top',fontsize=10)

                #for _q in [1,2,3,4,5]:
                #    _q_x = _.loc[_['quintile']==_q,'PMT'].max()
                #    _q_y = max(_.loc[_['quintile']<=_q,['pcwgt','dk0']].prod(axis=1).sum()*to_usd*1E-6,
                #               _.loc[_['quintile']<=_q,['pcwgt','dw_no']].prod(axis=1).sum()*to_usd*1E-6)
                #    if _q == 1: _q_yprime = _q_y/20

                #    plt.plot([_q_x,_q_x],[0,_q_y],color=greys_pal[4],ls=':',linewidth=1.5,zorder=91)
                #    plt.annotate(quint_labels[_q-1],xy=(_q_x,_q_y+_q_yprime),color=greys_pal[6],ha='right',va='bottom',style='italic',fontsize=8,zorder=91)

                plt.xlabel('Upper PMT threshold for post-disaster support',labelpad=8,fontsize=12)
                plt.ylabel('Cost & benefit [mil. USD]',labelpad=8,fontsize=12)
                plt.xlim(825)#;plt.ylim(0)

                plt.title(' '+str(_rp)+'-year '+haz_dict[_haz].lower()+'\n  in '+_loc,loc='left',color=greys_pal[7],pad=25,fontsize=15)
                plt.annotate(pds_dict[my_PDS],xy=(0.02,1.03),xycoords='axes fraction',color=greys_pal[6],ha='left',va='bottom',weight='bold',style='italic',fontsize=8,zorder=91,clip_on=False)

                plt.plot(plt.gca().get_xlim(),[0,0],color=greys_pal[2],linewidth=0.90)
                sns.despine(bottom=True)
                plt.grid(False)

                plt.gcf().savefig('../output_plots/SL/PMT/pmt_dk_vs_dw_'+_loc+'_'+_haz+'_'+str(_rp)+'_'+my_PDS+'.pdf',format='pdf',bbox_inches='tight')
                plt.close('all')            

                #####################################
                ### Cost vs benefit of PMT
                _ = _.fillna(0)
                #_ = _.loc[_['pcwgt_'+my_PDS]!=0].copy()
                _ = _.loc[(_['help_received_'+my_PDS]!=0)&(_['pcwgt_'+my_PDS]!=0)].copy()

                #_['dw_cum_'+my_PDS] = _[['pcwgt_'+my_PDS,'dw_'+my_PDS]].prod(axis=1).cumsum()
                #_['dw_cum_'+my_PDS] += ext_costs_pds_pub + ext_costs_pds_soc*_['cost_frac_'+my_PDS]
                # ^ unchanged from above

                _c1,_c1b = paired_pal[2],paired_pal[3]
                _c2,_c2b = paired_pal[0],paired_pal[1]

                _window = 100
                if _.shape[0] < 100: _window = int(_.shape[0]/5)

                plt.cla()

                _y_values_A = (_['cost_cum_'+my_PDS]*to_usd).diff()/_['pcwgt_'+my_PDS]
                _y_values_B = pd.rolling_mean((_['cost_cum_'+my_PDS]*to_usd).diff()/_['pcwgt_'+my_PDS],_window)

                if _y_values_A.max() >= 1.25*_y_values_B.max() or _y_values_A.min() <= 0.75*_y_values_B.min():
                    plt.scatter(_['PMT'],(_['cost_cum_'+my_PDS]*to_usd).diff()/_['pcwgt_'+my_PDS],color=_c1,s=4,zorder=98,alpha=0.25)
                    plt.plot(_['PMT'],pd.rolling_mean((_['cost_cum_'+my_PDS]*to_usd).diff()/_['pcwgt_'+my_PDS],_window),color=_c1b,lw=1.0,zorder=98)
                else: 
                    plt.plot(_['PMT'],(_['cost_cum_'+my_PDS]*to_usd).diff()/_['pcwgt_'+my_PDS],color=_c1b,lw=1.0,zorder=98)

                plt.scatter(_['PMT'],(_['delta_dw_cum_'+my_PDS]*to_usd).diff()/_['pcwgt_'+my_PDS],color=_c2,s=4,zorder=98,alpha=0.25)
                plt.plot(_['PMT'],pd.rolling_mean((_['delta_dw_cum_'+my_PDS]*to_usd).diff()/_['pcwgt_'+my_PDS],_window),color=_c2b,lw=1.0,zorder=98)
                _y_min = 1.05*pd.rolling_mean((_['delta_dw_cum_'+my_PDS]*to_usd).diff()/_['pcwgt_'+my_PDS],_window).min()
                _y_max = 1.1*max(pd.rolling_mean((_['delta_dw_cum_'+my_PDS]*to_usd).diff()/_['pcwgt_'+my_PDS],_window).max(),
                                 1.05*((_['cost_cum_'+my_PDS]*to_usd).diff()/_['pcwgt_'+my_PDS]).mean()+_q_yprime)

                for _q in [1,2,3,4,5]:
                    _q_x = min(1150,_.loc[_['quintile']==_q,'PMT'].max())
                    #_q_y = max(_.loc[_['quintile']<=_q,['pcwgt_'+my_PDS,'dk0']].prod(axis=1).sum()*to_usd,
                    #           _.loc[_['quintile']<=_q,['pcwgt_'+my_PDS,'dw_no']].prod(axis=1).sum()*to_usd))
                    if _q == 1: 
                        _q_xprime = (_q_x-840)/40
                        _q_yprime = _y_max/200

                    plt.plot([_q_x,_q_x],[_y_min,_y_max],color=greys_pal[4],ls=':',linewidth=1.5,zorder=91)
                    plt.annotate(quint_labels[_q-1],xy=(_q_x-_q_xprime,_y_max),color=greys_pal[6],ha='right',va='top',style='italic',fontsize=7,zorder=99)

                #toggle this
                plt.annotate('PDS cost',xy=(_['PMT'].max()-_q_xprime,((_['cost_cum_'+my_PDS]*to_usd).diff()/_['pcwgt_'+my_PDS]).mean()+_q_yprime),
                             color=_c1b,weight='bold',ha='right',va='bottom',fontsize=8,annotation_clip=False)

                #plt.annotate('Avoided\nwelfare losses',xy=(_['PMT'].max()-_q_xprime,pd.rolling_mean((_['delta_dw_cum']*to_usd/_['pcwgt_'+my_PDS]).diff(),_window).min()+_q_yprime),
                #             color=_c2b,weight='bold',ha='right',va='bottom',fontsize=8)

                plt.xlabel('Upper PMT threshold for post-disaster support',labelpad=8,fontsize=12)
                plt.ylabel('Marginal impact at threshold [USD per next enrollee]',labelpad=8,fontsize=12)

                plt.title(str(_rp)+'-year '+haz_dict[_haz].lower()+' in '+_loc,loc='right',color=greys_pal[7],pad=20,fontsize=15)
                plt.annotate(pds_dict[my_PDS],xy=(0.99,1.02),xycoords='axes fraction',color=greys_pal[6],ha='right',va='bottom',weight='bold',style='italic',fontsize=8,zorder=91,clip_on=False)

                plt.plot([840,1150],[0,0],color=greys_pal[2],linewidth=0.90)
                plt.xlim(840,1150);plt.ylim(_y_min,_y_max)
                sns.despine(bottom=True)
                plt.grid(False)

                plt.gcf().savefig('../output_plots/SL/PMT/pmt_slope_cost_vs_benefit_'+_loc+'_'+_haz+'_'+str(_rp)+'_'+my_PDS+'.pdf',format='pdf',bbox_inches='tight')
                plt.close('all')  


##################################################################
# This code generates output on poverty dimensions
# ^ this is by household (iah != iah_avg here)
if False:

    _myiah = myiah.reset_index().set_index(event_level+['hhid','affected_cat','helped_cat'])[['pcwgt_no',
                                                                                              'c_initial','c_post_reco',
                                                                                              'i_pre_reco','c_pre_reco','dc_pre_reco',
                                                                                              'pov_line','sub_line']].copy()
    _myiah = _myiah.loc[_myiah['pcwgt_no']!=0]
    
    run_poverty_duration_plot(myCountry,myHaz[1][0])
    #                                   ^ first hazard in the country we're running
    run_poverty_tables_and_maps(myCountry,_myiah,event_level,myHaz[1][0],drop_spots=places_to_drop)
    map_recovery_time(myCountry,myHaz[1][0],drop_spots=places_to_drop)

##################################################################
# This code generates the histograms showing income before & after disaster (in USD)
# ^ this is at household level (iah != iah_avg here)
if False:         
    with Pool(processes=2,maxtasksperchild=1) as pool:
        print('LAUNCHING',len(list(product(myHaz[0],myHaz[1],myHaz[2]))),'THREADS')
        pool.starmap(plot_income_and_consumption_distributions,list(product([myCountry],[myiah.copy()],myHaz[0],myHaz[1],myHaz[2],[True],['USD'])))

##################################################################
# This code generates the histograms showing income before & after disaster (in local_curr)
# ^ this is at household level (iah != iah_avg here)
if False:
    with Pool(processes=2,maxtasksperchild=1) as pool:
        print('LAUNCHING',len(list(product(myHaz[0],myHaz[1],myHaz[2]))),'THREADS')
        pool.starmap(plot_income_and_consumption_distributions,list(product([myCountry],[myiah.copy()],myHaz[0],myHaz[1],myHaz[2],[True])))

##################################################################
# This code generates the histograms including [k,dk,dc,dw,&pds]
# ^ this is by province/region, so it will use myiah (iah = iah_avg here)
if True:
    with Pool(processes=2,maxtasksperchild=1) as pool:
        print('LAUNCHING',len(list(product(myHaz[0],myHaz[1],myHaz[2],[my_PDS]))),'THREADS')
        pool.starmap(plot_impact_by_quintile,list(product([myCountry],myHaz[0],myHaz[1],myHaz[2],[myiah.copy()],[my_PDS])))


##################################################################
# This code generates the histograms 
# ^ this is only for affected households (iah = iah_avg here) <--because we're summing, not averaging
if True:
    with Pool(processes=2,maxtasksperchild=1) as pool:
        print('LAUNCHING',len(list(product(myHaz[0],myHaz[1],myHaz[2]))),'THREADS')
        pool.starmap(plot_relative_losses,list(product([myCountry],myHaz[0],myHaz[1],myHaz[2],[myiah.copy()])))  
