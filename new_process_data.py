####################################
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

places_to_drop = ['Jaffna','Matara','Kilinochchi'] if myCountry == 'SL' else None
#if myCountry == 'MW': places_to_drop = ['Blantyre City','Lilongwe City','Mzuzu City','Zomba City']

_mapres = 1000


##################################
# Set directories (where to look for files)
out_files = os.getcwd()+'/../output_country/'+myCountry+'/'


##################################
# Set policy params
drm_pov_sign = -1 # toggle subtraction or addition of dK to affected people's incomes

my_PDS = 'unif_poor'#'scaleout_samurdhi_universal'#'samurdhi_scaleup'#'samurdhi_scaleup00'#
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
#pds_options = all_pds_options

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
if True:
    df_prov = df[['dKtot','dWtot_currency']].copy()
    df_prov['gdp'] = df[['pop','gdp_pc_prov']].prod(axis=1).copy()
    results_df = macro.reset_index().set_index([economy,'hazard'])
    results_df = results_df.loc[results_df.rp==100,'dk_event'].sum(level='hazard')
    results_df = results_df.rename(columns={'dk_event':'dk_event_100'})
    results_df = pd.concat([results_df,df_prov.reset_index().set_index([economy,'hazard']).sum(level='hazard')['dKtot']],axis=1,join='inner')
    results_df.columns = ['dk_event_100','AAL']
    #results_df.to_csv(out_files+'results_table_new.csv')
    print('Writing '+out_files+'results_table_new.csv')
    

#####
# HACK here to select subsets of the population
#select_ethnicity = True if myCountry == 'SL' else False
select_ethnicity = False
sub_eth = ''

_df = pd.read_csv('../intermediate/'+myCountry+'/cat_info.csv').set_index(economy)
pop_df = _df['pcwgt'].sum(level=economy).to_frame(name='pop')

if myCountry == 'SL' and select_ethnicity:
    #sub_eth = 'Sinhalese'
    sub_eth = 'non-Sinhalese'
 
    iah = iah.loc[iah.eval('ethnicity==1' if sub_eth == 'Sinhalese' else 'ethnicity!=1')]

    _df = pd.read_csv('../intermediate/'+myCountry+'/cat_info.csv').set_index(economy)
    pop_df = _df.loc[_df.eval('ethnicity==1' if sub_eth == 'Sinhalese' else 'ethnicity!=1'),'pcwgt'].sum(level=economy).to_frame(name='pop')


##################################
# Create additional dfs
#

# 1) Get dataframe with expectation values: one entry per household, averaged over (a/na & helped/not_helped)
iah_avg = get_expectation_values_df(myCountry,economy,iah,pds_options,base_str=base_str,use_aewgt=use_aewgt)

# 1.5) get a dataframe with average pre-disaster consumption, by district & ethnicity (sin & non-sin)
if select_ethnicity: hh_consumption = pd.read_csv('../intermediate/SL/hh_consumption_pc.csv').set_index('district')
else:
    try:
        _ = iah.loc[iah.pcwgt_no!=0].reset_index().set_index('hhid')
        hh_consumption = pd.merge(_[['c_initial']].mean(level='hhid').reset_index(),
                                  _.loc[~(_.index.duplicated()),['district','quintile','ethnicity','pcwgt_no']].reset_index(),on='hhid').set_index(['district','hhid'])
        
        hh_consumption['ethnicity'] = hh_consumption.apply(lambda x:('Sinhalese' if x.ethnicity==1 else 'non-Sinhalese'),axis=1)
        hh_consumption = hh_consumption.reset_index().set_index(['district','ethnicity'])
        hh_consumption['dist_c'] = (hh_consumption[['pcwgt_no','c_initial']].prod(axis=1).groupby(['district','ethnicity']).transform('sum')
                                    /hh_consumption['pcwgt_no'].groupby(['district','ethnicity']).transform('sum'))
        hh_consumption = hh_consumption.loc[~hh_consumption.index.duplicated(),['dist_c']].unstack('ethnicity').fillna(0)
        
        hh_consumption.columns = hh_consumption.columns.get_level_values(1)
        hh_consumption.index.name = 'district'
        
        hh_consumption.to_csv('../intermediate/SL/hh_consumption_pc.csv')

    except: pass

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
iah_out.to_csv(out_files+'risk_by_economy_hazard'+('_'+sub_eth.lower().replace('-','').replace(' ','') if select_ethnicity else '')+'.csv')
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
            color_maper=(plt.cm.get_cmap('GnBu')), 
            label=(haz_dict[_h]+' risk [mil. US$ per year]\nNational total = US$ '
                   +str(int(round(float((iah_out.loc[(slice(None),(_h)),'Asset risk'].sum(level=economy)*1E-6*to_usd).sum()),0)))+'M'),
            new_title='',
            do_qualitative=False,
            res=_mapres,
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
if len(_.columns)>1:
    _['Total'] = _.sum(axis=1)
    _.loc['Total'] = _.sum()
    _.sort_values('Total',ascending=False).round(0).to_latex('latex/'+myCountry+'/risk_by_economy_hazard.tex')

    # USD
    _ = (no_pds['Asset risk']/1.E6).round(1).unstack().copy()*to_usd
    _['Total'] = _.sum(axis=1)
    _.loc['Total'] = _.sum(axis=0)
    _.sort_values('Total',ascending=False).round(1).to_latex('latex/'+myCountry+'/risk_by_economy_hazard_usd.tex')

else: 
    _['USD'] = _*to_usd*1E3 if myCountry == 'SL' else _*to_usd
    # For SL, losses are coming out in thousands of USD, everywhere else in millions
    _.loc['Total'] = _.sum()
    _.sort_values('USD',ascending=False).round(0).astype('int').to_latex('latex/'+myCountry+'/risk_by_economy_hazard.tex')    


# Sum over hazards
no_pds = no_pds.sum(level=economy)
no_pds[['Asset risk','Well-being risk']]/=1.E9 # no_pds is thousands [1E3]
no_pds['SE capacity']  = 100.*no_pds['Asset risk']/no_pds['Well-being risk']

no_pds.loc['Total'] = [float(no_pds['Asset risk'].sum()),
                        float(no_pds['Well-being risk'].sum()),
                        float(no_pds['Asset risk'].sum()/no_pds['Well-being risk'].sum())]
no_pds['SE capacity']  = 100.*no_pds['Asset risk']/no_pds['Well-being risk']

#put population in
no_pds = pd.concat([no_pds,pop_df],axis=1,join_axes=[no_pds.index])
no_pds.loc['Total','pop'] = no_pds['pop'].sum()

# Write out risk to assets & welfare & resilience by economy
# Local currency
no_pds.to_csv(out_files+'risk_by_economy'+('_'+sub_eth.lower().replace('-','').replace(' ','') if select_ethnicity else '')+'.csv')

no_pds[['Asset risk',
         'SE capacity',
         'Well-being risk']].sort_values(['Well-being risk'],ascending=False).round(0).to_latex('latex/'+myCountry+'/risk_by_economy.tex')
#USD
no_pds = no_pds.drop('Total')
no_pds[['Asset risk','Well-being risk']]*=to_usd*1E3 # no_pds is millions of USD now
no_pds.loc['Total'] = [float(no_pds['Asset risk'].sum()),
                       float(no_pds['Well-being risk'].sum()),
                       float(no_pds['Asset risk'].sum()/no_pds['Well-being risk'].sum()),
                       float(no_pds['pop'].sum())]

no_pds['SE capacity']  = (100.*no_pds['Asset risk']/no_pds['Well-being risk'])/1E3# offset below

(1.E3*no_pds).round(1).to_csv(out_files+'risk_by_economy_usd'+('_'+sub_eth.lower().replace('-','').replace(' ','') if select_ethnicity else '')+'.csv')
(1.E3*no_pds[['Asset risk','SE capacity','Well-being risk']]).sort_values(['Well-being risk'],ascending=False).round(0).astype('int').to_latex('latex/'+myCountry+'/risk_by_economy_usd.tex')
print('Wrote latex! Sums:\n',no_pds.loc['Total',['Asset risk','Well-being risk']])


# LOAD SVG FILES
svg_file = get_svg_file(myCountry)

#######################
if False:
    curr_dict = {'PH':'PHP',
                 'SL':'SLR',
                 'FJ':'FJD',
                 'RO':'RON',
                 'MW':'MWK',
                 'BO':'BOB'}
    
    # GDP prov
    grdp = 1E-6*df[['pop','gdp_pc_prov']].prod(axis=1).mean(level=economy).squeeze()

    lcurr = get_currency(myCountry)[0]
    for _curr in ['USD',(curr_dict[myCountry] if myCountry in curr_dict else 'XXX')]:
        
        print('--> Map asset losses in '+_curr)
        natl_sum = no_pds['Asset risk'].drop('Total').sum()
        if _curr == 'USD': grdp*=get_currency(myCountry)[2]
        else: natl_sum *= 1/get_currency(myCountry)[2]

        if natl_sum >= 1E5: 
            natl_sum*=1E-3
            natl_mil_vs_bil = ' billion'
        else: 
            natl_mil_vs_bil = ' million'
        natl_sum = round(natl_sum,1)

        make_map_from_svg(
            no_pds['Asset risk'].drop('Total')/(1E3*get_currency(myCountry)[2]) if _curr != 'USD' else no_pds['Asset risk'].drop('Total'),
            svg_file,
            outname=myCountry+'_asset_risk_'+_curr.lower(),
            color_maper=plt.cm.get_cmap('GnBu'),
            #svg_handle = 'reg',
            label=('Multihazard asset risk by '+economy.lower()+' ['+_curr.replace('USD','mil. US\$').replace('PHP','bil. PHP')+' per year]'
                   +'\nNational total = '+_curr.replace('USD','US\$')+str(natl_sum)+natl_mil_vs_bil),
            new_title='Asset losses ['+_curr.replace('USD','mil. US\$')+'per year]',
            do_qualitative=False,
            drop_spots=places_to_drop,
            res=_mapres)
        
        natl_avg = str(round((100.*no_pds['Asset risk'].drop('Total').sum()/grdp.sum()),2))
        make_map_from_svg(
            100.*(no_pds['Asset risk']/grdp).drop('Total'),
            svg_file,
            outname=myCountry+'_asset_risk_over_reg_gdp',
            color_maper=plt.cm.get_cmap('GnBu'),
            #svg_handle = 'reg',
            label='Multihazard asset risk [% of AHE per year]\nNational avg. = '+natl_avg+'%',
            new_title='Asset losses [% of AHE per year]',
            do_qualitative=False,
            drop_spots=places_to_drop,
            res=_mapres)   

        print('--> Map welfare losses in '+_curr)
        natl_sum = no_pds['Well-being risk'].drop('Total').sum()
        if _curr == 'USD': pass#grdp*=get_currency(myCountry)[2] # already done above
        else: natl_sum *= 1/get_currency(myCountry)[2]

        if natl_sum >= 1E5: 
            natl_sum*=1E-3
            natl_mil_vs_bil = ' billion'
        else: 
            natl_mil_vs_bil = ' million'
        natl_sum = round(natl_sum,1)

        make_map_from_svg(
            no_pds['Well-being risk'].drop('Total')/(1E3*get_currency(myCountry)[2]) if _curr != 'USD' else no_pds['Well-being risk'].drop('Total'),
            svg_file,
            outname=myCountry+'_welf_risk_'+_curr.lower(),
            color_maper=plt.cm.get_cmap('OrRd'),
            #svg_handle = 'reg',
            label=('Wellbeing losses by '+economy.lower()+' ['+_curr.replace('USD','mil. US\$').replace('PHP','bil. PHP')+' per year]'
                   +'\nNational total = '+_curr.replace('USD','US\$')+str(natl_sum)+natl_mil_vs_bil),
            new_title='Wellbeing losses ['+_curr.replace('USD','mUS\$').replace('PHP','bil. PHP')+' per year]',
            do_qualitative=False,
            drop_spots=places_to_drop,
            res=_mapres)
        
        natl_avg = str(round(100*(no_pds['Well-being risk'].drop('Total').sum()/grdp.sum()),2))
        make_map_from_svg(
            (100.*no_pds['Well-being risk']/grdp).drop('Total'),
            svg_file,
            outname=myCountry+'_welf_risk_over_reg_gdp',
            color_maper=plt.cm.get_cmap('OrRd'),
            #svg_handle = 'reg',
            label='Wellbeing losses [% of AHE per year]\nNational avg. = '+natl_avg+'%',
            new_title='Wellbeing losses [% of AHE per year]',
            do_qualitative=False,
            drop_spots=places_to_drop,
            res=_mapres)
        

    #######################
    print('Map Resilience')
    natl_average = 100.*(no_pds['Asset risk'].drop('Total').sum()/no_pds['Well-being risk'].drop('Total').sum()) 
    make_map_from_svg(
        100.*(no_pds['Asset risk']/no_pds['Well-being risk']).drop('Total'), 
        svg_file,
        outname=myCountry+'_resilience'+('_'+sub_eth.lower().replace('-','') if select_ethnicity else ''),
        color_maper=plt.cm.get_cmap('RdYlGn'),
        #svg_handle = 'reg',
        label='Socioeconomic resilience'+(r' $\endash$ '+sub_eth+' ' if select_ethnicity else ' ')+'[%]\n'+r'National avg. = '+str(int(round(natl_average,0)))+'%',
        new_title='Socioeconomic resilience [%]',
        do_qualitative=False,
        res=_mapres,
        #force_min = 0,
        drop_spots='Jaffna' if sub_eth == 'non-Sinhalese' else None,
        force_max = 110 if myCountry=='SL' else None)
    
    #######################
    print('Map per capita risk')
    plt.close('all')
        
    natl_average = str(round(1E6*no_pds['Asset risk'].drop('Total').sum()/no_pds['pop'].drop('Total').sum(),1))+'0'
    make_map_from_svg(
        1E6*(no_pds['Asset risk']/no_pds['pop']).drop('Total'),
        svg_file,
        outname=myCountry+'_asset_risk_per_cap_'+sub_eth.lower().replace('-','')+'_usd',
        color_maper=plt.cm.get_cmap('GnBu'),
        #svg_handle = 'reg',
        label=('Asset losses '+r'$\endash$ '+sub_eth+' [per person, per year]\nNational avg. = $'+natl_average if myCountry == 'SL' and select_ethnicity 
               else 'Asset losses [per person, per year]\nNational avg. = $'+natl_average+' per cap.'),
        new_title='Asset losses [US\$ per person, per year]',
        do_qualitative=False,
        drop_spots=places_to_drop,
        #force_max=60,
        res=_mapres)
    
    natl_average = str(round(1E6*no_pds['Well-being risk'].dropna().drop('Total').sum()/no_pds['pop'].drop('Total').sum(),1))+'0'
    make_map_from_svg(
        1E6*(no_pds['Well-being risk']/no_pds['pop']).drop('Total'),
        svg_file,
        outname=myCountry+'_welf_risk_per_cap_'+sub_eth.lower().replace('-','')+'_usd',
        color_maper=plt.cm.get_cmap('OrRd'),
        #svg_handle = 'reg',
        label=('Wellbeing losses '+r'$\endash$ '+sub_eth+' [US\$ per person, per year]\nNational avg. = $'+natl_average if myCountry == 'SL' and select_ethnicity 
               else 'Wellbeing losses [US\$ per person, per year]\nNational avg. = $'+natl_average+' per cap.'),
        new_title='Wellbeing losses [US\$ per person, per year]',
        do_qualitative=False,
        drop_spots=places_to_drop,
        #force_max=60,
        res=_mapres)
        

    purge('img/','map_of_*.png')
    purge('img/','legend_of_*.png')
    purge('img/','map_of_*.svg')
    purge('img/','legend_of_*.svg')
    plt.close('all')

####################################
# Make plot comparing resileince of sinhalese * non-sinhalese
def is_majority(df): return df['Sinhalese population'] >= df['non-Sinhalese population'] 
def calc_minority_frac(df): 
    try: return 100.*df['non-Sinhalese population']/df[['non-Sinhalese population','Sinhalese population']].sum()
    except ZeroDivisionError: return 100.

color_sin = paired_pal[9]
color_ns = paired_pal[3]

if myCountry == 'SL':

    _sin = pd.read_csv(out_files+'risk_by_economy_usd_sinhalese.csv').set_index(economy).rename(columns={'SE capacity':'Resilience of Sinhalese',
                                                                                                         'pop':'Sinhalese population'})
    _nos = pd.read_csv(out_files+'risk_by_economy_usd_nonsinhalese.csv').set_index(economy).rename(columns={'SE capacity':'Resilience of non-Sinhalese',
                                                                                                            'pop':'non-Sinhalese population'})
    _comp = pd.concat([_sin[['Resilience of Sinhalese','Sinhalese population']],
                       _nos[['Resilience of non-Sinhalese','non-Sinhalese population']],
                       hh_consumption],axis=1).fillna(0)
    _comp.index.name = 'district'


    _comp['is_majority'] = _comp.apply(lambda x:is_majority(x),axis=1)
    _comp['minority_frac'] = _comp.apply(lambda x:calc_minority_frac(x),axis=1)    

    # plot sinhalese resilience vs non-sinhalese resilience
    plt.figure(figsize=(6,6))
    ax = plt.gca()

    plt.plot([0,110],[0,110],color=greys_pal[6],alpha=0.30,lw=1.0,ls=':')
    plt.scatter(_comp.loc[_comp['is_majority'],'Resilience of Sinhalese'],
                _comp.loc[_comp['is_majority'],'Resilience of non-Sinhalese'],
                label='Majority Sinhalese districts',alpha=0.7,s=25,color=color_sin)
    plt.scatter(_comp.loc[_comp['is_majority'] == False,'Resilience of Sinhalese'],
                _comp.loc[_comp['is_majority'] == False,'Resilience of non-Sinhalese'],
                label='Majority-minority districts',alpha=0.9,s=25,color=color_ns)

    #    plt.scatter(_comp.loc[_comp.district!='Total','diff'],_comp.loc[_comp.district!='Total'].index,alpha=0.5,s=20,zorder=91,edgecolor=color_ns,color=color_ns)
    #    plt.scatter(_comp.loc[_comp.district=='Total','diff'],_comp.loc[_comp.district=='Total'].index,alpha=0.95,s=24,zorder=91,edgecolor=color_ns,facecolors='none')


    title_legend_labels(plt.gca(),pais='',
                        lab_x=r'Sinhalese $\endash$ socioeconomic resilience (%)',
                        lab_y=r'non-Sinhalese $\endash$ socioeconomic resilience (%)',lim_x=[0,110],lim_y=[0,110],leg_fs=9)
    sns.despine()
    plt.grid(False)

    plt.gcf().savefig('../output_plots/SL/conflict/resilience_sin_vs_ns.pdf',format='pdf',bbox_inches='tight')
    plt.close('all')

    # plot ratio (non-Sinhalese/Sinhalese) resilience vs ratio (non-Sinhalese/Sinhalese) population
    plt.scatter(_comp['minority_frac'],_comp['Resilience of non-Sinhalese']/_comp['Resilience of Sinhalese'],alpha=0.5,s=20,color=greys_pal[6])

    plt.plot([-5,105],[1,1],color=greys_pal[6],alpha=0.30,lw=1.0,ls=':')
    
    linear_fit = np.polyfit(_comp['minority_frac'].drop(['Batticaloa','Mannar']),
                            (_comp['Resilience of non-Sinhalese']/_comp['Resilience of Sinhalese']).drop(['Batticaloa','Mannar']),1)
    r_x, r_y = zip(*((i, i*linear_fit[0] + linear_fit[1]) for i in 100.*_comp['non-Sinhalese population']/(_comp['Sinhalese population']+_comp['non-Sinhalese population'])))
    plt.plot(r_x, r_y,color=greys_pal[6],alpha=0.9,lw=1.25,ls='-')

    batticaloa_x = str(round(_comp.loc['Batticaloa','minority_frac'],1))
    batticaloa_y = str(round((_comp['Resilience of non-Sinhalese']/_comp['Resilience of Sinhalese'])['Batticaloa'],1))

    mannar_x = str(round(_comp.loc['Mannar','minority_frac'],1))
    mannar_y = str(round((_comp['Resilience of non-Sinhalese']/_comp['Resilience of Sinhalese'])['Mannar'],1))
    
    plt.annotate('Outliers:\nMannar: ('+mannar_x+','+mannar_y+')\nBatticaloa: ('+batticaloa_x+','+batticaloa_y+')',
                 xy=(0.95,1.20),xycoords='axes fraction',xytext=(0.95,1.10),
                 arrowprops=dict(arrowstyle="-",facecolor=greys_pal[8],connectionstyle="angle,angleA=0,angleB=90,rad=5"),
                 annotation_clip=False,size=8,weight='light',ha='right',va='top',color=greys_pal[8],clip_on=False)
    
    title_legend_labels(plt.gca(),pais='',
                        lab_x=r'non-Sinhalese fraction of population',
                        lab_y=r'Ratio: non-Sinhalese/Sinhalese resilience',lim_y=[0,2],lim_x=[0,101],leg_fs=9,do_leg=False)
    sns.despine()
    plt.grid(False)

    plt.gcf().savefig('../output_plots/SL/conflict/pop_vs_res_sin_vs_ns.pdf',format='pdf',bbox_inches='tight')
    plt.close('all')   
    
    ###########################
    _comp = _comp.sort_values('Resilience of Sinhalese',ascending=True).reset_index()

    if True:
        for _ix in _comp.index:
            plt.plot([_comp.loc[_ix,'Resilience of non-Sinhalese'],_comp.loc[_ix,'Resilience of Sinhalese']],[_ix,_ix],
                     ls=':',lw=0.5,color=greys_pal[1],zorder=90)
            plt.annotate(_comp.loc[_ix,'district'].replace('Total','National avg.'),xy=(_comp.loc[_ix,['Resilience of non-Sinhalese','Resilience of Sinhalese']].max()+2.5,_ix),
                         ha='left',va='center',clip_on=False,annotation_clip=False,
                         fontsize=(6.5 if _comp.loc[_ix,'is_majority']==False else 6),
                         weight=('bold' if _comp.loc[_ix,'is_majority']==False else 'light'))
        plt.scatter(_comp['Resilience of non-Sinhalese'],_comp.index,alpha=0.5,s=20,color=color_ns,zorder=91)
        plt.scatter(_comp['Resilience of Sinhalese'],_comp.index,alpha=0.5,s=20,color=color_sin,zorder=91)
        title_legend_labels(plt.gca(),pais='',lab_x=r'Socioeconomic resilience',lab_y='',lim_x=[0,104],leg_fs=9,do_leg=False)
        sns.despine(left=True)
        plt.gca().yaxis.set_ticks([])

    if False:
        for _ix in _comp.index:               
            plt.plot([_comp.loc[_ix,'minority_frac'],_comp.loc[_ix,'minority_frac']],
                     [_comp.loc[_ix,'Resilience of non-Sinhalese'],_comp.loc[_ix,'Resilience of Sinhalese']],
                     ls=':',lw=0.5,color=greys_pal[1])

        plt.scatter(_comp['minority_frac'],_comp['Resilience of non-Sinhalese'],alpha=0.5,s=20,color=color_ns,label='non-Sinhalese')    
        plt.scatter(_comp['minority_frac'],_comp['Resilience of Sinhalese'],alpha=0.5,s=20,color=color_sin,label='Sinhalese')
        title_legend_labels(plt.gca(),pais='',lab_x=r'Minority fraction of total population',lab_y='Socioeconomic resilience',lim_x=[0,101],leg_fs=9,do_leg=True)
        sns.despine()

    plt.grid(False)

    plt.gcf().savefig('../output_plots/SL/conflict/pop_vs_res_sin_vs_ns_2.pdf',format='pdf',bbox_inches='tight')
    plt.close('all')   


    ###########################
    _comp['diff'] = _comp['Resilience of non-Sinhalese']-_comp['Resilience of Sinhalese']
    _comp = _comp.sort_values('diff',ascending=False).reset_index().drop('index',axis=1)

    for _ix in _comp.index:
        plt.plot([0,_comp.loc[_ix,'diff']],[_ix,_ix],
                 ls=':',lw=0.5,color=greys_pal[1],zorder=90)
        plt.annotate(_comp.loc[_ix,'district'].replace('Total','National average'),xy=(max(float(_comp.loc[_ix,['diff']]),0)+2.5,_ix),
                     ha='left',va='center',clip_on=False,annotation_clip=False,
                     fontsize=(6.5 if _comp.loc[_ix,'is_majority']==False else 6),
                     weight=('bold' if _comp.loc[_ix,'is_majority']==False else 'light'))
        
    plt.scatter(_comp.loc[_comp.district!='Total','diff'],_comp.loc[_comp.district!='Total'].index,alpha=0.5,s=20,zorder=91,edgecolor=color_ns,color=color_ns)
    plt.scatter(_comp.loc[_comp.district=='Total','diff'],_comp.loc[_comp.district=='Total'].index,alpha=0.95,s=24,zorder=91,edgecolor=color_ns,facecolors='none')
    
    plt.scatter([0 for _ in _comp.index][:-1],_comp.loc[_comp.district!='Total'].index,alpha=0.5,s=20,zorder=91,edgecolor=color_sin,color=color_sin)
    plt.scatter([0],_comp.loc[_comp.district=='Total'].index,alpha=0.95,s=24,zorder=91,edgecolor=color_sin,facecolors='none')

    _ylim = plt.gca().get_ylim()

    _shift_x = 3.5
    _shift_y = 0.9
    plt.annotate('non-Sinhalese',xy=(_comp.iloc[-1]['diff'],max(_comp.index)+0.35),xytext=(_comp.iloc[-1]['diff']+_shift_x,max(_comp.index)+_shift_y),fontsize=7,fontstyle='italic',
                 arrowprops=dict(arrowstyle="-",color=greys_pal[8],connectionstyle="angle,angleA=0,angleB=90,rad=5",alpha=0.65))
    plt.annotate('Sinhalese',xy=(0,max(_comp.index)+0.35),xytext=(0+_shift_x,max(_comp.index)+_shift_y),fontsize=7,fontstyle='italic',
                 arrowprops=dict(arrowstyle="-",color=greys_pal[8],connectionstyle="angle,angleA=0,angleB=90,rad=5",alpha=0.65))

    #plt.annotate(r'$\it{Majority-minority}$'+'\n'+r'$\it{districts\ in\ }$'+r'$\bf{bold}$',xy=(33,20),ha='left',fontsize=7)
    plt.annotate('Majority-minority\ndistricts in '+r'$\bf{bold}$',xy=(33,20),ha='left',fontsize=7,fontstyle='italic')

    plt.plot([0,0],_ylim,zorder=1,color=greys_pal[2],alpha=0.75,lw=0.5)
    plt.ylim(_ylim)

    title_legend_labels(plt.gca(),pais='',lab_x=r'$\Delta$ (Socioeconomic resilience)',lab_y='',lim_x=[-55,55],leg_fs=9,do_leg=False)
    sns.despine(left=True)
    plt.gca().yaxis.set_ticks([])

    plt.grid(False)

    plt.gcf().savefig('../output_plots/SL/conflict/pop_vs_res_sin_vs_ns_3.pdf',format='pdf',bbox_inches='tight')
    plt.close('all')   
    

    #######################################
    print(_comp)
    _comp = _comp.dropna(how='any').loc[_comp.district!='Total']
    _comp.to_csv('~/Desktop/tmp/comp.csv')

    outlier_query = "(district!='Mannar')&(district!='Batticaloa')&(district!='Jaffna')"
    for _e in ['Sinhalese','non-Sinhalese']:
        plt.scatter(_comp.loc[_comp.eval(outlier_query),_e]/1E3,
                    _comp.loc[_comp.eval(outlier_query),'Resilience of '+_e],color=(color_ns if 'non-' in _e else color_sin),alpha=0.4)
        #
        linear_fit = np.polyfit(_comp.loc[_comp.eval(outlier_query),_e]*1E-3,
                                _comp.loc[_comp.eval(outlier_query),'Resilience of '+_e],1)
        r_x, r_y = zip(*((i, i*linear_fit[0] + linear_fit[1]) for i in _comp.loc[_comp.eval(outlier_query),_e]*1E-3))
        plt.plot(r_x, r_y,color=(color_ns if 'non-' in _e else color_sin),alpha=0.6,lw=1.25,ls='-')

        __y = _comp.sort_values('Resilience of '+_e,ascending=True).iloc[-2:]['Resilience of '+_e].mean()
        plt.annotate(_e,xy =(235,__y),
                     fontsize=7,fontstyle='italic',color=greys_pal[8],
                     ha=('left' if 'non-' in _e else 'right'))
                     

    title_legend_labels(plt.gca(),pais='',lab_x='Average income by district\n[,000 LKR per capita & year]',
                        lab_y='Socioeconomic resilience [%]',lim_x=[40,275],lim_y=[0,110],leg_fs=9,do_leg=False)
    sns.despine()

    plt.grid(False)

    plt.gcf().savefig('../output_plots/SL/conflict/income_vs_resil.pdf',format='pdf',bbox_inches='tight')
    plt.close('all')  



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

no_pds_q1.loc['Total',['% total RA']] = 100*no_pds_q1.loc['Total','Asset risk']/no_pds_q1['TOTAL Asset risk'].drop('Total').sum()
no_pds_q1.loc['Total',['% total RW']] = 100*no_pds_q1.loc['Total','Well-being risk']/no_pds_q1['TOTAL Well-being risk'].drop('Total').sum()


no_pds_q1 = no_pds_q1.fillna(0)

no_pds_q1[['Asset risk','% total RA','SE capacity',
           'Well-being risk','% total RW']].sort_values(['Well-being risk'],ascending=False).round(1).to_latex('latex/'+myCountry+'/risk_q1_by_economy.tex')

no_pds_q1['Asset risk']*=to_usd*(1000 if myCountry == 'SL' else 1)
no_pds_q1['Well-being risk']*=to_usd*(1000 if myCountry == 'SL' else 1)
no_pds_q1[['Asset risk','% total RA','SE capacity',
           'Well-being risk','% total RW']].sort_values(['Well-being risk'],ascending=False).round(0).astype('int').to_latex('latex/'+myCountry+'/risk_q1_by_economy_usd.tex')
print('Wrote latex! Q1 sums: ',no_pds_q1.sum())

no_pds_q1['pop_q1']  = iah.loc[iah.quintile==1,'pcwgt_no'].sum(level=event_level).mean(level=event_level[0])/1.E3
#iah_out_q1['grdp_q1'] = iah.loc[iah.quintile==1,['pcwgt_no','c_initial']].prod(axis=1).sum(level=event_level).mean(level=event_level[0])

_ = (no_pds_q1.drop('Total',axis=0)[['pop_q1','Asset risk','Well-being risk']]).dropna(how='any').copy()

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
#if myCountry == 'PH': myHaz = [['V - Bicol','II - Cagayan Valley','NCR','IVA - CALABARZON','ARMM','CAR'],['HU','EQ'],[10,25,50,100,250,500]]
#if myCountry == 'PH': myHaz = [['ompong'],['HU','PF','SS'],[10,50,100,200,500]]
#if myCountry == 'PH': myHaz = [['I - Ilocos','II - Cagayan Valley','CAR'],['HU','EQ'],[25,100]]
if myCountry == 'PH': myHaz = [['VIII - Eastern Visayas'],['HU'],[100]]
if myCountry == 'SL': myHaz = [['Rathnapura','Colombo'],get_all_hazards(myCountry,myiah),[25,50,100]]
#if myCountry == 'SL': myHaz = [['Rathnapura','Colombo'],get_all_hazards(myCountry,myiah),get_all_rps(myCountry,myiah)]
if myCountry == 'MW': myHaz = [['Lilongwe','Chitipa'],get_all_hazards(myCountry,myiah),get_all_rps(myCountry,myiah)]
if myCountry == 'RO': myHaz = [['Bucharest-Ilfov'],get_all_hazards(myCountry,myiah),get_all_rps(myCountry,myiah)]
if myCountry == 'BO': myHaz = [['La Paz','Beni'],get_all_hazards(myCountry,myiah),[50]]


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

                wb_str = 'Total wellbeing losses = \$'+str(round(_['dw_cum_'+base_str].max()*to_usd*1E-6,1))+' million' 
                #wb_natl_str = '(+\$'+str(round(ext_costs_base_sum*to_usd*1E-6,1))+')'
                wb_natl_str = 'National welfare losses\n  $'+str(round(ext_costs_base_sum*to_usd*1E-6,1))+' million'

                plt.annotate(wb_str,xy=(0.02,_y2),xycoords='axes fraction',color=q_colors[3],ha='left',va='top',fontsize=10,annotation_clip=False)
                #plt.annotate(wb_natl_str,xy=(0.02,0.77),xycoords='axes fraction',color=q_colors[3],ha='left',va='top',fontsize=10)            

                for _q in [1,2,3,4,5]:
                    _q_x = _.loc[_['quintile']==_q,'PMT'].max()
                    _q_y = max(_.loc[_['quintile']<=_q,['pcwgt_'+base_str,'dk0']].prod(axis=1).sum()*to_usd*1E-6,
                               _.loc[_['quintile']<=_q,['pcwgt_'+base_str,'dw_'+base_str]].prod(axis=1).sum()*to_usd*1E-6)
                    if _q == 1: _q_yprime = _q_y/25

                    plt.plot([_q_x,_q_x],[0,_q_y],color=greys_pal[4],ls=':',linewidth=1.5,zorder=91)
                    plt.annotate(quint_labels[_q-1],xy=(_q_x,_q_y+7*_q_yprime),color=greys_pal[6],ha='right',va='bottom',style='italic',fontsize=8,zorder=91,annotation_clip=False)


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
                plt.ylabel('Cumulative losses [mil. US$]',labelpad=8,fontsize=10)
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
                plt.ylabel('Cost & benefit [mil. US$]',labelpad=8,fontsize=12)
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

                plt.xlabel('Upper PMT threshold for post-disaster support',labelpad=10,fontsize=10)
                plt.ylabel('Marginal impact at threshold [US$ per next enrollee]',labelpad=10,fontsize=10)

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

if myCountry=='BO': 
    myiah['pov_line'] = 714.9/to_usd
    myiah['sub_line'] = 350.0/to_usd
    print('\nPoverty line in Bolivia:\n',to_usd,714.9*to_usd,350.*to_usd,'\n\n')

if True:

    _myiah = myiah.reset_index().set_index(event_level+['hhid','affected_cat','helped_cat'])[['pcwgt_no','ispoor',
                                                                                              'c_initial','c_post_reco',
                                                                                              'i_pre_reco','c_pre_reco','dc_pre_reco',
                                                                                              'pov_line','sub_line']].copy()
    _myiah = _myiah.loc[_myiah['pcwgt_no']!=0]
    
    run_poverty_duration_plot(myCountry,myHaz[1][0])
    #                                   ^ first hazard in the country we're running
    run_poverty_tables_and_maps(myCountry,_myiah,event_level,myHaz[1][0],drop_spots=places_to_drop,_mapres=_mapres)
    map_recovery_time(myCountry,myHaz[1][0],RP=[200],drop_spots=places_to_drop,_mapres=_mapres)

##################################################################
# This code generates the histograms showing income before & after disaster (in USD)
# ^ this is at household level (iah != iah_avg here)
if True:         
    with Pool(processes=1,maxtasksperchild=1) as pool:
        print('LAUNCHING',len(list(product(myHaz[0],myHaz[1],myHaz[2]))),'THREADS')
        pool.starmap(plot_income_and_consumption_distributions,list(product([myCountry],[myiah.copy()],myHaz[0],myHaz[1],myHaz[2],[True],['USD'])))

##################################################################
# This code generates the histograms showing income before & after disaster (in local_curr)
# ^ this is at household level (iah != iah_avg here)
if True:
    with Pool(processes=1,maxtasksperchild=1) as pool:
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
