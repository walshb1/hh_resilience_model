#This script processes data outputs for the resilience indicator multihazard model for the Philippines. Developed by Brian Walsh.

#import IPython
from IPython import get_ipython, display
get_ipython().magic('reset -f')
get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')

import sys
import os, glob
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as pystat
import matplotlib.pyplot as plt

from libraries.lib_average_over_rp import *
from libraries.lib_common_plotting_functions import *
from libraries.lib_country_dir import get_all_rps, get_poverty_line, get_currency
from libraries.lib_gather_data import match_percentiles, perc_with_spline, reshape_data

pairs_pal = sns.color_palette('Paired', n_colors=12)
greys_pal = sns.color_palette('Greys', n_colors=9)

myCountry = 'PH'
try: myCountry = sys.argv[1]
except: pass

# Get poverty line
_pov = get_poverty_line(myCountry)

# Decide whether to do this at decile or quintile level
agglev = 'decile'

try: _q = pd.read_csv('../output_country/'+myCountry+'/sp_comparison_by_'+agglev+'.csv').set_index(agglev)
except:

    # Load file
    _f = pd.read_csv('../output_country/'+myCountry+'/poverty_duration_no.csv')
    _f_up = pd.read_csv('../output_country/'+myCountry+'/poverty_duration_unif_poor.csv')

    if 'quintile' not in _f.columns or 'decile' not in _f.columns:

        # Assign deciles
        _deciles=np.arange(0.10, 1.01, 0.10)
        _f = _f.groupby(['hazard','rp'],sort=True).apply(lambda x:match_percentiles(x,perc_with_spline(reshape_data(x.c),reshape_data(x.pcwgt),_deciles),'decile','c'))
        _f_up = _f_up.groupby(['hazard','rp'],sort=True).apply(lambda x:match_percentiles(x,perc_with_spline(reshape_data(x.c),reshape_data(x.pcwgt),_deciles),'decile','c'))

        # Assign quintiles
        _quintiles=np.arange(0.20, 1.01, 0.20)
        _f = _f.groupby(['hazard','rp'],sort=True).apply(lambda x:match_percentiles(x,perc_with_spline(reshape_data(x.c),reshape_data(x.pcwgt),_quintiles),'quintile','c'))
        _f_up = _f_up.groupby(['hazard','rp'],sort=True).apply(lambda x:match_percentiles(x,perc_with_spline(reshape_data(x.c),reshape_data(x.pcwgt),_quintiles),'quintile','c'))

        _f.to_csv('../output_country/'+myCountry+'/poverty_duration_no.csv')
        _f_up.to_csv('../output_country/'+myCountry+'/poverty_duration_unif_poor.csv')

    # Put quintile, hhid into index
    _f = _f.reset_index().set_index(['decile','quintile']).sort_index()
    _f_up = _f_up.reset_index().set_index(['decile','quintile']).sort_index()

    # Quintile-level (or decile-level) info
    _q = pd.DataFrame(index=_f.sum(level=agglev).index)

    tp_delta = {}
    tp_delta['rps'] = get_all_rps(myCountry,_f)

    for _rp in get_all_rps(myCountry,_f):
        _srp = str(_rp)
        print('\nRP = ',_rp)

        _crit = '(dk0!=0)&(region=="V - Bicol")&(hazard=="HU")&(rp==@_rp)'

        _f_wgt_dnm = _f.loc[_f.eval(_crit),'pcwgt'].sum(level=agglev)
        _f_up_wgt_dnm = _f_up.loc[_f_up.eval(_crit),'pcwgt'].sum(level=agglev)

        _q['t_pov_cons_no'+_srp] = _f.loc[_f.eval(_crit),['pcwgt','t_pov_cons']].prod(axis=1).sum(level=agglev)/_f_wgt_dnm
        _q['t_pov_cons_unif_poor'+_srp] = _f_up.loc[_f_up.eval(_crit),['pcwgt','t_pov_cons']].prod(axis=1).sum(level=agglev)/_f_up_wgt_dnm

        _q['t_pov_cons_pct_change'+_srp] = 1.-_q['t_pov_cons_unif_poor'+_srp]/_q['t_pov_cons_no'+_srp]

        _q['poverty_gap'] = 100.*(_f.loc[_f.eval(_crit)].eval('pcwgt*(@_pov-c)/@_pov').clip(lower=0)).sum(level=agglev)/_f_wgt_dnm
        _q['poverty_gap_no'+_srp] = 100.*(_f.loc[_f.eval(_crit)].eval('pcwgt*(@_pov-(c-dc_net_t0))/@_pov').clip(lower=0)).sum(level=agglev)/_f_wgt_dnm
        _q['poverty_gap_unif_poor'+_srp] = 100.*(_f_up.loc[_f_up.eval(_crit)].eval('pcwgt*(@_pov-(c-dc_net_t0))/@_pov')).clip(lower=0).sum(level=agglev)/_f_up_wgt_dnm

        _q['poverty_gap_pct_change'+_srp] = 1.-_q['poverty_gap_unif_poor'+_srp]/_q['poverty_gap_no'+_srp]

    _q.to_csv('../output_country/'+myCountry+'/sp_comparison_by_'+agglev+'.csv')

#choose rp to plot:
_rp = '50'

ax = plt.plot(_q.index,_q['t_pov_cons_no'+_rp])
plt.plot(_q.index,_q['t_pov_cons_unif_poor'+_rp])
plt.savefig('../output_plots/'+myCountry+'/poverty_duration_by_'+agglev+'.pdf',format='pdf')
plt.clf()

ax = plt.scatter(_q.loc[_q.index<=5].index,_q.loc[_q.index<=5,'poverty_gap_no'+_rp],label='no',zorder=31,s=30,color=pairs_pal[5],marker='x',linewidth=2)
plt.scatter(_q.loc[_q.index<=5].index,_q.loc[_q.index<=5,'poverty_gap_unif_poor'+_rp],label='unif_poor',zorder=29,s=35,color=pairs_pal[4])
plt.scatter(_q.loc[_q.index<=5].index,_q.loc[_q.index<=5,'poverty_gap'],label='Pre-disaster',zorder=30,s=35,color=greys_pal[5],marker='s')

for _i in _q.loc[_q.index<=5].index:
    plt.plot([_i,_i],[_q.loc[_i,'poverty_gap'],_q.loc[_i,'poverty_gap_no'+_rp]],color=greys_pal[1],linewidth=1.25,zorder=10)

    if _i == 1:
        plt.annotate('Pre-disaster',xy=(0.92*_i,_q.loc[_i,'poverty_gap']),ha='right',va='center',fontsize=6.5,weight='bold',linespacing=0.9)
        plt.annotate('No PDS',xy=(0.92*_i,_q.loc[_i,'poverty_gap_no'+_rp]),ha='right',va='center',fontsize=6.5,weight='bold',linespacing=0.9)
        plt.annotate('Uniform PDS',xy=(0.92*_i,_q.loc[_i,'poverty_gap_unif_poor'+_rp]),ha='right',va='center',fontsize=6.5,weight='bold',linespacing=0.9)

    if _i <= 3:

        try: _valA = '+'+str(int(round(100.*(_q.loc[_i,'poverty_gap_no'+_rp]/_q.loc[_i,'poverty_gap']-1.),0)))+'%'
        except: _valA = ''
        
        try: _valB = '+'+str(int(round(100.*(_q.loc[_i,'poverty_gap_unif_poor'+_rp]/_q.loc[_i,'poverty_gap']-1.),0)))+'%'
        except: _valB = ''

        plt.plot([_i,_i+.21],[_q.loc[_i,'poverty_gap'],_q.loc[_i,'poverty_gap']],color=greys_pal[3],linewidth=1.25,zorder=10)

        plt.annotate('',xy=(_i+.10,_q.loc[_i,'poverty_gap_no'+_rp]),xytext=(_i+.10,_q.loc[_i,'poverty_gap']),
                     arrowprops=dict(arrowstyle='simple,tail_width=0.15',connectionstyle="arc3,rad=0.0",fc=greys_pal[3],ec='none',shrinkA=0,shrinkB=0))
        plt.annotate('',xy=(_i+.18,_q.loc[_i,'poverty_gap_unif_poor'+_rp]),xytext=(_i+.18,_q.loc[_i,'poverty_gap']),
                     arrowprops=dict(arrowstyle='simple,tail_width=0.15',connectionstyle="arc3,rad=0.0",fc=greys_pal[3],ec='none',shrinkA=0,shrinkB=0))
        
        plt.annotate(_valA,xy=(_i+.14,_q.loc[_i,'poverty_gap_no'+_rp]),va='center',ha='left',weight='bold',fontsize=6.5,color=greys_pal[7])
        plt.annotate(_valB,xy=(_i+.22,_q.loc[_i,'poverty_gap_unif_poor'+_rp]),va='center',ha='left',weight='bold',fontsize=6.5,color=greys_pal[7])

#plt.xlabel('Decile',fontsize=11,weight='bold',labelpad=8)
plt.ylabel('Consumption poverty gap',fontsize=11,weight='bold',labelpad=8)

plt.xticks(np.linspace(1,5,5))
plt.gca().set_xticklabels(['Poorest\ndecile','Second\ndecile','Third\ndecile','Fourth\ndecile','Fifth\ndecile'],weight='bold')

plt.xlim(0.2,5.2)
plt.ylim(0)
plt.yticks(np.linspace(0,50,6))
plt.gca().set_yticklabels(['0%','10%','20%','30%','40%','50%'])

plt.gca().grid(False,axis='x')
sns.despine()

plt.savefig('../output_plots/'+myCountry+'/poverty_gap_by_'+agglev+'.pdf',format='pdf')
plt.close('all')


plt.close('all')
#################################################
#################################################
# Plot cost of SP policy by return period

rp = 10         # Return period,
p = 1/rp        # Probability of getting "red" at the roulette
hh = pystat.binom(rp, p)

event_cost = 1.E3
total_cost = 0
for k in range(0,rp+1):  # DO NOT FORGET THAT THE LAST INDEX IS NOT USED
    print(k,hh.pmf(k), event_cost*k*hh.pmf(k))
    total_cost += event_cost*k*hh.pmf(k)
print(total_cost)


path = os.getcwd()+'/../output_country/'+myCountry+'/'
pattern = 'sp_costs_*.csv'

_spdict = {'no':'No PDS',
           'unif_poor':'Uniform Q1-Q5',
           'unif_poor_only':'Uniform Q1',
           'prop_q1':'Proportional Q1',
           'prop':'Proportional Q1-Q5'}
_spcols = {'no':greys_pal[4],
           'unif_poor':pairs_pal[0],
           'unif_poor_only':pairs_pal[1],
           'prop_q1':pairs_pal[3],
           'prop':pairs_pal[2]}

ax = plt.gca()
df_max = 1000
for f in glob.glob(path+pattern):
    _spcode = f.replace(path,'').replace('sp_costs_','').replace('.csv','')
    if _spcode == 'no': continue

    _spdf = pd.read_csv(f).set_index(['region','hazard','rp']).sum(level='rp').drop(2000)*get_currency(myCountry)[2]*1E-6

    _spdf.plot(_spdf.index.values,'event_cost',ax=ax,label=_spdict[_spcode],lw=2.,color=_spcols[_spcode],legend=False)

    plt.annotate(_spdict[_spcode],xy=(df_max,_spdf.loc[df_max,'event_cost']),xycoords='data',ha='right',va='bottom',weight='bold',fontsize=7,clip_on=False)

title_legend_labels(ax,'',lab_x='Return period [years]',lab_y='Event cost (mUSD)',lim_x=None,lim_y=None,leg_fs=9,do_leg=False)

sns.despine()
ax.xaxis.grid(False)
#plt.gca().grid(False)
plt.xlim(0,df_max)
plt.ylim(0)
plt.gcf().savefig('../output_plots/'+myCountry+'/sp/cost_by_sp.pdf',format='pdf')



#################################################
#################################################
# Plot cost of SP policy by RP (inclusive of larger events)
plt.clf()
ax=plt.gca()
df_max = 1000
for f in glob.glob(path+pattern):
    _spcode = f.replace(path,'').replace('sp_costs_','').replace('.csv','')
    if _spcode == 'no': continue

    _spdf = pd.read_csv(f).set_index(['region','hazard','rp']).sum(level='rp').drop(2000)*get_currency(myCountry)[2]*1E-6
    _spdf['annual_event_cost'] = _spdf['event_cost']/_spdf.index.values
    _spdf.loc[_spdf.index.values!=1].plot(_spdf.loc[_spdf.index.values!=1].index.values,'annual_event_cost',ax=ax,
                                          label=_spdict[_spcode],lw=2.,color=_spcols[_spcode],legend=False)

    _va = 'top'
    if _spcode == 'unif_poor_only': _va = 'bottom'

    plt.annotate(_spdict[_spcode],xy=(df_max,_spdf.loc[df_max,'annual_event_cost']),xycoords='data',ha='right',va=_va,weight='bold',fontsize=7,clip_on=False)

title_legend_labels(ax,'',lab_x='Return period (inclusive of larger events) [years]',lab_y='Annual cost of SP (mUSD)',lim_x=None,lim_y=None,leg_fs=9,do_leg=False)

sns.despine()
ax.xaxis.grid(False)
#plt.gca().grid(False)
plt.xlim(0,df_max)
plt.ylim(0)
plt.gcf().savefig('../output_plots/'+myCountry+'/sp/annual_cost_by_sp.pdf',format='pdf')



#################################################
#################################################
# Plot cost of SP policy by RP (inclusive of smaller events)
plt.clf()
rp_max = 200

ax=plt.gca()
for f in glob.glob(path+pattern):
    _spcode = f.replace(path,'').replace('sp_costs_','').replace('.csv','')
    if _spcode == 'no': continue

    _spdf = pd.read_csv(f).set_index(['region','hazard','rp']).sum(level='rp').drop(['avg_admin_cost','avg_natl_cost'],axis=1)*get_currency(myCountry)[2]*1E-6
    _spavgdf = pd.read_csv(f).set_index(['region','hazard','rp']).mean(level='rp').drop('event_cost',axis=1)*get_currency(myCountry)[2]*1E-6
    
    _spdf,_ = average_over_rp(_spdf)
    _spdf['rp'] = _spavgdf.index.values
    _spdf = _spdf.reset_index().set_index('rp').drop('index',axis=1)

    _spdf['cumsum'] = _spdf['event_cost'].cumsum()

    _spdf = _spdf.drop([_rp for _rp in _spdf.index.values if _rp > rp_max])
    _spdf.plot(_spdf.index.values,'cumsum',ax=ax,label=_spdict[_spcode],lw=2.,color=_spcols[_spcode],legend=False)

    plt.annotate(_spdict[_spcode],xy=(rp_max,_spdf.loc[rp_max,'cumsum']+2),xycoords='data',ha='right',va='bottom',weight='bold',fontsize=7,clip_on=False)

title_legend_labels(ax,'',lab_x='Maximum event covered by SP [return period, years]',lab_y='Annual cost (mUSD)',lim_x=None,lim_y=None,leg_fs=9,do_leg=False)

sns.despine()
plt.xticks([0,50,100,150,200])
ax.xaxis.grid(False)
plt.xlim(0,rp_max)
plt.ylim(0)
plt.gcf().savefig('../output_plots/'+myCountry+'/sp/avg_cum_cost_by_sp.pdf',format='pdf',bbox_inches='tight')
