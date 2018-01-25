#This script processes data outputs for the resilience indicator multihazard model for the Philippines. Developed by Brian Walsh.
from IPython import get_ipython
get_ipython().magic('reset -f')
get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')

#Import packages for data analysis
from lib_compute_resilience_and_risk import *
from replace_with_warning import *
from lib_country_dir import *
from lib_gather_data import *
from maps_lib import *

from scipy.stats import norm
import matplotlib.mlab as mlab

import matplotlib.patches as patches
from pandas import isnull
import pandas as pd
import numpy as np
import os, time
import sys

#Aesthetics
import seaborn as sns
import brewer2mpl as brew
from matplotlib import colors
sns.set_style('darkgrid')
brew_pal = brew.get_map('Set1', 'qualitative', 8).mpl_colors
sns_pal = sns.color_palette('Set1', n_colors=17, desat=None)
greys_pal = sns.color_palette('Greys', n_colors=9)
reds_pal = sns.color_palette('Reds', n_colors=9)
q_labels = ['Q1 (Poorest)','Q2','Q3','Q4','Q5 (Wealthiest)']
q_colors = [sns_pal[0],sns_pal[1],sns_pal[2],sns_pal[3],sns_pal[5]]

reg_pal = sns.color_palette(['#e6194b','#3cb44b','#ffe119','#0082c8','#f58231','#911eb4','#46f0f0','#f032e6','#d2f53c','#008080','#e6beff','#fabebe','#800000','#808000','#000080','#808080','#000000'],n_colors=17,desat=None)

params = {'savefig.bbox': 'tight', #or 'standard'
          #'savefig.pad_inches': 0.1 
          'xtick.labelsize': 8,
          'ytick.labelsize': 8,
          'legend.fontsize': 9,
          'legend.facecolor': 'white',
          #'legend.linewidth': 2, 
          'legend.fancybox': True,
          'savefig.facecolor': 'white',   # figure facecolor when saving
          #'savefig.edgecolor': 'white'    # figure edgecolor when saving
          }
plt.rcParams.update(params)

font = {'family' : 'sans serif',
    'size'   : 10}
plt.rc('font', **font)

import warnings
warnings.filterwarnings('always',category=UserWarning)

myCountry = 'PH'
if len(sys.argv) < 2:
    print('Could list country. Using PH.')
else: myCountry = sys.argv[1]

model  = os.getcwd() #get current directory
output = model+'/../output_country/'+myCountry+'/'

# Load output files
pol_str = ''#'_v95'#could be {'_v95'}
base_str = 'no'
pds_str = 'unif_poor'

macro = pd.read_csv(output+'macro_tax_'+pds_str+'_.csv')

try:
    iah_pds = pd.read_csv('/Users/brian/Desktop/BANK/hh_resilience_model/check_plots/test_hh.csv').reset_index()
except:
    iah_pds = pd.read_csv(output+'iah_tax_'+pds_str+'_'+pol_str+'.csv').reset_index()
    #iah_pds = iah_pds.loc[(iah_pds.hhid==153829114)&(iah_pds.hazard=='EQ')&(iah_pds.rp==100)&(iah_pds.affected_cat=='a')&(iah_pds.helped_cat=='helped')]
    iah_pds = iah_pds.loc[(iah_pds.help_received>0)&(iah_pds.hazard=='EQ')&(iah_pds.rp==100)&(iah_pds.affected_cat=='a')&(iah_pds.helped_cat=='helped')].head(1)
    iah_pds.to_csv('/Users/brian/Desktop/BANK/hh_resilience_model/check_plots/test_hh.csv')
print(iah_pds.columns)

# k recovery
const_dk_reco = np.log(1/0.05)/float(iah_pds['hh_reco_rate'])
const_pds     = (np.log(1/0.05)/3.)*6. # PDS consumed in first half year of recovery 
const_prod_k  = float(macro.avg_prod_k.mean())

print(iah_pds.head())

c   = float(iah_pds['c'])
dc0 = float(iah_pds['dc0'])
pds = float(iah_pds['help_received'])

k     = float(iah_pds['k'])
dk0   = float(iah_pds['dk0'])
dkprv = float(iah_pds['dk_private'])
dkpub = float(iah_pds['dk_public'])

c_t       = [] 
dc_k_t    = []
dc_reco_t = []
dc_pds_t  = []

t_lins = np.linspace(0,10,200)
for t in t_lins:
    c_t.append(c)
    dc_k_t.append(dk0*const_prod_k*np.e**(-(t)*const_dk_reco))
    dc_reco_t.append(dk0*const_dk_reco*np.e**(-(t)*const_dk_reco))
    dc_pds_t.append(pds*const_pds*np.e**(-(t)*const_pds))
    
#step_dt*((1.-(temp['dc0']/temp['c'])*math.e**(-i_dt*const_reco_rate)+temp['help_received']*const_pds_rate*math.e**(-i_dt*const_pds_rate))**(1-const_ie)-1)*math.e**(-i_dt*const_rho)
# Indicate k(t): private and public 

# Lost income from capital
plt.fill_between(t_lins,c_t,[i-j for i,j in zip(c_t,dc_k_t)],facecolor=reds_pal[3],alpha=0.45)
plt.scatter(0,c_t[0]-dc_k_t[0],color=reds_pal[3],zorder=100)
plt.annotate('Income\nlosses',[-0.50,(c_t[0]+(c_t[0]-dc_k_t[0]))/2.],fontsize=8,ha='center',va='center')

# Reconstruction costs
plt.fill_between(t_lins,[i-j for i,j in zip(c_t,dc_k_t)],[i-j-k for i,j,k in zip(c_t,dc_k_t,dc_reco_t)],facecolor=reds_pal[4],alpha=0.45)
plt.scatter(0,c_t[0]-dc_k_t[0]-dc_reco_t[0],color=reds_pal[4],zorder=100)
plt.annotate('Reconstruction\ncosts',[-0.50,((c_t[0]-dc_k_t[0])+(c_t[0]-dc_k_t[0]-dc_reco_t[0]))/2.],fontsize=8,ha='center',va='center')

# PDS
plt.fill_between(t_lins,c_t,[i+j for i,j in zip(c_t,dc_pds_t)],facecolor=sns_pal[2],alpha=0.45)
plt.annotate('PDS\nspend down',[-0.50,(c_t[0]+(c_t[0]+dc_pds_t[0]))/2.],fontsize=8,ha='center',va='center')

plt.plot(t_lins,[i-j-k+l for i,j,k,l in zip(c_t,dc_k_t,dc_reco_t,dc_pds_t)],ls=':',color=reds_pal[8])

# Draw c
plt.plot([-1,5],[c,c],color=greys_pal[8])

plt.xlim(-1,5)
#plt.ylim((c-dc0)*0.98,c*1.02)

plt.xlabel(r'Time $t$ after disaster ($\tau_h \equiv 3$) [years]')
plt.ylabel(r'Household consumption ($c_h$)')
plt.xticks([-1,0,1,2,3,4,5],['-1',r'$t_0$','1','2','3','4','5'])
plt.yticks([c_t[0]],[r'$c_0$'])

plt.draw()
fig=plt.gcf()
fig.savefig('/Users/brian/Desktop/Dropbox/Bank/unbreakable_writeup/Figures/dc.pdf',format='pdf')

plt.clf()
plt.close('all')

# Draw k
plt.plot([-1,5],[k,k],color=greys_pal[8])

# points at t0
plt.scatter(0,k-dk0,color=reds_pal[5],zorder=100)
plt.scatter(0,k-dkprv,color=reds_pal[3],zorder=100)
# Annotate 
plt.annotate('Private\nasset\nlosses',[-0.65,k-dkprv/2.],fontsize=9,ha='center',va='center')
plt.annotate(r'$\Delta k^{prv}_0$',[-0.2,k-dkprv/2.],fontsize=10,ha='center',va='center')
plt.plot([-0.2,-0.2],[k-dkprv,k-1.1*dkprv/2.],color=reds_pal[3])
plt.plot([-0.2,-0.2],[k-0.9*dkprv/2.,k],color=reds_pal[3])
plt.plot([-0.22,-0.18],[k,k],color=reds_pal[3])
plt.plot([-0.22,-0.18],[k-dkprv*0.997,k-dkprv*0.997],color=reds_pal[3],zorder=100)

plt.annotate('Public\nasset\nlosses',[-0.65,k-dk0+dkpub/2.],fontsize=9,ha='center',va='center')
plt.annotate(r'$\Delta k^{pub}_0$',[-0.2,k-dk0+dkpub/2.],fontsize=10,ha='center',va='center')
plt.plot([-0.2,-0.2],  [k-dkprv-dkpub,(k-dkprv)-1.5*dkpub/2.],color=reds_pal[5])
plt.plot([-0.2,-0.2],  [(k-dkprv)-0.5*dkpub/2.,k-dkprv],color=reds_pal[5])
plt.plot([-0.22,-0.18],[k-dkprv*1.003,k-dkprv*1.003],color=reds_pal[5])
plt.plot([-0.22,-0.18],[k-dkprv-dkpub,k-dkprv-dkpub],color=reds_pal[5])

plt.annotate('Disaster\n'+r'(t = t$_0$)',[0,k*1.005],fontsize=9,ha='center',weight='bold')
plt.plot([0,0],[k-dk0,k],color=sns_pal[0])

# k recovery
k_t     = []
dk0_t   = []
dkprv_t = []
dkpub_t = []

for t in t_lins:
    k_t.append(k)
    dk0_t.append(k-(dk0*np.e**(-t*const_dk_reco)))
    dkprv_t.append(k-(dkprv*np.e**(-t*const_dk_reco)))
    dkpub_t.append(k-(dkpub*np.e**(-t*const_dk_reco)))

# Indicate k(t): private and public 
plt.fill_between(t_lins,k_t,dkprv_t,facecolor=reds_pal[3],alpha=0.45)
plt.fill_between(t_lins,dkprv_t,[i-(k-j) for i,j,k in zip(dkprv_t,dkpub_t,k_t)],facecolor=reds_pal[5],alpha=0.45)

plt.plot([3,3],[k-0.05*dk0,k],color=reds_pal[8])
plt.plot([2.98,3.02],[k-0.05*dk0,k-0.05*dk0],color=reds_pal[8],zorder=100)
plt.plot([2.98,3.02],[k,k],color=reds_pal[8],zorder=100)

plt.gca().add_patch(patches.Rectangle((3.45,k-0.12*dk0),1.60,6500,facecolor='white',zorder=98,clip_on=False))
plt.gca().annotate(r'$\Delta k_h^{eff}|_{t=\tau_h}$ = 0.05$\times\Delta k_0^{eff}$',
                   xy=(3,k-0.025*dk0), xycoords='data',
                   xytext=(3.5,k-0.075*dk0), textcoords='data', fontsize=10,
                   arrowprops=dict(arrowstyle="->",connectionstyle="arc3,rad=-0.05",lw=1.5),
                   ha='left',va='center',zorder=99)

plt.plot(t_lins,dk0_t,color=reds_pal[8],ls='--',lw=0.75)
plt.gca().add_patch(patches.Rectangle((1.40,dk0_t[10]*1.005),1.70,7000,facecolor='white',zorder=98))
plt.gca().annotate(r'$\Delta k_h^{eff}(t) = \Delta k_0^{eff}e^{-R_{\tau}\cdot t}$',
                   xy=(t_lins[20],dk0_t[20]), xycoords='data',
                   xytext=(1.45,dk0_t[10]*1.01), textcoords='data', fontsize=12,
                   arrowprops=dict(arrowstyle="->",connectionstyle="arc3,rad=0.2",lw=1.5),
                   ha='left',va='center',zorder=99)

plt.xlim(-1,5)
plt.ylim((k-dk0)*0.98,k*1.02)

plt.xlabel(r'Time $t$ after disaster ($\tau_h \equiv 3$) [years]')
#plt.xlabel(r'Time $t$ after disaster (years)')
plt.ylabel(r'Effective household capital ($k_h^{eff}$)')
plt.xticks([-1,0,1,2,3,4,5],['-1',r'$t_0$','1','2','3','4','5'])
plt.yticks([k_t[0]],[r'$k_h^{eff}$'])

plt.draw()
fig=plt.gcf()
fig.savefig('/Users/brian/Desktop/Dropbox/Bank/unbreakable_writeup/Figures/dk.pdf',format='pdf')

summary_df = pd.read_csv('/Users/brian/Desktop/BANK/debug/my_summary_no.csv').reset_index()
summary_df = summary_df.loc[summary_df.rp!=2000].sort_values(by=['hazard','region','rp'])

all_regions = np.unique(summary_df['region'].dropna())

for iHaz in ['SS','PF','HU','EQ']:
    _ = summary_df.loc[(summary_df.hazard==iHaz)]

    regions = _.groupby('region')
    
    fig, ax = plt.subplots()
    ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling
    col_ix = 0
    for name, iReg in regions:

        while name != all_regions[col_ix]:
            col_ix += 1
        
        ax.semilogx(iReg.rp, (100*iReg.res_tot).clip(upper=100), marker='.', linestyle='', ms=9, label=name,color=reg_pal[col_ix])
        col_ix+=1

    plt.xlabel('Return period [years]')
    plt.ylabel('Socio-economic capacity [%]')
    plt.ylim(0,130)

    leg = ax.legend(loc='best',labelspacing=0.75,ncol=4,fontsize=6,borderpad=0.75,fancybox=True,frameon=True,framealpha=1.0,title='Region')

    col_ix = 0
    for name, iReg in regions:

        while name != all_regions[col_ix]:
            col_ix += 1

        ax.plot(iReg.rp, (100*iReg.res_tot).clip(upper=100),color=reg_pal[col_ix])
        col_ix+=1

    fig.savefig('/Users/brian/Desktop/BANK/hh_resilience_model/check_plots/reg_resilience_'+iHaz+'.pdf',format='pdf')
    plt.clf()
