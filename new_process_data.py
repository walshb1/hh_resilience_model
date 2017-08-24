#Import packages for data analysis
from lib_compute_resilience_and_risk import *
from replace_with_warning import *
from lib_country_dir import *
from lib_gather_data import *
from maps_lib import *

from scipy.stats import norm
import matplotlib.mlab as mlab

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
sns_pal = sns.color_palette('Set1', n_colors=8, desat=.4)
greys_pal = sns.color_palette('Greys', n_colors=9)

font = {'family' : 'sans serif',
    'size'   : 20}
plt.rc('font', **font)
mpl.rcParams['xtick.labelsize'] = 16

import warnings
warnings.filterwarnings('always',category=UserWarning)

myCountry = 'FJ'
if len(sys.argv) >= 2: myCountry = sys.argv[1]
print('Running '+myCountry)

model  = os.getcwd() #get current directory
output = model+'/../output_country/'+myCountry+'/'

economy = get_economic_unit(myCountry)
event_level = [economy, 'hazard', 'rp']
dem = get_demonym(myCountry)

# Set policy params
base_str = 'no'
pds_str = 'fiji_SPS'#'no'
pol_str = ''
drm_pov_sign = -1 # toggle subtraction or addition of dK to affected people's incomes

# Load base and PDS files
iah_base = pd.read_csv(output+'iah_tax_'+base_str+'_'+pol_str+'.csv', index_col=[economy,'hazard','rp','hhid'])
iah = pd.read_csv(output+'iah_tax_'+pds_str+'_'+pol_str+'.csv', index_col=[economy,'hazard','rp','hhid'])
df = pd.read_csv(output+'results_tax_'+pds_str+'_'+pol_str+'.csv', index_col=[economy,'hazard','rp'])
df_base = pd.read_csv(output+'results_tax_'+base_str+'_'+pol_str+'.csv', index_col=[economy,'hazard','rp'])
macro = pd.read_csv(output+'macro_tax_'+pds_str+'_'+pol_str+'.csv', index_col=[economy,'hazard','rp'])

# Manipulate iah
iah['c_initial']   = (iah[['c','hhsize']].prod(axis=1)/iah['hhsize_ae']).fillna(0)# c per AE
iah['delta_c']     = (df['avg_prod_k'].mean()+1/df['T_rebuild_K'].mean())*(iah[['dk','pcwgt']].prod(axis=1)/iah['pcwgt_ae']).fillna(0)
iah['pds_nrh']     = iah['help_fee']-iah['help_received']
iah['c_final']     = (iah['c_initial'] + drm_pov_sign*iah['delta_c'])
iah['c_final_pds'] = (iah['c_initial'] - iah['delta_c'] - iah['pds_nrh'])



# Clone index of iah with just one entry/hhid
iah_res = pd.DataFrame(index=(iah.sum(level=[economy,'hazard','rp','hhid'])).index)
# Clone index of iah at national level
iah_ntl = pd.DataFrame(index=(iah_res.sum(level=['hazard','rp'])).index)

## Translate from iah by suming over hh categories [(a,na)x(helped,not_helped)]
# These are special--pcwgt has been distributed among [(a,na)x(helped,not_helped)] categories
iah_res['pcwgt']    =    iah['pcwgt'].sum(level=[economy,'hazard','rp','hhid'])
iah_res['pcwgt_ae'] = iah['pcwgt_ae'].sum(level=[economy,'hazard','rp','hhid'])
iah_res['hhwgt']    =    iah['hhwgt'].sum(level=[economy,'hazard','rp','hhid'])

#These are the same across [(a,na)x(helped,not_helped)] categories 
iah_res['pcinc']     =     iah['pcinc'].mean(level=[economy,'hazard','rp','hhid'])
iah_res['pcinc_ae']  =  iah['pcinc_ae'].mean(level=[economy,'hazard','rp','hhid'])
iah_res['hhsize_ae'] = iah['hhsize_ae'].mean(level=[economy,'hazard','rp','hhid'])
iah_res['quintile']  =  iah['quintile'].mean(level=[economy,'hazard','rp','hhid'])
iah_res['pov_line']  =  iah['pov_line'].mean(level=[economy,'hazard','rp','hhid'])

# These need to be averaged across [(a,na)x(helped,not_helped)] categories (weighted by pcwgt)
# ^ values still reported per capita
iah_res['pcinc'] = iah[['pcinc','pcwgt']].prod(axis=1).sum(level=[economy,'hazard','rp','hhid'])/iah_res['pcwgt']
iah_res['k'] = iah[['k','pcwgt']].prod(axis=1).sum(level=[economy,'hazard','rp','hhid'])/iah_res['pcwgt']
iah_res['dk'] = iah[['dk','pcwgt']].prod(axis=1).sum(level=[economy,'hazard','rp','hhid'])/iah_res['pcwgt']
iah_res['dc'] = iah[['dc','pcwgt']].prod(axis=1).sum(level=[economy,'hazard','rp','hhid'])/iah_res['pcwgt']
iah_res['help_received'] = iah[['help_received','pcwgt']].prod(axis=1).sum(level=[economy,'hazard','rp','hhid'])/iah_res['pcwgt']
iah_res['help_fee'] = iah[['help_fee','pcwgt']].prod(axis=1).sum(level=[economy,'hazard','rp','hhid'])/iah_res['pcwgt']
iah_res['dc_npv_pre'] = iah[['dc_npv_pre','pcwgt']].prod(axis=1).sum(level=[economy,'hazard','rp','hhid'])/iah_res['pcwgt']
iah_res['dc_npv_post'] = iah[['dc_npv_post','pcwgt']].prod(axis=1).sum(level=[economy,'hazard','rp','hhid'])/iah_res['pcwgt']

# Note that we're pulling dw in from iah_base here
iah_res['dw'] = iah_base[['dw','pcwgt']].prod(axis=1).sum(level=[economy,'hazard','rp','hhid'])/iah_res['pcwgt']
iah_res['pds_dw'] = iah[['dw','pcwgt']].prod(axis=1).sum(level=[economy,'hazard','rp','hhid'])/iah_res['pcwgt']

iah_res['c_initial']   = iah[['c_initial'  ,'pcwgt_ae']].prod(axis=1).sum(level=[economy,'hazard','rp','hhid'])/iah_res['pcwgt_ae'] # c per AE
iah_res['delta_c']     = iah[['delta_c'    ,'pcwgt_ae']].prod(axis=1).sum(level=[economy,'hazard','rp','hhid'])/iah_res['pcwgt_ae'] # dc per AE
iah_res['pds_nrh']     = iah[['pds_nrh'    ,'pcwgt_ae']].prod(axis=1).sum(level=[economy,'hazard','rp','hhid'])/iah_res['pcwgt_ae'] # nrh per AE
iah_res['c_final']     = iah[['c_final'    ,'pcwgt_ae']].prod(axis=1).sum(level=[economy,'hazard','rp','hhid'])/iah_res['pcwgt_ae'] # c per AE
iah_res['c_final_pds'] = iah[['c_final_pds','pcwgt_ae']].prod(axis=1).sum(level=[economy,'hazard','rp','hhid'])/iah_res['pcwgt_ae'] # c per AE

iah = iah.reset_index()
iah_res  = iah_res.reset_index().set_index([economy,'hazard','rp','hhid'])

iah_ntl['pop'] = iah_res.pcwgt.sum(level=['hazard','rp'])
iah_ntl['pov_pc_i'] = iah_res.loc[(iah_res.pcinc_ae <= iah_res.pov_line),'pcwgt'].sum(level=['hazard','rp'])
iah_ntl['pov_hh_i'] = iah_res.loc[(iah_res.pcinc_ae <= iah_res.pov_line),'hhwgt'].sum(level=['hazard','rp'])
iah_ntl['pov_pc_f'] = iah_res.loc[(iah_res.c_final < iah_res.pov_line),'pcwgt'].sum(level=['hazard','rp'])
iah_ntl['pov_hh_f'] = iah_res.loc[(iah_res.c_final < iah_res.pov_line),'hhwgt'].sum(level=['hazard','rp'])
iah_ntl['pov_pc_D'] = iah_ntl['pov_pc_f'] - iah_ntl['pov_pc_i']
iah_ntl['pov_hh_D'] = iah_ntl['pov_hh_f'] - iah_ntl['pov_hh_i']
iah_ntl['pov_pc_pds_f'] = iah_res.loc[(iah_res.c_final_pds < iah_res.pov_line),'pcwgt'].sum(level=['hazard','rp'])
iah_ntl['pov_hh_pds_f'] = iah_res.loc[(iah_res.c_final_pds < iah_res.pov_line),'hhwgt'].sum(level=['hazard','rp'])
iah_ntl['pov_pc_pds_D'] = iah_ntl['pov_pc_pds_f'] - iah_ntl['pov_pc_i']
iah_ntl['pov_hh_pds_D'] = iah_ntl['pov_hh_pds_f'] - iah_ntl['pov_hh_i']

iah_ntl['eff_pds'] = iah_ntl['pov_pc_pds_D'] - iah_ntl['pov_pc_D']

# Print out plots for iah_res
iah_res = iah_res.reset_index()
iah_ntl = iah_ntl.reset_index()
#myHaz = [['Ba'],['TC'],[100]]
myHaz = [['Ba','Lau','Tailevu'],['TC'],[1,100,500]]

q_labels = ['Poorest quintile','Q2','Q3','Q4','Wealthiest quintile']
q_colors = [sns_pal[0],sns_pal[1],sns_pal[2],sns_pal[3],sns_pal[5]]

##################################################################
# This code generates the histograms showing income before & after disaster
# ^ this is nationally, so we'll use iah_res & iah_ntl
upper_clip = 2E4
scale_fac = 1.4913458221542086

for aDis in ['TC','flood_fluv_undef','flood_pluv']:
    for anRP in [1,100,500]:        

        ax=plt.gca()

        cf_heights, cf_bins = np.histogram((iah_res.loc[(iah_res.hazard==aDis)&(iah_res.rp==anRP),'c_final']/scale_fac).clip(upper=upper_clip), bins=50, 
                                           weights=iah_res.loc[(iah_res.hazard==aDis)&(iah_res.rp==anRP),'pcwgt']/get_scale_fac(myCountry)[0])
        ci_heights, ci_bins = np.histogram((iah_res.loc[(iah_res.hazard==aDis)&(iah_res.rp==anRP),'c_initial']/scale_fac).clip(upper=upper_clip), bins=cf_bins, 
                                           weights=iah_res.loc[(iah_res.hazard==aDis)&(iah_res.rp==anRP),'pcwgt']/get_scale_fac(myCountry)[0])
        
        ax.bar(ci_bins[:-1], ci_heights, width=(ci_bins[1]-ci_bins[0]), label='Initial', facecolor=q_colors[1],alpha=0.4)
        ax.bar(cf_bins[:-1], cf_heights, width=(ci_bins[1]-ci_bins[0]), label='Post-disaster', facecolor=q_colors[0],alpha=0.4)
        
        plt.plot([iah_res.pov_line.mean()/scale_fac,iah_res.pov_line.mean()/scale_fac],[0,1.25*cf_heights[:-2].max()],'k-',lw=1.5,color='black',zorder=100,alpha=0.85)

        plt.xlim(0,upper_clip)
        plt.xlabel(r'Income ['+get_currency(myCountry)+' yr$^{-1}$]')
        plt.ylabel('Population'+get_scale_fac(myCountry)[1])
        leg = ax.legend(loc='best',labelspacing=0.75,ncol=1,fontsize=9,borderpad=0.75,fancybox=True,frameon=True,framealpha=0.9)
        leg.get_frame().set_color('white')
        leg.get_frame().set_edgecolor(greys_pal[7])
        leg.get_frame().set_linewidth(0.2)

        ax.annotate('Poverty line',xy=(1.1*iah_res.pov_line.mean()/scale_fac,1.20*cf_heights[:-2].max()),xycoords='data',ha='left',va='top',fontsize=8,annotation_clip=False,weight='bold')

        new_pov = int(iah_ntl.loc[(iah_ntl.hazard==aDis)&(iah_ntl.rp==anRP),'pov_pc_D'])
        new_pov_pct = round(100.*float(new_pov)/float(iah_ntl.loc[(iah_ntl.hazard==aDis)&(iah_ntl.rp==anRP),'pop']),1)

        ax.annotate(r'$\Delta$N$_p$ = +'+str(new_pov)[:-3]+','+str(new_pov)[-3:]+' ('+str(new_pov_pct)+'% of population)',xy=(1.1*iah_res.pov_line.mean()/scale_fac,1.15*cf_heights[:-2].max()),xycoords='data',ha='left',va='top',fontsize=8,annotation_clip=False)

        print('poverty_k_'+aDis+'_'+str(anRP)+'.pdf')
        fig = ax.get_figure()
        fig.savefig('../output_plots/'+myCountry+'/poverty_k_'+aDis+'_'+str(anRP)+'.pdf',format='pdf')
        #fig.savefig('../output_plots/'+myCountry+'/png/poverty_k_'+aDis+'_'+str(anRP)+'.png',format='png')
        plt.cla()    

##################################################################
# This code generates the histograms including [k,dk,dc,dw,&pds]
# ^ this is by province, so it will use iah_res
for aProv in myHaz[0]:
    for aDis in myHaz[1]:
        for anRP in myHaz[2]:

            ax = plt.subplot(111)
            for myQ in range(1,6): #nQuintiles
    
                print(aProv,aDis,anRP,'shape:',iah_res.loc[(iah_res.Division==aProv)&(iah_res.hazard==aDis)&(iah_res.rp==anRP)&(iah_res.quintile==myQ),'pcwgt'].shape[0])
                
                k = (0.01*iah_res.loc[(iah_res.Division==aProv)&(iah_res.hazard==aDis)&(iah_res.rp==anRP)&(iah_res.quintile==myQ),['k','pcwgt']].prod(axis=1).sum()/
                     iah_res.loc[(iah_res.Division==aProv)&(iah_res.hazard==aDis)&(iah_res.rp==anRP)&(iah_res.quintile==myQ),'pcwgt'].sum())
                
                dk = (iah_res.loc[(iah_res.Division==aProv)&(iah_res.hazard==aDis)&(iah_res.rp==anRP)&(iah_res.quintile==myQ),['dk','pcwgt']].prod(axis=1).sum()/
                      iah_res.loc[(iah_res.Division==aProv)&(iah_res.hazard==aDis)&(iah_res.rp==anRP)&(iah_res.quintile==myQ),'pcwgt'].sum())
                
                dc = (iah_res.loc[(iah_res.Division==aProv)&(iah_res.hazard==aDis)&(iah_res.rp==anRP)&(iah_res.quintile==myQ),['dc_npv_pre','pcwgt']].prod(axis=1).sum()/
                      iah_res.loc[(iah_res.Division==aProv)&(iah_res.hazard==aDis)&(iah_res.rp==anRP)&(iah_res.quintile==myQ),'pcwgt'].sum())
                
                dw = (iah_res.loc[(iah_res.Division==aProv)&(iah_res.hazard==aDis)&(iah_res.rp==anRP)&(iah_res.quintile==myQ),['dw','pcwgt']].prod(axis=1).sum()/
                      iah_res.loc[(iah_res.Division==aProv)&(iah_res.hazard==aDis)&(iah_res.rp==anRP)&(iah_res.quintile==myQ),'pcwgt'].sum())/df.wprime.mean()
                
                pds_nrh = (iah_res.loc[(iah_res.Division==aProv)&(iah_res.hazard==aDis)&(iah_res.rp==anRP)&(iah_res.quintile==myQ),['pds_nrh','pcwgt']].prod(axis=1).sum()/
                           iah_res.loc[(iah_res.Division==aProv)&(iah_res.hazard==aDis)&(iah_res.rp==anRP)&(iah_res.quintile==myQ),'pcwgt'].sum())

                pds_dw = (iah_res.loc[(iah_res.Division==aProv)&(iah_res.hazard==aDis)&(iah_res.rp==anRP)&(iah_res.quintile==myQ),['pds_dw','pcwgt']].prod(axis=1).sum()/
                          iah_res.loc[(iah_res.Division==aProv)&(iah_res.hazard==aDis)&(iah_res.rp==anRP)&(iah_res.quintile==myQ),'pcwgt'].sum())/df.wprime.mean()
                
                ax.bar([6*ii+myQ for ii in range(1,7)],[k,dk,dc,dw,pds_nrh,pds_dw],
                       color=q_colors[myQ-1],alpha=0.7,label=q_labels[myQ-1])
            
            out_str = ['1% of assets','Asset loss','Consumption\nloss','Well-being\nloss','Net cost \nof help','Well-being loss\npost-support']
            for ni, ii in enumerate(range(1,7)):
                ax.annotate(out_str[ni],xy=(6*ii+1,ax.get_ylim()[0]/4.),xycoords='data',ha='left',va='top',weight='bold',fontsize=8,annotation_clip=False)

            fig = ax.get_figure()    
            leg = ax.legend(loc='best',labelspacing=0.75,ncol=1,fontsize=9,borderpad=0.75,fancybox=True,frameon=True,framealpha=0.9)
            leg.get_frame().set_color('white')
            leg.get_frame().set_edgecolor(greys_pal[7])
            leg.get_frame().set_linewidth(0.2)
            
            plt.plot([xlim for xlim in ax.get_xlim()],[0,0],'k-',lw=0.50,color=greys_pal[7],zorder=100,alpha=0.85)

            ax.xaxis.set_ticks([])
            plt.ylabel('Disaster losses ['+get_currency(myCountry)+' per capita]')

            print('poverty_k_'+aDis+'_'+str(anRP)+'.pdf')
            fig.savefig('../output_plots/'+myCountry+'/'+aProv+'_'+aDis+'_'+str(anRP)+'.pdf',format='pdf')#+'.pdf',format='pdf')
            plt.clf()
            plt.close('all')
                

#final_out,_ = average_over_rp(iah_ntl,'default_rp')

iah_ntl.to_csv('~/Desktop/poverty_ntl_by_haz.csv')

iah_ntl = iah_ntl.reset_index().set_index('rp').sum(level='rp')
iah_ntl.to_csv('~/Desktop/poverty_ntl.csv')
iah_sum = average_over_rp(iah_ntl,'default_rp').to_frame()
iah_sum.to_csv('~/Desktop/poverty_sum.csv')
