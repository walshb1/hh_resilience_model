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
q_labels = ['Poorest quintile','Q2','Q3','Q4','Wealthiest quintile']
q_colors = [sns_pal[0],sns_pal[1],sns_pal[2],sns_pal[3],sns_pal[5]]

font = {'family' : 'sans serif',
    'size'   : 20}
plt.rc('font', **font)
mpl.rcParams['xtick.labelsize'] = 16

import warnings
warnings.filterwarnings('always',category=UserWarning)

myCountry = 'PH'
if len(sys.argv) >= 2: myCountry = sys.argv[1]
print('Running '+myCountry)

model  = os.getcwd() #get current directory
output = model+'/../output_country/'+myCountry+'/'
output_plots = model+'/../output_plots/'+myCountry+'/'

economy = get_economic_unit(myCountry)
event_level = [economy, 'hazard', 'rp']
dem = get_demonym(myCountry)

# Set policy params
base_str = 'no'
pds_str = 'unif_poor'
pds2_str = 'no'
if myCountry == 'FJ':
    pds_str  = 'fiji_SPS'
    pds2_str = 'fiji_SPP'

drm_pov_sign = -1 # toggle subtraction or addition of dK to affected people's incomes
all_policies = []#['_exp095','_exr095','_ew100','_vul070','_vul070r','_rec067']

# Load base and PDS files
iah_base = pd.read_csv(output+'iah_tax_'+base_str+'_.csv', index_col=[economy,'hazard','rp','hhid'])
iah = pd.read_csv(output+'iah_tax_'+pds_str+'_.csv', index_col=[economy,'hazard','rp','hhid'])
df = pd.read_csv(output+'results_tax_'+pds_str+'_.csv', index_col=[economy,'hazard','rp'])
df_base = pd.read_csv(output+'results_tax_'+base_str+'_.csv', index_col=[economy,'hazard','rp'])
macro = pd.read_csv(output+'macro_tax_'+pds_str+'_.csv', index_col=[economy,'hazard','rp'])

iah_SP2, df_SP2, iah_infra = None, None,None
try:
    iah_SP2 = pd.read_csv(output+'iah_tax_'+pds2_str+'_.csv', index_col=[economy,'hazard','rp','hhid'])
    df_SP2  = pd.read_csv(output+'results_tax_'+pds2_str+'_.csv', index_col=[economy,'hazard','rp'])
    iah_infra = pd.read_csv(output+'iah_tax_'+base_str+'__ip.csv', index_col=[economy,'hazard','rp','hhid'])
    print('loaded all 3 extra files')
except: assert(False)

for iPol in all_policies:
    iah_pol = pd.read_csv(output+'iah_tax_'+pds_str+'_'+iPol+'.csv', index_col=[economy,'hazard','rp','hhid'])
    df_pol  = pd.read_csv(output+'results_tax_'+pds_str+'_'+iPol+'.csv', index_col=[economy,'hazard','rp'])

    iah['dk'+iPol] = iah_pol[['dk','pcwgt']].prod(axis=1)
    iah['dw'+iPol] = iah_pol[['dw','pcwgt']].prod(axis=1)/df_pol.wprime.mean()

    print(iPol,'added to iah (these policies are run *with* PDS)')

    del iah_pol
    del df_pol

# SAVE OUT SOME RESULTS FILES
df_prov = df[['dKtot','dWtot_currency']].copy()
df_prov['gdp'] = df[['pop','gdp_pc_pp_prov']].prod(axis=1)
results_df = macro.reset_index().set_index([economy,'hazard'])
results_df = results_df.loc[results_df.rp==100,'dk_event'].sum(level='hazard')
results_df = results_df.rename(columns={'dk_event':'dk_event_100'})
results_df = pd.concat([results_df,df_prov.reset_index().set_index([economy,'hazard']).sum(level='hazard')['dKtot']],axis=1,join='inner')
results_df.columns = ['dk_event_100','AAL']
results_df.to_csv(output+'results_table_new.csv')
print('Writing '+output+'results_table_new.csv')

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
if get_subsistence_line(myCountry) != None: iah_res['sub_line'] = get_subsistence_line(myCountry)

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

try:
    iah_res['dc_inf_npv_pre'] = iah_infra[['dc_npv_pre','pcwgt']].prod(axis=1).sum(level=[economy,'hazard','rp','hhid'])/iah_res['pcwgt']
    iah_res['dc_inf_npv_post'] = iah_infra[['dc_npv_post','pcwgt']].prod(axis=1).sum(level=[economy,'hazard','rp','hhid'])/iah_res['pcwgt']
    iah_res['dc_delta'] = iah_res['dc_inf_npv_pre'] - iah_res['dc_npv_pre']

    iah_res = iah_res.reset_index()
    grab = [['TC'],[100,1000]]
    for aHaz in grab[0]:
        for anRP in grab[1]:
            
            ax = iah_res.loc[(iah_res.hazard==aHaz)&(iah_res.rp==anRP)].plot.scatter('dc_npv_pre','dc_inf_npv_pre',c='quintile')   
            ax.plot()
            fig = plt.gcf()
            fig.savefig(output_plots+'experiments/infra_dc_npv_pre.pdf',format='pdf')
            print('infra plot 1/2')
            plt.clf()

            ax = iah_res.loc[(iah_res.hazard==aHaz)&(iah_res.rp==anRP)&(iah_res.quintile==1)].plot.scatter('dc_npv_pre','dc_delta',color=q_colors[0],label='Q1')
            for iq in range(2,6): 
                iah_res.loc[(iah_res.hazard==aHaz)&(iah_res.rp==anRP)&(iah_res.quintile==iq)].plot.scatter('dc_npv_pre','dc_delta', color=q_colors[iq-1], label='Q'+str(iq), ax=ax)
            ax.plot()
            plt.xlim(0,6000)
            plt.ylim(-5000,5000)
            fig = plt.gcf()
            fig.savefig(output_plots+'experiments/infra_dc_npv_pre_delta_'+aHaz+str(anRP)+'.pdf',format='pdf')#flag
            print('infra plot 2/2')
            plt.clf()
            plt.close('all')

    iah_res = iah_res.reset_index().set_index(event_level+['hhid']).drop(['index'],axis=1)

except:
    assert(False)
#except: print('DID NOT load infra') 

assert(False)

# These are the other policies (scorecard)
# ^ already weighted by pcwgt from their respective files
for iPol in all_policies:
    print('dk'+iPol)
    iah_res['dk'+iPol] = iah['dk'+iPol].sum(level=[economy,'hazard','rp','hhid'])/iah_res['pcwgt']
    iah_res['dw'+iPol] = iah['dw'+iPol].sum(level=[economy,'hazard','rp','hhid'])/iah_res['pcwgt']

# Note that we're pulling dw in from iah_base here
iah_res['dw']          = (iah_base[['dw','pcwgt']].prod(axis=1).sum(level=[economy,'hazard','rp','hhid'])/iah_res['pcwgt'])/df.wprime.mean()
iah_res['pds_dw']      = (     iah[['dw','pcwgt']].prod(axis=1).sum(level=[economy,'hazard','rp','hhid'])/iah_res['pcwgt'])/df.wprime.mean()
try: iah_res['pds_plus_dw'] = ( iah_SP2[['dw','pcwgt']].prod(axis=1).sum(level=[economy,'hazard','rp','hhid'])/iah_res['pcwgt'])/df_SP2.wprime.mean()
except: pass

iah_res['c_initial']   = iah[['c_initial'  ,'pcwgt_ae']].prod(axis=1).sum(level=[economy,'hazard','rp','hhid'])/iah_res['pcwgt_ae'] # c per AE
iah_res['delta_c']     = iah[['delta_c'    ,'pcwgt_ae']].prod(axis=1).sum(level=[economy,'hazard','rp','hhid'])/iah_res['pcwgt_ae'] # dc per AE
iah_res['pds_nrh']     = iah[['pds_nrh'    ,'pcwgt_ae']].prod(axis=1).sum(level=[economy,'hazard','rp','hhid'])/iah_res['pcwgt_ae'] # nrh per AE
iah_res['c_final']     = iah[['c_final'    ,'pcwgt_ae']].prod(axis=1).sum(level=[economy,'hazard','rp','hhid'])/iah_res['pcwgt_ae'] # c per AE
iah_res['c_final_pds'] = iah[['c_final_pds','pcwgt_ae']].prod(axis=1).sum(level=[economy,'hazard','rp','hhid'])/iah_res['pcwgt_ae'] # c per AE

# Huge file
del iah_base
# Calc people who fell into povery on the regional level for each disaster
#iah_res.loc[(iah_res.pcinc_ae > iah_res.pov_line)&(iah_res.c_final < iah_res.pov_line),'pcwgt'].sum(level=[economy,'hazard','rp'])

iah = iah.reset_index()
iah_res  = iah_res.reset_index().set_index([economy,'hazard','rp','hhid'])

# Save out iah by economic unit
iah_out = pd.DataFrame(index=iah_res.sum(level=[economy,'hazard','rp']).index)
for iPol in ['']+all_policies:
    iah_out['Asset risk'+iPol] = iah_res[['dk'+iPol,'pcwgt']].prod(axis=1).sum(level=[economy,'hazard','rp'])
    iah_out['Well-being risk'+iPol] = iah_res[['dw'+iPol,'pcwgt']].prod(axis=1).sum(level=[economy,'hazard','rp'])
#iah_out['pds_dw']      = iah_res[['pds_dw','pcwgt']].prod(axis=1).sum(level=[economy,'hazard','rp'])
#iah_out['help_fee']    = iah_res[['help_fee','pcwgt']].prod(axis=1).sum(level=[economy,'hazard','rp'])
#iah_out['pds_plus_dw'] = iah_res[['pds_plus_dw','pcwgt']].prod(axis=1).sum(level=[economy,'hazard','rp'])
iah_out['SE capacity']  = iah_out['Asset risk']/iah_out['Well-being risk']

iah_out.to_csv(output+'geo_sums.csv')
print(iah_out.head(10))
iah_out,_ = average_over_rp(iah_out,'default_rp')
iah_out['SE capacity']  = iah_out['Asset risk']/iah_out['Well-being risk']
iah_out.to_csv(output+'geo_haz_aal_sums.csv')

iah_out = iah_out.sum(level=economy)
iah_out['SE capacity']  = iah_out['Asset risk']/iah_out['Well-being risk']
iah_out.to_csv(output+'geo_aal_sums.csv')
iah_out[['Asset risk','Well-being risk']]/=1.E6
iah_out[['SE capacity']]*=100.
iah_out[['Asset risk','SE capacity','Well-being risk']].sort_values(['Well-being risk'],ascending=False).astype(int).to_latex(output+'geo_aal_sums.tex')
print('Wrote latex! Sum: ',iah_out.sum())

# Save out iah by economic unit, *only for poorest quintile*
iah_out = pd.DataFrame(index=iah_res.sum(level=[economy,'hazard','rp']).index)
for iPol in ['']+all_policies:
    iah_out['Asset risk'+iPol] = iah_res.loc[(iah_res.quintile==1),['dk'+iPol,'pcwgt']].prod(axis=1).sum(level=[economy,'hazard','rp'])
    iah_out['Well-being risk'+iPol] = iah_res.loc[(iah_res.quintile==1),['dw'+iPol,'pcwgt']].prod(axis=1).sum(level=[economy,'hazard','rp'])
#iah_out['pds_dw']      = iah_res.loc[(iah_res.quintile==1),['pds_dw','pcwgt']].prod(axis=1).sum(level=[economy,'hazard','rp'])
#iah_out['help_fee']    = iah_res.loc[(iah_res.quintile==1),['help_fee','pcwgt']].prod(axis=1).sum(level=[economy,'hazard','rp'])
#iah_out['pds_plus_dw'] = iah_res.loc[(iah_res.quintile==1),['pds_plus_dw','pcwgt']].prod(axis=1).sum(level=[economy,'hazard','rp'])
iah_out['SE capacity']  = iah_out['Asset risk']/iah_out['Well-being risk']

iah_out.to_csv(output+'geo_sums_q1.csv')
iah_out,_ = average_over_rp(iah_out,'default_rp')
iah_out['SE capacity']  = iah_out['Asset risk']/iah_out['Well-being risk']
iah_out.to_csv(output+'geo_haz_aal_sums_q1.csv')

iah_out = iah_out.sum(level=economy)
iah_out['SE capacity']  = iah_out['Asset risk']/iah_out['Well-being risk']
iah_out.to_csv(output+'geo_aal_sums_q1.csv')
iah_out[['Asset risk','Well-being risk']]/=1.E6
iah_out[['SE capacity']]*=100.
iah_out[['Asset risk','SE capacity','Well-being risk']].sort_values(['Well-being risk'],ascending=False).astype(int).to_latex(output+'geo_aal_sums_q1.tex',bold_rows=True)
print('Wrote latex! Q1 sums: ',iah_out.sum())

# Save out iah
iah_out = pd.DataFrame(index=iah_res.sum(level=['hazard','rp']).index)
for iPol in ['']+all_policies:
    iah_out['dk'+iPol] = iah_res[['dk'+iPol,'pcwgt']].prod(axis=1).sum(level=['hazard','rp'])
    iah_out['dw'+iPol] = iah_res[['dw'+iPol,'pcwgt']].prod(axis=1).sum(level=['hazard','rp'])
iah_out['pds_dw'] = iah_res[['pds_dw','pcwgt']].prod(axis=1).sum(level=['hazard','rp'])
iah_out['help_fee'] = iah_res[['help_fee','pcwgt']].prod(axis=1).sum(level=['hazard','rp'])
try: iah_out['pds_plus_dw'] = iah_res[['pds_plus_dw','pcwgt']].prod(axis=1).sum(level=['hazard','rp'])
except: pass

iah_out.to_csv(output+'haz_sums.csv')
print(iah_out.head(10))
iah_out,_ = average_over_rp(iah_out,'default_rp')
iah_out.to_csv(output+'sums.csv')

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
print('Initial poverty incidence:',iah_ntl[['pov_pc_i','pov_hh_i']].mean())

iah_ntl['eff_pds'] = iah_ntl['pov_pc_pds_D'] - iah_ntl['pov_pc_D']

# Print out plots for iah_res
iah_res = iah_res.reset_index()
iah_ntl = iah_ntl.reset_index()

myHaz = None
if myCountry == 'FJ': myHaz = [['Ba','Lau','Tailevu'],get_all_hazards(myCountry,iah_res),[1,100,500]]
elif myCountry == 'PH': myHaz = [['NCR'],get_all_hazards(myCountry,iah_res),get_all_rps(myCountry,iah_res)]

##################################################################
# This code generates the histograms showing income before & after disaster
# ^ this is nationally, so we'll use iah_res 
upper_clip = 2E5
scale_fac = 1.0
if myCountry == 'FJ': 
    scale_fac = 2.321208
    upper_clip = 2E4

for aReg in myHaz[0]:
    for aDis in get_all_hazards(myCountry,iah_res):
        
        myC_ylim = None
        c_bins = [None,50]
        for anRP in myHaz[2][::-1]:        

            ax=plt.gca()

            cf_heights, cf_bins = np.histogram((iah_res.loc[(iah_res[economy]==aReg)&(iah_res.hazard==aDis)&(iah_res.rp==anRP),'c_final']/scale_fac).clip(upper=upper_clip), bins=c_bins[1], 
                                               weights=iah_res.loc[(iah_res[economy]==aReg)&(iah_res.hazard==aDis)&(iah_res.rp==anRP),'pcwgt']/get_scale_fac(myCountry)[0])
            if c_bins[0] == None: c_bins = [cf_bins,cf_bins]
            ci_heights, ci_bins = np.histogram((iah_res.loc[(iah_res[economy]==aReg)&(iah_res.hazard==aDis)&(iah_res.rp==anRP),'c_initial']/scale_fac).clip(upper=upper_clip), bins=c_bins[1], 
                                               weights=iah_res.loc[(iah_res[economy]==aReg)&(iah_res.hazard==aDis)&(iah_res.rp==anRP),'pcwgt']/get_scale_fac(myCountry)[0])
        
            ax.bar(c_bins[1][:-1], ci_heights, width=(c_bins[1][1]-c_bins[1][0]), label=aReg+' - Initial', facecolor=q_colors[1],alpha=0.4)
            ax.bar(c_bins[1][:-1], cf_heights, width=(c_bins[1][1]-c_bins[1][0]), label=aReg+' - Post-disaster', facecolor=q_colors[0],alpha=0.4)
            
            if myC_ylim == None: myC_ylim = ax.get_ylim()
            plt.ylim(myC_ylim[0],1.4*myC_ylim[1])

            plt.xlim(0,upper_clip)
            plt.xlabel(r'Income ('+get_currency(myCountry)+' yr$^{-1}$)')
            plt.ylabel('Population'+get_scale_fac(myCountry)[1])
            plt.title(str(anRP)+'-year '+aDis+' event in '+aReg)
            leg = ax.legend(loc='best',labelspacing=0.75,ncol=1,fontsize=9,borderpad=0.75,fancybox=True,frameon=True,framealpha=0.9)
            leg.get_frame().set_color('white')
            leg.get_frame().set_edgecolor(greys_pal[7])
            leg.get_frame().set_linewidth(0.2)

            ax.annotate('Total asset losses: '+str(round(iah_res.loc[(iah_res[economy]==aReg)&(iah_res.hazard==aDis)&(iah_res.rp==anRP),['pcwgt','dk']].prod(axis=1).sum()/1.E9,1))+' bill. '+get_currency(myCountry), 
                        xy=(0.03,-0.18), xycoords=leg.get_frame(),size=7,va='top',ha='left',annotation_clip=False,zorder=100)
            ax.annotate('Total well-being losses: '+str(round(iah_res.loc[(iah_res[economy]==aReg)&(iah_res.hazard==aDis)&(iah_res.rp==anRP),['pcwgt','dw']].prod(axis=1).sum()/1.E9,1))+' bill. '+get_currency(myCountry), 
                        xy=(0.03,-0.50), xycoords=leg.get_frame(),size=7,va='top',ha='left',annotation_clip=False,zorder=100)

            new_pov = int(iah_res.loc[((iah_res[economy]==aReg)&(iah_res.hazard==aDis)&(iah_res.rp==anRP)
                                       &(iah_res.pcinc_ae > iah_res.pov_line)&(iah_res.c_final < iah_res.pov_line)),'pcwgt'].sum())
            new_pov_pct = round(100.*float(new_pov)/float(iah_res.loc[(iah_res[economy]==aReg)&(iah_res.hazard==aDis)&(iah_res.rp==anRP),'pcwgt'].sum()),1)
            plt.plot([iah_res.pov_line.mean()/scale_fac,iah_res.pov_line.mean()/scale_fac],[0,1.21*cf_heights[:-2].max()],'k-',lw=2.5,color=greys_pal[7],zorder=100,alpha=0.85)
            ax.annotate('Poverty line',xy=(1.1*iah_res.pov_line.mean()/scale_fac,1.21*cf_heights[:-2].max()),xycoords='data',ha='left',va='top',fontsize=8,annotation_clip=False,weight='bold')
            ax.annotate(r'$\Delta$N$_p$ = +'+int_w_commas(new_pov)+' ('+str(new_pov_pct)+'% of population)',
                        xy=(1.1*iah_res.pov_line.mean()/scale_fac,1.14*cf_heights[:-2].max()),xycoords='data',ha='left',va='top',fontsize=8,annotation_clip=False)

            sub_line, new_sub = get_subsistence_line(myCountry), None
            if sub_line is not None:
                new_sub = int(iah_res.loc[((iah_res[economy]==aReg)&(iah_res.hazard==aDis)&(iah_res.rp==anRP)
                                           &(iah_res.pcinc_ae > sub_line)&(iah_res.c_final < sub_line)),'pcwgt'].sum())
                new_sub_pct = round(100.*float(new_sub)/float(iah_res.loc[(iah_res[economy]==aReg)&(iah_res.hazard==aDis)&(iah_res.rp==anRP),'pcwgt'].sum()),1)
                plt.plot([sub_line,sub_line],[0,1.41*cf_heights[:-2].max()],'k-',lw=2.5,color=greys_pal[7],zorder=100,alpha=0.85)
                ax.annotate('Subsistence line',xy=(1.1*sub_line,1.41*cf_heights[:-2].max()),xycoords='data',ha='left',va='top',fontsize=9,annotation_clip=False,weight='bold')
                ax.annotate(r'$\Delta$N$_s$ = +'+int_w_commas(new_sub)+' ('+str(new_sub_pct)+'% of population)',
                            xy=(1.1*sub_line,1.34*cf_heights[:-2].max()),xycoords='data',ha='left',va='top',fontsize=8,annotation_clip=False)

            print(aReg,aDis,anRP,new_pov,'people into poverty &',new_sub,'into subsistence') 

            fig = ax.get_figure()
            fig.savefig(output_plots+'npr_poverty_k_'+aReg+'_'+aDis+'_'+str(anRP)+'.pdf',format='pdf')
            fig.savefig(output_plots+'png/npr_poverty_k_'+aReg+'_'+aDis+'_'+str(anRP)+'.png',format='png')
            plt.clf()
            plt.close('all')
            print(aReg+'_poverty_k_'+aDis+'_'+str(anRP)+'.pdf')

##################################################################
# This code generates the histograms including [k,dk,dc,dw,&pds]
# ^ this is by province, so it will use iah_res
for aProv in myHaz[0]:
    for aDis in myHaz[1]:
        for anRP in myHaz[2]:

            plt.figure(1)
            ax = plt.subplot(111)

            plt.figure(2)
            ax2 = plt.subplot(111)

            for myQ in range(1,6): #nQuintiles
    
                print(aProv,aDis,anRP,'shape:',iah_res.loc[(iah_res[economy]==aProv)&(iah_res.hazard==aDis)&(iah_res.rp==anRP)&(iah_res.quintile==myQ),'pcwgt'].shape[0])
                
                k = (0.01*iah_res.loc[(iah_res[economy]==aProv)&(iah_res.hazard==aDis)&(iah_res.rp==anRP)&(iah_res.quintile==myQ),['k','pcwgt']].prod(axis=1).sum()/
                     iah_res.loc[(iah_res[economy]==aProv)&(iah_res.hazard==aDis)&(iah_res.rp==anRP)&(iah_res.quintile==myQ),'pcwgt'].sum())
                
                dk = (iah_res.loc[(iah_res[economy]==aProv)&(iah_res.hazard==aDis)&(iah_res.rp==anRP)&(iah_res.quintile==myQ),['dk','pcwgt']].prod(axis=1).sum()/
                      iah_res.loc[(iah_res[economy]==aProv)&(iah_res.hazard==aDis)&(iah_res.rp==anRP)&(iah_res.quintile==myQ),'pcwgt'].sum())
                
                dc = (iah_res.loc[(iah_res[economy]==aProv)&(iah_res.hazard==aDis)&(iah_res.rp==anRP)&(iah_res.quintile==myQ),['dc_npv_pre','pcwgt']].prod(axis=1).sum()/
                      iah_res.loc[(iah_res[economy]==aProv)&(iah_res.hazard==aDis)&(iah_res.rp==anRP)&(iah_res.quintile==myQ),'pcwgt'].sum())
                
                dw = (iah_res.loc[(iah_res[economy]==aProv)&(iah_res.hazard==aDis)&(iah_res.rp==anRP)&(iah_res.quintile==myQ),['dw','pcwgt']].prod(axis=1).sum()/
                      iah_res.loc[(iah_res[economy]==aProv)&(iah_res.hazard==aDis)&(iah_res.rp==anRP)&(iah_res.quintile==myQ),'pcwgt'].sum())
                
                pds_nrh = (iah_res.loc[(iah_res[economy]==aProv)&(iah_res.hazard==aDis)&(iah_res.rp==anRP)&(iah_res.quintile==myQ),['pds_nrh','pcwgt']].prod(axis=1).sum()/
                           iah_res.loc[(iah_res[economy]==aProv)&(iah_res.hazard==aDis)&(iah_res.rp==anRP)&(iah_res.quintile==myQ),'pcwgt'].sum())

                pds_dw = (iah_res.loc[(iah_res[economy]==aProv)&(iah_res.hazard==aDis)&(iah_res.rp==anRP)&(iah_res.quintile==myQ),['pds_dw','pcwgt']].prod(axis=1).sum()/
                          iah_res.loc[(iah_res[economy]==aProv)&(iah_res.hazard==aDis)&(iah_res.rp==anRP)&(iah_res.quintile==myQ),'pcwgt'].sum())

                try:
                    pds_plus_dw = (iah_res.loc[(iah_res[economy]==aProv)&(iah_res.hazard==aDis)&(iah_res.rp==anRP)&(iah_res.quintile==myQ),['pds_plus_dw','pcwgt']].prod(axis=1).sum()/
                                   iah_res.loc[(iah_res[economy]==aProv)&(iah_res.hazard==aDis)&(iah_res.rp==anRP)&(iah_res.quintile==myQ),'pcwgt'].sum())
                except: pds_plus_dw = 0

                ax.bar([6*ii+myQ for ii in range(1,6)],[dk,dc,dw,pds_nrh,pds_dw],
                       color=q_colors[myQ-1],alpha=0.7,label=q_labels[myQ-1])

                lbl= None
                if myQ==1: 
                    ax2.bar([0],[0],color=[q_colors[0]],alpha=0.7,label='No post-disaster support')
                    ax2.bar([0],[0],color=[q_colors[1]],alpha=0.7,label='Winston-like response')
                    ax2.bar([0],[0],color=[q_colors[2]],alpha=0.7,label='Wider & stronger response')
                ax2.bar([4*myQ+ii for ii in range(1,4)],[dw,pds_dw,pds_plus_dw],color=[q_colors[0],q_colors[1],q_colors[2]],alpha=0.7)
                
            out_str = None
            if myCountry == 'FJ': out_str = ['Asset loss','Consumption\nloss','Well-being\nloss','Net cost of\nWinston-like\nsupport','Well-being loss\npost support']
            elif myCountry == 'PH': out_str = ['Asset loss','Consumption\nloss','Well-being\nloss','Net cost of\nsupport','Well-being loss\npost support']

            for ni, ii in enumerate(range(1,6)):
                ax.annotate(out_str[ni],xy=(6*ii+1,ax.get_ylim()[0]/4.),xycoords='data',ha='left',va='top',weight='bold',fontsize=8,annotation_clip=False)

            fig = ax.get_figure()    
            leg = ax.legend(loc='best',labelspacing=0.75,ncol=1,fontsize=9,borderpad=0.75,fancybox=True,frameon=True,framealpha=0.9)
            leg.get_frame().set_color('white')
            leg.get_frame().set_edgecolor(greys_pal[7])
            leg.get_frame().set_linewidth(0.2)
            
            plt.figure(1)
            plt.plot([xlim for xlim in ax.get_xlim()],[0,0],'k-',lw=0.50,color=greys_pal[7],zorder=100,alpha=0.85)
            ax.xaxis.set_ticks([])
            plt.ylabel('Disaster losses ('+get_currency(myCountry)+' per capita)')

            print('losses_k_'+aDis+'_'+str(anRP)+'.pdf')
            fig.savefig(output_plots+'npr_'+aProv+'_'+aDis+'_'+str(anRP)+'.pdf',format='pdf')#+'.pdf',format='pdf')
            fig.savefig(output_plots+'png/npr_'+aProv+'_'+aDis+'_'+str(anRP)+'.png',format='png')

            plt.figure(2)
            fig2 = ax2.get_figure()
            leg = ax2.legend(loc='best',labelspacing=0.75,ncol=1,fontsize=9,borderpad=0.75,fancybox=True,frameon=True,framealpha=0.9)
            leg.get_frame().set_color('white')
            leg.get_frame().set_edgecolor(greys_pal[7])
            leg.get_frame().set_linewidth(0.2)

            ann_y = -ax2.get_ylim()[1]/50

            out_str = ['Q1','Q2','Q3','Q4','Q5']
            for ni, ii in enumerate(range(1,6)):
                ax2.annotate(out_str[ni],xy=(4*ii+1.05,ann_y),zorder=100,xycoords='data',
                             ha='left',va='center',weight='bold',fontsize=8,annotation_clip=False)
                plt.plot([4*ii+1.50,4*ii+4.00],[ann_y,ann_y],'k-',lw=0.50,
                         color=greys_pal[7],zorder=100,alpha=0.85)
                plt.plot([4*ii+4.00,4*ii+4.00],[ann_y*0.9,ann_y*1.1],'k-',lw=0.50,color=greys_pal[7],zorder=100,alpha=0.85)

            ax2.xaxis.set_ticks([])
            plt.xlim(3,26)
            plt.plot([i for i in ax2.get_xlim()],[0,0],'k-',lw=2.5,color=greys_pal[7],zorder=100,alpha=0.85)
            plt.ylabel('Well-being losses ('+get_currency(myCountry)+' per capita)')
            fig2.savefig(output_plots+'npr_pds_schemes_'+aProv+'_'+aDis+'_'+str(anRP)+'.pdf',format='pdf')
            fig2.savefig(output_plots+'png/npr_pds_schemes_'+aProv+'_'+aDis+'_'+str(anRP)+'.png',format='png')
        
            plt.clf()
            plt.close('all')
                

iah_ntl.to_csv(output+'poverty_ntl_by_haz.csv')
iah_ntl = iah_ntl.reset_index().set_index(['hazard','rp'])
iah_ntl_haz,_ = average_over_rp(iah_ntl,'default_rp')
iah_ntl_haz.sum(level='hazard').to_csv(output+'poverty_haz_sum.csv')

iah_ntl = iah_ntl.reset_index().set_index('rp').sum(level='rp')
iah_ntl.to_csv(output+'poverty_ntl.csv')
iah_sum,_ = average_over_rp(iah_ntl,'default_rp')
iah_sum.sum().to_csv(output+'poverty_sum.csv')
