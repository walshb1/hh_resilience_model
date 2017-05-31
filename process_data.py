#This script processes data outputs for the resilience indicator multihazard model for the Philippines. Developed by Brian Walsh.
from IPython import get_ipython
get_ipython().magic('reset -f')
get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')

#Import package for data analysis
from replace_with_warning import *
from lib_gather_data import *
from maps_lib import *

#from wquantiles import median, quantile

from pandas import isnull
import pandas as pd
import numpy as np
import os, time
import sys

from lib_compute_resilience_and_risk import *

#Aesthetics
import seaborn as sns
import brewer2mpl as brew
from matplotlib import colors
sns.set_style('darkgrid')
brew_pal = brew.get_map('Set1', 'qualitative', 8).mpl_colors
sns_pal = sns.color_palette('Set1', n_colors=8, desat=.5)

import warnings
warnings.filterwarnings('always',category=UserWarning)

if len(sys.argv) < 2:
    print('Need to list country. Try PH')
    assert(False)
else: myCountry = sys.argv[1]

model  = os.getcwd() #get current directory
output = model+'/../output_country/'+myCountry+'/'

event_level = ['province', 'hazard', 'rp']

res_base = pd.read_csv(output+'results_tax_no_.csv', index_col=['province','hazard','rp'])
res_unif_poor_only = pd.read_csv(output+'results_tax_unif_poor_only_.csv', index_col=['province','hazard','rp'])

df = pd.read_csv(output+'results_tax_no_.csv', index_col=['province','hazard','rp'])

print(df[['fa','n','weight']].head(25))
assert(False)

iah = pd.read_csv(output+'iah_tax_no_.csv', index_col=['province','hazard','rp'])
iah_pds = pd.read_csv(output+'iah_tax_unif_poor_only_.csv', index_col=['province','hazard','rp'])

#cats = pd.read_csv(output+'cats_tax_no_.csv', index_col=['province','hazard','rp'])

print(iah.columns)

# Transform dw:
wprime = df.wprime.mean()
print(wprime)

iah['dw'] = iah['dw']/wprime
iah['pds_dw']  = iah_pds['dw']/wprime

iah['pds_nrh'] = iah_pds['help_fee']-iah_pds['help_received'] # Net received help for Q1-2
#iah['no_pds_nrh'] = iah['help_received']-iah['help_fee'] # Net received help for Q3-5

q_labels = ['Poorest quintile','Q2','Q3','Q4','Wealthiest quintile']
q_colors = [sns_pal[0],sns_pal[1],sns_pal[2],sns_pal[3],sns_pal[5]]

# Look at single event:
myHaz = [['Aurora','Bulacan'],['flood','earthquake'],[1,50,1500]]

iah = iah.reset_index()
for myProv in myHaz[0]:
    for myDis in myHaz[1]:
        for myRP in myHaz[2]:
    
            print('len_t=',iah.shape[0])
            cut = iah.loc[(iah.province == myProv) & (iah.hazard == myDis) & (iah.rp == myRP)].set_index(['province','hazard','rp'])

            print(cut.head(50))
            assert(False)

            q1 = cut.loc[cut.quintile == 1].reset_index()
            q2 = cut.loc[cut.quintile == 2].reset_index()
            q3 = cut.loc[cut.quintile == 3].reset_index()
            q4 = cut.loc[cut.quintile == 4].reset_index()
            q5 = cut.loc[cut.quintile == 5].reset_index()

            k_meds = get_weighted_median(q1,q2,q3,q4,q5,'k')
            k_mean = get_weighted_mean(q1,q2,q3,q4,q5,'k')
            
            dk_meds = get_weighted_median(q1,q2,q3,q4,q5,'dk')
            dk_mean = get_weighted_mean(q1,q2,q3,q4,q5,'dk')

            dc_meds = get_weighted_median(q1,q2,q3,q4,q5,'dc_npv_pre')
            dc_mean = get_weighted_mean(q1,q2,q3,q4,q5,'dc_npv_pre')

            dw_meds = get_weighted_median(q1,q2,q3,q4,q5,'dw')
            dw_mean = get_weighted_mean(q1,q2,q3,q4,q5,'dw')

            nrh_meds = get_weighted_median(q1,q2,q3,q4,q5,'pds_nrh')
            nrh_mean = get_weighted_mean(q1,q2,q3,q4,q5,'pds_nrh')

            dw_pds_meds = get_weighted_median(q1,q2,q3,q4,q5,'pds_dw')
            dw_pds_mean = get_weighted_mean(q1,q2,q3,q4,q5,'pds_dw')

            print('\nk/10 (med) = ',0.1*np.array(k_meds))
            print('k/10 (avg) = ',0.1*np.array(k_mean))

            print('\ndk (med) = ',dk_meds)
            print('dk (avg) = ',dk_mean)

            print('\ndc (med) = ',dc_meds)
            print('dc (avg) = ',dc_mean)

            print('\ndw (med) = ',dw_meds)
            print('dw (avg) = ',dw_mean)

            # histograms
            df_wgt = pd.DataFrame({'q1_w': q1['weight'],'q2_w': q2['weight'],'q3_w': q3['weight'],'q4_w': q4['weight'],'q5_w': q5['weight']}, 
                                  columns=['q1_w', 'q2_w', 'q3_w', 'q4_w', 'q5_w']).dropna()
            for istr in ['dk','dc','dw']:
                df_tmp = pd.DataFrame({'q1': q1[istr],'q2': q2[istr],'q3': q3[istr],'q4': q4[istr],'q5': q5[istr]},columns=['q1', 'q2', 'q3', 'q4', 'q5']).dropna()

                q1_heights, q1_bins = np.histogram(df_tmp['q1'],weights=df_wgt['q1_w'])
                q2_heights, q2_bins = np.histogram(df_tmp['q2'],weights=df_wgt['q2_w'],bins=q1_bins)
                q3_heights, q3_bins = np.histogram(df_tmp['q3'],weights=df_wgt['q3_w'],bins=q1_bins)
                q4_heights, q4_bins = np.histogram(df_tmp['q4'],weights=df_wgt['q4_w'],bins=q1_bins)
                q5_heights, q5_bins = np.histogram(df_tmp['q5'],weights=df_wgt['q5_w'],bins=q1_bins)

                width = (q1_bins[1] - q1_bins[0])/6
            
                ax = plt.gca()
                ax.bar(q1_bins[:-1], q1_heights, width=width, label='q1', facecolor=q_colors[0],alpha=0.5)
                ax.bar(q2_bins[:-1]+1*width, q2_heights, width=width, label='q2', facecolor=q_colors[1],alpha=0.5)
                ax.bar(q3_bins[:-1]+2*width, q3_heights, width=width, label='q3', facecolor=q_colors[2],alpha=0.5)
                ax.bar(q4_bins[:-1]+3*width, q4_heights, width=width, label='q4', facecolor=q_colors[3],alpha=0.5)
                ax.bar(q5_bins[:-1]+4*width, q5_heights, width=width, label='q5', facecolor=q_colors[4],alpha=0.5)

                plt.title(myDis+' in '+myProv+' (rp = '+str(myRP)+') - '+istr)
                plt.xlabel(istr,fontsize=12)
                plt.legend()
                
                fig = ax.get_figure()
                print('Saving: hists/'+istr+'_'+myProv+'_'+myDis+'_'+str(myRP)+'.pdf')
                fig.savefig('../output_plots/PH/'+istr+'_'+myProv+'_'+myDis+'_'+str(myRP)+'.pdf',format='pdf')
                plt.cla()

            # Median
            ax1 = plt.subplot(111)
            for ij in range(0,5):
                ax1.bar([6*ii+ij for ii in range(1,7)],[0.1*np.array(k_meds[ij]),dk_meds[ij],dc_meds[ij],dw_meds[ij],nrh_meds[ij],dw_pds_meds[ij]],color=q_colors[ij],alpha=0.7,label=q_labels[ij])
            ax1.xaxis.set_ticks([])
            plt.ylabel('Median PHP ('+myProv+', '+myDis+', rp='+str(myRP)+' yr)')
            ax1.annotate('10% of assets',             xy=( 6,-1500),xycoords='data',ha='left',va='top',fontsize=8,annotation_clip=False)
            ax1.annotate('Asset loss',                xy=(12,-1500),xycoords='data',ha='left',va='top',fontsize=8,annotation_clip=False)
            ax1.annotate('Consumption\nloss',         xy=(18,-1500),xycoords='data',ha='left',va='top',fontsize=8,annotation_clip=False)
            ax1.annotate('Welfare loss',              xy=(24,-1500),xycoords='data',ha='left',va='top',fontsize=8,annotation_clip=False)
            ax1.annotate('Net cost \nof help',        xy=(30,-1500),xycoords='data',ha='left',va='top',fontsize=8,annotation_clip=False)
            ax1.annotate('Welfare loss\npost-support',xy=(36,-1500),xycoords='data',ha='left',va='top',fontsize=8,annotation_clip=False)
            #ax1.annotate('Consumption\nloss (post, NPV)',xy=(36,-500),xycoords='data',ha='left',va='top',annotation_clip=False)
            ax1.legend()

            print('Saving: histo_'+myProv+'_'+myDis+'_'+str(myRP)+'.pdf\n')
            plt.savefig('../output_plots/PH/medians_'+myProv+'_'+myDis+'_'+str(myRP)+'.pdf',bbox_inches='tight',format='pdf')
            plt.cla()

            # Means
            ax1 = plt.subplot(111)
            for ij in range(0,5):
                ax1.bar([6*ii+ij for ii in range(1,7)],[0.1*np.array(k_mean[ij]),dk_mean[ij],dc_mean[ij],dw_mean[ij],nrh_mean[ij],dw_pds_mean[ij]],color=q_colors[ij],alpha=0.7,label=q_labels[ij])
            ax1.xaxis.set_ticks([])
            plt.ylabel('Mean PHP ('+myProv+', '+myDis+', rp='+str(myRP)+' yr)')
            ax1.annotate('10% of assets',             xy=( 6,-1500),xycoords='data',ha='left',va='top',fontsize=8,annotation_clip=False)
            ax1.annotate('Asset loss',                xy=(12,-1500),xycoords='data',ha='left',va='top',fontsize=8,annotation_clip=False)
            ax1.annotate('Consumption\nloss',         xy=(18,-1500),xycoords='data',ha='left',va='top',fontsize=8,annotation_clip=False)
            ax1.annotate('Welfare loss',              xy=(24,-1500),xycoords='data',ha='left',va='top',fontsize=8,annotation_clip=False)
            ax1.annotate('Net cost \nof help',        xy=(30,-1500),xycoords='data',ha='left',va='top',fontsize=8,annotation_clip=False)
            ax1.annotate('Welfare loss\npost-support',xy=(36,-1500),xycoords='data',ha='left',va='top',fontsize=8,annotation_clip=False)
            #ax1.annotate('Consumption\nloss (post, NPV)',xy=(36,-500),xycoords='data',ha='left',va='top',annotation_clip=False)
            ax1.legend()

            print('Saving: histo_'+myProv+'_'+myDis+'_'+str(myRP)+'.pdf\n')
            plt.savefig('../output_plots/PH/means_'+myProv+'_'+myDis+'_'+str(myRP)+'.pdf',bbox_inches='tight',format='pdf')
            plt.cla()

# Don't think this is the correct weighting...
df_prov_mh = pd.DataFrame(index=df[['risk','risk_to_assets']].sum(level='province').index)
df_prov_mh['risk'] = (df.risk*df.weight/(df.weight.sum(level=event_level))).sum(level='province')
df_prov_mh['risk_to_assets'] = (df.risk_to_assets*df.weight/(df.weight.sum(level=event_level))).sum(level='province')
df_prov_mh['resilience'] = df_prov_mh['risk_to_assets']/df_prov_mh['risk']

# Also not sure why 'risk' and 'risk_to_assets' are off by a factor of 10:
df_prov_mh['risk'] *= 10
df_prov_mh['risk_to_assets'] *= 10

df_orig = pd.read_csv('~/Desktop/Dropbox/Bank/resilience_model/output/gdpNTL/PHL_results_tax_no.csv',index_col='province')[['risk','risk_to_assets','resilience']]

# path to the blank map 
svg_file_path = '../map_files/'+myCountry+'/BlankSimpleMap.svg'
inp_res = 800

make_map_from_svg(
        df_prov_mh.risk_to_assets, #data 
        svg_file_path,                  #path to blank map
        outname='asset_risk_',  #base name for output  (will create img/map_of_asset_risk.png, img/legend_of_asset_risk.png, etc.)
        color_maper=plt.cm.get_cmap('Blues'), #color scheme (from matplotlib. Chose them from http://colorbrewer2.org/)
        label='Annual asset losses (% of GDP)',
        new_title='Map of asset risk in the Philippines',  #title for the colored SVG
        do_qualitative=False,
        res=inp_res)

make_map_from_svg(
        df_prov_mh.resilience, 
        svg_file_path,
        outname='se_resilience_',
        color_maper=plt.cm.get_cmap('RdYlGn'), 
        label='Socio-economic capacity (%)',
        new_title='Map of socio-economic resilience in the Philippines',
        do_qualitative=False,
        res=inp_res)

make_map_from_svg(
        df_prov_mh.risk, 
        svg_file_path,
        outname='welfare_risk_',
        color_maper=plt.cm.get_cmap('Purples'), 
        label='Annual welfare losses (% of GDP)',
        new_title='Map of welfare risk in the Philippines',
        do_qualitative=False,
        res=inp_res)
