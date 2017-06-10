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

font = {'family' : 'sans serif',
    'size'   : 20}
plt.rc('font', **font)
mpl.rcParams['xtick.labelsize'] = 16

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
res_unif_poor = pd.read_csv(output+'results_tax_unif_poor_.csv', index_col=['province','hazard','rp'])

df = pd.read_csv(output+'results_tax_no_.csv', index_col=['province','hazard','rp'])

def format_delta_p(delta_p):
    delta_p_int = int(delta_p)
    delta_p = int(delta_p)

    if delta_p_int >= 1E6:
        delta_p = str(delta_p)[:-6]+','+str(delta_p)[-6:]
    if delta_p_int >= 1E3:         
        delta_p = str(delta_p)[:-3]+','+str(delta_p)[-3:]
    return(str(delta_p))
        
iah = pd.read_csv(output+'iah_tax_no_.csv', index_col=['province','hazard','rp'])
iah_pds = pd.read_csv(output+'iah_tax_unif_poor_.csv', index_col=['province','hazard','rp'])

#cats = pd.read_csv(output+'cats_tax_no_.csv', index_col=['province','hazard','rp'])

# Transform dw:
wprime = df.wprime.mean()
print(wprime)

iah['dw'] = iah['dw']/wprime
iah['pds_dw']  = iah_pds['dw']/wprime

iah['pds_nrh'] = iah_pds['help_fee']-iah_pds['help_received'] # Net received help
iah['pds_help_fee'] = iah_pds['help_fee']
iah['pds_help_received'] = iah_pds['help_received']

iah['hhwgt'] = iah['hhwgt'].fillna(0)
iah['weight'] = iah['weight'].fillna(0)

# Convert all these hh variables to per cap
iah['c']  = iah[['c','hhwgt']].prod(axis=1)/iah['weight']
iah['k']  = iah[['k','hhwgt']].prod(axis=1)/iah['weight']
iah['dk'] = iah[['dk','hhwgt']].prod(axis=1)/iah['weight']
iah['dc'] = iah[['dc','hhwgt']].prod(axis=1)/iah['weight']
iah['dc_npv_pre'] = iah[['dc_npv_pre','hhwgt']].prod(axis=1)/iah['weight']

iah['dw'] = iah[['dw','hhwgt']].prod(axis=1)/iah['weight']
iah['pds_dw'] = iah[['pds_dw','hhwgt']].prod(axis=1)/iah['weight']

iah['pds_nrh'] = iah[['pds_nrh','hhwgt']].prod(axis=1)/iah['weight']
iah['pds_help_fee'] = iah[['pds_help_fee','hhwgt']].prod(axis=1)/iah['weight']
iah['pds_help_received'] = iah[['pds_help_received','hhwgt']].prod(axis=1)/iah['weight']

cf_ppp = 17.889
avg_hhsize = iah['weight'].sum(level=event_level).mean(skipna=True)/iah['hhwgt'].sum(level=event_level).mean(skipna=True) # weight is head count

q_labels = ['Poorest quintile','Q2','Q3','Q4','Wealthiest quintile']
q_colors = [sns_pal[0],sns_pal[1],sns_pal[2],sns_pal[3],sns_pal[5]]

# Look at single event:
myHaz = [['Mountain Province','Davao Occidental','Negros Occidental','Abra','Aurora','Bulacan','Pangasinan','Sulu','Davao','Palawan','Eastern Samar','Cebu','Manila'],['flood','wind'],[25,100,500,1000]]

#pov_line = (9064*12.)*(avg_hhsize/5)
#pov_line = 1.90*365*cf_ppp
pov_line = 6329.*(12./5.)

iah = iah.reset_index()
for myDis in ['flood','earthquake','surge','wind']:

    cut_rps = iah.loc[(iah.hazard == myDis)].set_index(['province','hazard','rp']).fillna(0)
    if (cut_rps['weight'].sum() == 0 or cut_rps.shape[0] == 0): continue

    cut_rps['c_initial'] = cut_rps['c']#cut_rps['k']*df['avg_prod_k'].mean()
    cut_rps['delta_c']   = cut_rps['dk']*(df['avg_prod_k'].mean()+1/df['T_rebuild_K'].mean())
    cut_rps['c_final']   = cut_rps['c_initial'] - cut_rps['delta_c']
    
    cut_rps['c_initial'] = cut_rps['c_initial'].clip(upper=100000)
    cut_rps['c_final']   = cut_rps['c_final'].clip(upper=100000)

    cut_rps['pre_dis_n_pov'] = 0
    cut_rps.loc[(cut_rps.c_initial <= pov_line), 'pre_dis_n_pov'] = cut_rps.loc[(cut_rps.c_initial <= pov_line), 'weight']
    print('Pop below pov line before disaster:',cut_rps['pre_dis_n_pov'].sum(level=['hazard','rp']).mean())

    cut_rps['disaster_n_pov'] = 0
    cut_rps.loc[(cut_rps.c_final <= pov_line) & (cut_rps.c_initial > pov_line), 'disaster_n_pov'] = cut_rps.loc[(cut_rps.c_final <= pov_line) & (cut_rps.c_initial > pov_line), 'weight']

    n_pov = pd.DataFrame(cut_rps['disaster_n_pov'].sum(level=['province','rp']).reset_index(),columns=['province','rp','disaster_n_pov']).set_index(['province','rp'])
    n_pov['disaster_n_pov_pct'] = (n_pov['disaster_n_pov']/cut_rps.weight.sum(level=['province','rp']).reset_index().set_index(['province','rp']).T).T
    
    n_pov.disaster_n_pov/=100.
    n_pov = n_pov.reset_index().set_index(['province','rp'])

    n_pov = sum_with_rp(n_pov[['disaster_n_pov','disaster_n_pov_pct']],['disaster_n_pov','disaster_n_pov_pct'],sum_provinces=False)

    make_map_from_svg(
        n_pov.disaster_n_pov, 
        '../map_files/'+myCountry+'/BlankSimpleMap.svg',
        outname='new_poverty_incidence_'+myDis+'_allRPs',
        color_maper=plt.cm.get_cmap('RdYlGn_r'), 
        label='Number of Filipinos pushed into poverty each year by '+myDis+'s',
        new_title='Number of Filipinos pushed into poverty each year by '+myDis+'s',
        do_qualitative=False,
        res=800)
    
    make_map_from_svg(
        n_pov.disaster_n_pov_pct, 
        '../map_files/'+myCountry+'/BlankSimpleMap.svg',
        outname='new_poverty_incidence_pct_'+myDis+'_allRPs',
        color_maper=plt.cm.get_cmap('RdYlGn_r'), 
        label='Filipinos pushed into poverty each year by '+myDis+'s [%]',
        new_title='Filipinos pushed into poverty by '+myDis+'s [%]',
        do_qualitative=False,
        res=800)
    
    for myRP in [1,10,25,50,100,200,250,500,1000]:
        
        cutA = iah.loc[(iah.hazard == myDis) & (iah.rp == myRP)].set_index(['province','hazard','rp']).fillna(0)
        #cutA = iah.loc[(iah.hazard == myDis) & (iah.rp == myRP) & (iah.helped_cat == 'helped')].set_index(['province','hazard','rp']).fillna(0)
        if (cutA['weight'].sum() == 0 or cutA.shape[0] == 0): continue

        # look at instantaneous dk
        ax=plt.gca()

        cutA['c_initial'] = cutA['c']#cutA['k']*df['avg_prod_k'].mean()
        cutA['delta_c']   = cutA['dk']*(df['avg_prod_k'].mean()+1/df['T_rebuild_K'].mean())
        cutA['c_final']   = cutA['c_initial'] - cutA['delta_c']

        cutA['c_initial'] = cutA['c_initial'].clip(upper=100000)
        cutA['c_final']   = cutA['c_final'].clip(upper=100000)

        cutA['disaster_n_pov'] = 0
        cutA.loc[(cutA.c_final <= pov_line) & (cutA.c_initial > pov_line), 'disaster_n_pov'] = cutA.loc[(cutA.c_final <= pov_line) & (cutA.c_initial > pov_line), 'weight']

        disaster_n_pov = pd.DataFrame(cutA['disaster_n_pov'].sum(level=event_level).reset_index(),columns=['province','disaster_n_pov']).set_index('province')
        disaster_n_pov['disaster_n_pov_pct'] = (disaster_n_pov['disaster_n_pov']/cutA.weight.sum(level='province').reset_index().set_index('province').T).T

        disaster_n_pov.disaster_n_pov/=100.
        disaster_n_pov = disaster_n_pov.reset_index().set_index('province')

        ci_heights, ci_bins = np.histogram(cutA['c_initial'],       bins=50, weights=cutA['weight'])
        cf_heights, cf_bins = np.histogram(cutA['c_final'],    bins=ci_bins, weights=cutA['weight'])

        ax.bar(ci_bins[:-1], ci_heights, width=ci_bins[1], label='Initial Consumption', facecolor=q_colors[0],alpha=0.4)
        ax.bar(cf_bins[:-1], cf_heights, width=ci_bins[1], label='Post-disaster Consumption', facecolor=q_colors[1],alpha=0.4)

        delta_p = cutA.loc[(cutA.c_initial > pov_line) & (cutA.c_final <= pov_line),'weight'].sum()

        p_str = format_delta_p(delta_p)
        p_pct = ' ('+str(round((delta_p/cutA['weight'].sum())*100.,2))+'% of population)'

        plt.plot([pov_line,pov_line],[0,1.2*cf_heights[:-2].max()],'k-',lw=1.5,color='black',zorder=100,alpha=0.85)
        ax.annotate('Subsistence line',xy=(1.1*pov_line,1.20*cf_heights[:-2].max()),xycoords='data',ha='left',va='top',fontsize=8,annotation_clip=False)
        ax.annotate(r'$\Delta$N$_s$ = +'+p_str+p_pct,xy=(pov_line*1.1,1.12*cf_heights[:-2].max()),xycoords='data',ha='left',va='top',fontsize=8,annotation_clip=False)

        fig = ax.get_figure()
        plt.xlabel(r'Consumption [Philippine pesos yr$^{-1}$]')
        plt.ylabel('Population')
        plt.legend(loc='best')
        print('poverty_k_'+myDis+'_'+str(myRP)+'.pdf')
        fig.savefig('../output_plots/PH/poverty_k_'+myDis+'_'+str(myRP)+'.png',format='png')#+'.pdf',format='pdf')
        plt.cla()    
        
        ##

        # look at affected people
        ax=plt.gca()

        ci_heights, ci_bins = np.histogram(cutA.loc[(cutA.affected_cat =='a'),'c_initial'],       bins=50, weights=cutA.loc[(cutA.affected_cat =='a'),'weight'])
        cf_heights, cf_bins = np.histogram(cutA.loc[(cutA.affected_cat =='a'),'c_final'],    bins=ci_bins, weights=cutA.loc[(cutA.affected_cat =='a'),'weight'])

        ax.bar(ci_bins[:-1], ci_heights, width=ci_bins[1], label='Initial Consumption', facecolor=q_colors[0],alpha=0.4)
        ax.bar(cf_bins[:-1], cf_heights, width=ci_bins[1], label='Post-disaster Consumption', facecolor=q_colors[1],alpha=0.4)

        print('All people: ',cutA['weight'].sum())
        print('Affected people: ',cutA.loc[(cutA.affected_cat =='a'),'weight'].sum())

        delta_p = cutA.loc[(cutA.affected_cat =='a') & (cutA.c_final <= pov_line),'weight'].sum() 
        delta_p -= cutA.loc[(cutA.affected_cat =='a') & (cutA.c_initial <= pov_line),'weight'].sum()
        p_str = format_delta_p(delta_p)
        p_pct = ' ('+str(round((delta_p/cutA['weight'].sum())*100.,2))+'% of population)'

        plt.plot([pov_line,pov_line],[0,1.2*cf_heights[:-2].max()],'k-',lw=1.5,color='black',zorder=100,alpha=0.85)
        ax.annotate('Subsistence line',xy=(1.1*pov_line,1.20*cf_heights[:-2].max()),xycoords='data',ha='left',va='top',fontsize=8,annotation_clip=False)
        ax.annotate(r'$\Delta$N$_p$ = '+p_str+p_pct,xy=(1.1*pov_line,1.12*cf_heights[:-2].max()),xycoords='data',ha='left',va='top',fontsize=8,annotation_clip=False)

        fig = ax.get_figure()
        plt.xlabel(r'Consumption [Philippine pesos yr$^{-1}$]')
        plt.ylabel('Population')
        #plt.ylim(0,400000)
        plt.legend(loc='best')
        print('poverty_k_aff_'+myDis+'_'+str(myRP)+'.pdf')
        fig.savefig('../output_plots/PH/poverty_k_aff_'+myDis+'_'+str(myRP)+'.png',format='png')#+'.pdf',format='pdf')
        plt.cla()

        make_map_from_svg(
            disaster_n_pov.disaster_n_pov, 
            '../map_files/'+myCountry+'/BlankSimpleMap.svg',
            outname='new_poverty_incidence_'+myDis+'_'+str(myRP),
            color_maper=plt.cm.get_cmap('RdYlGn_r'), 
            label='Number of Filipinos pushed into poverty by '+myDis+' (RP = '+str(myRP)+')',
            new_title='Number of Filipinos pushed into poverty by '+myDis+' (RP = '+str(myRP)+')',
            do_qualitative=False,
            res=800)
        
        make_map_from_svg(
            disaster_n_pov.disaster_n_pov_pct, 
            '../map_files/'+myCountry+'/BlankSimpleMap.svg',
            outname='new_poverty_incidence_pct_'+myDis+'_'+str(myRP),
            color_maper=plt.cm.get_cmap('RdYlGn_r'), 
            label='Filipinos pushed into poverty by '+myDis+' (RP = '+str(myRP)+') [%]',
            new_title='Filipinos pushed into poverty by '+myDis+' (RP = '+str(myRP)+') [%]',
            do_qualitative=False,
            res=800)
    
for myProv in myHaz[0]:
    for myDis in myHaz[1]:
        for myRP in myHaz[2]:
    
            print('len_t=',iah.shape[0])
            cut = iah.loc[(((iah.affected_cat == 'a') & (iah.helped_cat == 'helped')) | ((iah.affected_cat == 'na') & (iah.helped_cat == 'not_helped')))  & (iah.province == myProv) & (iah.hazard == myDis) & (iah.rp == myRP)].set_index(['province','hazard','rp'])

            if cut.shape[0] == 0: continue
        
            cut.to_csv('~/Desktop/my_post_file.csv',encoding='utf-8', header=True)

            # look at quintiles
            q1 = cut.loc[cut.quintile == 1].reset_index()
            q2 = cut.loc[cut.quintile == 2].reset_index()
            q3 = cut.loc[cut.quintile == 3].reset_index()
            q4 = cut.loc[cut.quintile == 4].reset_index()
            q5 = cut.loc[cut.quintile == 5].reset_index()
            
            k_mean = get_weighted_mean(q1,q2,q3,q4,q5,'k')
            dk_mean = get_weighted_mean(q1,q2,q3,q4,q5,'dk')
            dc_mean = get_weighted_mean(q1,q2,q3,q4,q5,'dc_npv_pre')
            dw_mean = get_weighted_mean(q1,q2,q3,q4,q5,'dw')
            nrh_mean = get_weighted_mean(q1,q2,q3,q4,q5,'pds_nrh')
            dw_pds_mean = get_weighted_mean(q1,q2,q3,q4,q5,'pds_dw')
            pds_help_fee_mean = get_weighted_mean(q1,q2,q3,q4,q5,'pds_help_fee')
            pds_help_rec_mean = get_weighted_mean(q1,q2,q3,q4,q5,'pds_help_received')

            print('k/100 (avg) = ',0.01*np.array(k_mean))
            print('dk (avg) = ',dk_mean)
            print('dc (avg) = ',dc_mean)
            print('dw (pc avg) = ',dw_mean)
            print('pds_help_fee_mean (avg) = ',pds_help_fee_mean)
            print('pds_help_rec_mean (avg) = ',pds_help_rec_mean)

            # histograms
            df_wgt = pd.DataFrame({'q1_w': q1['weight'],'q2_w': q2['weight'],'q3_w': q3['weight'],'q4_w': q4['weight'],'q5_w': q5['weight']}, 
                                  columns=['q1_w', 'q2_w', 'q3_w', 'q4_w', 'q5_w']).fillna(0)

            df_wgt.to_csv('~/Desktop/weights.csv')

            for istr in ['dk','dc','dw']:

                continue

                df_tmp = pd.DataFrame({'q1': q1[istr],'q2': q2[istr],'q3': q3[istr],'q4': q4[istr],'q5': q5[istr]},columns=['q1', 'q2', 'q3', 'q4', 'q5']).fillna(0)

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
                plt.legend(loc='best')
                
                fig = ax.get_figure()
                print('Saving: hists/'+istr+'_'+myProv+'_'+myDis+'_'+str(myRP)+'.pdf')
                fig.savefig('../output_plots/PH/'+istr+'_'+myProv.replace(' ','_')+'_'+myDis+'_'+str(myRP)+'.png',format='png')#+'.pdf',format='pdf')
                plt.cla()

            # Means
            ax1 = plt.subplot(111)
            for ij in range(0,5):
                #ax1.bar([6*ii+ij for ii in range(1,5)],[0.01*np.array(k_mean[ij]),dk_mean[ij],dc_mean[ij],dw_mean[ij]],color=q_colors[ij],alpha=0.7,label=q_labels[ij])
                ax1.bar([6*ii+ij for ii in range(1,7)],[0.01*np.array(k_mean[ij]),dk_mean[ij],dc_mean[ij],dw_mean[ij],nrh_mean[ij],dw_pds_mean[ij]],color=q_colors[ij],alpha=0.7,label=q_labels[ij])
            ax1.xaxis.set_ticks([])
            plt.ylabel('Mean PHP ('+myProv+', '+myDis+', rp='+str(myRP)+' yr)')
            ax1.annotate('1% of assets',              xy=( 6,ax.get_ylim()[0]),xycoords='data',ha='left',va='top',fontsize=8,annotation_clip=False)
            ax1.annotate('Asset loss',                xy=(12,ax.get_ylim()[0]),xycoords='data',ha='left',va='top',fontsize=8,annotation_clip=False)
            ax1.annotate('Consumption\nloss',         xy=(18,-500),xycoords='data',ha='left',va='top',fontsize=8,annotation_clip=False)
            ax1.annotate('Welfare loss',              xy=(24,-500),xycoords='data',ha='left',va='top',fontsize=8,annotation_clip=False)
            #ax1.annotate('PDS Help\nFee',             xy=(30,-500),xycoords='data',ha='left',va='top',fontsize=8,annotation_clip=False)
            #ax1.annotate('PDS Help\nRec',             xy=(36,-500),xycoords='data',ha='left',va='top',fontsize=8,annotation_clip=False)
            ax1.annotate('Net cost \nof help',        xy=(30,-500),xycoords='data',ha='left',va='top',fontsize=8,annotation_clip=False)
            ax1.annotate('Welfare loss\npost-support',xy=(36,-500),xycoords='data',ha='left',va='top',fontsize=8,annotation_clip=False)
            ax1.legend(loc='best')

            print('Saving: histo_'+myProv+'_'+myDis+'_'+str(myRP)+'.pdf\n')
            plt.savefig('../output_plots/PH/means_'+myProv.replace(' ','_')+'_'+myDis+'_'+str(myRP)+'.png',format='png')#+'.pdf',bbox_inches='tight',format='pdf')
            plt.cla()
