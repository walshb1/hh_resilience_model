#This script processes data outputs for the resilience indicator multihazard model for the Philippines. Developed by Brian Walsh.
from IPython import get_ipython
get_ipython().magic('reset -f')
get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')

#Import packages for data analysis
from replace_with_warning import *
from lib_gather_data import *
from maps_lib import *

from scipy.stats import norm
import matplotlib.mlab as mlab

from pandas import isnull
import pandas as pd
import numpy as np
import os, time
import sys

from lib_country_dir import *
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

economy = get_economic_unit(myCountry)
event_level = [economy, 'hazard', 'rp']

# Load output files
res_base = pd.read_csv(output+'results_tax_no_.csv', index_col=[economy,'hazard','rp'])
res_unif_poor = pd.read_csv(output+'results_tax_unif_poor_.csv', index_col=[economy,'hazard','rp'])
df = pd.read_csv(output+'results_tax_no_.csv', index_col=[economy,'hazard','rp'])

iah = pd.read_csv(output+'iah_tax_no_.csv', index_col=[economy,'hazard','rp'])
iah_pds = pd.read_csv(output+'iah_tax_unif_poor_.csv', index_col=[economy,'hazard','rp'])

def format_delta_p(delta_p):
    delta_p_int = int(delta_p)
    delta_p = int(delta_p)

    if delta_p_int >= 1E6:
        delta_p = str(delta_p)[:-6]+','+str(delta_p)[-6:]
    if delta_p_int >= 1E3:         
        delta_p = str(delta_p)[:-3]+','+str(delta_p)[-3:]
    return(str(delta_p))
        
#cats = pd.read_csv(output+'cats_tax_no_.csv', index_col=[economy,'hazard','rp'])

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
myHaz = [['Manila','Mountain Province','Bukidnon','Negros Oriental','Bulacan','Northern Samar','Cebu'],['flood','wind'],[1,10,25,30,50,100,250,500,1000]]

#pov_line = (9064*12.)*(avg_hhsize/5)
#pov_line = 1.90*365*cf_ppp
#pov_line = 6329.*(12./5.)
sub_line = 14832.0962*(22302.6775/21240.2924)
pov_line = 22302.6775#21240.2924

iah = iah.reset_index()
for myDis in ['flood','earthquake','surge','wind']:

    cut_rps = iah.loc[(iah.hazard == myDis)].set_index([economy,'hazard','rp']).fillna(0)
    if (cut_rps['weight'].sum() == 0 or cut_rps.shape[0] == 0): continue

    cut_rps['c_initial'] = cut_rps['c']#cut_rps['k']*df['avg_prod_k'].mean()
    cut_rps['delta_c']   = cut_rps['dk']*(df['avg_prod_k'].mean()+1/df['T_rebuild_K'].mean())
    cut_rps['c_final']   = cut_rps['c_initial'] - cut_rps['delta_c']
    
    cut_rps['c_initial'] = cut_rps['c_initial'].clip(upper=100000)
    cut_rps['c_final']   = cut_rps['c_final'].clip(upper=100000)

    cut_rps['pre_dis_n_pov'] = 0
    cut_rps['pre_dis_n_sub'] = 0
    cut_rps.loc[(cut_rps.c_initial <= pov_line), 'pre_dis_n_pov'] = cut_rps.loc[(cut_rps.c_initial <= pov_line), 'weight']
    cut_rps.loc[(cut_rps.c_initial <= sub_line), 'pre_dis_n_sub'] = cut_rps.loc[(cut_rps.c_initial <= sub_line), 'weight']
    print('\n\nPop below pov line before disaster:',cut_rps['pre_dis_n_pov'].sum(level=['hazard','rp']).mean(),'\n')
    print('\n\nPop below sub line before disaster:',cut_rps['pre_dis_n_sub'].sum(level=['hazard','rp']).mean(),'\n')

    print('poor, below pov',cut_rps.loc[(cut_rps.poorhh == 1) & (cut_rps.c_initial <= pov_line), 'weight'].sum(level=['hazard','rp']).mean())
    print('poor, above pov',cut_rps.loc[(cut_rps.poorhh == 1) & (cut_rps.c_initial > pov_line), 'weight'].sum(level=['hazard','rp']).mean())
    print('rich, below pov',cut_rps.loc[(cut_rps.poorhh == 0) & (cut_rps.c_initial <= pov_line), 'weight'].sum(level=['hazard','rp']).mean())
    print('rich, above pov',cut_rps.loc[(cut_rps.poorhh == 0) & (cut_rps.c_initial > pov_line), 'weight'].sum(level=['hazard','rp']).mean())

    print('poor, below sub',cut_rps.loc[(cut_rps.poorhh == 1) & (cut_rps.c_initial <= sub_line), 'weight'].sum(level=['hazard','rp']).mean())
    print('poor, above sub',cut_rps.loc[(cut_rps.poorhh == 1) & (cut_rps.c_initial > sub_line), 'weight'].sum(level=['hazard','rp']).mean())
    print('rich, below sub',cut_rps.loc[(cut_rps.poorhh == 0) & (cut_rps.c_initial <= sub_line), 'weight'].sum(level=['hazard','rp']).mean())
    print('rich, above sub',cut_rps.loc[(cut_rps.poorhh == 0) & (cut_rps.c_initial > sub_line), 'weight'].sum(level=['hazard','rp']).mean())

    cut_rps['disaster_n_pov'] = 0
    cut_rps['disaster_n_sub'] = 0
    cut_rps.loc[(cut_rps.c_final <= pov_line) & (cut_rps.c_initial > pov_line), 'disaster_n_pov'] = cut_rps.loc[(cut_rps.c_final <= pov_line) & (cut_rps.c_initial > pov_line), 'weight']
    cut_rps.loc[(cut_rps.c_final <= sub_line) & (cut_rps.c_initial > sub_line), 'disaster_n_sub'] = cut_rps.loc[(cut_rps.c_final <= sub_line) & (cut_rps.c_initial > sub_line), 'weight']

    n_pov = pd.DataFrame(cut_rps[['disaster_n_pov','disaster_n_sub']].sum(level=[economy,'rp']).reset_index(),
                         columns=[economy,'rp','disaster_n_pov','disaster_n_sub']).set_index([economy,'rp'])
    n_pov['disaster_n_pov_pct'] = (n_pov['disaster_n_pov']/cut_rps.weight.sum(level=[economy,'rp']).reset_index().set_index([economy,'rp']).T).T
    n_pov['disaster_n_sub_pct'] = (n_pov['disaster_n_sub']/cut_rps.weight.sum(level=[economy,'rp']).reset_index().set_index([economy,'rp']).T).T
    
    n_pov.disaster_n_pov/=100.
    n_pov.disaster_n_sub/=100.
    n_pov = n_pov.reset_index().set_index([economy,'rp'])

    n_pov = sum_with_rp(n_pov[['disaster_n_pov','disaster_n_pov_pct','disaster_n_sub','disaster_n_sub_pct']],
                        ['disaster_n_pov','disaster_n_pov_pct','disaster_n_sub','disaster_n_sub_pct'],sum_provinces=False)


    my_n_pov = n_pov.copy().drop(['Batanes'],axis=0)

    make_map_from_svg(
        my_n_pov.disaster_n_pov, 
        '../map_files/'+myCountry+'/BlankSimpleMap.svg',
        outname='new_poverty_incidence_'+myDis+'_allRPs',
        color_maper=plt.cm.get_cmap('RdYlGn_r'), 
        label='Number of Filipinos pushed into poverty each year by '+myDis+'s',
        new_title='Number of Filipinos pushed into poverty each year by '+myDis+'s',
        do_qualitative=False,
        res=800)
    
    make_map_from_svg(
        my_n_pov.disaster_n_pov_pct, 
        '../map_files/'+myCountry+'/BlankSimpleMap.svg',
        outname='new_poverty_incidence_pct_'+myDis+'_allRPs',
        color_maper=plt.cm.get_cmap('RdYlGn_r'), 
        label='Filipinos pushed into poverty each year by '+myDis+'s [%]',
        new_title='Filipinos pushed into poverty by '+myDis+'s [%]',
        do_qualitative=False,
        res=800)

    make_map_from_svg(
        my_n_pov.disaster_n_sub, 
        '../map_files/'+myCountry+'/BlankSimpleMap.svg',
        outname='new_subsistence_incidence_'+myDis+'_allRPs',
        color_maper=plt.cm.get_cmap('RdYlGn_r'), 
        label='Number of Filipinos pushed into subsistence each year by '+myDis+'s',
        new_title='Number of Filipinos pushed into subsistence each year by '+myDis+'s',
        do_qualitative=False,
        res=800)
    
    make_map_from_svg(
        my_n_pov.disaster_n_sub_pct, 
        '../map_files/'+myCountry+'/BlankSimpleMap.svg',
        outname='new_subsistence_incidence_pct_'+myDis+'_allRPs',
        color_maper=plt.cm.get_cmap('RdYlGn_r'), 
        label='Filipinos pushed into subsistence each year by '+myDis+'s [%]',
        new_title='Filipinos pushed into subsistence by '+myDis+'s [%]',
        do_qualitative=False,
        res=800)
    
    for myRP in [1,10,25,50,100,200,250,500,1000]:
        
        cutA = iah.loc[(iah.hazard == myDis) & (iah.rp == myRP)].set_index([economy,'hazard','rp']).fillna(0)
        #cutA = iah.loc[(iah.hazard == myDis) & (iah.rp == myRP) & (iah.helped_cat == 'helped')].set_index([economy,'hazard','rp']).fillna(0)
        if (cutA['weight'].sum() == 0 or cutA.shape[0] == 0): continue

        # look at instantaneous dk
        ax=plt.gca()

        cutA['c_initial'] = cutA['c']#cutA['k']*df['avg_prod_k'].mean()
        cutA['delta_c']   = cutA['dk']*(df['avg_prod_k'].mean()+1/df['T_rebuild_K'].mean())
        cutA['c_final']   = cutA['c_initial'] - cutA['delta_c']

        cutA['c_initial'] = cutA['c_initial'].clip(upper=100000)
        cutA['c_final']   = cutA['c_final'].clip(upper=100000)

        cutA['disaster_n_pov'] = 0
        cutA['disaster_n_sub'] = 0
        cutA.loc[(cutA.c_final <= pov_line) & (cutA.c_initial > pov_line), 'disaster_n_pov'] = cutA.loc[(cutA.c_final <= pov_line) & (cutA.c_initial > pov_line), 'weight']
        cutA.loc[(cutA.c_final <= sub_line) & (cutA.c_initial > sub_line), 'disaster_n_sub'] = cutA.loc[(cutA.c_final <= sub_line) & (cutA.c_initial > sub_line), 'weight']

        disaster_n_pov = pd.DataFrame(cutA[['disaster_n_pov','disaster_n_sub']].sum(level=event_level).reset_index(),columns=[economy,'disaster_n_pov','disaster_n_sub']).set_index(economy)
        disaster_n_pov['disaster_n_pov_pct'] = (disaster_n_pov['disaster_n_pov']/cutA.weight.sum(level=economy).reset_index().set_index(economy).T).T
        disaster_n_pov['disaster_n_sub_pct'] = (disaster_n_pov['disaster_n_sub']/cutA.weight.sum(level=economy).reset_index().set_index(economy).T).T

        disaster_n_pov.disaster_n_pov/=100.
        disaster_n_pov.disaster_n_sub/=100.
        disaster_n_pov = disaster_n_pov.reset_index().set_index(economy)

        ci_heights, ci_bins = np.histogram(cutA['c_initial'],       bins=50, weights=cutA['weight'])
        cf_heights, cf_bins = np.histogram(cutA['c_final'],    bins=ci_bins, weights=cutA['weight'])

        ci_heights /= 1.E6
        cf_heights /= 1.E6

        ax.bar(ci_bins[:-1], ci_heights, width=ci_bins[1], label='Initial Consumption', facecolor=q_colors[0],alpha=0.4)
        ax.bar(cf_bins[:-1], cf_heights, width=ci_bins[1], label='Post-disaster Consumption', facecolor=q_colors[1],alpha=0.4)

        ##### Experiment
        #print('rich, below pov',cutA.loc[(cutA.poorhh == 0) & (cutA.c_initial <= pov_line), 'weight'].sum(level=['hazard','rp']).mean())
        #crb_heights, crb_bins = np.histogram(cutA.loc[(cutA.poorhh == 0) & (cutA.c_initial <= pov_line),'c_initial'].fillna(0),    
        #                                     bins=ci_bins, weights=cutA.loc[(cutA.poorhh == 0) & (cutA.c_initial <= pov_line),'weight'].fillna(0))
        
        
        #print('rich, above pov',cutA.loc[(cutA.poorhh == 0) & (cutA.c_initial > pov_line), 'weight'].sum(level=['hazard','rp']).mean())
        #cra_heights, cra_bins = np.histogram(cutA.loc[(cutA.poorhh == 0) & (cutA.c_initial > pov_line),'c_initial'].fillna(0),    
        #                                     bins=ci_bins, weights=cutA.loc[(cutA.poorhh == 0) & (cutA.c_initial > pov_line),'weight'].fillna(0))
        
        #cra_heights /= 1.E6
        #crb_heights /= 1.E6

        #ax.bar(crb_bins[:-1], crb_heights, width=ci_bins[1], label='Initial Consumption, Rich', facecolor=q_colors[2],alpha=0.8)
        #ax.bar(cra_bins[:-1], cra_heights, width=ci_bins[1], label='Initial Consumption, Rich', facecolor=q_colors[3],alpha=0.8)
        
        ###############

        # Change in poverty incidence
        delta_p = cutA.loc[(cutA.c_initial > pov_line) & (cutA.c_final <= pov_line),'weight'].sum()
        p_str = format_delta_p(delta_p)
        p_pct = ' ('+str(round((delta_p/cutA['weight'].sum())*100.,2))+'% of population)'

        # Change in subsistence incidence
        delta_s = cutA.loc[(cutA.c_initial > sub_line) & (cutA.c_final <= sub_line),'weight'].sum()
        s_str = format_delta_p(delta_s)
        s_pct = ' ('+str(round((delta_s/cutA['weight'].sum())*100.,2))+'% of population)'

        plt.plot([pov_line,pov_line],[0,1.25*cf_heights[:-2].max()],'k-',lw=1.5,color='black',zorder=100,alpha=0.85)
        ax.annotate('Poverty line',xy=(1.1*pov_line,1.25*cf_heights[:-2].max()),xycoords='data',ha='left',va='top',fontsize=9,annotation_clip=False,weight='bold')
        ax.annotate(r'$\Delta$N$_s$ = +'+p_str+p_pct,xy=(pov_line*1.1,1.15*cf_heights[:-2].max()),xycoords='data',ha='left',va='top',fontsize=9,annotation_clip=False)

        plt.plot([sub_line,sub_line],[0,1.6*cf_heights[:-2].max()],'k-',lw=1.5,color='black',zorder=100,alpha=0.85)
        ax.annotate('Subsistence line',xy=(1.1*sub_line,1.60*cf_heights[:-2].max()),xycoords='data',ha='left',va='top',fontsize=9,annotation_clip=False,weight='bold')
        ax.annotate(r'$\Delta$N$_s$ = +'+s_str+s_pct,xy=(sub_line*1.1,1.5*cf_heights[:-2].max()),xycoords='data',ha='left',va='top',fontsize=9,annotation_clip=False)

        fig = ax.get_figure()
        plt.title(str(myRP)+'-Year '+myDis[:1].upper()+myDis[1:]+' Event')
        plt.xlabel(r'Consumption [Philippine pesos yr$^{-1}$]')
        plt.ylabel('Population [Millions]')
        plt.legend(loc='best')
        print('poverty_k_'+myDis+'_'+str(myRP)+'.pdf')
        fig.savefig('../output_plots/PH/poverty_k_'+myDis+'_'+str(myRP)+'.png',format='png')#+'.pdf',format='pdf')
        plt.cla()    
        
        ##

        # Same as above, for affected people
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
        ax.annotate('Poverty line',xy=(1.1*pov_line,1.20*cf_heights[:-2].max()),xycoords='data',ha='left',va='top',fontsize=8,annotation_clip=False)
        ax.annotate(r'$\Delta$N$_s$ = '+p_str+p_pct,xy=(1.1*pov_line,1.12*cf_heights[:-2].max()),xycoords='data',ha='left',va='top',fontsize=8,annotation_clip=False)

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

df_out_sum = pd.DataFrame()
df_out = pd.DataFrame()

rp_all = []
dk_all = []
dw_all = []

dk_q1 = []
dw_q1 = []

for myRP in myHaz[2]:

    # Don't care about province or hazard, but I DO still need to separate out by RP
    # try means for all of the Philippines
    all_q1 = iah.loc[(((iah.affected_cat == 'a') & (iah.helped_cat == 'helped')) | ((iah.affected_cat == 'na') & (iah.helped_cat == 'not_helped'))) 
                     & (iah.quintile == 1) & (iah.rp == myRP)]
    all_q2 = iah.loc[(((iah.affected_cat == 'a') & (iah.helped_cat == 'helped')) | ((iah.affected_cat == 'na') & (iah.helped_cat == 'not_helped'))) 
                     & (iah.quintile == 2) & (iah.rp == myRP)]
    all_q3 = iah.loc[(((iah.affected_cat == 'a') & (iah.helped_cat == 'helped')) | ((iah.affected_cat == 'na') & (iah.helped_cat == 'not_helped'))) 
                     & (iah.quintile == 3) & (iah.rp == myRP)]
    all_q4 = iah.loc[(((iah.affected_cat == 'a') & (iah.helped_cat == 'helped')) | ((iah.affected_cat == 'na') & (iah.helped_cat == 'not_helped'))) 
                     & (iah.quintile == 4) & (iah.rp == myRP)]
    all_q5 = iah.loc[(((iah.affected_cat == 'a') & (iah.helped_cat == 'helped')) | ((iah.affected_cat == 'na') & (iah.helped_cat == 'not_helped'))) 
                     & (iah.quintile == 5) & (iah.rp == myRP)]

    print('RP = ',myRP,'dk =',iah.loc[(((iah.affected_cat == 'a') & (iah.helped_cat == 'helped')) | ((iah.affected_cat == 'na') & (iah.helped_cat == 'not_helped'))) 
                  & (iah.rp == myRP),['dk','weight']].prod(axis=1).sum())
    print('RP = ',myRP,'dw =',iah.loc[(((iah.affected_cat == 'a') & (iah.helped_cat == 'helped')) | ((iah.affected_cat == 'na') & (iah.helped_cat == 'not_helped'))) 
                  & (iah.rp == myRP),['dw','weight']].prod(axis=1).sum())          

    rp_all.append(myRP)
    dk_all.append(iah.loc[(((iah.affected_cat == 'a') & (iah.helped_cat == 'helped')) | ((iah.affected_cat == 'na') & (iah.helped_cat == 'not_helped'))) 
                          & (iah.rp == myRP),['dk','weight']].prod(axis=1).sum())
    dw_all.append(iah.loc[(((iah.affected_cat == 'a') & (iah.helped_cat == 'helped')) | ((iah.affected_cat == 'na') & (iah.helped_cat == 'not_helped'))) 
                          & (iah.rp == myRP),['dw','weight']].prod(axis=1).sum())

    dk_q1.append(iah.loc[(((iah.affected_cat == 'a') & (iah.helped_cat == 'helped')) | ((iah.affected_cat == 'na') & (iah.helped_cat == 'not_helped'))) 
                         & (iah.rp == myRP) & (iah.quintile == 1),['dk','weight']].prod(axis=1).sum())
    dw_q1.append(iah.loc[(((iah.affected_cat == 'a') & (iah.helped_cat == 'helped')) | ((iah.affected_cat == 'na') & (iah.helped_cat == 'not_helped'))) 
                         & (iah.rp == myRP) & (iah.quintile == 1),['dw','weight']].prod(axis=1).sum())

    k_mean = get_weighted_mean(all_q1,all_q2,all_q3,all_q4,all_q5,'k')
    dk_mean = get_weighted_mean(all_q1,all_q2,all_q3,all_q4,all_q5,'dk')
    dc_mean = get_weighted_mean(all_q1,all_q2,all_q3,all_q4,all_q5,'dc_npv_pre')
    dw_mean = get_weighted_mean(all_q1,all_q2,all_q3,all_q4,all_q5,'dw')
    nrh_mean = get_weighted_mean(all_q1,all_q2,all_q3,all_q4,all_q5,'pds_nrh')
    dw_pds_mean = get_weighted_mean(all_q1,all_q2,all_q3,all_q4,all_q5,'pds_dw')
    pds_help_fee_mean = get_weighted_mean(all_q1,all_q2,all_q3,all_q4,all_q5,'pds_help_fee')
    pds_help_rec_mean = get_weighted_mean(all_q1,all_q2,all_q3,all_q4,all_q5,'pds_help_received')
    
    df_this_sum = pd.DataFrame({     'k_q1':     k_mean[0],     'k_q2':     k_mean[1],     'k_q3':     k_mean[2],     'k_q4':     k_mean[3],     'k_q5':     k_mean[4],
                                     'dk_q1':    dk_mean[0],    'dk_q2':    dk_mean[1],    'dk_q3':    dk_mean[2],    'dk_q4':    dk_mean[3],    'dk_q5':    dk_mean[4], 
                                     'dc_q1':    dc_mean[0],    'dc_q2':    dc_mean[1],    'dc_q3':    dc_mean[2],    'dc_q4':    dc_mean[3],    'dc_q5':    dc_mean[4],
                                     'dw_q1':    dw_mean[0],    'dw_q2':    dw_mean[1],    'dw_q3':    dw_mean[2],    'dw_q4':    dw_mean[3],    'dw_q5':    dw_mean[4],
                                     'nrh_q1':   nrh_mean[0],   'nrh_q2':   nrh_mean[1],   'nrh_q3':   nrh_mean[2],   'nrh_q4':   nrh_mean[3],   'nrh_q5':   nrh_mean[4],
                                     'dw_pds_q1':dw_pds_mean[0],'dw_pds_q2':dw_pds_mean[1],'dw_pds_q3':dw_pds_mean[2],'dw_pds_q4':dw_pds_mean[3],'dw_pds_q5':dw_pds_mean[4]},
                               columns=[     'k_q1',     'k_q2',     'k_q3',     'k_q4',     'k_q5',
                                             'dk_q1',    'dk_q2',    'dk_q3',    'dk_q4',    'dk_q5',
                                             'dc_q1',    'dc_q2',    'dc_q3',    'dc_q4',    'dc_q5',
                                             'dw_q1',    'dw_q2',    'dw_q3',    'dw_q4',    'dw_q5',
                                             'nrh_q1',   'nrh_q2',   'nrh_q3',   'nrh_q4',   'nrh_q5',
                                             'dw_pds_q1','dw_pds_q2','dw_pds_q3','dw_pds_q4','dw_pds_q5'],
                               index=[myRP])

    if df_out_sum.empty: df_out_sum = df_this_sum
    else: df_out_sum = df_out_sum.append(df_this_sum)

    print('--> WHOLE COUNTRY (rp =',myRP,')')
    print('k/100 (avg) = ',0.01*np.array(k_mean))
    print('dk (avg) = ',dk_mean)
    print('dc (avg) = ',dc_mean)
    print('dw (pc avg) = ',dw_mean)
    print('pds_help_fee_mean (avg) = ',pds_help_fee_mean)
    print('pds_help_rec_mean (avg) = ',pds_help_rec_mean)
    print('\n')

    for myProv in myHaz[0]:
        for myDis in myHaz[1]:

            cut = iah.loc[(((iah.affected_cat == 'a') & (iah.helped_cat == 'helped')) | ((iah.affected_cat == 'na') & (iah.helped_cat == 'not_helped')))  & (iah.province == myProv) & (iah.hazard == myDis) & (iah.rp == myRP)].set_index([economy,'hazard','rp'])

            if cut.shape[0] == 0: continue
        
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

            df_this = pd.DataFrame({     'k_q1':     k_mean[0],     'k_q2':     k_mean[1],     'k_q3':     k_mean[2],     'k_q4':     k_mean[3],     'k_q5':     k_mean[4],
                                        'dk_q1':    dk_mean[0],    'dk_q2':    dk_mean[1],    'dk_q3':    dk_mean[2],    'dk_q4':    dk_mean[3],    'dk_q5':    dk_mean[4], 
                                        'dc_q1':    dc_mean[0],    'dc_q2':    dc_mean[1],    'dc_q3':    dc_mean[2],    'dc_q4':    dc_mean[3],    'dc_q5':    dc_mean[4],
                                        'dw_q1':    dw_mean[0],    'dw_q2':    dw_mean[1],    'dw_q3':    dw_mean[2],    'dw_q4':    dw_mean[3],    'dw_q5':    dw_mean[4],
                                       'nrh_q1':   nrh_mean[0],   'nrh_q2':   nrh_mean[1],   'nrh_q3':   nrh_mean[2],   'nrh_q4':   nrh_mean[3],   'nrh_q5':   nrh_mean[4],
                                    'dw_pds_q1':dw_pds_mean[0],'dw_pds_q2':dw_pds_mean[1],'dw_pds_q3':dw_pds_mean[2],'dw_pds_q4':dw_pds_mean[3],'dw_pds_q5':dw_pds_mean[4]},
                                      columns=[     'k_q1',     'k_q2',     'k_q3',     'k_q4',     'k_q5',
                                                   'dk_q1',    'dk_q2',    'dk_q3',    'dk_q4',    'dk_q5',
                                                   'dc_q1',    'dc_q2',    'dc_q3',    'dc_q4',    'dc_q5',
                                                   'dw_q1',    'dw_q2',    'dw_q3',    'dw_q4',    'dw_q5',
                                                  'nrh_q1',   'nrh_q2',   'nrh_q3',   'nrh_q4',   'nrh_q5',
                                               'dw_pds_q1','dw_pds_q2','dw_pds_q3','dw_pds_q4','dw_pds_q5'],
                                      index=[[myProv],[myDis],[myRP]])

            if df_out.empty: df_out = df_this
            else: df_out = df_out.append(df_this)

            # histograms
            df_wgt = pd.DataFrame({'q1_w': q1.loc[(q1.affected_cat=='a') & (q1.c <= 100000),'weight'],
                                   'q2_w': q2.loc[(q2.affected_cat=='a') & (q2.c <= 100000),'weight'],
                                   'q3_w': q3.loc[(q3.affected_cat=='a') & (q3.c <= 100000),'weight'],
                                   'q4_w': q4.loc[(q4.affected_cat=='a') & (q4.c <= 100000),'weight'],
                                   'q5_w': q5.loc[(q5.affected_cat=='a') & (q5.c <= 100000),'weight']}, 
                                  columns=['q1_w', 'q2_w', 'q3_w', 'q4_w', 'q5_w']).fillna(0)

            #df_wgt.to_csv('~/Desktop/weights.csv')

            for istr in ['dk','dc','dw']:

                upper_clip = 75000
                if istr == 'dw': upper_clip =  200000

                df_tmp = pd.DataFrame({'q1': q1.loc[(q1.affected_cat=='a') & (q1.c <= 100000),istr],
                                       'q2': q2.loc[(q2.affected_cat=='a') & (q2.c <= 100000),istr],
                                       'q3': q3.loc[(q3.affected_cat=='a') & (q3.c <= 100000),istr],
                                       'q4': q4.loc[(q4.affected_cat=='a') & (q4.c <= 100000),istr],
                                       'q5': q5.loc[(q5.affected_cat=='a') & (q5.c <= 100000),istr]},columns=['q1', 'q2', 'q3', 'q4', 'q5']).fillna(0)

                q1_heights, q1_bins = np.histogram(df_tmp['q1'].clip(upper=upper_clip),weights=df_wgt['q1_w'],bins=15)                
                #q2_heights, q2_bins = np.histogram(df_tmp['q2'].clip(upper=upper_clip),weights=df_wgt['q2_w'],bins=q1_bins)
                q3_heights, q3_bins = np.histogram(df_tmp['q3'].clip(upper=upper_clip),weights=df_wgt['q3_w'],bins=q1_bins)
                #q4_heights, q4_bins = np.histogram(df_tmp['q4'].clip(upper=upper_clip),weights=df_wgt['q4_w'],bins=q1_bins)
                q5_heights, q5_bins = np.histogram(df_tmp['q5'].clip(upper=upper_clip),weights=df_wgt['q5_w'],bins=q1_bins)

                width = (q1_bins[1] - q1_bins[0])*2/7
            
                ax = plt.gca()
                ax.bar(q1_bins[:-1],         q1_heights, width=width, label='q1', facecolor=q_colors[0],alpha=0.3)
                #ax.bar(q2_bins[:-1], q2_heights, width=width, label='q2', facecolor=q_colors[1],alpha=0.3)
                ax.bar(q3_bins[:-1]+1*width, q3_heights, width=width, label='q3', facecolor=q_colors[2],alpha=0.3)
                #ax.bar(q4_bins[:-1], q4_heights, width=width, label='q4', facecolor=q_colors[3],alpha=0.3)
                ax.bar(q5_bins[:-1]+2*width, q5_heights, width=width, label='q5', facecolor=q_colors[4],alpha=0.3)

                mu = np.average(df_tmp['q1'], weights=df_wgt['q1_w'])
                sigma = np.sqrt(np.average((df_tmp['q1']-mu)**2, weights=df_wgt['q1_w']))
                y = df_wgt['q1_w'].sum()*mlab.normpdf(q1_bins, mu, sigma)
                l = plt.plot(q1_bins, y, 'r--', linewidth=2,color=q_colors[0]) 

                mu = np.average(df_tmp['q3'], weights=df_wgt['q3_w'])
                sigma = np.sqrt(np.average((df_tmp['q3']-mu)**2, weights=df_wgt['q3_w']))      
                y = df_wgt['q3_w'].sum()*mlab.normpdf(q3_bins, mu, sigma)
                l = plt.plot(q3_bins, df_wgt['q3_w'].sum()*y, 'r--', linewidth=2,color=q_colors[2]) 

                mu = np.average(df_tmp['q5'], weights=df_wgt['q5_w'])
                sigma = np.sqrt(np.average((df_tmp['q5']-mu)**2, weights=df_wgt['q5_w']))     
                y = df_wgt['q5_w'].sum()*mlab.normpdf(q5_bins, mu, sigma)
                l = plt.plot(q5_bins, df_wgt['q5_w'].sum()*y, 'r--', linewidth=2,color=q_colors[4]) 

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
                ax1.bar([6*ii+ij for ii in range(1,3)],[dk_mean[ij],dw_mean[ij]],color=q_colors[ij],alpha=0.7,label=q_labels[ij])
                #ax1.bar([6*ii+ij for ii in range(1,7)],[0.01*np.array(k_mean[ij]),dk_mean[ij],dc_mean[ij],dw_mean[ij],nrh_mean[ij],dw_pds_mean[ij]],color=q_colors[ij],alpha=0.7,label=q_labels[ij])
                        
            label_y_val = 0.2*np.array(nrh_mean).min()

            ax1.xaxis.set_ticks([])
            plt.title(str(myRP)+'-Year '+myDis[:1].upper()+myDis[1:]+' Event in '+myProv)
            plt.ylabel('Disaster losses [PhP per capita]')
            #ax1.annotate('1% of assets',              xy=( 6,label_y_val),xycoords='data',ha='left',va='top',weight='bold',fontsize=8,annotation_clip=False)
            ax1.annotate('Asset loss',                xy=(6,label_y_val),xycoords='data',ha='left',va='top',weight='bold',fontsize=12,annotation_clip=False)
            #ax1.annotate('Consumption\nloss',         xy=(18,label_y_val),xycoords='data',ha='left',va='top',weight='bold',fontsize=8,annotation_clip=False)
            ax1.annotate('Well-being loss',              xy=(12,label_y_val),xycoords='data',ha='left',va='top',weight='bold',fontsize=12,annotation_clip=False)
            #ax1.annotate('Net cost \nof help',        xy=(30,label_y_val),xycoords='data',ha='left',va='top',weight='bold',fontsize=8,annotation_clip=False)
            #ax1.annotate('Well-being loss\npost-support',xy=(18,label_y_val),xycoords='data',ha='left',va='top',weight='bold',fontsize=12,annotation_clip=False)
            ax1.legend(loc='best')

            plt.xlim(5.5,17.5)

            print('Saving: histo_'+myProv+'_'+myDis+'_'+str(myRP)+'.pdf\n')
            plt.savefig('../output_plots/PH/means_'+myProv.replace(' ','_')+'_'+myDis+'_'+str(myRP)+'.png',format='png')#+'.pdf',bbox_inches='tight',format='pdf')
            plt.cla()

df_out.to_csv('~/Desktop/my_means.csv')
df_out_sum.to_csv('~/Desktop/my_means_ntl.csv')

print(rp_all,'\n',dk_all,'\n',dw_all,'\n',dk_q1,'\n',dw_q1)
