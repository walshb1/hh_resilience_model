import glob, os
import pandas as pd
import matplotlib.pyplot as plt
from libraries.lib_country_dir import get_currency

import seaborn as sns
sns_pal = sns.color_palette('tab20b', n_colors=20)
greys_pal = sns.color_palette('Greys', n_colors=9)

pol_dict = {'no_ew100':'Universal early warning',
            'no_exp095':'Reduce exposure of\nbottom 40% by 5%',
            'no_exr095':'Reduce exposure of\ntop 40% by 5%',
            'no_increase_social':'Increase pre-disaster social\ntransfers to bottom 40%, by 25%',
            'no_pcinc_p_110':'Increase income of the poor 10%',
            'no_vul070p':'Reduce vulnerability\nof hh in poverty by 30%',
            'social_scaleup':'2-month top-up of all social\ntransfers to affected hh',
            'unif_poor':'Uniform payout\nto affected',
            'unif_poor_only':'Uniform payout to affected\nin bottom quintile',
            'unif_poor_q12':'Uniform payout to affected\nin bottom 40%',
            'no_social_scaleup':'Increase social transfers\nto bottom 40% by 25%',
            'no_protection_5yr':'Eliminate losses to\n5-year events',
            'no_protection_10yr':'Eliminate losses to\n10-year events',
            'no_protection_20yr':'Eliminate losses to\n20-year events'}


def select_event(country,df,event):
    _loc, _haz, _rp = event

    if country == 'RO': event_level = ['Region','hazard','rp']
    else: assert(False)

    if _rp == 'aal': return df.loc[(df[event_level[0]]==_loc)&(df[event_level[1]]==_haz)].copy()
    else: return df.loc[(df[event_level[0]]==_loc)&(df[event_level[1]]==_haz)&(df[event_level[2]]==_rp)].copy()


def plot_impact_for_single_event(country,event):
    results_dir = '../output_country/{}/'.format(country)

    abs_results = {}

    # get nominal (baseline) AAL:
    df_base = pd.read_csv(results_dir+'my_summary_no.csv'.format(country))
    df_base = select_event(country,df_base,event)


    # find all other files (including baseline here)
    for file in glob.glob(results_dir+"my_summary_*.csv"):
        if event[2] == 'aal' and 'aal' not in file: continue
        elif event[2] != 'aal' and 'aal' in file: continue
        if event[1] == 'EQ' and 'ew' in file: continue

        #if 'unif_poor' in file: continue

        pol_str = file.replace(results_dir+'my_summary_','').replace('aal_','').replace('.csv','')
        #
        _df = pd.read_csv(file)
        _df = select_event(country,_df,event)

        abs_results[pol_str] = (round(_df['dk_tot'].sum(),3)*get_currency(country)[2],
                                round(_df['dw_tot'].sum(),3)*get_currency(country)[2],
                                round(1E2*(_df['dk_tot'].sum()/_df['dw_tot'].sum()),1))
        
    results = pd.DataFrame(abs_results).T
    results.columns = ['dk','dw','resilience']
    
    baseline = results.T['no'].squeeze()
    results.drop('no',axis=0,inplace=True)

    ax = plt.gca()
    
    results['dk'] -= baseline['dk']
    results['dw'] -= baseline['dw']

    results = results.loc[(abs(results.dk)>1E-1)|(abs(results.dw)>1E-1)].copy()
    results = results.reset_index().sort_values('dw',ascending=False)# values are negative

    format_dk = int(round(baseline['dk'],0))
    if format_dk >= 1E3: format_dk = '{},{}'.format(str(format_dk)[:-3],str(format_dk)[-3:])

    format_dw = int(round(baseline['dw'],0))
    if format_dw >= 1E3: format_dw = '{},{}'.format(str(format_dw)[:-3],str(format_dw)[-3:])

    ax.bar([3*ii for ii in range(len(results.index))],-1*results['dk'],color=sns_pal[4],alpha=0.6,width=1,
           label='assets (base = US${} million{})'.format(format_dk,('/year' if event[2] == 'aal' else '')))
    ax.bar([3*ii+1 for ii in range(len(results.index))],-1*results['dw'],color=sns_pal[0],alpha=0.6,width=1,
           label='well-being (base = US${} million{})'.format(format_dw,('/year' if event[2] == 'aal' else '')))
           
    ax.legend(loc='upper left',bbox_to_anchor=(0.05,0.95))
    #ax.scatter((0.6),(0.5), s=81, c="limegreen", transform=ax.transAxes)
    #plt.legend()

    if event[2] == 'aal': _str = 'Expected losses (AAL) to {} in {}'.format(event[1],event[0]) 
    else:  _str = 'Expected losses to {}-year {} in {}'.format(event[2],event[1],event[0])
    plt.annotate(_str,xy=(0.05,0.97),xycoords='axes fraction',va='bottom',style='italic')

    plt.yticks(size=8)
    plt.ylabel('Avoided losses [US$, millions{}]'.format('/year' if event[2]=='aal' else ''))

    plt.xticks([3*ii+1.25 for ii in range(len(results.index))],size=8)
    ax.tick_params(axis='x', which='major', pad=10,rotation=90)
    ax.set_xticklabels([pol_dict[_] for _ in results['index']],size=8,color=greys_pal[8])
    ax.tick_params(axis=u'x', which=u'both',length=0)

    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_horizontalalignment('right')

    #ax = results.plot(results.index*3,'dk',kind='bar',color=sns_pal[1],zorder=99)
    #results.plot(results.index*3+2,'dw',kind='bar',color=sns_pal[5],ax=ax,zorder=90)

    plt.grid(False)
    sns.despine()
    plt.gcf().savefig('../output_plots/{}/policy_scorecard_{}_{}_{}.pdf'.format(country,event[0],event[1],event[2]),format='pdf',bbox_inches='tight')
    plt.close('all')
