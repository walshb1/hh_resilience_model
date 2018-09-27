import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import os, glob

from libraries.lib_common_plotting_functions import title_legend_labels,q_colors

import seaborn as sns
pairs_pal = sns.color_palette('Paired', n_colors=12)

sns.set_style('whitegrid')
mpl.rcParams['legend.facecolor'] = 'white'

myCountry = 'SL'
out = '../output_plots/'+myCountry+'/'
to_usd = 1./153

def get_sp_costs(myCountry):
    sp_pols, sp_files = [], {}

    dir = '../output_country/'+myCountry+'/sp_costs'
    
    for f in glob.glob(dir+'*'):
        
        sp_pol = f.replace(dir+'_','').replace('.csv','')
        sp_pols.append(sp_pol)

        sp_files[sp_pol] = pd.read_csv(f,index_col=['district','hazard','rp'])

    return sp_pols, sp_files

def get_sort_order(sp_pols,sp_files,crit):
    sort_order = []

    while len(sort_order) != len(sp_pols):
        val = [0,'']
        for _sp in sp_pols:

            if _sp not in sort_order and val[0] <= sp_files[_sp][crit].mean(): 
                val[0] = sp_files[_sp][crit].mean()
                val[1] = _sp

        sort_order.append(val[1])

    return sort_order
            

sp_pols, sp_files = get_sp_costs(myCountry)
pds_effects = pd.read_csv('../output_country/'+myCountry+'/pds_effects.csv').set_index(['district','hazard','rp'])

# Plot costs of each SP when event occurs
sort_order = get_sort_order(sp_pols,sp_files,'avg_natl_cost')
print(sort_order)

_ix = 0
for _sp in sort_order:
    if 'samurdhi' not in _sp: continue

    cost = ' (annual cost = '+str(round(sp_files[_sp]['avg_natl_cost'].mean()*to_usd/1.E6,2))+'M)'
    if _sp == 'no': cost = ''
    
    plt.plot(sp_files[_sp].sum(level='rp').index,sp_files[_sp]['event_cost'].sum(level='rp')*to_usd/1.E6,label=_sp+cost,color=pairs_pal[_ix])
    plt.plot(sp_files[_sp].sum(level='rp').index,pds_effects['dw_DELTA_'+_sp].sum(level='rp')*to_usd,label=_sp+' benefit',ls=':',color=pairs_pal[_ix+1])

    _ix+=2

title_legend_labels(plt,'Sri Lanka','Return Period [years]','SP cost when event occurs [M USD]',[1,1000])
plt.xscale('log')
plt.gca().get_figure().savefig(out+'sp_costs.pdf',format='pdf')



########################################################################
# Samurdhi is a constant payout, so show bar plot without RP dependence
# - cost of disbursement (annual & when event occurs)
# - well-being benefit (annual & when event occurs)
# - cost-benefit ratio

# x-axis is targeting error
# y-axis is in mUSD
pds_effects = pds_effects.reset_index('rp')

_wid = 0.45
rp_range = [5]
affected_place = ['Colombo','Rathnapura','all']
for _place in affected_place:
    for _rp in rp_range:
        plt.close('all')
        plt.figure(figsize=(6,6))

        for _x, _sp in enumerate(['samurdhi_scaleup','samurdhi_scaleup66','samurdhi_scaleup33','samurdhi_scaleup00']):
            sp_files[_sp] = sp_files[_sp].reset_index()

            _criteria = '(rp==@_rp)'
            if _place != 'all': _criteria+='&(district==@_place)'

            _benefit = pds_effects.loc[pds_effects.eval(_criteria),'dw_DELTA_'+_sp].sum()*to_usd
            _cost = sp_files[_sp].loc[sp_files[_sp].eval(_criteria),'event_cost'].sum()*to_usd/1.E6

            if _x == 0: 
                _xos = _benefit/100
                _cb = r'$\frac{benefit}{cost}$ = '+str(round(_benefit/_cost,2))
                _cstr = 'Cost\n'+str(round(_cost,1))
                _bstr = 'Benefit\n'+str(round(_benefit,1))
            else: 
                _cb = str(round(_benefit/_cost,2))
                _cstr = str(round(_cost,2))
                _bstr = str(round(_benefit,2))

            plt.bar(_x,(_benefit-_cost),bottom=_cost,color=q_colors[_x+1],alpha=0.6,width=_wid)
        
            # ratio
            plt.annotate(_cb,xy=(_x,_benefit+_xos),ha='center',va='bottom',weight='bold')

            if _x != 3:
                # benefit
                plt.plot([_x-_wid/2,_x+_wid/2],[_benefit,_benefit],lw=2,color=q_colors[_x+1])
                plt.annotate(_bstr,xy=(_x,_benefit-_xos),ha='center',va='top')

                # cost
                plt.plot([_x-_wid/2,_x+_wid/2],[_cost,_cost],lw=2,color=q_colors[_x+1])
                plt.annotate(_cstr,xy=(_x,_cost+_xos),ha='center',va='bottom')
                
            else:            
                # benefit
                plt.plot([_x-_wid/2,_x+_wid/2],[_benefit,_benefit],lw=2,color=q_colors[_x+1])
                plt.annotate(_bstr,xy=(_x-_wid/2,_benefit+_xos/20),ha='right',va='bottom',fontsize=8.5)
                
                # cost
                plt.plot([_x-_wid/2,_x+_wid/2],[_cost,_cost],lw=2,color=q_colors[_x+1])
                plt.annotate(_cstr,xy=(_x-_wid/2,_cost+_xos/20),ha='right',va='bottom',fontsize=8.5)
        
        plt.xlim(-0.75,3.5)
        plt.ylim(-0.1)

        plt.xticks([0,1,2,3])
        plt.gca().set_xticklabels(['Scaleup to\nall Samurdhi\nbeneficiaries',
                                   'Scaleup to\nall affected &\n66% of unaffected',
                                   'Scaleup to\nall affected &\n33% of unaffected',
                                   'Perfect\ntargeting within\nSamurdhi'],size=8)
        plt.xlabel('Inclusion error',labelpad=8,weight='bold')
        plt.ylabel('Cost of Samurdhi scaleup [mil. USD]',labelpad=8,weight='bold')
        plt.grid(False)
        sns.despine(bottom=True)

        plt.title(('1 month-equivalent top-up to existing\nSamurdhi enrollees in '+_place+'\nafter qualifying flood').replace('in all ',''),loc='right',pad=-25)

        plt.gca().get_figure().savefig('../output_plots/SL/samurdhi/cost_benefit_'+(_place+'_').replace('all_','')+'rp'+str(_rp)+'.pdf',format='pdf',bbox_inches='tight')
