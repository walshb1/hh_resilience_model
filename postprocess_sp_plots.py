import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import os, glob

from libraries.lib_common_plotting_functions import title_legend_labels,q_colors,greys_pal
from libraries.lib_country_dir import average_over_rp
from libraries.lib_pds_dict import pds_crit_dict,pds_colors

import seaborn as sns
pairs_pal = sns.color_palette('Paired', n_colors=12)

sns.set_style('whitegrid')
mpl.rcParams['legend.facecolor'] = 'white'
pd.set_option('display.width', 220)

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
    
    print(_sp)
    plt.plot(sp_files[_sp].sum(level='rp').index,sp_files[_sp]['event_cost'].sum(level='rp')*to_usd/1.E6,label=_sp+cost,color=pairs_pal[_ix]) 
    plt.plot(sp_files[_sp].sum(level='rp').index,pds_effects['dw_DELTA_'+_sp].sum(level='rp')*to_usd,label=_sp+' benefit',ls=':',color=pairs_pal[_ix+1])

    _ix+=2

title_legend_labels(plt,'Sri Lanka','Return Period [years]','SP cost when event occurs [M USD]',[1,1000])
plt.xscale('log')
plt.gca().get_figure().savefig(out+'sp_costs.pdf',format='pdf')


########################################################################
# Samurdhi is a constant payout, but there is still RP-dependence
# - cost of disbursement (annual & when event occurs)
# - wellbeing benefit (annual & when event occurs)
# - cost-benefit ratio

# x-axis is targeting error
# y-axis is in mUSD
pds_effects = pds_effects.reset_index('rp')

_wid = 0.45
rp_range = [5,10,25,50,100]
affected_place = ['Colombo','Rathnapura','all']
for _place in affected_place:
    for _rp in rp_range:
        plt.close('all')
        plt.figure(figsize=(6,6))

        for _x, _sp in enumerate(['samurdhi_scaleup','samurdhi_scaleup66','samurdhi_scaleup33','samurdhi_scaleup00']):
            sp_files[_sp] = sp_files[_sp].reset_index().drop([_c for _c in ['index','level_0'] if _c in sp_files[_sp].columns],axis=1)

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
        plt.xlabel('Inclusion error',labelpad=10,fontsize=10)
        plt.ylabel('Cost of Samurdhi scaleup [mil. USD]',labelpad=10,fontsize=10)
        plt.grid(False)
        sns.despine(bottom=True)

        plt.title(('1 month-equivalent top-up to existing\nSamurdhi enrollees in '+_place+'\nafter '+str(_rp)+'-year flood').replace('in all ',''),loc='right',pad=-25)

        plt.gca().get_figure().savefig('../output_plots/SL/samurdhi/cost_benefit_'+(_place+'_').replace('all_','')+'rp'+str(_rp)+'.pdf',format='pdf',bbox_inches='tight')


########################################################################
# This one compares scaling up vs out. 
# - Samurdhi is a constant payout, but there is still RP-dependence
# - cost of disbursement (annual & when event occurs)
# - wellbeing benefit (annual & when event occurs)
# - cost-benefit ratio

# number of beneficiaries
# --> this can vary totally independently between scaleup & out
# --> ... depends for scaleup on coverage of samurdhi
# --> ... depends for scaleout on the PMT threshold 

# x-axis is targeting error
# y-axis is in mUSD

_wid = 0.45
rp_range = [5,10,25,50,100]
affected_place = ['Colombo','Rathnapura','all']
for _place in affected_place:
    for _rp in rp_range:
        plt.close('all')
        plt.figure(figsize=(6,6))

        n_info = []
        for _x, _sp in enumerate(['samurdhi_scaleup00','scaleout_samurdhi']):
            # Both scenarios = perfect targeting. 
            # NB: means I'm a dum-dum for setting different conventions for up & out

            sp_files[_sp] = sp_files[_sp].reset_index().drop([_c for _c in ['index','level_0'] if _c in sp_files[_sp].columns],axis=1)

            _criteria = '(rp==@_rp)'
            if _place != 'all': _criteria+='&(district==@_place)'

            _benefit = pds_effects.loc[pds_effects.eval(_criteria),'dw_DELTA_'+_sp].sum()*to_usd
            _cost = sp_files[_sp].loc[sp_files[_sp].eval(_criteria),'event_cost'].sum()*to_usd/1.E6

            if _x == 1: _xos = _benefit/100
            _cb = r'$\frac{benefit}{cost}$ = '+str(round(_benefit/_cost,2))
            _cstr = 'Cost: '+str(round(_cost,2))
            _bstr = 'Benefit: '+str(round(_benefit,2))

            plt.bar(_x,(_benefit-_cost),bottom=_cost,color=q_colors[_x+1],alpha=0.6,width=_wid)
        
            # ratio
            plt.annotate(_cb,xy=(_x,_benefit+_xos),ha='center',va='bottom',weight='bold')

            if _x == 1:
                # benefit
                plt.plot([_x-_wid/2,_x+_wid/2],[_benefit,_benefit],lw=2,color=q_colors[_x+1])
                plt.annotate(_bstr+'M',xy=(_x-1.02*_wid/2,_benefit+_xos/10),ha='right',va='bottom',fontsize=8,style='italic')

                # cost
                plt.plot([_x-_wid/2,_x+_wid/2],[_cost,_cost],lw=2,color=q_colors[_x+1])
                plt.annotate(_cstr+'M',xy=(_x-1.02*_wid/2,_cost+_xos/10),ha='right',va='bottom',fontsize=8,style='italic')
                
            else:            
                # benefit
                plt.plot([_x-_wid/2,_x+_wid/2],[_benefit,_benefit],lw=2,color=q_colors[_x+1])
                plt.annotate(_bstr+'M',xy=(_x+1.02*_wid/2,_benefit+_xos/10),ha='left',va='bottom',fontsize=8,style='italic')
                
                # cost
                plt.plot([_x-_wid/2,_x+_wid/2],[_cost,_cost],lw=2,color=q_colors[_x+1])
                plt.annotate(_cstr+'M',xy=(_x+1.02*_wid/2,_cost+_xos/10),ha='left',va='bottom',fontsize=8,style='italic')
        
            n_info.append(str(float(round(sp_files[_sp].loc[sp_files[_sp].eval(_criteria)].eval('n_enrolled').sum()/
                                          sp_files[_sp].loc[sp_files[_sp].eval(_criteria)].eval('n_enrolled/reg_frac_enrolled').sum(),1))))
                              
        #plt.xlim(-0.75,3.5)
        plt.ylim(-0.05)

        plt.xticks([0,1])

        plt.gca().set_xticklabels(['Scaleup to affected\nSamurdhi enrollees\n('+n_info[0]+'% of district\nare beneficiaries)',
                                   'Scaleout to affected\nusing PMT\n('+n_info[1]+'% of district\nare beneficiaries)'],size=9)
        plt.ylabel('Cost, benefit of post-disaster support [mil. USD]',labelpad=10,fontsize=10,linespacing=1.5)
        plt.grid(False)
        sns.despine(left=True)

        plt.title(('1 month-equivalent Samurdhi\npayment to '+_place+'\nafter '+str(_rp)+'-year flood').replace('to all ',''),loc='left',color=greys_pal[7],pad=-25)
        plt.annotate('Perfect targeting',xy=(0.0,0.89),xycoords='axes fraction',color=greys_pal[5],style='italic')

        plt.gca().get_figure().savefig('../output_plots/SL/samurdhi/up_vs_out_cb_'+(_place+'_').replace('all_','')+'rp'+str(_rp)+'.pdf',format='pdf',bbox_inches='tight')
        plt.cla()

plt.close('all')

########################################################################
# This one plots dW for several PDS options, with RP on the x-axis
# --> ROI labeled
plt.cla()

pds_effects = pd.read_csv('../output_country/'+myCountry+'/pds_effects.csv').set_index(['district','hazard','rp'])

pds_to_plot = ['samurdhi_scaleup','samurdhi_scaleup00','scaleout_samurdhi','scaleout_samurdhi_universal']
for _pds in pds_to_plot:
    pds_effects['dw_DELTA_'+_pds]/=150
    pds_effects['cost_'+_pds] = pds_effects['dw_DELTA_'+_pds]/pds_effects['ROI_event_'+_pds]
    pds_effects = pds_effects.drop('ROI_event_'+_pds,axis=1)
pds_effects = pds_effects.drop([_c for _c in pds_effects.columns if 'ROI_event_' in _c],axis=1)

pds_effects = pds_effects.sum(level=['rp']).reset_index()
pds_effects = pds_effects.loc[pds_effects.rp>=10]

pds_effects_avg,_ = average_over_rp(pds_effects.set_index('rp'))
pds_effects_avg = pds_effects_avg.sum().to_frame().T
print(pds_effects_avg)

for _pds in pds_to_plot:

    if int(pds_effects_avg['cost_'+_pds]) < 1:
        _cost_str = str(int(round(pds_effects_avg['cost_'+_pds]*1E3,0)))+'K'
    else: _cost_str = str(float(round(pds_effects_avg['cost_'+_pds],2)))+'M'

    plt.loglog(pds_effects.rp,pds_effects['dw_DELTA_'+_pds],color=pds_colors[_pds],linewidth=2)
    plt.annotate(pds_crit_dict[_pds].replace(' (','\n('),xy=(1.1*pds_effects.iloc[-1]['rp'],1.01*pds_effects.iloc[-1]['dw_DELTA_'+_pds]),
                 ha='left',va='bottom',fontsize=8,color=greys_pal[7],weight='bold')
    plt.annotate('Annual cost = '+_cost_str+'\nProgram BCR = '+str(round(float(pds_effects_avg.eval('dw_DELTA_'+_pds+'/cost_'+_pds)),1)),
                 xy=(1.1*pds_effects.iloc[-1]['rp'],0.98*pds_effects.iloc[-1]['dw_DELTA_'+_pds]),va='top',fontsize=8,color=greys_pal[7])

    _event_cost = float(pds_effects.loc[pds_effects.rp==50,'cost_'+_pds])

    _event_label = 'M'
    if _event_cost < 1: 
        _event_cost = str(int(round(_event_cost*1E3,0)))
        _event_label='K'
    else: _event_cost = str(round(_event_cost,1))

    _event_ben = float(pds_effects.loc[pds_effects.rp==50,'dw_DELTA_'+_pds])

    plt.annotate('$'+_event_cost+_event_label,xy=(50,_event_ben),xytext=(55,_event_ben*0.94),
                 arrowprops=dict(shrink=0.01,headwidth=0.5,headlength=0,width=0.5,color=greys_pal[6],connectionstyle='arc3'),
                 ha='left',va='top',fontsize=8,color=greys_pal[6],weight='light')
    
plt.plot([50,50],[0,1E4],color=greys_pal[5],ls=':')
plt.annotate('Cost of ASP\nafter 50-year flood',xy=(55,39),ha='left',va='top',fontsize=8,color=greys_pal[6],weight='bold')

plt.title('Expected benefit of ASP (1 month Samurdhi top-up)\nby return period & beneficiary group',loc='left',fontsize=15,color=greys_pal[7],linespacing=1.65,pad=20)
#plt.annotate('One month Samurdhi topup to',xy=(0.96,1.04),xycoords='axes fraction',color=greys_pal[7],annotation_clip=False,ha='left',va='bottom')
plt.xlabel('Return period [years]',labelpad=10,fontsize=10)
plt.xlim(9,1100)
plt.xticks([10,25,50,100,500,1000])
plt.gca().set_xticklabels([10,25,50,100,500,1000],size=8)

plt.ylabel('Avoided wellbeing losses [mil. US$]',labelpad=10,fontsize=10)
plt.ylim(9E-1,4E1)
plt.yticks([1E0,5E0,1E1,2.5E1])
plt.gca().set_yticklabels(['1','5','10','25'],size=8)

sns.despine()
plt.grid(False)

plt.gca().get_figure().savefig('../output_plots/SL/samurdhi/benefit_vs_rp.pdf',format='pdf',bbox_inches='tight')
