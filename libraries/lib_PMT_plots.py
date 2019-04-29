import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from libraries.lib_scaleout import get_pmt
from libraries.lib_pds_dict import pds_dict
from libraries.lib_gather_data import match_percentiles, perc_with_spline, reshape_data
from libraries.lib_common_plotting_functions import *
import seaborn as sns

haz_dict = {'SS':'Storm surge',
            'PF':'Precipitation flood',
            'HU':'Hurricane',
            'EQ':'Earthquake',
            'DR':'Drought',
            'FF':'Fluvial flood',
            'CY':'Cyclone Idai'}

def SL_PMT_plots(myCountry,economy,event_level,myiah,myHaz,my_PDS,_wprime,base_str,to_usd):
    out_files = os.getcwd()+'/../output_country/'+myCountry+'/'

    listofquintiles=np.arange(0.20, 1.01, 0.20)
    quint_labels = ['Poorest\nquintile','Second','Third','Fourth','Wealthiest\nquintile']

    myiah['hhid'] = myiah['hhid'].astype('str')
    myiah = myiah.set_index('hhid')
    pmt,_ = get_pmt(myiah)
    myiah['PMT'] = pmt

    for _loc in myHaz[0]:
        for _haz in myHaz[1]:
            for _rp in myHaz[2]:

                plt.cla()
                _ = myiah.loc[(myiah[economy]==_loc)&(myiah['hazard']==_haz)&(myiah['rp']==_rp)].copy()

                _ = _.reset_index().groupby(economy,sort=True).apply(lambda x:match_percentiles(x,perc_with_spline(reshape_data(x.PMT),reshape_data(x.pcwgt_no),listofquintiles),'quintile','PMT'))
                
                for _sort in ['PMT']:

                    _ = _.sort_values(_sort,ascending=True)

                    _['pcwgt_cum_'+base_str] = _['pcwgt_'+base_str].cumsum()
                    _['pcwgt_cum_'+my_PDS] = _['pcwgt_'+my_PDS].cumsum()
                    
                    _['dk0_cum'] = _[['pcwgt_'+base_str,'dk0']].prod(axis=1).cumsum()
                    
                    _['cost_cum_'+my_PDS] = _[['pcwgt_'+my_PDS,'help_received_'+my_PDS]].prod(axis=1).cumsum()
                    # ^ cumulative cost
                    _['cost_frac_'+my_PDS] = _[['pcwgt_'+my_PDS,'help_received_'+my_PDS]].prod(axis=1).cumsum()/_[['pcwgt_'+my_PDS,'help_received_'+my_PDS]].prod(axis=1).sum()
                    # ^ cumulative cost as fraction of total

                    # GET WELFARE COSTS
                    _['dw_cum_'+base_str] = _[['pcwgt_'+base_str,'dw_'+base_str]].prod(axis=1).cumsum()
                    # Include public costs in baseline (dw_cum)
                    ext_costs_base = pd.read_csv(out_files+'public_costs_tax_'+base_str+'_.csv').set_index([economy,'hazard','rp'])
                    
                    ext_costs_base['dw_pub_curr'] = ext_costs_base['dw_pub']/_wprime
                    ext_costs_base['dw_soc_curr'] = ext_costs_base['dw_soc']/_wprime
                    ext_costs_base['dw_tot_curr'] = ext_costs_base[['dw_pub','dw_soc']].sum(axis=1)/_wprime

                    ext_costs_base_sum = ext_costs_base.loc[ext_costs_base['contributer']!=ext_costs_base.index.get_level_values(event_level[0]),
                                                            ['dw_pub_curr','dw_soc_curr','dw_tot_curr']].sum(level=[economy,'hazard','rp']).reset_index()
                    
                    ext_costs_base_pub = float(ext_costs_base_sum.loc[(ext_costs_base_sum[economy]==_loc)&ext_costs_base_sum.eval('(hazard==@_haz)&(rp==@_rp)'),'dw_pub_curr'])
                    ext_costs_base_soc = float(ext_costs_base_sum.loc[(ext_costs_base_sum[economy]==_loc)&ext_costs_base_sum.eval('(hazard==@_haz)&(rp==@_rp)'),'dw_soc_curr'])            
                    ext_costs_base_sum = float(ext_costs_base_sum.loc[(ext_costs_base_sum[economy]==_loc)&ext_costs_base_sum.eval('(hazard==@_haz)&(rp==@_rp)'),'dw_tot_curr'])
                    

                    _['dw_cum_'+my_PDS] = _[['pcwgt_'+my_PDS,'dw_'+my_PDS]].prod(axis=1).cumsum()
                    # ^ cumulative DW, with my_PDS implemented

                    # Include public costs in pds_dw_cum
                    ext_costs_pds = pd.read_csv(out_files+'public_costs_tax_'+my_PDS+'_.csv').set_index([economy,'hazard','rp'])

                    ext_costs_pds['dw_pub_curr'] = ext_costs_pds['dw_pub']/_wprime
                    ext_costs_pds['dw_soc_curr'] = ext_costs_pds['dw_soc']/_wprime
                    ext_costs_pds['dw_tot_curr'] = ext_costs_pds[['dw_pub','dw_soc']].sum(axis=1)/_wprime

                    ext_costs_pds_sum = ext_costs_pds.loc[(ext_costs_pds['contributer']!=ext_costs_pds.index.get_level_values(event_level[0])),
                                                          ['dw_pub_curr','dw_soc_curr','dw_tot_curr']].sum(level=[economy,'hazard','rp']).reset_index()

                    ext_costs_pds_pub = float(ext_costs_pds_sum.loc[(ext_costs_pds_sum[economy]==_loc)&ext_costs_pds_sum.eval('(hazard==@_haz)&(rp==@_rp)'),'dw_pub_curr'])
                    ext_costs_pds_soc = float(ext_costs_pds_sum.loc[(ext_costs_pds_sum[economy]==_loc)&ext_costs_pds_sum.eval('(hazard==@_haz)&(rp==@_rp)'),'dw_soc_curr'])
                    ext_costs_pds_sum = float(ext_costs_pds_sum.loc[(ext_costs_pds_sum[economy]==_loc)&ext_costs_pds_sum.eval('(hazard==@_haz)&(rp==@_rp)'),'dw_tot_curr'])

                    _['dw_cum_'+my_PDS] += (ext_costs_pds_pub + ext_costs_pds_soc)*_['cost_frac_'+my_PDS]
                    _['delta_dw_cum_'+my_PDS] = _['dw_cum_'+base_str]-_['dw_cum_'+my_PDS]


                    ### PMT-ranked population coverage [%]
                    plt.plot(100.*_['pcwgt_cum_'+base_str]/_['pcwgt_'+base_str].sum(),
                             100.*_['dk0_cum']/_[['pcwgt_'+base_str,'dk0']].prod(axis=1).sum())
                    plt.annotate('Total asset losses\n$'+str(round(1E-6*to_usd*_.iloc[-1]['dk0_cum'],1))+' mil.',xy=(0.1,0.85),xycoords='axes fraction',color=greys_pal[7],fontsize=10)
                    if False:
                        plt.plot(100.*_['pcwgt_cum_'+base_str]/_['pcwgt_'+base_str].sum(),
                                 100.*_['dk0_cum']/_[['pcwgt_'+base_str,'dk0']].prod(axis=1).sum())

                plt.xlabel('Population percentile [%]',labelpad=8,fontsize=10)
                plt.ylabel('Cumulative asset losses [%]',labelpad=8,fontsize=10)
                plt.xlim(0);plt.ylim(-0.1)
                plt.gca().xaxis.set_ticks([20,40,60,80,100])
                sns.despine()
                plt.grid(False)

                plt.gcf().savefig('../output_plots/SL/PMT/pcwgt_vs_dk0_'+_loc+'_'+_haz+'_'+str(_rp)+'.pdf',format='pdf',bbox_inches='tight')
                plt.cla()


                #####################################
                ### PMT threshold vs dk (normalized)
                _ = _.sort_values('PMT',ascending=True)
                plt.plot(_['PMT'],100.*_['dk0_cum']/_[['pcwgt_'+base_str,'dk0']].prod(axis=1).sum(),linewidth=1.8,zorder=99,color=q_colors[1])

                for _q in [1,2,3,4,5]:
                    _q_x = _.loc[_['quintile']==_q,'PMT'].max()
                    _q_y = 100.*_.loc[_['quintile']<=_q,['pcwgt_'+base_str,'dk0']].prod(axis=1).sum()/_[['pcwgt_'+base_str,'dk0']].prod(axis=1).sum()
                    if _q == 1: _q_yprime = _q_y/20

                    plt.plot([_q_x,_q_x],[0,_q_y],color=greys_pal[4],ls=':',linewidth=1.5,zorder=91)

                    _usd = ' mil.'
                    plt.annotate((quint_labels[_q-1]+'\n$'+str(round(1E-6*to_usd*_.loc[_['quintile']==_q,['pcwgt_'+base_str,'dk0']].prod(axis=1).sum(),1))+_usd),
                                 xy=(_q_x,_q_y+_q_yprime),color=greys_pal[6],ha='right',va='bottom',style='italic',fontsize=8,zorder=91)

                if False:
                    plt.scatter(_['PMT'],100.*_['dk0_cum']/_[['pcwgt_'+base_str,'dk0']].prod(axis=1).sum(),alpha=0.08,s=6,zorder=10,color=q_colors[1])

                plt.xlabel('Household income [PMT]',labelpad=8,fontsize=10)
                plt.ylabel('Cumulative asset losses [%]',labelpad=8,fontsize=10)
                plt.annotate('Total asset losses\n$'+str(round(1E-6*to_usd*_.iloc[-1]['dk0_cum'],1))+' mil.',xy=(0.1,0.85),xycoords='axes fraction',color=greys_pal[7],fontsize=10)
                plt.xlim(825);plt.ylim(-0.1)

                sns.despine()
                plt.grid(False)

                plt.gcf().savefig('../output_plots/SL/PMT/pmt_vs_dk_norm_'+_loc+'_'+_haz+'_'+str(_rp)+'.pdf',format='pdf',bbox_inches='tight')

                #####################################
                ### PMT threshold vs dk & dw
                plt.cla()
                plt.plot(_['PMT'],_['dk0_cum']*to_usd*1E-6,color=q_colors[1],linewidth=1.8,zorder=99)
                plt.plot(_['PMT'],_['dw_cum_'+base_str]*to_usd*1E-6,color=q_colors[3],linewidth=1.8,zorder=99)

                _y1 = 1.08; _y2 = 1.03
                if _['dk0_cum'].max() < _['dw_cum_'+base_str].max(): _y1 = 1.03; _y2 = 1.08

                plt.annotate('Total asset losses = $'+str(round(_['dk0_cum'].max()*to_usd*1E-6,1))+' million',xy=(0.02,_y1),
                             xycoords='axes fraction',color=q_colors[1],ha='left',va='top',fontsize=10,annotation_clip=False)

                wb_str = 'Total wellbeing losses = \$'+str(round(_['dw_cum_'+base_str].max()*to_usd*1E-6,1))+' million' 
                #wb_natl_str = '(+\$'+str(round(ext_costs_base_sum*to_usd*1E-6,1))+')'
                wb_natl_str = 'National welfare losses\n  $'+str(round(ext_costs_base_sum*to_usd*1E-6,1))+' million'

                plt.annotate(wb_str,xy=(0.02,_y2),xycoords='axes fraction',color=q_colors[3],ha='left',va='top',fontsize=10,annotation_clip=False)
                #plt.annotate(wb_natl_str,xy=(0.02,0.77),xycoords='axes fraction',color=q_colors[3],ha='left',va='top',fontsize=10)            

                for _q in [1,2,3,4,5]:
                    _q_x = _.loc[_['quintile']==_q,'PMT'].max()
                    _q_y = max(_.loc[_['quintile']<=_q,['pcwgt_'+base_str,'dk0']].prod(axis=1).sum()*to_usd*1E-6,
                               _.loc[_['quintile']<=_q,['pcwgt_'+base_str,'dw_'+base_str]].prod(axis=1).sum()*to_usd*1E-6)
                    if _q == 1: _q_yprime = _q_y/25

                    plt.plot([_q_x,_q_x],[0,_q_y],color=greys_pal[4],ls=':',linewidth=1.5,zorder=91)
                    plt.annotate(quint_labels[_q-1],xy=(_q_x,_q_y+7*_q_yprime),color=greys_pal[6],ha='right',va='bottom',style='italic',fontsize=8,zorder=91,annotation_clip=False)


                    # This figures out label ordering (are cumulative asset or cum welfare lossers higher?)
                    _cumk = round(_.loc[_['quintile']<=_q,['pcwgt_'+base_str,'dk0']].prod(axis=1).sum()*to_usd*1E-6,1)
                    _cumw = round(_.loc[_['quintile']<=_q,['pcwgt_'+base_str,'dw_'+base_str]].prod(axis=1).sum()*to_usd*1E-6,1)               
                    if _cumk >= _cumw: 
                        _yprime_k = 4*_q_yprime
                        _yprime_w = 1*_q_yprime
                    else: 
                        _yprime_k = 1*_q_yprime
                        _yprime_w = 4*_q_yprime


                    _qk = round(_.loc[_['quintile']==_q,['pcwgt_'+base_str,'dk0']].prod(axis=1).sum()*to_usd*1E-6,1)
                    _qw = round(_.loc[_['quintile']==_q,['pcwgt_'+base_str,'dw_'+base_str]].prod(axis=1).sum()*to_usd*1E-6,1)

                    plt.annotate('$'+str(_qk)+' mil.',xy=(_q_x,_q_y+_yprime_k),color=q_colors[1],ha='right',va='bottom',style='italic',fontsize=8,zorder=91,annotation_clip=False)
                    plt.annotate('$'+str(_qw)+' mil.',xy=(_q_x,_q_y+_yprime_w),color=q_colors[3],ha='right',va='bottom',style='italic',fontsize=8,zorder=91,annotation_clip=False)

                plt.xlabel('Household income [PMT]',labelpad=8,fontsize=10)
                plt.ylabel('Cumulative losses [mil. US$]',labelpad=8,fontsize=10)
                plt.xlim(825);plt.ylim(-0.1)

                plt.title(' '+str(_rp)+'-year '+haz_dict[_haz].lower()+' in '+_loc,loc='left',color=greys_pal[7],pad=30,fontsize=15)

                sns.despine()
                plt.grid(False)

                plt.gcf().savefig('../output_plots/SL/PMT/pmt_vs_dk0_'+_loc+'_'+_haz+'_'+str(_rp)+'.pdf',format='pdf',bbox_inches='tight')
                plt.close('all')


                #####################################
                ### Cost vs benefit of PMT

                show_net_benefit = False
                if show_net_benefit:
                    _['dw_cum_'+base_str] += (ext_costs_base_pub+ext_costs_base_soc)*_['cost_frac_'+my_PDS]
                #*_[['pcwgt_'+base_str,'dk0']].prod(axis=1).cumsum()/_[['pcwgt_'+base_str,'dk0']].prod(axis=1).sum()
                # ^ include national costs in baseline dw
                _['delta_dw_cum_'+my_PDS] = _['dw_cum_'+base_str]-_['dw_cum_'+my_PDS]
                # redefine this because above changed

                plt.cla()
                plt.plot(_['PMT'],_['cost_cum_'+my_PDS]*to_usd*1E-6,color=q_colors[1],linewidth=1.8,zorder=99)
                plt.plot(_['PMT'],_['delta_dw_cum_'+my_PDS]*to_usd*1E-6,color=q_colors[3],linewidth=1.8,zorder=99)

                plt.annotate('PDS cost =\n$'+str(round(_['cost_cum_'+my_PDS].max()*to_usd*1E-6,2))+' mil.',
                             xy=(_['PMT'].max(),_['cost_cum_'+my_PDS].max()*to_usd*1E-6),color=q_colors[1],weight='bold',ha='left',va='top',fontsize=10,annotation_clip=False)
                plt.annotate('Avoided wellbeing\nlosses = $'+str(round(_.iloc[-1]['delta_dw_cum_'+my_PDS]*to_usd*1E-6,2))+' mil.',
                             xy=(_['PMT'].max(),_.iloc[-1]['delta_dw_cum_'+my_PDS]*to_usd*1E-6),color=q_colors[3],weight='bold',ha='left',va='top',fontsize=10)

                #for _q in [1,2,3,4,5]:
                #    _q_x = _.loc[_['quintile']==_q,'PMT'].max()
                #    _q_y = max(_.loc[_['quintile']<=_q,['pcwgt','dk0']].prod(axis=1).sum()*to_usd*1E-6,
                #               _.loc[_['quintile']<=_q,['pcwgt','dw_no']].prod(axis=1).sum()*to_usd*1E-6)
                #    if _q == 1: _q_yprime = _q_y/20

                #    plt.plot([_q_x,_q_x],[0,_q_y],color=greys_pal[4],ls=':',linewidth=1.5,zorder=91)
                #    plt.annotate(quint_labels[_q-1],xy=(_q_x,_q_y+_q_yprime),color=greys_pal[6],ha='right',va='bottom',style='italic',fontsize=8,zorder=91)

                plt.xlabel('Upper PMT threshold for post-disaster support',labelpad=8,fontsize=12)
                plt.ylabel('Cost & benefit [mil. US$]',labelpad=8,fontsize=12)
                plt.xlim(825)#;plt.ylim(0)

                plt.title(' '+str(_rp)+'-year '+haz_dict[_haz].lower()+'\n  in '+_loc,loc='left',color=greys_pal[7],pad=25,fontsize=15)
                plt.annotate(pds_dict[my_PDS],xy=(0.02,1.03),xycoords='axes fraction',color=greys_pal[6],ha='left',va='bottom',weight='bold',style='italic',fontsize=8,zorder=91,clip_on=False)

                plt.plot(plt.gca().get_xlim(),[0,0],color=greys_pal[2],linewidth=0.90)
                sns.despine(bottom=True)
                plt.grid(False)

                plt.gcf().savefig('../output_plots/SL/PMT/pmt_dk_vs_dw_'+_loc+'_'+_haz+'_'+str(_rp)+'_'+my_PDS+'.pdf',format='pdf',bbox_inches='tight')
                plt.close('all')            
                continue


                #####################################
                ### Cost vs benefit of PMT
                _ = _.fillna(0)
                #_ = _.loc[_['pcwgt_'+my_PDS]!=0].copy()
                _ = _.loc[(_['help_received_'+my_PDS]!=0)&(_['pcwgt_'+my_PDS]!=0)].copy()

                #_['dw_cum_'+my_PDS] = _[['pcwgt_'+my_PDS,'dw_'+my_PDS]].prod(axis=1).cumsum()
                #_['dw_cum_'+my_PDS] += ext_costs_pds_pub + ext_costs_pds_soc*_['cost_frac_'+my_PDS]
                # ^ unchanged from above

                _c1,_c1b = paired_pal[2],paired_pal[3]
                _c2,_c2b = paired_pal[0],paired_pal[1]

                _window = 100
                if _.shape[0] < 100: _window = int(_.shape[0]/5)

                plt.cla()

                _y_values_A = (_['cost_cum_'+my_PDS]*to_usd).diff()/_['pcwgt_'+my_PDS]
                _y_values_B = pd.rolling_mean((_['cost_cum_'+my_PDS]*to_usd).diff()/_['pcwgt_'+my_PDS],_window)

                if _y_values_A.max() >= 1.25*_y_values_B.max() or _y_values_A.min() <= 0.75*_y_values_B.min():
                    plt.scatter(_['PMT'],(_['cost_cum_'+my_PDS]*to_usd).diff()/_['pcwgt_'+my_PDS],color=_c1,s=4,zorder=98,alpha=0.25)
                    plt.plot(_['PMT'],pd.rolling_mean((_['cost_cum_'+my_PDS]*to_usd).diff()/_['pcwgt_'+my_PDS],_window),color=_c1b,lw=1.0,zorder=98)
                else: 
                    plt.plot(_['PMT'],(_['cost_cum_'+my_PDS]*to_usd).diff()/_['pcwgt_'+my_PDS],color=_c1b,lw=1.0,zorder=98)

                plt.scatter(_['PMT'],(_['delta_dw_cum_'+my_PDS]*to_usd).diff()/_['pcwgt_'+my_PDS],color=_c2,s=4,zorder=98,alpha=0.25)
                plt.plot(_['PMT'],pd.rolling_mean((_['delta_dw_cum_'+my_PDS]*to_usd).diff()/_['pcwgt_'+my_PDS],_window),color=_c2b,lw=1.0,zorder=98)
                _y_min = 1.05*pd.rolling_mean((_['delta_dw_cum_'+my_PDS]*to_usd).diff()/_['pcwgt_'+my_PDS],_window).min()
                _y_max = 1.1*max(pd.rolling_mean((_['delta_dw_cum_'+my_PDS]*to_usd).diff()/_['pcwgt_'+my_PDS],_window).max(),
                                 1.05*((_['cost_cum_'+my_PDS]*to_usd).diff()/_['pcwgt_'+my_PDS]).mean()+_q_yprime)

                for _q in [1,2,3,4,5]:
                    _q_x = min(1150,_.loc[_['quintile']==_q,'PMT'].max())
                    #_q_y = max(_.loc[_['quintile']<=_q,['pcwgt_'+my_PDS,'dk0']].prod(axis=1).sum()*to_usd,
                    #           _.loc[_['quintile']<=_q,['pcwgt_'+my_PDS,'dw_no']].prod(axis=1).sum()*to_usd))
                    if _q == 1: 
                        _q_xprime = (_q_x-840)/40
                        _q_yprime = _y_max/200

                    plt.plot([_q_x,_q_x],[_y_min,_y_max],color=greys_pal[4],ls=':',linewidth=1.5,zorder=91)
                    plt.annotate(quint_labels[_q-1],xy=(_q_x-_q_xprime,_y_max),color=greys_pal[6],ha='right',va='top',style='italic',fontsize=7,zorder=99)

                #toggle this
                plt.annotate('PDS cost',xy=(_['PMT'].max()-_q_xprime,((_['cost_cum_'+my_PDS]*to_usd).diff()/_['pcwgt_'+my_PDS]).mean()+_q_yprime),
                             color=_c1b,weight='bold',ha='right',va='bottom',fontsize=8,annotation_clip=False)

                #plt.annotate('Avoided\nwellbeing losses',xy=(_['PMT'].max()-_q_xprime,pd.rolling_mean((_['delta_dw_cum']*to_usd/_['pcwgt_'+my_PDS]).diff(),_window).min()+_q_yprime),
                #             color=_c2b,weight='bold',ha='right',va='bottom',fontsize=8)

                plt.xlabel('Upper PMT threshold for post-disaster support',labelpad=10,fontsize=10)
                plt.ylabel('Marginal impact at threshold [US$ per next enrollee]',labelpad=10,fontsize=10)

                plt.title(str(_rp)+'-year '+haz_dict[_haz].lower()+' in '+_loc,loc='right',color=greys_pal[7],pad=20,fontsize=15)
                plt.annotate(pds_dict[my_PDS],xy=(0.99,1.02),xycoords='axes fraction',color=greys_pal[6],ha='right',va='bottom',weight='bold',style='italic',fontsize=8,zorder=91,clip_on=False)

                plt.plot([840,1150],[0,0],color=greys_pal[2],linewidth=0.90)
                plt.xlim(840,1150);plt.ylim(_y_min,_y_max)
                sns.despine(bottom=True)
                plt.grid(False)

                plt.gcf().savefig('../output_plots/SL/PMT/pmt_slope_cost_vs_benefit_'+_loc+'_'+_haz+'_'+str(_rp)+'_'+my_PDS+'.pdf',format='pdf',bbox_inches='tight')
                plt.close('all')  
