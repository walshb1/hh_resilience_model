import matplotlib
matplotlib.use('AGG')

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

from libraries.lib_country_dir import get_currency,get_pop_scale_fac,get_subsistence_line,get_economic_unit,int_w_commas
from libraries.lib_common_plotting_functions import greys_pal,q_colors,blues_pal

def axis_data_coords_sys_transform(axis_obj_in,xin,yin,inverse=False):
    """ inverse = False : Axis => Data
                = True  : Data => Axis
    """
    xlim = axis_obj_in.get_xlim()
    ylim = axis_obj_in.get_ylim()

    xdelta = xlim[1] - xlim[0]
    ydelta = ylim[1] - ylim[0]
    if not inverse:
        xout =  xlim[0] + xin * xdelta
        yout =  ylim[0] + yin * ydelta
    else:
        xdelta2 = xin - xlim[0]
        ydelta2 = yin - ylim[0]
        xout = xdelta2 / xdelta
        yout = ydelta2 / ydelta
    return xout,yout

def plot_income_and_consumption_distributions(myCountry,iah,aReg,aDis,anRP):

    try: plt.close('all')
    except: pass     
    

    economy = get_economic_unit(myCountry)
    output_plots = os.getcwd()+'/../output_plots/'+myCountry+'/'

    upper_clip = 1E6
    if myCountry == 'PH': upper_clip = 1.25E5
    if myCountry == 'FJ': upper_clip = 2E4
    if myCountry == 'SL': upper_clip = 4.0E5
    if myCountry == 'MW': upper_clip = 1.E4

    c_bins = [None,50]

    haz_dict = {'SS':'Storm surge',
                'PF':'Precipitation flood',
                'HU':'Hurricane',
                'EQ':'Earthquake',
                'DR':'Drought',
                'FF':'Fluvial flood'}

    simple_plot = True
    sf_x = get_currency(myCountry)[2]
    for _fom,_fom_lab in [('i','Income'),
                          ('c','Consumption')]:

        try:

            ax=plt.gca()

            plt.xlim(0,sf_x*upper_clip)
            if aReg == 'II - Cagayan Valley' and aDis == 'HU' and anRP == 25: plt.ylim(0,400)
        
            mny = get_currency(myCountry)
            plt.xlabel(_fom_lab+r' [USD per person, per year]')
            plt.ylabel('Population'+get_pop_scale_fac(myCountry)[1])
            plt.title(str(anRP)+'-year '+haz_dict[aDis].lower()+' in '+aReg)

            # Income/Cons dist immediately after disaster
            cf_heights, cf_bins = np.histogram(sf_x*iah.loc[(iah[economy]==aReg)&(iah.hazard==aDis)&(iah.rp==anRP),_fom+'_pre_reco'].clip(upper=upper_clip), bins=c_bins[1],
                                               weights=iah.loc[(iah[economy]==aReg)&(iah.hazard==aDis)&(iah.rp==anRP),'pcwgt']/get_pop_scale_fac(myCountry)[0])
            if c_bins[0] is None: c_bins = [cf_bins,cf_bins]

            # Income dist before disaster
            ci_heights, _bins = np.histogram(sf_x*iah.loc[(iah[economy]==aReg)&(iah.hazard==aDis)&(iah.rp==anRP),'c_initial'].clip(upper=upper_clip), bins=c_bins[1],
                                             weights=iah.loc[(iah[economy]==aReg)&(iah.hazard==aDis)&(iah.rp==anRP),'pcwgt']/get_pop_scale_fac(myCountry)[0])

            # Income dist after reconstruction
            #cf_reco_hgt, _bins = np.histogram(sf_x*iah.loc[(iah[economy]==aReg)&(iah.hazard==aDis)&(iah.rp==anRP),'c_post_reco'].clip(upper=upper_clip), bins=c_bins[1],
            #                                  weights=iah.loc[(iah[economy]==aReg)&(iah.hazard==aDis)&(iah.rp==anRP),'pcwgt']/get_pop_scale_fac(myCountry)[0])

            sns.despine()
            plt.gca().grid(False,axis='x')

            ax.step(c_bins[1][1:], ci_heights, label=aReg+' - FIES income', linewidth=1.2,color=greys_pal[6])
            #leg = ax.legend(loc='best',labelspacing=0.75,ncol=1,fontsize=9,borderpad=0.75,fancybox=True,frameon=True,framealpha=0.9)             
            ax.get_figure().savefig(output_plots+'npr_poverty_'+_fom+'_'+aReg.replace(' ','').replace('-','')+'_'+aDis+'_'+str(anRP)+'_1of3.pdf',format='pdf')

            #ax.step(c_bins[1][:-1], cf_heights, label=aReg+' - post-disaster', facecolor=q_colors[1],alpha=0.45)
            ax.bar(c_bins[1][:-1], cf_heights, width=(c_bins[1][1]-c_bins[1][0]), align='edge', 
                   label=aReg+' - post-disaster', facecolor=q_colors[1],edgecolor=None,linewidth=0,alpha=0.65)
            #leg = ax.legend(loc='best',labelspacing=0.75,ncol=1,fontsize=9,borderpad=0.75,fancybox=True,frameon=True,framealpha=0.9)
            ax.get_figure().savefig(output_plots+'npr_poverty_'+_fom+'_'+aReg.replace(' ','').replace('-','')+'_'+aDis+'_'+str(anRP)+'_2of3.pdf',format='pdf')

            plt.annotate('Pre-disaster '+_fom_lab.lower()+'\n(reported)',xy=(c_bins[1][-2],ci_heights[-1]),xytext=(c_bins[1][-4],ci_heights[-1]*1.04),
                         arrowprops=dict(arrowstyle="-",facecolor=greys_pal[8],connectionstyle="angle,angleA=0,angleB=90,rad=5"),
                         clip_on=False,size=7,weight='light',ha='right',va='center',color=greys_pal[8])
            plt.annotate('Post-disaster '+_fom_lab.lower()+'\n(modeled)',xy=((c_bins[1][-2]+c_bins[1][-1])/1.99,cf_heights[-1]*0.95),xytext=(c_bins[1][-4],cf_heights[-1]*0.95),
                         arrowprops=dict(arrowstyle="-",facecolor=greys_pal[4]),clip_on=False,size=7,weight='light',ha='right',va='center',color=blues_pal[8])

            #ax.bar(c_bins[1][:-1], cf_reco_hgt, width=(c_bins[1][1]-c_bins[1][0]), label=aReg+' - post-reconstruction', facecolor=q_colors[1],edgecolor=q_colors[1],alpha=0.65)
            #ax.step(c_bins[1][1:], ci_heights, label=aReg+' - FIES income', linewidth=1.2,color=greys_pal[8])            

            #if myC_ylim == None: myC_ylim = ax.get_ylim()
            #plt.ylim(myC_ylim[0],2.5*myC_ylim[1])
            # ^ Need this for movie making, but better to let the plot limits float if not

            #leg = ax.legend(loc='best',labelspacing=0.75,ncol=1,fontsize=9,borderpad=0.75,fancybox=True,frameon=True,framealpha=0.9)
            #leg.get_frame().set_color('white')
            #leg.get_frame().set_edgecolor(greys_pal[7])
            #leg.get_frame().set_linewidth(0.2)

            if not simple_plot:
                ax.annotate('Total asset losses: '+str(round(iah.loc[(iah[economy]==aReg)&(iah.hazard==aDis)&(iah.rp==anRP),['pcwgt','dk0']].prod(axis=1).sum()/mny[1],1))+mny[0],
                            xy=(0.03,-0.18), xycoords=leg.get_frame(),size=8,va='top',ha='left',annotation_clip=False,zorder=100)
                ax.annotate('Reg. well-being losses: '+str(round(iah.loc[(iah[economy]==aReg)&(iah.hazard==aDis)&(iah.rp==anRP),['pcwgt','dw']].prod(axis=1).sum()/(df.wprime.mean()*mny[1]),1))+mny[0],
                            xy=(0.03,-0.50), xycoords=leg.get_frame(),size=8,va='top',ha='left',annotation_clip=False,zorder=100)
                ax.annotate('Natl. liability: '+str(round(float(public_costs.loc[(public_costs.contributer!=aReg)&(public_costs[economy]==aReg)&(public_costs.hazard==aDis)&(public_costs.rp==anRP),['transfer_pub']].sum()*1.E3/mny[1]),1))+mny[0],
                            xy=(0.03,-0.92), xycoords=leg.get_frame(),size=8,va='top',ha='left',annotation_clip=False,zorder=100)
                ax.annotate('Natl. well-being losses: '+str(round(float(public_costs.loc[(public_costs.contributer!=aReg)&(public_costs[economy]==aReg)&(public_costs.hazard==aDis)&(public_costs.rp==anRP),'dw_tot_curr'].sum()*1.E3/mny[1]),1))+mny[0].replace('b','m'),
                            xy=(0.03,-1.24), xycoords=leg.get_frame(),size=8,va='top',ha='left',annotation_clip=False,zorder=100)

            try:
                net_chg_pov_c = int(iah.loc[iah.eval('region==@aReg & hazard==@aDis & rp==@anRP & c_pre_reco<=pov_line'),'pcwgt'].sum()
                                    -iah.loc[iah.eval('region==@aReg & hazard==@aDis & rp==@anRP & c_initial<=pov_line'),'pcwgt'].sum())
                net_chg_pov_i = int(iah.loc[iah.eval('region==@aReg & hazard==@aDis & rp==@anRP & i_pre_reco<=pov_line'),'pcwgt'].sum()
                                    -iah.loc[iah.eval('region==@aReg & hazard==@aDis & rp==@anRP & c_initial<=pov_line'),'pcwgt'].sum())
            except:
                net_chg_pov_c = int(iah.loc[iah.eval('district==@aReg & hazard==@aDis & c_pre_reco<=pov_line'),'pcwgt'].sum()
                                    -iah.loc[iah.eval('district==@aReg & hazard==@aDis & c_initial<=pov_line'),'pcwgt'].sum())
                net_chg_pov_i = int(iah.loc[iah.eval('district==@aReg & hazard==@aDis & i_pre_reco<=pov_line'),'pcwgt'].sum()
                                    -iah.loc[iah.eval('district==@aReg & hazard==@aDis & c_initial<=pov_line'),'pcwgt'].sum())

            net_chg_pov = int(round(net_chg_pov_i/100.,0)*100)
            if _fom == 'c': net_chg_pov = int(round(net_chg_pov_c/100.,0)*100)
            #print('c:',net_chg_pov_c,' i:',net_chg_pov_i)

            try: net_chg_pov_pct = abs(round(100.*float(net_chg_pov)/float(iah.loc[(iah[economy]==aReg)&(iah.hazard==aDis)&(iah.rp==anRP),'pcwgt'].sum()),1))
            except: net_chg_pov_pct = 0

            trans = ax.get_xaxis_transform() # x in data units, y in axes fraction

            pov_anno_y = 0.75
            sub_anno_y = 0.90
            anno_y_offset = 0.045

            _,pov_anno_y_data = axis_data_coords_sys_transform(ax,0,pov_anno_y,inverse=False)
            _,sub_anno_y_data = axis_data_coords_sys_transform(ax,0,sub_anno_y,inverse=False)

            plt.plot([sf_x*iah.pov_line.mean(),sf_x*iah.pov_line.mean()],[0,pov_anno_y_data],'k-',lw=2.,color=greys_pal[8],zorder=100,alpha=0.85)
            ax.annotate('Poverty line',xy=(sf_x*1.1*iah.pov_line.mean(),pov_anno_y),xycoords=trans,ha='left',va='top',fontsize=9,
                        annotation_clip=False,weight='bold',color=greys_pal[7])

            ax.annotate('Increase of '+int_w_commas(net_chg_pov)+' ('+str(net_chg_pov_pct)+'%)\nFilipinos in '+_fom_lab.lower()+' poverty',
                        weight='light',color=greys_pal[7],xy=(sf_x*1.1*iah.pov_line.mean(),pov_anno_y-anno_y_offset),
                        xycoords=trans,ha='left',va='top',fontsize=9,annotation_clip=False)

            sub_line, net_chg_sub = get_subsistence_line(myCountry), None
            if sub_line is not None:
                net_chg_sub = int(round((iah.loc[((iah[economy]==aReg)&(iah.hazard==aDis)&(iah.rp==anRP)&(iah.c_pre_reco <= sub_line)),'pcwgt'].sum()
                                         -iah.loc[((iah[economy]==aReg)&(iah.hazard==aDis)&(iah.rp==anRP)&(iah.c_initial <= sub_line)),'pcwgt'].sum())/100.,0)*100)
                if _fom == 'i': net_chg_sub = int(round((iah.loc[((iah[economy]==aReg)&(iah.hazard==aDis)&(iah.rp==anRP)&(iah.i_pre_reco<=sub_line)),'pcwgt'].sum()
                                                         -iah.loc[((iah[economy]==aReg)&(iah.hazard==aDis)&(iah.rp==anRP)&(iah.c_initial<=sub_line)),'pcwgt'].sum())/100.,0)*100)

                try: net_chg_sub_pct = round(100.*float(net_chg_sub)/float(iah.loc[(iah[economy]==aReg)&(iah.hazard==aDis)&(iah.rp==anRP),'pcwgt'].sum()),1)
                except: net_chg_sub_pct = 0

                plt.plot([sf_x*sub_line,sf_x*sub_line],[0,sub_anno_y_data ],'k-',lw=2.5,color=greys_pal[8],zorder=100,alpha=0.85)
                ax.annotate('Subsistence line',xy=(sf_x*1.1*sub_line,sub_anno_y),xycoords=trans,ha='left',va='top',
                            color=greys_pal[7],fontsize=9,annotation_clip=False,weight='bold')
                ax.annotate('Increase of '+int_w_commas(net_chg_sub)+' ('+str(net_chg_sub_pct)+'%)\nFilipinos in '+_fom_lab.lower()+' subsistence',
                            weight='light',color=greys_pal[7],xy=(sf_x*1.1*sub_line,sub_anno_y-anno_y_offset),
                            xycoords=trans,ha='left',va='top',fontsize=9,annotation_clip=False)

            #print(aReg,aDis,anRP,net_chg_pov,'people into poverty &',net_chg_sub,'into subsistence') 

            fig = ax.get_figure()
            fig.savefig(output_plots+'npr_poverty_'+_fom+'_'+aReg.replace(' ','').replace('-','')+'_'+aDis+'_'+str(anRP)+'.pdf',format='pdf')
            #fig.savefig(output_plots+'png/npr_poverty_k_'+aReg.replace(' ','').replace('-','')+'_'+aDis+'_'+str(anRP)+'.png',format='png')
            plt.clf()
            plt.close('all')
            print('wrote '+aReg+'_poverty_'+_fom+'_'+aDis+'_'+str(anRP)+'.pdf')

        except: print('Error running '+aDis+' '+aDis+' '+anRP)
