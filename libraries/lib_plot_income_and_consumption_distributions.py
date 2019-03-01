import matplotlib
matplotlib.use('AGG')

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

from libraries.lib_country_dir import get_currency,get_pop_scale_fac,get_subsistence_line,get_poverty_line,get_economic_unit,int_w_commas,get_demonym
from libraries.lib_common_plotting_functions import greys_pal,q_colors,blues_pal,paired_pal

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

def plot_income_and_consumption_distributions(myC,iah,aReg,aDis,anRP,label_subsistence=True,currency=''):
    iah = iah.reset_index()
    economy = get_economic_unit(myC)

    iah = iah.loc[iah.pcwgt_no!=0].copy()

    try: plt.close('all')
    except: pass

    if aReg == 'ompong':
        reg_crit = "((region=='I - Ilocos')|(region=='II - Cagayan Valley')|(region=='CAR'))"
        aReg = 'path of Typhoon Mangkhut'
    else: reg_crit = '('+economy+'==@aReg)'
        
    economy = get_economic_unit(myC)
    output_plots = os.getcwd()+'/../output_plots/'+myC+'/'

    # Number of bins in histograms. Ignore the None argument
    c_bins = [None,50]

    # Dictionary for labeling
    haz_dict = {'SS':'Storm surge',
                'PF':'Precipitation flood',
                'HU':'Typhoon',
                'EQ':'Earthquake',
                'DR':'Drought',
                'FF':'Fluvial flood'}

    simple_plot = True
    stack_wealthy = True

    upper_clip = 1E6
    if myC == 'PH': 
        if aReg == 'VIII - Eastern Visayas': 
            upper_clip = 1.25E5
            # Hack hack hack
        else: upper_clip = 1.5E5
    if myC == 'FJ': upper_clip = 2E4
    if myC == 'SL': 
        upper_clip = 3.25E5
        if aReg == 'Rathnapura': upper_clip = 3.0E5
    if myC == 'MW': 
        if aReg == 'Lilongwe': upper_clip = 4.0E5
        else: upper_clip = 2.5E5
    
    sf_x = 1
    if currency.lower() == 'usd': sf_x = get_currency(myC)[2]
    elif myC == 'PH': currency = 'kPhP'; sf_x = 1E-3
    elif myC == 'MW': currency = ',000 MWK'; sf_x = 1E-3
    elif myC == 'SL': currency = ',000 LKR'; sf_x = 1E-3
    else: currency = get_currency(myC)[0]

    for _fom,_fom_lab in [('i','Income'),
                          ('c','Consumption')]:

        ax=plt.gca()
        plt.cla()

        plt.xlim(0,sf_x*upper_clip)
        if not stack_wealthy: 
            upper_clip*=1.1
            if currency.lower() != 'usd' and myC == 'PH': plt.xlim([0,100])

        if aReg == 'II - Cagayan Valley' and aDis == 'HU': plt.ylim(0,400)
        elif aReg == 'VIII - Eastern Visayas' and aDis == 'HU': plt.ylim(0,500)
        elif aReg == 'Rathnapura': plt.ylim(0,130)

        plt.xlabel(_fom_lab+r' ('+currency+' per person, per year)',labelpad=8,fontsize=8)
        plt.ylabel('Population'+get_pop_scale_fac(myC)[1],labelpad=8,fontsize=8)
        if _fom == 'i': plt.title(str(anRP)+'-year '+haz_dict[aDis].lower()+' in '+aReg)
        
        # Income/Cons dist immediately after disaster
        cf_heights, cf_bins = np.histogram(sf_x*iah.loc[iah.eval(reg_crit+'&(hazard==@aDis)&(rp==@anRP)'),_fom+'_pre_reco'].clip(upper=upper_clip), bins=c_bins[1],
                                           weights=iah.loc[iah.eval(reg_crit+'&(hazard==@aDis)&(rp==@anRP)'),'pcwgt_no']/get_pop_scale_fac(myC)[0]) 

        if c_bins[0] is None: c_bins = [cf_bins,cf_bins]

        # Income dist before disaster
        ci_heights, _bins = np.histogram(sf_x*iah.loc[iah.eval(reg_crit+'&(hazard==@aDis)&(rp==@anRP)'),'c_initial'].clip(upper=upper_clip), bins=c_bins[1],
                                         weights=iah.loc[iah.eval(reg_crit+'&(hazard==@aDis)&(rp==@anRP)'),'pcwgt_no']/get_pop_scale_fac(myC)[0])

        # Income dist after reconstruction
        #cf_reco_hgt, _bins = np.histogram(sf_x*iah.loc[iah.eval(reg_crit+'&(hazard==@aDis)&(rp==@anRP)'),'c_post_reco'].clip(upper=upper_clip), bins=c_bins[1],
        #                                  weights=iah.loc[iah.eval(reg_crit+'&(hazard==@aDis)&(rp==@anRP)'),'pcwgt_no']/get_pop_scale_fac(myC)[0])

        sns.despine()
        plt.gca().grid(False)

        pre_step = ax.step(c_bins[1][1:], ci_heights, label=aReg+' - FIES income', linewidth=1.25,color=greys_pal[7])
        #leg = ax.legend(loc='best',labelspacing=0.75,ncol=1,fontsize=9,borderpad=0.75,fancybox=True,frameon=True,framealpha=0.9)
        plt.ylim(0)
        
        if stack_wealthy: pre_ann = plt.annotate('Pre-disaster '+_fom_lab.lower()+'\n(FIES data)',xy=(c_bins[1][-2],ci_heights[-1]),xytext=(c_bins[1][-4],ci_heights[-1]*1.075),
                                                 arrowprops=dict(arrowstyle="-",color=greys_pal[8],connectionstyle="angle,angleA=0,angleB=90,rad=5"),
                                                 annotation_clip=False,size=8,weight='light',ha='right',va='center',color=greys_pal[8])
        else: pre_ann = plt.annotate('Pre-disaster '+_fom_lab.lower()+'\n(FIES data)',xy=((c_bins[1][5]+c_bins[1][6])/2,ci_heights[5]),xytext=(c_bins[1][8],ci_heights[5]*1.02),
                                     arrowprops=dict(arrowstyle="-",color=greys_pal[8],connectionstyle="angle,angleA=0,angleB=90,rad=5"),
                                     annotation_clip=False,size=8,weight='light',ha='left',va='bottom',color=greys_pal[8])

        _success = False
        _counter = 0
        while not _success and _counter < 15:
            try:
                _fout = output_plots+'npr_poverty_'+_fom+'_'+aReg.replace(' ','').replace('-','')+'_'+aDis+'_'+str(anRP)+'_'+currency[-3:].lower()+'_1of3.pdf'
                ax.get_figure().savefig(_fout,format='pdf',bbox_inches='tight')                    
                _success = True
            except:
                print('no good! try again in plot_income_and_consumption_distributions (1/3-'+str(_counter)+')')
                _counter+=1

        

        #ax.step(c_bins[1][:-1], cf_heights, label=aReg+' - post-disaster', facecolor=q_colors[1],alpha=0.45)
        #ax.bar(c_bins[1][:-1], -(ci_heights-cf_heights), width=(c_bins[1][1]-c_bins[1][0]), align='edge', 
        #       label=aReg+' - post-disaster', facecolor=paired_pal[4],edgecolor=None,linewidth=0,alpha=0.65,bottom=ci_heights)
        ax.bar(c_bins[1][:-1],cf_heights, width=(c_bins[1][1]-c_bins[1][0]), align='edge', 
               label=aReg+' - post-disaster', facecolor=paired_pal[4],edgecolor=None,linewidth=0,alpha=0.75)
        #leg = ax.legend(loc='best',labelspacing=0.75,ncol=1,fontsize=9,borderpad=0.75,fancybox=True,frameon=True,framealpha=0.9)

        if stack_wealthy: post_ann = plt.annotate('Post-disaster '+_fom_lab.lower()+'\n(modeled)',xy=((c_bins[1][-2]+c_bins[1][-1])/1.99,cf_heights[-1]*0.90),
                                                  xytext=(c_bins[1][-4],cf_heights[-1]*0.90),arrowprops=dict(arrowstyle="-",facecolor=greys_pal[8]),
                                                  annotation_clip=False,size=8,weight='light',ha='right',va='center',color=paired_pal[5])
        else: 
            ax.lines.remove(ax.lines[0])
            pre_step = ax.step(c_bins[1][1:], ci_heights, label=aReg+' - FIES income', linewidth=1.0,color=greys_pal[7],alpha=0.65)

            pre_ann.remove()
            pre_ann = plt.annotate('Pre-disaster '+_fom_lab.lower()+'\n(FIES data)',xy=((c_bins[1][5]+c_bins[1][6])/2,ci_heights[5]),xytext=(c_bins[1][8],ci_heights[5]*1.02),
                                   arrowprops=dict(arrowstyle="-",color=greys_pal[8],connectionstyle="angle,angleA=0,angleB=90,rad=5",alpha=0.65),
                                   annotation_clip=False,size=8,weight='light',ha='left',va='bottom',color=greys_pal[8],alpha=0.65)
            post_ann = plt.annotate('Post-disaster '+_fom_lab.lower()+'\n(modeled)',xy=(c_bins[1][5],1.1*cf_heights[6]),
                                    xytext=(c_bins[1][9],1.1*cf_heights[6]),annotation_clip=False,size=8,weight='light',ha='left',va='center',color=paired_pal[5],
                                    arrowprops=dict(arrowstyle="-",color=paired_pal[5],connectionstyle="angle,angleA=0,angleB=90,rad=5"))

        _success = False; _counter = 0
        while not _success and _counter < 10:
            try:
                _fout = output_plots+'npr_poverty_'+_fom+'_'+aReg.replace(' ','').replace('-','')+'_'+aDis+'_'+str(anRP)+'_'+currency[-3:].lower()+'_2of3.pdf'
                ax.get_figure().savefig(_fout,format='pdf',bbox_inches='tight')
                _success = True
            except:
                print('no good! try again in plot_income_and_consumption_distributions (2/3)')
                _counter+=1

        # These are done above
        #plt.annotate('Pre-disaster '+_fom_lab.lower()+'\n(FIES data)',xy=(c_bins[1][-2],ci_heights[-1]),xytext=(c_bins[1][-4],ci_heights[-1]*1.075),
        #             arrowprops=dict(arrowstyle="-",facecolor=greys_pal[8],connectionstyle="angle,angleA=0,angleB=90,rad=5"),
        #             annotation_clip=False,size=7,weight='light',ha='right',va='center',color=greys_pal[8])
        #plt.annotate('Post-disaster '+_fom_lab.lower()+'\n(modeled)',xy=((c_bins[1][-2]+c_bins[1][-1])/1.99,cf_heights[-1]*0.90),xytext=(c_bins[1][-4],cf_heights[-1]*0.90),
        #             arrowprops=dict(arrowstyle="-",facecolor=greys_pal[8]),annotation_clip=False,size=7,weight='light',ha='right',va='center',color=paired_pal[5])
        if not stack_wealthy: pre_ann.remove(); post_ann.remove()

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
            ax.annotate('Total asset losses: '+str(round(sf_x*iah.loc[iah.eval(reg_crit+'&(hazard==@aDis)&(rp==@anRP)'),['pcwgt_no','dk0']].prod(axis=1).sum(),1))+currency,
                        xy=(0.03,-0.18), xycoords=leg.get_frame(),size=8,va='top',ha='left',annotation_clip=False,zorder=100)
            ax.annotate('Reg. well-being losses: '+str(round(sf_x*iah.loc[iah.eval(reg_crit+'&(hazard==@aDis)&(rp==@anRP)'),['pcwgt_no','dw']].prod(axis=1).sum()/df.wprime.mean(),1))+currency,
                        xy=(0.03,-0.50), xycoords=leg.get_frame(),size=8,va='top',ha='left',annotation_clip=False,zorder=100)
            ax.annotate('Natl. liability: '+str(round(float(sf_x*public_costs.loc[(public_costs.contributer!=aReg)&(public_costs[economy]==aReg)&(public_costs.hazard==aDis)&(public_costs.rp==anRP),['transfer_pub']].sum()*1.E3),1))+currency,
                        xy=(0.03,-0.92), xycoords=leg.get_frame(),size=8,va='top',ha='left',annotation_clip=False,zorder=100)
            ax.annotate('Natl. well-being losses: '+str(round(sf_x*float(public_costs.loc[(public_costs.contributer!=aReg)&(public_costs[economy]==aReg)&(public_costs.hazard==aDis)&(public_costs.rp==anRP),'dw_tot_curr'].sum()),1))+',000 '+currency,
                        xy=(0.03,-1.24), xycoords=leg.get_frame(),size=8,va='top',ha='left',annotation_clip=False,zorder=100)

        try:
            net_chg_pov_c = int(iah.loc[iah.eval(reg_crit+'&(hazard==@aDis)&(rp==@anRP)&(c_pre_reco<=pov_line)'),'pcwgt_no'].sum()
                                -iah.loc[iah.eval(reg_crit+'&(hazard==@aDis)&(rp==@anRP)&(c_initial<=pov_line)'),'pcwgt_no'].sum())
            net_chg_pov_i = int(iah.loc[iah.eval(reg_crit+'&(hazard==@aDis)&(rp==@anRP)&(i_pre_reco<=pov_line)'),'pcwgt_no'].sum()
                                -iah.loc[iah.eval(reg_crit+'&(hazard==@aDis)&(rp==@anRP)&(c_initial<=pov_line)'),'pcwgt_no'].sum())
        except:
            net_chg_pov_c = int(iah.loc[iah.eval('district==@aReg & hazard==@aDis & c_pre_reco<=pov_line'),'pcwgt_no'].sum()
                                -iah.loc[iah.eval('district==@aReg & hazard==@aDis & c_initial<=pov_line'),'pcwgt_no'].sum())
            net_chg_pov_i = int(iah.loc[iah.eval('district==@aReg & hazard==@aDis & i_pre_reco<=pov_line'),'pcwgt_no'].sum()
                                -iah.loc[iah.eval('district==@aReg & hazard==@aDis & c_initial<=pov_line'),'pcwgt_no'].sum())

        net_chg_pov = int(round(net_chg_pov_i/100.,0)*100)
        if _fom == 'c': net_chg_pov = int(round(net_chg_pov_c/100.,0)*100)
        #print('c:',net_chg_pov_c,' i:',net_chg_pov_i)

        try: net_chg_pov_pct = abs(round(100.*float(net_chg_pov)/float(iah.loc[iah.eval(reg_crit+'&(hazard==@aDis)&(rp==@anRP)'),'pcwgt_no'].sum()),1))
        except: net_chg_pov_pct = 0

        trans = ax.get_xaxis_transform() # x in data units, y in axes fraction

        pov_anno_y = 0.80
        sub_anno_y = 0.95
        anno_y_offset = 0.045

        _,pov_anno_y_data = axis_data_coords_sys_transform(ax,0,pov_anno_y,inverse=False)
        _,sub_anno_y_data = axis_data_coords_sys_transform(ax,0,sub_anno_y,inverse=False)
    
        plt.plot([sf_x*iah.pov_line.mean(),sf_x*iah.pov_line.mean()],[0,pov_anno_y_data],'k-',lw=1.0,color=greys_pal[8],zorder=100,alpha=0.85,ls=':')
        ax.annotate('Poverty line',xy=(sf_x*1.1*iah.pov_line.mean(),pov_anno_y),xycoords=trans,ha='left',va='top',fontsize=9,
                    annotation_clip=False,weight='bold',color=greys_pal[7])

        ax.annotate('Increase of '+int_w_commas(net_chg_pov)+' ('+str(net_chg_pov_pct)+'% of regional pop.)\n in '+_fom_lab.lower()+' poverty',
                    weight='light',color=greys_pal[7],xy=(sf_x*1.1*iah.pov_line.mean(),pov_anno_y-anno_y_offset),
                    xycoords=trans,ha='left',va='top',fontsize=9,annotation_clip=False)

        sub_line, net_chg_sub = get_subsistence_line(myC), None
        if not label_subsistence: sub_line = None

        if sub_line is not None:
            net_chg_sub = int(round((iah.loc[iah.eval(reg_crit+'&(hazard==@aDis)&(rp==@anRP)&(c_pre_reco<=@sub_line)'),'pcwgt_no'].sum()
                                     -iah.loc[iah.eval(reg_crit+'&(hazard==@aDis)&(rp==@anRP)&(c_initial<=@sub_line)'),'pcwgt_no'].sum())/100.,0)*100)
            if _fom == 'i': net_chg_sub = int(round((iah.loc[iah.eval(reg_crit+'&(hazard==@aDis)&(rp==@anRP)&(i_pre_reco<=@sub_line)'),'pcwgt_no'].sum()
                                                     -iah.loc[iah.eval(reg_crit+'&(hazard==@aDis)&(rp==@anRP)&(c_initial<=@sub_line)'),'pcwgt_no'].sum())/100.,0)*100)

            try: net_chg_sub_pct = round(100.*float(net_chg_sub)/float(iah.loc[iah.eval(reg_crit+'&(hazard==@aDis)&(rp==@anRP)'),'pcwgt_no'].sum()),1)
            except: net_chg_sub_pct = 0

            plt.plot([sf_x*sub_line,sf_x*sub_line],[0,sub_anno_y_data ],'k-',lw=1.0,color=greys_pal[8],zorder=100,alpha=0.85,ls=':')
            ax.annotate('Subsistence line',xy=(sf_x*1.1*sub_line,sub_anno_y),xycoords=trans,ha='left',va='top',
                        color=greys_pal[7],fontsize=9,annotation_clip=False,weight='bold')
            ax.annotate('Increase of '+int_w_commas(net_chg_sub)+' ('+str(net_chg_sub_pct)+'% of regional pop.)\n in '+_fom_lab.lower()+' subsistence',
                        weight='light',color=greys_pal[7],xy=(sf_x*1.1*sub_line,sub_anno_y-anno_y_offset),
                        xycoords=trans,ha='left',va='top',fontsize=9,annotation_clip=False)

        #print(aReg,aDis,anRP,net_chg_pov,'people into poverty &',net_chg_sub,'into subsistence') 

        fig = ax.get_figure()
        
        _success = False; _counter = 0
        while not _success and _counter < 4:
            try:
                _fout = output_plots+'npr_poverty_'+_fom+'_'+aReg.replace(' ','').replace('-','')+'_'+aDis+'_'+str(anRP)+'_'+currency[-3:].lower()+'.pdf'
                fig.savefig(_fout,format='pdf',bbox_inches='tight')
                plt.clf(); plt.close('all')
                _success = True
                print('wrote '+aReg+'_poverty_'+_fom+'_'+aDis+'_'+str(anRP)+'.pdf')
            except:
                print('no good! try again in plot_income_and_consumption_distributions')
                _counter+= 1
