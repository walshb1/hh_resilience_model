import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from libraries.lib_pds_dict import pds_dict
from libraries.lib_common_plotting_functions import q_colors,q_labels,greys_pal
from libraries.lib_country_dir import get_currency,get_economic_unit,int_w_commas

def plot_relative_losses(myCountry,aProv,aDis,anRP,iah):
    economy = get_economic_unit(myCountry)

    iah = iah.loc[iah.affected_cat=='a']
    iah = iah.reset_index().set_index('quintile').sort_index()

    _poor = 100.*iah.loc[((iah.ispoor==1)
                          &(iah[economy]==aProv)
                          &(iah.hazard==aDis)
                          &(iah.rp==anRP)),['dk0','pcwgt_no']].prod(axis=1).sum()/iah.loc[((iah.ispoor==1)
                                                                                           &(iah[economy]==aProv)
                                                                                           &(iah.hazard==aDis)
                                                                                           &(iah.rp==anRP)),['k','pcwgt_no']].prod(axis=1).sum()
    _nonpoor = 100*iah.loc[((iah.ispoor==0)
                            &(iah[economy]==aProv)
                            &(iah.hazard==aDis)
                            &(iah.rp==anRP)),['dk0','pcwgt_no']].prod(axis=1).sum()/iah.loc[((iah.ispoor==0)
                                                                                             &(iah[economy]==aProv)
                                                                                             &(iah.hazard==aDis)
                                                                                             &(iah.rp==anRP)),['k','pcwgt_no']].prod(axis=1).sum()
    print(aProv,aDis,anRP)
    print('Poor:',_poor)
    print('Non-poor:',_nonpoor)
    print('Ratio:',_poor/_nonpoor,'\n\n')


def plot_impact_by_quintile(myCountry,aProv,aDis,anRP,iah,my_PDS='no',currency='USD'):
    iah = iah.loc[iah.affected_cat=='a']

    output_plots = '../output_plots/'+myCountry+'/'
    economy = get_economic_unit(myCountry)

    plt.figure(1)
    ax = plt.subplot(111)

    plt.figure(2)
    ax2 = plt.subplot(111)

    _curr = 1
    _curr_sf = 1E-3; _curr_sf_str = ',000 ' 
    if currency == 'USD': 
        _curr = get_currency(myCountry)[2]
        _curr_sf = 1
        _curr_sf_str = ''

    for myQ in range(1,6): #nQuintiles

        #print(aProv,aDis,anRP,'shape:',iah.loc[(iah[economy]==aProv)&(iah.hazard==aDis)&(iah.rp==anRP)&(iah.quintile==myQ),'pcwgt_no'].shape[0])

        k = _curr*_curr_sf*(0.01*iah.loc[(iah[economy]==aProv)&(iah.hazard==aDis)&(iah.rp==anRP)&(iah.quintile==myQ),['k','pcwgt_no']].prod(axis=1).sum()/
                            iah.loc[(iah[economy]==aProv)&(iah.hazard==aDis)&(iah.rp==anRP)&(iah.quintile==myQ),'pcwgt_no'].sum())

        dk = _curr*_curr_sf*(iah.loc[(iah[economy]==aProv)&(iah.hazard==aDis)&(iah.rp==anRP)&(iah.quintile==myQ),['dk0','pcwgt_no']].prod(axis=1).sum()/
                             iah.loc[(iah[economy]==aProv)&(iah.hazard==aDis)&(iah.rp==anRP)&(iah.quintile==myQ),'pcwgt_no'].sum())

        dc = _curr*_curr_sf*(iah.loc[(iah[economy]==aProv)&(iah.hazard==aDis)&(iah.rp==anRP)&(iah.quintile==myQ),['dc_pre_reco','pcwgt_no']].prod(axis=1).sum()/
                             iah.loc[(iah[economy]==aProv)&(iah.hazard==aDis)&(iah.rp==anRP)&(iah.quintile==myQ),'pcwgt_no'].sum())

        dw = _curr*_curr_sf*(iah.loc[(iah[economy]==aProv)&(iah.hazard==aDis)&(iah.rp==anRP)&(iah.quintile==myQ),['dw_no','pcwgt_no']].prod(axis=1).sum()/
                             iah.loc[(iah[economy]==aProv)&(iah.hazard==aDis)&(iah.rp==anRP)&(iah.quintile==myQ),'pcwgt_no'].sum())

        pds_nrh = _curr*_curr_sf*(iah.loc[(iah[economy]==aProv)&(iah.hazard==aDis)&(iah.rp==anRP)&(iah.quintile==myQ),['net_help_received_'+my_PDS,'pcwgt_'+my_PDS]].prod(axis=1).sum()/
                                  iah.loc[(iah[economy]==aProv)&(iah.hazard==aDis)&(iah.rp==anRP)&(iah.quintile==myQ),'pcwgt_'+my_PDS].sum())

        pds_dw =  _curr*_curr_sf*(iah.loc[(iah[economy]==aProv)&(iah.hazard==aDis)&(iah.rp==anRP)&(iah.quintile==myQ),['dw_'+my_PDS,'pcwgt_'+my_PDS]].prod(axis=1).sum()/
                                  iah.loc[(iah[economy]==aProv)&(iah.hazard==aDis)&(iah.rp==anRP)&(iah.quintile==myQ),'pcwgt_'+my_PDS].sum())

        pds2_dw = _curr*0
        pds3_dw = _curr*0

        ax.bar([6*ii+myQ for ii in range(1,5)],[dk,dw,pds_nrh,pds_dw],
               color=q_colors[myQ-1],alpha=0.7,label=q_labels[myQ-1])

        #np.savetxt('/Users/brian/Desktop/BANK/hh_resilience_model/output_plots/PH/dk_dc_dw_pds_'+aProv+'_'+aDis+'_'+str(anRP)+'_Q'+str(myQ)+'.csv',[dk,dc,dw,pds_nrh,pds_dw],delimiter=',')

        if myQ==1: 
            ax2.bar([0],[0],color=[q_colors[0]],alpha=0.7,label='No post-disaster support')
            ax2.bar([0],[0],color=[q_colors[2]],alpha=0.7,label='80% of avg Q1 losses covered for Q1')
            ax2.bar([0],[0],color=[q_colors[3]],alpha=0.7,label='80% of avg Q1 losses covered for Q1-Q2')
            ax2.bar([0],[0],color=[q_colors[1]],alpha=0.7,label='80% of avg Q1 losses covered for Q1-Q5')
        try: 
            ax2.bar([5*myQ+ii for ii in range(0,4)],[dw,pds3_dw,pds3_dw,pds_dw],color=[q_colors[0],q_colors[1],q_colors[2],q_colors[3]],alpha=0.7)
            np.savetxt('/Users/brian/Desktop/BANK/hh_resilience_model/output_plots/PH/pds_comparison_'+aProv+'_'+aDis+'_'+str(anRP)+'_Q'+str(myQ)+'.csv',
                       [dw,pds_dw,pds2_dw], delimiter=',')
        except: pass

    out_str = None
    if myCountry == 'FJ': out_str = ['Asset loss','Consumption\nloss (NPV)','Well-being loss','Net cost of\nWinston-like\nsupport','Well-being loss\npost support']
    else: out_str = ['Asset loss','Well-being loss','Net cash benefit of\n'+pds_dict[my_PDS],'Well-being loss\nwith '+pds_dict[my_PDS]]

    trans = ax.get_xaxis_transform() # x in data untis, y in axes fraction
    for ni, ii in enumerate(range(1,5)):
        ann = ax.annotate(out_str[ni], xy=(6*ii+0.5,+0.005), xycoords=trans,ha='left',va='top',fontsize=8,annotation_clip=False)
        #ax.annotate(out_str[ni],xy=(6*ii+0.5,ax.get_ylim()[0]),xycoords='data',ha='left',va='top',weight='bold',fontsize=9,annotation_clip=False)

    fig = ax.get_figure()    
    leg = ax.legend(loc='best',labelspacing=0.75,ncol=1,fontsize=8,borderpad=0.75,fancybox=True,frameon=True,framealpha=0.9)
    leg.get_frame().set_color('white')
    leg.get_frame().set_edgecolor(greys_pal[7])
    leg.get_frame().set_linewidth(0.2)

    plt.figure(1)
    myxlim = ax.get_xlim()
    plt.plot([xlim for xlim in myxlim],[0,0],'k-',lw=0.75,color=greys_pal[3],zorder=100,alpha=0.85)
    plt.xlim(myxlim)

    ax.xaxis.set_ticks([])
    ax.tick_params(axis='y',labelsize=8)
    
    #plt.plot(ax.get_xlim(),[0,0],color=greys_pal[2],linewidth=0.5,zorder=10)
    if currency != 'USD': currency = get_currency(myCountry)[0][-3:]
    plt.ylabel('Disaster losses ['+_curr_sf_str+currency+' per affected person]',labelpad=8,fontsize=8)
    sns.despine(bottom=True)
    plt.grid(False)


    _success = False
    while not _success:
        try:
            fig.savefig(output_plots+'npr_'+aProv.replace(' ','')+'_'+aDis+'_'+str(anRP)+'_'+my_PDS+'_'+currency.lower()+'.pdf',format='pdf',bbox_inches='tight')
            #fig.savefig(output_plots+'png/npr_'+aProv.replace(' ','')+'_'+aDis+'_'+str(anRP)+'_'+my_PDS+'_'+currency.lower()+'.png',format='png',bbox_inches='tight')
            _success = True
        except: 
            print('having trouble...')

    plt.figure(2)
    fig2 = ax2.get_figure()
    leg = ax2.legend(loc='best',labelspacing=0.75,ncol=1,fontsize=9,borderpad=0.75,fancybox=True,frameon=True,framealpha=0.9)
    leg.get_frame().set_color('white')
    leg.get_frame().set_edgecolor(greys_pal[7])
    leg.get_frame().set_linewidth(0.2)

    ann_y = -ax2.get_ylim()[1]/30

    n_pds_options = 4
    out_str = ['Q1','Q2','Q3','Q4','Q5']
    for ni, ii in enumerate(range(1,6)): # quintiles
        ax2.annotate(out_str[ni],xy=(5*ii+0.05,ann_y),zorder=100,xycoords='data',
                     ha='left',va='center',weight='bold',fontsize=9,annotation_clip=False)
        plt.plot([5*ii+1.20,5*ii+3.78],[ann_y,ann_y],'k-',lw=1.50,color=greys_pal[7],zorder=100,alpha=0.85)
        plt.plot([5*ii+3.78,5*ii+3.78],[ann_y*0.9,ann_y*1.1],'k-',lw=1.50,color=greys_pal[7],zorder=100,alpha=0.85)

    ax2.xaxis.set_ticks([])
    plt.xlim(3,32)
    plt.plot([i for i in ax2.get_xlim()],[0,0],'k-',lw=1.5,color=greys_pal[7],zorder=100,alpha=0.85)

    if currency != 'USD': currency = get_currency(myCountry)[0][-3:]
    plt.ylabel('Well-being losses ['+currency+' per capita]',labelpad=8,weight='bold')

    _success = False
    try:
        fig2.savefig(output_plots+'npr_pds_schemes_'+aProv+'_'+aDis+'_'+str(anRP)+'_'+currency.lower()+'.pdf',format='pdf',bbox_inches='tight')
        #fig2.savefig(output_plots+'png/npr_pds_schemes_'+aProv+'_'+aDis+'_'+str(anRP)+'.png',format='png',bbox_inches='tight')
        _success = True
    except: print('not working..NOT trying again on npr_pds_schemes_***')

    plt.clf()
    plt.close('all')
