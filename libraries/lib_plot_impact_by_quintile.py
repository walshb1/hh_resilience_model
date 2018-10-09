import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from libraries.lib_country_dir import get_currency,get_economic_unit,int_w_commas
from libraries.lib_common_plotting_functions import q_colors,q_labels,greys_pal

def plot_relative_losses(myCountry,aProv,aDis,anRP,iah):
    economy = get_economic_unit(myCountry)
    
    iah = iah.reset_index().set_index('quintile').sort_index()
    _poor = 100.*iah.loc[((iah.ispoor==1)
                          &(iah[economy]==aProv)
                          &(iah.hazard==aDis)
                          &(iah.rp==anRP)),['dk0','pcwgt']].prod(axis=1).sum()/iah.loc[((iah.ispoor==1)
                                                                                        &(iah[economy]==aProv)
                                                                                        &(iah.hazard==aDis)
                                                                                        &(iah.rp==anRP)),['k','pcwgt']].prod(axis=1).sum()
    _nonpoor = 100*iah.loc[((iah.ispoor==0)
                            &(iah[economy]==aProv)
                            &(iah.hazard==aDis)
                            &(iah.rp==anRP)),['dk0','pcwgt']].prod(axis=1).sum()/iah.loc[((iah.ispoor==0)
                                                                                          &(iah[economy]==aProv)
                                                                                          &(iah.hazard==aDis)
                                                                                          &(iah.rp==anRP)),['k','pcwgt']].prod(axis=1).sum()
    print(aProv,aDis,anRP)
    print('Poor:',_poor)
    print('Non-poor:',_nonpoor)
    print('Ratio:',_poor/_nonpoor,'\n\n')


def plot_impact_by_quintile(myCountry,aProv,aDis,anRP,iah_res,optionPDS):

    output_plots = '../output_plots/'+myCountry+'/'
    economy = get_economic_unit(myCountry)

    plt.figure(1)
    ax = plt.subplot(111)

    plt.figure(2)
    ax2 = plt.subplot(111)

    for myQ in range(1,6): #nQuintiles

        #print(aProv,aDis,anRP,'shape:',iah_res.loc[(iah_res[economy]==aProv)&(iah_res.hazard==aDis)&(iah_res.rp==anRP)&(iah_res.quintile==myQ),'pcwgt'].shape[0])

        k = (0.01*iah_res.loc[(iah_res[economy]==aProv)&(iah_res.hazard==aDis)&(iah_res.rp==anRP)&(iah_res.quintile==myQ),['k','pcwgt']].prod(axis=1).sum()/
             iah_res.loc[(iah_res[economy]==aProv)&(iah_res.hazard==aDis)&(iah_res.rp==anRP)&(iah_res.quintile==myQ),'pcwgt'].sum())

        dk = (iah_res.loc[(iah_res[economy]==aProv)&(iah_res.hazard==aDis)&(iah_res.rp==anRP)&(iah_res.quintile==myQ),['dk0','pcwgt']].prod(axis=1).sum()/
              iah_res.loc[(iah_res[economy]==aProv)&(iah_res.hazard==aDis)&(iah_res.rp==anRP)&(iah_res.quintile==myQ),'pcwgt'].sum())

        dc = (iah_res.loc[(iah_res[economy]==aProv)&(iah_res.hazard==aDis)&(iah_res.rp==anRP)&(iah_res.quintile==myQ),['dc_npv_pre','pcwgt']].prod(axis=1).sum()/
              iah_res.loc[(iah_res[economy]==aProv)&(iah_res.hazard==aDis)&(iah_res.rp==anRP)&(iah_res.quintile==myQ),'pcwgt'].sum())

        dw = (iah_res.loc[(iah_res[economy]==aProv)&(iah_res.hazard==aDis)&(iah_res.rp==anRP)&(iah_res.quintile==myQ),['dw','pcwgt']].prod(axis=1).sum()/
              iah_res.loc[(iah_res[economy]==aProv)&(iah_res.hazard==aDis)&(iah_res.rp==anRP)&(iah_res.quintile==myQ),'pcwgt'].sum())

        pds_nrh = (iah_res.loc[(iah_res[economy]==aProv)&(iah_res.hazard==aDis)&(iah_res.rp==anRP)&(iah_res.quintile==myQ),['pds_nrh','pcwgt']].prod(axis=1).sum()/
                   iah_res.loc[(iah_res[economy]==aProv)&(iah_res.hazard==aDis)&(iah_res.rp==anRP)&(iah_res.quintile==myQ),'pcwgt'].sum())

        pds_dw = (iah_res.loc[(iah_res[economy]==aProv)&(iah_res.hazard==aDis)&(iah_res.rp==anRP)&(iah_res.quintile==myQ),['pds_dw','pcwgt']].prod(axis=1).sum()/
                  iah_res.loc[(iah_res[economy]==aProv)&(iah_res.hazard==aDis)&(iah_res.rp==anRP)&(iah_res.quintile==myQ),'pcwgt'].sum())

        try: pds2_dw = (iah_res.loc[(iah_res[economy]==aProv)&(iah_res.hazard==aDis)&(iah_res.rp==anRP)&(iah_res.quintile==myQ),['pds2_dw','pcwgt']].prod(axis=1).sum()/
                        iah_res.loc[(iah_res[economy]==aProv)&(iah_res.hazard==aDis)&(iah_res.rp==anRP)&(iah_res.quintile==myQ),'pcwgt'].sum())
        except: pds2_dw = 0

        try: pds3_dw = (iah_res.loc[(iah_res[economy]==aProv)&(iah_res.hazard==aDis)&(iah_res.rp==anRP)&(iah_res.quintile==myQ),['pds3_dw','pcwgt']].prod(axis=1).sum()/
                        iah_res.loc[(iah_res[economy]==aProv)&(iah_res.hazard==aDis)&(iah_res.rp==anRP)&(iah_res.quintile==myQ),'pcwgt'].sum())
        except: pds3_dw = 0

        ax.bar([6*ii+myQ for ii in range(1,5)],[dk,dw,pds_nrh,pds_dw],
               color=q_colors[myQ-1],alpha=0.7,label=q_labels[myQ-1])

        #np.savetxt('/Users/brian/Desktop/BANK/hh_resilience_model/output_plots/PH/dk_dc_dw_pds_'+aProv+'_'+aDis+'_'+str(anRP)+'_Q'+str(myQ)+'.csv',[dk,dc,dw,pds_nrh,pds_dw],delimiter=',')

        if myQ==1: 
            ax2.bar([0],[0],color=[q_colors[0]],alpha=0.7,label='No post-disaster support')
            ax2.bar([0],[0],color=[q_colors[2]],alpha=0.7,label='80% of avg Q1 losses covered for Q1')
            ax2.bar([0],[0],color=[q_colors[3]],alpha=0.7,label='80% of avg Q1 losses covered for Q1-Q2')
            ax2.bar([0],[0],color=[q_colors[1]],alpha=0.7,label='80% of avg Q1 losses covered for Q1-Q5')
        ax2.bar([5*myQ+ii for ii in range(0,4)],[dw,pds2_dw,pds3_dw,pds_dw],color=[q_colors[0],q_colors[1],q_colors[2],q_colors[3]],alpha=0.7)
        
        np.savetxt('/Users/brian/Desktop/BANK/hh_resilience_model/output_plots/PH/pds_comparison_'+aProv+'_'+aDis+'_'+str(anRP)+'_Q'+str(myQ)+'.csv',[dw,pds_dw,pds2_dw], delimiter=',')

    out_str = None
    if myCountry == 'FJ': out_str = ['Asset loss','Consumption\nloss (NPV)','Well-being loss','Net cost of\nWinston-like\nsupport','Well-being loss\npost support']
    else: out_str = ['Asset loss','Well-being loss','Net cost of\nUniform PDS','Well-being loss\nwith Uniform PDS']

    trans = ax.get_xaxis_transform() # x in data untis, y in axes fraction
    for ni, ii in enumerate(range(1,5)):
        ann = ax.annotate(out_str[ni], xy=(6*ii+0.5,+0.005), xycoords=trans,ha='left',va='top',weight='bold',fontsize=9,annotation_clip=False)
        #ax.annotate(out_str[ni],xy=(6*ii+0.5,ax.get_ylim()[0]),xycoords='data',ha='left',va='top',weight='bold',fontsize=9,annotation_clip=False)

    fig = ax.get_figure()    
    leg = ax.legend(loc='best',labelspacing=0.75,ncol=1,fontsize=9,borderpad=0.75,fancybox=True,frameon=True,framealpha=0.9)
    leg.get_frame().set_color('white')
    leg.get_frame().set_edgecolor(greys_pal[7])
    leg.get_frame().set_linewidth(0.2)

    plt.figure(1)
    #myxlim = ax.get_xlim()
    #plt.plot([xlim for xlim in myxlim],[0,0],'k-',lw=0.50,color=greys_pal[7],zorder=100,alpha=0.85)
    #plt.xlim(myxlim)

    ax.xaxis.set_ticks([])
    plt.ylabel('Disaster losses ['+get_currency(myCountry)[0][-3:]+' per capita]',labelpad=8,weight='bold',fontsize=11)
    sns.despine(bottom=True)

    print('wrote losses_k_'+aDis+'_'+str(anRP)+'.pdf')
    fig.savefig(output_plots+'npr_'+aProv.replace(' ','')+'_'+aDis+'_'+str(anRP)+'.pdf',format='pdf',bbox_inches='tight')
    fig.savefig(output_plots+'png/npr_'+aProv.replace(' ','')+'_'+aDis+'_'+str(anRP)+'.png',format='png',bbox_inches='tight')

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
    plt.ylabel('Well-being losses ['+get_currency(myCountry)[0][-3:]+' per capita]',labelpad=8,weight='bold')
    fig2.savefig(output_plots+'npr_pds_schemes_'+aProv+'_'+aDis+'_'+str(anRP)+'.pdf',format='pdf',bbox_inches='tight')
    fig2.savefig(output_plots+'png/npr_pds_schemes_'+aProv+'_'+aDis+'_'+str(anRP)+'.png',format='png',bbox_inches='tight')
    plt.clf()
    plt.close('all')
