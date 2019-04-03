import pandas as pd
import numpy as np
import seaborn as sns
import brewer2mpl as brew
import matplotlib.pyplot as plt

from libraries.lib_country_dir import get_poverty_line

sns.set_style('whitegrid')
brew_pal = brew.get_map('Set1', 'qualitative', 8).mpl_colors
greys_pal = sns.color_palette('Greys', n_colors=9)
pubugn_pal = sns.color_palette('PuBuGn', n_colors=9)
blues_pal = sns.color_palette('Blues', n_colors=9)

pairs_pal = sns.color_palette('Paired', n_colors=12)
reg_colorsA = [pairs_pal[0],pairs_pal[2],pairs_pal[4]]
reg_colorsB = [pairs_pal[1],pairs_pal[3],pairs_pal[5]]

q_colors = [pubugn_pal[1],pubugn_pal[3],pubugn_pal[5],pubugn_pal[6],pubugn_pal[8]]
q_labels = ['Poorest\nquintile','Second','Third','Fourth','Wealthiest\nquintile']

############################################
# Do quick agriculture plot
if True:
    try: _df = pd.read_csv('../inputs/PH/FIES2015_ompong.csv')
    except:
        _df = pd.read_csv('../inputs/PH/FIES2015.csv')
        _df = _df.loc[_df.eval('(w_regn==1)|(w_regn==2)|(w_regn==14)')]
        _df.to_csv('../inputs/PH/FIES2015_ompong.csv')

    _df['pcwgt'] = _df.eval('hhwgt*fsize')
    _df['agpcinc'] = _df.eval('aginc/fsize')
    
    pov_line = get_poverty_line('PH')
    sf_aginc = (26.8*_df[['hhwgt','aginc']].prod(axis=1).sum()*1E-9/202.2)/(_df[['hhwgt','aginc']].prod(axis=1).sum()*1E-9)
    #           ^ total ag losses from Ompong                       ^ AG national accounts
    
    _df['pcinc_final'] = _df.eval('pcinc-@sf_aginc*agpcinc')

    # Calculate poverty headcount
    pov_increase = _df.loc[(_df['pcinc']>pov_line)&(_df.eval('pcinc-@sf_aginc*agpcinc<=@pov_line')),'pcwgt'].sum()
    pov_gap_init = 1.-_df.loc[_df['pcinc']<pov_line,['pcinc','pcwgt']].prod(axis=1).sum()/(_df.loc[_df['pcinc']<pov_line,'pcwgt'].sum()*pov_line)
    pov_gap_final = 1.-(_df.loc[_df.eval('(pcinc_final<=@pov_line)'),['pcinc_final','pcwgt']].prod(axis=1).sum()
                        /(_df.loc[_df.eval('(pcinc_final<=@pov_line)'),'pcwgt'].sum()*pov_line))

    pov_gap_povfinal = 1.-(_df.loc[_df.eval('(pcinc<=@pov_line)'),['pcinc_final','pcwgt']].prod(axis=1).sum()
                           /(_df.loc[_df.eval('(pcinc<=@pov_line)'),'pcwgt'].sum()*pov_line))

    print('ag fraction of total hh income:',_df[['agpcinc','pcwgt']].prod(axis=1).sum()/_df[['pcinc','pcwgt']].prod(axis=1).sum())
    print('poverty increse:',pov_increase)
    print('initial poverty gap:',pov_gap_init)
    print('final poverty gap:',pov_gap_final)
    print('final povgap for hh in pov before ompong:',pov_gap_povfinal)
    
    pov_line/=1E3
    _df['pcwgt']/=1E3
    _df['agpcinc']/=1E3
    _df['pcinc']/=1E3

    cf_heights, bins = np.histogram(_df.eval('pcinc-(@sf_aginc*agpcinc)'), bins=800,weights=_df['pcwgt'])   
    ci_heights, _ = np.histogram(_df['pcinc'], bins=bins,weights=_df['pcwgt'])

    plt.gca().bar(bins[:-1], cf_heights, width=(bins[1]-bins[0]), align='edge', 
                  label='Less '+str(round(100*sf_aginc,1))+'% of\nagricultural income', facecolor=q_colors[1],edgecolor=None,linewidth=0,alpha=0.65,zorder=97)
    plt.gca().step(bins[1:], ci_heights, label='FIES income', linewidth=1.2,color=greys_pal[5],zorder=98) 

    plt.plot([pov_line,pov_line],[0,700],color=greys_pal[7],zorder=97.5)
    plt.plot([pov_line-0.67,pov_line],[598.5,598.5],color=greys_pal[7],zorder=97.5,clip_on=False)
    plt.annotate('Poverty line',xy=(get_poverty_line('PH')/1E3,14),xycoords='data',rotation=90,weight='bold',zorder=99,va='bottom',ha='right',fontsize=8)
    plt.legend(loc='upper right')

    plt.xlim(0,60)
    plt.ylim(0,600)
    plt.xlabel('Per capita income [,000 PhP per year]',weight='bold',labelpad=8)
    plt.ylabel('Population [,000]',weight='bold',labelpad=8)

    sns.despine();plt.grid(False)
    plt.gcf().savefig('/Users/brian/Desktop/Dropbox/Bank/ompong/plots/agriculture.pdf',format='pdf',bbox_inches='tight')


assert(False)

############################################
# Get fa
if False:
    fa_file = pd.read_csv('../intermediate/PH/hazard_ratios.csv')
    
    fa_file = fa_file.loc[fa_file.hazard!='EQ']
    fa_file = fa_file.loc[(fa_file.region == 'I - Ilocos')|(fa_file.region=='II - Cagayan Valley')|(fa_file.region=='CAR')]
    
    fa_file = fa_file.reset_index().set_index(['region','hazard','rp'])
    
    fa_file = fa_file['fa'].mean(level=['region','hazard','rp'])
    
    print(fa_file)




#############################################
# Plot dk_tot vs poverty increase (headcount)

if True:
    summary = pd.read_csv('../output_country/PH/my_summary_no.csv')

    summary = summary.loc[summary.hazard!='EQ']
    summary = summary.loc[(summary.region == 'I - Ilocos')|(summary.region=='II - Cagayan Valley')|(summary.region=='CAR')]

    summary = summary.reset_index().set_index(['region','hazard','rp']).drop(['index','ratio_dw_lim_tot',
                                                                              'dk_sub','dk_lim',
                                                                              'dw_lim','dw_sub',
                                                                              'res_lim','res_sub'],axis=1)

    try: 
        pov_hh = pd.read_csv('~/Desktop/tmp/ompong.csv').set_index(['region','hazard','rp'])

    except:
        pov_hh = pd.read_csv('../output_country/PH/net_chg_pov_reg_haz_rp.csv')
    
        pov_hh = pov_hh.loc[pov_hh.hazard!='EQ']
        pov_hh = pov_hh.loc[(pov_hh.region == 'I - Ilocos')|(pov_hh.region=='II - Cagayan Valley')|(pov_hh.region=='CAR')]
        pov_hh = pov_hh.reset_index().set_index(['region','hazard','rp'])

        pov_hh.to_csv('/Users/brian/Desktop/tmp/ompong.csv')

    pline = get_poverty_line('PH')
    
    ##############
    summary['poverty_change'] = pov_hh['net_chg_pov_c']

    summary = summary[['dk_tot','poverty_change']].sum(level=['hazard','rp'])

    summary = summary.reset_index()
    summary['dk_tot'] *= 1E-3 # was already in millions; now in billions
    summary['poverty_change'] *= 1E-3 # now in thousands

    for _d in [['HU','Wind',1],['PF','Precipitation flood',2]]:
        plt.cla()
        ax = summary.loc[(summary.hazard==_d[0])&(summary.rp<=500)].plot('dk_tot','poverty_change',
                                                                         label=_d[1],color=brew_pal[_d[2]],lw=2.0,legend=False,zorder=99)
        #summary.loc[(summary.hazard=='PF')&(summary.rp!=2000)].plot('dk_tot','poverty_change',ax=ax,color=brew_pal[2],label='Precipitation flood',lw=2.0)
        #summary.loc[(summary.hazard=='SS')&(summary.rp!=2000)].plot('dk_tot','poverty_change',ax=ax,color=brew_pal[5],label='Storm surge')

        for _rp in [1,10,25]:
            _x = float(summary.loc[(summary.hazard==_d[0])&(summary.rp==_rp),'dk_tot'])
            _yhi = float(summary.loc[(summary.hazard==_d[0])&(summary.rp==_rp),'poverty_change'])
            plt.plot([_x,_x],[0,_yhi],linestyle=':',color=greys_pal[4],zorder=10)
            plt.plot([0,_x],[_yhi,_yhi],linestyle=':',color=greys_pal[4],zorder=10)

            _lb = ' '
            if _rp != 10: _lb = '\n'
            plt.annotate(str(_rp)+'-year'+_lb+'event',xy=(_x,_yhi),xycoords='data',ha='right',va='bottom',fontsize=6.5,weight='bold',color=greys_pal[6])

        plt.title('Potential '+_d[1].lower()+' impacts in\nthe path of Typhoon Ompong')#\n(Ilocos, Cagayan Valley, & CAR)')
        
        plt.xlabel(_d[1]+' damage to assets [bil. PhP]',labelpad=8,weight='bold')
        plt.ylabel('Consumption poverty increase (thousands)',labelpad=8,weight='bold')

        if _d[0] == 'HU':
            plt.xlim(0,50)
            plt.ylim(0,800)
        elif _d[0] == 'PF': 
            plt.xlim(0,8)
            plt.ylim(0,110)

        plt.grid(False)
        sns.despine()

        plt.gcf().savefig('/Users/brian/Desktop/Dropbox/Bank/ompong/plots/ompong_'+_d[0]+'.pdf',format='pdf')


####################################################
# Plot asset losses by decile
if True:

    # --> Bar plot: Absolute & Relative losses, by quintile
    # load file (household level)
    try:
        _hh = pd.read_csv('/Users/brian/Desktop/Dropbox/Bank/ompong/data/hh_losses_ompong.csv').set_index(['region','hazard','rp'])
        _hh = _hh.drop([_c for _c in ['index','Unnamed: 0'] if _c in _hh.columns],axis=1)

    except:
        _hh = pd.read_csv('../output_country/PH/iah_tax_no_.csv')
        _hh = _hh[['region','hazard','rp','hhid','quintile','pcwgt','c','dk0','k','dw']]
    
        _hh = _hh.loc[((_hh.region=='I - Ilocos')|(_hh.region=='II - Cagayan Valley')|(_hh.region=='CAR'))&(_hh.rp!=2000)&(_hh.hazard!='EQ')]
        _hh = _hh.drop([_c for _c in ['Unnamed: 0','Unnamed: 0.1'] if _c in _hh.columns],axis=1)

        if True:
            print('Redefining quintiles in the region(s)')
            _hh = _hh.drop('quintile',axis=1)
            _hh['commonality'] = 'ompong'
            listofquintiles=np.arange(0.20, 1.01, 0.20)
            _hh = _hh.reset_index().groupby('commonality',sort=True).apply(lambda x:match_percentiles(x,perc_with_spline(reshape_data(x.c),
                                                                                                                         reshape_data(x.pcwgt),
                                                                                                                         listofquintiles),'quintile'))
            _hh = _hh.reset_index().set_index(['region','hazard','rp'])

        _hh.to_csv('/Users/brian/Desktop/Dropbox/Bank/ompong/data/hh_losses_ompong.csv')
        

    # These should be equal (asset losses in summary in millions)
    #print(_hh[['dk0','pcwgt']].prod(axis=1).sum(level=['hazard','rp']).sort_index())
    #print(summary.head())

    # Now the region = 'path of Ompong'
    _hh = _hh.reset_index().set_index(['hazard','rp','quintile','region','hhid'])

    _quintiles = pd.DataFrame(index=_hh.index.copy())
    _quintiles['tot_asset_loss'] = _hh[['dk0','pcwgt']].prod(axis=1) 
    _quintiles['tot_asset'] = _hh[['k','pcwgt']].prod(axis=1)
    #_quintiles = _quintiles.squeeze()
        
    _quintiles = _quintiles.sum(level=['hazard','rp','quintile']).sort_index()
    _quintiles['rel_asset_loss'] = _quintiles['tot_asset_loss']/_quintiles['tot_asset']

    _quintiles.to_csv('/Users/brian/Desktop/Dropbox/Bank/ompong/data/ompong_quintiles.csv')
    #print(_quintiles.head())

    # These should be equal (asset losses in summary in millions)
    #print(_quintiles.sum(level=['hazard','rp']))
    #print(summary.head())

    rp_range = [1,10,25]
    quints = [1,2,3,4,5]
    wid = 0.8
    
    # Plot losses for 'HU' & 'PF' separately:
    _quintiles = _quintiles.reset_index()
    for _fom in ['tot_asset_loss','rel_asset_loss']:
        for _d in [['HU','Wind'],['PF','Precipitation flooding']]:
            
            #### Plot absolute asset losses
            plt.cla()
        
            _quint = _quintiles.loc[_quintiles.hazard==_d[0]].copy()
            #print(_quint.head())

            if _fom == 'tot_asset_loss':
                plt.bar(quints,np.array(_quint.loc[_quint.rp==rp_range[-1],'tot_asset_loss'])/1.E9-np.array(_quint.loc[_quint.rp==rp_range[0],'tot_asset_loss']/1E9),
                        bottom=np.array(_quint.loc[_quint.rp==rp_range[0],'tot_asset_loss'].copy()/1E9),
                        width=wid,color=q_colors,alpha=0.8)

                plt.annotate(('Total '+_d[1].lower()+' damage to\nhousing & infrastructure:\n     '+str(round(_quint.loc[_quint.rp==rp_range[0],'tot_asset_loss'].sum()/1E9,1))
                              +' billion PhP\n    '+r'$-$'+str(round(np.array(_quint.loc[_quint.rp==rp_range[-1],'tot_asset_loss'].sum())/1.E9,1))+' billion PhP')
                             ,xy=(0.1,0.9),xycoords='axes fraction',ha='left',va='top',weight='bold',linespacing=1.5)
                    
            elif _fom =='rel_asset_loss':
                plt.bar(quints,np.array(100*_quint.loc[_quint.rp==rp_range[-1],'rel_asset_loss'])-np.array(100*_quint.loc[_quint.rp==rp_range[0],'rel_asset_loss']),
                        bottom=np.array(100*_quint.loc[_quint.rp==rp_range[0],'rel_asset_loss'].copy()),
                        width=wid,color=q_colors,alpha=0.8)

            for _nrp,_rp in enumerate(rp_range):
            
                if _fom == 'tot_asset_loss':
                    _rp_df = np.array(_quint.loc[_quint.rp==_rp,'tot_asset_loss']/1E9)
                elif _fom == 'rel_asset_loss':
                    _rp_df = np.array(100*_quint.loc[_quint.rp==_rp,'rel_asset_loss'].copy())

                for _nll, _loss_amount in enumerate(_rp_df):
                    if _nll == 4:
                        plt.annotate(str(_rp)+'-year\nevent',xy=((quints[_nll])+1.15*wid/2,_loss_amount),xycoords='data',weight='bold',ha='left',va='center',fontsize=6.5)

                    plt.plot([(quints[_nll])-wid/2,quints[_nll]+wid/2],[_loss_amount,_loss_amount],color=greys_pal[(_nrp+1)*2],lw=2)

            plt.xticks(quints)
            plt.gca().set_xticklabels(['Poorest\nquintile','Second','Third','Fourth','Wealthiest\nquintile'],size=10,weight='bold')
            for tick in ax.get_xaxis().get_major_ticks():
                tick.set_pad(8.)
            plt.ylim(0)

            if _fom == 'tot_asset_loss':
                plt.ylabel(_d[1]+' damage to assets [bil. PhP]',labelpad=8.,weight='bold')
            elif _fom == 'rel_asset_loss':
                plt.ylabel(_d[1]+' damage to assets [% of total assets]',labelpad=8.,weight='bold') 

            sns.despine()
            plt.grid(False)
            #plt.title('Asset losses from '+_d[1].lower()+' \nin the path of Typhoon Ompong')
        
            if _fom == 'tot_asset_loss':
                plt.gcf().savefig('/Users/brian/Desktop/Dropbox/Bank/ompong/plots/asset_losses_by_quint_abs_'+_d[0]+'.pdf',format='pdf',bbox_inches='tight')
            elif _fom == 'rel_asset_loss':
                plt.gcf().savefig('/Users/brian/Desktop/Dropbox/Bank/ompong/plots/asset_losses_by_quint_rel_'+_d[0]+'.pdf',format='pdf',bbox_inches='tight')
        
    wprime = pd.read_csv('../output_country/PH/results_tax_no_.csv', index_col=['region','hazard','rp'])['wprime'].mean()

    _regions = pd.DataFrame(index=_hh.index.copy())
    _regions['tot_asset_loss'] = _hh[['dk0','pcwgt']].prod(axis=1) 
    _regions['tot_welf_loss'] = _hh[['dw','pcwgt']].prod(axis=1)/wprime

    _regions = _regions.sum(level=['region','hazard','rp']).reset_index(['hazard','rp'])
    
    _regions['resilience'] = _regions['tot_asset_loss']/_regions['tot_welf_loss']
    _regions.to_csv('/Users/brian/Desktop/Dropbox/Bank/ompong/data/regional_impacts.csv')

    reg_x = np.array([0,1.5,3.0])
    
    for _d in [['HU','Wind'],['PF','Precipitation flooding']]:
        plt.cla()
    
        _cut = _regions.loc[_regions.hazard==_d[0]].copy()
        #print(np.array(_cut.loc[_cut.rp==_rp,'tot_asset_loss']))
        
        plt.xticks(2*reg_x+0.5)
        x_ticks = [reg.replace('II - ','').replace('I - ','') for reg in np.array(_cut.loc[_cut.rp==rp_range[0]].index.values)]

        plt.gca().set_xticklabels(x_ticks,size=10,weight='bold')
        for tick in ax.get_xaxis().get_major_ticks():
            tick.set_pad(8.)


        plt.bar(2*reg_x,(np.array(_cut.loc[_cut.rp==rp_range[-1],'tot_asset_loss']/1E9)
                         -np.array(_cut.loc[_cut.rp==rp_range[0],'tot_asset_loss']/1E9)),
                bottom=np.array(_cut.loc[_cut.rp==rp_range[0],'tot_asset_loss']/1E9),width=wid,color=reg_colorsA,alpha=0.8)
        plt.bar(2*reg_x+1,(np.array(_cut.loc[_cut.rp==rp_range[-1],'tot_welf_loss']/1E9)
                         -np.array(_cut.loc[_cut.rp==rp_range[0],'tot_welf_loss']/1E9)),
                bottom=np.array(_cut.loc[_cut.rp==rp_range[0],'tot_welf_loss']/1E9),width=wid,color=reg_colorsB,alpha=0.8)
        _yspace = float((_cut.loc[_cut.rp==rp_range[-1],'tot_asset_loss']/1E9).max())/50

        for _nrp,_rp in enumerate(rp_range):
            
            _rp_df_k = np.array(_cut.loc[_cut.rp==_rp,'tot_asset_loss']/1E9)
            _rp_df_w = np.array(_cut.loc[_cut.rp==_rp,'tot_welf_loss']/1E9)
            
            for _nll, _loss_amount in enumerate(_rp_df_k):
                #if _nll == 2:
                if _rp == rp_range[-1]:
                    plt.annotate('Assets',xy=(2*reg_x[_nll],_loss_amount+_yspace),weight='bold',va='bottom',ha='center',fontsize=6.5,rotation=0)
                plt.plot([(2*reg_x[_nll])-wid/2,2*reg_x[_nll]+wid/2],[_loss_amount,_loss_amount],color=greys_pal[(_nrp+1)*2],lw=2)

            for _nll, _loss_amount in enumerate(_rp_df_w):
                if _nll == 2:
                    plt.annotate(str(_rp)+'-year\nevent',xy=((2*reg_x[_nll])+1+1.25*wid/2,_loss_amount),xycoords='data',weight='bold',ha='left',va='center',fontsize=6.5)
                if _rp == rp_range[-1]:
                    plt.annotate('Well-being',xy=(2*reg_x[_nll]+1,_loss_amount+_yspace),weight='bold',va='bottom',ha='center',fontsize=6.5,rotation=0)

                plt.plot([(2*reg_x[_nll]+1)-wid/2,2*reg_x[_nll]+1+wid/2],[_loss_amount,_loss_amount],color=greys_pal[(_nrp+1)*2],lw=2)
    
        plt.ylabel(_d[1]+' disaster losses [bil. PhP]',weight='bold',labelpad=8)
        plt.ylim(0)
        plt.grid(False)
        sns.despine()

        plt.gcf().savefig('/Users/brian/Desktop/Dropbox/Bank/ompong/plots/asset_welf_losses_by_reg_'+_d[0]+'.pdf',format='pdf',bbox_inches='tight')
