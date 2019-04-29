import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from libraries.lib_average_over_rp import *
from libraries.lib_country_dir import get_economic_unit
from libraries.lib_common_plotting_functions import greys_pal,paired_pal

def run_urban_plots(myC,df):
    # use isrural to sort households in Bolivia
    global _upper_cut
    _upper_cut = 1.5 #times poverty_line
    #
    yor_pal = sns.color_palette('tab20b', n_colors=20)
    #
    global _colUS; global _colRS
    global _colUP; global _colRP
    global _colUN; global _colRN
    global _colUW; global _colRW
    _colUS = yor_pal[12]; _colRS = yor_pal[15]
    _colUP = yor_pal[8];  _colRP = yor_pal[11]
    _colUN = yor_pal[4];  _colRN = yor_pal[7]
    _colUW = yor_pal[0];  _colRW = yor_pal[3]
    #
    global _colUV; global _colRV
    _colUV = yor_pal[0]; _colRV = yor_pal[3]
    #
    global _alphaU; global _alphaR
    _alphaU = 0.50; _alphaR = 0.8
    #
    global _e
    _e = get_economic_unit(myC)
        
    if 'isrural' not in df.columns: return False    
    else: 
        # plot poverty gap, by urban/rural, & by department
        df = df.reset_index().set_index([_e,'ispoor','isrural'])[['pcwgt','totper','c','pcsoc','has_ew','pov_line','sub_line','issub']]
        #
        social_scatter(myC,df,'rel')
        social_scatter(myC,df,'abs')
        #
        poverty_gap(myC,df)
        #
        for plot_rural in [True, False]:
            populations(myC,df,'rel',plot_rural)
            populations(myC,df,'abs',plot_rural)
        #
  
    # this code shouldn't be here!
    # we're analyzing model results in a function that gets called in gather_data
    # but it's easiest...
    #try: 
    model_results_plots(myC,'assets','abs',('PF','aal'),redo_intermediate=True)
    model_results_plots(myC,'assets','rel',('PF','aal'))
    model_results_plots(myC,'assets','tot',('PF','aal'))
    model_results_plots(myC,'wellbeing','abs',('PF','aal'))
    model_results_plots(myC,'wellbeing','rel',('PF','aal'))
    model_results_plots(myC,'wellbeing','tot',('PF','aal'))
    model_results_plots(myC,'resilience','',('PF','aal'))
    
    model_results_plots(myC,'assets','abs',('PF',50),redo_intermediate=True)
    model_results_plots(myC,'assets','rel',('PF',50))
    model_results_plots(myC,'assets','tot',('PF',50))
    model_results_plots(myC,'wellbeing','abs',('PF',50))
    model_results_plots(myC,'wellbeing','rel',('PF',50))
    model_results_plots(myC,'wellbeing','tot',('PF',50))
    model_results_plots(myC,'resilience','',('PF',50))
    #except: pass

def model_results_plots(myC,fom,ra_switch='abs',event_switch=('PF','aal'),redo_intermediate=False):
    if fom == 'resilience' and ra_switch != '': return True

    _haz,_rp = event_switch

    ##########################
    # post-model run plots
    try: 
        if redo_intermediate: assert(False)
        df = pd.read_csv('../output_country/BO/urban_intermediate.csv').set_index([_e]).sort_index()
    except: 
        print('did NOT work!')
        df = pd.read_csv('../output_country/BO/iah_tax_no_.csv').set_index(['hhid']).sort_index()
        _wprime = pd.read_csv('../output_country/BO/results_tax_no_.csv', index_col=[_e]).wprime.mean()
        #
        _cut = "(pcwgt!=0)&(hazard=='"+str(_haz)+"')"
        _sumto = ['hhid','rp'] 
        if str(_rp) != 'aal': 
            _cut += "&(rp=="+str(_rp)+")&(affected_cat=='a')"
            _sumto = 'hhid'
        #
        df = df.loc[df.eval(_cut)].reset_index().set_index(_sumto)
        #
        _df = (df.eval('pcwgt*c').sum(level=_sumto)
               /df.eval('pcwgt').sum(level=_sumto)).to_frame(name='c')
        _df['dk'] = (df.eval('pcwgt*dk0').sum(level=_sumto)
                     /df.eval('pcwgt').sum(level=_sumto))
        _df['dw'] = (df.eval('pcwgt*dw').sum(level=_sumto)/_wprime
                     /df.eval('pcwgt').sum(level=_sumto))
        _df['pcwgt'] = df['pcwgt'].sum(level=_sumto)
        #
        if _rp == 'aal': 
            _df = _df.groupby(['hhid']).apply(lambda x:average_over_rp(x,return_probs=False).squeeze()).reset_index().set_index('hhid')
        df = df.reset_index().set_index('hhid')

        _df = _df.join(df.loc[~df.index.duplicated(keep='first'),['isrural','ispoor','issub','pov_line',_e]],how='left')
        _df.to_csv('../output_country/BO/urban_intermediate.csv')
        #
        # rename to link this with existing code
        df = _df.dropna().reset_index().set_index(_e)
    #
    #
    plt.figure(figsize=(12,6))
    #
    if fom == 'resilience':
        df['_'] = df.eval('100*pcwgt*dk')
        df['__'] = df.eval('pcwgt*dw')
    elif fom == 'assets':
        if ra_switch == 'abs': df['_'] = df.eval('pcwgt*dk')
        if ra_switch == 'rel': df['_'] = df.eval('1E2*pcwgt*dk/c')
        df['__'] = df.eval('pcwgt')
    elif fom == 'wellbeing':
        if ra_switch == 'abs': df['_'] = df.eval('pcwgt*dw')
        if ra_switch == 'rel': df['_'] = df.eval('1E2*pcwgt')*df.eval('dw/c').clip(upper=1000)
        df['__'] = df.eval('pcwgt')
    if ra_switch == 'tot': 
        if fom == 'assets': df['_'] = df.eval('1E-6*pcwgt*dk')
        if fom == 'wellbeing': df['_'] = df.eval('1E-6*pcwgt*dw')
        df['__'] = None

    __df_slice = (df.loc[df.eval('~(isrural)'),'_'].sum(level=_e)
                  /(df.loc[df.eval('~(isrural)'),'__'].sum(level=_e) if ra_switch!='tot' else 1)).to_frame(name='avg_urban').sort_values(by='avg_urban',ascending=True)
    #__df_slice['avg_resil_rur_vul'] = (df.loc[df.eval('(isrural)&(c<=@_upper_cut*pov_line)'),'_'].sum(level=_e)
    #                                   /(df.loc[df.eval('(isrural)&(c<=@_upper_cut*pov_line)'),'__'].sum(level=_e) if ra_switch!='tot' else 1))
    
    # Populations subsets
    sets = [#['Rural extreme poor','(isrural)&(issub)',_colRS],
            #['Rural poor'        ,'(isrural)&~(issub)&(ispoor)',_colRP],        
            #['Rural near-poor'   ,'(isrural)&~(ispoor)&(c<=@_upper_cut*pov_line)',_colRN],   
            #['Rural non-poor'    ,'(isrural)&(c>@_upper_cut*pov_line)',_colRW],    
            ['Urban extreme poor','~(isrural)&(issub)',_colUS],
            ['Urban poor'        ,'~(isrural)&~(issub)&(ispoor)',_colUP],
            ['Urban near-poor'   ,'~(isrural)&~(ispoor)&(c<=@_upper_cut*pov_line)',_colUN],
            ['Urban non-poor'    ,'~(isrural)&(c>@_upper_cut*pov_line)',_colUW]]
    _nsets = int(round(float(len(sets))*1.5,0))
    _gap = int(_nsets/4); _n2 = 0
    
    for _n, _cut in enumerate(sets):
        _df_slice = df.loc[df.eval(_cut[1])].copy()    
        _df_slice = __df_slice.join((_df_slice['_'].sum(level=_e)/(_df_slice['__'].sum(level=_e) if ra_switch!='tot' else 1)).to_frame(name='_'),how='left')
        plt.bar([float(_)*(_nsets+_gap)+_n*(1+1/(_nsets+2)) for _ in range(_df_slice.shape[0])],_df_slice['_'],color=_cut[2],zorder=95,label=_cut[0],alpha=0.85)
        
        # ticks
        if _n == 0:
            plt.xticks([float(_)*(_nsets+_gap)+_n*(1+1/(_nsets+2)+_n2)-1/2 for _ in range(_df_slice.sum(level=_e).shape[0])])
            plt.gca().set_xticklabels(labels=_df_slice.sum(level=_e).index.values,ha='left',fontsize=8)
            plt.tick_params(axis='x', which='major',color=greys_pal[4])
            plt.tick_params(axis='y', which='major',color=greys_pal[4],labelsize=8)
        
        _n += 1
        if _n>= 0.5*len(sets) and _n2 == 0: _n2 = 0.75

    # national average
    if ra_switch != 'tot':
        _xlim = plt.gca().get_xlim()
        plt.plot(_xlim,[(df['_'].sum()/(df['__'].sum() if ra_switch!='tot' else 1)),(df['_'].sum()/(df['__'].sum() if ra_switch!='tot' else 1))],color=greys_pal[8],alpha=0.85,zorder=96,lw=0.5,ls=':')
        plt.xlim(_xlim)
        plt.annotate('National average',xy=(_xlim[0]+0.7,1.005*df['_'].sum()/(df['__'].sum() if ra_switch!='tot' else 1)),ha='left',va='bottom',color=greys_pal[5],fontsize=8,clip_on=False,zorder=99)

    sns.despine()
    plt.grid(False)
    plt.gca().legend(loc='best',labelspacing=0.75,ncol=1,fontsize=8,borderpad=0.75,fancybox=True,frameon=False)

    plt.tick_params(axis='y', which='major', labelsize=8)    
    if fom == 'resilience': 
        plt.ylabel('Socioeconomic resilience (%)',fontsize=8,labelpad=10,linespacing=1.5)
        plt.ylim(0)#,60)
    if fom == 'assets': 
        if _rp == 'aal':
            if ra_switch == 'abs': plt.ylabel('Asset risk\n(expected annual losses to floods in Bs. per capita)',fontsize=8,labelpad=10,linespacing=1.5)
            if ra_switch == 'rel': plt.ylabel('Asset risk\n(expected annual losses to floods as % of total consumption)',fontsize=8,labelpad=10,linespacing=1.5)
            if ra_switch == 'tot': plt.ylabel('Average annual asset losses (mil. Bs.)',fontsize=8,labelpad=10,linespacing=1.5)
        else:
            if ra_switch == 'abs': plt.ylabel('Asset losses in '+str(_rp)+'-year flood (Bs. per capita)',fontsize=8,labelpad=10,linespacing=1.5)
            if ra_switch == 'rel': plt.ylabel('Asset losses in '+str(_rp)+'-year flood (as % of total consumption)',fontsize=8,labelpad=10,linespacing=1.5)
            if ra_switch == 'tot': plt.ylabel('Total asset losses in '+str(_rp)+'-year flood (mil. Bs.)',fontsize=8,labelpad=10,linespacing=1.5)
    if fom == 'wellbeing': 
        if _rp == 'aal':
            if ra_switch == 'abs': plt.ylabel('Wellbeing risk\n(expected annual losses to floods in Bs. per capita)',fontsize=8,labelpad=10,linespacing=1.5)
            if ra_switch == 'rel': plt.ylabel('Wellbeing risk\n(expected annual losses to floods as % of total consumption)',fontsize=8,labelpad=10,linespacing=1.5)
            if ra_switch == 'tot': plt.ylabel('Average annual wellbeing losses (mil. Bs.)',fontsize=8,labelpad=10,linespacing=1.5)
        else: 
            if ra_switch == 'abs': plt.ylabel('Wellbeing losses in '+str(_rp)+'-year flood (Bs. per capita)',fontsize=8,labelpad=10,linespacing=1.5)
            if ra_switch == 'rel': plt.ylabel('Wellbeing losses in '+str(_rp)+'-year flood (as % of total consumption)',fontsize=8,labelpad=10,linespacing=1.5)
            if ra_switch == 'tot': plt.ylabel('Total wellbeing losses in '+str(_rp)+'-year flood (mil. Bs.)',fontsize=8,labelpad=10,linespacing=1.5)
        plt.ylim(0)

    fig = plt.gcf()
    fig.savefig(('../output_plots/'+myC+'/urban_'+fom
                 +('_' if ra_switch !='' else '')+ra_switch
                 +'_'+('aal' if _rp == 'aal' else str(_haz)+str(_rp))+'.pdf'),format='pdf',bbox_inches='tight')
    plt.close('all')
    return True

    

def social_scatter(myC,df,ra_switch='rel'):
    df = df.sort_index()
    
    plt.figure(figsize=(12,6))

    # df2 will be department-level averages for rich/poor & urban/rural
    df2 = pd.DataFrame(index=df.sum(level=[_e,'isrural','ispoor']).index).sort_index()
    for _c in ['totper','c','pcsoc','has_ew','pov_line','sub_line']:
        df2[_c] = df[['pcwgt',_c]].prod(axis=1).sum(level=[_e,'isrural','ispoor'])/df['pcwgt'].sum(level=[_e,'isrural','ispoor'])    
    df2 = df2.reset_index(['isrural','ispoor'])

    # Scatter plot, not interesting
    #df = df.reset_index()
    #_cut = '~(isrural)&(issub)&(pcsoc>0)'  
    #plt.scatter(1E-3*df.loc[df.eval(_cut),'c'],1E-3*df.loc[df.eval(_cut),'pcsoc'],alpha=_alphaU,color=_colUS,s=5,zorder=98)
    #_cut = '~(isrural)&(ispoor)&~(issub)&(pcsoc>0)'  
    #plt.scatter(1E-3*df.loc[df.eval(_cut),'c'],1E-3*df.loc[df.eval(_cut),'pcsoc'],alpha=_alphaU,color=_colUP,s=5,zorder=96)
    #_cut = '~(isrural)&(c<=@_upper_cut*pov_line)&(pcsoc>0)&~(ispoor)'    
    #plt.scatter(1E-3*df.loc[df.eval(_cut),'c'],1E-3*df.loc[df.eval(_cut),'pcsoc'],alpha=_alphaU,color=_colUN,s=5,zorder=94)
    _nsets = 8; _gap = 4.
    _n = 0; _n2 = 0

    if ra_switch == 'abs': df['_'] = df.eval('pcwgt*pcsoc')
    else: df['_'] = df.eval('100*pcwgt*pcsoc/c')

    __df_slice = (df.loc[df.eval('~(isrural)&(c<=@_upper_cut*pov_line)'),'_'].sum(level=_e)
                  /df.loc[df.eval('~(isrural)&(c<=@_upper_cut*pov_line)'),'pcwgt'].sum(level=_e)).to_frame(name='avg_pcsoc_urb_vul').sort_values(by='avg_pcsoc_urb_vul',ascending=True)
    __df_slice['avg_pcsoc_rur_vul'] = (df.loc[df.eval('(isrural)&(c<=@_upper_cut*pov_line)'),'_'].sum(level=_e)
                                       /df.loc[df.eval('(isrural)&(c<=@_upper_cut*pov_line)'),'pcwgt'].sum(level=_e))
    

    # rural, extreme poverty
    _cut = '(isrural)&(issub)'
    _df_slice = df.loc[df.eval(_cut)].copy()    
    _df_slice = __df_slice.join((_df_slice['_'].sum(level=_e)/_df_slice['pcwgt'].sum(level=_e)).to_frame(name='_'),how='left')
    plt.bar([float(_)*(_nsets+_gap)+_n*(1+1/(_nsets+2)) for _ in range(_df_slice.shape[0])],_df_slice['_'],color=_colRS,zorder=95,label='Rural extreme poor')
        
    # average value of rural transfers
    for _n_mean, _d_mean in enumerate(__df_slice['avg_pcsoc_rur_vul']):
        plt.plot([float(_n_mean)*(_nsets+_gap)+_n*(1+1/(_nsets+2))-1/2,float(_n_mean)*(_nsets+_gap)+(_n+3)*(1+1/(_nsets+2))-1/2],[_d_mean,_d_mean],
                 color=greys_pal[8],alpha=0.85,zorder=96,ls=':',lw=0.5)

    # ticks
    plt.xticks([float(_)*(_nsets+_gap)+_n*(1+1/(_nsets+2)+_n2)-1/2 for _ in range(_df_slice.sum(level=_e).shape[0])])
    plt.gca().set_xticklabels(labels=_df_slice.sum(level=_e).index.values,ha='left',fontsize=8)
    plt.tick_params(axis='x', which='major',color=greys_pal[4])
    plt.tick_params(axis='y', which='major',color=greys_pal[4],labelsize=8)
    _n += 1

    # rural, poverty
    _cut = '(isrural)&~(issub)&(ispoor)'
    _df_slice = df.loc[df.eval(_cut)].copy()
    _df_slice = __df_slice.join((_df_slice['_'].sum(level=_e)/_df_slice['pcwgt'].sum(level=_e)).to_frame(name='_'),how='left')
    plt.bar([float(_)*(_nsets+_gap)+_n*(1+1/(_nsets+2)) for _ in range(_df_slice.shape[0])],_df_slice['_'],color=_colRP,zorder=95,label='Rural poor')
    _n += 1

    # rural, near-poor
    _cut = '(isrural)&~(issub)&~(ispoor)&(c<=@_upper_cut*pov_line)'
    _df_slice = df.loc[df.eval(_cut)].copy()
    _df_slice = __df_slice.join((_df_slice['_'].sum(level=_e)/_df_slice['pcwgt'].sum(level=_e)).to_frame(name='_'),how='left')
    plt.bar([float(_)*(_nsets+_gap)+_n*(1+1/(_nsets+2)) for _ in range(_df_slice.shape[0])],_df_slice['_'],color=_colRN,zorder=95,label='Rural near-poor')
    _n += 1; _n2+= 0.75

    # urban, extreme poverty
    _cut = '~(isrural)&(issub)'
    _df_slice = df.loc[df.eval(_cut)].copy()
    _df_slice = __df_slice.join((_df_slice['_'].sum(level=_e)/_df_slice['pcwgt'].sum(level=_e)).to_frame(name='_'),how='left')
    plt.bar([float(_)*(_nsets+_gap)+_n*(1+1/(_nsets+2))+_n2 for _ in range(_df_slice.shape[0])],_df_slice['_'],color=_colUS,zorder=95,label='Urban extreme poor')

    # average value of urban transfers
    for _n_mean, _d_mean in enumerate(__df_slice['avg_pcsoc_urb_vul']):
        plt.plot([float(_n_mean)*(_nsets+_gap)+_n*(1+1/(_nsets+2))-1/2+_n2,float(_n_mean)*(_nsets+_gap)+(_n+3)*(1+1/(_nsets+2))-1/2+_n2],[_d_mean,_d_mean],
                 color=greys_pal[8],alpha=0.85,zorder=96,lw=0.5,ls=':')
    _n += 1

    # urban, poverty
    _cut = '~(isrural)&~(issub)&(ispoor)'
    _df_slice = df.loc[df.eval(_cut)].copy()
    _df_slice = __df_slice.join((_df_slice['_'].sum(level=_e)/_df_slice['pcwgt'].sum(level=_e)).to_frame(name='_'),how='left')
    plt.bar([float(_)*(_nsets+_gap)+_n*(1+1/(_nsets+2))+_n2 for _ in range(_df_slice.shape[0])],_df_slice['_'],color=_colUP,zorder=95,label='Urban poor')
    _n += 1

    # urban, near-poor
    _cut = '~(isrural)&~(issub)&~(ispoor)&(c<=@_upper_cut*pov_line)'
    _df_slice = df.loc[df.eval(_cut)].copy()
    _df_slice = __df_slice.join((_df_slice['_'].sum(level=_e)/_df_slice['pcwgt'].sum(level=_e)).to_frame(name='_'),how='left')
    plt.bar([float(_)*(_nsets+_gap)+_n*(1+1/(_nsets+2))+_n2 for _ in range(_df_slice.shape[0])],_df_slice['_'],color=_colUN,zorder=95,label='Urban near-poor')
    _n += 1


    # national average
    _xlim = plt.gca().get_xlim()
    plt.plot(_xlim,[(df['_'].sum()/df['pcwgt'].sum()),(df['_'].sum()/df['pcwgt'].sum())],color=greys_pal[8],alpha=0.85,zorder=96,lw=0.5,ls=':')
    plt.xlim(_xlim)
    plt.annotate('National average',xy=(_xlim[0]+0.7,1.005*df['_'].sum()/df['pcwgt'].sum()),ha='left',va='bottom',color=greys_pal[5],fontsize=8,clip_on=False,zorder=99)

    sns.despine()
    plt.grid(False)
    plt.gca().legend(loc='upper right',labelspacing=0.75,ncol=1,fontsize=8,borderpad=0.75,fancybox=True,frameon=False)

    plt.tick_params(axis='y', which='major', labelsize=8)
    if ra_switch == 'abs': 
        plt.ylabel('Social transfers, per capita (Bs.)',fontsize=8,labelpad=10,linespacing=1.5)
        plt.ylim(0,2500)
    else: 
        plt.ylabel('Social transfers (% of total income)',fontsize=8,labelpad=10,linespacing=1.5)
        plt.ylim(0,45)        

    fig = plt.gcf()
    fig.savefig('../output_plots/'+myC+'/urban_social_'+ra_switch+'.pdf',format='pdf',bbox_inches='tight')
    plt.close('all')

    return True


def populations(myC,df,ra_switch='rel',plot_rural=True):

    # df2 will be department-level averages for rich/poor & urban/rural
    df2 = pd.DataFrame(index=df.sum(level=[_e,'isrural','ispoor']).index).sort_index()        
    for _c in ['totper','c','pcsoc','has_ew','pov_line','sub_line']:
        df2[_c] = df[['pcwgt',_c]].prod(axis=1).sum(level=[_e,'isrural','ispoor'])/df['pcwgt'].sum(level=[_e,'isrural','ispoor'])
    df2['pov_gap'] = df2.eval('100.*(c/pov_line)')

    ## df is indexed by department
    df = df.reset_index(['ispoor','isrural'])
    df2 = df2.reset_index()

    # Poverty gap -- poor urban vs rural
    _df = pd.DataFrame(index=df2.set_index(_e).sum(level=_e).index)
    #
    _df['total_pop'] = df['pcwgt'].sum(level=_e)/(1E-2*df['pcwgt'].sum(level=_e) if ra_switch == 'rel' else 1E3)
    _df['total_vulnerable_pop'] = df.loc[df.eval('(c<=@_upper_cut*pov_line)'),'pcwgt'].sum(level=_e)/(1E-2*df['pcwgt'].sum(level=_e) if ra_switch == 'rel' else 1E3)
    _df['total_vulnerable_pop_urban'] = df.loc[df.eval('~(isrural)&(c<=@_upper_cut*pov_line)'),'pcwgt'].sum(level=_e)/(1E-2*df['pcwgt'].sum(level=_e) if ra_switch == 'rel' else 1E3)
    ##################################################
    _df['pop_sub_urban'] = (df.loc[df.eval('(ispoor)&~(isrural)&(issub)')].eval('pcwgt').sum(level=_e)
                            /(1E-2*df['pcwgt'].sum(level=_e) if ra_switch == 'rel' else 1E3))
    _df['pop_sub_rural'] = (df.loc[df.eval('(ispoor)&(isrural)&(issub)')].eval('pcwgt').sum(level=_e)
                            /(1E-2*df['pcwgt'].sum(level=_e) if ra_switch == 'rel' else 1E3))
    ##################################################
    _df['pop_poor_urban'] = (df.loc[df.eval('(ispoor)&~(isrural)&~(issub)')].eval('pcwgt').sum(level=_e)
                             /(1E-2*df['pcwgt'].sum(level=_e) if ra_switch == 'rel' else 1E3))
    _df['pop_poor_rural'] = (df.loc[df.eval('(ispoor)&(isrural)&~(issub)')].eval('pcwgt').sum(level=_e)
                             /(1E-2*df['pcwgt'].sum(level=_e) if ra_switch == 'rel' else 1E3))
    ##################################################
    _df['pop_nonpoor_urban'] = (df.loc[df.eval('~(ispoor)&~(isrural)&(c<=@_upper_cut*pov_line)')].eval('pcwgt').sum(level=_e)
                                /(1E-2*df['pcwgt'].sum(level=_e) if ra_switch == 'rel' else 1E3))
    _df['pop_nonpoor_rural'] = (df.loc[df.eval('~(ispoor)&(isrural)&(c<=@_upper_cut*pov_line)')].eval('pcwgt').sum(level=_e)
                                /(1E-2*df['pcwgt'].sum(level=_e) if ra_switch == 'rel' else 1E3))
    ##################################################
    _df['pop_rich_urban'] = (df.loc[df.eval('~(ispoor)&~(isrural)&(c>@_upper_cut*pov_line)')].eval('pcwgt').sum(level=_e)
                             /(1E-2*df['pcwgt'].sum(level=_e) if ra_switch == 'rel' else 1E3))
    _df['pop_rich_rural'] = (df.loc[df.eval('~(ispoor)&(isrural)&(c>@_upper_cut*pov_line)')].eval('pcwgt').sum(level=_e)
                             /(1E-2*df['pcwgt'].sum(level=_e) if ra_switch == 'rel' else 1E3))
    ##################################################
    #
    if plot_rural: _df = _df.sort_values('total_vulnerable_pop',ascending=True)
    else: _df = _df.sort_values('total_vulnerable_pop_urban',ascending=True)

    _xvals = [_ for _ in range(_df.shape[0])]
    _xoffset = +0.15
    _yoffset = -0.5
    __tmp = [0 for _ in range(len(_xvals))]; _tmp = __tmp.copy()
    #
    if plot_rural:
        _tmp = _df['pop_sub_rural'].squeeze()
        plt.plot(_xvals,__tmp,_tmp,clip_on=False,label='Rural extreme poor',alpha=_alphaR,color=_colRS)
        plt.fill_between(_xvals,__tmp,_tmp,color=_colRS,alpha=_alphaR)
        plt.annotate('Rural extreme poor',xy=(_xvals[-1]+_xoffset,_tmp[-1]+ _yoffset),fontsize=7.5,color=_colRS,va='top',ha='left',style='italic')

        __tmp = _tmp.copy(); _tmp += _df['pop_poor_rural'].squeeze()
        plt.plot(_xvals,_tmp,clip_on=False,label='Rural poor',alpha=_alphaR,color=_colRP)
        plt.fill_between(_xvals,__tmp,_tmp,color=_colRP,alpha=_alphaR)
        plt.annotate('Rural poor',xy=(_xvals[-1]+_xoffset,_tmp[-1]+ _yoffset),fontsize=7.5,color=_colRP,va='top',ha='left',style='italic') 

        __tmp = _tmp.copy(); _tmp += _df['pop_nonpoor_rural'].squeeze()
        plt.plot(_xvals,_tmp,clip_on=False,label='Rural near-poor',alpha=_alphaR,color=_colRN)
        plt.fill_between(_xvals,__tmp,_tmp,color=_colRN,alpha=_alphaR)
        plt.annotate('Rural near-poor',xy=(_xvals[-1]+_xoffset,_tmp[-1]+_yoffset),fontsize=7.5,color=_colRN,va='top',ha='left',style='italic')

        __tmp = _tmp.copy(); _tmp += _df['pop_rich_rural'].squeeze()
        plt.plot(_xvals,_tmp,clip_on=False,label='Rural wealthy',alpha=_alphaR,color=_colRW)
        plt.fill_between(_xvals,__tmp,_tmp,color=_colRW,alpha=_alphaR)
        plt.annotate('Rural wealthy',xy=(_xvals[-1]+_xoffset,_tmp[-1]+_yoffset),fontsize=7.5,color=_colRW,va='top',ha='left',style='italic')

    __tmp = _tmp.copy(); _tmp += _df['pop_sub_urban'].squeeze()
    plt.fill_between(_xvals,__tmp,_tmp,color=_colUS,alpha=_alphaU)
    plt.plot(_xvals,_tmp,clip_on=False,label='Urban extreme poor',alpha=_alphaU,color=_colUS)
    plt.annotate('Urban extreme poor',xy=(_xvals[-1]+_xoffset,_tmp[-1]+ _yoffset),fontsize=7.5,color=_colUS,va='top',ha='left',style='italic')
    #
    __tmp = _tmp.copy(); _tmp += _df['pop_poor_urban'].squeeze()
    plt.plot(_xvals,_tmp,clip_on=False,label='Urban poor',alpha=_alphaU,color=_colUP)
    plt.fill_between(_xvals,__tmp,_tmp,color=_colUP,alpha=_alphaU)
    plt.annotate('Urban poor',xy=(_xvals[-1]+_xoffset,_tmp[-1]+ _yoffset),fontsize=7.5,color=_colUP,va='top',ha='left',style='italic')
    #
    __tmp = _tmp.copy(); _tmp += _df['pop_nonpoor_urban'].squeeze()
    plt.plot(_xvals,_tmp,clip_on=False,label='Urban near-poor',alpha=_alphaU,color=_colUN)
    plt.fill_between(_xvals,__tmp,_tmp,color=_colUN,alpha=_alphaU)
    plt.annotate('Urban near-poor',xy=(_xvals[-1]+_xoffset,_tmp[-1]+_yoffset),fontsize=7.5,color=_colUN,va='top',ha='left',style='italic')
    #
    __tmp = _tmp.copy(); _tmp += _df['pop_rich_urban'].squeeze()
    plt.plot(_xvals,_tmp,clip_on=False,label='Urban wealthy',alpha=_alphaU,color=_colUW)
    plt.fill_between(_xvals,__tmp,_tmp,color=_colUW,alpha=_alphaU)
    plt.annotate('Urban wealthy',xy=(_xvals[-1]+_xoffset,_tmp[-1]+_yoffset),fontsize=7.5,color=_colUW,va='top',ha='left',style='italic')   
    #
    _xlim = plt.gca().get_xlim()    
    _ylim = plt.gca().get_ylim()
    #
    plt.xticks([_ for _ in range(len(_df.index))])
    plt.gca().set_xticklabels(labels=_df.index.values,rotation=45,ha='right',fontsize=8)
    plt.tick_params(axis='x', which='major',color=greys_pal[4])
    
    for _ in range(len(_df.index)):
        plt.plot([_,_],_ylim,color=greys_pal[5],alpha=0.8,ls=':',lw=0.5,zorder=99)
        
    plt.tick_params(axis='y', which='major', labelsize=8)
    plt.ylim(0)
    if ra_switch == 'rel': _ylabelstr = 'Urban'+(' & rural' if plot_rural else '')+' households, by poverty status\n(% of total population, stacked)'
    else: _ylabelstr = 'Urban'+(' & rural' if plot_rural else '')+' households, by poverty status\n(,000 & stacked)'
    plt.ylabel(_ylabelstr,fontsize=8,labelpad=10,linespacing=1.5)
    
    sns.despine()
    plt.grid(False)
    
    fig = plt.gcf()
    fig.savefig('../output_plots/'+myC+'/urban_populations_'+ra_switch+('_with_rural' if plot_rural else '')+'.pdf',format='pdf',bbox_inches='tight')
    plt.close('all')
    return True
    

    
def poverty_gap(myC,df):
    # use isrural to sort households in Bolivia

    # df2 will be department-level averages for rich/poor & urban/rural
    df2 = pd.DataFrame(index=df.sum(level=[_e,'isrural','ispoor']).index).sort_index()        
    for _c in ['totper','c','pcsoc','has_ew','pov_line','sub_line']:
        df2[_c] = df[['pcwgt',_c]].prod(axis=1).sum(level=[_e,'isrural','ispoor'])/df['pcwgt'].sum(level=[_e,'isrural','ispoor'])
    df2['pov_gap'] = df2.eval('100.*(c/pov_line)')

    ## df is indexed by department
    df = df.reset_index(['ispoor','isrural'])
    df2 = df2.reset_index()

    # Poverty gap -- poor urban vs rural
    _df = df2.loc[df2.eval('(ispoor)&~(isrural)')].set_index(_e)['pov_gap'].to_frame(name='PG_poor_urban')
    #_df['PG_poor_rural'] = df2.loc[df2.eval('(ispoor)&(isrural)'].set_index(_e)['pov_gap']
    #

    ##################################################
    _df['PG_sub_urban'] = 100.*(df.loc[df.eval('(ispoor)&~(isrural)&(issub)')].eval('(c/pov_line)*pcwgt').sum(level=_e)
                                /df.loc[df.eval('(ispoor)&~(isrural)&(issub)'),'pcwgt'].sum(level=_e))
        
    _df['PG_sub_rural'] = 100.*(df.loc[df.eval('(ispoor)&(isrural)&(issub)')].eval('(c/pov_line)*pcwgt').sum(level=_e)
                                /df.loc[df.eval('(ispoor)&(isrural)&(issub)'),'pcwgt'].sum(level=_e))      
    ##################################################
    _df['PG_poor_urban'] = 100.*(df.loc[df.eval('(ispoor)&~(isrural)&~(issub)')].eval('(c/pov_line)*pcwgt').sum(level=_e)
                                 /df.loc[df.eval('(ispoor)&~(isrural)&~(issub)'),'pcwgt'].sum(level=_e))
    
    _df['PG_poor_rural'] = 100.*(df.loc[df.eval('(ispoor)&(isrural)&~(issub)')].eval('(c/pov_line)*pcwgt').sum(level=_e)
                                  /df.loc[df.eval('(ispoor)&(isrural)&~(issub)'),'pcwgt'].sum(level=_e))     
    ##################################################
    _df['PG_nonpoor_urban'] = 100.*(df.loc[df.eval('~(ispoor)&~(isrural)&(c<=@_upper_cut*pov_line)')].eval('(c/pov_line)*pcwgt').sum(level=_e)
                                    /df.loc[df.eval('~(ispoor)&~(isrural)&(c<=@_upper_cut*pov_line)'),'pcwgt'].sum(level=_e))
    
    _df['PG_nonpoor_rural'] = 100.*(df.loc[df.eval('~(ispoor)&(isrural)&(c<=@_upper_cut*pov_line)')].eval('(c/pov_line)*pcwgt').sum(level=_e)
                                    /df.loc[df.eval('~(ispoor)&(isrural)&(c<=@_upper_cut*pov_line)'),'pcwgt'].sum(level=_e))
    ##################################################
    #
    _df = _df.sort_values('PG_sub_urban',ascending=True)
    _xvals = [_ for _ in range(_df.shape[0])]
    _xoffset = -0.15
    _yoffset = 4.25
    #
    plt.scatter(_xvals, _df['PG_sub_rural'],clip_on=False,label='Rural extreme poor',alpha=0.7,color=_colRS)
    plt.scatter(_xvals, _df['PG_sub_urban'],clip_on=False,label='Urban extreme poor',alpha=0.7,color=_colUS)
    #plt.annotate('Extreme poor',xy=(_xvals[-1],_df.iloc[-1]['PG_sub_rural']),xytext=(1.02*_xvals[-1],_df.iloc[-1]['PG_sub_rural']),fontsize=7,color=greys_pal[8])
    plt.annotate('Extreme poor',xy=(_xvals[0]+_xoffset,_df['PG_sub_urban'].max()+ _yoffset),fontsize=7.5,color=_colUS,va='center',ha='left',style='italic')
    #
    plt.scatter(_xvals, _df['PG_poor_rural'],clip_on=False,label='Rural poor',alpha=0.7,color=_colRP)
    plt.scatter(_xvals, _df['PG_poor_urban'],clip_on=False,label='Urban poor',alpha=0.7,color=_colUP)
    plt.annotate('Poor',xy=(_xvals[0]+_xoffset,_df['PG_poor_urban'].max()+ _yoffset),fontsize=7.5,color=_colUP,va='center',ha='left',style='italic')
    #
    plt.scatter(_xvals, _df['PG_nonpoor_rural'],clip_on=False,label='Rural near-poor',alpha=0.7,color=_colRN)
    plt.scatter(_xvals, _df['PG_nonpoor_urban'],clip_on=False,label='Urban near-poor',alpha=0.7,color=_colUN)
    plt.annotate('Near-poor',xy=(_xvals[0]+_xoffset,_df['PG_nonpoor_urban'].max()+_yoffset),fontsize=7.5,color=_colUN,va='center',ha='left',style='italic')
    
    plt.annotate('Urban',xy=(_xvals[-1],_df.iloc[-1]['PG_nonpoor_urban']),xytext=(_xvals[-1]+0.35,_df.iloc[-1]['PG_nonpoor_urban']+3.0),
                 arrowprops=dict(arrowstyle="-",facecolor=greys_pal[5],connectionstyle="angle,angleA=0,angleB=-90,rad=5"),
                 fontsize=7.5,color=greys_pal[6],va='center',ha='left',style='italic')
    
    plt.annotate('Rural',xy=(_xvals[-1],_df.iloc[-1]['PG_nonpoor_rural']),xytext=(_xvals[-1]+0.35,_df.iloc[-1]['PG_nonpoor_rural']-3.0),
                 arrowprops=dict(arrowstyle="-",facecolor=greys_pal[5],connectionstyle="angle,angleA=0,angleB=-90,rad=5"),
                 fontsize=7.5,color=greys_pal[6],va='center',ha='left',style='italic')
    #
    _xlim = plt.gca().get_xlim()
    plt.plot(_xlim,[100.,100.],lw=0.75,ls='-',color=greys_pal[3],alpha=0.75)
    plt.xlim(_xlim)
    
    plt.xticks([_ for _ in range(len(_df.index))])
    plt.gca().set_xticklabels(labels=_df.index.values,rotation=45,ha='right',fontsize=8)
    plt.tick_params(axis='x', which='major',color=greys_pal[4])
    
    for _ in range(len(_df.index)):
        plt.plot([_,_],[20,120],color=greys_pal[4],alpha=0.2,ls=':',lw=0.5)
        
    plt.tick_params(axis='y', which='major', labelsize=8)
    plt.ylim(20,120)
    plt.ylabel('Per capita income, as % of poverty line',fontsize=8,labelpad=10,linespacing=1.5)
    
    sns.despine(bottom=True)
    plt.grid(False)
    
    fig = plt.gcf()
    fig.savefig('../output_plots/'+myC+'/urban_poverty_gap.pdf',format='pdf',bbox_inches='tight')
    plt.close('all')
    return True
    
