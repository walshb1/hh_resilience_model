import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy.random as nprand
import seaborn as sns
from libraries.lib_common_plotting_functions import greys_pal, title_legend_labels
from libraries.lib_average_over_rp import *
from libraries.maps_lib import *
#from libraries.lib_country_dir import get_all_hazards

event_level = ['district','rp']#['district','hazard','rp']


def get_aal(myC='MW'):
    df = pd.read_csv('../output_country/'+myC+'/out_ag.csv').set_index(event_level+['hhid'])

    df_dist = pd.DataFrame(index=df.sum(level=event_level).index)
    for _c in ['di_ag','dw_ag_currency']:
        df_dist[_c] = df[['pcwgt','fa_ag',_c]].prod(axis=1).sum(level=event_level)
    for _c in ['c','v_ag']:
        df_dist[_c] = df[['pcwgt',_c]].prod(axis=1).sum(level=event_level)
    df_dist['c_ag'] = df.eval('pcwgt*di_ag/v_ag').sum(level=event_level)

    # Get rid of rp>50, because the exceedance curves stop there 
    # --> average_over_rp() would miscalculate AAL, otherwise
    df_dist = df_dist.reset_index('rp')
    df_dist = df_dist.loc[df_dist.rp<=50,:].reset_index().set_index(event_level)
    
    # Average over rp
    df_dist_aal,_ = average_over_rp(df_dist)

    # Put in USD
    df_dist_aal/=730.
    
    #Latex
    latex_out = (1E-3*df_dist_aal[['di_ag','dw_ag_currency']]).round(1).sort_values('di_ag',ascending=True).rename(columns={'di_ag':'Income loss','dw_ag_currency':'Wellbeing loss'})
    latex_out.loc['Total'] = latex_out.sum()
    latex_out['Socioeconomic resilience'] = (1E2*latex_out['Income loss']/latex_out['Wellbeing loss']).round(0).astype('int')
    latex_out.to_latex('~/Desktop/tmp/c_ag.tex')
    assert(False)
    return df_dist_aal.sort_values('di_ag',ascending=True)
#

def make_drought_maps(df_dist_aal,myC='MW',mapres=750):
    
    print(df_dist_aal.head())
    df_dist_aal_nocities = df_dist_aal.drop(['Zomba City','Mzuzu City','Blantyre City','Lilongwe City'])

    _label = 'Expected annual agricultural income losses\nto drought (,000 US\$ per year)' 
    make_map_from_svg(
        df_dist_aal_nocities['di_ag']*1E-3,
        get_svg_file(myC),
        outname=myC+'_drought_income_AAL',
        color_maper=(plt.cm.get_cmap('YlOrBr')), 
        label=_label,
        new_title='',
        do_qualitative=False,
        res=mapres,
        drop_spots=None,
        force_max=1300)

    _label = 'Expected annual wellbeing losses\nto drought (,000 US\$ per year)' 
    make_map_from_svg(
        df_dist_aal_nocities['dw_ag_currency']*1E-3,
        get_svg_file(myC),
        outname=myC+'_drought_wellbeing_AAL',
        color_maper=(plt.cm.get_cmap('YlOrBr')), 
        label=_label,
        new_title='',
        do_qualitative=False,
        res=mapres,
        drop_spots=None,
        force_max=1300)

    _label = 'Socioeconomic resilience to drought (%)' 
    make_map_from_svg(
        1E2*df_dist_aal_nocities.eval('di_ag/dw_ag_currency'),
        get_svg_file(myC),
        outname=myC+'_drought_resilience',
        color_maper=(plt.cm.get_cmap('RdYlGn')), 
        label=_label,
        new_title='',
        do_qualitative=False,
        res=mapres,
        drop_spots=None)


def load_drought_hazard(myCountry,fname,event_level,income_cats):
    
    hazard_ratios = pd.read_csv(fname, index_col=event_level+[income_cats])
    
    if 'DR' not in hazard_ratios.index.get_level_values('hazard'): return None

    # separate drought from other hazards
    hazard_ratios_drought = hazard_ratios.loc[hazard_ratios.index.get_level_values('hazard')=='DR',:].copy()
    hazard_ratios_drought = hazard_ratios_drought.drop(['v_mean','hh_reco_rate','public_loss_v','dy_over_dk','fa','k','v'],axis=1)

    # need consumption info in this df
    hazard_ratios_drought = pd.merge(hazard_ratios_drought.reset_index(),
                                     pd.read_csv('../intermediate/'+myCountry+'/cat_info.csv')[['hhid','c']],on='hhid').set_index(event_level+[income_cats]).sort_index()

    return hazard_ratios_drought

def calc_dw_drought(df_haz,_rho,_eta=1.5):
    #    
    df_haz['welf_i'] = 0
    df_haz['welf_f'] = 0    
    #
    _dt = 1/52
    _integral = 0

    _gamma = _rho/_eta
    _e_to_gamma = np.exp(-_gamma)

    _e = np.e
    for _t in np.linspace(0,1,int(1/_dt)):
        print(_t)
        #df_haz['welf_i'] += df_haz.eval('@_dt/(1-@_eta)*((c-pcinc_ag_gross)+pcinc_ag_gross*(@_e_to_gamma*@_t+@_e**(-@_rho*@_t)))**(1-@_eta)')
        #df_haz['welf_f'] += df_haz.eval('@_dt/(1-@_eta)*((c-pcinc_ag_gross)+pcinc_ag_gross*(1-v_ag)*(@_e_to_gamma*@_t+@_e**(-@_rho*@_t)))**(1-@_eta)')
        
        df_haz['welf_i'] += df_haz.eval('@_dt/(1-@_eta)*((c-pcinc_ag_gross)+pcinc_ag_gross*@_gamma/(1-@_e_to_gamma)*@_e**(-@_gamma*@_t))**(1-@_eta)*@_e**(-@_rho*@_t)')
        df_haz['welf_f'] += df_haz.eval('@_dt/(1-@_eta)*((c-pcinc_ag_gross)+pcinc_ag_gross*(1-v_ag)*@_gamma/(1-@_e_to_gamma)*@_e**(-@_gamma*@_t))**(1-@_eta)*@_e**(-@_rho*@_t)')        

    df_haz['dw'] = df_haz.eval('welf_i-welf_f')
    #print(df_haz[['welf_i','welf_f','dw']].head(30))
    #assert(False)
    #
    df_haz['di_ag'] = df_haz.eval('pcinc_ag_gross*v_ag')
    #
    return df_haz[['pcwgt','c','v_ag','fa_ag','di_ag','dw']]


#def get_dw_integral_coeff(rho=0.06,eta=1.5):
#    _dt = 1E-3
#    _integral = 0
#    for _t in np.linspace(0,1,int(1/_dt)):
#        _integral += _dt*(np.e**(-rho)*_dt+np.e**(-rho*_t))**(1-eta)*np.e**(-rho*_t)
#    return _integral

def get_agricultural_vulnerability_to_drought(myC,df):
    #set_directories(myC)
    df = df.reset_index('district')
    df['frac_c_ag_gross'] = df.eval('pcinc_ag_gross/c')
    #
    _df = pd.read_stata('../inputs/MW/MWI_2016_IHS-IV_v02_M_Stata/AG_MOD_D.dta').rename(columns={'case_id':'hhid'}).set_index('hhid')
    for _c in [['ag_d36','ag_applied_fert_org','YES'],
               ['ag_d38','ag_applied_fert_inorg','YES'],
               ['ag_d28a','ag_is_rainfed','Rainfed/No irrigation']]:

        df[_c[1]] = _df.loc[~_df.index.duplicated(keep='first'),_c[0]].apply(lambda x:(True if x==_c[2] else False))    
        df[_c[1]]  = df[_c[1]].fillna(False)  
    #
    df['v_ag'] = df.apply(lambda x:nprand.uniform(0.75*(0.667 if x.ag_is_rainfed else 0.333),
                                                  1.25*(0.667 if x.ag_is_rainfed else 0.333)),axis=1)
    
    df = df.reset_index().set_index(['district','hhid'])
    df.to_csv('~/Desktop/tmp/ag.csv')

    return df['v_ag']

def get_ag_value(myC,fom,yr=2015):
    #
    df = pd.read_csv('../inputs/'+myC+'/faostat_ag_production_value/FAOSTAT_data_4-13-2019.csv')
    df = df.loc[df.eval("(Element=='Gross Production Value (current million SLC)')&(Year=="+str(yr)+")"),:]
    #
    if fom == 'mz_frac_ag': return float(df.loc[df.eval("(Item=='Maize')"),'Value']/df['Value'].sum())
    if fom == 'ag_value': return float(df['Value'].sum())
    assert(False)
    

def drought_study(df):
    
    print('\n')
    print('Frac non-poor affected by drought:',df.loc[(df.impact_drought==1)&(df.ispoor=='Non-poor'),'pcwgt'].sum()/df.loc[(df.ispoor=='Non-poor'),'pcwgt'].sum())
    print('Frac poor affected by drought:',df.loc[(df.impact_drought==1)&(df.ispoor=='Non-poor'),'pcwgt'].sum()/df.loc[(df.ispoor=='Poor'),'pcwgt'].sum())
    print('Frac subsistence affected by drought:',df.loc[(df.impact_drought==1)&(df.issub=='Ultra-poor'),'pcwgt'].sum()/df.loc[(df.issub=='Ultra-poor'),'pcwgt'].sum())
    print('\n')

    print('Frac w/ ganyu income affected by drought:',df.loc[(df.impact_drought==1)&(df.income_ganyu>0),'pcwgt'].sum()/df.loc[(df.income_ganyu>0),'pcwgt'].sum())
    print('Frac w/o ganyu income affected by drought:',df.loc[(df.impact_drought==1)&(df.income_ganyu==0),'pcwgt'].sum()/df.loc[(df.income_ganyu==0),'pcwgt'].sum())
    print('\n')
    print('Frac w/ unpaid ag labor affected by drought:',df.loc[df.eval('(impact_drought==1)&(labor_ag_unpaid==0)'),'pcwgt'].sum()/df.loc[(df.labor_ag_unpaid>0),'pcwgt'].sum())
    print('Frac w/o unpaid ag labor affected by drought:',df.loc[df.eval('(impact_drought==1)&(labor_ag_unpaid==0)'),'pcwgt'].sum()/df.loc[(df.labor_ag_unpaid==0),'pcwgt'].sum())
    print('\n')
    print('Frac w/ ag wages affected by drought:',df.loc[df.eval('(impact_drought==1)&(main_wage_job_ag==1)'),'pcwgt'].sum()/df.loc[(df.main_wage_job_ag==1),'pcwgt'].sum())
    print('Frac w/o ag wages affected by drought:',df.loc[df.eval('(impact_drought==1)&(main_wage_job_ag==0)'),'pcwgt'].sum()/df.loc[(df.main_wage_job_ag==0),'pcwgt'].sum())
    print('\n')
    print('Frac w/ unpaid ag labor or ganyu affected by drought:',(df.loc[df.eval('(impact_drought==1)&((labor_ag_unpaid>0)|(income_ganyu>0))'),'pcwgt'].sum()
                                                                   /df.loc[df.eval('((labor_ag_unpaid>0)|(income_ganyu>0))'),'pcwgt'].sum()))
    
    print('Frac w/o unpaid ag labor or ganyu affected by drought:',(df.loc[df.eval('(impact_drought==1)&(labor_ag_unpaid==0)&(income_ganyu==0)'),'pcwgt'].sum()
                                                                    /df.loc[df.eval('(labor_ag_unpaid==0)&(income_ganyu==0)'),'pcwgt'].sum()))
  
    print('\n')
    print('Frac w/ unpaid ag labor/ganyu/ag wage affected by drought:',(df.loc[df.eval('(impact_drought==1)&((labor_ag_unpaid>0)|(income_ganyu>0)|(main_wage_job_ag==1))'),'pcwgt'].sum()
                                                                        /df.loc[df.eval('((labor_ag_unpaid>0)|(income_ganyu>0)|(main_wage_job_ag==1))'),'pcwgt'].sum()))
    print('Frac w/o unpaid ag labor/ganyu/ag wage affected by drought:',(df.loc[df.eval('(impact_drought==1)&(labor_ag_unpaid==0)&(income_ganyu==0)&(main_wage_job_ag==0)'),'pcwgt'].sum()
                                                                         /df.loc[df.eval('(labor_ag_unpaid==0)&(income_ganyu==0)&(main_wage_job_ag==0)'),'pcwgt'].sum()))  

    df = df.reset_index().set_index(['district','hhid'])
    df_dist = pd.DataFrame(index=df.sum(level='district').index)
    df_dist['frac_drought_impact'] = (df.loc[(df.impact_drought==1),'pcwgt'].sum(level='district')/df['pcwgt'].sum(level='district'))
    df_dist['frac_ganyu_income'] = (df.loc[(df.income_ganyu>0),'pcwgt'].sum(level='district')/df['pcwgt'].sum(level='district'))
    df_dist['frac_ganyu_plus'] = (df.loc[(df.income_ganyu>0)|(df.main_wage_job_ag==1),'pcwgt'].sum(level='district')/df['pcwgt'].sum(level='district'))
    df_dist['frac_ag_unpaid'] = (df.loc[(df.labor_ag_unpaid>0),'pcwgt'].sum(level='district')/df['pcwgt'].sum(level='district'))

    ax = df_dist.plot.scatter('frac_ganyu_income','frac_drought_impact',color='red')
    df_dist.plot.scatter('frac_ganyu_plus','frac_drought_impact',color='blue',ax=ax)
    df_dist.plot.scatter('frac_ag_unpaid','frac_drought_impact',color='green',ax=ax)

    plt.gca().get_figure().savefig('../output_plots/MW/drought_impacts.pdf',format='pdf')

    df_ag = pd.read_stata('../inputs/MW/MWI_2016_IHS-IV_v02_M_Stata/AG_MOD_M.dta',columns=['case_id','ag_m10a','ag_m10b']).rename(columns={'case_id':'hhid'}).set_index('hhid')
    print(df_ag.head())
    print(df_ag.loc[(df_ag.ag_m10a=='Drought')|(df_ag.ag_m10b=='Drought')])

    #print(df.head())
    assert(False)

def draw_drought_income_plot(df=None,_='initial',_rho=0.70):
    big_pal = sns.color_palette('tab20b', n_colors=20)

    if df is None: df = pd.read_csv('../output_country/MW/out_ag.csv').set_index(event_level+['hhid'])

    df['frac_ag'] = df.eval('(di_ag/v_ag)/c')
    df['i_nonag'] = 1E-3*df.eval('c*(1-frac_ag)')
    df['i_ag'] = 1E-3*df.eval('c*(frac_ag)')
    df = (df.loc[(df.frac_ag>=0.70)&(df.frac_ag<0.99)&(df.v_ag<=0.40)].sample(1)).squeeze()
    print(df)

    ## Non-agricultural income
    #plt.plot([0,2],[df['c_nonag'],df['c_nonag']])
    
    #########################
    # Agricultural income spenddown
    _x   = []; _y   = []; _x_lean   = []; _y_lean   = []
    _xDR = []; _yDR = []; _xDR_lean = []; _yDR_lean = []
    t_series,_dt = np.linspace(0,2,4E2,retstep=True)
    c_ag_init = _rho*df['i_ag']/(1-np.exp(-_rho))

    for _t in t_series:
        
        _sf = 1 if _t < 1 else (1-df['v_ag']) 
        _tcyc = _t if _t < 1 else _t-1
        #
        _cag_t = c_ag_init*_sf*np.exp(-_rho*_tcyc) 
        #
        if _t <= 1:
            _x.append(_t)
            _y.append(_cag_t)
            if _cag_t <= df['i_ag']*_sf:
                _x_lean.append(_t)
                _y_lean.append(_cag_t)
        else:
            _xDR.append(_t)
            _yDR.append(_cag_t)
            if _cag_t <= df['i_ag']*_sf:
                _xDR_lean.append(_t)
                _yDR_lean.append(_cag_t)


    # Plot c_ag(t)
    plt.plot(_x,_y,color=greys_pal[7],alpha=0.9,zorder=100)
    plt.plot(_xDR,_yDR,color=greys_pal[7],alpha=0.9,zorder=100)
    #
    plt.plot([_-1 for _ in _x],_y,color=greys_pal[7],alpha=0.9,zorder=100,clip_on=True)
    plt.plot([_+2 for _ in _x],_y,color=greys_pal[7],alpha=0.9,zorder=100,clip_on=True)
    #
    plt.plot([0,0],[_y[-1],_y[0]],color=greys_pal[7],alpha=0.9,zorder=100)   # connect year = -1 to year = 0
    plt.plot([1,1],[_y[-1],_yDR[0]],color=greys_pal[8],alpha=0.9,zorder=100) # connect year =  0 to year = 1
    plt.plot([2,2],[_yDR[-1],_y[0]],color=greys_pal[7],alpha=0.9,zorder=100) # connect year =  1 to year = 2

    # and do annotation
    #_str = 'Household consumption\nof agricultural income\n$c_{ag}(t)\,=\,C^{\prime}_{ag}\cdot e^{-\gamma_{ag}t}$\n(cf. Eqs. 3$\,$&$\,$7)'
    _str = 'Household consumption\nof agricultural income $c_{ag}(t)$\n(cf.$\,$Eqs.$\,$2-7)'
    plt.annotate(_str,
                 xy=(_x[5],_y[1]),
                 xytext=(_x[13],_y[0]),
                 arrowprops=dict(arrowstyle="-",facecolor=greys_pal[7],connectionstyle="angle,angleA=0,angleB=-135,rad=0"),
                 fontsize=7.5,color=greys_pal[7],va='bottom',ha='left',linespacing=1.4)

    #########################
    # Agricultural income spike
    for yr_no in [0,1,2]:
        # annual income squeezed into 1 month
        len_harvest = 4 # months

        _sf = 1
        if yr_no == 1: _sf = (1-df['v_ag']) 
        
        plt.plot([yr_no,yr_no],[0,(12/len_harvest)*df['i_ag']*_sf],color=big_pal[10],alpha=0.5)
        plt.plot([yr_no,yr_no+len_harvest/12],[(12/len_harvest)*df['i_ag']*_sf,(12/len_harvest)*df['i_ag']*_sf],color=big_pal[10],alpha=0.5)
        plt.fill_between([yr_no,yr_no+len_harvest/12],[(12/len_harvest)*df['i_ag']*_sf,(12/len_harvest)*df['i_ag']*_sf],color=big_pal[11],alpha=0.5)
        plt.plot([yr_no+len_harvest/12,yr_no+len_harvest/12],[0,(12/len_harvest)*df['i_ag']*_sf],color=big_pal[10],alpha=0.5)
        #
        # Mark lean season
        if yr_no != 2:
            plt.fill_between((_x_lean if yr_no == 0 else _xDR_lean)[:-1],
                             y1=(_y_lean if yr_no == 0 else _yDR_lean)[:-1],
                             #y2=[0 for _tcyc in _y_lean[:-1]],
                             color=big_pal[13],alpha=0.1)
            #  and do annotation
            if yr_no == 0: plt.annotate('Lean season',xy=((_xDR_lean if yr_no == 1 else _x_lean)[-4],df['i_ag']*(0.03)),
                                        fontsize=7,color=big_pal[13],va='bottom',ha='right')
            # 
            # Plot annualized income as dotted line
            plt.plot([yr_no,yr_no+1],[df['i_ag']*_sf,df['i_ag']*_sf],color=greys_pal[4],alpha=0.4,ls='--',lw=0.75)
            #  and do annotation
            _labstr = ('Income from main\nharvest, normal year' if yr_no == 0 else 'Income from main\nharvest, drought year')
            plt.annotate(_labstr,
                         xy=(yr_no+len_harvest/12*0.20,(12/len_harvest)*df['i_ag']*_sf),
                         xytext=(yr_no+len_harvest/12*(0.20*1.75),(12/len_harvest)*df['i_ag']*(_sf+0.06)),
                         arrowprops=dict(arrowstyle="-",facecolor=greys_pal[7],connectionstyle="angle,angleA=0,angleB=-90,rad=5"),
                         fontsize=7.5,color=greys_pal[7],va='center',ha='left')

    # ticks
    plt.xticks([0,0.25,0.5,0.75,1.0,1.25,1.50,1.75,2])
    plt.gca().set_xticklabels(labels=['April','July','October','January',
                                      'April','July','October','January','April'],ha='right',fontsize=8,rotation=45)
    plt.tick_params(axis='x', which='major',color=greys_pal[4])
    plt.tick_params(axis='y', which='major',color=greys_pal[4],labelsize=8)
    #
    title_legend_labels(plt.gca(),'',lab_x='',lab_y='Kwacha per capita (,000)',lim_x=None,lim_y=None,leg_fs=9,do_leg=False)
    #
    plt.grid(False)
    plt.xlim(-0.05,2.05)
    plt.ylim(0)
    sns.despine()
    plt.gcf().savefig('../output_plots/MW/drought/schematic.pdf',format='pdf',bbox_inches='tight')
    plt.close('all')
    
