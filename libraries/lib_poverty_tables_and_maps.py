import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from libraries.maps_lib import make_map_from_svg, purge
from libraries.lib_average_over_rp import average_over_rp
from libraries.lib_country_dir import get_demonym,get_poverty_line,get_currency
from libraries.lib_gather_data import match_percentiles,perc_with_spline,reshape_data
from libraries.lib_common_plotting_functions import title_legend_labels,sns_pal
import glob

purge('img/','map_of_*.png')
purge('img/','legend_of_*.png')
purge('img/','map_of_*.svg')
purge('img/','legend_of_*.svg')

col_cast_dict = {'net_chg_pov_i':'int', 'pct_pov_i':'float64',
                 'net_chg_sub_i':'int', 'pct_sub_i':'float64',
                 'net_chg_pov_c':'int', 'pct_pov_c':'float64',
                 'net_chg_sub_c':'int', 'pct_sub_c':'float64'}

haz_dict = {'SS':'Storm surge',
            'PF':'Precipitation flood',
            'HU':'Hurricane',
            'EQ':'Earthquake',
            'DR':'Drought',
            'FF':'Fluvial flood'}

def map_recovery_time(myC,HAZ=['PF'],RP=[100],RECO=['75','80','90']):
    df = pd.read_csv('../output_country/'+myC+'/time_to_recovery_no.csv')

    # hack
    if myC == 'MW':
        df.loc['Blantyre'] = df.loc[['Blantyre','Blantyre City']].sum()
        df.loc['Lilongwe'] = df.loc[['Lilongwe','Lilongwe City']].sum()
        df.loc['Mzimba'] = df.loc[['Mzimba','Mzuzu City']].sum()
        df.loc['Zomba Non-City'] = df.loc[['Zomba Non-City','Zomba City']].sum() 
        df = df_prov.drop(['Blantyre City','Lilongwe City', 'Mzuzu City', 'Zomba City'],axis=0)

    # Look for the map (svg) file here
    svg_file = ''
    if myC == 'PH': svg_file = '../map_files/'+myC+'/BlankSimpleMapRegional.svg'
    elif myC == 'SL': svg_file = '../map_files/'+myC+'/lk.svg'
    elif myC == 'MW': svg_file = '../map_files/'+myC+'/mw.svg'

    for _haz in HAZ:
        for _rp in RP:
            for _reco in RECO:

                _ = df.loc[(df.hazard == _haz)&(df.rp == _rp)].set_index(['region'])
                _.loc[_['time_recovery_'+_reco]==-1,'time_recovery_'+_reco] = 10

                make_map_from_svg(
                    _['time_recovery_'+_reco], 
                    svg_file,
                    outname='time_to_recover_'+_reco+'pct_'+_haz+str(_rp),
                    color_maper=plt.cm.get_cmap('Purples'), 
                    label='Time to reconstruct '+_reco+'% of assets destroyed \nby '+str(_rp)+'-year '+haz_dict[_haz].lower()+' [years]',
                    new_title='',
                    do_qualitative=False,
                    res=2000)

def run_poverty_duration_plot(myC):

    # Load file with geographical (region/province/district) as index
    df = pd.read_csv('../output_country/'+myC+'/poverty_duration_no.csv')
    df = df.reset_index().set_index(df.columns[1]).sort_index()
    
    geo = df.index.name
    all_geo = np.array(df[~df.index.duplicated(keep='first')].index)

    # used in groupby  
    df['country'] = myC

    # assign deciles
    listofdeciles=np.arange(0.10, 1.01, 0.10)
    df = df.reset_index().groupby('country',sort=True).apply(lambda x:match_percentiles(x,perc_with_spline(reshape_data(x.c),reshape_data(x.pcwgt),listofdeciles),'decile'))

    # Load additional SP runs    
    _sp = []
    for f in glob.glob('/Users/brian/Desktop/BANK/hh_resilience_model/output_country/'+myC+'/poverty_duration_*.csv'):
        _ = f.replace('/Users/brian/Desktop/BANK/hh_resilience_model/output_country/'+myC+'/poverty_duration_','').replace('.csv','')
        _sp.append(_)
        
    for iSP in _sp:
        _ = pd.read_csv('../output_country/'+myC+'/poverty_duration_'+iSP+'.csv')

        _['t_pov_bool'] = False        
        try: _.loc[_.c>_.pov_line,'t_pov_bool'] = True
        except: _.loc[_.c>get_poverty_line(myC,by_district=False),'t_pov_bool'] = True

        df[['t_pov_inc'+iSP,'t_pov_cons'+iSP,'t_pov_bool'+iSP]] = _[['t_pov_inc','t_pov_cons','t_pov_bool']]


    ############################
    # Do some plotting
    #plot_crit = '(t_pov_bool)&(hazard=="PF")&(rp==500)'

    #df.loc[df.eval(plot_crit)].plot.hexbin('dk0','t_pov_cons')
    #plt.gca().get_figure().savefig('../output_plots/'+myC+'/poverty_duration_hexbin_no.pdf',format='pdf')
    #plt.cla()

    #df.loc[df.eval(plot_crit)].plot.scatter('dk0','t_pov_cons')
    #plt.gca().get_figure().savefig('../output_plots/'+myC+'/poverty_duration_scatter_no.pdf',format='pdf')
    #plt.cla()    


    ############################
    df = df.reset_index().set_index(['hazard','rp','decile'])
    df['t_pov_bool'] = False
    try: df.loc[df.c>df.pov_line,'t_pov_bool'] = True
    except: df.loc[df.c>get_poverty_line(myC,by_district=False),'t_pov_bool'] = True

    df_dec = pd.DataFrame(index=df.sum(level=['hazard','rp','decile']).index).sort_index()
    # Populate the df_dec dataframe now, while its index is set to ['hazard','rp','decile']

    # Number of individuals who face income or consumption poverty
    df_dec['n_new_pov_inc']  = df.loc[df.t_pov_bool==True,'pcwgt'].sum(level=['hazard','rp','decile'])
    df_dec['n_new_pov_cons'] = df.loc[df.t_pov_bool==True,'pcwgt'].sum(level=['hazard','rp','decile'])

    # Individuals who face income or consumption poverty as fraction of all individuals
    df_dec['frac_new_pov_inc'] = df_dec['n_new_pov_inc']/df['pcwgt'].sum(level=['hazard','rp','decile'])
    df_dec['frac_new_pov_cons'] = df_dec['n_new_pov_cons']/df['pcwgt'].sum(level=['hazard','rp','decile'])

    # Among people pushed into pov: average time in poverty (months)
    for iSP in _sp:
        df_dec['t_pov_inc_avg'+iSP] = 12.*(df.loc[df.eval('t_pov_bool'+iSP+'==True'),['pcwgt','t_pov_inc'+iSP]].prod(axis=1).sum(level=['hazard','rp','decile'])
                                           /df.loc[df.eval('t_pov_bool'+iSP+'==True'),'pcwgt'].sum(level=['hazard','rp','decile']))
        df_dec['t_pov_cons_avg'+iSP] = 12.*(df.loc[df.eval('t_pov_bool'+iSP+'==True'),['pcwgt','t_pov_cons'+iSP]].prod(axis=1).sum(level=['hazard','rp','decile'])
                                            /df.loc[df.eval('t_pov_bool'+iSP+'==True'),'pcwgt'].sum(level=['hazard','rp','decile']))

    for iloc in all_geo:
        df_dec['t_pov_inc_avg_'+iloc] = 12.*(df.loc[df.eval('(t_pov_bool==True)&('+geo+'==@iloc)'),['pcwgt','t_pov_inc']].prod(axis=1).sum(level=['hazard','rp','decile'])
                                             /df.loc[df.eval('(t_pov_bool==True)&('+geo+'==@iloc)'),'pcwgt'].sum(level=['hazard','rp','decile']))

        df_dec['t_pov_cons_avg_'+iloc] = 12.*(df.loc[df.eval('(t_pov_bool==True)&('+geo+'==@iloc)'),['pcwgt','t_pov_cons']].prod(axis=1).sum(level=['hazard','rp','decile'])
                                              /df.loc[df.eval('(t_pov_bool==True)&('+geo+'==@iloc)'),'pcwgt'].sum(level=['hazard','rp','decile']))

    df_dec.to_csv('../output_country/'+myC+'/poverty_by_decile.csv')

    ######################
    # Scatter plot of hh that have to delay reconstruction
    upper_lim = 1E15
    df['t_reco'] = (np.log(1.0/0.05)/df['hh_reco_rate']).clip(upper=upper_lim)

    means = []
    xmax = 2.5E5
    step = xmax/10.
    for i in np.linspace(0,10,10):        
        means.append(df.loc[df.eval('(rp==1000)&(c>@i*@step)&(c<=(@i+1)*@step)&(t_reco!=@upper_lim)'),['pcwgt','t_reco']].prod(axis=1).sum()/
                     df.loc[df.eval('(rp==1000)&(c>@i*@step)&(c<=(@i+1)*@step)&(t_reco!=@upper_lim)'),['pcwgt']].sum())

    ax = df.loc[df.eval('(c<@xmax)&(t_reco<12)')].plot.hexbin('c','t_reco',gridsize=25,mincnt=1)
    plt.plot(step*np.linspace(0,10,10),means,zorder=100)

    # Do the formatting
    ax = title_legend_labels(ax,'Precipitation flood in '+myC,lab_x='Pre-disaster consumption [PhP per cap]',lab_y='Time to reconstruct [years]',leg_fs=9)
    
    # Do the saving
    plt.draw()
    plt.gca().get_figure().savefig('../output_plots/'+myC+'/t_start_reco_scatter.pdf',format='pdf')
    plt.cla()

    ######################
    # Latex table of poorest quintile poverty time

    _cons_to_tex = df_dec.drop([i for i in df_dec.columns if i not in ['t_pov_cons_avg_'+j for j in all_geo]],axis=1)
    _inc_to_tex = df_dec.drop([i for i in df_dec.columns if i not in ['t_pov_inc_avg_'+j for j in all_geo]],axis=1)

    _cons_to_tex = _cons_to_tex.rename(columns={'t_pov_cons_avg_'+j:j for j in all_geo}).stack()
    _inc_to_tex  =  _inc_to_tex.rename(columns={'t_pov_inc_avg_'+j:j for j in all_geo}).stack()
    _cons_to_tex.index.names = ['hazard','rp','decile',geo]
    _inc_to_tex.index.names = ['hazard','rp','decile',geo]

    _to_tex = pd.DataFrame(index=_cons_to_tex.index)
    _to_tex['Income'] = _inc_to_tex.round(1)
    _to_tex['Consumption'] = _cons_to_tex.round(1)
  
    _to_tex = _to_tex.reset_index().set_index(geo)
    _to_tex = _to_tex.loc[_to_tex.eval('(hazard=="PF")&(rp==10)&(decile==1)')].sort_values('Consumption',ascending=False)
    
    _to_tex[['Income','Consumption']].to_latex('latex/'+myC+'_poverty_duration.tex')

    ######################
    # Plot consumption and income poverty (separately)
    df_dec = df_dec.reset_index()

    _lab = {'t_pov_cons_avg':'Average time to exit poverty\n(income net of reconstruction & savings) [months]',
            't_pov_inc_avg':'Average time to exit poverty (income only) [months]'}

    print(df_dec.columns)

    for ipov in ['t_pov_cons_avg','t_pov_inc_avg']:
        # Do the plotting
        #ax = df_dec.loc[df_dec.eval('(hazard=="PF")&(rp==10)')].plot.scatter('decile',ipov+'no',color=sns_pal[1],lw=0,label='Natl. average (RP = 5 years)',zorder=99)
        #df_dec.loc[df_dec.eval('(hazard=="PF")&(rp==1000)')].plot.scatter('decile',ipov+'no',color=sns_pal[3],lw=0,label='Natl. average (RP = 1000 years)',zorder=98,ax=ax)

        try:
            ax = df_dec.loc[df_dec.eval('(hazard=="PF")&(rp==10)')].plot('decile',ipov+'no',color=sns_pal[1],zorder=97,label='')
            df_dec.loc[df_dec.eval('(hazard=="PF")&(rp==1000)')].plot('decile',ipov+'no',color=sns_pal[3],zorder=96,label='',ax=ax)
        except: pass

        icol = 4

        # Which areas to plot?
        if myC == 'SL': focus = ['Colombo','Rathnapura','Kalutara','Mannar']
        elif myC == 'PH': focus = ['NCR']
        elif myC == 'MW': focus = ['Lilongwe','Chitipa']

        for iloc in focus:
            df_dec.loc[df_dec.eval('(hazard=="PF")&(rp==10)')].plot.scatter('decile',ipov+'_'+iloc,color=sns_pal[icol],lw=0,label=iloc+' (RP = 5 years)',zorder=95,ax=ax)
            df_dec.loc[df_dec.eval('(hazard=="PF")&(rp==10)')].plot('decile',ipov+'_'+iloc,color=sns_pal[icol],zorder=94,label='',ax=ax)
            icol+=1

        # Do the formatting
        ax = title_legend_labels(ax,'Precipitation flood in '+myC,lab_x='Decile',lab_y=_lab[ipov],lim_x=[0.5,10.5],lim_y=[-0.1,42],leg_fs=9)
        ax.xaxis.set_ticks([1,2,3,4,5,6,7,8,9,10])
        ax.yaxis.set_ticks([0,6,12,18,24,30,36,42])

        # Do the saving
        ax.get_figure().savefig('../output_plots/'+myC+'/'+ipov+'_by_decile.pdf',format='pdf')
        plt.cla()    

    ######################
    # Plot consumption and income poverty (separately), with alternative SPs
    _lab = {'t_pov_cons_avg':'Average time to exit poverty\n(income net of reconstruction & savings) [months]',
            't_pov_inc_avg':'Average time to exit poverty (income only) [months]'}

    icol=0
    ax = plt.gca()
    for ipov in ['t_pov_cons_avg','t_pov_inc_avg']:
        # Do the plotting
        _df_5 = df_dec.loc[df_dec.eval('(hazard=="PF")&(rp==10)')].copy()
        _df_1000 = df_dec.loc[df_dec.eval('(hazard=="PF")&(rp==1000)')].copy()

        for iSP in _sp:

            plt.fill_between(_df_5['decile'].values,_df_5[ipov+iSP],_df_1000[ipov+iSP],alpha=0.3)

            _df_5.plot.scatter('decile',ipov+iSP,lw=0,label='',zorder=99,ax=ax)
            _df_1000.plot.scatter('decile',ipov+iSP,lw=0,label='',zorder=98,ax=ax)
            
            _df_5.plot('decile',ipov+iSP,zorder=97,linestyle=':',label=iSP+' natl. average (RP = 5 years)',ax=ax)
            _df_1000.plot('decile',ipov+iSP,zorder=96,label=iSP+' natl. average (RP = 1000 years)',ax=ax)
            
            icol+=1

        # Do the formatting
        ax = title_legend_labels(ax,'Precipitation flood in '+myC,lab_x='Decile',lab_y=_lab[ipov],lim_x=[0.5,10.5],lim_y=[-0.1],leg_fs=9)
        ax.xaxis.set_ticks([1,2,3,4,5,6,7,8,9,10])
        ax.yaxis.set_ticks([0,3,6,9,12,15,18])

        # Do the saving
        ax.get_figure().savefig('../output_plots/'+myC+'/'+ipov+'_with_sps_by_decile.pdf',format='pdf')
        plt.cla()

    plt.close('all')

    return True

def run_poverty_tables_and_maps(myC,pov_df,event_level=['region','hazard','rp']):

    # Load demonym for this country
    dem = get_demonym(myC)

    # Look for the map (svg) file here
    svg_file = '../map_files/'+myC+'/BlankSimpleMap.svg'
    if myC == 'PH' and event_level[0] == 'region':
        svg_file = '../map_files/'+myC+'/BlankSimpleMapRegional.svg'
    elif myC == 'SL': svg_file = '../map_files/'+myC+'/lk.svg'
    elif myC == 'MW': svg_file = '../map_files/'+myC+'/mw.svg'

    regional_poverty = pd.read_csv('../inputs/'+myC+'/regional_poverty_rate.csv').set_index(event_level[0])
    make_map_from_svg(
        regional_poverty['poverty_rate'], 
        svg_file,
        outname=myC+'_regional_poverty_incidence',
        color_maper=plt.cm.get_cmap('Reds'), 
        label='Regional poverty incidence [%]',
        new_title= dem+'Regional poverty incidence [%]',
        do_qualitative=False,
        res=2000)  

    # Get the poverty headcount info
    #try:
    # Count up the hh that fell into poverty & subsistence:
    pov_df_event = (pov_df.loc[pov_df.eval('(c_pre_reco<=pov_line)'),'pcwgt'].sum(level=event_level)
                    -pov_df.loc[pov_df.eval('(c_initial<=pov_line)'),'pcwgt'].sum(level=event_level)).to_frame(name='net_chg_pov_c')
    pov_df_event['net_chg_pov_i'] = (pov_df.loc[pov_df.eval('(i_pre_reco<=pov_line)'),'pcwgt'].sum(level=event_level)
                                     -pov_df.loc[pov_df.eval('(c_initial<=pov_line)'),'pcwgt'].sum(level=event_level))
    try: 
        pov_df_event['net_chg_pov_c_children'] = (pov_df.loc[pov_df.eval('(c_pre_reco<=pov_line)'),['N_children','hhwgt']].prod(axis=1).sum(level=event_level)
                                                  -pov_df.loc[pov_df.eval('(c_initial<=pov_line)'),['N_children','hhwgt']].prod(axis=1).sum(level=event_level))
        pov_df_event['net_chg_pov_i_children'] = (pov_df.loc[pov_df.eval('(i_pre_reco<=pov_line)'),['N_children','hhwgt']].prod(axis=1).sum(level=event_level)
                                                  -pov_df.loc[pov_df.eval('(c_initial<=pov_line)'),['N_children','hhwgt']].prod(axis=1).sum(level=event_level))
    except: pass

    # hack!
    if myC == 'MW':
        
        pov_df_event.loc['Blantyre'] = pov_df_event.loc[['Blantyre','Blantyre City']].sum()
        pov_df_event.loc['Lilongwe'] = pov_df_event.loc[['Lilongwe','Lilongwe City']].sum()
        pov_df_event.loc['Mzimba'] = pov_df_event.loc[['Mzimba','Mzuzu City']].sum()
        pov_df_event.loc['Zomba Non-City'] = pov_df_event.loc[['Zomba Non-City','Zomba City']].sum() 
        pov_df_event = pov_df_event.drop(['Blantyre City','Lilongwe City', 'Mzuzu City', 'Zomba City'],axis=0)
        
    pov_df_event['net_chg_sub_c'] = (pov_df.loc[pov_df.eval('(c_pre_reco<=sub_line)'),'pcwgt'].sum(level=event_level).fillna(0)
                                     -pov_df.loc[pov_df.eval('(c_initial<=sub_line)'),'pcwgt'].sum(level=event_level).fillna(0))
    pov_df_event['net_chg_sub_i'] = (pov_df.loc[pov_df.eval('(i_pre_reco<=sub_line)'),'pcwgt'].sum(level=event_level).fillna(0)
                                     -pov_df.loc[pov_df.eval('(c_initial<=sub_line)'),'pcwgt'].sum(level=event_level).fillna(0))

    pov_df_event['init_pov'] = pov_df.loc[pov_df.eval('(c_initial<=pov_line)'),'pcwgt'].sum(level=event_level).fillna(0)
    pov_df_event['init_sub'] = pov_df.loc[pov_df.eval('(c_initial<=sub_line)'),'pcwgt'].sum(level=event_level).fillna(0)        

    pov_df_event['reg_pop'] = pov_df['pcwgt'].sum(level=event_level)
    
    pov_df_event.to_csv('../output_country/'+myC+'/net_chg_pov_reg_haz_rp.csv')
    
    # Count up the hh still in poverty or subsistence after reconstruction (10 years)
    pov_df_later,_ = average_over_rp(pov_df.loc[pov_df.eval('(c_initial>pov_line)&(c_post_reco<=pov_line)'),'pcwgt'].sum(level=event_level).to_frame(name='new_pov_perm'),'default_rp')
    pov_df_later.to_csv('../output_country/'+myC+'/permanent_cons_poverty_by_reg.csv')
    pov_df_later.sum().to_csv('../output_country/'+myC+'/permanent_cons_poverty.csv')

    # Average over RPs (index = region, hazard)
    try: pov_df_reg_haz,_ = average_over_rp(pov_df_event[['net_chg_pov_i','net_chg_sub_i',
                                                          'net_chg_pov_c','net_chg_sub_c',
                                                          'net_chg_pov_c_children']],'default_rp')
    except: pov_df_reg_haz,_ = average_over_rp(pov_df_event[['net_chg_pov_i','net_chg_sub_i','net_chg_pov_c','net_chg_sub_c']],'default_rp')
    

    #if myC == 'MW':
    #    pov_df_reg_haz.loc['Blantyre'] = pov_df_reg_haz.loc[['Blantyre','Blantyre City']].sum()
    #    pov_df_reg_haz.loc['Lilongwe'] = pov_df_reg_haz.loc[['Lilongwe','Lilongwe City']].sum()
    #    pov_df_reg_haz.loc['Mzimba'] = pov_df_reg_haz.loc[['Mzimba','Mzuzu City']].sum()
    #    pov_df_reg_haz.loc['Zomba Non-City'] = pov_df_reg_haz.loc[['Zomba Non-City','Zomba City']].sum() 
    #    pov_df_reg_haz = pov_df_reg_haz.drop(['Blantyre City','Lilongwe City', 'Mzuzu City', 'Zomba City'],axis=0)

    pov_df_reg_haz['reg_pop'] = pov_df_event['reg_pop'].mean(level=[event_level[0],'hazard'])
    pov_df_reg_haz['init_pov'] = pov_df_event['init_pov'].mean(level=[event_level[0],'hazard'])
    pov_df_reg_haz['init_sub'] = pov_df_event['init_sub'].mean(level=[event_level[0],'hazard'])
 
    # Number pushed into poverty *& subsistence as % of individuals already there
    pov_df_reg_haz['pct_increase_pov_c'] = 1000.*pov_df_reg_haz['net_chg_pov_c']/pov_df_reg_haz['init_pov']
    pov_df_reg_haz['pct_increase_sub_c'] = 1000.*pov_df_reg_haz['net_chg_sub_c']/pov_df_reg_haz['init_sub']
    pov_df_reg_haz['pct_increase_pov_i'] = 1000.*pov_df_reg_haz['net_chg_pov_i']/pov_df_reg_haz['init_pov']
    pov_df_reg_haz['pct_increase_sub_i'] = 1000.*pov_df_reg_haz['net_chg_sub_i']/pov_df_reg_haz['init_sub']

    # Number pushed into poverty *& subsistence as % of regional population
    pov_df_reg_haz['pct_pop_pov_c'] = 1000.*pov_df_reg_haz['net_chg_pov_c']/pov_df_reg_haz['reg_pop'].astype('float')
    pov_df_reg_haz['pct_pop_sub_c'] = 1000.*pov_df_reg_haz['net_chg_sub_c']/pov_df_reg_haz['reg_pop'].astype('float')
    pov_df_reg_haz['pct_pop_pov_i'] = 1000.*pov_df_reg_haz['net_chg_pov_i']/pov_df_reg_haz['reg_pop'].astype('float')
    pov_df_reg_haz['pct_pop_sub_i'] = 1000.*pov_df_reg_haz['net_chg_sub_i']/pov_df_reg_haz['reg_pop'].astype('float')
    #
    pov_df_reg_haz.to_csv('../output_country/'+myC+'/net_chg_pov_reg_haz.csv')
    #
    if False:
        regional_poverty['pct_pop_pov_c'] = pov_df_reg_haz.reset_index('hazard')['pct_pop_pov_c'].copy()
        regional_poverty = regional_poverty.dropna(how='any')
        plt.scatter(regional_poverty['pct_pop_pov_c']/10.,regional_poverty['poverty_rate'])
        plt.gcf().savefig('/Users/brian/Desktop/tmp/corr.pdf',format='pdf',bbox_inches='tight')

    # Write out latex tables by hazard
    for _typ, _haz in pov_df_reg_haz.reset_index().set_index(event_level[0]).groupby(['hazard']):
        _haz = _haz.copy()
        _haz.loc['Total'] = _haz.sum()
        _haz[['net_chg_pov_i','net_chg_sub_i',
              'net_chg_pov_c','net_chg_sub_c']].fillna(0).sort_values(['net_chg_pov_c'],ascending=False).astype('int').to_latex('latex/poverty_net_change_by_haz_'+str(_typ)+'.tex')

    # Sum over hazards (index = region)
    pov_df_region = pov_df_reg_haz[['net_chg_pov_i','net_chg_sub_i','net_chg_pov_c','net_chg_sub_c']].sum(level=event_level[0])
    try: pov_df_region['net_chg_pov_c_children'] = pov_df_reg_haz[['net_chg_pov_c_children']].sum(level=event_level[0])
    except: pass

    pov_df_region['reg_pop'] = pov_df_event['reg_pop'].mean(level=event_level[0])
    pov_df_region['init_pov'] = pov_df_event['init_pov'].mean(level=event_level[0])
    pov_df_region['init_sub'] = pov_df_event['init_sub'].mean(level=event_level[0])

    pov_df_region['pct_pop_pov_i'] = 100.*pov_df_region['net_chg_pov_i']/pov_df_region['reg_pop']
    pov_df_region['pct_pop_sub_i'] = 100.*pov_df_region['net_chg_sub_i']/pov_df_region['reg_pop']
    pov_df_region['pct_increase_pov_i'] = 100.*pov_df_region['net_chg_pov_i']/pov_df_region['init_pov']
    pov_df_region['pct_increase_sub_i'] = 100.*pov_df_region['net_chg_sub_i']/pov_df_region['init_sub']

    pov_df_region['pct_pop_pov_c'] = 100.*pov_df_region['net_chg_pov_c']/pov_df_region['reg_pop']
    pov_df_region['pct_pop_sub_c'] = 100.*pov_df_region['net_chg_sub_c']/pov_df_region['reg_pop']
    pov_df_region['pct_increase_pov_c'] = 100.*pov_df_region['net_chg_pov_c']/pov_df_region['init_pov']
    pov_df_region['pct_increase_sub_c'] = 100.*pov_df_region['net_chg_sub_c']/pov_df_region['init_sub']

    pov_df_region.to_csv('../output_country/'+myC+'/net_chg_pov_reg.csv')

    pov_df_region.loc['Total'] = pov_df_region.sum()
    pov_df_region.loc['Total',['pct_pop_pov_i']] = 100.*pov_df_region['net_chg_pov_i'].sum()/pov_df_region['reg_pop'].sum()
    pov_df_region.loc['Total',['pct_pop_sub_i']] = 100.*pov_df_region['net_chg_sub_i'].sum()/pov_df_region['reg_pop'].sum()
    pov_df_region.loc['Total',['pct_pop_pov_c']] = 100.*pov_df_region['net_chg_pov_c'].sum()/pov_df_region['reg_pop'].sum()
    pov_df_region.loc['Total',['pct_pop_sub_c']] = 100.*pov_df_region['net_chg_sub_c'].sum()/pov_df_region['reg_pop'].sum()

    pov_df_region.loc['Total',['pct_increase_pov_i']] = 100.*pov_df_region['net_chg_pov_i'].sum()/pov_df_region['init_pov'].sum()
    pov_df_region.loc['Total',['pct_increase_sub_i']] = 100.*pov_df_region['net_chg_sub_i'].sum()/pov_df_region['init_sub'].sum()
    pov_df_region.loc['Total',['pct_increase_pov_c']] = 100.*pov_df_region['net_chg_pov_c'].sum()/pov_df_region['init_pov'].sum()
    pov_df_region.loc['Total',['pct_increase_sub_c']] = 100.*pov_df_region['net_chg_sub_c'].sum()/pov_df_region['init_sub'].sum()

    pov_df_region[['net_chg_pov_i','net_chg_sub_i',
                   'net_chg_pov_c','net_chg_sub_c']] = (1E-3*pov_df_region[['net_chg_pov_i','net_chg_sub_i','net_chg_pov_c','net_chg_sub_c']]).round(1)
    pov_df_region[['pct_increase_pov_i','pct_increase_sub_i',
                   'pct_increase_pov_c','pct_increase_sub_c']] = pov_df_region[['pct_increase_pov_i','pct_increase_sub_i',
                                                                                'pct_increase_pov_c','pct_increase_sub_c']].round(1)

    pov_df_region[['net_chg_pov_c','pct_increase_pov_c','net_chg_sub_c','pct_increase_sub_c']].fillna(0).sort_values(['net_chg_pov_c'],ascending=False).to_latex('latex/poverty_all_haz.tex')

    # Sum over hazards (just totals left)
    _ = pov_df_region.reset_index().copy()
    pov_df_total = _.loc[_[event_level[0]]!='Total',['net_chg_pov_i','net_chg_sub_i','net_chg_pov_c','net_chg_sub_c','reg_pop','init_pov','init_sub']].sum()

    pov_df_total['pct_pop_pov_i'] = 1E3*100*pov_df_total['net_chg_pov_i']/pov_df_total['reg_pop']
    pov_df_total['pct_pop_sub_i'] = 1E3*100*pov_df_total['net_chg_sub_i']/pov_df_total['reg_pop']
    pov_df_total['pct_pop_pov_c'] = 1E3*100*pov_df_total['net_chg_pov_c']/pov_df_total['reg_pop']
    pov_df_total['pct_pop_sub_c'] = 1E3*100*pov_df_total['net_chg_sub_c']/pov_df_total['reg_pop']

    pov_df_total['pct_increase_pov_i'] = 1E3*100*pov_df_total['net_chg_pov_i']/pov_df_total['init_pov']
    pov_df_total['pct_increase_sub_i'] = 1E3*100*pov_df_total['net_chg_sub_i']/pov_df_total['init_sub']
    pov_df_total['pct_increase_pov_c'] = 1E3*100*pov_df_total['net_chg_pov_c']/pov_df_total['init_pov']
    pov_df_total['pct_increase_sub_c'] = 1E3*100*pov_df_total['net_chg_sub_c']/pov_df_total['init_sub']
    pov_df_total.to_csv('../output_country/'+myC+'/net_chg_pov.csv')

    # Plot poverty incidence for specific RPs
    pov_df_event = pov_df_event.reset_index(['hazard','rp'])

    rp_PH = [[1,1E0,''],
             [10,1E0,''],
             [25,1E3,' (thousands)'],
             [30,1E3,' (thousands)'],
             [50,1E3,' (thousands)'],
             [100,1E3,' (thousands)'],
             [200,1E3,' (thousands)'],
             [250,1E3,' (thousands)'],
             [500,1E3,' (thousands)'],
             [1000,1E3,' (thousands)']]

    for myDis in ['PF']:
        for myRP in [[10,1E0,'']]:

            make_map_from_svg(
                pov_df_event.loc[(pov_df_event.hazard==myDis)&(pov_df_event.rp==myRP[0]),'net_chg_pov_c']/(myRP[1]*100.), 
                svg_file,
                outname='new_poverty_incidence_'+myDis+'_'+str(myRP[0]),
                color_maper=plt.cm.get_cmap('Reds'), 
                label=dem+' pushed into consumption poverty by '+str(myRP[0])+'-yr '+myDis+myRP[2],
                new_title=dem+' pushed into consumption poverty by '+str(myRP[0])+'-yr '+myDis+myRP[2],
                do_qualitative=False,
                res=2000)
            
            make_map_from_svg(
                (pov_df_event.loc[(pov_df_event.hazard==myDis)&(pov_df_event.rp==myRP[0]),'net_chg_pov_c']
                 /pov_df_event.loc[(pov_df_event.hazard==myDis)&(pov_df_event.rp==myRP[0]),'reg_pop']),
                svg_file,
                outname='new_poverty_incidence_pct_'+myDis+'_'+str(myRP[0]),
                color_maper=plt.cm.get_cmap('Reds'), 
                label='Percent of regional pop. pushed into\nconsumption poverty by '+str(myRP[0])+'-yr '+myDis,
                new_title='Percent of regional pop. pushed into\nconsumption poverty by '+str(myRP[0])+'-yr '+myDis,
                do_qualitative=False,
                res=2000)

            plt.close('all')

    pov_df_region = pov_df_region.drop('Total')
    try:
        pov_df_region.to_csv('~/Desktop/tmp/children.csv')
        make_map_from_svg(
            pov_df_region['net_chg_pov_c_children']/1E3,
            svg_file,
            outname=myC+'_new_child_poverty_incidence_allHaz_allRPs',
            color_maper=plt.cm.get_cmap('Reds'), 
            label='Net change in children in consumption poverty\neach year from all hazards (thousands)',
            new_title='Net change in children in consumption poverty each year from all hazards',
            do_qualitative=True,
            res=2000)
    except: assert(False)
    assert(False)

    make_map_from_svg(
        pov_df_region['net_chg_pov_c']/1E3,
        svg_file,
        outname=myC+'_new_poverty_incidence_allHaz_allRPs',
        color_maper=plt.cm.get_cmap('Reds'), 
        label='Net change in '+dem+' in consumption poverty\neach year from all hazards (thousands)',
        new_title='Net change in '+dem+' in consumption poverty each year from all hazards',
        do_qualitative=False,
        res=2000)

    make_map_from_svg(
        1E2*1.E3*(pov_df_region['net_chg_pov_c']/pov_df_region['init_pov']), 
        svg_file,
        outname=myC+'_new_poverty_as_pct_of_incidence_pct_allHaz_allRPs',
        color_maper=plt.cm.get_cmap('Reds'), 
        label='Annual consumption poverty increase\nas % of regional poverty incidence',
        new_title= dem+' pushed into poverty by all hazards [%]',
        do_qualitative=False,
        res=2000)

    make_map_from_svg(
        1E2*1.E3*(pov_df_region['net_chg_pov_c']/pov_df_region.reg_pop), 
        svg_file,
        outname=myC+'_new_poverty_incidence_pct_allHaz_allRPs',
        color_maper=plt.cm.get_cmap('Reds'), 
        label='Annual consumption poverty increase\nas % of regional population',
        new_title= dem+' pushed into poverty by all hazards [%]',
        do_qualitative=False,
        res=2000)
    
    make_map_from_svg(
        pov_df_region['net_chg_sub_c']/1E3, 
        svg_file,
        outname=myC+'_new_subsistence_incidence_allHaz_allRPs',
        color_maper=plt.cm.get_cmap('Reds'), 
        label='Number of '+dem+' pushed into consumption subsistence\neach year by all hazards (thousands)',
        new_title='Number of '+dem+' pushed into consumption\nsubsistence each year by all hazards',
        do_qualitative=False,
        res=2000)
    
    make_map_from_svg(
        100.*pov_df_region.net_chg_sub_c/pov_df_region.reg_pop,
        svg_file,
        outname=myC+'_new_subsistence_incidence_pct_allHaz_allRPs',
        color_maper=plt.cm.get_cmap('Reds'), 
        label= dem+' pushed into consumption subsistence\neach year by all hazards [% of regional pop.]',
        new_title= dem+' pushed into consumption subsistence\nby all hazards [%]',
        do_qualitative=False,
        res=2000)

    purge('img/','map_of_*.png')
    purge('img/','legend_of_*.png')
    purge('img/','map_of_*.svg')
    purge('img/','legend_of_*.svg')
    
#run_poverty_tables_and_maps(None)
#map_recovery_time('PH')
