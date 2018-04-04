import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from libraries.maps_lib import make_map_from_svg, purge
from libraries.lib_average_over_rp import average_over_rp
from libraries.lib_country_dir import get_demonym
from libraries.lib_gather_data import match_percentiles,perc_with_spline,reshape_data
from libraries.lib_common_plotting_functions import title_legend_labels,sns_pal
import glob

purge('img/','map_of_*.png')
purge('img/','legend_of_*.png')
purge('img/','map_of_*.svg')
purge('img/','legend_of_*.svg')

col_cast_dict = {'new_pov':'int',
                 'pct_pov':'float64',
                 'new_sub':'int',
                 'pct_sub':'float64'}

def run_poverty_duration_plot(myC):

    df = pd.read_csv('../output_country/'+myC+'/poverty_duration_no.csv')
    df['country'] = myC

    listofdeciles=np.arange(0.10, 1.01, 0.10)
    df = df.reset_index().groupby('country',sort=True).apply(lambda x:match_percentiles(x,perc_with_spline(reshape_data(x.c),reshape_data(x.pcwgt),listofdeciles),'decile'))

    # Load additional SP runs
    _pols = ['unif_poor']
    for iSP in _pols:
        _ = pd.read_csv('../output_country/'+myC+'/poverty_duration_'+iSP+'.csv')
        df[['t_pov_inc_'+iSP,'t_pov_cons_'+iSP,'t_pov_bool_'+iSP]] = _[['t_pov_inc','t_pov_cons','t_pov_bool']]

    # Do some plotting
    plot_crit = '(t_pov_bool>0)&(hazard=="PF")&(rp==500)'

    df.loc[df.eval(plot_crit)].plot.hexbin('dk0','t_pov_cons')
    #plt.gca().get_figure().savefig('../output_plots/'+myC+'/poverty_duration_hexbin_no.pdf',format='pdf')
    plt.cla()

    df.loc[df.eval(plot_crit)].plot.scatter('dk0','t_pov_cons')
    #plt.gca().get_figure().savefig('../output_plots/'+myC+'/poverty_duration_scatter_no.pdf',format='pdf')
    plt.cla()    

    df = df.reset_index().set_index(['hazard','rp','decile'])
    print(df.head())

    df_dec = pd.DataFrame(index=df.sum(level=['hazard','rp','decile']).index)
    # Populate the df_dec dataframe now, while its index is set to ['hazard','rp','decile']

    # Number of individuals who face income or consumption poverty
    df_dec['n_new_pov_inc']  = df.loc[df.t_pov_inc !=0,'pcwgt'].sum(level=['hazard','rp','decile'])
    df_dec['n_new_pov_cons'] = df.loc[df.t_pov_cons!=0,'pcwgt'].sum(level=['hazard','rp','decile'])

    # Individuals who face income or consumption poverty as fraction of all individuals
    df_dec['frac_new_pov_inc'] = df_dec['n_new_pov_inc']/df['pcwgt'].sum(level=['hazard','rp','decile'])
    df_dec['frac_new_pov_cons'] = df_dec['n_new_pov_cons']/df['pcwgt'].sum(level=['hazard','rp','decile'])

    # Among people pushed into pov: average time in poverty (months)
    df_dec['t_pov_inc_avg'] = 12.*(df.loc[df.eval('t_pov_bool==True'),['pcwgt','t_pov_inc']].prod(axis=1).sum(level=['hazard','rp','decile'])
                               /df.loc[df.eval('t_pov_bool==True'),'pcwgt'].sum(level=['hazard','rp','decile']))
    df_dec['t_pov_cons_avg'] = 12.*(df.loc[df.eval('t_pov_bool==True'),['pcwgt','t_pov_cons']].prod(axis=1).sum(level=['hazard','rp','decile'])
                               /df.loc[df.eval('t_pov_bool==True'),'pcwgt'].sum(level=['hazard','rp','decile']))

    focus = ['Rathnapura','Colombo','Kandy','Gampaha']
    for iloc in focus:
        df_dec['t_pov_inc_avg_'+iloc] = 12.*(df.loc[df.eval('(t_pov_bool==True)&(district==@iloc)'),['pcwgt','t_pov_inc']].prod(axis=1).sum(level=['hazard','rp','decile'])
                                             /df.loc[df.eval('(t_pov_bool==True)&(district==@iloc)'),'pcwgt'].sum(level=['hazard','rp','decile']))

        df_dec['t_pov_cons_avg_'+iloc] = 12.*(df.loc[df.eval('(t_pov_bool==True)&(district==@iloc)'),['pcwgt','t_pov_cons']].prod(axis=1).sum(level=['hazard','rp','decile'])
                                              /df.loc[df.eval('(t_pov_bool==True)&(district==@iloc)'),'pcwgt'].sum(level=['hazard','rp','decile']))

    df_dec = df_dec.reset_index()
    df_dec.to_csv('../output_country/'+myC+'/poverty_by_decile.csv')

    ######################
    # Plot consumption and income poverty (separately)
    _lab = {'t_pov_cons_avg':'Average time to exit poverty\n(income net of reconstruction & savings) [months]',
            't_pov_inc_avg':'Average time to exit poverty (income only) [months]'}

    for ipov in ['t_pov_cons_avg','t_pov_inc_avg']:
        # Do the plotting
        ax = df_dec.loc[df_dec.eval('(hazard=="PF")&(rp==5)')].plot.scatter('decile',ipov,color=sns_pal[1],lw=0,label='Natl. average (RP = 5 years)',zorder=99)
        df_dec.loc[df_dec.eval('(hazard=="PF")&(rp==1000)')].plot.scatter('decile',ipov,color=sns_pal[3],lw=0,label='Natl. average (RP = 1000 years)',zorder=98,ax=ax)

        df_dec.loc[df_dec.eval('(hazard=="PF")&(rp==5)')].plot('decile',ipov,color=sns_pal[1],zorder=97,label='',ax=ax)
        df_dec.loc[df_dec.eval('(hazard=="PF")&(rp==1000)')].plot('decile',ipov,color=sns_pal[3],zorder=96,label='',ax=ax)

        icol = 4
        for iloc in focus:
            df_dec.loc[df_dec.eval('(hazard=="PF")&(rp==5)')].plot.scatter('decile',ipov+'_'+iloc,color=sns_pal[icol],lw=0,label=iloc+' (RP = 5 years)',zorder=95,ax=ax)
            df_dec.loc[df_dec.eval('(hazard=="PF")&(rp==5)')].plot('decile',ipov+'_'+iloc,color=sns_pal[icol],zorder=94,label='',ax=ax)
            icol+=1

        # Do the formatting
        ax = title_legend_labels(ax,'Precipitation flood in '+myC,lab_x='Decile',lab_y=_lab[ipov],lim_x=[0.5,10.5],lim_y=[-0.1,42],leg_fs=9)
        ax.xaxis.set_ticks([1,2,3,4,5,6,7,8,9,10])
        ax.yaxis.set_ticks([0,6,12,18,24,30,36,42])

        # Do the saving
        ax.get_figure().savefig('../output_plots/'+myC+'/'+ipov+'_by_decile.pdf',format='pdf')
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
    elif myC == 'SL':
        svg_file = '../map_files/'+myC+'/lk.svg'

    # Get the poverty headcount info
    try:
        # Count up the hh that fell into poverty & subsistence:
        pov_df_event = pov_df.loc[pov_df.eval('(c_initial>pov_line)&(i_pre_reco<=pov_line)&(i_pre_reco>sub_line)'),'pcwgt'].sum(level=event_level).to_frame(name='new_pov')
        pov_df_event['new_sub'] = pov_df.loc[pov_df.eval('(c_initial>sub_line)&(i_pre_reco<=sub_line)'),'pcwgt'].sum(level=event_level).fillna(0)

        pov_df_event['init_pov'] = pov_df.loc[pov_df.eval('(c_initial<=pov_line)&(c_initial>sub_line)'),'pcwgt'].sum(level=event_level).fillna(0)
        pov_df_event['init_sub'] = pov_df.loc[pov_df.eval('(c_initial<=sub_line)'),'pcwgt'].sum(level=event_level).fillna(0)        

        pov_df_event['reg_pop'] = pov_df['pcwgt'].sum(level=event_level)
                                             
        pov_df_event.to_csv('tmp/new_pov_reg_haz_rp.csv')
        
        # Count up the hh still in poverty or subsistence after reconstruction (10 years)
        pov_df_later,_ = average_over_rp(pov_df.loc[pov_df.eval('(c_initial>pov_line)&(c_post_reco<=pov_line)'),'pcwgt'].sum(level=event_level).to_frame(name='new_pov_perm'),'default_rp')
        pov_df_later.to_csv('tmp/permanent_poverty_by_reg.csv')
        pov_df_later.sum().to_csv('tmp/permanent_poverty.csv')

    except:
        try: 
            pov_df_event = pd.read_csv('tmp/new_pov_reg_haz_rp.csv', index_col=event_level)
            print('working with saved file')
        except: print('\n\n***Could not load poverty info***\n\n'); return False
 
    # Average over RPs (index = region, hazard)
    pov_df_reg_haz,_ = average_over_rp(pov_df_event[['new_pov','new_sub']],'default_rp')

    pov_df_reg_haz['reg_pop'] = pov_df_event['reg_pop'].mean(level=[event_level[0],'hazard'])
    pov_df_reg_haz['init_pov'] = pov_df_event['init_pov'].mean(level=[event_level[0],'hazard'])
    pov_df_reg_haz['init_sub'] = pov_df_event['init_sub'].mean(level=[event_level[0],'hazard'])
 
    # Number pushed into poverty *& subsistence as % of individuals already there
    pov_df_reg_haz['pct_increase_pov'] = 1000.*pov_df_reg_haz['new_pov']/pov_df_reg_haz['init_pov']# Remember to divide by 10 later
    pov_df_reg_haz['pct_increase_sub'] = 1000.*pov_df_reg_haz['new_sub']/pov_df_reg_haz['init_sub']# Remember to divide by 10 later

    # Number pushed into poverty *& subsistence as % of regional population
    pov_df_reg_haz['pct_pop_pov'] = 1000.*pov_df_reg_haz['new_pov']/pov_df_reg_haz['reg_pop'].astype('float')# Remember to divide by 10 later
    pov_df_reg_haz['pct_pop_sub'] = 1000.*pov_df_reg_haz['new_sub']/pov_df_reg_haz['reg_pop'].astype('float')# Remember to divide by 10 later

    pov_df_reg_haz.to_csv('tmp/new_pov_reg_haz.csv')

    # Write out latex tables by hazard
    for _typ, _haz in pov_df_reg_haz.reset_index().set_index(event_level[0]).groupby(['hazard']):
        _haz = _haz.copy()
        _haz.loc['Total'] = _haz.sum()
        _haz[['new_pov','new_sub']].fillna(0).sort_values(['new_pov'],ascending=False).astype('int').to_latex('latex/poverty_by_haz_'+str(_typ)+'.tex')

    # Sum over hazards (index = region)
    pov_df_region = pov_df_reg_haz[['new_pov','new_sub']].sum(level=event_level[0]).astype('int')
    pov_df_region['reg_pop'] = pov_df_event['reg_pop'].mean(level=event_level[0])
    pov_df_region['init_pov'] = pov_df_event['init_pov'].mean(level=event_level[0])
    pov_df_region['init_sub'] = pov_df_event['init_sub'].mean(level=event_level[0])

    pov_df_region['pct_pop_pov'] = 1000.*pov_df_region['new_pov']/pov_df_region['reg_pop']# Remember to divide by 10 later
    pov_df_region['pct_pop_sub'] = 1000.*pov_df_region['new_sub']/pov_df_region['reg_pop']# Remember to divide by 10 later
    pov_df_region['pct_increase_pov'] = 1000.*pov_df_region['new_pov']/pov_df_region['init_pov']# Remember to divide by 10 later
    pov_df_region['pct_increase_sub'] = 1000.*pov_df_region['new_sub']/pov_df_region['init_sub']# Remember to divide by 10 later
    pov_df_region.to_csv('tmp/new_pov_reg.csv')

    pov_df_region.loc['Total'] = pov_df_region.sum()
    pov_df_region.loc['Total',['pct_pop_pov']] = round(1000.*pov_df_region['new_pov'].sum()/pov_df_region['reg_pop'].sum(),0)# Remember to divide by 10 later
    pov_df_region.loc['Total',['pct_pop_sub']] = round(1000.*pov_df_region['new_sub'].sum()/pov_df_region['reg_pop'].sum(),0)# Remember to divide by 10 later
    pov_df_region.loc['Total',['pct_increase_pov']] = round(1000.*pov_df_region['new_pov'].sum()/pov_df_region['init_pov'].sum(),0)# Remember to divide by 10 later
    pov_df_region.loc['Total',['pct_increase_sub']] = round(1000.*pov_df_region['new_sub'].sum()/pov_df_region['init_sub'].sum(),0)# Remember to divide by 10 later

    pov_df_region[['new_pov','pct_increase_pov','new_sub','pct_increase_sub']].fillna(0).sort_values(['new_pov'],ascending=False).astype('int').to_latex('latex/poverty_all_haz.tex')

    # Sum over hazards (just totals left)
    _ = pov_df_region.reset_index().copy()
    pov_df_total = _.loc[_[event_level[0]]!='Total',['new_pov','new_sub','reg_pop','init_pov','init_sub']].sum()
    pov_df_total['pct_pop_pov'] = 100*pov_df_total['new_pov']/pov_df_total['reg_pop']
    pov_df_total['pct_pop_sub'] = 100*pov_df_total['new_sub']/pov_df_total['reg_pop']
    pov_df_total['pct_increase_pov'] = 100*pov_df_total['new_pov']/pov_df_total['init_pov']
    pov_df_total['pct_increase_sub'] = 100*pov_df_total['new_sub']/pov_df_total['init_sub']
    pov_df_total.to_csv('tmp/new_pov.csv')

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
                pov_df_event.loc[(pov_df_event.hazard==myDis)&(pov_df_event.rp==myRP[0]),['new_pov','new_sub']].sum(axis=1)/(myRP[1]*100.), 
                svg_file,
                outname='new_poverty_incidence_'+myDis+'_'+str(myRP[0]),
                color_maper=plt.cm.get_cmap('RdYlGn_r'), 
                label=dem+' pushed into poverty by '+str(myRP[0])+'-yr '+myDis+myRP[2],
                new_title=dem+' pushed into poverty by '+str(myRP[0])+'-yr '+myDis+myRP[2],
                do_qualitative=False,
                res=2000)
            
            make_map_from_svg(
                (pov_df_event.loc[(pov_df_event.hazard==myDis)&(pov_df_event.rp==myRP[0]),['new_pov','new_sub']].sum(axis=1)
                 /pov_df_event.loc[(pov_df_event.hazard==myDis)&(pov_df_event.rp==myRP[0]),'reg_pop']),
                svg_file,
                outname='new_poverty_incidence_pct_'+myDis+'_'+str(myRP[0]),
                color_maper=plt.cm.get_cmap('RdYlGn_r'), 
                label='Percent of regional pop. pushed into poverty by '+str(myRP[0])+'-yr '+myDis,
                new_title='Percent of regional pop. pushed into poverty by '+str(myRP[0])+'-yr '+myDis,
                do_qualitative=False,
                res=2000)

            plt.close('all')

    make_map_from_svg(
        pov_df_region[['new_pov','new_sub']].sum(axis=1)/(1E3*100.),
        svg_file,
        outname=myC+'_new_poverty_incidence_allHaz_allRPs',
        color_maper=plt.cm.get_cmap('RdYlGn_r'), 
        label='Number of '+dem+' pushed into poverty each year by all hazards (thousands)',
        new_title='Number of '+dem+' pushed into poverty each year by all hazards',
        do_qualitative=False,
        res=2000)
    
    make_map_from_svg(
        pov_df_region[['new_pov','new_sub']].sum(axis=1)/pov_df_region.reg_pop, 
        svg_file,
        outname=myC+'_new_poverty_incidence_pct_allHaz_allRPs',
        color_maper=plt.cm.get_cmap('RdYlGn_r'), 
        label=dem+' pushed into poverty each year by all hazards [% of regional pop.]',
        new_title= dem+' pushed into poverty by all hazards [%]',
        do_qualitative=False,
        res=2000)
    
    make_map_from_svg(
        pov_df_region['new_sub']/(100.*1E3), 
        svg_file,
        outname=myC+'_new_subsistence_incidence_allHaz_allRPs',
        color_maper=plt.cm.get_cmap('RdYlGn_r'), 
        label='Number of '+dem+' pushed into subsistence each year by all hazards (thousands)',
        new_title='Number of '+dem+' pushed into subsistence each year by all hazards',
        do_qualitative=False,
        res=2000)
    
    make_map_from_svg(
        pov_df_region.new_sub/pov_df_region.reg_pop,
        svg_file,
        outname=myC+'_new_subsistence_incidence_pct_allHaz_allRPs',
        color_maper=plt.cm.get_cmap('RdYlGn_r'), 
        label= dem+' pushed into subsistence each year by all hazards [% of regional pop.]',
        new_title= dem+' pushed into subsistence by all hazards [%]',
        do_qualitative=False,
        res=2000)

    purge('img/','map_of_*.png')
    purge('img/','legend_of_*.png')
    purge('img/','map_of_*.svg')
    purge('img/','legend_of_*.svg')
    
#run_poverty_tables_and_maps(None)
