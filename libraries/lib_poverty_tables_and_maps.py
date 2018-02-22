import pandas as pd
import matplotlib.pyplot as plt
from libraries.maps_lib import make_map_from_svg, purge
from libraries.lib_average_over_rp import average_over_rp
from libraries.lib_country_dir import get_demonym
import glob

purge('img/','map_of_*.png')
purge('img/','legend_of_*.png')
purge('img/','map_of_*.svg')
purge('img/','legend_of_*.svg')

col_cast_dict = {'new_pov':'int',
                 'pct_pov':'float64',
                 'new_sub':'int',
                 'pct_sub':'float64'}

def run_poverty_tables_and_maps(pov_df,event_level=['region','hazard','rp'],myC='PH'):

    # Load demonym for this country
    dem = get_demonym(myC)

    # Look for the map (svg) file here
    svg_file = '../map_files/'+myC+'/BlankSimpleMap.svg'
    if myC == 'PH' and event_level[0] == 'region':
        svg_file = '../map_files/'+myC+'/BlankSimpleMapRegional.svg'    

    # Get the poverty headcount info
    try:
        # Count up the hh that fell into poverty & subsistence:
        pov_df_event = pov_df.loc[pov_df.eval('(c_initial>pov_line)&(i_pre_reco<=pov_line)&(i_pre_reco>sub_line)'),'pcwgt'].sum(level=event_level).to_frame(name='new_pov')
        pov_df_event['new_sub'] = pov_df.loc[pov_df.eval('(c_initial>sub_line)&(i_pre_reco<=sub_line)'),'pcwgt'].sum(level=event_level).fillna(0)

        pov_df_event['init_pov'] = pov_df.loc[pov_df.eval('(c_initial<=pov_line)&(c_initial>sub_line)'),'pcwgt'].sum(level=event_level).fillna(0)
        pov_df_event['init_sub'] = pov_df.loc[pov_df.eval('(c_initial<=sub_line)'),'pcwgt'].sum(level=event_level).fillna(0)        

        pov_df_event['reg_pop'] = pov_df['pcwgt'].sum(level=event_level)
                                             
        pov_df_event.to_csv('debug/new_pov_reg_haz_rp.csv')
        
        # Count up the hh still in poverty or subsistence after reconstruction (10 years)
        pov_df_later,_ = average_over_rp(pov_df.loc[pov_df.eval('(c_initial>pov_line)&(c_post_reco<=pov_line)'),'pcwgt'].sum(level=event_level).to_frame(name='new_pov_perm'),'default_rp')
        pov_df_later.sum().to_csv('debug/permanent_poverty.csv')

    except:
        try: 
            pov_df_event = pd.read_csv('debug/new_pov_reg_haz_rp.csv', index_col=event_level)
            print('working with saved file')
        except: print('\n\n***Could not load poverty info***\n\n'); return False
 
    # Average over RPs (index = region, hazard)
    pov_df_reg_haz,_ = average_over_rp(pov_df_event[['new_pov','new_sub']],'default_rp')

    pov_df_reg_haz['reg_pop'] = pov_df_event['reg_pop'].mean(level=['region','hazard'])
    pov_df_reg_haz['init_pov'] = pov_df_event['init_pov'].mean(level=['region','hazard'])
    pov_df_reg_haz['init_sub'] = pov_df_event['init_sub'].mean(level=['region','hazard'])
 
    # Number pushed into poverty *& subsistence as % of individuals already there
    pov_df_reg_haz['pct_increase_pov'] = 1000.*pov_df_reg_haz['new_pov']/pov_df_reg_haz['init_pov']# Remember to divide by 10 later
    pov_df_reg_haz['pct_increase_sub'] = 1000.*pov_df_reg_haz['new_sub']/pov_df_reg_haz['init_sub']# Remember to divide by 10 later

    # Number pushed into poverty *& subsistence as % of regional population
    pov_df_reg_haz['pct_pop_pov'] = 1000.*pov_df_reg_haz['new_pov']/pov_df_reg_haz['reg_pop'].astype('float')# Remember to divide by 10 later
    pov_df_reg_haz['pct_pop_sub'] = 1000.*pov_df_reg_haz['new_sub']/pov_df_reg_haz['reg_pop'].astype('float')# Remember to divide by 10 later

    pov_df_reg_haz.to_csv('debug/new_pov_reg_haz.csv')

    # Write out latex tabels by hazard
    for _typ, _haz in pov_df_reg_haz.reset_index().set_index('region').groupby(['hazard']):
        _haz = _haz.copy()
        _haz.loc['Total'] = _haz.sum()
        _haz[['new_pov','new_sub']].fillna(0).sort_values(['new_pov'],ascending=False).astype('int').to_latex('debug/poverty_by_haz_'+str(_typ)+'.tex')

    # Sum over hazards (index = region)
    pov_df_region = pov_df_reg_haz[['new_pov','new_sub']].sum(level='region').astype('int')
    pov_df_region['reg_pop'] = pov_df_event['reg_pop'].mean(level=event_level[0])
    pov_df_region['init_pov'] = pov_df_event['init_pov'].mean(level=event_level[0])
    pov_df_region['init_sub'] = pov_df_event['init_sub'].mean(level=event_level[0])

    pov_df_region['pct_pop_pov'] = 1000.*pov_df_region['new_pov']/pov_df_region['reg_pop']# Remember to divide by 10 later
    pov_df_region['pct_pop_sub'] = 1000.*pov_df_region['new_sub']/pov_df_region['reg_pop']# Remember to divide by 10 later
    pov_df_region['pct_increase_pov'] = 1000.*pov_df_region['new_pov']/pov_df_region['init_pov']# Remember to divide by 10 later
    pov_df_region['pct_increase_sub'] = 1000.*pov_df_region['new_sub']/pov_df_region['init_sub']# Remember to divide by 10 later
    pov_df_region.to_csv('debug/new_pov_reg.csv')

    pov_df_region.loc['Total'] = pov_df_region.sum()
    pov_df_region.loc['Total',['pct_pop_pov']] = 1000.*pov_df_region['new_pov'].sum()/pov_df_region['reg_pop'].sum()# Remember to divide by 10 later
    pov_df_region.loc['Total',['pct_pop_sub']] = 1000.*pov_df_region['new_sub'].sum()/pov_df_region['reg_pop'].sum()# Remember to divide by 10 later
    pov_df_region.loc['Total',['pct_increase_pov']] = 1000.*pov_df_region['new_pov'].sum()/pov_df_region['init_pov'].sum()# Remember to divide by 10 later
    pov_df_region.loc['Total',['pct_increase_sub']] = 1000.*pov_df_region['new_sub'].sum()/pov_df_region['init_sub'].sum()# Remember to divide by 10 later

    pov_df_region[['new_pov','pct_increase_pov','new_sub','pct_increase_sub']].fillna(0).sort_values(['new_pov'],ascending=False).astype('int').to_latex('debug/poverty_all_haz.tex')

    # Sum over hazards (just totals left)
    _ = pov_df_region.reset_index().copy()
    pov_df_total = _.loc[_.region!='Total',['new_pov','new_sub','reg_pop','init_pov','init_sub']].sum()
    pov_df_total['pct_pop_pov'] = 100*pov_df_total['new_pov']/pov_df_total['reg_pop']
    pov_df_total['pct_pop_sub'] = 100*pov_df_total['new_sub']/pov_df_total['reg_pop']
    pov_df_total['pct_increase_pov'] = 100*pov_df_total['new_pov']/pov_df_total['init_pov']
    pov_df_total['pct_increase_sub'] = 100*pov_df_total['new_sub']/pov_df_total['init_sub']
    pov_df_total.to_csv('debug/new_pov.csv')
    assert(False)

    # Plot poverty incidence for specific RPs
    pov_df_event = pov_df_event.reset_index(['hazard','rp'])

    for myDis in ['EQ','HU']:
        for myRP in [[1,1E0,''],
                     [10,1E0,''],
                     [25,1E3,' (thousands)'],
                     [30,1E3,' (thousands)'],
                     [50,1E3,' (thousands)'],
                     [100,1E3,' (thousands)'],
                     [200,1E3,' (thousands)'],
                     [250,1E3,' (thousands)'],
                     [500,1E3,' (thousands)'],
                     [1000,1E3,' (thousands)']]:

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
