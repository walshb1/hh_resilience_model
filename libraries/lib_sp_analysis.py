import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from libraries.lib_common_plotting_functions import q_colors, paired_pal

def run_sp_analysis(myCountry,cat_info):
    """Runs analysis on social payments, calculates for each economy
    1. The fraction of people receiving social payments
    2. The average value of those social payments, averaged over both total population and only those who receive some kind of social payment
    3. Proportion of social payment over total income,
    """

    economy = cat_info.index.names[0]
    cat_info = cat_info.reset_index().set_index(['hhid',economy])

    # Create empty index with just district keys
    sp_out = pd.DataFrame(index=cat_info.sum(level=economy).index.copy())
    # This caused an error where nothing is actually computed
    # sp_out.index = sp_out.index.astype(str)

    # fraction of poeple receiving social payments is ratio of people receiving social payments over total
    sp_out['frac_receive_SP'] = 100*cat_info.loc[cat_info['pcsoc']!=0,'pcwgt'].sum(level=economy)/cat_info['pcwgt'].sum(level=economy) # Doesn't work because sp_out key is a string
    # average value of social payments is total of all payments over population receiving socila payments
    sp_out['avg_value_SP'] = cat_info.loc[cat_info['pcsoc']!=0,['pcsoc','pcwgt']].prod(axis=1).sum(level=economy)/cat_info.loc[cat_info['pcsoc']!=0,'pcwgt'].sum(level=economy)
    # average value of social payments all is total of all payments over ALL population
    sp_out['avg_value_SPall'] = cat_info[['pcsoc','pcwgt']].prod(axis=1).sum(level=economy)/cat_info['pcwgt'].sum(level=economy)
    # household-weighted social payments over consumption by district, for only persons receiving social payments
    sp_out['sp_over_c'] = cat_info.loc[cat_info['pcsoc']!=0].eval('pcwgt*pcsoc/c').sum(level=economy)/cat_info.loc[cat_info['pcsoc']!=0,'pcwgt'].sum(level=economy)
    # household-weighted social payments over consumption by district, for all population
    sp_out['sp_over_call'] = cat_info.eval('pcwgt*pcsoc/c').sum(level=economy)/cat_info['pcwgt'].sum(level=economy)

    # Calculation of national averages for comparison
    sp_out.loc['NATL_AVG','frac_receive_SP'] = 100*cat_info.loc[cat_info['pcsoc']!=0,'pcwgt'].sum()/cat_info['pcwgt'].sum()
    sp_out.loc['NATL_AVG','avg_value_SP'] = cat_info.loc[cat_info['pcsoc']!=0,['pcsoc','pcwgt']].prod(axis=1).sum()/cat_info.loc[cat_info['pcsoc']!=0,'pcwgt'].sum()
    sp_out.loc['NATL_AVG','avg_value_SPall'] = cat_info[['pcsoc','pcwgt']].prod(axis=1).sum()/cat_info['pcwgt'].sum()
    sp_out.loc['NATL_AVG','sp_over_c'] = cat_info.loc[cat_info['pcsoc']!=0,['pcsoc','pcwgt']].prod(axis=1).sum()/cat_info.loc[cat_info['pcsoc']!=0,['c','pcwgt']].prod(axis=1).sum()
    sp_out.loc['NATL_AVG','sp_over_call'] = cat_info.eval('pcwgt*pcsoc/c').sum()/cat_info['pcwgt'].sum()

    if myCountry == 'SL':
        # fraction of population receiving samurdhi
        sp_out['frac_receive_samurdhi'] = 100*cat_info.loc[cat_info['pcsamurdhi']!=0,'pcwgt'].sum(level=economy)/cat_info['pcwgt'].sum(level=economy)

        # Plot coverage by ethnicity
        cat_info = cat_info.reset_index().set_index(['ethnicity']).sort_index()
        plt.scatter(cat_info.sum(level='ethnicity').index.values,
                    100*cat_info.loc[cat_info['pcsamurdhi']>0,'pcwgt'].sum(level='ethnicity')/cat_info['pcwgt'].sum(level='ethnicity'),
                    color=q_colors)

        plt.xticks(np.linspace(1,5,5),size=10)
        plt.gca().set_xticklabels(['Sinhala','Sri Lankan\nTamil','Indian\nTamil','Sri Lankan\nMoor','Malay'],size=10)

        plt.xlim(0.5,5.5)
        plt.ylabel('Population coverage in Samurdhi [%]',labelpad=8)
        plt.ylim(0,30)

        sns.despine(bottom=True)
        plt.grid(False)
        plt.gcf().savefig('../output_plots/SL/samurdhi/coverage_by_ethnicity.pdf',format='pdf',bbox_inches='tight')

        # Plot value by ethnicity
        plt.close('all')
        cat_info = cat_info.reset_index().set_index(['ethnicity']).sort_index()
        plt.scatter(cat_info.sum(level='ethnicity').index.values[:5],
                    (1/12)*cat_info.loc[cat_info['pcsamurdhi']>0,['pcsamurdhi','pcwgt']].prod(axis=1).sum(level='ethnicity')/cat_info.loc[cat_info['pcsamurdhi']>0,'pcwgt'].sum(level='ethnicity'),
                    color=q_colors)

        plt.xticks(np.linspace(1,5,5),size=10)
        plt.gca().set_xticklabels(['Sinhala','Sri Lankan\nTamil','Indian\nTamil','Sri Lankan\nMoor','Malay'],size=10)

        plt.xlim(0.5,5.5)
        plt.ylabel('Enrolled population, average receipt\n[LKR per person, per month]',labelpad=8)
        plt.ylim(0)

        sns.despine(bottom=True)
        plt.grid(False)
        plt.gcf().savefig('../output_plots/SL/samurdhi/value_by_ethnicity.pdf',format='pdf',bbox_inches='tight')

        # Plot enrollment by religion
        plt.close('all')
        cat_info = cat_info.reset_index().set_index(['religion']).sort_index()
        plt.scatter(cat_info.sum(level='religion').index.values,
                    100*cat_info.loc[cat_info['pcsamurdhi']>0,'pcwgt'].sum(level='religion')/cat_info['pcwgt'].sum(level='religion'),
                    color=[paired_pal[1],paired_pal[3],paired_pal[5],paired_pal[7]])

        plt.xticks(np.linspace(1,4,4),size=10)
        plt.gca().set_xticklabels(['Buddhist','Hindu','Muslim','Christian'],size=10)

        plt.xlim(0.5,4.5)
        plt.ylabel('Population coverage in Samurdhi [%]',labelpad=8)
        plt.ylim(0,30)

        sns.despine(bottom=True)
        plt.grid(False)
        plt.gcf().savefig('../output_plots/SL/samurdhi/enrollment_by_religion.pdf',format='pdf',bbox_inches='tight')

        # Plot value by religion
        plt.close('all')
        cat_info = cat_info.reset_index().set_index(['religion']).sort_index()
        plt.scatter(cat_info.sum(level='religion').index.values[:4],
                    (1/12)*cat_info.loc[cat_info['pcsamurdhi']>0,['pcsamurdhi','pcwgt']].prod(axis=1).sum(level='religion')/cat_info.loc[cat_info['pcsamurdhi']>0,'pcwgt'].sum(level='religion'),
                    color=[paired_pal[1],paired_pal[3],paired_pal[5],paired_pal[7]])

        plt.xticks(np.linspace(1,4,4),size=10)
        plt.gca().set_xticklabels(['Buddhist','Hindu','Muslim','Christian'],size=10)

        plt.xlim(0.5,4.5)
        plt.ylabel('Enrolled population, average receipt\n[LKR per person, per month]',labelpad=8)
        plt.ylim(0)

        sns.despine(bottom=True)
        plt.grid(False)
        plt.gcf().savefig('../output_plots/SL/samurdhi/value_by_religion.pdf',format='pdf',bbox_inches='tight')

    sp_out.to_csv('../output_country/'+myCountry+'/sp_receipts_by_region.csv')
