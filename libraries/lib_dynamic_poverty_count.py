import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from libraries.lib_country_dir import get_poverty_line

def dynamic_poverty_count(myC='PH'):
    df = pd.read_csv('../output_country/'+myC+'/poverty_duration_no.csv')
    
    df = df.loc[(df.hazard=='HU')&(df.rp==10)]
    df = df.loc[(df.region == 'I - Ilocos')|(df.region=='II - Cagayan Valley')|(df.region=='CAR')]

    df = df.reset_index().set_index(['region','hazard','rp']).drop([_c for _c in ['index','ratio_dw_lim_tot',
                                                                                  'dk_sub','dk_lim',
                                                                                  'dw_lim','dw_sub',
                                                                                  'res_lim','res_sub'] if _c in df.columns],axis=1)
    print(df.head())

    n_pov = []
    for _wk in np.linspace(0,521,450)/52.:
        n_pov.append(1E-3*df.loc[(df['c']>get_poverty_line(myC))&(df['t_pov_cons']>_wk),'pcwgt'].sum())

    plt.plot(np.linspace(0,521,450)/52.,n_pov)
    plt.xlim(0,10.2); plt.ylim(0)

    plt.xlabel('Years after disaster',weight='bold',labelpad=8)
    plt.ylabel('Filipinos in consumption poverty\ndue to wind damages in Ompong (,000)',weight='bold',labelpad=8)
    sns.despine()
    plt.grid(False)

    plt.gcf().savefig('/Users/brian/Desktop/Dropbox/Bank/ompong/plots/t_pov_ompong.pdf',format='pdf')

    
    print(df.loc[df['c']>get_poverty_line(myC),'pcwgt'].sum())
    print(df.loc[(df['c']>get_poverty_line(myC))&(df['t_pov_cons']>12/52),'pcwgt'].sum())
    assert(False)

dynamic_poverty_count()
