import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def housing_plots(RO_hhid,df,_df):
    # RO_hhid is the columns that are concatenated to form unique identifier for each hh
    # df is the dataframe with household characteristics, which we'll compare with housing class
    # _df is the raw dataframe with housing characteristics
    #print(df.head())
    #print(_df.head())

    _df = _df.dropna(subset=['MATCONS','ANNC'])
    _df['ANNC'] = _df['ANNC'].astype('int').clip(lower=1900)

    # Make histogram with year of construction, by construction material
    ax = plt.gca()
    year_bins = np.linspace(1900,2021,60)

    for thecat in _df.MATCONS.unique(): 

        heights, bins = np.histogram(_df.loc[_df['MATCONS']==thecat,'ANNC'],
                                     weights=1E-3*_df.loc[_df['MATCONS']==thecat,'hhwgt'],bins=year_bins)
        ax.bar(bins[:-1],heights, width=(bins[1]-bins[0]), align='edge', 
           #label=aReg+' - post-disaster', facecolor=paired_pal[4],edgecolor=None,
           linewidth=0,alpha=0.3,label=thecat)
        
    plt.legend()
    plt.xlabel('Construction year')
    plt.ylabel('Households (,000)')

    sns.despine()

    ax.get_figure().savefig('../output_plots/RO/construction_year_by_material.pdf',format='pdf')
    plt.cla()
    #assert(False)
