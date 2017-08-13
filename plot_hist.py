import os
import pandas as pd
from lib_gather_data import *
import matplotlib.pyplot as plt

import seaborn as sns
sns.set_style('darkgrid')
sns_pal = sns.color_palette('Set1', n_colors=8, desat=.5)

def plot_simple_hist(df,cols,labels,fout,nBins=50,uclip=None,lclip=None):
    q_colors = [sns_pal[0],sns_pal[1],sns_pal[2],sns_pal[3],sns_pal[5]]

    plt.cla()
    ax = plt.gca()

    bin0 = None
    for nCol, aCol in enumerate(cols):
        heights, bins = np.histogram(df[aCol].clip(upper=uclip,lower=lclip), bins=nBins, weights=df[['hhwgt']].prod(axis=1))

        if bin0 == None: bin0 = bins
        ax.bar(bin0[:-1], heights, width=(bin0[1]-bin0[0]), label=labels[nCol], facecolor=q_colors[nCol],alpha=0.4)
        
    fig = ax.get_figure()
    plt.xlabel(r'Income')
    plt.ylabel('Households')
    plt.legend(loc='best')
    fig.savefig(fout,format='pdf')#+'.pdf',format='pdf')
