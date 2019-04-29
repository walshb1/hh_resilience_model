import matplotlib.pyplot as plt
import seaborn as sns
import brewer2mpl as brew
from matplotlib import colors

##################################
#Aesthetics
sns.set_style('whitegrid')
brew_pal = brew.get_map('Set1', 'qualitative', 8).mpl_colors
greys_pal = sns.color_palette('Greys', n_colors=9)
q_labels = ['Poorest quintile','Second','Third',
            'Fourth','Wealthiest quintile']

paired_pal = sns.color_palette('Paired', n_colors=12)
sns_pal = sns.color_palette('Set1', n_colors=8, desat=.4)
#q_colors = [sns_pal[0],sns_pal[1],sns_pal[2],sns_pal[3],sns_pal[5]]

pubugn_pal = sns.color_palette('PuBuGn', n_colors=9)
blues_pal = sns.color_palette('Blues', n_colors=9)
q_colors = [pubugn_pal[1],pubugn_pal[3],pubugn_pal[5],pubugn_pal[6],pubugn_pal[8]]

def title_legend_labels(ax,pais,lab_x=None,lab_y=None,lim_x=None,lim_y=None,leg_fs=9,do_leg=True):
    
    try:plt.title(iso_to_name[pais])
    except:plt.title(pais)

    if lim_x is not None:
        try: plt.xlim(lim_x)
        except:
            try: plt.xlim(lim_x[0])
            except: pass

    if lim_y is not None:
        try: plt.ylim(lim_y)
        except:
            try: plt.ylim(lim_y[0])
            except: pass
            
    plt.xlabel(lab_x,labelpad=10,fontsize=10,linespacing=1.75)
    plt.ylabel(lab_y,labelpad=10,fontsize=10,linespacing=1.75)

    plt.xticks(size=8)
    plt.yticks(size=8)

    if do_leg:
        ax.legend(loc='best',labelspacing=0.75,ncol=1,fontsize=leg_fs,borderpad=0.75,
                  fancybox=True,frameon=True,framealpha=0.9)

    return ax

def pretty_float(_f):
    
    if _f >= 1E3:
        _u = 'k'
        _f /= 1E3

        if _f >= 1E3:
            _u = 'm'
            _f /= 1E3

            if _f >=1E3:
                _u = 'b'
                _f /= 1E3

    return str(round(_f,2))+_u
