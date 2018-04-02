import matplotlib.pyplot as plt

def title_legend_labels(ax,pais,lab_x=None,lab_y=None,lim_x=None,lim_y=None,leg_fs=9):
    
    try:plt.title(iso_to_name[pais])
    except:plt.title(pais)

    try: plt.xlim(lim_x)
    except:
        try: plt.xlim(lim_x[0])
        except: pass

    try: plt.ylim(lim_y)
    except:
        try: plt.ylim(lim_y[0])
        except: pass

    plt.xlabel(lab_x)
    plt.ylabel(lab_y)

    ax.legend(loc='best',labelspacing=0.75,ncol=1,fontsize=leg_fs,borderpad=0.75,fancybox=True,frameon=True,framealpha=0.9)

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
