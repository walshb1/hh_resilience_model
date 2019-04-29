import pandas as pd
import matplotlib.pyplot as plt
from libraries.maps_lib import *
from libraries.lib_drought import *

#
draw_drought_income_plot()
#
df_dist_aal = get_aal()
make_drought_maps(df_dist_aal,myC='MW')

