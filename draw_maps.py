import pandas as pd

loc_nat = 'gdpLOC'
df_with_results = pd.read_csv('output/'+loc_nat+'/PHL_results_tax_no.csv', index_col=0)

#this library contains ad hoc functions, coded for this project, that allow to produce maps 
from maps_lib import *
import sys

#ploting capacities
import matplotlib as mpl
import matplotlib.pyplot as plt 
#make plots appear in this notebook
#matplotlib inline  

#Default options for plots: 
#this controls the font used in the legend
font = {'family' : 'sans serif',
    'size'   : 26}
plt.rc('font', **font)
mpl.rcParams['xtick.labelsize'] = 20

# path to the blank map 
svg_file_path = "map/BlankSimpleMap.svg"

inp_do_qual = False
inp_res = 600
if len(sys.argv[1:]) > 0:
    inp_do_qual = sys.argv[1]
    inp_res = 2000

make_map_from_svg(
        df_with_results.risk_to_assets, #data 
        svg_file_path,                  #path to blank map
        outname="asset_risk_"+loc_nat,  #base name for output  (will create img/map_of_asset_risk.png, img/legend_of_asset_risk.png, etc.)
        color_maper=plt.cm.get_cmap("Blues"), #color scheme (from matplotlib. Chose them from http://colorbrewer2.org/)
        label="Annual asset losses (% of GDP)",
        new_title="Map of asset risk in the Philippines",  #title for the colored SVG
        do_qualitative=inp_do_qual,
        res=inp_res)

make_map_from_svg(
        df_with_results.resilience, 
        svg_file_path,
        outname="se_resilience_"+loc_nat, 
        color_maper=plt.cm.get_cmap("RdYlGn"), 
        label="Socio-economic capacity (%)",
        new_title="Map of socio-economic resilience in the Philippines",
        do_qualitative=inp_do_qual,
        res=inp_res)

make_map_from_svg(
        df_with_results.risk, 
        svg_file_path,
        outname="welfare_risk_"+loc_nat, 
        color_maper=plt.cm.get_cmap("Purples"), 
        label="Annual welfare losses (% of GDP)",
        new_title="Map of welfare risk in the Philippines",
        do_qualitative=inp_do_qual,
        res=inp_res)
