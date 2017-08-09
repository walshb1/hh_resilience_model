import os
import pandas as pd
from lib_gather_data import *
from lib_country_dir import *
import matplotlib.pyplot as plt

from maps_lib import *

import seaborn as sns
sns.set_style('darkgrid')
sns_pal = sns.color_palette('Set1', n_colors=8, desat=.5)

global model
model = os.getcwd()
inputs        = model+'/../inputs/FJ/'       # get inputs data directory

# LOAD FILES (by hazard, asset class) and merge hazards
# load all building values
#df_bld_edu_tc =   pd.read_csv(inputs+'fiji_tc_buildings_edu_tikina.csv').set_index('Tikina').drop('Country_ID',axis=1)
df_bld_edu_tc =   pd.read_csv(inputs+'fiji_tc_buildings_health_tikina.csv').set_index('Tikina').drop('Country_ID',axis=1)

df_bld_edu_tc['Division'] = (df_bld_edu_tc['Tikina_ID']/100).astype('int')
prov_code = pd.read_excel(inputs+'Fiji_provinces.xlsx')[['code','name']].set_index('code').squeeze() 
df_bld_edu_tc['Division'] = df_bld_edu_tc.Division.replace(prov_code)
df_bld_edu_tc.drop('Tikina_ID',axis=1,inplace=True)

df_bld_edu_tc = df_bld_edu_tc.reset_index().set_index(['Division','Tikina']).rename(columns={'exceed_2':2475,'exceed_5':975,'exceed_10':475,
                                                                                             'exceed_20':224,'exceed_40':100,'exceed_50':72,
                                                                                             'exceed_65':50,'exceed_90':22,'exceed_99':10,'AAL':1})

df_bld_edu_tc = df_bld_edu_tc.stack()
df_bld_edu_tc /= 1.E6 # put into millions
df_bld_edu_tc = df_bld_edu_tc.unstack()

df_bld_edu_tc = df_bld_edu_tc.reset_index().set_index(['Division','Tikina','Exp_Value']).stack().to_frame(name='losses')
df_bld_edu_tc.index.names = ['Division','Tikina','Exp_Value','rp']
df_bld_edu_tc = df_bld_edu_tc.reset_index().set_index(['Division','Tikina','rp'])

summed = sum_with_rp('FJ',df_bld_edu_tc,['Exp_Value','losses'],sum_provinces=False,national=False)
#summed['Exp_Value'] /= 10.
print(summed)

df_bld_edu_tc.sum(level=['Division','rp']).to_csv('~/Desktop/my_plots/health_assets.csv')
summed.to_csv('~/Desktop/my_plots/health_assets_AAL.csv')


df_bld_edu_tc['Exp_Value'] /= 100. # map code multiplies by 100 for a percentage
make_map_from_svg(
    df_bld_edu_tc['Exp_Value'].sum(level=['Division','rp']).mean(level='Division'), 
    '../map_files/FJ/BlankSimpleMap.svg',
    outname='FJ_health_assets',
    color_maper=plt.cm.get_cmap('Blues'),
    label='Health assets [million USD]',
    new_title='Health assets [million USD]',
    do_qualitative=False,
    res=2000)
