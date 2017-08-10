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


# Subsistence Fishing & Farming
df = pd.read_excel(inputs+'HIES 2013-14 Income Data.xlsx',usecols=['HHID','Division','AE','HHsize','Ethnicity','Weight','Sector','TotalAgri','Fishprawnscrabsetc','TotalIncome']).set_index('HHID')

# Plot total income from Fishprawnscrabsetc:
ci_heights, ci_bins = np.histogram(df.loc[df.Fishprawnscrabsetc > 0,'Fishprawnscrabsetc'].clip(upper=20000), bins=50, weights=df.loc[df.Fishprawnscrabsetc > 0,'Weight'])

total_fishermen = df.loc[df.Fishprawnscrabsetc > 0,['Weight','HHsize']].prod(axis=1).sum()
frac_fishermen_ita = df.loc[(df.Fishprawnscrabsetc > 0)&(df.Ethnicity=='ITAUKEI'),['Weight','HHsize']].prod(axis=1).sum()/total_fishermen
frac_fishermen_ind = df.loc[(df.Fishprawnscrabsetc > 0)&(df.Ethnicity=='INDIAN-FIJIAN'),['Weight','HHsize']].prod(axis=1).sum()/total_fishermen

ax = plt.gca()
q_colors = [sns_pal[0],sns_pal[1],sns_pal[2],sns_pal[3],sns_pal[5]]
ax.bar(ci_bins[:-1], ci_heights, width=(ci_bins[1]-ci_bins[0]), facecolor=q_colors[0],alpha=0.4)
    
fig = ax.get_figure()
plt.xlabel(r'Income from fisheries [FJD yr$^{-1}$]')
plt.ylabel('Population')
plt.legend(loc='best')
ax.annotate('Total income: '+str(round(df[['Fishprawnscrabsetc','Weight']].prod(axis=1).sum()/1.E6,1))+'M FJD per year',xy=(10000,650),xycoords='data',ha='left',va='top',fontsize=9,annotation_clip=False,weight='bold')
ax.annotate(str(int(total_fishermen))+' Fijians gain some income from fisheries',xy=(10000,600),xycoords='data',ha='left',va='top',fontsize=9,annotation_clip=False,weight='bold')
ax.annotate(str(round(frac_fishermen_ita*100.,1))+'% i-Taukei',xy=(10000,550),xycoords='data',ha='left',va='top',fontsize=9,annotation_clip=False,weight='bold')
ax.annotate(str(round(frac_fishermen_ind*100.,1))+'% Indian-Fijian',xy=(10000,500),xycoords='data',ha='left',va='top',fontsize=9,annotation_clip=False,weight='bold')

fig.savefig('../output_plots/FJ/sectoral/fisheries_income.pdf',format='pdf')#+'.pdf',format='pdf')
plt.cla()

print(frac_fishermen_ita/total_fishermen)
print(frac_fishermen_ind/total_fishermen)

# Plot fraction of income from Fishprawnscrabsetc:
ci_heights, ci_bins = np.histogram(df.loc[df.Fishprawnscrabsetc > 0,'Fishprawnscrabsetc']/df.loc[df.Fishprawnscrabsetc > 0,'TotalIncome'], 
                                   bins=50, weights=df.loc[df.Fishprawnscrabsetc > 0,'Weight'])

ax = plt.gca()
q_colors = [sns_pal[0],sns_pal[1],sns_pal[2],sns_pal[3],sns_pal[5]]
ax.bar(ci_bins[:-1], ci_heights, width=(ci_bins[1]-ci_bins[0]), facecolor=q_colors[0],alpha=0.4)
    
fig = ax.get_figure()
plt.xlabel(r'Fraction of income from fisheries')
plt.ylabel('Population')
plt.legend(loc='best')
fig.savefig('../output_plots/FJ/sectoral/fisheries_income_frac.pdf',format='pdf')#+'.pdf',format='pdf')
plt.cla()

# Plot total income from Farming:
df['TotalAgri'] -= df['Fishprawnscrabsetc']
ci_heights, ci_bins = np.histogram(df.loc[df.TotalAgri > 0,'TotalAgri'].clip(upper=20000), bins=50, weights=df.loc[df.TotalAgri > 0,'Weight'])

total_farmers = df.loc[df.TotalAgri > 0,['Weight','HHsize']].prod(axis=1).sum()
frac_farmers_ita = df.loc[(df.TotalAgri > 0)&(df.Ethnicity=='ITAUKEI'),['Weight','HHsize']].prod(axis=1).sum()/total_farmers
frac_farmers_ind = df.loc[(df.TotalAgri > 0)&(df.Ethnicity=='INDIAN-FIJIAN'),['Weight','HHsize']].prod(axis=1).sum()/total_farmers

ax = plt.gca()
q_colors = [sns_pal[0],sns_pal[1],sns_pal[2],sns_pal[3],sns_pal[5]]
ax.bar(ci_bins[:-1], ci_heights, width=(ci_bins[1]-ci_bins[0]), facecolor=q_colors[0],alpha=0.4)
    
fig = ax.get_figure()
plt.xlabel(r'Income from agriculture [FJD yr$^{-1}$]')
plt.ylabel('Population')
plt.legend(loc='best')
ax.annotate('Total income: '+str(round(df[['TotalAgri','Weight']].prod(axis=1).sum()/1.E6,1))+'M FJD per year',xy=(10000,3200),xycoords='data',ha='left',va='top',fontsize=9,annotation_clip=False,weight='bold')
ax.annotate(str(int(total_farmers))+' Fijians gain some income from agriculture',xy=(10000,3000),xycoords='data',ha='left',va='top',fontsize=9,annotation_clip=False,weight='bold')
ax.annotate(str(round(frac_farmers_ita*100.,1))+'% i-Taukei',xy=(10000,2800),xycoords='data',ha='left',va='top',fontsize=9,annotation_clip=False,weight='bold')
ax.annotate(str(round(frac_farmers_ind*100.,1))+'% Indian-Fijian',xy=(10000,2600),xycoords='data',ha='left',va='top',fontsize=9,annotation_clip=False,weight='bold')

fig.savefig('../output_plots/FJ/sectoral/agri_income.pdf',format='pdf')#+'.pdf',format='pdf')
plt.cla()

# Plot fraction of income from Fishprawnscrabsetc:
ci_heights, ci_bins = np.histogram(df.loc[df.TotalAgri > 0,'TotalAgri']/df.loc[df.TotalAgri > 0,'TotalIncome'], 
                                   bins=50, weights=df.loc[df.TotalAgri > 0,'Weight'])

ax = plt.gca()
q_colors = [sns_pal[0],sns_pal[1],sns_pal[2],sns_pal[3],sns_pal[5]]
ax.bar(ci_bins[:-1], ci_heights, width=(ci_bins[1]-ci_bins[0]), facecolor=q_colors[0],alpha=0.4)
    
fig = ax.get_figure()
plt.xlabel(r'Fraction of income from agriculture')
plt.ylabel('Population')
plt.legend(loc='best')
fig.savefig('../output_plots/FJ/sectoral/agri_income_frac.pdf',format='pdf')#+'.pdf',format='pdf')
plt.cla()

df = df.reset_index()
prov_code = pd.read_excel(inputs+'Fiji_provinces.xlsx')[['code','name']].set_index('code').squeeze() 
df['Division'] = df.Division.replace(prov_code)
df = df.set_index('Division')

ag_df = df[['Weight','TotalIncome']].prod(axis=1).sum(level='Division').to_frame(name='TotalIncome')
ag_df['AgriIncome'] = df[['Weight','TotalAgri']].prod(axis=1).sum(level='Division')
ag_df['FishIncome'] = df[['Weight','Fishprawnscrabsetc']].prod(axis=1).sum(level='Division')
ag_df.to_csv('~/Desktop/my_plots/FJ_ag_incomes.csv')

#print(df)
assert(False)

# LOAD FILES (by hazard, asset class) and merge hazards
# load all building values
<<<<<<< HEAD
#df_bld_edu_tc =   pd.read_csv(inputs+'fiji_tc_buildings_edu_tikina.csv').set_index('Tikina').drop('Country_ID',axis=1)
#desc_str = 'education'
df_bld_edu_tc =   pd.read_csv(inputs+'fiji_tc_buildings_health_tikina.csv').set_index('Tikina').drop('Country_ID',axis=1)
desc_str = 'health'
=======
df_bld_edu_tc =   pd.read_csv(inputs+'fiji_tc_buildings_edu_tikina.csv').set_index('Tikina').drop('Country_ID',axis=1)
df_bld_hea_tc =   pd.read_csv(inputs+'fiji_tc_buildings_health_tikina.csv').set_index('Tikina').drop('Country_ID',axis=1)
>>>>>>> parent of 168dda1... new inf file

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

summed = sum_with_rp('FJ',df_bld_edu_tc,['losses'],sum_provinces=False,national=False)

<<<<<<< HEAD
df_bld_edu_tc.sum(level=['Division','rp']).to_csv('~/Desktop/my_plots/'+desc_str+'_assets.csv')
summed.to_csv('~/Desktop/my_plots/'+desc_str+'_assets_AAL.csv')
=======
df_bld_edu_tc.sum(level=['Division','rp']).to_csv('~/Desktop/my_plots/educational_assets.csv')
summed.to_csv('~/Desktop/my_plots/educational_assets_AAL.csv')
>>>>>>> parent of 168dda1... new inf file


df_bld_edu_tc['Exp_Value'] /= 100. # map code multiplies by 100 for a percentage
make_map_from_svg(
    df_bld_edu_tc['Exp_Value'].sum(level=['Division','rp']).mean(level='Division'), 
    '../map_files/FJ/BlankSimpleMap.svg',
<<<<<<< HEAD
    outname='FJ_'+desc_str+'_assets',
    color_maper=plt.cm.get_cmap('Blues'),
    label = desc_str[0].upper()+desc_str[1:]+' assets [million USD]',
    new_title = desc_str[0].upper()+desc_str[1:]+' assets [million USD]',
=======
    outname='FJ_educational_assets',
    color_maper=plt.cm.get_cmap('Blues'),
    label='Educational assets [million USD]',
    new_title='Educational assets [million USD]',
>>>>>>> parent of 168dda1... new inf file
    do_qualitative=False,
    res=2000)
