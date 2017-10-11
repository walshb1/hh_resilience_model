import os
import pandas as pd
from lib_gather_data import *
import matplotlib.pyplot as plt
from plot_hist import *

df_c = pd.read_excel(os.getcwd()+'/../inputs/FJ/HIES 2013-14 Consumption Data.xlsx',sheetname='by_Individual items')[['HHID','Grand Total']].dropna()
df_c.columns.names = ['consumption']

df_i = pd.read_excel(os.getcwd()+'/../inputs/FJ/HIES 2013-14 Income Data.xlsx',sheetname='Sheet1')[['HHID','New Total']].dropna()
df_i.columns.names = ['income']

df_c['HHID'] = df_c['HHID'].astype('int')
df_i['HHID'] = df_i['HHID'].astype('int')

#df_c['income'] = df_i['New Total']

#print(df_c.shape[0])
#print(df_i.shape[0])

#df = pd.concat([df_c,df_i],axis=1,join='inner')
df = pd.merge(df_c,df_i,on=['HHID'],how='outer')

df.to_csv('~/Desktop/tmp.csv')
#print(df.head(5))
#print(df.shape[0])
