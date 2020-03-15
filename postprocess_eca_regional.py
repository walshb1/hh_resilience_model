import os,time,gc,glob
import pandas as pd

countries = ['HR','GR','RO','BG','TR','GE','AL','AM']
df_cty = pd.DataFrame()

for _cty in countries:
    print(_cty)

    df = pd.read_csv('../output_country/{}/iah_tax_no_.csv'.format(_cty)).set_index('quintile')
    
    for col in ['dk0','di_aggregate','dc_aggregate']:
        
        try: df_cty[_cty+col] = 1E-6*df[['pcwgt',col]].prod(axis=1).sum(level='quintile')
        except: pass
        
df_cty.to_csv('~/Desktop/Dropbox/Bank/ECA ASA/analytics/quintile_info.csv')
