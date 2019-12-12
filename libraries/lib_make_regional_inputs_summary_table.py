import pandas as pd
import subprocess
import glob, os

from libraries.lib_country_dir import *
from libraries.lib_gather_data import *
from libraries.lib_get_hh_savings import get_hh_savings


def weighted_avg(_c,_df,wgt='pcwgt'):       
    return _df[[wgt,_c]].prod(axis=1).sum(level='region')/_df[wgt].sum(level='region')

def make_regional_inputs_summary_table(myC,df):
    units = []
    
    df_out = (1E-6*df['pcwgt'].sum(level='region').to_frame(name='Population')).round(3)
    units.append('(millions)')
    #
    df_out['Households'] = (1E-6*df['hhwgt'].sum(level='region')).round(3)
    units.append('(millions)')
    #
    df_out['Assets'] = (1E-9*df[['k','pcwgt']].prod(axis=1).sum(level='region')).round(1)
    units.append('(bPhP)')
    #
    df_out['Consumption'] = (1E-3*weighted_avg('c',df)).round(1)
    sort_col = 'Consumption'  
    units.append('(kPHP per cap)')
    #
    df_out['Poverty'] = (1E2*weighted_avg('ispoor',df)).round(0).astype('int')
    units.append('(%)')
    #
    df['cons_gap'] = df.eval('1E2*(1-c/pov_line)')
    df['pcwgt_poor'] = df.eval('ispoor*pcwgt') 
    df_out['Poverty gap'] = (weighted_avg('cons_gap',df,wgt='pcwgt_poor').round(0)).astype('int')
    units.append('(%)')
    #
    df_out['Vulnerability'] = (1E2*weighted_avg('v',df)).round(0).astype('int')
    units.append('(% assets lost)')
    #
    df_out['Social transfers'] = (1E2*weighted_avg('social',df)).round(0).astype('int')
    units.append('(% consumption)')
    #
    df_out['Savings'] = (1E-3*get_hh_savings(myC,'region',pol=None,return_regional_avg=True)).round(1)
    units.append('(kPHP per cap)')
    #
    df_out['Early warning'] = (1E2*weighted_avg('has_ew',df)).astype('int')
    units.append('(% population)')
    #
    df_out = df_out.sort_values(sort_col,ascending=False)
    #df_out.index.name = 'Region'
    _grab = df_out.index[0]
    #
    df_units = pd.DataFrame(columns=df_out.columns,index={'Region'})
    df_units.loc['Region'] = units
    df_out = df_units.append(df_out)
    #
    col_form = '@{}l|rrrrrrrrrr@{}'
    df_out.to_latex('../inputs/{}/_regional_inputs_summary_table.tex'.format(myC),column_format=col_form)
    df_out.to_csv('../inputs/{}/regional_inputs_summary_table.csv'.format(myC))

    with open('../inputs/{}/_regional_inputs_summary_table.tex'.format(myC), 'r') as f:
        with open('../inputs/{}/regional_inputs_summary_table.tex'.format(myC), 'w') as f2:

                    f2.write(r'\documentclass[preview=true]{standalone}'+'\n')
                    f2.write(r'\usepackage{amssymb} %maths'+'\n')
                    f2.write(r'\usepackage{amsmath} %maths'+'\n')
                    f2.write(r'\usepackage{booktabs}'+'\n')
                    f2.write(r'\usepackage{rotating}'+'\n')
                    f2.write(r'\begin{document}'+'\n')
                    
                    reading_is_fundamental = f.read()
                    reading_is_fundamental = reading_is_fundamental.replace(r'\midrule','')
                    reading_is_fundamental = reading_is_fundamental.replace('{}'.format(_grab),r'\midrule {}'.format(_grab))

                    _dupes = []
                    for _n in df_out:
                        reading_is_fundamental = reading_is_fundamental.replace(_n,r'\footnotesize {'+_n+'}')
                        _dupes.append(_n)

                    for _u in units:
                        if _u in _dupes: continue
                        if '%' in _u: 
                            _u = _u.replace('%','\%')
                            reading_is_fundamental = reading_is_fundamental.replace(_u,r'\footnotesize{'+_u+'}')
                        else: reading_is_fundamental = reading_is_fundamental.replace(_u,r'\footnotesize{'+_u+'}')
                        _dupes.append(_u)
                        
                    f2.write(reading_is_fundamental)

                    f2.write(r'\end{document}')
                    f2.close()

    subprocess.call(('cd ~/Desktop/BANK/hh_resilience_model/inputs/{}/; pdflatex regional_inputs_summary_table.tex'.format(myC)),shell=True)

    for f in glob.glob('../inputs/{}/*.aux'.format(myC)): os.remove(f)
    for f in glob.glob('../inputs/{}/*.log'.format(myC)): os.remove(f)
    for f in glob.glob('../inputs/{}/_*.tex'.format(myC)): os.remove(f)
    assert(False)
    
