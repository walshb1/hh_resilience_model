import pandas as pd
import numpy as np

def load_aspire_data(myCountry,cat_info):

    try: aspire = pd.read_excel('../inputs/eca_aspire.xlsx',sheet_name=myCountry)
    except: return False
    
    random_df = pd.DataFrame({'quintile':cat_info['quintile'].copy().astype('int'),
                              'random':np.random.uniform(0,1,cat_info.shape[0]),
                              'pcsoc':0.},
                             index=cat_info.index)
    
    most_recent = aspire.columns[-1]
    for _q in [1,2,3,4,5]:

        most_recent = aspire.columns[-1]
        coverage_val = 1E-2*float(aspire.loc[aspire.series_code=='per_allsp.cov_q{}_tot'.format(_q),most_recent])

        # set value: annual ppp, per cap (note *365 below)
        # --> includes coverage criterion
        random_df.loc[(random_df.quintile==_q)&(random_df.random<=coverage_val),'pcsoc'] = 365*float(aspire.loc[aspire.series_code=='per_allsp.avt_q{}_tot'.format(_q),most_recent])
        
    cat_info['pcsoc'] = random_df['pcsoc'].clip(upper=0.99*cat_info['pcinc'])
    cat_info['hhsoc'] = cat_info[['pcsoc','hhsize']].prod(axis=1)
    print('\n\n--> added aspire info to cat_info')
    return True
