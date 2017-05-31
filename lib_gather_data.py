import pandas as pd
from pandas_helper import get_list_of_index_names, broadcast_simple, concat_categories
import numpy as np
from scipy.interpolate import UnivariateSpline,interp1d

def mystriper(string):
    '''strip blanks and converts everythng to lower case''' 
    if type(string)==str:
        return str.strip(string).lower()
    else:
        return string

def get_hhid_FIES(df):
    df['hhid'] =  df['w_regn'].astype('str')
    df['hhid'] += df['w_prov'].astype('str')    
    df['hhid'] += df['w_mun'].astype('str')   
    df['hhid'] += df['w_bgy'].astype('str')   
    df['hhid'] += df['w_ea'].astype('str')   
    df['hhid'] += df['w_shsn'].astype('str')   
    df['hhid'] += df['w_hcn'].astype('str')   

#weighted average		
def wavg(data,weights): 
    df_matched =pd.DataFrame({'data':data,'weights':weights}).dropna()
    return (df_matched.data*df_matched.weights).sum()/df_matched.weights.sum()

#gets share per agg category from the data in one of the sheets in PAGER_XL	
def get_share_from_sheet(PAGER_XL,pager_code_to_aggcat,iso3_to_wb,sheetname='Rural_Non_Res'):
    data = pd.read_excel(PAGER_XL,sheetname=sheetname).set_index('ISO-3digit') #data as provided in PAGER
    #rename column to aggregate category
    data_agg =    data[pager_code_to_aggcat.index].rename(columns = pager_code_to_aggcat) #only pick up the columns that are the indice in paper_code_to_aggcat, and change each name to median, fragile etc. based on pager_code_to_aggcat 
    #group by category and sum
    data_agg= data_agg.sum(level=0,axis=1) #sum each category up and shows only three columns with fragile, median and robust.

    data_agg = data_agg.set_index(data_agg.reset_index()['ISO-3digit'].replace(iso3_to_wb));
    
    data_agg.index.name='country'
    return data_agg[data_agg.index.isin(iso3_to_wb)] #keeps only countries
	
def social_to_tx_and_gsp(economy,cat_info):       
        '''(tx_tax, gamma_SP) from cat_info[['social','c','n']] '''
        
        tx_tax = cat_info[['social','c','n']].prod(axis=1, skipna=False).sum(level=economy) / \
                 cat_info[         ['c','n']].prod(axis=1, skipna=False).sum(level=economy)

        #income from social protection PER PERSON as fraction of PER CAPITA social protection
        gsp=     cat_info[['social','c']].prod(axis=1,skipna=False) /\
             cat_info[['social','c','n']].prod(axis=1, skipna=False).sum(level=economy)
        
        return tx_tax, gsp
		
		
def perc_with_spline(data, wt, percentiles):
	assert np.greater_equal(percentiles, 0.0).all(), 'Percentiles less than zero' 
	assert np.less_equal(percentiles, 1.0).all(), 'Percentiles greater than one' 
	data = np.asarray(data) 
	assert len(data.shape) == 1 
	if wt is None: 
		wt = np.ones(data.shape, np.float) 
	else: 
		wt = np.asarray(wt, np.float) 
		assert wt.shape == data.shape 
		assert np.greater_equal(wt, 0.0).all(), 'Not all weights are non-negative.' 
	assert len(wt.shape) == 1 
	i = np.argsort(data) 
	sd = np.take(data, i, axis=0)
	sw = np.take(wt, i, axis=0) 
	aw = np.add.accumulate(sw) 
	if not aw[-1] > 0: 
	 raise ValueError('Nonpositive weight sum' )
	w = (aw)/aw[-1] 
	# f = UnivariateSpline(w,sd,k=1)
	f = interp1d(np.append([0],w),np.append([0],sd))
	return f(percentiles)	 
	
def match_quintiles(hhdataframe,quintiles):
    hhdataframe.loc[hhdataframe['c']<=quintiles[0],'quintile']=1

    for j in np.arange(1,len(quintiles)):
        hhdataframe.loc[(hhdataframe['c']<=quintiles[j])&(hhdataframe['c']>quintiles[j-1]),'quintile']=j+1
        
    return hhdataframe
	
def match_quintiles_score(hhdataframe,quintiles):
    hhdataframe.loc[hhdataframe['score']<=quintiles[0],'quintile_score']=1
    for j in np.arange(1,len(quintiles)):
        hhdataframe.loc[(hhdataframe['score']<=quintiles[j])&(hhdataframe['score']>quintiles[j-1]),'quintile_score']=j+1
    return hhdataframe
	
	
def reshape_data(income):
	data = np.reshape(income.values,(len(income.values))) 
	return data
