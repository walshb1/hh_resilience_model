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
        '''(tx_tax, gamma_SP) from cat_info[['social','c','weight']] '''

        tx_tax = cat_info[['social','c','pcwgt']].prod(axis=1, skipna=False).sum() / cat_info[['c','pcwgt']].prod(axis=1, skipna=False).sum()
        #income from social protection PER PERSON as fraction of PER CAPITA social protection
        gsp= cat_info[['social','c']].prod(axis=1,skipna=False) / cat_info[['social','c','pcwgt']].prod(axis=1, skipna=False).sum()
        
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
	
def match_percentiles(hhdataframe,quintiles,col_label):
    hhdataframe.loc[hhdataframe['c']<=quintiles[0],col_label]=1

    for j in np.arange(1,len(quintiles)):
        hhdataframe.loc[(hhdataframe['c']<=quintiles[j])&(hhdataframe['c']>quintiles[j-1]),col_label]=j+1
        
    return hhdataframe
	
def match_quintiles_score(hhdataframe,quintiles):
    hhdataframe.loc[hhdataframe['score']<=quintiles[0],'quintile_score']=1
    for j in np.arange(1,len(quintiles)):
        hhdataframe.loc[(hhdataframe['score']<=quintiles[j])&(hhdataframe['score']>quintiles[j-1]),'quintile_score']=j+1
    return hhdataframe
	
	
def reshape_data(income):
	data = np.reshape(income.values,(len(income.values))) 
	return data

def get_AIR_data(fname,sname,keep_sec,keep_per):
    # AIR dataset province code to province name
    AIR_prov_lookup = pd.read_excel(fname,sheetname='Lookup_Tables',usecols=['province_code','province'],index_col='province_code')
    AIR_prov_lookup = AIR_prov_lookup['province'].to_dict()

    AIR_prov_corrections = {'Tawi-Tawi':'Tawi-tawi',
                            #'Metropolitan Manila':'Manila',
                            #'Davao del Norte':'Davao',
                            #'Batanes':'Batanes_off',
                            'North Cotabato':'Cotabato'}
    
    # AIR dataset peril code to peril name
    AIR_peril_lookup_1 = pd.read_excel(fname,sheetname='Lookup_Tables',usecols=['perilsetcode','peril'],index_col='perilsetcode')
    AIR_peril_lookup_1 = AIR_peril_lookup_1['peril'].dropna().to_dict()
    #AIR_peril_lookup_2 = {'EQ':'EQ', 'HUSSPF':'TC', 'HU':'wind', 'SS':'surge', 'PF':'flood'}

    AIR_value_destroyed = pd.read_excel(fname,sheetname='Loss_Results',
                                        usecols=['perilsetcode','province','Perspective','Sector','EP1','EP10','EP25','EP30','EP50','EP100','EP200','EP250','EP500','EP1000']).squeeze()
    AIR_value_destroyed.columns=['hazard','province','Perspective','Sector',1,10,25,30,50,100,200,250,500,1000]

    # Change province code to province name
    #AIR_value_destroyed = AIR_value_destroyed.reset_index().set_index(['hazard','Perspective','Sector'])
    AIR_value_destroyed['province'].replace(AIR_prov_lookup,inplace=True)
    AIR_value_destroyed['province'].replace(AIR_prov_corrections,inplace=True) 
    #AIR_prov_corrections

    AIR_value_destroyed = AIR_value_destroyed.reset_index().set_index('province')
    AIR_value_destroyed = AIR_value_destroyed.drop('All Provinces')

    AIR_value_destroyed = AIR_value_destroyed.reset_index().set_index(['hazard','province','Perspective','Sector'])
    AIR_value_destroyed = AIR_value_destroyed.drop(['index'],axis=1, errors='ignore')

    # Stack return periods column
    AIR_value_destroyed.columns.name='rp'
    AIR_value_destroyed = AIR_value_destroyed.stack()

    # Name values
    #AIR_value_destroyed.name='v'

    # Choose only Sector = 0 (Private Assets) 
    # --> Alternative: 15 = All Assets (Private + Govt (16) + Emergency (17))
    sector_dict = {'Private':0, 'private':0,
                   'Public':16, 'public':16,
                   'Government':16, 'government':16,
                   'Emergency':17, 'emergency':17,
                   'All':15, 'all':15}
    
    AIR_value_destroyed = AIR_value_destroyed.reset_index().set_index(['Sector'])
    AIR_value_destroyed = AIR_value_destroyed.drop([iSec for iSec in range(0,30) if iSec != sector_dict[keep_sec]])
    
    # Choose only Perspective = Occurrence ('Occ') OR Aggregate ('Agg')
    AIR_value_destroyed = AIR_value_destroyed.reset_index().set_index(['Perspective'])
    AIR_value_destroyed = AIR_value_destroyed.drop([iPer for iPer in ['Occ', 'Agg'] if iPer != keep_per])
    
    # Map perilsetcode to perils to hazard
    AIR_value_destroyed = AIR_value_destroyed.reset_index().set_index(['hazard'])
    AIR_value_destroyed = AIR_value_destroyed.drop(-1)

    # Drop Sector and Perspective columns
    AIR_value_destroyed = AIR_value_destroyed.reset_index().set_index(['province','hazard','rp'])
    AIR_value_destroyed = AIR_value_destroyed.drop(['Sector','Perspective'],axis=1, errors='ignore')
    
    AIR_value_destroyed = AIR_value_destroyed.reset_index().set_index('province')
    AIR_value_destroyed['hazard'].replace(AIR_peril_lookup_1,inplace=True)

    # Keep only earthquake (EQ) and typhoon (TC = wind + storm surge + precipitation flood)
    AIR_value_destroyed = AIR_value_destroyed.reset_index().set_index('hazard')   
    AIR_value_destroyed = AIR_value_destroyed.drop(['HU','SS','PF'],axis=0)

    AIR_value_destroyed = AIR_value_destroyed.reset_index().set_index(['province','hazard','rp'])

    AIR_value_destroyed = AIR_value_destroyed.sort_index().squeeze()

    return AIR_value_destroyed
