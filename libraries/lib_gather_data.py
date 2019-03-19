from libraries.pandas_helper import get_list_of_index_names, broadcast_simple, concat_categories
from scipy.interpolate import UnivariateSpline,interp1d
from libraries.lib_average_over_rp import *
import pandas as pd
import numpy as np

def mystriper(string):
    '''strip blanks and converts everything to lower case'''
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
def get_share_from_sheet(PAGER_XL,pager_code_to_aggcat,iso3_to_wb,sheet_name='Rural_Non_Res'):
    data = pd.read_excel(PAGER_XL,sheet_name=sheetname).set_index('ISO-3digit') #data as provided in PAGER
    #rename column to aggregate category
    data_agg =    data[pager_code_to_aggcat.index].rename(columns = pager_code_to_aggcat)
    #only pick up the columns that are the indices in paper_code_to_aggcat, and change each name to median, fragile etc. based on pager_code_to_aggcat
    #group by category and sum
    data_agg= data_agg.sum(level=0,axis=1) #sum each category up and shows only three columns with fragile, median and robust.

    data_agg = data_agg.set_index(data_agg.reset_index()['ISO-3digit'].replace(iso3_to_wb));

    data_agg.index.name='country'
    return data_agg[data_agg.index.isin(iso3_to_wb)] #keeps only countries

def social_to_tx_and_gsp(economy,cat_info):
    """tx_tax Tax revenue used to fund social payments.
    gsp
    (tx_tax, gamma_SP) from cat_info[['social','c','weight']]

    Try to exclude remittances (depends on whether df has frac_remittance),
    otherwise include them bothself.

    Returns
    -------
    tau_tax = total value of social as fraction of total C
    gamma_SP = Fraction of social that goes to each hh
    """
    try:
        tx_tax = cat_info.eval('(1-frac_remittance)*social*c*pcwgt').sum() / cat_info.eval('c*pcwgt').sum()
        gsp = cat_info.eval('(1-frac_remittance)*social*c')/cat_info.eval('(1-frac_remittance)*social*c*pcwgt').sum()
        print('\n\nCalculating tax_social & gamma_SP excluding remittances')
    except:
        tx_tax = cat_info[['social','c','pcwgt']].prod(axis=1, skipna=False).sum() / cat_info[['c','pcwgt']].prod(axis=1, skipna=False).sum()
        gsp = cat_info[['social','c']].prod(axis=1,skipna=False) / cat_info[['social','c','pcwgt']].prod(axis=1, skipna=False).sum()
        print('\n\nCalculating tax_social & gamma_SP *including* remittances')
    return tx_tax, gsp


def perc_with_spline(data, wt, percentiles):
    """Find the cutoffs in data at each percentile given household weights

    Parameters
    ----------
    data : nparray
        data wanted to find percentiles of
    wt : type
        weighting of each row - number of households or people at each data point.
    percentiles : list-like or ndarray
        list of percentiles wanted to calculate

    Returns
    -------
    ndarray
        percentile cutoffs for data
    """
    assert np.greater_equal(percentiles, 0.0).all(), 'Percentiles less than zero'
    assert np.less_equal(percentiles, 1.0).all(), 'Percentiles greater than one'
    data = np.asarray(data)
    assert len(data.shape) == 1, 'not a 1d array'
    if wt is None:
        wt = np.ones(data.shape, np.float)
    else:
        wt = np.asarray(wt, np.float)
        assert wt.shape == data.shape
        assert np.greater_equal(wt, 0.0).all(), 'Not all weights are non-negative.'
    assert len(wt.shape) == 1
    # Take the indices that would sort data, and then sorts both data and wt by those indices
    i = np.argsort(data)
    sd = np.take(data, i, axis=0)
    sw = np.take(wt, i, axis=0)
    # Cumulative population
    aw = np.add.accumulate(sw)
    if not aw[-1] > 0:
     raise ValueError('Nonpositive weight sum')
    # Calculate percentile by household
    w = (aw)/aw[-1]
    # f = UnivariateSpline(w,sd,k=1)
    f = interp1d(np.append([0],w),np.append([0],sd))
    return f(percentiles)

def match_percentiles(hhdataframe,quintiles,col_label,sort_val='c'):
    """Compares hhdataframe to 'quintiles' and returns the data frame with
    with another column that gives the numerical quintile/decile/percentile

    Parameters
    ----------
    hhdataframe : df
        households dataframe, input from df.groupby('district').apply(lambda x:match_percentiles(*))
    quintiles : type
        quintile cutoffs, from perc_with_spline
    col_label : type
        label for the column that is output
    sort_val : type
        what column to compare to the quintile cutoffs

    Returns
    -------
    df
        hhdataframe with a percentile column
    """
    # Label households below the lowest cutoff of col_label
    hhdataframe.loc[hhdataframe[sort_val]<=quintiles[0],col_label]=1

    for j in np.arange(1,len(quintiles)):
        hhdataframe.loc[(hhdataframe[sort_val]<=quintiles[j])&(hhdataframe[sort_val]>quintiles[j-1]),col_label]=j+1

    return hhdataframe

def match_quintiles_score(hhdataframe,quintiles):
    hhdataframe.loc[hhdataframe['score']<=quintiles[0],'quintile_score']=1
    for j in np.arange(1,len(quintiles)):
        hhdataframe.loc[(hhdataframe['score']<=quintiles[j])&(hhdataframe['score']>quintiles[j-1]),'quintile_score']=j+1
    return hhdataframe


def reshape_data(income):
    """Basically returns income.values"""
    data = np.reshape(income.values,(len(income.values)))
    return data

def get_AIR_data(fname,sname,keep_sec,keep_per):
    # AIR dataset province code to province name
    AIR_prov_lookup = pd.read_excel(fname,sheet_name='Lookup_Tables',index_col='province_code')[['province']]
    AIR_prov_lookup = AIR_prov_lookup['province'].to_dict()
    # NOTE: the coding in AIR differs from the latest PSA coding for Zamboanga Peninsula
    # --> AIR: 80: 'Zamboanga del Norte', 81: 'Zamboanga del Sur', 82: 'Zamboanga Sibugay'
    # --> PSA:

    AIR_prov_corrections = {'Tawi-Tawi':'Tawi-tawi',
                            #'Metropolitan Manila':'Manila',
                            #'Davao del Norte':'Davao',
                            #'Batanes':'Batanes_off',
                            'North Cotabato':'Cotabato'}

    # AIR dataset peril code to peril name
    AIR_peril_lookup_1 = pd.read_excel(fname,sheet_name='Lookup_Tables',index_col='perilsetcode')[['peril']]
    AIR_peril_lookup_1 = AIR_peril_lookup_1['peril'].dropna().to_dict()
    #AIR_peril_lookup_2 = {'EQ':'EQ', 'HUSSPF':'TC', 'HU':'wind', 'SS':'surge', 'PF':'flood'}

    AIR_value_destroyed = pd.read_excel(fname,sheet_name='Loss_Results')[['perilsetcode','province','Perspective','Sector','AAL',
                                                                          'EP1','EP10','EP25','EP30','EP50','EP100','EP200','EP250','EP500','EP1000']].squeeze()
    AIR_value_destroyed.columns=['hazard','province','Perspective','Sector','AAL',1,10,25,30,50,100,200,250,500,1000]

    # Change province code to province name
    #AIR_value_destroyed = AIR_value_destroyed.reset_index().set_index(['hazard','Perspective','Sector'])
    AIR_value_destroyed['province'].replace(AIR_prov_lookup,inplace=True)
    AIR_value_destroyed['province'].replace(AIR_prov_corrections,inplace=True)
    #AIR_prov_corrections

    AIR_value_destroyed = AIR_value_destroyed.reset_index().set_index('province')
    AIR_value_destroyed = AIR_value_destroyed.drop('All Provinces')

    AIR_value_destroyed = AIR_value_destroyed.reset_index().set_index(['hazard','province','Perspective','Sector'])
    AIR_value_destroyed = AIR_value_destroyed.drop(['index'],axis=1, errors='ignore')

    AIR_aal = AIR_value_destroyed['AAL']

    # Stack return periods column
    AIR_value_destroyed = AIR_value_destroyed.drop('AAL',axis=1)
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

    AIR_aal = AIR_aal.reset_index().set_index(['Sector'])
    AIR_aal = AIR_aal.drop([iSec for iSec in range(0,30) if iSec != sector_dict[keep_sec]])

    # Choose only Perspective = Occurrence ('Occ') OR Aggregate ('Agg')
    AIR_value_destroyed = AIR_value_destroyed.reset_index().set_index(['Perspective'])
    AIR_value_destroyed = AIR_value_destroyed.drop([iPer for iPer in ['Occ', 'Agg'] if iPer != keep_per])

    AIR_aal = AIR_aal.reset_index().set_index(['Perspective'])
    AIR_aal = AIR_aal.drop([iPer for iPer in ['Occ', 'Agg'] if iPer != keep_per])

    # Map perilsetcode to perils to hazard
    AIR_value_destroyed = AIR_value_destroyed.reset_index().set_index(['hazard'])
    AIR_value_destroyed = AIR_value_destroyed.drop(-1)

    AIR_aal = AIR_aal.reset_index().set_index(['hazard'])
    AIR_aal = AIR_aal.drop(-1)

    # Drop Sector and Perspective columns
    AIR_value_destroyed = AIR_value_destroyed.reset_index().set_index(['province','hazard','rp'])
    AIR_value_destroyed = AIR_value_destroyed.drop(['Sector','Perspective'],axis=1, errors='ignore')

    AIR_aal = AIR_aal.reset_index().set_index(['province','hazard'])
    AIR_aal = AIR_aal.drop(['Sector','Perspective'],axis=1, errors='ignore')

    AIR_value_destroyed = AIR_value_destroyed.reset_index().set_index('province')
    AIR_value_destroyed['hazard'].replace(AIR_peril_lookup_1,inplace=True)

    AIR_aal = AIR_aal.reset_index().set_index('province')
    AIR_aal['hazard'].replace(AIR_peril_lookup_1,inplace=True)

    # Keep earthquake (EQ), wind (HU), storm surge (SS), and precipitation flood (PF)
    AIR_value_destroyed = AIR_value_destroyed.reset_index().set_index('hazard')
    AIR_value_destroyed = AIR_value_destroyed.drop(['HUSSPF'],axis=0)

    AIR_aal = AIR_aal.reset_index().set_index('hazard')
    AIR_aal = AIR_aal.drop(['HUSSPF'],axis=0)

    AIR_value_destroyed = AIR_value_destroyed.reset_index().set_index(['province','hazard','rp'])
    AIR_value_destroyed = AIR_value_destroyed.sort_index().squeeze()

    AIR_aal = AIR_aal.reset_index().set_index(['province','hazard'])
    AIR_aal = AIR_aal.sort_index().squeeze()

    AIR_value_destroyed = AIR_extreme_events(AIR_value_destroyed,AIR_aal,sec=keep_sec,per=keep_per)

    return AIR_value_destroyed

def AIR_extreme_events(df_air,df_aal,sec='',per=''):

    # add frequent events
    last_rp = 20
    new_rp = 1

    added_proba = 1/new_rp - 1/last_rp

    # places where new values are higher than values for 10-yr RP
    test = df_air.unstack().replace(0,np.nan).dropna().assign(test=lambda x:x[new_rp]/x[10]).test

    max_relative_exp = .8

    overflow_frequent_countries = test[test>max_relative_exp].index
    print("overflow in {n} (region, event)".format(n=len(overflow_frequent_countries)))
    #print(test[overflow_frequent_countries].sort_values(ascending=False))

    # for this places, add infrequent events
    hop=df_air.unstack()


    hop[1]=hop[1].clip(upper=max_relative_exp*hop[10])
    df_air = hop.stack()
    #^ changed from: frac_value_destroyed_gar = hop.stack()
    #print(frac_value_destroyed_gar_completed)

    print(df_air.head(10))

    new_rp = 2000
    added_proba = 1/2000

    df_air_avg, _ = average_over_rp(df_air)

    new_frac_destroyed = (df_aal - df_air_avg.squeeze())/(added_proba)

    #REMOVES 'tsunamis' and 'earthquakes' from this thing
    # new_frac_destroyed = pd.DataFrame(new_frac_destroyed).query("hazard in ['tsunami', 'earthquake']").squeeze()

    hop = df_air.unstack()
    hop[new_rp]=   new_frac_destroyed
    hop= hop.sort_index(axis=1)

    df_air = hop.stack()
    #frac_value_destroyed_gar_completed.head(10)

    test = df_air.unstack().replace(0,np.nan).dropna().assign(test=lambda x:x[new_rp]/x[1000]).test
    #print(frac_value_destroyed_gar_completed["United States"])

    df_air_averages, _ = average_over_rp(df_air)

    pd.DataFrame((df_air_averages.squeeze()/df_aal).replace(0,np.nan).dropna().sort_values())

    print('GAR preprocessing script: writing out intermediate/frac_value_destroyed_gar_completed.csv')
    df_air.to_csv('../inputs/PH/Risk_Profile_Master_With_Population_with_EP1_and_EP2000'+sec+'_'+per+'.csv', encoding="utf-8", header=True)

    return df_air


def get_subnational_gdp_macro(myCountry,_hr,avg_prod_k):
    hr_init = _hr.shape[0]

    if myCountry == 'PH':

        grdp = pd.read_csv('../inputs/PH/phil_grdp.csv')[['region','2015']]
        grdp.columns = ['_region','grdp']
        grdp['region_lower'] = grdp['_region'].str.lower()

        grdp['grdp_assets'] = grdp['grdp'].str.replace(',','').astype('int')*1000./avg_prod_k

        _hr = _hr.reset_index()
        _hr['region_lower'] = _hr['region'].str.lower()
        _hr = pd.merge(_hr.reset_index(),grdp.reset_index(),on=['region_lower']).reset_index().set_index(['region','hazard','rp']).sort_index()

        print(_hr.shape[0],hr_init)
        assert(_hr.shape[0] == hr_init)
        return _hr['grdp_assets']
