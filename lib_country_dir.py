import os
import pandas as pd
from lib_gather_data import *

global model
model = os.getcwd()

# People/hh will be affected or not_affected, and helped or not_helped
affected_cats = pd.Index(['a', 'na'], name='affected_cat')	     # categories for social protection
helped_cats   = pd.Index(['helped','not_helped'], name='helped_cat')

# These parameters could vary by country
reconstruction_time = 3.00 #time needed for reconstruction
reduction_vul       = 0.20 # how much early warning reduces vulnerability
inc_elast           = 1.50 # income elasticity
discount_rate       = 0.06 # discount rate
asset_loss_covered  = 0.80 # becomes 'shareable' 
max_support         = 0.05 # fraction of GDP

# Define directories
def set_directories(myCountry):  # get current directory
    global inputs, intermediate
    inputs        = model+'/../inputs/'+myCountry+'/'       # get inputs data directory
    intermediate  = model+'/../intermediate/'+myCountry+'/' # get outputs data directory

    # If the depository directories don't exist, create one:
    if not os.path.exists(inputs): 
        print('You need to put the country survey files in a directory titled ','/inputs/'+myCountry+'/')
        assert(False)
    if not os.path.exists(intermediate):
        os.makedirs(intermediate)

    return intermediate

def get_economic_unit(myC):
    
    if myC == 'PH': return 'province'
    elif myC == 'FJ': return 'Division'#'tikina'
    elif myC == 'SL': return 'district'#'tikina'
    else: return None

def get_currency(myC):
    
    if myC == 'PH': return 'PhP'
    elif myC == 'FJ': return 'FJD'
    elif myC == 'SL': return 'LKR'
    else: return 'XXX'

def get_places(myC,economy):
    # This df should have just province code/name and population

    if myC == 'PH': 
        df = pd.read_excel(inputs+'population_2015.xlsx',sheetname='population').set_index(economy)
        df['psa_pop']      = df['population']    # Provincial population
        df.drop(['population'],axis=1,inplace=True)
        return df

    if myC == 'FJ':
        df = pd.read_excel(inputs+'HIES 2013-14 Housing Data.xlsx',sheetname='Sheet1').set_index('Division').dropna(how='all')[['HHsize','Weight']].prod(axis=1).sum(level='Division').to_frame()
        df.columns = ['population']
        return df

    if myC == 'SL':
        df = pd.read_csv(inputs+'/Sri_Lanka/dataframe/finalhhdataframes/finalhhframe.csv').set_index('district').dropna(how='all')[['weight','np']].prod(axis=1).sum(level='district').to_frame()
        df.columns = ['population']
        return df

    else: return None

def get_places_dict(myC):

    if myC == 'PH': 
        return pd.read_excel(inputs+'FIES_provinces.xlsx')[['province_code','province_AIR']].set_index('province_code').squeeze() 

    if myC == 'FJ':
        return pd.read_excel(inputs+'Fiji_provinces.xlsx')[['code','name']].set_index('code').squeeze() 

    else: return None

def load_survey_data(myC):
    
    #Each survey/country should have the following:
    # -> hhid
    # -> hhinc
    # -> pcinc
    # -> hhwgt
    # -> pcwgt
    # -> hhsize
    # -> hhsize_ae
    # -> hhsoc
    # -> pcsoc
    # -> ispoor

    if myC == 'PH':
        df = pd.read_csv(inputs+'fies2015.csv',usecols=['w_regn','w_prov','w_mun','w_bgy','w_ea','w_shsn','w_hcn','walls','roof','totex','cash_abroad',
                                                          'cash_domestic','regft','hhwgt','fsize','poorhh','totdis','tothrec','pcinc_s','pcinc_ppp11','pcwgt'])
        df = df.rename(columns={'tothrec':'hhsoc','pcinc_s':'pcinc','poorhh':'ispoor'})
        
        df['pcinc_ae']   = df['pcinc']
        df['pcwgt_ae']   = df['pcwgt']

        df['hhsize']     = df['pcwgt']/df['hhwgt']
        df['hhsize_ae']  = df['pcwgt']/df['hhwgt']        

        df['hhinc'] = df[['pcinc','hhsize']].prod(axis=1)

        df['pcsoc']  = df['hhsoc']/df['hhsize']

        return df

    elif myC == 'FJ':
        df = pd.read_excel(inputs+'HIES 2013-14 Income Data.xlsx',usecols=['HHID','Division','Nchildren','Nadult','AE','HHsize',
                                                                           'Sector','Weight','TOTALTRANSFER','TotalIncome','New Total']).set_index('HHID')
        df = df.rename(columns={'HHID':'hhid','TotalIncome':'hhinc','HHsize':'hhsize','Weight':'hhwgt','TOTALTRANSFER':'hhsoc'})

        #df['hhsize_ae'] = df['Nadult'] # assuming this is 'Adult Equivalents'
        df['hhsize_ae'] = df['AE'] # assuming this is 'Adult Equivalents'
        # Should be equivalent to 0.5*df['Nchildren']+df['Nadult']

        df['pcwgt'] = df[['hhsize','hhwgt']].prod(axis=1)
        df['pcwgt_ae'] = df[['AE','hhwgt']].prod(axis=1)
        df['pcinc'] = df['hhinc']/df['hhsize']
        df['pcinc_ae'] = df['hhinc']/df['hhsize_ae']
        df['pcsoc'] = df['hhsoc']/df['hhsize']

        df_housing = pd.read_excel(inputs+'HIES 2013-14 Housing Data.xlsx',sheetname='Sheet1').set_index('HHID').dropna(how='all')[['Constructionofouterwalls',
                                                                                                                                    'Conditionofouterwalls']]
        
        df_poor = pd.read_excel(inputs+'HIES 2013-14 Demographic Data.xlsx',sheetname='Sheet1').set_index('HHID').dropna(how='all')['Poor']
        df_poor = df_poor[~df_poor.index.duplicated(keep='first')]

        df = pd.concat([df,df_housing,df_poor],axis=1).reset_index().set_index('Division')

        df = df.rename(columns={'Poor':'ispoor'})

        return df

    else: return None


def get_df2(myC):

    if myC == 'PH':
        df2 = pd.read_excel(inputs+'PSA_compiled.xlsx',skiprows=1)[['province','gdp_pc_pp','pop','shewp','shewr']].set_index('province')
        df2['gdp'] = df2['gdp_pc_pp']*df2['pop']
        return df2

    else: return None

def get_vul_curve(myC,struct):

    if myC == 'PH':
        return pd.read_excel(inputs+'vulnerability_curves_FIES.xlsx',sheetname=struct)[['desc','v']]

    if myC == 'FJ':
        df = pd.read_excel(inputs+'vulnerability_curves_Fiji.xlsx',sheetname=struct)[['desc','v']]
        return df

    else: return None

def get_hazard_df(myC,economy):

    if myC == 'PH': 
        df = get_AIR_data(inputs+'/Risk_Profile_Master_With_Population.xlsx','Loss_Results','Private','Agg').reset_index()
        df.columns = [economy,'hazard','rp','value_destroyed']
        return df
    
    if myC == 'FJ':
        df = pd.read_csv(inputs+'fj_tikina_aal.csv')[['TIKINA','PROVINCE','TID','Total_AAL','Total_Valu']].set_index(['TIKINA','PROVINCE','TID']).sum(level='PROVINCE').reset_index()
        df.columns = [economy,'value_destroyed','total_value']
        df = df.set_index([df.Division.replace({'Nadroga':'Nadroga-Navosa'})])        
        return df

    else: return None

def get_poverty_line(myC):
    
    if myC == 'PH':
        return 22302.6775#21240.2924

    if myC == 'FJ':
        # 55.12 per week for an urban adult
        # 49.50 per week for a rural adult
        # children under age 14 are counted as half an adult
        return 55.12*52. #this is for an urban adult
    

def get_subsistence_line(myC):
    
    if myC == 'PH':
        return 14832.0962*(22302.6775/21240.2924)
    
    else: return None

def get_to_USD(myC):

    if myC == 'PH': return 50.0
    elif myC == 'FJ': return 0.48
    else: return 0.

def get_scale_fac(myC):
    
    if myC == 'PH': return [1.E6,' [Millions]']
    elif myC == 'FJ': return [1.E3,' [Thousands]']
    else: return [1,'']

def get_avg_prod(myC):
    
    if myC == 'PH': return 0.337960802589002
    elif myC == 'FJ': return 0.336139019412

def get_demonym(myC):
    
    if myC == 'PH': return 'Filipinos'
    elif myC == 'FJ': return 'Fijians'
