import os, glob
import pandas as pd
pd.set_option('display.width', 220)

import matplotlib.pyplot as plt
from libraries.plot_hist import plot_simple_hist
from libraries.lib_drought import *
from libraries.lib_gather_data import *

import seaborn as sns
sns.set_style('whitegrid')
sns_pal = sns.color_palette('Set1', n_colors=8, desat=.5)

global model_
model = os.getcwd()

# People/hh will be affected or not_affected, and helped or not_helped
affected_cats = pd.Index(['a', 'na'], name='affected_cat') # categories for social protection
helped_cats   = pd.Index(['helped','not_helped'], name='helped_cat')

# These parameters could vary by country
reconstruction_time = 3.00 # time needed for reconstruction
reduction_vul       = 0.20 # how much early warning reduces vulnerability
inc_elast           = 1.50 # income elasticity
max_support         = 0.05 # max expenditure on PDS as fraction of GDP (set to 5% based on PHL)
nominal_asset_loss_covered_by_PDS = 0.80 # also, elsewhere, called 'shareable' 

# Dictionary for standardizing HIES column names that is SPECIFIC TO ROMANIA
hbs_dict = {'regiune':'Region','judet':'County','codla':'hhcode','coefj':'hhwgt','cpers':'hhsize','R44':'hhinc',
            'REGIUNE':'Region','JUDET':'County','CODLA':'hhcode','COEFJ':'hhwgt','CPERS':'hhsize',
            'NRGL':'nrgl','MEDIU':'mediu','CENTRA':'centra'}

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
    
    if myC == 'PH': return 'region'#'province'
    if myC == 'FJ': return 'Division'#'tikina'
    if myC == 'SL': return 'district'
    if myC == 'MW': return 'district'
    if myC == 'RO': return 'Region'
    if myC == 'BO': return 'departamento'
    assert(False)

def get_currency(myC):
    
    if myC == 'PH': return ['b. PhP',1.E9,1./50.]
    if myC == 'FJ': return ['k. F\$',1.E3,1./2.]
    if myC == 'SL': return ['LKR',1.E9,1./150.]
    if myC == 'MW': return ['MWK',1.E9,1./724.64]
    if myC == 'RO': return ['RON',1.E9,1/4.166667]
    if myC == 'BO': return ['BOB',1.E9,1/7.143]
    return ['XXX',1.E0,1E0]

def get_hhid_elements(myC):
    if myC == 'RO': return ['Region','County','centra','hhcode','nrgl','mediu']
    return None

def get_places(myC,economy):
    # This df should have just province code/name and population

    if myC == 'RO':

        # Dictionary for standardized column names
        RO_hhid = get_hhid_elements(myC)
        # nrgl	= Order number of the household within the place of residence

        # Load df0 <- this is going to me the master index of households in RO 
        hbs_str = inputs+'ROU_2016_HBS_v01_M/Data/Stata/s0.dta'
        df0 = pd.read_stata(hbs_str).rename(columns=hbs_dict)[RO_hhid+['ri','hhwgt']].set_index(RO_hhid)

        # BELOW: i think these are duplicative, ie hhwgt=0 when hh refused interview
        df0 = df0.loc[(df0['hhwgt']!=0)&(df0['ri']=='The household accepts the interview')]
        df0 = df0.drop('ri',axis=1)

        #df0 = df0.reset_index().set_index('hhid')
        #df0.loc[df0.index.duplicated(keep=False)].to_csv('~/Desktop/tmp/df0_dupes.csv')
        #assert(False)

        # Load df1
        df1 = pd.read_stata(inputs+'ROU_2016_HBS_v01_M/Data/Stata/s1.dta').rename(columns=hbs_dict)[RO_hhid+['hhsize']].set_index(RO_hhid)
        df1 = df1['hhsize'].max(level=RO_hhid)

        # Merge df0 and df1
        df  = pd.merge(df0.reset_index(),df1.reset_index(),on=RO_hhid)
        df[RO_hhid] = df[RO_hhid].astype('int')

        # Create unique hhid ('codla' seems not to be unique) 
        df['hhid'] = ''
        for id_col in RO_hhid: df['hhid'] = df['hhid'].astype('str') + df[id_col].astype('str')+('_' if id_col != RO_hhid[-1] else '')
        df = df.reset_index().set_index('hhid')
        
        df['pcwgt'] = df[['hhsize','hhwgt']].prod(axis=1)
        df.to_csv(inputs+'ROU_2016_HBS_v01_M/ro_weights.csv')

        assert(df.loc[df.index.duplicated(keep=False)].shape[0]==0)

        
        # "County" field is populated with garbage
        df = df.set_index(economy)['pcwgt'].sum(level=economy).to_frame()
        df.columns = ['pop']
    
        return df

    if myC == 'PH':
        df = pd.read_excel(inputs+'population_2015.xlsx',sheet_name='population').set_index('province').rename(columns={'population':'psa_pop'})
        return df

    if myC == 'FJ':
        df = pd.read_excel(inputs+'HIES 2013-14 Housing Data.xlsx',sheet_name='Sheet1').set_index('Division').dropna(how='all')[['HHsize','Weight']].prod(axis=1).sum(level='Division').to_frame()
        df.columns = ['population']
        return df
    
    if myC == 'SL':

        df_hhwgt = pd.read_csv(inputs+'HIES2016/weight2016.csv')
        df_hhwgt = df_hhwgt.rename(columns={'Finalweight':'hhwgt'})
        
        df_hhsize = pd.read_csv(inputs+'HIES2016/sec_1_demographic_information.csv')

        df_hhsize['hhsize'] = df_hhsize.groupby(['District','Sector','Psu','Snumber','Hhno'])['Person_Serial_No'].transform('count')

        # Set flag if HH has children
        df_hhsize['is_child'] = 0
        df_hhsize.loc[(df_hhsize['Birth_Year']>=98)|(df_hhsize['Birth_Year']<18),'is_child'] = 1
            
        df_hhsize = pd.merge(df_hhsize.reset_index(),df_hhwgt.reset_index(),on=['District','Sector','Psu'])

        df_hhsize['hhid'] = (df_hhsize['District'].astype('str')+df_hhsize['Sector'].astype('str')
                             +df_hhsize['Psu'].astype('str')+df_hhsize['Snumber'].astype('str')+df_hhsize['Hhno'].astype('str'))
        df_hhsize = df_hhsize.reset_index().set_index(['District','Sector','Psu','Snumber','Hhno']).sort_index()
        
        # Now get rid of duplicates: 
        df = df_hhsize[['hhsize','hhwgt']].mean(level=['District','Sector','Psu','Snumber','Hhno'])
        df['N_children'] = df_hhsize['is_child'].sum(level=['District','Sector','Psu','Snumber','Hhno'])
        # count children
        df['ethnicity'] = df_hhsize.loc[df_hhsize['Person_Serial_No']==1,'Ethnicity']
        # label ethnicity (1=sinhala, 2=tamil, 3=indian tamil, 4=moors, 5=malay, 6=burgher, 7=other)
        df['religion'] = df_hhsize.loc[df_hhsize['Person_Serial_No']==1,'Religion']
        # label religion (1=buddhist, 2=hindu, 3=islam, 4=christian, 9=other0

        df.to_csv(inputs+'/HIES2016/df_hhwgt_and_hhsize.csv')
        # this will be used in get_survey_data, rather than redoing hhsize
            
        df = df.reset_index().rename(columns={'District':'district'}).set_index('district')
        df['headcount'] = df[['hhsize','hhwgt']].prod(axis=1)
            
        df = df['headcount'].sum(level='district').to_frame(name='pop')
        # ^ return this

        return df

    if myC == 'MW':

        df_agg = pd.read_stata(inputs+'consumption_aggregates_poverty_ihs4.dta')
        df_agg['district'].replace('Nkhatabay','Nkhata Bay',inplace=True)
        df_agg['district'].replace('Zomba Non-City','Zomba',inplace=True)
        df_agg = df_agg.set_index(['district','case_id']).dropna(how='all').drop([_c for _c in ['index'] if _c in df_agg.columns],axis=1)

        df_agg = df_agg[['hh_wgt','hhsize']].prod(axis=1).sum(level='district').to_frame()
        df_agg.columns = ['population']

        df_agg = df_agg.sort_values(by='population',ascending=False)
        
        return df_agg   

    else: return None

def get_places_dict(myC):
    p_code,r_code = None,None

    if myC == 'PH': 
        p_code = pd.read_excel(inputs+'FIES_provinces.xlsx')[['province_code','province_AIR']].set_index('province_code').squeeze()
        p_code[97] = 'Zamboanga del Norte'
        p_code[98] = 'Zamboanga Sibugay'
        r_code = pd.read_excel(inputs+'FIES_regions.xlsx')[['region_code','region_name']].set_index('region_code').squeeze()
        
    if myC == 'FJ':
        p_code = pd.read_excel(inputs+'Fiji_provinces.xlsx')[['code','name']].set_index('code').squeeze()

    if myC == 'SL':
        p_code = pd.read_excel(inputs+'Admin_level_3__Districts.xls')[['DISTRICT_C','DISTRICT_N']].set_index('DISTRICT_C').squeeze()
        p_code.index.name = 'district'

    if myC == 'MW':
        p_code = pd.read_csv(inputs+'MW_code_region_district.csv')[['code','district']].set_index('code').dropna(how='all')
        p_code.index = p_code.index.astype(int)

        r_code = pd.read_csv(inputs+'MW_code_region_district.csv')[['code','region']].set_index('code').dropna(how='all')
        r_code.index = r_code.index.astype(int)   

    if myC == 'RO':
        r_code = pd.read_csv(inputs+'region_info.csv')[['HBS_code','Region']].set_index('HBS_code').dropna(how='all').squeeze()
        r_code.index = r_code.index.astype(int)  

    try: p_code = p_code.to_dict()
    except: pass
    try: r_code = r_code.to_dict()
    except: pass

    return p_code,r_code

def load_survey_data(myC):

    df = None
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
    # -> has_ew

    if myC == 'RO':

        # hbs_dict defined above as Dictionary for standard column names 
        # Array to recreate hhid:
        RO_hhid = get_hhid_elements(myC)

        # LOAD: hhid, hhwgt, pcwgt, hhsize, hhsize_ae
        df_wgts = pd.read_csv(inputs+'ROU_2016_HBS_v01_M/ro_weights.csv').set_index('hhid')
        df_wgts['hhsize_ae'] = df_wgts['hhsize'].copy()
        df_wgts['aewgt'] = df_wgts['pcwgt'].copy()

        # LOAD: hhinc, pcinc, hsoc, pcsoc
        _df = pd.read_stata(inputs+'ROU_2016_HBS_v01_M/Data/Stata/s7.dta').rename(columns=hbs_dict)[RO_hhid+['hhinc','R14','R15','R16','R17','R18','R19','R20','R21','R22','R23','R24',
                                                                                                             'R25','R26','R27','R28','R29','R54','R55','R56','R57','R58','R59','R60',
                                                                                                             'R47','R52']]
        _df['hhsoc'] = _df[['R14','R15','R16','R17','R18','R19','R20','R21','R22','R23','R24','R25','R26','R27','R28','R29','R54','R55','R56','R57','R58','R59','R60']].sum(axis=1)
        _df = _df.rename(columns={'R52':'precautionary_savings','R47':'loans_and_credit'})
        # -->  R47 = Loans and credits taken
        # -->  R52 = SAVINGS!!!

        _df[RO_hhid] = _df[RO_hhid].astype('int')
        df = pd.merge(df_wgts.reset_index(),_df.reset_index()[RO_hhid+['hhinc','hhsoc','precautionary_savings','loans_and_credit']],on=RO_hhid).set_index(RO_hhid)

        df['hhinc'] *= 52.
        df['pcinc'] = df.eval('hhinc/hhsize')
        df = df.loc[df.pcinc!=0]
        #
        df['hhsoc'] *= 52.
        df['pcsoc'] = df.eval('hhsoc/hhsize')
        #
        df['ispoor'] = 0
        #
        #print(df[['pcinc','pcwgt']].prod(axis=1).sum()/df['pcwgt'].sum())        
        #print(df[['pcsoc','pcwgt']].prod(axis=1).sum()/df['pcwgt'].sum())
        #

        # Load: construction materials
        _df = pd.read_stata(inputs+'ROU_2016_HBS_v01_M/Data/Stata/s10a.dta').rename(columns=hbs_dict).rename(columns={'MATCONS':'walls'})[RO_hhid+['walls']]
        _df[RO_hhid] = _df[RO_hhid].astype('int')
        _df = _df.set_index(RO_hhid)[['walls']]
        df = pd.merge(df.reset_index(),_df.reset_index(),on=RO_hhid).set_index(RO_hhid)

        # LOAD: has_ew
        _df = pd.read_stata(inputs+'ROU_2016_HBS_v01_M/Data/Stata/s10b.dta').rename(columns=hbs_dict)[RO_hhid+['INZES_12','INZES_17','INZES_18','INZES_25']]
        _df['has_ew'] = _df[['INZES_12','INZES_17','INZES_18','INZES_25']].sum(axis=1).clip(upper=1)
        
        _df[RO_hhid] = _df[RO_hhid].astype('int')
        _df = _df.set_index(RO_hhid)[['has_ew']]
        
        df = pd.merge(df.reset_index(),_df.reset_index(),on=RO_hhid).set_index(RO_hhid)

        # Load file with region names
        region_dict = pd.read_csv(inputs+'region_info.csv').set_index('HBS_code')['Region'].to_dict()
        
        df = df.reset_index(['Region'])
        df['Region'] = df['Region'].replace(region_dict)
        df = df.reset_index().set_index(RO_hhid)

        df['c'] = df['pcinc'].copy()
        #df['social'] = df.eval('pcsoc/pcinc')

        # pop & write out hh savings & loans/credit
        pd.concat([df['hhid']]+[df.pop(x) for x in ['precautionary_savings','loans_and_credit']], 1).to_csv('../intermediate/RO/hh_savings.csv')
                
    if myC == 'MW':

        df_agg = pd.read_stata(inputs+'consumption_aggregates_poverty_ihs4.dta').set_index(['case_id']).dropna(how='all')
        df = df_agg[['district','hh_wgt','rexpagg','pline_2016','upline_2016','poor','upoor','hhsize','adulteq','urban']].copy()
        df['district'].replace('Nkhatabay','Nkhata Bay',inplace=True)
        df['district'].replace('Zomba Non-City','Zomba',inplace=True)

        df = df.reset_index().rename(columns={'hh_wgt':'hhwgt',
                                              'rexpagg':'hhinc',
                                              'pline_2016':'pov_line',
                                              'upline_2016':'sub_line',
                                              'case_id':'hhid',
                                              'adulteq':'hhsize_ae'}).set_index(['hhid']).sort_index()
        df.district = df.district.astype('object')
        df['ispoor'],df['issub'] = 0,0
        df.loc[df['poor']=='Poor','ispoor'] = 1
        df.loc[df['upoor']=='Ultra-poor','issub'] = 1
        df = df.drop(['poor','upoor'],axis=1)

        df['pcwgt'] = df[['hhwgt','hhsize']].prod(axis=1)
        df['pcinc'] = df['hhinc']/df['hhsize']

        df['aeinc'] = df['hhinc']/df['hhsize_ae']
        df['aewgt'] = df[['hhwgt','hhsize_ae']].prod(axis=1)        
    
        # Still need hhsoc & pcsoc
        dfR = pd.read_stata(inputs+'MWI_2016_IHS-IV_v02_M_Stata/HH_MOD_R.dta').rename(columns={'case_id':'hhid'}).set_index('hhid')
        df['hhsoc'] = dfR[['hh_r02a','hh_r02b']].sum(axis=1).sum(level='hhid')
        df['pcsoc'] = df['hhsoc']/df['hhsize']
    
        df['hhsoc_kg_maize'] = dfR['hh_r02c'].sum(level='hhid')
        # ^ not clear what to do with this

        #need dwelling info
        dfF = pd.read_stata(inputs+'MWI_2016_IHS-IV_v02_M_Stata/HH_MOD_F.dta').rename(columns={'case_id':'hhid'}).set_index('hhid')
        #df['general_construction'] = dfF['hh_f06'].copy()
        df['walls'] = dfF['hh_f07'].copy().astype('object')
        df['roof'] = dfF['hh_f08'].copy().astype('object')
        #df['floor'] = dfF['hh_f09'].copy()

        # AXFIN (also in mod F)
        dfF['axfin'] = 0.
        dfF.loc[(dfF.hh_f48=='YES')|(dfF.hh_f50=='YES')|(dfF.hh_f52=='YES')|(dfF.hh_f54=='YES'),'axfin'] = 1.
        df['axfin'] = dfF['axfin'].copy()

        # Early warning info
        dfL = pd.read_stata(inputs+'MWI_2016_IHS-IV_v02_M_Stata/HH_MOD_L.dta').rename(columns={'case_id':'hhid'}).set_index('hhid')
        df['has_ew'] = dfL.loc[((dfL.hh_l02=="Radio ('wireless')")
                                |(dfL.hh_l02=="Radio with flash drive/micro CD")
                                |(dfL.hh_l02=="Television")
                                |(dfL.hh_l02=="Computer equipment & accessories")
                                |(dfL.hh_l02=="Sattelite dish")|(dfL.hh_l02=="Satellite dish")),'hh_l03'].sum(level='hhid').clip(upper=1.0)

        # Agricultural income
        # --> Does the hh have an agricultural enterprise?
        dfN1 = pd.read_stata(inputs+'MWI_2016_IHS-IV_v02_M_Stata/HH_MOD_N1.dta').rename(columns={'case_id':'hhid'}).set_index('hhid')
        #dfN2 = pd.read_stata(inputs+'MWI_2016_IHS-IV_v02_M_Stata/HH_MOD_N2.dta').rename(columns={'case_id':'hhid'}).set_index('hhid')
        df['enterprise_ag'] = 0
        df.loc[dfN1.hh_n02=='YES','enterprise_ag'] = 1

        # --> Income from sale of agricultural assets (crops?)
        dfP = pd.read_stata(inputs+'MWI_2016_IHS-IV_v02_M_Stata/HH_MOD_P.dta').rename(columns={'case_id':'hhid'}).set_index('hhid')
        df['income_ag_assets'] = dfP.loc[(dfP['hh_p0a']=='Income from Household Agricultural/Fishing Asset Sales'),'hh_p02'].fillna(0)

        # --> GANYU income (MK) over last 12 months
        dfE = pd.read_stata(inputs+'MWI_2016_IHS-IV_v02_M_Stata/HH_MOD_E.dta').rename(columns={'case_id':'hhid'}).set_index('hhid')
        dfE[['hh_e56','hh_e57','hh_e58','hh_e59']] = dfE[['hh_e56','hh_e57','hh_e58','hh_e59']].fillna(0)
        df['income_ganyu'] = dfE[['hh_e56','hh_e57','hh_e58','hh_e59']].prod(axis=1).sum(level='hhid')

        # --> N hh members working in unpaid agricultural labor
        dfE['labor_ag_unpaid'] = 0
        dfE.loc[dfE['hh_e06_8a']=='UNPAID HOUSEHOLD LABOR(AGRIC)','labor_ag_unpaid'] = 1.
        df['labor_ag_unpaid'] = dfE['labor_ag_unpaid'].sum(level='hhid')

        # --> N hh members working in ganyu        
        dfE['labor_ganyu'] = 0
        dfE.loc[dfE['hh_e06_8a']=='GANYU','labor_ganyu'] = 1.
        df['labor_ganyu'] = dfE['labor_ganyu'].sum(level='hhid')

        # --> Main wage job
        dfE['main_wage_job_ag'] = 0
        dfE.loc[(dfE.hh_e19b==62),'main_wage_job_ag'] = 1
        df['main_wage_job_ag'] = dfE['main_wage_job_ag'].sum(level='hhid')

        # Drought impact Questionnaire
        dfU = pd.read_stata(inputs+'MWI_2016_IHS-IV_v02_M_Stata/HH_MOD_U.dta').rename(columns={'case_id':'hhid'}).set_index('hhid')
        df['impact_drought'] = dfU.loc[(dfU.hh_u0a=='Drought'),'hh_u01'].replace('Yes',1).replace('No',0)

        # Irrigated?
        dfK = pd.read_stata(inputs+'MWI_2016_IHS-IV_v02_M_Stata/AG_MOD_K.dta').rename(columns={'case_id':'hhid'}).set_index('hhid')
        df['has_irrigation'] = None
        df.loc[(dfK.loc[(dfK['ag_k29a']=='Rainfed/No irrigation')|(dfK['ag_k29a']=='Bucket')]).index.tolist(),'has_irrigation'] = False
        df.loc[(dfK.loc[(dfK['ag_k29a']!='Rainfed/No irrigation')&(dfK['ag_k29a']!='Bucket')]).index.tolist(),'has_irrigation'] = True

        print(df.loc[(df['impact_drought']==1)&(df['has_irrigation']==False),'pcwgt'].sum()/df.loc[(df['has_irrigation']==False),'pcwgt'].sum())
        print(df.loc[(df['impact_drought']==1)&(df['has_irrigation']==True),'pcwgt'].sum()/df.loc[(df['has_irrigation']==True),'pcwgt'].sum())

        # Total value of sales
        _df = pd.read_stata(inputs+'MWI_2016_IHS-IV_v02_M_Stata/AG_MOD_I.dta').rename(columns={'case_id':'hhid'}).set_index('hhid')       
        df['income_crop_wet'] = _df['ag_i03'].sum(level='hhid')
        df['income_crop_wet'] = df['income_crop_wet'].fillna(0)

        _df = pd.read_stata(inputs+'MWI_2016_IHS-IV_v02_M_Stata/AG_MOD_O.dta').rename(columns={'case_id':'hhid'}).set_index('hhid')
        df['income_crop_dry'] = _df['ag_o03'].sum(level='hhid')
        df['income_crop_dry'] = df['income_crop_dry'].fillna(0)

        _df = pd.read_stata(inputs+'MWI_2016_IHS-IV_v02_M_Stata/AG_MOD_Q.dta').rename(columns={'case_id':'hhid'}).set_index('hhid')
        df['income_permcrop'] = _df['ag_q03'].sum(level='hhid')
        df['income_permcrop'] = df['income_permcrop'].fillna(0)

        _df = pd.read_stata(inputs+'MWI_2016_IHS-IV_v02_M_Stata/AG_MOD_R1.dta').rename(columns={'case_id':'hhid'}).set_index('hhid')
        df['income_livestock'] = _df['ag_r17'].sum(level='hhid')
        df['income_livestock'] = df['income_livestock'].fillna(0)

        _df = pd.read_stata(inputs+'MWI_2016_IHS-IV_v02_M_Stata/AG_MOD_S.dta').rename(columns={'case_id':'hhid'}).set_index('hhid')
        df['income_animprod'] = _df['ag_s06'].sum(level='hhid')
        df['income_animprod'] = df['income_animprod'].fillna(0)

        df['income_ag_gross'] = df[['income_ganyu','income_crop_wet','income_crop_dry','income_permcrop','income_livestock','income_animprod']].sum(axis=1)
        df['income_ag_gross'] = df['income_ag_gross'].fillna(0)
        #print(df[['income_ganyu','income_crop_wet','income_crop_dry','income_permcrop','income_livestock','income_animprod']].head())

        # Total cost
        df['ag_input_cost'] = 0

        # labor
        _df = pd.read_stata(inputs+'MWI_2016_IHS-IV_v02_M_Stata/AG_MOD_D.dta').rename(columns={'case_id':'hhid'}).set_index('hhid')
        df['ag_input_cost'] += _df[['ag_d46a1','ag_d46b1']].prod(axis=1).sum(level='hhid')
        df['ag_input_cost'] += _df[['ag_d46a2','ag_d46b2']].prod(axis=1).sum(level='hhid')
        df['ag_input_cost'] += _df[['ag_d46a3','ag_d46b3']].prod(axis=1).sum(level='hhid')

        # rent
        _df = pd.read_stata(inputs+'MWI_2016_IHS-IV_v02_M_Stata/AG_MOD_B2.dta').rename(columns={'case_id':'hhid'}).set_index('hhid')
        df['ag_input_cost'] += _df[['ag_b209a','ag_b209b']].sum(axis=1).sum(level='hhid')        

        # fertilizer purchased with coupon
        _df = pd.read_stata(inputs+'MWI_2016_IHS-IV_v02_M_Stata/AG_MOD_E2.dta').rename(columns={'case_id':'hhid'}).set_index('hhid')
        df['ag_input_cost'] += _df['ag_e04'].sum(level='hhid')
        
        # fertilizer purchased without coupon
        _df = pd.read_stata(inputs+'MWI_2016_IHS-IV_v02_M_Stata/AG_MOD_F.dta').rename(columns={'case_id':'hhid'}).set_index('hhid')
        df['ag_input_cost'] += _df[['ag_f09','ag_f10']].sum(axis=1).sum(level='hhid')        
        
        # seed
        _df = pd.read_stata(inputs+'MWI_2016_IHS-IV_v02_M_Stata/AG_MOD_H.dta').rename(columns={'case_id':'hhid'}).set_index('hhid')
        df['ag_input_cost'] += _df[['ag_h09','ag_h10']].sum(axis=1).sum(level='hhid')

        # cost of transport for sales
        _df = pd.read_stata(inputs+'MWI_2016_IHS-IV_v02_M_Stata/AG_MOD_I.dta').rename(columns={'case_id':'hhid'}).set_index('hhid')
        df['ag_input_cost'] += _df['ag_i10'].sum(level='hhid')
        _df = pd.read_stata(inputs+'MWI_2016_IHS-IV_v02_M_Stata/AG_MOD_O.dta').rename(columns={'case_id':'hhid'}).set_index('hhid')
        df['ag_input_cost'] += _df['ag_o10'].sum(level='hhid')
        _df = pd.read_stata(inputs+'MWI_2016_IHS-IV_v02_M_Stata/AG_MOD_Q.dta').rename(columns={'case_id':'hhid'}).set_index('hhid')
        df['ag_input_cost'] += _df['ag_q10'].sum(level='hhid')
  
        # cost of livestock inputs
        _df = pd.read_stata(inputs+'MWI_2016_IHS-IV_v02_M_Stata/AG_MOD_R2.dta').rename(columns={'case_id':'hhid'}).set_index('hhid')
        print(_df.columns)
        df['ag_input_cost'] += _df[['ag_r25','ag_r26','ag_r27','ag_r28','ag_r29']].sum(axis=1).sum(level='hhid')

        # cost of advice
        _df = pd.read_stata(inputs+'MWI_2016_IHS-IV_v02_M_Stata/AG_MOD_T2.dta').rename(columns={'case_id':'hhid'}).set_index('hhid')
        df['ag_input_cost'] += _df['ag_t10'].sum(level='hhid')

        df['ag_input_cost'] = df['ag_input_cost'].fillna(0)

        ##########
        df['income_ag_net'] = df['income_ag_gross'].fillna(0)-df['ag_input_cost']
        #df['ratio_net_to_gross'] = (df['income_ag_net']/df['income_ag_gross']).fillna(0)

        print('Net income =',df[['income_ag_net','hhwgt']].prod(axis=1).sum())
        print('Gross income =',df[['income_ag_gross','hhwgt']].prod(axis=1).sum())

        # Plot net to gross income ratio
        #ax=plt.gca()
        #heights, bins = np.histogram(df.loc[df.ratio_net_to_gross>0,'ratio_net_to_gross'],bins=50,weights=df.loc[df.ratio_net_to_gross>0,'hhwgt']/1.E3)
        #ax.bar(bins[:-1], heights, width=(bins[1]-bins[0]), facecolor='blue',edgecolor=None,linewidth=0,alpha=0.45)
        #ax.get_figure().savefig('../output_plots/MW/ag_income_net_to_gross_ratio.pdf',format='pdf')        
        #plt.cla()  
        
        # Plot net income
        uclip = 3.E5

        ax=plt.gca()
        heights, bins = np.histogram(df['income_ag_net'].clip(lower=-1E4,upper=uclip),bins=50,weights=df['hhwgt']/1.E3)
        ax.bar(bins[:-1], heights, width=(bins[1]-bins[0]), facecolor='blue',edgecolor=None,linewidth=0,alpha=0.45)
        ax.get_figure().savefig('../output_plots/MW/ag_income_net.pdf',format='pdf')        
        plt.cla()        
        

        # Plot income by stream 
        ax=plt.gca()
        heights1, bins = np.histogram(df.loc[(df.income_ag_net>0)&(df.income_ganyu!=0),'income_ganyu'].clip(upper=uclip),bins=50,
                                      weights=df.loc[(df.income_ag_net>0)&(df.income_ganyu!=0),'hhwgt']/1.E3)

        heights2, bins = np.histogram(df.loc[(df.income_ag_net>0)&(df.income_crop_wet!=0),'income_crop_wet'].clip(upper=uclip),bins=50,
                                      weights=df.loc[(df.income_ag_net>0)&(df.income_crop_wet!=0),'hhwgt']/1.E3)

        heights3, bins = np.histogram(df.loc[(df.income_ag_net>0)&(df.income_crop_dry!=0),'income_crop_dry'].clip(upper=uclip),bins=50,
                                      weights=df.loc[(df.income_ag_net>0)&(df.income_crop_dry!=0),'hhwgt']/1.E3)

        heights4, bins = np.histogram(df.loc[(df.income_ag_net>0)&(df.income_permcrop!=0),'income_permcrop'].clip(upper=uclip),bins=50,
                                      weights=df.loc[(df.income_ag_net>0)&(df.income_permcrop!=0),'hhwgt']/1.E3)

        heights5, bins = np.histogram(df.loc[(df.income_ag_net>0)&(df.income_livestock!=0),'income_livestock'].clip(upper=uclip),bins=50,
                                      weights=df.loc[(df.income_ag_net>0)&(df.income_livestock!=0),'hhwgt']/1.E3)

        heights6, bins = np.histogram(df.loc[(df.income_ag_net>0)&(df.income_animprod!=0),'income_animprod'].clip(upper=uclip),bins=50,
                                      weights=df.loc[(df.income_ag_net>0)&(df.income_animprod!=0),'hhwgt']/1.E3)

        ax.bar(bins[:-1], heights1, width=(bins[1]-bins[0]), facecolor='blue',edgecolor=None,linewidth=0,alpha=0.45)
        ax.bar(bins[:-1], heights2, width=(bins[1]-bins[0]), facecolor='red',edgecolor=None,linewidth=0,alpha=0.45,bottom=heights1)
        ax.bar(bins[:-1], heights3, width=(bins[1]-bins[0]), facecolor='purple',edgecolor=None,linewidth=0,alpha=0.45,bottom=heights1+heights2)
        ax.bar(bins[:-1], heights4, width=(bins[1]-bins[0]), edgecolor=None,linewidth=0,alpha=0.45,bottom=heights1+heights2+heights3)
        ax.bar(bins[:-1], heights5, width=(bins[1]-bins[0]), edgecolor=None,linewidth=0,alpha=0.45,bottom=heights1+heights2+heights3+heights4)
        ax.bar(bins[:-1], heights6, width=(bins[1]-bins[0]), edgecolor=None,linewidth=0,alpha=0.45,bottom=heights1+heights2+heights3+heights4+heights5)
        ax.get_figure().savefig('../output_plots/MW/ag_income.pdf',format='pdf')        
        plt.cla()

        df.loc[(df.hhinc<7.5E5)&(df.income_ag_net>0)&(df.income_ag_net<7.5E5)].plot.scatter('hhinc','income_ag_net')
        plt.gca().get_figure().savefig('../output_plots/MW/ag_income_vs_total.pdf',format='pdf')    
        plt.cla()

        df['income_ag_net'] = df['income_ag_net'].clip(upper=df['hhinc'])
        df.loc[(df.hhinc<7.5E5)&(df.income_ag_net>0)&(df.income_ag_net<7.5E5)].plot.scatter('hhinc','income_ag_net')
        plt.gca().get_figure().savefig('../output_plots/MW/ag_income_clipped_vs_total.pdf',format='pdf')    
        plt.cla()
        

        print('Frac reporting drought:',df.loc[(df.impact_drought==1),'pcwgt'].sum()/df['pcwgt'].sum())        

        print('Frac reporting ganyu income:',df.loc[(df.income_ganyu>0),'pcwgt'].sum()/df['pcwgt'].sum())
        print('Frac w/ ganyu income reporting drought:',df.loc[(df.income_ganyu>0)&(df.impact_drought==1),'pcwgt'].sum()/df.loc[(df.income_ganyu>0),'pcwgt'].sum())
        print('Frac w/o ganyu income reporting drought:',df.loc[(df.income_ganyu==0)&(df.impact_drought==1),'pcwgt'].sum()/df.loc[(df.income_ganyu==0),'pcwgt'].sum())

        #drought_study(df)
        df = df[[i for i in df.columns if i not in ['enterprise_ag', 'income_ag_assets','labor_ag_unpaid','labor_ganyu', 'main_wage_job_ag',
                                                    'has_irrigation',
                                                    'income_ganyu','income_crop_wet', 'income_crop_dry', 'income_permcrop','income_livestock', 'income_animprod','income_ag_gross',
                                                    'ag_input_cost']]]

        #print(df.loc[(df.impact_drought==1)&(df.income_ag_net>0),'pcwgt'].sum()/df.loc[(df.income_ag_net>0),'pcwgt'].sum())
        #print(df.loc[(df.impact_drought==1)&(df.income_ag_net==0),'pcwgt'].sum()/df.loc[(df.income_ag_net==0),'pcwgt'].sum())
        
        print('Setting c to pcinc') 
        df['c'] = df['pcinc'].copy()
        
        df = df.reset_index().set_index('district').drop([_i for _i in ['index'] if _i in df.columns])

    elif myC == 'PH':
        df = pd.read_csv(inputs+'fies2015.csv')[['w_regn','w_prov','w_mun','w_bgy','w_ea','w_shsn','w_hcn',
                                                 'walls','roof',
                                                 'totex','cash_abroad','cash_domestic','regft',
                                                 'hhwgt','fsize','poorhh','totdis','tothrec','pcinc_s','pcinc_ppp11','pcwgt',
                                                 'radio_qty','tv_qty','cellphone_qty','pc_qty',
                                                 'savings','invest']]

        df = df.rename(columns={'tothrec':'hhsoc','poorhh':'ispoor','totex':'hhexp'})

        df['hhsize']     = df['pcwgt']/df['hhwgt']
        df['hhsize_ae']  = df['pcwgt']/df['hhwgt']
        df['aewgt']   = df['pcwgt'].copy()

        # Per capita expenditures
        df['pcexp'] = df['hhexp']/df['hhsize']

        # These lines use income as income
        df = df.rename(columns={'pcinc_s':'pcinc'})
        df['hhinc'] = df[['pcinc','hhsize']].prod(axis=1)

        # These lines use disbursements as proxy for income
        #df = df.rename(columns={'totdis':'hhinc'}) 
        #df['pcinc'] = df['hhinc']/df['hhsize']

        df['aeinc']   = df['pcinc'].copy()

        df['pcsoc']  = df['hhsoc']/df['hhsize']

        df['tot_savings'] = df[['savings','invest']].sum(axis=1,skipna=False)
        df['savings'] = df['savings'].fillna(-1)
        df['invest'] = df['invest'].fillna(-1)
        
        df['axfin']  = 0
        df.loc[(df.savings>0)|(df.invest>0),'axfin'] = 1

        df['est_sav'] = df[['axfin','pcinc']].prod(axis=1)/2.

        df['has_ew'] = df[['radio_qty','tv_qty','cellphone_qty','pc_qty']].sum(axis=1).clip(upper=1)
        df = df.drop(['radio_qty','tv_qty','cellphone_qty','pc_qty'],axis=1)

        # plot 1
        plot_simple_hist(df.loc[df.axfin==1],['tot_savings'],['hh savings'],'../output_plots/PH/hh_savings.pdf',uclip=None,nBins=25)

        # plot 2
        ax = df.loc[df.tot_savings>=0].plot.scatter('pcinc','tot_savings')
        ax.plot()
        plt.gcf().savefig('../output_plots/PH/hh_savings_scatter.pdf',format='pdf')

        # plot 3
        ax = df.loc[df.tot_savings>=0].plot.scatter('est_sav','tot_savings')
        plt.xlim(0,60000)
        plt.ylim(0,60000)
        ax.plot()
        plt.gcf().savefig('../output_plots/PH/hh_est_savings_scatter.pdf',format='pdf')      
        
        print(str(round(100*df[['axfin','hhwgt']].prod(axis=1).sum()/df['hhwgt'].sum(),2))+'% of hh report expenses on savings or investments\n')
    
        # Run savings script
        df['country'] = 'PH'
        listofquintiles=np.arange(0.10, 1.01, 0.10)
        df = df.reset_index().groupby('country',sort=True).apply(lambda x:match_percentiles(x,perc_with_spline(reshape_data(x.pcinc),reshape_data(x.pcwgt),listofquintiles),
                                                                                            'decile_nat',sort_val='pcinc')).drop(['index'],axis=1)
        df = df.reset_index().groupby('w_regn',sort=True).apply(lambda x:match_percentiles(x,perc_with_spline(reshape_data(x.pcinc),reshape_data(x.pcwgt),listofquintiles),
                                                                                           'decile_reg',sort_val='pcinc')).drop(['index'],axis=1)
        df = df.reset_index().set_index(['w_regn','decile_nat','decile_reg']).drop('index',axis=1)
        
        df['precautionary_savings'] = df['pcinc']-df['pcexp']

        # Savings rate by national decile
        _ = pd.DataFrame(index=df.sum(level='decile_nat').index)
        _['income'] = df[['pcinc','pcwgt']].prod(axis=1).sum(level='decile_nat')/df['pcwgt'].sum(level='decile_nat')
        _['expenditures'] = df[['pcexp','pcwgt']].prod(axis=1).sum(level='decile_nat')/df['pcwgt'].sum(level='decile_nat')
        _['precautionary_savings'] = _['income']-_['expenditures']
        _.sort_index().to_csv('../intermediate/'+myC+'/hh_savings_by_decile.csv')

        # Savings rate by decile (regionally-defined) & region
        _ = pd.DataFrame(index=df.sum(level=['w_regn','decile_reg']).index)
        _['income'] = df[['pcinc','pcwgt']].prod(axis=1).sum(level=['w_regn','decile_reg'])/df['pcwgt'].sum(level=['w_regn','decile_reg'])
        _['expenditures'] = df[['pcexp','pcwgt']].prod(axis=1).sum(level=['w_regn','decile_reg'])/df['pcwgt'].sum(level=['w_regn','decile_reg'])
        _['precautionary_savings'] = _['income']-_['expenditures']
        _.sort_index().to_csv('../intermediate/'+myC+'/hh_savings_by_decile_and_region.csv')

        # Savings rate for hh in subsistence (natl average)
        listofquartiles=np.arange(0.25, 1.01, 0.25)
        df = df.reset_index().groupby('country',sort=True).apply(lambda x:match_percentiles(x,perc_with_spline(reshape_data(x.precautionary_savings),reshape_data(x.pcwgt),listofquartiles),
                                                                                            'nat_sav_quartile',sort_val='precautionary_savings'))
        df = df.reset_index().groupby('w_regn',sort=True).apply(lambda x:match_percentiles(x,perc_with_spline(reshape_data(x.precautionary_savings),reshape_data(x.pcwgt),listofquartiles),
                                                                                           'reg_sav_quartile',sort_val='precautionary_savings')).drop(['index'],axis=1)
        df = df.reset_index().set_index(['w_regn','decile_nat','decile_reg']).drop('index',axis=1).sort_index()

        _ = pd.DataFrame()
        _.loc['subsistence_savings_rate','hh_avg'] = (df.loc[df.pcinc<get_subsistence_line(myC)].eval('pcwgt*(pcinc-pcexp)').sum()
                                                      /df.loc[df.pcinc<get_subsistence_line(myC),'pcwgt'].sum())
        _.loc['subsistence_savings_rate','hh_q1'] = df.loc[df.nat_sav_quartile==1,'precautionary_savings'].max()
        _.loc['subsistence_savings_rate','hh_q2'] = df.loc[df.nat_sav_quartile==2,'precautionary_savings'].max()
        _.loc['subsistence_savings_rate','hh_q3'] = df.loc[df.nat_sav_quartile==3,'precautionary_savings'].max()


        _.sort_index().to_csv('../intermediate/'+myC+'/hh_savings_in_subsistence_natl.csv')

        # Savings rate for hh in subsistence (by region)
        _ = pd.DataFrame()
        _['hh_avg'] = (df.loc[df.pcinc<get_subsistence_line(myC)].eval('pcwgt*(pcinc-pcexp)').sum(level='w_regn')
                       /df.loc[df.pcinc<get_subsistence_line(myC),'pcwgt'].sum(level='w_regn'))
        _['hh_q1'] = df.loc[df.reg_sav_quartile==1,'precautionary_savings'].max(level='w_regn')
        _['hh_q2'] = df.loc[df.reg_sav_quartile==2,'precautionary_savings'].max(level='w_regn')
        _['hh_q3'] = df.loc[df.reg_sav_quartile==3,'precautionary_savings'].max(level='w_regn')
        _.sort_index().to_csv('../intermediate/'+myC+'/hh_savings_in_subsistence_reg.csv')

        if False:
            _.plot.scatter('income','expenditures')
            plt.gcf().savefig('../output_plots/PH/income_vs_exp_by_decile_PH.pdf',format='pdf')
            plt.cla()
            
            _.plot.scatter('income','precautionary_savings')
            plt.gcf().savefig('../output_plots/PH/net_income_vs_exp_by_decile_PH.pdf',format='pdf')
            plt.cla()
            
            df.boxplot(column='aprecautionary_savings',by='decile')
            plt.ylim(-1E5,1E5)
            plt.gcf().savefig('../output_plots/PH/net_income_by_exp_decile_boxplot_PH.pdf',format='pdf')
            plt.cla()

        # Rename one column to be 'c' --> consumption
        print('Using per cap income as c')
        df['c'] = df['pcinc'].copy()

        # Drop unused columns
        df = df.reset_index().set_index(['w_regn','w_prov','w_mun','w_bgy','w_ea','w_shsn','w_hcn'])
        df = df.drop([_c for _c in ['country','decile_nat','decile_reg','est_sav','tot_savings','savings','invest',
                                    'precautionary_savings','index','level_0','cash_domestic'] if _c in df.columns],axis=1)

    elif myC == 'FJ':
        
        # This is an egregious hack, but deadlines are real
        # 4.632E9 = GDP in USD (2016, WDI)
        # 0.48 = conversion to FJD
        # --> inc_sf is used to scale PCRAFI/AIR losses to hh assets, 
        # ----> because hh-based estimate of assets in country is VERY different (factor of 4 for some asset types) from PCRAFI estimate
        inc_sf = (4.632E9/0.48)

        df = pd.read_excel(inputs+'HIES 2013-14 Income Data.xlsx')[['HHID','Division','Nchildren','Nadult','AE','HHsize',
                                                                    'Sector','Weight','TOTALTRANSFER','TotalIncome','New Total',
                                                                    'CareandProtectionProgrampaymentfromSocialWelfare',
                                                                    'FamilyAssistanceProgrampaymentfromSocialWelfare',
                                                                    'FijiNationalProvidentFundPension',
                                                                    'FNPFWithdrawalsEducationHousingInvestmentsetc',
                                                                    'SocialPensionScheme',
                                                                    'TotalBusiness','TotalPropertyIncome'
                                                                    ]]
        df = df.rename(columns={'TotalIncome':'hhinc','HHsize':'hhsize','Weight':'hhwgt','TOTALTRANSFER':'hhsoc',
                                'CareandProtectionProgrampaymentfromSocialWelfare':'SP_CPP',
                                'FamilyAssistanceProgrampaymentfromSocialWelfare':'SP_FAP',
                                'FijiNationalProvidentFundPension':'SP_FNPF',
                                'FNPFWithdrawalsEducationHousingInvestmentsetc':'SP_FNPF2',
                                'SocialPensionScheme':'SP_SPS'}).set_index('HHID')

        df['pov_line'] = 0.
        df.loc[df.Sector=='Urban','pov_line'] = get_poverty_line(myC,sec='Urban')
        df.loc[df.Sector=='Rural','pov_line'] = get_poverty_line(myC,sec='Rural')

        if inc_sf != None: df['hhinc'], df['pov_line'] = scale_hh_income_to_match_GDP(df[['hhinc','hhwgt','hhsize','AE','Sector','pov_line']],inc_sf,flat=True)

        df['hh_pov_line'] = df[['pov_line','AE']].prod(axis=1)

        df['hhsize_ae'] = df['AE'] # This is 'Adult Equivalents'
        # ^ Equivalent to 0.5*df['Nchildren']+df['Nadult']

        df['pcwgt']    = df[['hhsize','hhwgt']].prod(axis=1)
        df['aewgt'] = df[['AE','hhwgt']].prod(axis=1)

        df['pcinc']    = df['hhinc']/df['hhsize']
        df['aeinc'] = df['hhinc']/df['hhsize_ae']

        df['pcsoc']    = df['hhsoc']/df['hhsize']
            
        df_housing = pd.read_excel(inputs+'HIES 2013-14 Housing Data.xlsx',sheet_name='Sheet1').set_index('HHID').dropna(how='all')[['Constructionofouterwalls',
                                                                                                                                    'Conditionofouterwalls',
                                                                                                                                     'Radio','TV','TelephoneLines','Mobilephones',
                                                                                                                                     'Computers','Internet','Ipads','Smartphones']]
        df_housing = df_housing.rename(columns={'Constructionofouterwalls':'walls','Conditionofouterwalls':'walls_condition'})
        df_housing['has_ew'] = df_housing[['Radio','TV','TelephoneLines','Mobilephones','Computers','Internet','Ipads','Smartphones']].sum(axis=1).clip(upper=1)
        df_housing = df_housing.drop(['Radio','TV','TelephoneLines','Mobilephones','Computers','Internet','Ipads','Smartphones'],axis=1)

        df_dems = pd.read_excel(inputs+'HIES 2013-14 Demographic Data.xlsx',sheet_name='Sheet1').set_index('HHID').dropna(how='all')[['Poor','Age']].fillna(0)

        df_dems['isOld'] = 0
        df_dems.loc[df_dems.Age >=68,'isOld'] = 1
        df_dems['nOlds'] = df_dems['isOld'].sum(level='HHID')
        df_dems = df_dems[~df_dems.index.duplicated(keep='first')].drop(['Age','isOld'],axis=1)

        df = pd.concat([df,df_housing,df_dems],axis=1).reset_index().set_index('Division')

        df = df.rename(columns={'Poor':'ispoor'})

        # Fiji also has social safety net-- set flag if household gets income from each program
        # plot income from these programs
        plot_simple_hist(df.loc[(df.SP_CPP != 0)],['SP_CPP'],['Care & Protection Program'],'../output_plots/FJ/sectoral/SP_CPP_income.pdf',uclip=1500,nBins=25)
        plot_simple_hist(df.loc[(df.SP_FAP != 0)],['SP_FAP'],['Family Assistance Program'],'../output_plots/FJ/sectoral/SP_FAP_income.pdf',uclip=1500,nBins=25)
        plot_simple_hist(df.loc[(df.SP_FAP != 0)|(df.SP_CPP != 0)],['SP_FAP','SP_CPP'],
                         ['Family Assistance Program','Care & Protection Program'],'../output_plots/FJ/sectoral/SP_income.pdf',uclip=1000)

        # SP_CPP = CareandProtectionProgrampaymentfromSocialWelfare
        df.loc[df.SP_CPP != 0,'SP_CPP'] = True
        df.loc[df.SP_CPP == 0,'SP_CPP'] = False
        # SP_FAP = FamilyAssistanceProgrampaymentfromSocialWelfare
        df.loc[df.SP_FAP != 0,'SP_FAP'] = True
        df.loc[df.SP_FAP == 0,'SP_FAP'] = False
    
        # SP_PBS = PovertyBenefitsScheme
        df['SP_PBS'] = False
        df.sort_values(by='pcinc',ascending=True,inplace=True)
        hhno,hhsum = 0,0

        while (hhno < df.shape[0]) and (hhsum < 24000):
            hhsum = df.iloc[:hhno].hhwgt.sum()
            hhno+=1
        df.iloc[:hhno].SP_PBS = True
        # This line generates warning about setting value on a copy..
        print(df.loc[df.SP_PBS == True,'hhwgt'].sum())

        # SP_SPS = SocialProtectionScheme
        # --> or retirees (over 65/formerly, 68) not enrolled in FNPF
        #df.loc[df.SP_SPS != 0,'SP_SPS'] = True
        #df.loc[df.SP_SPS == 0,'SP_SPS'] = False
        # SP_PBS = PovertyBenefitsScheme
        df['SP_SPS'] = False
        df.sort_values(by='aeinc',ascending=True,inplace=True)
        hhno,sumOlds = 0,0
        while (hhno < df.shape[0]) and (sumOlds < 15000):
            sumOlds = (df.iloc[:hhno])[['hhwgt','nOlds']].prod(axis=1).sum()
            hhno+=1
        df.iloc[:hhno].SP_SPS = True
        df.loc[df.nOlds == 0,'SP_SPS'] = False

        # SP_FNPF = FijiNationalPensionFund
        # SP_FNPF2 = FNPFWithdrawalsEducationHousingInvestmentsetc
        df.loc[(df.SP_FNPF != 0)|(df.SP_FNPF2 != 0),'SP_FNPF'] = True
        df.loc[df.SP_FNPF == 0,'SP_FNPF'] = False

        # SPP_core = Social Protection+, core beneficiaries
        # SPP_add  = Social Protection+, additional beneficiaries
        df['SPP_core'] = False
        df['SPP_add']  = False
        df.sort_values(by='aeinc',ascending=True,inplace=True)

        # BELOW: these lines assume we're counting AE 
        # admission: this is before I learned about cumsum -- it was a very late night in Sydney office.
        # it's really stupid, but it works and I don't have time to fix now
        hhno,hhsum = 0,0
        while (hhno < df.shape[0]) and (hhsum < 25000):
            hhsum = df.iloc[:hhno].hhwgt.sum()
            hhno+=1
        df.iloc[:hhno].SPP_core = True
        hhno_add, hhsum_add = hhno, 0
        while (hhno_add < df.shape[0]) and (hhsum_add < 29000):
            hhsum_add = df.iloc[hhno:hhno_add].hhwgt.sum()
            hhno_add+=1
        df.iloc[hhno:hhno_add].SPP_add = True

        df = df.rename(columns={'HHID':'hhid'})
        print('Setting c to pcinc') 
        df['c'] = df['pcinc'].copy()
        

    elif myC == 'SL':
            
        # This file has household weights
        #df = pd.read_csv(inputs+'/HIES2016/weight2016.csv')
        df = pd.read_csv(inputs+'/HIES2016/df_hhwgt_and_hhsize.csv')
        df['hhid'] = df['District'].astype('str')+df['Sector'].astype('str')+df['Psu'].astype('str')+df['Snumber'].astype('str')+df['Hhno'].astype('str')
           
        dist_names = pd.read_excel(inputs+'Admin_level_3__Districts.xls')[['DISTRICT_C','DISTRICT_N']].set_index('DISTRICT_C').to_dict()['DISTRICT_N']
        df['dist_name'] = [dist_names[_dn] for _dn in df['District'].values]
 
        df = df.reset_index().set_index(['District','hhid']).drop(['index','Psu','Snumber','Hhno'],axis=1)

        ##########################
        ## Add ethnicity info:
        #df_demog = pd.read_csv(inputs+'HIES2016/sec_1_demographic_information.csv')
        #df_demog['hhid'] = df_demog['District'].astype('str')+df_demog['Sector'].astype('str')+df_demog['Psu'].astype('str')+df_demog['Snumber'].astype('str')+df_demog['Hhno'].astype('str')
        #df_demog = df_demog.set_index(['District','hhid'])[['Ethnicity']]
        #df_demog['eth_2'] = df_demog.groupby('hhid')['Ethnicity'].transform('mean')
        #df_demog.loc[df_demog.Ethnicity!=df_demog.eth_2,'Ethnicity'] = -1
        #df_demog = df_demog.mean(level=['District','hhid'])[['Ethnicity']]

        #print(df.head())
        #print(df_demog.head())
        

        #df['ethn'] = df_demog['Ethnicity'].copy()
        #print(df['ethn'].shape[0],df['ethn'].dropna().shape[0])
        #assert(df['ethn'].shape[0]==df['ethn'].dropna().shape[0])
        #df['ethn'] = df['ethn'].fillna(0)

        #########################
        # Add expenditures info:
        # Food:
        df_food = pd.read_csv(inputs+'/HIES2016/sec_4_1_food_consumption_and_expenditure.csv')
        df_food['hhid'] = df_food['District'].astype('str')+df_food['Sector'].astype('str')+df_food['Psu'].astype('str')+df_food['Snumber'].astype('str')+df_food['Hhno'].astype('str')
        df_food = df_food.reset_index().set_index(['District','hhid'])
        #
        hh_food = pd.DataFrame(index=df_food.sum(level=['District','hhid']).index.copy())
        hh_food['total_value'] = df_food['Value'].sum(level=['District','hhid'])
        #
        df['c_food'] = hh_food['total_value'].copy()*52.
        
        # Non-Food:
        df_nonfood = pd.read_csv(inputs+'/HIES2016/sec_4_2_non_food_consumption_and_expenditure.csv').fillna(0)
        df_nonfood['hhid'] = (df_nonfood['District'].astype('str')+df_nonfood['Sector'].astype('str')
                              +df_nonfood['Psu'].astype('str')+df_nonfood['Snumber'].astype('str')+df_nonfood['Hhno'].astype('str'))
        df_nonfood = df_nonfood.reset_index().set_index(['District','hhid'])
        #
        df_nonfood['frequency'] = 0. # monthly
        df_nonfood.loc[((df_nonfood['Nf_Code']>=2000)&(df_nonfood['Nf_Code']<2100)  # Housing (average per month)
                        |(df_nonfood['Nf_Code']>=2100)&(df_nonfood['Nf_Code']<2200) # Fuel & light (avg. per month)
                        |(df_nonfood['Nf_Code']>=2200)&(df_nonfood['Nf_Code']<2300) # Personal care expense (last month)
                        |(df_nonfood['Nf_Code']>=2300)&(df_nonfood['Nf_Code']<2400) # Health expenses (last month)
                        |(df_nonfood['Nf_Code']>=2400)&(df_nonfood['Nf_Code']<2500) # Transport & transport fees (last month)
                        |(df_nonfood['Nf_Code']>=2500)&(df_nonfood['Nf_Code']<2600) # Communication (last month)
                        |(df_nonfood['Nf_Code']>=2600)&(df_nonfood['Nf_Code']<2700) # Education (last month)
                        |(df_nonfood['Nf_Code']>=2700)&(df_nonfood['Nf_Code']<2800) # Recreation, entertaintment, and cultural activities (last month)
                        |(df_nonfood['Nf_Code']>=2800)&(df_nonfood['Nf_Code']<2900) # Nondurable household goods (last month)
                        |(df_nonfood['Nf_Code']>=2900)&(df_nonfood['Nf_Code']<3000) # Household services (last month)
                        ),'frequency'] = 12. # monthly
        df_nonfood.loc[(df_nonfood['Nf_Code']>=3000)&(df_nonfood['Nf_Code']<3300),'frequency'] = 2.  # Clothing & textiles, footwear, & durables (last 6 months)
        df_nonfood.loc[(df_nonfood['Nf_Code']>=3300)&(df_nonfood['Nf_Code']<3400),'frequency'] = 1.  # Durables (last year)
        df_nonfood.loc[(df_nonfood['Nf_Code']>=3400)&(df_nonfood['Nf_Code']<3500),'frequency'] = 12. # Other expenses (last month)
        df_nonfood.loc[(df_nonfood['Nf_Code']>=3500)&(df_nonfood['Nf_Code']<3600),'frequency'] = 1.  # annually            
        print('\n\n',df_nonfood.loc[df_nonfood.frequency==0].shape[0],' item(s) in hh survey not counted:\n',df_nonfood.loc[df_nonfood.frequency==0],'\n\n')
        # 
        hh_nonfood = pd.DataFrame(index=df_nonfood.sum(level=['District','hhid']).index.copy())
        hh_nonfood['total_value'] = df_nonfood.eval('(Nf_Value)*frequency').sum(level=['District','hhid'])
        #
        df['c_nonfood'] = hh_nonfood['total_value'].copy()
        #
        
        # Servants!
        df_servants = pd.read_csv(inputs+'/HIES2016/sec_4_3_boarders.csv').fillna(0)
        df_servants['hhid'] = (df_servants['District'].astype('str')+df_servants['Sector'].astype('str')
                               +df_servants['Psu'].astype('str')+df_servants['Snumber'].astype('str')+df_servants['Hhno'].astype('str'))
        #
        #hh_servants = pd.DataFrame(index=df_servants.sum(level=['District','hhid']).index.copy())
        hh_servants = df_servants.reset_index().set_index(['District','hhid']).sort_index().sum(level=['District','hhid'])
        hh_servants['total_value'] = hh_servants.eval('52.*(Col_3)+12.*(Col_4+Col_5+Col_6+Col_7+Col_8+Col_9+Col_10+Col_11+Col_12+Col_13)+2.*(Col_14)+1.*(Col_15)')
        #
        df['c_boarders'] = hh_servants['total_value'].copy()
        df = df.fillna(0)
        
        #####################################
        # Sum these streams
        df['hhinc'] = df.eval('c_food+c_nonfood')
        df['pcinc'] = df.eval('hhinc/hhsize')
        #
        df['pcwgt'] = df.eval('hhwgt*hhsize')
        df['aewgt'] = df['pcwgt'].copy()
        df['hhsize_ae'] = df['hhsize'].copy()
        #####################################            
                        
        # Get hhsoc
        df_social = pd.read_csv(inputs+'/HIES2016/sec_5_5_1_other_income.csv').fillna(0)
        df_social['hhid'] = (df_social['District'].astype('str')+df_social['Sector'].astype('str')
                             +df_social['Psu'].astype('str')+df_social['Snumber'].astype('str')+df_social['Hhno'].astype('str'))
        df_social = df_social.reset_index().set_index(['District','hhid']).sort_index().drop(['index','Sector','Month','Psu','Snumber','Hhno','Nhh','Result','Serial_5_5_1'],axis=1)
        df_social = df_social.sum(level=['District','hhid'])
        df_social['hhsoc'] = df_social.eval('12.*(Pension+Disability_And_Relief+Samurdhi+Elder+Tb+Scholar+Sc_Lunch+Threeposha)+Income_Forign+Income_Local')
        df_social['hhremittance'] = df_social.eval('Income_Forign+Income_Local')
        df_social['hhsamurdhi'] = df_social.eval('12.*Samurdhi')
        df_social['frac_remittance'] = df_social.eval('hhremittance/hhsoc')
        
        df['hhsoc'] = 0; df['frac_remittance'] = 0
        df['hhsoc'].update(df_social['hhsoc'])
        df['frac_remittance'].update(df_social['frac_remittance'])

        df['hhsoc'] = df['hhsoc'].clip(upper=df['hhinc'])
        #df['hhremittance'] = df['hhremittance'].clip(upper=df['hhinc'])
        # ^ By construction, can't be above 1
        df['pcsoc'] = df.eval('hhsoc/hhsize')
        
        df['hhsamurdhi'] = 0
        df['hhsamurdhi'].update(df_social['hhsamurdhi'])
        df['pcsamurdhi'] = df.eval('hhsamurdhi/hhsize')
        df['gsp_samurdhi'] = df['pcsamurdhi']/df[['pcwgt','pcsamurdhi']].prod(axis=1).sum()
        df = df.drop('hhsamurdhi',axis=1)

        # Get early warning info
        df_ew = pd.read_csv(inputs+'/HIES2016/sec_6a_durable_goods.csv').fillna(0)  
        df_ew['hhid'] = (df_ew['District'].astype('str')+df_ew['Sector'].astype('str')
                         +df_ew['Psu'].astype('str')+df_ew['Snumber'].astype('str')+df_ew['Hhno'].astype('str'))
        df_ew = df_ew.reset_index().set_index(['District','hhid']).sort_index().drop(['index','Sector','Month','Psu','Snumber','Hhno','Nhh','Result'],axis=1)    
        
        df_ew['has_ew'] = 0
        df_ew.loc[df_ew.eval('(Radio==1)|(Tv==1)|(Telephone==1)|(Telephone_Mobile==1)|(Computers==1)'),'has_ew'] = 1
        df['has_ew'] = df_ew['has_ew'].copy()
        
        # Set flag for whether household is poor
        sl_pov_by_dist = get_poverty_line('SL',by_district=True).reset_index()
        sl_pov_by_dist.columns = ['dist_name','pov_line']
        
        df = pd.merge(df.reset_index(),sl_pov_by_dist.reset_index(),on='dist_name').reset_index().set_index(['District','hhid']).drop(['index','level_0'],axis=1)
        assert(df['pov_line'].dropna().shape[0] == df['pov_line'].shape[0]) # <-- check that all districts have a poverty line
        
        df['ispoor'] = False
        df.loc[df['pcinc']<df['pov_line'],'ispoor'] = True
        
        print('Poverty rate:\n',100.*df.loc[df.ispoor==True,'pcwgt'].sum()/df['pcwgt'].sum())
        # ADB stat: In Sri Lanka, 4.1% of the population lives below the national poverty line in 2016.
        # https://www.adb.org/countries/sri-lanka/poverty
        
        # Get housing construction & materials info
        df_housing = pd.read_csv(inputs+'/HIES2016/sec_8_housing.csv').fillna(0)
        df_housing['hhid'] = (df_housing['District'].astype('str')+df_housing['Sector'].astype('str')
                              +df_housing['Psu'].astype('str')+df_housing['Snumber'].astype('str')+df_housing['Hhno'].astype('str'))
        df_housing = df_housing.reset_index().set_index(['District','hhid']).sort_index().drop(['index','Sector','Month','Psu',
                                                                                                'Snumber','Hhno','Nhh','Result'],axis=1)[['Walls','Floor','Roof']]
        df_housing = df_housing.rename(columns={_c:_c.lower() for _c in df_housing.columns})
        
        df[['walls','floor','roof']] = df_housing[['walls','floor','roof']].copy()
        
        df['c'] = df['pcinc'].copy()
        df = df.reset_index().rename(columns={'District':'district'}).set_index(['district','hhid']).drop(['c_food','c_nonfood','c_boarders'],axis=1)  

    # Assing weighted household consumption to quintiles within each province
    print('Finding quintiles')

    economy = df.index.names[0]
    listofquintiles=np.arange(0.20, 1.01, 0.20)
    df = df.reset_index().groupby(economy,sort=True).apply(lambda x:match_percentiles(x,perc_with_spline(reshape_data(x.c),reshape_data(x.pcwgt),listofquintiles),'quintile'))

    listofdeciles = np.arange(0.10, 1.01, 0.10)
    df = df.reset_index().groupby(economy,sort=True).apply(lambda x:match_percentiles(x,perc_with_spline(reshape_data(x.c),reshape_data(x.pcwgt),listofdeciles),'decile'))

    df.drop([icol for icol in ['level_0','index','pctle_05','pctle_05_nat'] if icol in df.columns],axis=1,inplace=True)

    # Last thing: however 'c' was set (income or consumption), pcsoc can't be higher than 0.99*that!
    df['pcsoc'] = df['pcsoc'].clip(upper=0.99*df['c'])

    return df

def get_df2(myC):

    if myC == 'PH':
        df2 = pd.read_excel(inputs+'PSA_compiled.xlsx',skiprows=1)[['province','gdp_pc_pp','pop','shewp','shewr']].set_index('province')
        df2['gdp_pp'] = df2['gdp_pc_pp']*df2['pop']
        return df2

    else: return None

def get_vul_curve(myC,struct):
    df = None
    if myC == 'PH': df = pd.read_excel(inputs+'vulnerability_curves_FIES.xlsx',sheet_name=struct)[['desc','v']]
    if myC == 'FJ': df = pd.read_excel(inputs+'vulnerability_curves_Fiji.xlsx',sheet_name=struct)[['desc','v']]
    if myC == 'SL': df = pd.read_excel(inputs+'vulnerability_curves.xlsx',sheet_name=struct)[['key','v']].rename(columns={'key':'desc'})
    if myC == 'MW': df = pd.read_excel(inputs+'vulnerability_curves_MW.xlsx',sheet_name=struct)[['desc','v']]
    if myC == 'RO': df = pd.read_excel(inputs+'vulnerability_curves_RO.xlsx',sheet_name=struct)[['desc','v']]
    return df
    
def get_infra_stocks_data(myC):
    if myC == 'FJ':
        infra_stocks = pd.read_csv(inputs+'infra_stocks.csv',index_col='sector')
        return infra_stocks
    else:return None
    
def get_wb_or_penn_data(myC):
    #iso2 to iso3 table
    names_to_iso2 = pd.read_csv(inputs+'names_to_iso.csv')[['iso2','country']].drop_duplicates().set_index('country').squeeze()
    K = pd.read_csv(inputs+'avg_prod_k_with_gar_for_sids.csv',index_col='Unnamed: 0')
    wb = pd.read_csv(inputs+'wb_data.csv',index_col='country')
    wb['Ktot'] = wb.gdp_pc_pp*wb['pop']/K.avg_prod_k
    wb['GDP'] = wb.gdp_pc_pp*wb['pop']
    wb['avg_prod_k'] = K.avg_prod_k
    print(wb['avg_prod_k'])
    assert(False)

    wb['iso2'] = names_to_iso2
    return wb.set_index('iso2').loc[myC,['Ktot','GDP','avg_prod_k']]
    
def get_rp_dict(myC):
    return pd.read_csv(inputs+"rp_dict.csv").set_index("old_rp").new_rp
    
def get_infra_destroyed(myC,df_haz):

    #print(get_infra_stocks_data(myC))

    infra_stocks = get_infra_stocks_data(myC).loc[['transport','energy','water'],:]
    infra_stocks['infra_share'] = infra_stocks.value_k/infra_stocks.value_k.sum()

    print(infra_stocks)
   
    hazard_ratios_infra = broadcast_simple(df_haz[['frac_inf','frac_destroyed_inf']],infra_stocks.index)
    hazard_ratios_infra = pd.merge(hazard_ratios_infra.reset_index(),infra_stocks.infra_share.reset_index(),on='sector',how='outer').set_index(['Division','hazard','rp','sector'])
    hazard_ratios_infra['share'] = hazard_ratios_infra['infra_share']*hazard_ratios_infra['frac_inf']
        
    transport_losses = pd.read_csv(inputs+"frac_destroyed_transport.csv").rename(columns={"ti_name":"Tikina"})
    transport_losses['Division'] = (transport_losses['tid']/100).astype('int')
    prov_code,_ = get_places_dict(myC)
    rp_dict   = get_rp_dict(myC)
    transport_losses['Division'] = transport_losses.Division.replace(prov_code)
    #sums at Division level to be like df_haz
    transport_losses = transport_losses.set_index(['Division','hazard','rp']).sum(level=['Division','hazard','rp'])
    transport_losses["frac_destroyed"] = transport_losses.damaged_value/transport_losses.value
    #if there is no result in transport_losses, use the PCRAFI data (from df_haz):
    transport_losses = pd.merge(transport_losses.reset_index(),hazard_ratios_infra.frac_destroyed_inf.unstack('sector')['transport'].to_frame(name="frac_destroyed_inf").reset_index(),on=['Division','hazard','rp'],how='outer')
    transport_losses['frac_destroyed'] = transport_losses.frac_destroyed.fillna(transport_losses.frac_destroyed_inf)
    transport_losses = transport_losses.set_index(['Division','hazard','rp'])
    
    hazard_ratios_infra = hazard_ratios_infra.reset_index('sector')
    hazard_ratios_infra.ix[hazard_ratios_infra.sector=='transport','frac_destroyed_inf'] = transport_losses["frac_destroyed"]
    hazard_ratios_infra = hazard_ratios_infra.reset_index().set_index(['Division','hazard','rp','sector'])

    return hazard_ratios_infra.rename(columns={'frac_destroyed_inf':'frac_destroyed'})
    
def get_service_loss(myC):
    if myC == 'FJ':
        service_loss = pd.read_csv(inputs+'service_loss.csv').set_index(['hazard','rp'])[['transport','energy','water']]
        service_loss.columns.name='sector'
        a = service_loss.stack()
        a.name = 'cost_increase'
        infra_stocks = get_infra_stocks_data(myC).loc[['transport','energy','water'],:]
        service_loss = pd.merge(pd.DataFrame(a).reset_index(),infra_stocks.e.reset_index(),on=['sector'],how='outer').set_index(['sector','hazard','rp'])
        return service_loss
    else:return None

def get_hazard_df(myC,economy,agg_or_occ='Occ',rm_overlap=False):

    if myC == 'PH': 
        df_prv = get_AIR_data(inputs+'/Risk_Profile_Master_With_Population_with_EP1.xlsx','Loss_Results','Private',agg_or_occ).reset_index()
        df_pub = get_AIR_data(inputs+'/Risk_Profile_Master_With_Population_with_EP1.xlsx','Loss_Results','Public',agg_or_occ).reset_index()

        df_prv.columns = ['province','hazard','rp','value_destroyed_prv']
        df_pub.columns = ['province','hazard','rp','value_destroyed_pub']
        
        df_prv = df_prv.reset_index().set_index(['province','hazard','rp'])
        df_pub = df_pub.reset_index().set_index(['province','hazard','rp'])

        df_prv['value_destroyed_pub'] = df_pub['value_destroyed_pub']
        df_prv['hh_share'] = df_prv['value_destroyed_prv']/(df_prv['value_destroyed_pub']+df_prv['value_destroyed_prv'])
                
        df_prv = df_prv.reset_index().drop('index',axis=1).fillna(0)
        
        return df_prv,df_prv

    elif myC == 'MW':
        df_haz = pd.read_excel(inputs+'GAR/GAR_PML_curve_MW.xlsx',sheet_name='PML(mUSD)')[['rp','Earthquake','Flood','Drought']].set_index('rp')
        tot_exposure = float(pd.read_excel(inputs+'GAR/GAR_PML_curve_MW.xlsx',sheet_name='total_exposed_val').loc['Malawi','Total Exposure'].squeeze())

        ag_exposure_gross = float(pd.read_excel(inputs+'GAR/GAR_PML_curve_MW.xlsx',sheet_name='total_exposed_val').loc['Malawi','Gross value, maize'].squeeze())
        ag_exposure_net = float(pd.read_excel(inputs+'GAR/GAR_PML_curve_MW.xlsx',sheet_name='total_exposed_val').loc['Malawi','Net value, maize'].squeeze())

        df_haz = df_haz.rename(columns={'Earthquake':'EQ','Flood':'FF','Drought':'DR'})

        df_haz_AAL = df_haz.loc['AAL']
        df_haz = df_haz.drop('AAL').stack().to_frame()
        df_haz.index.names = ['rp','hazard']
        df_haz.columns = ['PML']
        
        # For all hazards except drought, frac_destroyed = PML/total_assets
        df_haz['frac_destroyed'] = df_haz['PML']/tot_exposure
        # For drought, frac_destroyed = PML/gross_production_maize
        df_haz.loc[df_haz.index.get_level_values('hazard') == 'DR','frac_destroyed'] = df_haz['PML']/ag_exposure_gross

        df_haz = df_haz.reset_index().set_index(['hazard','rp']).sort_index()

        df_haz['hh_share'] = 1.0
        
        # Need to broadcast to district
        df_geo = get_places('MW','district')

        df_geo['country'] = 'MW'
        df_haz['country'] = 'MW'
        
        rms_haz = pd.read_excel(inputs+'masdap/malawi_exposure.xls',sheet_name='malawi_exposure').rename(columns={'Dist_Name':'district','POP':'rms_pop'}).drop(['FID','the_geom'],axis=1)
        rms_haz['district'].replace(pd.read_csv(inputs+'MW_code_region_district.csv')[['code','district']].astype('str').set_index('code').to_dict()['district'],inplace=True)
        rms_haz = rms_haz.reset_index().set_index('district').sort_index().drop('index',axis=1)
        df_geo['rms_population'] = rms_haz['rms_pop'].copy()
        df_geo.to_csv(inputs+'population_hies_vs_rms.csv')
        # --> HIES sums to 16.3M; RMS to 12.8M, and pop of most districts in RMS is about 80% of HIES

        ################################
        # Work with RMS_haz
        # 1) stack RPs

        rms_haz = rms_haz.reset_index()
        _init_haz = rms_haz.copy()
        
        # stack affected_population first
        rms_haz = rms_haz.set_index([_c for _c in rms_haz.columns if 'APOP_RP' not in _c])
        rms_haz = rms_haz.stack()
        rms_haz.name = 'aff_pop'
        rms_haz = rms_haz.reset_index().set_index('district')
        rms_haz = rms_haz.rename(columns={'level_41':'rp'})
        rms_haz['rp'] = rms_haz['rp'].str.replace('APOP_RP','')
        rms_haz = rms_haz.reset_index().set_index(['district','rp'])

        # stack asset categories into RP
        for exp_data in [['AH_RP','aff_hh'],['ASPS_RP','aff_semiperm'],['ATS_RP','aff_trad'],['APS_RP','aff_perm']]:
            _ = _init_haz.copy()
            rms_haz = rms_haz.drop([_c for _c in _.columns if exp_data[0] in _c],axis=1)

            _ = _.set_index([_c for _c in _.columns if exp_data[0] not in _c])        
            _.columns = [_c.replace(exp_data[0],'') for _c in _.columns]
            _.columns.name = 'rp'
            _ = _.stack().reset_index().set_index(['district','rp'])
            rms_haz[exp_data[1]] = _[0].copy()

        for _fa in [['aff_pop','rms_pop'],['aff_hh','HH'],['aff_semiperm','N_SPST'],['aff_trad','N_TS'],['aff_perm','N_PST']]:
            rms_haz['fa_'+_fa[0][4:]] = rms_haz[_fa[0]]/rms_haz[_fa[1]]
            #rms_haz = rms_haz.drop(_fa,axis=1)

        rms_haz.to_csv(inputs+'rms_haz.csv')
        # I can see that the types of structures don't add information, and there is minimal RP-dependence
        # --> solution: average over RP, and find characteristic fa for each district
        # fix index
        rhx = rms_haz.index
        rms_haz.index = rms_haz.index.set_levels([rhx.levels[0].astype(str), rhx.levels[-1].astype(int)])
        # average over rp, keeping only 'fa_pop'
        rms_haz_dist,_ = average_over_rp(rms_haz['fa_pop'])
        rms_haz_dist.columns = ['fa']
        rms_haz_dist['avg_n_aff'],_ = average_over_rp(rms_haz['aff_pop'])
        rms_haz_dist['rms_pop'] = rms_haz['rms_pop'].mean(level='district')

        rms_fa_national = float(rms_haz_dist['avg_n_aff'].sum()/rms_haz_dist['rms_pop'].sum())

        # calc fraction of total affected
        #rms_haz_dist['frac_aff'] = rms_haz_dist['avg_n_aff']/rms_haz_dist['avg_n_aff'].sum()
        rms_haz_dist.to_csv(inputs+'rms_haz_district_level.csv')

        # Merge the datasets
        df_haz = pd.merge(df_geo.reset_index(),df_haz.reset_index(),on=['country'],how='outer').set_index(['district']).sort_index()
        df_haz['ff_frac_aff'] = 1;
        df_haz.loc[df_haz.hazard=='FF','ff_frac_aff'] = rms_haz_dist['fa']/rms_fa_national
        # Special hack for cities
        df_haz = df_haz.reset_index()
        
        for _city in [['Blantyre City','Blantyre'],
                      ['Lilongwe City','Lilongwe'],
                      ['Mzuzu City','Mzimba'],
                      ['Zomba City','Zomba']]:          
            _s = '(district=="'+_city[0]+'")&(hazard=="FF")'
            df_haz.loc[df_haz.eval(_s),'ff_frac_aff'] = df_haz.loc[df_haz.eval(_s),'ff_frac_aff'].fillna(df_haz.loc[df_haz.eval(_s.replace(_city[0],_city[1])),'ff_frac_aff'].mean())

        # Combine columns
        df_haz['frac_destroyed'] = df_haz[['frac_destroyed','ff_frac_aff']].prod(axis=1)

        df_haz.to_csv('~/Desktop/tmp/df_haz.csv')
        df_haz = df_haz.drop(['country','population','PML','ff_frac_aff'],axis=1) 

        return df_haz, df_haz
        
    elif myC == 'FJ':

        df_all_ah = pd.read_csv(inputs+'map_tikinas.csv').set_index('Tikina').drop('Country_ID',axis=1)
        df_all_ah['hazard'] = 'All Hazards'
        df_all_ah['asset_class'] = 'all'
        df_all_ah['asset_subclass'] = 'all'

        # LOAD FILES (by hazard, asset class) and merge hazards
        # load all building values
        df_bld_oth_tc =   pd.read_csv(inputs+'fiji_tc_buildings_tikina.csv').set_index('Tikina').drop('Country_ID',axis=1)
        df_bld_oth_et = pd.read_csv(inputs+'fiji_eqts_buildings_tikina.csv').set_index('Tikina').drop('Country_ID',axis=1)
        df_bld_oth_tc['hazard'] = 'TC'
        df_bld_oth_et['hazard'] = 'EQTS'
        df_bld_oth = pd.concat([df_bld_oth_tc,df_bld_oth_et])
        
        df_bld_oth['asset_class'] = 'bld_oth'
        df_bld_oth['asset_subclass'] = 'oth'
        df_bld_oth = df_bld_oth.reset_index().set_index(['Tikina','Tikina_ID','hazard','asset_class','asset_subclass'])  

        # load residential building values
        df_bld_res_tc =   pd.read_csv(inputs+'fiji_tc_buildings_res_tikina.csv').set_index('Tikina').drop('Country_ID',axis=1)
        df_bld_res_et = pd.read_csv(inputs+'fiji_eqts_buildings_res_tikina.csv').set_index('Tikina').drop('Country_ID',axis=1)
        df_bld_res_tc['hazard'] = 'TC'
        df_bld_res_et['hazard'] = 'EQTS'
        df_bld_res = pd.concat([df_bld_res_tc,df_bld_res_et])

        df_bld_res['asset_class'] = 'bld_res'
        df_bld_res['asset_subclass'] = 'res'
        df_bld_res = df_bld_res.reset_index().set_index(['Tikina','Tikina_ID','hazard','asset_class','asset_subclass'])  

        # Get PCRAFI estimate of total building stock
        df_bld_oth_EV = df_bld_oth['Exp_Value'].sum(level=['hazard']).mean()
        df_bld_res_EV = df_bld_res['Exp_Value'].sum(level=['hazard']).mean()
      
        # Stack RPs in building exposure files
        df_bld_oth.columns.name = 'rp'
        df_bld_res.columns.name = 'rp'
        df_bld_oth = df_bld_oth.reset_index().set_index(['Tikina','Tikina_ID','hazard','asset_class','asset_subclass','Exp_Value']).stack().to_frame(name='losses')
        df_bld_res = df_bld_res.reset_index().set_index(['Tikina','Tikina_ID','hazard','asset_class','asset_subclass','Exp_Value']).stack().to_frame(name='losses')
                
        df_bld_oth = df_bld_oth.reset_index().set_index(['Tikina','Tikina_ID','hazard','rp'])
        df_bld_res = df_bld_res.reset_index().set_index(['Tikina','Tikina_ID','hazard','rp'])

        # Scale building assets to Rashmin's analysis
        df_bld_oth['Exp_Value'] *= 6.505E9/df_bld_oth_EV
        df_bld_res['Exp_Value'] *= 4.094E9/df_bld_res_EV
        df_bld_oth['Exp_Value'] -= df_bld_res['Exp_Value']

        df_bld_oth['losses'] *= 6.505E9/df_bld_oth_EV
        df_bld_res['losses'] *= 4.094E9/df_bld_res_EV
        df_bld_oth['losses'] -= df_bld_res['losses']

        df_bld = pd.concat([df_bld_oth,df_bld_res])

        #############################
        # load infrastructure values
        df_inf_tc =   pd.read_csv(inputs+'fiji_tc_infrastructure_tikina.csv').set_index('Tikina').drop('Country_ID',axis=1)
        df_inf_et = pd.read_csv(inputs+'fiji_eqts_infrastructure_tikina.csv').set_index('Tikina').drop('Country_ID',axis=1)
        
        ##### corrects the infrastructure values
        if True:
        #just put False here if the new infrastructure values mess up the results
            df_inf_correction = pd.read_excel(inputs+"fj_infrastructure_v3.xlsx","Pivot by Tikina",skiprows=[0]).rename(columns={"Unnamed: 2":"Tikina","Tikina":"new_tikina","Tikina_ID":"new_Tikina_ID"})
            df_inf_correction = df_inf_correction[df_inf_correction.Region2!="Grand Total"]
            df_inf_correction = df_inf_correction.rename(columns={"Region2":"Tikina_ID"})
            df_inf_tc = df_inf_tc.reset_index().merge(df_inf_correction[["Tikina_ID","Total"]].dropna(),on="Tikina_ID",how="outer")
            df_inf_et = df_inf_et.reset_index().merge(df_inf_correction[["Tikina_ID","Total"]].dropna(),on="Tikina_ID",how="outer")
            df_inf_et["Total"] = df_inf_et.Total.fillna(df_inf_et.Exp_Value)
            df_inf_tc["Total"] = df_inf_tc.Total.fillna(df_inf_tc.Exp_Value)
            df_inf_et['Exp_Value'] = df_inf_et.Total
            df_inf_tc['Exp_Value'] = df_inf_tc.Total     
            df_inf_et = df_inf_et.drop(["Total"],axis=1).set_index("Tikina")
            df_inf_tc = df_inf_tc.drop(["Total"],axis=1).set_index("Tikina")        
        
        df_inf_tc['hazard'] = 'TC'        
        df_inf_et['hazard'] = 'EQTS'
        df_inf = pd.concat([df_inf_tc,df_inf_et])

        df_inf['asset_class'] = 'inf'
        df_inf['asset_subclass'] = 'all'

        # Get PCRAFI estimate of total infrastructure stock
        df_inf_EV = df_inf.loc[df_inf.hazard=='TC','Exp_Value'].sum()

        # Stack and scale RPs in infrastructure exposure file
        df_inf.columns.name = 'rp'
        df_inf = df_inf.reset_index().set_index(['Tikina','Tikina_ID','hazard','asset_class','asset_subclass','Exp_Value']).stack().to_frame(name='losses')
        df_inf = df_inf.reset_index().set_index(['Tikina','Tikina_ID','hazard','asset_class','asset_subclass','rp'])

        df_inf['losses'] *= (3.E09+9.6E08+5.15E08)/df_inf_EV
        df_inf['Exp_Value'] *= (3.E09+9.6E08+5.15E08)/df_inf_EV

        # load agriculture values
        df_agr_tc =   pd.read_csv(inputs+'fiji_tc_crops_tikina.csv').set_index('Tikina').drop('Country_ID',axis=1)
        df_agr_et = pd.read_csv(inputs+'fiji_eqts_crops_tikina.csv').set_index('Tikina').drop('Country_ID',axis=1)
        df_agr_tc['hazard'] = 'TC'
        df_agr_et['hazard'] = 'EQTS'
        df_agr = pd.concat([df_agr_tc,df_agr_et])
 
        df_agr['asset_class'] = 'agr'
        df_agr['asset_subclass'] = 'all'

        df_agr.columns.name = 'rp'
        df_agr = df_agr.reset_index().set_index(['Tikina','Tikina_ID','hazard','asset_class','asset_subclass','Exp_Value']).stack().to_frame(name='losses')
        df_agr = df_agr.reset_index().set_index(['Tikina','Tikina_ID','hazard','asset_class','asset_subclass','rp'])

        ############
        # Merge
        df_bld = df_bld.reset_index().set_index(['Tikina'])
        df_inf = df_inf.reset_index().set_index(['Tikina'])
        df_agr = df_agr.reset_index().set_index(['Tikina'])
        df = pd.concat([df_bld,df_inf,df_agr])
        #df = df.loc[df.rp != 'AAL']

        df = df.reset_index().set_index(['Tikina','Tikina_ID','asset_class','asset_subclass','Exp_Value','hazard','rp'])    
        #df.to_csv('~/Desktop/my_csv.csv')
        df = df.unstack()

        df = df.rename(columns={'exceed_2':2475,'exceed_5':975,'exceed_10':475,
                                'exceed_20':224,'exceed_40':100,'exceed_50':72,
                                'exceed_65':50,'exceed_90':22,'exceed_99':10,'AAL':1})
        
        df.columns.name = 'rp'
        df = df.stack()

        df = df.reset_index().set_index(['Tikina','Tikina_ID','asset_class','asset_subclass','hazard','rp'])
        df = df.rename(columns={'losses':'value_destroyed'})

        df = df.sort_index().reset_index()

        df['Division'] = (df['Tikina_ID']/100).astype('int')
        prov_code,_ = get_places_dict(myC)
        df = df.reset_index().set_index([df.Division.replace(prov_code)]).drop(['index','Division','Tikina_ID','asset_subclass'],axis=1) #replace district code with its name
        df_tikina = df.copy()
        
        df = df.reset_index().set_index(['Division','Tikina','hazard','rp','asset_class'])

        df_sum = ((df['value_destroyed'].sum(level=['Division','hazard','rp']))/(df['Exp_Value'].sum(level=['Division','hazard','rp']))).to_frame(name='frac_destroyed')
        # ^ Taking fa from all asset classes
        
        df = df.sum(level=['Division','hazard','rp','asset_class'])
        df = df.reset_index().set_index(['Division','hazard','rp'])

        # record affected assets for each asset class, hazard, rp
        df['frac_destroyed'] = df['value_destroyed']/df['Exp_Value']

        df_sum['Exp_Value'] = df['Exp_Value'].sum(level=['Division','hazard','rp'])
        #
        df_sum['frac_bld_res'] = df.loc[df.asset_class == 'bld_res','Exp_Value']/df['Exp_Value'].sum(level=['Division','hazard','rp'])
        df_sum['frac_bld_oth'] = df.loc[df.asset_class == 'bld_oth','Exp_Value']/df['Exp_Value'].sum(level=['Division','hazard','rp'])
        df_sum['frac_inf']     = df.loc[df.asset_class == 'inf','Exp_Value']/df['Exp_Value'].sum(level=['Division','hazard','rp'])
        df_sum['frac_agr']     = df.loc[df.asset_class == 'agr','Exp_Value']/df['Exp_Value'].sum(level=['Division','hazard','rp'])
        #

        #df_sum = ((df.loc[(df.asset_class == 'bld_res')|(df.asset_class == 'agr'),'value_destroyed'].sum(level=['Division','hazard','rp']))/(df.loc[(df.asset_class == 'bld_res')|(df.asset_class == 'agr'),'Exp_Value'].sum(level=['Division','hazard','rp']))).to_frame(name='fa')
        #
        df_sum['frac_destroyed_inf']     = df.loc[df.asset_class == 'inf','value_destroyed']/df.loc[df.asset_class == 'inf','Exp_Value']
        df_sum['frac_destroyed_bld_oth'] = df.loc[df.asset_class == 'bld_oth','value_destroyed']/df.loc[df.asset_class == 'bld_oth','Exp_Value']
        df_sum['frac_destroyed_bld_res'] = df.loc[df.asset_class == 'bld_res','value_destroyed']/df.loc[df.asset_class == 'bld_res','Exp_Value']
        df_sum['frac_destroyed_agr']     = df.loc[df.asset_class == 'agr','value_destroyed']/df.loc[df.asset_class == 'agr','Exp_Value']
        
        #################
        #adds SSBN floods
        if True:
            df_floods = pd.read_csv(inputs+"flood_fa.csv").rename(columns={"tid":"Tikina_ID","LS2012_pop":"Exp_Value"})
            df_floods['Division'] = (df_floods['Tikina_ID']/100).astype('int').replace(prov_code)
            
            product = [df_sum.reset_index().Division.unique(),df_floods.reset_index().hazard.unique(),df_floods.reset_index().rp.unique()]
            idx = pd.MultiIndex.from_product(product, names=['Division', 'hazard','rp'])
            df_floods_sum = pd.DataFrame(index=idx)

            df_floods_sum["frac_destroyed"] = (df_floods.set_index(['Division','hazard','rp'])[["frac_destroyed","Exp_Value"]].prod(axis=1).sum(level=['Division','hazard','rp'])/df_floods.set_index(['Division','hazard','rp'])["Exp_Value"].sum(level=['Division','hazard','rp']))
            df_floods_sum["frac_destroyed_inf"] = df_floods_sum["frac_destroyed"]
            df_floods_sum["frac_inf"] = broadcast_simple(df_sum.frac_inf.mean(level="Division"),df_floods_sum.index)
            
            df_sum = df_sum.append(df_floods_sum.fillna(0)) #the floods are appended in df_sum but only the frac_destroyed and frac_inf columns will have numbers
        
        print('\n')
        print('--> Total BLD =',round(df.loc[(df.asset_class == 'bld_oth')|(df.asset_class == 'bld_res'),'Exp_Value'].sum(level=['hazard','rp']).mean()/1.E6,1),'M USD (',
              round((100.*df.loc[(df.asset_class == 'bld_oth')|(df.asset_class == 'bld_res'),'Exp_Value'].sum(level=['hazard','rp'])/df['Exp_Value'].sum(level=['hazard','rp'])).mean(),1),'%)')
        print('--> Total INF =',round(df.loc[df.asset_class == 'inf','Exp_Value'].sum(level=['hazard','rp']).mean()/1.E6,1),'M USD (',
              round((100.*df.loc[df.asset_class == 'inf','Exp_Value'].sum(level=['hazard','rp'])/df['Exp_Value'].sum(level=['hazard','rp'])).mean(),1),'%)')
        print('--> Total AG =',round(df.loc[df.asset_class == 'agr','Exp_Value'].sum(level=['hazard','rp']).mean()/1.E6,1),'M USD (', 
              round((100.*df.loc[df.asset_class == 'agr','Exp_Value'].sum(level=['hazard','rp'])/df['Exp_Value'].sum(level=['hazard','rp'])).mean(),1),'%)\n')
        
        df_sum['Exp_Value'] *= (1.0/0.48) # AIR-PCRAFI in USD(2009?) --> switch to FJD

        df_sum = df_sum.reset_index().set_index(['Division'])
        df_sum.Exp_Value = df_sum.Exp_Value.mean(level='Division',skipna=True)

        return df_sum,df_tikina

    elif myC == 'SL':
        df = pd.read_excel(inputs+'hazards_data.xlsx',sheet_name='hazard').dropna(how='any')
        df.hazard = df.hazard.replace({'flood':'PF'})
        df = df.set_index(['district','hazard','rp'])

        # What else do we have:
        # loss data exceedance curves from GAR15
        # regional exposure data from Suranga
        # PDNS with total regional losses from 2017 landslides

        # try landslides
        path = os.getcwd()+'/../inputs/SL/data_hunting/suranga/Landslide/*.xls'
        landslide_df = None
        for f in glob.glob(path):
            
            new_reg = pd.read_excel(f).set_index(['District','Division'])
            new_reg = new_reg.rename(columns={' Houses Damaged':'Houses Damaged',
                                              ' Houses Destroyed':'Houses Destroyed',
                                              ' Affected_L':'Affected_L'})
            new_reg = new_reg.drop([_c for _c in new_reg.columns if _c not in ['Houses Damaged','Houses Destroyed','Affected_L']],axis=1)
            new_reg = new_reg.sum(level='District')

            if landslide_df is not None:
                landslide_df = pd.concat([landslide_df,new_reg])
            else: landslide_df = new_reg.copy()

        # This file has # hh (affected, damaged, and destroyed, by District)
        landslide_df.to_csv('../inputs/SL/data_hunting/suranga/Landslide/fa_landslide.csv')

        # Need to compare to household survey
        # ^ but what to do if I have a vulnerability
        
        return df,df
    
    elif myC == 'RO':

        # this file is created in libraries/lib_collect_hazard_data_RO
        df_haz = pd.read_csv('../inputs/RO/hazard/romania_multihazard_fa.csv').set_index(['Region','hazard','rp'])
        
        ## this file has total GDP (by county), which we'll use as denominator
        #df_gdp = pd.read_csv(inputs+'county_gdp.csv').set_index('County')[['GDP']]
        #
        ## this file links counties to regions, because HBS has only regional info
        #df_geo_info = pd.read_csv(inputs+'county_to_region.csv').set_index('County')[['Region']]# also has NUTS codes
        #
        ## merge above
        #df_counties = pd.merge(df_gdp.reset_index(),df_geo_info.reset_index(),on='County').set_index('County')
        #
        ## This file has capital loss to EQ (by county), which we'll use as numerator
        #df_EQ_caploss = pd.read_csv(inputs+'hazard/_EQ_caploss_by_county.csv').set_index('County')
        #df_EQ_caploss['hazard'] = 'EQ'
        #
        ## Merge above
        #df_haz = pd.merge(df_counties.reset_index(),df_EQ_caploss.reset_index(),on='County').set_index(['Region','hazard','rp']).drop(['County'],axis=1)
        #
        ## Sum county-level results to regions
        #df_haz = df_haz.sum(level=['Region','hazard','rp'])
        #
        ## Choose which exceedance curve to use:
        ## - Options:
        ## - 1) PML: probably maximum loss
        ## - 2) AAL: average annual loss
        ## - 3) AEP: annual exceedance probability
        #loss_measure_to_use = 'AAL'
        #
        #df_haz = df_haz.drop([_l for _l in ['AAL','AEP','PML'] if _l != loss_measure_to_use],axis=1)
        #df_haz['frac_destroyed'] = df_haz.eval(loss_measure_to_use+'/GDP').clip(upper=0.99)
        #
        return df_haz[['fa']], df_haz[['fa']]

    else: return None,None

def get_poverty_line(myC,by_district=True,sec=None):
    pov_line = 0.0
    
    if myC == 'PH': pov_line = 22302.6775#21240.2924
    if myC == 'MW': pov_line = 137427.98
    if myC == 'FJ':
        # 55.12 per week for an urban adult
        # 49.50 per week for a rural adult
        # children under age 14 are counted as half an adult
        if (sec.lower() == 'urban' or sec.lower() == 'u'):
            pov_line = 55.12*52.
        elif (sec.lower() == 'rural' or sec.lower() == 'r'):
            povline = 49.50*52.
        else: 
            print('Pov line is variable for urb/rur Fijians! Need to specify which you\'re looking for!')
            pov_line = 0.0    

    if myC == 'SL':
        pov_line = (pd.read_excel('../inputs/SL/poverty_def_by_district.xlsx').T['2017 Aug Rs.']*12).to_frame()
        if not by_district:
            pov_line = float(pd.read_excel('../inputs/SL/poverty_def_by_district.xlsx').T.loc['National','2017 Aug Rs.']*12.)

        # apply PPP to estimate 2016 value...
        pov_line *= 11445.5/11669.1

    if myC == 'RO': pov_line = 40*1.645*365
    if myC == 'BO': pov_line = 365*1.90*3.43

    return pov_line

def get_subsistence_line(myC):

    if myC == 'PH': return 14832.0962*(22302.6775/21240.2924)
    if myC == 'MW': return 85260.164
    if myC == 'RO': return 40*(1.25/1.90)*1.645*365
    if myC == 'SL':

        pov_line = float(pd.read_excel('../inputs/SL/poverty_def_by_district.xlsx').T.loc['National','2017 Aug Rs.']*12.)
        # apply PPP to estimate 2016 value...
        pov_line *= 11445.5/11669.1
        # scale from $1.90/day to $1.25/day
        return (1.09/1.25)*pov_line
    if myC == 'BO': return 365*1.25*3.43

    else:
        print('No subsistence info. Returning False') 
        return False

def get_to_USD(myC):

    if myC == 'PH': return 50.70
    if myC == 'FJ': return 2.01
    if myC == 'SL': return 153.76
    if myC == 'MW': return 720.0
    if myC == 'RO': return 4.0
    assert(False)

def get_pop_scale_fac(myC):
    #if myC == 'PH' or myC == 'FJ' or myC == 'MW' or myC == 'SL':  
    return [1.E3,' (,000)']

def get_avg_prod(myC):
    
    if myC == 'PH': return 0.273657188280276
    if myC == 'FJ': return 0.336139019412
    if myC == 'SL': return 0.337960802589002
    if myC == 'MW': return 0.253076569219416
    if myC == 'RO': return (277174.8438/1035207.75)
    assert(False)

def get_demonym(myC):
    
    if myC == 'PH': return 'Filipinos'
    if myC == 'FJ': return 'Fijians'
    if myC == 'SL': return 'Sri Lankans'
    if myC == 'MW': return 'Malawians'
    if myC == 'RO': return 'Romanians'
    if myC == 'BO': return 'Bolivians'
    return 'individuals'

def scale_hh_income_to_match_GDP(df_o,new_total,flat=False):

    df = df_o.copy()
    tot_inc = df.loc[:,['hhinc','hhwgt']].prod(axis=1).sum()

    if flat == True:
        print('\nScaling up income and the poverty line by',round((new_total/tot_inc),6),'!!\n')

        df['hhinc']*=(new_total/tot_inc)
        df['pov_line']*=(new_total/tot_inc)
        return df['hhinc'], df['pov_line']
    
    #[['hhinc','hhwgt','AE','Sector']]
    tot_inc_urb = df.loc[df.Sector=='Urban',['hhinc','hhwgt']].prod(axis=1).sum()
    tot_inc_rur = df.loc[df.Sector=='Rural',['hhinc','hhwgt']].prod(axis=1).sum()

    nAE = df[['AE','hhwgt']].prod(axis=1).sum()
    nAE_urb = df.loc[df.Sector=='Urban',['AE','hhwgt']].prod(axis=1).sum()
    nAE_rur = df.loc[df.Sector=='Rural',['AE','hhwgt']].prod(axis=1).sum()
    
    f_inc_urb = tot_inc_urb/tot_inc
    f_inc_rur = tot_inc_rur/tot_inc

    new_inc_urb = f_inc_urb*new_total
    new_inc_rur = f_inc_rur*new_total

    print('New inc urb',new_inc_urb)
    print('New inc rur',new_inc_rur)
    
    #ep_urb = 0.295#(np.log(new_inc_urb/nAE_urb)-np.log(tot_inc_urb/nAE_urb))/(np.log(tot_inc_urb/nAE_urb)-np.log(55.12*52))-1
    #ep_rur = 0.295#(np.log(new_inc_rur/nAE_rur)-np.log(tot_inc_rur/nAE_rur))/(np.log(tot_inc_rur/nAE_rur)-np.log(49.50*52))-1  

    ep_urb = 0.30
    ep_rur = 0.30

    #print(tot_inc)
    #print(ep_urb)
    #print(ep_rur)

    df['AEinc'] = df['hhinc']/df['AE']
    df['new_AEinc'] = df['AEinc']
    df.loc[(df.Sector=='Urban')&(df.AEinc>1.5*df.pov_line),'new_AEinc'] = (55.12*52)*(df.loc[df.Sector=='Urban','AEinc']/(55.12*52))**(1+ep_urb)
    df.loc[(df.Sector=='Rural')&(df.AEinc>1.5*df.pov_line),'new_AEinc'] = (49.50*52)*(df.loc[df.Sector=='Rural','AEinc']/(49.50*52))**(1+ep_rur)

    df['ratio'] = df['new_AEinc']/df['AEinc']
    
    #print(df[['AEinc','new_AEinc','ratio']])

    print('Old sum:',df[['hhwgt','AE','AEinc']].prod(axis=1).sum())
    print('New sum:',df[['hhwgt','AE','new_AEinc']].prod(axis=1).sum())

    df['new_hhinc'] = df[['AE','new_AEinc']].prod(axis=1)

    ci_heights, ci_bins = np.histogram(df['AEinc'].clip(upper=20000), bins=50, weights=df[['hhwgt','hhsize']].prod(axis=1))
    cf_heights, cf_bins = np.histogram(df['new_AEinc'].clip(upper=20000), bins=50, weights=df[['hhwgt','hhsize']].prod(axis=1))

    ax = plt.gca()
    q_colors = [sns_pal[0],sns_pal[1],sns_pal[2],sns_pal[3],sns_pal[5]]
    ax.bar(ci_bins[:-1], ci_heights, width=(ci_bins[1]-ci_bins[0]), label='Initial', facecolor=q_colors[0],alpha=0.4)
    ax.bar(ci_bins[:-1], cf_heights, width=(ci_bins[1]-ci_bins[0]), label='Post-shift', facecolor=q_colors[1],alpha=0.4)

    print('in pov before shift:',df.loc[(df.AEinc <= df.pov_line),['hhwgt','hhsize']].prod(axis=1).sum())
    print('in pov after shift:',df.loc[(df.new_AEinc <= df.pov_line),['hhwgt','hhsize']].prod(axis=1).sum())    

    fig = ax.get_figure()
    plt.xlabel(r'Income [FJD yr$^{-1}$]')
    plt.ylabel('Population')
    plt.legend(loc='best')
    fig.savefig('../output_plots/FJ/income_shift.pdf',format='pdf')#+'.pdf',format='pdf')

    return df['new_hhinc'], df['pov_line']

def get_all_hazards(myC,df):
    temp = (df.reset_index().set_index(['hazard'])).copy()
    temp = temp[~temp.index.duplicated(keep='first')]
    return [i for i in temp.index.values if i != 'x']
        
def get_all_rps(myC,df):
    temp = (df.reset_index().set_index(['rp'])).copy()
    temp = temp[~temp.index.duplicated(keep='first')]
    return [int(i) for i in temp.index.values]
        
def int_w_commas(in_int):
    in_str = str(in_int)
    in_list = list(in_str)
    out_str = ''

    if in_int < 1E3:  return in_str
    if in_int < 1E6:  return in_str[:-3]+','+in_str[-3:] 
    if in_int < 1E9:  return in_str[:-6]+','+in_str[-6:-3]+','+in_str[-3:] 
    if in_int < 1E12: return in_str[:-9]+','+in_str[-9:-6]+','+in_str[-6:-3]+','+in_str[-3:] 
    
