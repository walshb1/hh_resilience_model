import os
import pandas as pd
import matplotlib.pyplot as plt

from libraries.plot_hist import *
from libraries.lib_drought import *
from libraries.lib_gather_data import *

import seaborn as sns
sns.set_style('whitegrid')
sns_pal = sns.color_palette('Set1', n_colors=8, desat=.5)

global model
model = os.getcwd()

# People/hh will be affected or not_affected, and helped or not_helped
affected_cats = pd.Index(['a', 'na'], name='affected_cat') # categories for social protection
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
    
    if myC == 'PH': return 'region'#'province'
    elif myC == 'FJ': return 'Division'#'tikina'
    elif myC == 'SL': return 'district'
    elif myC == 'MW': return 'district'
    else: return None

def get_currency(myC):
    
    if myC == 'PH': return ['b. PhP',1.E9,1./50.]
    elif myC == 'FJ': return ['k. F\$',1.E3,1./2.]
    elif myC == 'SL': return ['b. LKR',1.E9,1./153.]
    elif myC == 'MW': return ['MWK',1.E9,1./724.64]
    else: return ['XXX',1.E0]

def get_places(myC,economy):
    # This df should have just province code/name and population

    if myC == 'PH':
        df = pd.read_excel(inputs+'population_2015.xlsx',sheetname='population').set_index('province')
        df['psa_pop']      = df['population']    # Provincial population        
        df.drop(['population'],axis=1,inplace=True)
        return df

    if myC == 'FJ':
        df = pd.read_excel(inputs+'HIES 2013-14 Housing Data.xlsx',sheetname='Sheet1').set_index('Division').dropna(how='all')[['HHsize','Weight']].prod(axis=1).sum(level='Division').to_frame()
        df.columns = ['population']
        return df

    if myC == 'SL':
        df = pd.read_csv(inputs+'/finalhhframe.csv').set_index('district').dropna(how='all')[['weight','np']].prod(axis=1).sum(level='district').to_frame()
        df.columns = ['population']
        return df

    if myC == 'MW':

        df_agg = pd.read_stata(inputs+'consumption_aggregates_poverty_ihs4.dta').set_index(['district','case_id']).dropna(how='all')
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

    elif myC == 'SL':
        p_code = pd.read_excel(inputs+'Admin_level_3__Districts.xls')[['DISTRICT_C','DISTRICT_N']].set_index('DISTRICT_C').squeeze()
        p_code.index.name = 'district'

    elif myC == 'MW':
        p_code = pd.read_csv(inputs+'MW_code_region_district.csv',usecols=['code','district']).set_index('code').dropna(how='all')
        p_code.index = p_code.index.astype(int)

        r_code = pd.read_csv(inputs+'MW_code_region_district.csv',usecols=['code','region']).set_index('code').dropna(how='all')
        r_code.index = r_code.index.astype(int)   

    return p_code,r_code

def load_survey_data(myC,inc_sf=None):
    
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
    
    if myC == 'MW':

        df_agg = pd.read_stata(inputs+'consumption_aggregates_poverty_ihs4.dta').set_index(['case_id']).dropna(how='all')
        df = df_agg[['district','hh_wgt','rexpagg','pline_2016','upline_2016','poor','upoor','hhsize','adulteq']].copy()
        
        df = df.reset_index().rename(columns={'hh_wgt':'hhwgt',
                                              'rexpagg':'hhinc',
                                              'pline_2016':'pov_line',
                                              'upline_2016':'sub_line',
                                              'poor':'ispoor',
                                              'upoor':'issub',
                                              'case_id':'hhid',
                                              'adulteq':'hhsize_ae'}).set_index(['hhid']).sort_index()
        df['pcwgt'] = df[['hhwgt','hhsize']].prod(axis=1)
        df['pcinc'] = df['hhinc']/df['hhsize']

        df['pcinc_ae'] = df['hhinc']/df['hhsize_ae']
        df['pcwgt_ae'] = df[['hhwgt','hhsize_ae']].prod(axis=1)        

        # Still need hhsoc & pcsoc
        dfR = pd.read_stata(inputs+'MWI_2016_IHS-IV_v02_M_Stata/HH_MOD_R.dta').rename(columns={'case_id':'hhid'}).set_index('hhid')
        df['hhsoc'] = dfR[['hh_r02a','hh_r02b']].sum(axis=1).sum(level='hhid')
        df['pcsoc'] = df['hhsoc']/df['hhsize']

        df['hhsoc_kg_maize'] = dfR['hh_r02c'].sum(level='hhid')
        # ^ not clear what to do with this

        #need dwelling info
        dfF = pd.read_stata(inputs+'MWI_2016_IHS-IV_v02_M_Stata/HH_MOD_F.dta').rename(columns={'case_id':'hhid'}).set_index('hhid')
        #df['general_construction'] = dfF['hh_f06'].copy()
        df['wall'] = dfF['hh_f07'].copy()
        df['roof'] = dfF['hh_f08'].copy()
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

        print(dfE.head())
        assert(False)

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
        
        return df

    elif myC == 'PH':
        df = pd.read_csv(inputs+'fies2015.csv',usecols=['w_regn','w_prov','w_mun','w_bgy','w_ea','w_shsn','w_hcn',
                                                        'walls','roof',
                                                        'totex','cash_abroad','cash_domestic','regft',
                                                        'hhwgt','fsize','poorhh','totdis','tothrec','pcinc_s','pcinc_ppp11','pcwgt',
                                                        'radio_qty','tv_qty','cellphone_qty','pc_qty',
                                                        'savings','invest'])
        df = df.rename(columns={'tothrec':'hhsoc','poorhh':'ispoor'})
        df['hhsize']     = df['pcwgt']/df['hhwgt']
        df['hhsize_ae']  = df['pcwgt']/df['hhwgt']

        df['pcwgt_ae']   = df['pcwgt']

        # These lines use income as income
        df = df.rename(columns={'pcinc_s':'pcinc'})
        df['hhinc'] = df[['pcinc','hhsize']].prod(axis=1)

        # These lines use disbursements as proxy for income
        #df = df.rename(columns={'totdis':'hhinc'}) 
        #df['pcinc'] = df['hhinc']/df['hhsize']

        df['pcinc_ae']   = df['pcinc']

        df['pcsoc']  = df['hhsoc']/df['hhsize']

        df['tot_savings'] = df[['savings','invest']].sum(axis=1,skipna=False)
        df['savings'] = df['savings'].fillna(-1)
        df['invest'] = df['invest'].fillna(-1)
        
        df['axfin']  = 0
        df.loc[(df.savings>0)|(df.invest>0),'axfin'] = 1

        df['est_sav'] = df[['axfin','pcinc']].prod(axis=1)/2.

        df['has_ew'] = df[['radio_qty','tv_qty','cellphone_qty','pc_qty']].sum(axis=1).clip(upper=1)
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
    
        return df.drop(['savings','invest'],axis=1)

    elif myC == 'FJ':
        df = pd.read_excel(inputs+'HIES 2013-14 Income Data.xlsx',usecols=['HHID','Division','Nchildren','Nadult','AE','HHsize',
                                                                           'Sector','Weight','TOTALTRANSFER','TotalIncome','New Total',
                                                                           'CareandProtectionProgrampaymentfromSocialWelfare',
                                                                           'FamilyAssistanceProgrampaymentfromSocialWelfare',
                                                                           'FijiNationalProvidentFundPension',
                                                                           'FNPFWithdrawalsEducationHousingInvestmentsetc',
                                                                           'SocialPensionScheme',
                                                                           'TotalBusiness','TotalPropertyIncome'
                                                                           ]).set_index('HHID')
        df = df.rename(columns={'HHID':'hhid','TotalIncome':'hhinc','HHsize':'hhsize','Weight':'hhwgt','TOTALTRANSFER':'hhsoc',
                                'CareandProtectionProgrampaymentfromSocialWelfare':'SP_CPP',
                                'FamilyAssistanceProgrampaymentfromSocialWelfare':'SP_FAP',
                                'FijiNationalProvidentFundPension':'SP_FNPF',
                                'FNPFWithdrawalsEducationHousingInvestmentsetc':'SP_FNPF2',
                                'SocialPensionScheme':'SP_SPS'})

        df['pov_line'] = 0.
        df.loc[df.Sector=='Urban','pov_line'] = get_poverty_line(myC,'Urban')
        df.loc[df.Sector=='Rural','pov_line'] = get_poverty_line(myC,'Rural')

        if inc_sf != None: df['hhinc'], df['pov_line'] = scale_hh_income_to_match_GDP(df[['hhinc','hhwgt','hhsize','AE','Sector','pov_line']],inc_sf,flat=True)

        df['hh_pov_line'] = df[['pov_line','AE']].prod(axis=1)

        df['hhsize_ae'] = df['AE'] # This is 'Adult Equivalents'
        # ^ Equivalent to 0.5*df['Nchildren']+df['Nadult']

        df['pcwgt']    = df[['hhsize','hhwgt']].prod(axis=1)
        df['pcwgt_ae'] = df[['AE','hhwgt']].prod(axis=1)

        df['pcinc']    = df['hhinc']/df['hhsize']
        df['pcinc_ae'] = df['hhinc']/df['hhsize_ae']

        df['pcsoc']    = df['hhsoc']/df['hhsize']
            
        df_housing = pd.read_excel(inputs+'HIES 2013-14 Housing Data.xlsx',sheetname='Sheet1').set_index('HHID').dropna(how='all')[['Constructionofouterwalls',
                                                                                                                                    'Conditionofouterwalls']]

        df_dems = pd.read_excel(inputs+'HIES 2013-14 Demographic Data.xlsx',sheetname='Sheet1').set_index('HHID').dropna(how='all')[['Poor','Age']]
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
        df.sort_values(by='pcinc_ae',ascending=True,inplace=True)
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
        df.sort_values(by='pcinc_ae',ascending=True,inplace=True)

        # BELOW: these lines assume we're counting AE 
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
        return df

    elif myC == 'SL':
        
        #df = pd.read_csv(inputs+'finalhhframe.csv').set_index('hhid')
        df = pd.read_csv(inputs+'hhdata_samurdhi.csv',usecols=['district','hhid','poor','pov_line','rpccons',
                                                               'weight','hhsize','income_other_income_pc',
                                                               'income_local_pc','walls','floor','roof']).set_index('hhid')

        df['has_ew'] = pd.read_csv(inputs+'hhdata_samurdhi.csv',usecols=['hhid','asset_telephone_mobile',
                                                                         'asset_telephone','asset_radio',
                                                                         'asset_tv','asset_computers'],index_col='hhid').sum(axis=1).clip(lower=0,upper=1)

        pmt = pd.read_csv(inputs+'pmt_2012_hh_model1_score.csv').set_index('hhid')
        df[['pmt','pmt_rpccons']] = pmt[['score','rpccons']]

        df = df.rename(columns={'rpccons':'pcinc',
                                'pmt_rpccons':'pcinc_pmt',
                                'weight':'hhwgt',
                                'income_other_income_pc':'other_inc',
                                'income_local_pc':'income_local',
                                'poor':'ispoor'})

        df[['pcinc','pcinc_pmt','other_inc','income_local']] *= 12.
        # pcinc is monthly
        
        hies_gdp_lkr = round(df[['pcinc','hhsize','hhwgt']].prod(axis=1).sum()/1E9,3)
        
        lkr_to_usd = 1./156.01
        print('hies GPD = ',hies_gdp_lkr,'billion LKR and',round(hies_gdp_lkr*lkr_to_usd,3),'billion USD')

        #test_df = pd.read_csv(inputs+'finalhhframe.csv').set_index('hhid')
        #print(test_df.head())
        #print(round(test_df[['exp','weight','np']].prod(axis=1).sum()/1E9,3)*lkr_to_usd)

        df['pcinc_ae'] = df['pcinc'].copy()
        df['pcwgt'] = df[['hhwgt','hhsize']].prod(axis=1)
        print('avg income per cap & month = ',int(df[['pcinc','pcwgt']].prod(axis=1).sum()/df['pcwgt'].sum()/12.))
        print('avg income per hh & month = ',int(df[['pcinc','hhsize','hhwgt']].prod(axis=1).sum()/df['hhwgt'].sum()/12.))

        df['hhsize_ae'] = df['hhsize']
        df['pcwgt_ae'] = df['pcwgt']

        df['pcsoc'] = df[['other_inc','income_local']].sum(axis=1)
        
        df = df.reset_index().set_index('district')
        
        return df

    else: return None

def get_df2(myC):

    if myC == 'PH':
        df2 = pd.read_excel(inputs+'PSA_compiled.xlsx',skiprows=1)[['province','gdp_pc_pp','pop','shewp','shewr']].set_index('province')
        df2['gdp_pp'] = df2['gdp_pc_pp']*df2['pop']
        return df2

    else: return None

def get_vul_curve(myC,struct):
    df = None

    if myC == 'PH':
        df = pd.read_excel(inputs+'vulnerability_curves_FIES.xlsx',sheetname=struct)[['desc','v']]

    elif myC == 'FJ':
        df = pd.read_excel(inputs+'vulnerability_curves_Fiji.xlsx',sheetname=struct)[['desc','v']]

    elif myC == 'SL':
        df = pd.read_excel(inputs+'vulnerability_curves.xlsx',sheetname=struct)[['key','v']]
        df = df.rename(columns={'key':'desc'})

    elif myC == 'MW':
        df = pd.read_excel(inputs+'vulnerability_curves_MW.xlsx',sheetname=struct)[['desc','v']]

    return df
    
def get_infra_stocks_data(myC):
    if myC == 'FJ':
        infra_stocks = pd.read_csv(inputs+'infra_stocks.csv',index_col='sector')
        return infra_stocks
    else:return None
    
def get_wb_or_penn_data(myC):
    #iso2 to iso3 table
    names_to_iso2 = pd.read_csv(inputs+'names_to_iso.csv', usecols=['iso2','country']).drop_duplicates().set_index('country').squeeze()
    K = pd.read_csv(inputs+'avg_prod_k_with_gar_for_sids.csv',index_col='Unnamed: 0')
    wb = pd.read_csv(inputs+'wb_data.csv',index_col='country')
    wb['Ktot'] = wb.gdp_pc_pp*wb['pop']/K.avg_prod_k
    wb['GDP'] = wb.gdp_pc_pp*wb['pop']
    wb['avg_prod_k'] = K.avg_prod_k
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
        df_haz = pd.read_excel(inputs+'/GAR_PML_curve_MW.xlsx',sheetname='PML(mUSD)')[['rp','Earthquake','Flood','Drought']].set_index('rp')
        tot_exposure = float(pd.read_excel(inputs+'/GAR_PML_curve_MW.xlsx',sheetname='total_exposed_val').loc['Malawi','Total Exposure'].squeeze())

        ag_exposure_gross = float(pd.read_excel(inputs+'/GAR_PML_curve_MW.xlsx',sheetname='total_exposed_val').loc['Malawi','Gross value, maize'].squeeze())
        ag_exposure_net = float(pd.read_excel(inputs+'/GAR_PML_curve_MW.xlsx',sheetname='total_exposed_val').loc['Malawi','Net value, maize'].squeeze())

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
    
        df_geo = df_geo.reset_index().set_index('country')
        df_haz = df_haz.reset_index().set_index('country')
        
        df_haz = pd.merge(df_geo.reset_index(),df_haz.reset_index(),on=['country'],how='outer').set_index(['district','hazard','rp']).sort_index()
        df_haz = df_haz.drop(['country','population','PML'],axis=1)

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
        df = pd.read_excel(inputs+'hazards_data.xlsx',sheetname='hazard').dropna(how='any')
        df.hazard = df.hazard.replace({'flood':'PF'})
        df = df.set_index(['district','hazard','rp'])
        return df,df

    else: return None,None

def get_poverty_line(myC,sec=None):
    
    if myC == 'PH': pov_line = 22302.6775#21240.2924

    elif myC == 'FJ':
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

    elif myC == 'SL': pov_line = float(pd.read_csv('../inputs/SL/hhdata_samurdhi.csv',index_col='hhid')['pov_line'].mean())*12.    
    elif myC == 'MW': pov_line = 137427.98
    else: pov_line = 0.0
    
    return pov_line

def get_subsistence_line(myC):
    
    if myC == 'PH': sub_line = 14832.0962*(22302.6775/21240.2924)
    elif myC == 'SL':
        sub_line = float(pd.read_csv('../inputs/SL/hhdata_samurdhi.csv',index_col='hhid')['pline_125'].mean())*12.
        print('--> subsistence line:',sub_line,'\n')
    elif myC == 'MW': sub_line = 85260.164
    else: sub_line = 0; print('No subsistence info. Returning 0')

    return sub_line

def get_to_USD(myC):

    if myC == 'PH': return 50.70
    elif myC == 'FJ': return 2.01
    elif myC == 'SL': return 153.76
    elif myC == 'MW': return 720.0
    else: return 0.

def get_pop_scale_fac(myC):
    
    if myC == 'PH': return [1.E3,' [Thousands]']
    elif myC == 'FJ': return [1.E3,' [Thousands]']
    elif myC == 'MW': return [1.E3,' [Thousands]']
    else: return [1,'']

def get_avg_prod(myC):
    
    if myC == 'PH': return 0.3379608025890020#0.273657188280276
    elif myC == 'FJ': return 0.336139019412
    elif myC == 'SL': return 0.337960802589002
    elif myC == 'MW': return 0.253076569219416
    assert(False)

def get_demonym(myC):
    
    if myC == 'PH': return 'Filipinos'
    elif myC == 'FJ': return 'Fijians'
    elif myC == 'SL': return 'Sri Lankans'
    elif myC == 'MW': return 'Malawians'

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
    return [i for i in temp.index.values]
        
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
    
