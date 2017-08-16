import os
import pandas as pd
from lib_gather_data import *
import matplotlib.pyplot as plt
from plot_hist import *

import seaborn as sns
sns.set_style('darkgrid')
sns_pal = sns.color_palette('Set1', n_colors=8, desat=.5)

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
        df = pd.read_csv(inputs+'/finalhhframe.csv').set_index('district').dropna(how='all')[['weight','np']].prod(axis=1).sum(level='district').to_frame()
        df.columns = ['population']
        return df

    else: return None

def get_places_dict(myC):

    if myC == 'PH': 
        return pd.read_excel(inputs+'FIES_provinces.xlsx')[['province_code','province_AIR']].set_index('province_code').squeeze() 

    if myC == 'FJ':
        return pd.read_excel(inputs+'Fiji_provinces.xlsx')[['code','name']].set_index('code').squeeze() 

    elif myC == 'SL':
        return pd.read_excel(inputs+'Admin_level_3__Districts.xls')[['DISTRICT_C','DISTRICT_N']].set_index('DISTRICT_C').squeeze()

    else: return None

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
                                                                           'Sector','Weight','TOTALTRANSFER','TotalIncome','New Total',
                                                                           'CareandProtectionProgrampaymentfromSocialWelfare',
                                                                           'FamilyAssistanceProgrampaymentfromSocialWelfare',
                                                                           'SocialPensionScheme']).set_index('HHID')
        df = df.rename(columns={'HHID':'hhid','TotalIncome':'hhinc','HHsize':'hhsize','Weight':'hhwgt','TOTALTRANSFER':'hhsoc',
                                'CareandProtectionProgrampaymentfromSocialWelfare':'SP_CPP',
                                'FamilyAssistanceProgrampaymentfromSocialWelfare':'SP_FAP',
                                'SocialPensionScheme':'SP_SPS'})

        df['pov_line'] = 0.
        df.loc[df.Sector=='Urban','pov_line'] = 55.12*52*df.loc[df.Sector=='Urban','AE']
        df.loc[df.Sector=='Rural','pov_line'] = 49.50*52*df.loc[df.Sector=='Rural','AE']        

        if inc_sf != None: df['hhinc'] = scale_hh_income_to_match_GDP(df[['hhinc','hhwgt','hhsize','AE','Sector','pov_line']],inc_sf)

        #df_cons = pd.read_excel(inputs+'HIES 2013-14 Consumption Data.xlsx',usecols=['HHID','Weights','Grand Total'],sheetname='by_Individual items').set_index('HHID').drop('HHID').fillna(0)
        #print(df_cons[['Weights','Grand Total']].prod(axis=1).sum())
        #assert(False)

        #df['hhsize_ae'] = df['Nadult'] # assuming this is 'Adult Equivalents'
        df['hhsize_ae'] = df['AE'] # assuming this is 'Adult Equivalents'
        # Should be equivalent to 0.5*df['Nchildren']+df['Nadult']

        df['pcwgt'] = df[['hhsize','hhwgt']].prod(axis=1)
        df['pcwgt_ae'] = df[['AE','hhwgt']].prod(axis=1)
        df['pcinc'] = df['hhinc']/df['hhsize']
        df['pcinc_ae'] = df['hhinc']/df['hhsize_ae']
        df['pcsoc'] = df['hhsoc']/df['hhsize']

        #print('Total income:',df[['pcinc','pcwgt']].prod(axis=1).sum())
        #print('Total income (new):',df[['New Total','hhwgt']].prod(axis=1).sum())
        #print(df[['hhsize','hhwgt']].prod(axis=1).sum()/df['hhwgt'].sum())
        #print(df[['hhsize_ae','hhwgt']].prod(axis=1).sum()/df['hhwgt'].sum())
            
        df_housing = pd.read_excel(inputs+'HIES 2013-14 Housing Data.xlsx',sheetname='Sheet1').set_index('HHID').dropna(how='all')[['Constructionofouterwalls',
                                                                                                                                    'Conditionofouterwalls']]
        
        df_poor = pd.read_excel(inputs+'HIES 2013-14 Demographic Data.xlsx',sheetname='Sheet1').set_index('HHID').dropna(how='all')['Poor']
        df_poor = df_poor[~df_poor.index.duplicated(keep='first')]

        df = pd.concat([df,df_housing,df_poor],axis=1).reset_index().set_index('Division')

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
        # SP_SPS = SocialProtectionScheme
        df.loc[df.SP_SPS != 0,'SP_SPS'] = True
        df.loc[df.SP_SPS == 0,'SP_SPS'] = False

        return df

    elif myC == 'SL':
        
        df = pd.read_csv(inputs+'finalhhframe.csv').set_index('hhid')
        pmt = pd.read_csv(inputs+'pmt_2012_hh_model1_score.csv').set_index('hhid')
        df2 = pd.read_csv(inputs+'hhdata_samurdhi.csv').set_index('hhid')

        df[['score','rpccons']] = pmt[['score','rpccons']]

        df['ispoor'] = df2['poor']
        df['pov_line'] = df2['pov_line']        

        df = df.rename(columns={'rpccons':'pcinc','weight':'hhwgt','np':'hhsize'})

        df['pcinc'] *= 12.

        df['pcinc_ae'] = df['pcinc']
        df['pcwgt'] = df[['hhwgt','hhsize']].prod(axis=1)

        df['hhsize_ae'] = df['hhsize']
        df['pcwgt_ae'] = df['pcwgt']

        df['pcsoc'] = df[['other_inc','income_local']].sum(axis=1)
        
        df = df.reset_index().set_index('district')
        
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
        df = pd.read_excel(inputs+'vulnerability_curves_FIES.xlsx',sheetname=struct)[['desc','v']]
        return df

    if myC == 'FJ':
        df = pd.read_excel(inputs+'vulnerability_curves_Fiji.xlsx',sheetname=struct)[['desc','v']]
        return df

    if myC == 'SL':
        df = pd.read_excel(inputs+'vulnerability_curves.xlsx',sheetname=struct)[['key','v']]
        df = df.rename(columns={'key':'desc'})
        return df        
        
    else: return None
    
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

    print(get_infra_stocks_data(myC))

    infra_stocks = get_infra_stocks_data(myC).loc[['transport','energy','water'],:]
    infra_stocks['infra_share'] = infra_stocks.value_k/infra_stocks.value_k.sum()
        
    hazard_ratios_infra = broadcast_simple(df_haz[['frac_inf','frac_destroyed_inf']],infra_stocks.index)
    hazard_ratios_infra = pd.merge(hazard_ratios_infra.reset_index(),infra_stocks.infra_share.reset_index(),on='sector',how='outer').set_index(['Division','hazard','rp','sector'])
    hazard_ratios_infra['share'] = hazard_ratios_infra['infra_share']*hazard_ratios_infra['frac_inf']
        
    transport_losses = pd.read_csv(inputs+"frac_destroyed_transport.csv").rename(columns={"ti_name":"Tikina"})
    transport_losses['Division'] = (transport_losses['tid']/100).astype('int')
    prov_code = get_places_dict(myC)
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

def get_hazard_df(myC,economy):

    if myC == 'PH': 
        df = get_AIR_data(inputs+'/Risk_Profile_Master_With_Population.xlsx','Loss_Results','Private','Agg').reset_index()
        df.columns = [economy,'hazard','rp','value_destroyed']
        return df
    
    elif myC == 'FJ':

        df_all_ah = pd.read_csv(inputs+'map_tikinas.csv').set_index('Tikina').drop('Country_ID',axis=1)
        df_all_ah['hazard'] = 'All Hazards'
        df_all_ah['asset_class'] = 'all'
        df_all_ah['asset_subclass'] = 'all'

        # Hazard = tropical cyclone
        #df_bld_tc = pd.read_csv(inputs+'fiji_tc_buildings_tikina.csv').set_index('Tikina').drop('Country_ID',axis=1)
        #df_bld_tc['hazard'] = 'TC'
        #df_bld_tc['asset_class'] = 'bld'
        #df_bld_tc['asset_subclass'] = 'all'
        #df_bld_tc.reset_index().set_index(['Tikina','Tikina_ID','hazard','asset_class','asset_subclass','Exp_Value'])  

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
        prov_code = get_places_dict(myC)
        df = df.reset_index().set_index([df.Division.replace(prov_code)]).drop(['index','Division','Tikina_ID','asset_subclass'],axis=1) #replace district code with its name
        
        df = df.reset_index().set_index(['Division','Tikina','hazard','rp','asset_class'])
        df_sum = ((df['value_destroyed'].sum(level=['Division','hazard','rp']))/(df['Exp_Value'].sum(level=['Division','hazard','rp']))).to_frame(name='frac_destroyed')
        # ^ what if we run with fa from buildings?

        #df = df.reset_index().set_index(['Division','Tikina','hazard','rp'])
        #df_sum = ((df.loc[(df.asset_class == 'bld_res')|(df.asset_class == 'agr'),'value_destroyed'].sum(level=['Division','hazard','rp']))/(df.loc[(df.asset_class == 'bld_res')|(df.asset_class == 'agr'),'Exp_Value'].sum(level=['Division','hazard','rp']))).to_frame(name='fa')
        #df = df.reset_index().set_index(['Division','Tikina','hazard','rp','asset_class'])
        
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
        
        #df_sum['bldg_stock'] = df_sum[['Exp_Value','frac_bld_res']].prod(axis=1)+df_sum[['Exp_Value','frac_bld_oth']].prod(axis=1)
        #print(df_sum.reset_index().set_index(['rp']).ix[1,'bldg_stock'].sum())
        df_sum['Exp_Value'] *= (1.0/0.48) # AIR-PCRAFI in USD(2009?) --> switch to FJD

        #print(df_sum)
        #df_sum.to_csv('~/Desktop/df_hazard.csv')
        #assert(False)

        return df_sum

    elif myC == 'SL':
        df = pd.read_excel(inputs+'hazards_data.xlsx',sheetname='hazard').dropna(how='any').set_index(['district','hazard','rp'])
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

    if myC == 'PH': return 50.70
    elif myC == 'FJ': return 2.01
    elif myC == 'SL': return 153.76
    else: return 0.

def get_scale_fac(myC):
    
    if myC == 'PH': return [1.E6,' [Millions]']
    elif myC == 'FJ': return [1.E3,' [Thousands]']
    else: return [1,'']

def get_avg_prod(myC):
    
    if myC == 'PH': return 0.337960802589002
    elif myC == 'FJ': return 0.336139019412
    elif myC == 'SL': return 0.337960802589002

def get_demonym(myC):
    
    if myC == 'PH': return 'Filipinos'
    elif myC == 'FJ': return 'Fijians'
    elif myC == 'SL': return 'Sri Lankans'

def scale_hh_income_to_match_GDP(df,new_total):

    df = df.copy()
    
    #[['hhinc','hhwgt','AE','Sector']]

    tot_inc = df[['hhinc','hhwgt']].prod(axis=1).sum()
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
    
    ep_urb = 0.295#(np.log(new_inc_urb/nAE_urb)-np.log(tot_inc_urb/nAE_urb))/(np.log(tot_inc_urb/nAE_urb)-np.log(55.12*52))-1
    ep_rur = 0.295#(np.log(new_inc_rur/nAE_rur)-np.log(tot_inc_rur/nAE_rur))/(np.log(tot_inc_rur/nAE_rur)-np.log(49.50*52))-1  

    #ep_urb = 0.35
    #ep_rur = 0.

    #print(tot_inc)
    #print(ep_urb)
    #print(ep_rur)

    df['AEinc'] = df['hhinc']/df['AE']
    df['new_AEinc'] = 0.    
    df.loc[df.Sector=='Urban','new_AEinc'] = (55.12*52)*(df.loc[df.Sector=='Urban','AEinc']/(55.12*52))**(1+ep_urb)
    df.loc[df.Sector=='Rural','new_AEinc'] = (49.50*52)*(df.loc[df.Sector=='Rural','AEinc']/(49.50*52))**(1+ep_rur)

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

    print('in pov before shift:',df.loc[(df.hhinc <= df.pov_line),['hhwgt','hhsize']].prod(axis=1).sum())
    print('in pov after shift:',df.loc[(df.new_hhinc <= df.pov_line),['hhwgt','hhsize']].prod(axis=1).sum())    

    fig = ax.get_figure()
    plt.xlabel(r'Income [FJD yr$^{-1}$]')
    plt.ylabel('Population')
    plt.legend(loc='best')
    fig.savefig('../output_plots/FJ/income_shift.pdf',format='pdf')#+'.pdf',format='pdf')

    return df['new_hhinc']
