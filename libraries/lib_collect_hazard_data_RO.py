import pandas as pd
from dev.qgis_to_csv import qgis_to_csv
from libraries.lib_country_dir import get_all_rps #get_economic_unit
import numpy as np

def collect_EQ_hazard_data_RO():

    # National-level losses
    df_caploss = pd.read_csv('../inputs/RO/hazard/future_earth_2016/EQ/EQ.2010.country.revised.csv').set_index('WB Country Name')
    df_caploss = df_caploss.loc['Romania'].to_frame().T

    pml_caploss = df_caploss[[_c for _c in df_caploss.columns if 'CAPLOSS' in _c and '.1' not in _c and '.2' not in _c and 'AAL' not in _c]].stack().reset_index()
    pml_caploss['level_1'] = pml_caploss['level_1'].str.replace('CAPLOSS','').astype('int')
    pml_caploss = pml_caploss.rename(columns={'level_1':'rp',0:'PML'})
    pml_caploss = pml_caploss.set_index('rp').drop(['level_0'],axis=1).sort_index()
    
    aep_caploss = df_caploss[[_c for _c in df_caploss.columns if 'CAPLOSS' in _c and '.1' in _c and 'AAL' not in _c]].stack().reset_index()
    aep_caploss['level_1'] = aep_caploss['level_1'].str.replace('CAPLOSS','')
    aep_caploss['level_1'] = aep_caploss['level_1'].str.replace('.1','').astype('int')
    aep_caploss = aep_caploss.rename(columns={'level_1':'rp',0:'AEP'})
    aep_caploss = aep_caploss.set_index('rp').drop(['level_0'],axis=1).sort_index()
    
    aal_caploss = df_caploss[[_c for _c in df_caploss.columns if 'CAPLOSS' in _c and '.2' in _c and 'AAL' not in _c]].stack().reset_index()
    aal_caploss['level_1'] = aal_caploss['level_1'].str.replace('CAPLOSS','')
    aal_caploss['level_1'] = aal_caploss['level_1'].str.replace('.2','').astype('int')
    aal_caploss = aal_caploss.rename(columns={'level_1':'rp',0:'AAL'})
    aal_caploss = aal_caploss.set_index('rp').drop(['level_0'],axis=1).sort_index()
    
    caploss_out = pd.concat([pml_caploss,aep_caploss,aal_caploss],axis=1)
    caploss_out.to_csv('../inputs/RO/hazard/_EQ_caploss_by_country.csv')
    #
    # Above: national
    # Below: provincial
    # 
    df = pd.read_excel('../inputs/RO/hazard/future_earth_2016/EQ/ECA Earthquake Final Spreadsheet 12112014.xlsx',sheet_name='Province',skiprows=1)
    df = df.loc[df['WB Country Name']=='Romania'].rename(columns={'WB Province Name':'province'}).set_index('province')

    for _out in ['CAPLOSS','POP','GDP','DEATHS']:
        print(_out)

        _pml = df[[_c for _c in df.columns if _out in _c and '.1' not in _c and '.2' not in _c and 'AAL' not in _c and 'CATDAT' not in _c]].stack().reset_index(-1)
        _pml['level_1'] = _pml['level_1'].str.replace(_out,'').astype('int')
        _pml = _pml.rename(columns={'level_1':'rp',0:'PML'})
        _pml = _pml.reset_index().set_index(['province','rp']).sort_index()

        _aep = df[[_c for _c in df.columns if _out in _c and '.1' in _c and 'AAL' not in _c and 'CATDAT' not in _c]].stack().reset_index(-1)
        _aep['level_1'] = _aep['level_1'].str.replace(_out,'')
        _aep['level_1'] = _aep['level_1'].str.replace('.1','').astype('int')
        _aep = _aep.rename(columns={'level_1':'rp',0:'AEP'})
        _aep = _aep.reset_index().set_index(['province','rp']).sort_index()

        _aal = df[[_c for _c in df.columns if _out in _c and '.2' in _c and 'AAL' not in _c and 'CATDAT' not in _c]].stack().reset_index(-1)
        _aal['level_1'] = _aal['level_1'].str.replace(_out,'')
        _aal['level_1'] = _aal['level_1'].str.replace('.2','').astype('int')
        _aal = _aal.rename(columns={'level_1':'rp',0:'AAL'})
        _aal = _aal.reset_index().set_index(['province','rp']).sort_index()

        _outdf = pd.concat([_pml,_aep,_aal],axis=1)
        _outdf.to_csv('../inputs/RO/hazard/_EQ_'+_out.lower().replace('pop','popaff').replace('gdp','gdpaff')+'_by_province.csv')
    #####
    return True

def collect_FL_hazard_data_RO():
    
    try: 
        ro_codes = pd.read_csv('../inputs/RO/hazard/Shapefiles/ECAMerge2.csv').set_index('OBJECTID_1')
    except:
        qgis_to_csv('../inputs/RO/hazard/Shapefiles/ECAMerge2.shp','../inputs/RO/hazard/Shapefiles/ECAMerge2.csv')
        ro_codes = pd.read_csv('../inputs/RO/hazard/Shapefiles/ECAMerge2.csv').set_index('OBJECTID_1')

    ro_codes = ro_codes.loc[ro_codes.WB_ADM0_NA=='Romania']
    ro_dict = ro_codes['WB_ADM1_NA'].to_dict()

    ro_risk = pd.read_csv('../inputs/RO/hazard/future_earth_2016/Flood/flood.RP.people.all_BiH_Level1.csv').set_index('OBJECTID_1')

    ro_risk = ro_risk.loc[ro_codes.index.values].reset_index()

    ro_risk['County'] = ro_risk['OBJECTID_1'].replace(ro_dict)
    ro_risk = ro_risk.set_index('County')

    #print(ro_risk.columns)
    ro_risk = ro_risk[[_c for _c in ro_risk.columns if 'present-day' in _c and '2030' not in _c and '2080' not in _c]].stack().reset_index()
    ro_risk = ro_risk.rename(columns={'level_1':'rp',0:'pop_affected'})
    ro_risk['rp'] = ro_risk['rp'].str.replace('data_present-day_2010_RP_','').astype('int')

    ro_risk = ro_risk.set_index('County')

    ro_pop = pd.read_csv('../inputs/RO/hazard/Romania/Romania.province.data.unround.csv')[['Province',
                                                                                           'Data.Population']].rename(columns={'Province':'County','Data.Population':'population'})
    ro_pop = ro_pop.set_index('County').squeeze()
    ro_risk['fa'] = ro_risk['pop_affected']/ro_pop

    ro_risk = ro_risk.reset_index().set_index(['County','rp'])
    
    ro_risk[['pop_affected']].to_csv('../inputs/RO/hazard/_PF_pop_aff_by_county.csv')
    ro_risk[['fa']].to_csv('../inputs/RO/hazard/_PF_fa_by_county.csv')
    return True



############################
# The 2 functions above report losses, by *county*
# I think we can't just sum these to the Regional level
# --> easiest thing to do is:
# 1) simulate losses in each county over 50K years, 
# 2) sum to regional level for each simulated year
# 3) recreate the exceedance curve at the regional level

def random_to_loss(County,pval):

    for _nrp, _rp in enumerate(inv_rps):
        if pval > _rp: 
            try: return int(_df.loc[(County,rps[_nrp-1])])
            except: return 0
            # ^ this is because the RP=0 isn't in the df
        
    return int(_df.loc[(County,rps[-1])])

def RO_county_to_region(_haz='EQ'):
    global _df
    if _haz == 'PF':
        _df = pd.read_csv('../inputs/RO/hazard/_PF_pop_aff_by_county.csv').set_index(['County','rp'])
    if _haz == 'EQ':
        #_df = pd.read_csv('../inputs/RO/hazard/_EQ_popaff_by_county.csv').drop(['AAL','AEP'],axis=1).set_index(['County','rp'])
        _df = pd.read_csv('../inputs/RO/hazard/_EQ_caploss_by_county.csv').drop(['AAL','AEP'],axis=1).set_index(['County','rp'])
    #multihaz = pd.merge(_pf,_eq,on=['County','rp']).set_index(['County','rp'])



    # In any case, the EQ data has more return periods, so we'll generate that exceedance curve at the regional level
    final_exceedance = pd.DataFrame(index=pd.read_csv('../inputs/RO/hazard/_EQ_caploss_by_county.csv').set_index(['County','rp']).index).reset_index()
    final_exceedance['Region'] = final_exceedance['County'].replace(pd.read_csv('../inputs/RO/county_to_region.csv').set_index('County')['Region'].to_dict())
    final_exceedance = final_exceedance.set_index(['Region','rp']).sum(level=['Region','rp']).drop('County',axis=1)
    final_exceedance['loss'] = None



    # array with return periods
    global rps,inv_rps
    rps = get_all_rps('RO',_df)    
    if rps[0] != 1.: rps = np.append([1.],[rps])
    inv_rps = [1/i for i in rps]

    #print(rps)
    final_rps = [2, 5, 10, 20, 50, 100, 200, 250, 500, 1000, 2000, 5000, 10000]

    # create dataframe to store random numbers
    loss_cty = pd.DataFrame(index=_df.sum(level='County').index).reset_index()
    loss_cty['Region'] = loss_cty['County'].replace(pd.read_csv('../inputs/RO/county_to_region.csv').set_index('County')['Region'].to_dict())

    loss_reg = loss_cty.set_index(['County','Region']).sum(level='Region')
    loss_cty = loss_cty.reset_index().set_index('Region').drop('index',axis=1)

    # generate random numbers
    NYEARS = int(100E4) # <-- any multiple of 10K
    for _yn in range(NYEARS):
        loss_cty['_'] = [np.random.uniform(0,1) for i in range(loss_cty.shape[0])]
        loss_cty['y'+str(_yn)] = loss_cty.apply(lambda x:random_to_loss(x.County,x['_']),axis=1)

        if _yn != 0 and (_yn+1)%500 == 0: 
            
            loss_reg = pd.concat([loss_reg,loss_cty.drop('_',axis=1).sum(level='Region')],axis=1)
            loss_cty = loss_cty[['County']]
            print(_yn+1)
    
    for _reg in loss_cty.index.values:
        aReg = loss_reg.loc[_reg].sort_values(ascending=False).reset_index()

        for _frp in final_rps:
            final_exceedance.loc[(_reg,_frp),'loss'] = float(aReg.iloc[int(NYEARS/_frp)][_reg])

    final_exceedance.to_csv('../inputs/RO/hazard/regional_exceedance_'+_haz+'.csv')
    return True



def exceedance_to_fa():
    eq_metric = 'caploss'
    pf_metric = 'popaff'

    fa = pd.merge(pd.read_csv('../inputs/RO/hazard/regional_exceedance_EQ_'+eq_metric+'_1M.csv').rename(columns={'loss':'EQ'}).fillna(0),
                  pd.read_csv('../inputs/RO/hazard/regional_exceedance_PF_'+pf_metric+'_1M.csv').rename(columns={'loss':'PF'}).fillna(0),on=['Region','rp']).set_index(['Region','rp'])

    #######################################################
    # Load population df
    ro_pop = pd.read_csv('../inputs/RO/hazard/Romania/Romania.province.data.unround.csv')[['Province',
                                                                                           'Data.Population']].rename(columns={'Province':'County','Data.Population':'population'})
    ro_pop['Region'] = ro_pop['County'].replace(pd.read_csv('../inputs/RO/county_to_region.csv').set_index('County')['Region'].to_dict())
    ro_pop = ro_pop.set_index('Region').sum(level='Region').squeeze()

    #######################################################
    # Load total exposed capital (for use with caploss)
    ro_cap = pd.read_excel('../inputs/RO/hazard/future_earth_2016/EQ/ECA Earthquake Final Spreadsheet 12112014.xlsx',sheet_name='Province',skiprows=1)
    ro_cap = ro_cap.loc[ro_cap['WB Country Name']=='Romania'].rename(columns={'WB Province Name':'County'})
    ro_cap['Region'] = ro_cap['County'].replace(pd.read_csv('../inputs/RO/county_to_region.csv').set_index('County')['Region'].to_dict())
    ro_cap = ro_cap.set_index(['Region','County'])
    #
    # CATDAT provides gross and net capital stock.
    # for all Counties in RO, net = 0.62597*gross
    ro_cap['gross_assets'] = ro_cap[['GDP (CATDAT)','Gross Capital Stock Multiplier (CATDAT)']].prod(axis=1)
    ro_cap['net_assets'] = ro_cap[['GDP (CATDAT)','Net Capital Stock Multiplier (CATDAT)']].prod(axis=1)
    ro_cap = ro_cap[['gross_assets','net_assets']]
    #
    ro_cap = ro_cap.sum(level='Region')['gross_assets'].squeeze()
    #



    #######################################################    
    # Convert PF risk (whichever metric it is) to fa
    if pf_metric == 'popaff': fa['PF'] = fa['PF']/ro_pop
    elif pf_metric == 'caploss': fa['PF'] = fa['PF']/ro_cap

    # Convert EQ risk (whichever metric that is) to fa
    if eq_metric == 'popaff': fa['EQ'] = fa['EQ']/ro_pop
    elif eq_metric == 'caploss': fa['EQ'] = fa['EQ']/ro_cap


    ########################################################
    # Process/package for saving out, to be picked up by gather_data
    fa = fa[['EQ','PF']].stack().to_frame()
    fa.index.names = ['Region','rp','hazard']
    fa.columns = ['fa']
    
    fa = fa.reset_index().set_index(['Region','hazard','rp']).sort_index()

    fa.to_csv('../inputs/RO/hazard/romania_multihazard_fa.csv')



############################
#collect_FL_hazard_data_RO()
#collect_EQ_hazard_data_RO()
#RO_county_to_region('EQ')
#RO_county_to_region('PF')
exceedance_to_fa()
