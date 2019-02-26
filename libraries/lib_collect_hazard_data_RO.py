import pandas as pd

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
    
    

    return True


###########################
collect_FL_hazard_data_RO()
assert(False)
collect_EQ_hazard_data_RO()

