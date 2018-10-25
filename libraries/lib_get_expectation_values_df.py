from libraries.lib_country_dir import get_subsistence_line
import pandas as pd

def get_expectation_values_df(myC,economy,iah,pds_options,base_str='no',use_aewgt=False):
    if use_aewgt: assert(False)

    wprime = pd.read_csv('../output_country/'+myC+'/results_tax_'+base_str+'_.csv', index_col=[economy,'hazard','rp']).wprime.mean()

    # Clone index of iah with just one entry/hhid
    iah_avg = pd.DataFrame(index=(iah.sum(level=[economy,'hazard','rp','hhid'])).index)

    ## Translate from iah by summing over hh categories [(a,na)x(helped,not_helped)]
    # These are special--pcwgt has been distributed among [(a,na)x(helped,not_helped)] categories
    for _pds in [base_str]+pds_options:
        try: iah_avg['pcwgt_'+_pds] = iah['pcwgt_'+_pds].sum(level=[economy,'hazard','rp','hhid'])
        except: iah_avg['aewgt_'+_pds] = iah['aewgt_'+_pds].sum(level=[economy,'hazard','rp','hhid'])
        #iah_avg['hhwgt_'+base_str] = iah['hhwgt_'+_pds].sum(level=[economy,'hazard','rp','hhid'])

    #These are the same across [(a,na)x(helped,not_helped)] categories 
    iah_avg['k']         = iah['k'].mean(level=[economy,'hazard','rp','hhid'])
    iah_avg['c_initial']         = iah['c_initial'].mean(level=[economy,'hazard','rp','hhid'])
    iah_avg['quintile']  = iah['quintile'].mean(level=[economy,'hazard','rp','hhid'])
    iah_avg['pov_line']  = iah['pov_line'].mean(level=[economy,'hazard','rp','hhid'])

    # Get subsistence line
    if get_subsistence_line(myC) != None:
        iah_avg['sub_line'] = get_subsistence_line(myC)
        
    # These need to be averaged across [(a,na)x(helped,not_helped)] categories (weighted by pcwgt)
    # ^ values still reported per capita
    iah_avg['dk0']         = iah[[  'dk0','pcwgt_'+base_str]].prod(axis=1).sum(level=[economy,'hazard','rp','hhid'])/iah_avg['pcwgt_'+base_str]
    iah_avg['i_pre_reco'] = iah[['i_pre_reco','pcwgt_'+base_str]].prod(axis=1).sum(level=[economy,'hazard','rp','hhid'])/iah_avg['pcwgt_'+base_str]
    iah_avg['di_pre_reco'] = iah[['di_pre_reco','pcwgt_'+base_str]].prod(axis=1).sum(level=[economy,'hazard','rp','hhid'])/iah_avg['pcwgt_'+base_str]
    iah_avg['c_pre_reco'] = iah[['c_pre_reco','pcwgt_'+base_str]].prod(axis=1).sum(level=[economy,'hazard','rp','hhid'])/iah_avg['pcwgt_'+base_str]

    for _ipds in [_c for _c in iah.columns if ('dw_' in _c) or ('help_received_' in _c)]:
        iah_avg[_ipds]     = (iah[[_ipds,'pcwgt_'+base_str]].prod(axis=1).sum(level=[economy,'hazard','rp','hhid'])/iah_avg['pcwgt_'+base_str])
        
    # Calc people who fell into poverty on the regional level for each disaster
    iah_avg['delta_pov_pre_reco']  = iah.loc[(iah.c_initial > iah.pov_line)&(iah.c_pre_reco <= iah.pov_line),'pcwgt_'+base_str].sum(level=[economy,'hazard','rp','hhid'])
    iah_avg['delta_pov_post_reco'] = iah.loc[(iah.c_initial > iah.pov_line)&(iah.c_post_reco <= iah.pov_line),'pcwgt_'+base_str].sum(level=[economy,'hazard','rp','hhid'])

    iah_avg = iah_avg.reset_index()
    iah_avg['delta_pov_pre_reco'] = iah_avg.groupby([economy,'hazard','rp'])['delta_pov_pre_reco'].transform('sum')
    iah_avg['delta_pov_post_reco'] = iah_avg.groupby([economy,'hazard','rp'])['delta_pov_post_reco'].transform('sum')
    iah_avg = iah_avg.reset_index().set_index([economy,'hazard','rp','hhid']).drop('index',axis=1)

    return iah_avg
