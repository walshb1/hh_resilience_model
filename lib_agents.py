import pandas as pd

def smart_savers(temp,avg_prod_k,const_pds_rate):

    #avg_prod_k = macro['avg_prod_k'].mean()

    _a = temp.loc[temp.dk0!=0].copy()

    _a['sav_offset_to'] = ((_a['dk0']*(avg_prod_k+_a['hh_reco_rate'])
                           +_a['help_received']*const_pds_rate
                           +_a['sav_i']*(_a['help_received']*const_pds_rate**2-_a[['dk0','hh_reco_rate']].prod(axis=1)*(avg_prod_k+_a['hh_reco_rate']))
                           /(_a['dk0']*(avg_prod_k+_a['hh_reco_rate'])+_a['help_received']*const_pds_rate))
                           /(1-(_a['help_received']*const_pds_rate**2-_a[['dk0','hh_reco_rate']].prod(axis=1)*(avg_prod_k+_a['hh_reco_rate']))
                             /(_a['dk0']*(avg_prod_k+_a['hh_reco_rate'])+_a['help_received']*const_pds_rate))**2)
    
    return _a['sav_offset_to']
