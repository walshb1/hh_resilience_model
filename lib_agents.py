import pandas as pd

def smart_savers(temp,avg_prod_k,const_pub_reco_rate,const_pds_rate):

    _a = temp.copy()
    _a['sav_offset_to'] = 0. # <-- will remain zero for hh that don't reconstruct

    _a.loc[(_a.hh_reco_rate!=0),'sav_offset_to'] = 0.65*((_a['dc0_prv']+_a['dc0_pub']-_a['help_received']*const_pds_rate
                                                         -_a['sav_i']*((_a['help_received']*(const_pds_rate)**2-_a[['dc0_prv','hh_reco_rate']].prod(axis=1)-_a['dc0_pub']*const_pub_reco_rate)
                                                                       /(_a['help_received']*const_pds_rate-_a['dc0_prv']-_a['dc0_pub'])))
                                                        /(1.-_a['sav_i']*((_a['help_received']*(const_pds_rate)**2-_a[['dc0_prv','hh_reco_rate']].prod(axis=1)-_a['dc0_pub']*const_pub_reco_rate)
                                                                          /(_a['help_received']*const_pds_rate-_a['dc0_prv']-_a['dc0_pub'])**2)))
    
    return _a['sav_offset_to'].clip(lower=0.)
