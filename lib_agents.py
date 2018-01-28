import pandas as pd

def smart_savers(temp,avg_prod_k,const_pub_reco_rate,const_pds_rate):

    _a = temp[['dc0_prv','dc0_pub','help_received','sav_f','hh_reco_rate']].copy()

    _a['sav_offset_to'] = 0.25*(_a['dc0_prv']+_a['dc0_pub']-_a['help_received']*const_pds_rate) 
    # ^ will remain at this level for hh that don't reconstruct

    savings_offset = '((dc0_prv+dc0_pub-help_received*@const_pds_rate-sav_f*((help_received*(@const_pds_rate)**2-dc0_prv*hh_reco_rate-dc0_pub*@const_pub_reco_rate)/(help_received*@const_pds_rate-dc0_prv-dc0_pub)))/(1.-sav_f*((help_received*(@const_pds_rate)**2-dc0_prv*hh_reco_rate-dc0_pub*@const_pub_reco_rate)/(help_received*@const_pds_rate-dc0_prv-dc0_pub)**2)))'

    _a.loc[(_a.hh_reco_rate!=0),'sav_offset_to'] = 0.65*_a.loc[(_a.hh_reco_rate!=0)].eval(savings_offset)

    return _a['sav_offset_to'].clip(lower=0.)
