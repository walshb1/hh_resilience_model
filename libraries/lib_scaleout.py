import pandas as pd

def get_pmt(hh_df, aid_cutoff=887):
    hhdf_index = hh_df.index.names
    hh_df = hh_df.reset_index()

    # This function recreates the PMT from "A proxy means test for Sri Lanka" by Sebastian, et al. (June, 2018)
    pmt_df = (hh_df['hhid'].astype('str')).reset_index().set_index('hhid').copy()
    pmt_df = pmt_df[~pmt_df.index.duplicated(keep='first')].drop([_c for _c in pmt_df.columns],axis=1)
    # get a dataframe with index = hhid (just one entry per household--no hazard, rp levels) 
    
    # HIES INPUT FILES:
    df_demog = pd.read_csv('../inputs/SL/HIES2016/sec_1_demographic_information.csv')
    df_demog['hhid'] = (df_demog['District'].astype('str')+df_demog['Sector'].astype('str')
                        +df_demog['Psu'].astype('str')+df_demog['Snumber'].astype('str')+df_demog['Hhno'].astype('str'))
    df_demog = df_demog.reset_index().set_index('hhid').sort_index()


    #########################
    # HH head characteristics:
    # 1) is Female? (weight = 0.41)
    pmt_df['is_female'] = 0.41*(df_demog.loc[df_demog['Person_Serial_No']==1,'Sex']-1)
    # in HIES: 1=M & 2 =F, so just subtract 1 to get binary/bool

    # 2) IF is never married? (weight = 0); ELIF is married? (0.46); ELIF is widowed? (0.57); ELIF is divorced? (-1.44); ELIF is separated? (-4.5)
    pmt_df['marital_status'] = df_demog.loc[df_demog['Person_Serial_No']==1,'Marital_Status'].replace({1:0.00,
                                                                                                       2:0.46,
                                                                                                       3:0.57,
                                                                                                       4:-1.44,
                                                                                                       5:-4.50})

    # 3) Age? (weight = -0.24 per year)
    pmt_df['age'] = -0.24*df_demog.loc[df_demog['Person_Serial_No']==1,'Age']

    # 4) Employment: IF is govt or semi-govt? (weight = 10.27); ELIF is private employee? (0.25); ELIF is employer? (20.31); ELIF is own-account/casual worker (0.0)
    pmt_df['employment'] = df_demog.loc[df_demog['Person_Serial_No']==1,'Employment_Status'].replace({1:10.27,
                                                                                                      2:10.27,
                                                                                                      3:0.25,
                                                                                                      4:20.31,
                                                                                                      5:0,
                                                                                                      6:0}).fillna(0)

    # HH demographics
    # 1) Family size: 1 (weight = 81.81); 2 (57.65); 3 (40.52); 4 (26.91) 5 (15.63); 6+ (0)
    pmt_df['hhsize'] = (df_demog.groupby('hhid')['Person_Serial_No'].transform('count')).mean(level='hhid')
    pmt_df.loc[pmt_df['hhsize']>=6,'hhsize'] = 0.
    pmt_df['hhsize'] = pmt_df['hhsize'].replace({1:81.81,
                                                 2:57.65,
                                                 3:40.52,
                                                 4:26.91,
                                                 5:15.63})
      
    # 2) Highest education level (*of members not currently enrolled in school*)
    #      < Grade10 (weight = 0.0); G10 (9.1); O/L (14.28); A/L (22.19); University (35.15)
    pmt_df['education'] = df_demog.loc[df_demog.eval('(Curr_Educ==9) & (Education<=17)'),'Education'].max(level='hhid')
    pmt_df['education_score'] = 0
    pmt_df.loc[pmt_df.eval('(education==10)'),'education_score'] = 9.1
    pmt_df.loc[pmt_df.eval('(education==11)|(education==12)'),'education_score'] = 14.28
    pmt_df.loc[pmt_df.eval('(education==13)|(education==14)'),'education_score'] = 22.19
    pmt_df.loc[pmt_df.eval('(education==15)|(education==16)|(education==17)'),'education_score'] = 35.15
    pmt_df = pmt_df.drop('education',axis=1)

    # 3) Dependency ratio: (N_under_15 + N_over_65)/hh_size (weight = -12.2)
    pmt_df['dependency_ratio'] = -12.2*(((df_demog.loc[df_demog.eval('(Age<15)|(Age>64)')]).groupby('hhid')['Person_Serial_No'].transform('count')).mean(level='hhid')
                                        /(df_demog.groupby('hhid')['Person_Serial_No'].transform('count')).mean(level='hhid')).fillna(0)
    
    ################################
    # Assets:
    df_assets = pd.read_csv('../inputs/SL/HIES2016/sec_6a_durable_goods.csv').fillna(0)  
    df_assets['hhid'] = (df_assets['District'].astype('str')+df_assets['Sector'].astype('str')
                         +df_assets['Psu'].astype('str')+df_assets['Snumber'].astype('str')+df_assets['Hhno'].astype('str'))
    df_assets = df_assets.reset_index().set_index('hhid').rename(columns={'Washing_Mechine':'Washing_Machine'})

    # 1) has computer (weight = 12.37)
    pmt_df['has_computer'] = 0
    pmt_df.loc[df_assets['Computers']==1,'has_computer'] = 12.37
    # 2) has cooker (19.01)
    pmt_df['has_cooker'] = 0
    pmt_df.loc[df_assets['Cookers']==1,'has_cooker'] = 19.01
    # 3) has electric fan (11.89)
    pmt_df['has_electricfan'] = 0
    pmt_df.loc[df_assets['Electric_Fans']==1,'has_electricfan'] = 11.89
    # 4) has refrigerator (10.89)
    pmt_df['has_fridge'] = 0
    pmt_df.loc[df_assets['Fridge']==1,'has_fridge'] = 10.89
    # 5) has washing machine (13.27)
    pmt_df['has_washingmachine'] = 0
    pmt_df.loc[df_assets['Washing_Machine']==1,'has_washingmachine'] = 13.27
    # 6) has land phone (4.81)
    pmt_df['has_landphone'] = 0
    pmt_df.loc[df_assets['Telephone']==1,'has_landphone'] = 4.81
    # 7) has water pump (15.5)
    pmt_df['has_waterpump'] = 0
    pmt_df.loc[df_assets['Waterpumps']==1,'has_waterpump'] = 15.50
    # 8) has motorcycle (10.0)
    pmt_df['has_motorcycle'] = 0
    pmt_df.loc[df_assets['Motor_Bicycle']==1,'has_motorcycle'] = 10.0
    # 9) has car/van (39.69)
    pmt_df['has_car'] = 0
    pmt_df.loc[df_assets['Motor_Car_Van']==1,'has_car'] = 39.69
    # 10) has three-wheeler (14.02)
    pmt_df['has_3wheel'] = 0
    pmt_df.loc[df_assets['Three_Wheeler']==1,'has_3wheel'] = 14.02
    # 11) has four-wheel tractor (19.67)
    pmt_df['has_4wheel'] = 0
    pmt_df.loc[df_assets['Tractor_4_Wheel']==1,'has_4wheel'] = 19.67

    ################################
    # Housing quality & facilities
    df_house = pd.read_csv('../inputs/SL/HIES2016/sec_8_housing.csv').fillna(0)  
    df_house['hhid'] = (df_house['District'].astype('str')+df_house['Sector'].astype('str')
                         +df_house['Psu'].astype('str')+df_house['Snumber'].astype('str')+df_house['Hhno'].astype('str'))
    df_house = df_house.reset_index().set_index('hhid').rename(columns={'Tioilet_Use':'Toilet_Use'})

    # 1) bedrooms per person (DCS definition) (weight = 15.38 per bedroom)
    pmt_df['bdrms_per_person'] = 15.38*(df_house['Bed_Rooms']/(df_demog.groupby('hhid')['Person_Serial_No'].transform('count')).mean(level='hhid'))
    # 2) have floor tiles/terasso (14.41)
    pmt_df['has_flrtiles'] = 0
    pmt_df['has_flrtiles'].update(14.41*(df_house.loc[df_house['Floor']==2,'Floor'].clip(upper=1)))
    # 3) drinking water source inside unit (2.52)
    pmt_df['has_drinking'] = 0
    pmt_df['has_drinking'].update(2.52*(df_house.loc[df_house.eval('(Drinking_Water==1)|(Drinking_Water==4)|(Drinking_Water==5)'),'Drinking_Water'].clip(upper=1)))    
    # 4) electricity for lighting (8.52)
    pmt_df['has_eleclights'] = 0
    pmt_df['has_eleclights'].update(8.52*(df_house.loc[df_house['Lite_Source']==2,'Lite_Source'].clip(upper=1)))
    # 5) have wall of brick/kabok/cement (1.64)
    pmt_df['has_robustwall'] = 0
    pmt_df.loc[df_house.eval('(Walls>=1)&(Walls<=3)'),'has_robustwall'] = 1.64
    # 6) have toilet within unit (7.31)
    pmt_df['toilet_use'] = 0
    pmt_df.loc[df_house.eval('(Toilet_Use==1)|(Toilet_Use==2)'),'toilet_use'] = 7.31

    pmt_df['PMT'] = pmt_df.sum(axis=1)+833.5

    #hh_df['hhid'] = hh_df['hhid'].astype('str')
    #hh_df = pd.merge(hh_df.reset_index(),pmt_df['PMT'].reset_index(),on='hhid').set_index(hhdf_index)

    return pmt_df['PMT'], aid_cutoff


def get_scaleout_recipients(optionPDS,hh_df,ranking_var,aid_cutoff):
    hh_df_ix = hh_df.index.names

    # This function returns a list of recipients of PDS, along with the value of their receipt
    # notes: make sure it's done in a way that can account for targeting (incl/excl) error

    # Set a flag based on whether hh already receives samurdhi
    hh_df['enrolled_in_samurdhi'] = False
    hh_df.loc[hh_df['pcsamurdhi']!=0,'enrolled_in_samurdhi'] = True
    hh_df = hh_df.reset_index().set_index(hh_df_ix+['enrolled_in_samurdhi'])

    
    # for each group (location, hazard, RP, Samurdhi enrollment) set "help_received" = average
    hh_df = hh_df.reset_index(['hhid','affected_cat','helped_cat'])
    hh_df['help_received'] = (1/12)*(hh_df[['pcwgt','pcsamurdhi']].prod(axis=1).groupby(hh_df.index.names).transform('sum')
                                     /hh_df['pcwgt'].groupby(hh_df.index.names).transform('sum'))

    # so far, payout doesn't go to anyone not enrolled in samurdhi
    hh_df = hh_df.reset_index('enrolled_in_samurdhi')
    # ^ pull "enrolled_in_samurdhi" out of index
    hh_df['help_received'] = hh_df['help_received'].groupby(hh_df.index.names).transform('max')
    # ^ use transform('max') to apply payout to all hh

    # GET PMT (function defined in this file, above)
    pmt_df, aid_cutoff = get_pmt(hh_df,aid_cutoff)

    hh_df = pd.merge(hh_df.reset_index(),pmt_df.reset_index(),on='hhid').set_index(hh_df_ix)

    # So now all hh get a payout.
    # Which groups don't get PDS?
    hh_df = hh_df.reset_index('helped_cat')
    hh_df.loc[hh_df['helped_cat'] == 'not_helped','help_received'] = 0
    hh_df = hh_df.reset_index().set_index(hh_df_ix)
    # ^ 1) helped_cat = na
    hh_df.loc[hh_df['PMT'] > aid_cutoff,'help_received'] = 0
    # ^ 2) anyone whose PMT exceeds threshold

    #hh_df.loc[hh_df['enrolled_in_samurdhi']==True,'help_received'] = 0
    ## ^ 3) anyone already receiving Samurdhi

    return(hh_df['help_received'])
