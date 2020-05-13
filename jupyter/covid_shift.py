import numpy as np
import pandas as pd 
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import math
import glob,os
import scipy

myCountry = 'PH'
poverty_type = 'cons'

# formatting & aesthetics
font = {'family':'sans serif', 'size':10}
plt.rc('font', **font)
mpl.rcParams['xtick.labelsize'] = 10
mpl.rcParams['ytick.labelsize'] = 10
mpl.rcParams['legend.facecolor'] = 'white'
sns.set_style("white")

sns_pal = sns.color_palette('Set1', n_colors=8, desat=.4)
greys_pal = sns.color_palette('Greys', n_colors=9)

haz_dict = {'EQ':'earthquakes','HU':'wind events','SS':'storm surges','PF':'precipitation\nflooding'}

def get_economy(myCountry):
    if myCountry == 'PH': return 'region'
    else: return 'Region'

def get_currency(myCountry):
    if myCountry == 'PH': return [1E-9,'bil. ','PhP']
    if myCountry == 'RO': return [1,'','XXX']
    return [1E-6,'mil. ','PPP$'] # all ECA
    
def get_expectation_values(df,myCountry):
    
    #ix_levels = np.array(df.index.names)
    #ix_levels = np.array(ix_levels[((ix_levels!='affected_cat')&(ix_levels!='helped_cat'))])
    ix_levels = [get_economy(myCountry),'hazard','rp','hhid','quintile']
    #print(ix_levels)
    
    df_ev = pd.DataFrame(index=(df.sum(level=ix_levels)).index)
    
    for _col in df:
        if _col == 'pcwgt': df[['pcwgt']].sum(level=ix_levels)
        df_ev[_col] = df[['pcwgt',_col]].prod(axis=1).sum(level=ix_levels)/df['pcwgt'].sum(level=ix_levels)
        
    return df_ev
        

def load_hh_df(myCountry,pds_scenario='no'):
    
    in_file = '../../output_country/{}/poverty_duration_{}.csv'.format(myCountry,pds_scenario)
    master_ix = [get_economy(myCountry),'hazard','rp','hhid','affected_cat','helped_cat']

    df = pd.read_csv(in_file).set_index(master_ix).dropna(how='any')
    df = get_quintile_info(myCountry,df,master_ix+['quintile'])
    df = load_income_classification(myCountry,df,master_ix+['quintile'])

    df = load_wellbeing_losses(myCountry,df,master_ix+['quintile'],pds_scenario)
    
    return df.sort_index()

def get_rho(myCountry):
    in_file = '../../output_country/{}/macro_tax_no_.csv'.format(myCountry)
    return float(pd.read_csv(in_file).dropna(how='any')['rho'].mean())

def get_average_income(myCountry):
    in_file = '../../output_country/{}/iah_tax_no_.csv'.format(myCountry)
    iah = pd.read_csv(in_file).dropna(how='any')[['c','pcwgt']]
    c = iah[['c','pcwgt']].prod(axis=1).sum()/iah['pcwgt'].sum()
    return c
avg_c = get_average_income(myCountry)

def load_wellbeing_losses(myCountry,df,master_ix,pds_scenario):
    in_file = '../../output_country/{}/iah_tax_{}_.csv'.format(myCountry,pds_scenario)
    iah = pd.read_csv(in_file).set_index(master_ix).dropna(how='any')[['c','dw','pcwgt']]
    
    wprime = float(iah[['c','pcwgt']].prod(axis=1).sum()/iah['pcwgt'].sum())**(-1.5)
    iah['dw'] /= wprime
    
    df = pd.merge(df.reset_index(),iah[['dw']].reset_index(),on=master_ix,how='left').set_index(master_ix)
    return df

def get_quintile_info(myCountry,df,master_ix):
    quints = pd.read_csv('../../intermediate/{}/cat_info.csv'.format(myCountry),usecols=['hhid','quintile']).set_index('hhid')
    df = pd.merge(quints.reset_index(),df.reset_index(),on='hhid').set_index(master_ix)
    return df

def slice_on_pop(df,segment=None,poverty_type=None):
    if segment == 'poor': print('DANGER: not returning poor hh. is intended?')
    encode = {'poor':'pov',
              'vulnerable':'vul',
              'secure':'sec',
              'middleclass':'mc'}
    
    try: code = 't_{}_{}'.format(encode[segment],poverty_type)
    except: code = 'no_encoding'
        
    if segment == 'poor': 
        pop_slice = (False,True,True,True)
        df = df.loc[(df.ispoor==pop_slice[0])
                    &((df.isvulnerable==pop_slice[1])|(df.issecure==pop_slice[2])|(df.ismiddleclass==pop_slice[3]))]
    elif segment is not None:
        if segment == 'vulnerable': pop_slice = (False,True,False,False)
        if segment == 'secure': pop_slice = (False,False,True,False)
        if segment == 'middleclass': pop_slice = (False,False,False,True)
        df = df.loc[(df.ispoor==pop_slice[0])
                    &(df.isvulnerable==pop_slice[1])&(df.issecure==pop_slice[2])&(df.ismiddleclass==pop_slice[3])] 
    else: pass

    return df,code

def get_population(myCountry,segment):
    _cols = [get_economy(myCountry),'quintile','pcwgt','ispoor','ismiddleclass','issecure','isvulnerable']
    tot_pop = pd.read_csv('../../intermediate/{}/cat_info.csv'.format(myCountry),usecols=_cols).set_index([get_economy(myCountry),'quintile'])
    
    if segment != 'poor': tot_pop,_ = slice_on_pop(tot_pop,segment)
    else: tot_pop = tot_pop.loc[tot_pop.ispoor==True]
    
    tot_pop = tot_pop[['pcwgt']].sum(level=[get_economy(myCountry),'quintile'])
    
    return tot_pop
    
def load_income_classification(myCountry,df,master_ix):
    cat_info = pd.read_csv('../../intermediate/{}/cat_info.csv'.format(myCountry))
    
    if 'ispoor' not in cat_info.columns or cat_info['ispoor'].astype('int').sum()==0: cat_info['ispoor'] = cat_info.c<=5.5*365
    if 'isvulnerable' not in cat_info.columns: cat_info['isvulnerable'] = (cat_info.c>5.5*365)&(cat_info.c<=10.*365)
    if 'issecure' not in cat_info.columns: cat_info['issecure'] = (cat_info.c>10*365)&(cat_info.c<15.*365)
    if 'ismiddleclass' not in cat_info.columns: cat_info['ismiddleclass'] = (cat_info.c>=15.*365)

    cat_info.to_csv('../../intermediate/{}/cat_info.csv'.format(myCountry))
    
    cat_info = cat_info[['hhid','ispoor','ismiddleclass','issecure','isvulnerable']]
    df = pd.merge(df.reset_index(),cat_info.reset_index(),on='hhid').set_index(master_ix)    
    return df


def average_over_rp(df,default_rp='default_rp',protection=None,return_probs=True):        
    """Aggregation of the outputs over return periods"""    
    if protection is None:
        protection=pd.Series(0,index=df.index)        

    #just drops rp index if df contains default_rp
    if default_rp in df.index.get_level_values('rp'):
        print('default_rp detected, dropping rp')
        return (df.T/protection).T.reset_index('rp',drop=True)
           
    df=df.copy().reset_index('rp')

    #computes frequency of each return period
    return_periods=np.unique(df['rp'].dropna())

    proba = pd.Series(np.diff(np.append(1/return_periods,0)[::-1])[::-1],index=return_periods) #removes 0 from the rps 

    #matches return periods and their frequency
    proba_serie=df['rp'].replace(proba).rename('prob')
    proba_serie1 = pd.concat([df.rp,proba_serie],axis=1)

    #handles cases with multi index and single index (works around pandas limitation)
    idxlevels = list(range(df.index.nlevels))

    if idxlevels==[0]:
        idxlevels =0
        
    #average weighted by proba
    averaged = df.mul(proba_serie,axis=0).sum(level=idxlevels).drop('rp',axis=1) # frequency times each variables in the columns including rp.


    if return_probs: return averaged,proba_serie1 #here drop rp.
    else: return averaged


def set_directories(myCountry):
    path = os.getcwd()+'/figures/{}'.format(myCountry)
    if not os.path.isdir(path): os.mkdir(path)
    path+='/duration'
    if not os.path.isdir(path): os.mkdir(path)




#def main():
 
set_directories(myCountry)
master_df = load_hh_df(myCountry)
print(master_df.head())
#master_df.head(500).to_csv('test.csv') 