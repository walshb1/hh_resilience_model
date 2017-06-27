from IPython import get_ipython
get_ipython().magic('reset -f')
get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')

import gc
import sys
import os, time
import warnings
import numpy as np
import pandas as pd

from lib_country_dir import *
from lib_compute_resilience_and_risk import *
from replace_with_warning import *

warnings.filterwarnings('always',category=UserWarning)

if len(sys.argv) < 2:
    print('Need to list country. Try PH or FJ')
    assert(False)
else: myCountry = sys.argv[1]

# Setup directories
output       = model+'/../output_country/'+myCountry+'/'
intermediate = model+'/../intermediate/'+myCountry+'/'
if not os.path.exists(output):
    os.makedirs(output)

########################################################
########################################################

# How is it paid for? 
# --> 'tax' = based on income 
# --> 'insurance_premium' = based on vulnerability
optionFee = 'tax'

if optionFee=='insurance_premium':
    optionB='unlimited'
    optionT='perfect'
else:
    optionB='data'
    optionT='perfect'#'data'

# How much is disbursed?
# --> 'unif_poor' = uniform disbursement based of average asset losses of poor
# --> 'unif' = uniform disbursement based of average losses
optionPDS = 'no'#'unif_poor'#'no'

# Cap on benefits (bool)
option_CB = 1 #0 is for calculation of benefits only; 1 by default

if option_CB==0:
    option_CB_name = 'benefits_only'
else:
    option_CB_name = ''

print('optionFee =',optionFee, 'optionPDS =', optionPDS, 'optionB =', optionB, 'optionT =', optionT, 'option_CB =', option_CB_name)

#Options and parameters
nat_economy   = 'national'
global economy
economy       = get_economic_unit(myCountry)
event_level   = [economy, 'hazard', 'rp']                            #levels of index at which one event happens
default_rp    = 'default_rp'                                         #return period to use when no rp is provided (mind that this works with protection)
income_cats   = 'hhid'                                               #categories of households
affected_cats = pd.Index(['a', 'na'], name='affected_cat')           #categories for social protection
helped_cats   = pd.Index(['helped','not_helped'], name='helped_cat')

#read data
macro = pd.read_csv(intermediate+'macro.csv', index_col=economy)
cat_info = pd.read_csv(intermediate+'cat_info.csv',  index_col=[economy, income_cats])

# First function: compute_with_hazard_ratios 
# --> This is a shell that loads hazard_ratios, then calls process_input
#  Inputs:
#  - macro has province-level info
#  - cat_info has household-level info
#  - hazard_ratios has fa for each household (which varies not by hh, but by province, hazard, & RP) 
macro_event, cats_event, hazard_ratios_event = compute_with_hazard_ratios(myCountry,intermediate+'hazard_ratios.csv',macro,cat_info,economy,event_level,income_cats,default_rp,verbose_replace=True)

gc.collect()
print('A')
#verbose_replace=True by default, replace common columns in macro_event and cats_event with those in hazard_ratios_event

# compute_dK does the following:
# -- adds dk_event column to macro_event
# -- adds affected/na categories to cats_event
macro_event, cats_event_ia = compute_dK(macro_event, cats_event,event_level,affected_cats) #calculate the actual vulnerability, the potential damange to capital, and consumption
print('B\n\n')

macro_event, cats_event_iah = calculate_response(macro_event,cats_event_ia,event_level,helped_cats,default_rp,option_CB,optionFee=optionFee,optionT=optionT, optionPDS=optionPDS, optionB=optionB,loss_measure='dk',fraction_inside=1, share_insured=.25)
print('C\n\n')

#optionFee: tax or insurance_premium  optionFee='insurance_premium',optionT='perfect', optionPDS='prop', optionB='unlimited',optionFee='tax',optionT='data', optionPDS='unif_poor', optionB='data',
#optionT(targeting errors):perfect, prop_nonpoor_lms, data, x33, incl, excl.
#optionB:one_per_affected, one_per_helped, one, unlimited, data, unif_poor, max01, max05
#optionPDS: unif_poor, no, 'prop', 'prop_nonpoor'
macro_event.to_csv(output+'macro_'+optionFee+'_'+optionPDS+'_'+option_CB_name+'.csv',encoding='utf-8', header=True)
print('D')

cats_event_iah.to_csv(output+'cats_'+optionFee+'_'+optionPDS+'_'+option_CB_name+'.csv',encoding='utf-8', header=True)
print('E')

out = compute_dW(macro_event,cats_event_iah,event_level,option_CB,return_stats=True,return_iah=True)
print('F')

# Flag: running local welfare
print('running national welfare')
results,iah = process_output(out,macro_event,economy,default_rp,return_iah=True,is_local_welfare=False)
print('G')

results.to_csv(output+'results_'+optionFee+'_'+optionPDS+'_'+option_CB_name+'.csv',encoding='utf-8', header=True)
print('H')

iah.to_csv(output+'iah_'+optionFee+'_'+optionPDS+'_'+option_CB_name+'.csv',encoding='utf-8', header=True)
print('I')

# result1=pd.read_csv('output-old/results.csv', index_col=economy)
# iah1=pd.read_csv('output-old/iah.csv', index_col=event_level+['income_cat','affected_cat','helped_cat'])
# print(((result1-results)/results).max())
# print(((iah1-iah.reset_index().set_index(event_level+['income_cat','affected_cat','helped_cat']))/iah1).max())
