import pandas as pd
from libraries.lib_country_dir import set_directories, load_survey_data

def get_idai_loss():
    _f = '/Users/brian/Desktop/BANK/hh_resilience_model/inputs/MW/CY_Idai/losses_table.xlsx'
    caploss = pd.read_excel(_f,sheet_name='capital_loss').set_index('district')
    return caploss

def get_pop_aff():
    total_pop = pd.read_csv('/Users/brian/Desktop/BANK/hh_resilience_model/inputs/MW/population_hies_vs_rms.csv').set_index('district')

    set_directories('MW')
    #
    #total_cap = load_survey_data('MW')
    #print(total_cap.head())
    #assert(False)
    #
    popaff = pd.read_excel(_f,sheet_name='pop_aff').set_index('district')
    popaff = popaff.join(total_pop[['population']],how='left')
    #
    popaff['fa'] = popaff.eval('people_affected/population')
    popaff['hazard'] = 'CY'
    popaff['rp'] = 1
    #
    #
    out = popaff.reset_index()[['district','hazard','rp','fa']]
    out.to_csv('/Users/brian/Desktop/BANK/hh_resilience_model/inputs/MW/CY_Idai/fa.csv')
    #


