import pandas as pd
from libraries.lib_gather_data import get_hh_savings

df = pd.read_csv('../../debug/temp_init.csv')

print(get_hh_savings(df[['province','ispoor','axfin','c']],'PH','','../inputs/PH/Socioeconomic Resilience (Provincial)_Print Version_rev1.xlsx').head())
