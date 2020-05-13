import pandas as pd

def get_hhid_lfs(df):

	df['hhid_lfs'] = (df['w_prov'].map(str)
    				  + df['w_mun'].map(str)
                      + df['w_bgy'].map(str)
                      + df['w_ea'].map(str)
                      + df['w_shsn'].map(str)
                      + df['w_hcn'].map(str)).astype(str)

	# if not df.index.is_unique:
    	# df.loc[df.index.duplicated()].to_csv('csv/get_hhid_lfs_nonunique.csv')
    	# assert(False)