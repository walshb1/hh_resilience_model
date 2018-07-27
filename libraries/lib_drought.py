import pandas as pd
import matplotlib.pyplot as plt

def drought_study(df):
    
    print('\n')
    print('Frac non-poor affected by drought:',df.loc[(df.impact_drought==1)&(df.ispoor=='Non-poor'),'pcwgt'].sum()/df.loc[(df.ispoor=='Non-poor'),'pcwgt'].sum())
    print('Frac poor affected by drought:',df.loc[(df.impact_drought==1)&(df.ispoor=='Non-poor'),'pcwgt'].sum()/df.loc[(df.ispoor=='Poor'),'pcwgt'].sum())
    print('Frac subsistence affected by drought:',df.loc[(df.impact_drought==1)&(df.issub=='Ultra-poor'),'pcwgt'].sum()/df.loc[(df.issub=='Ultra-poor'),'pcwgt'].sum())
    print('\n')

    print('Frac w/ ganyu income reporting affected by drought:',df.loc[(df.impact_drought==1)&(df.income_ganyu>0),'pcwgt'].sum()/df.loc[(df.income_ganyu>0),'pcwgt'].sum())
    print('Frac w/o ganyu income reporting affected by drought:',df.loc[(df.impact_drought==1)&(df.income_ganyu==0),'pcwgt'].sum()/df.loc[(df.income_ganyu==0),'pcwgt'].sum())
    print('\n')
    print('Frac w/ unpaid ag labor reporting affected by drought:',df.loc[(df.impact_drought==1)&(df.labor_ag_unpaid>0),'pcwgt'].sum()/df.loc[(df.labor_ag_unpaid>0),'pcwgt'].sum())
    print('Frac w/o unpaid ag labor reporting affected by drought:',df.loc[(df.impact_drought==1)&(df.labor_ag_unpaid==0),'pcwgt'].sum()/df.loc[(df.labor_ag_unpaid==0),'pcwgt'].sum())
    print('\n')
    print('Frac w/ ag wages reporting affected by drought:',df.loc[(df.impact_drought==1)&(df.main_wage_job_ag==1),'pcwgt'].sum()/df.loc[(df.main_wage_job_ag==1),'pcwgt'].sum())
    print('Frac w/o ag wages reporting affected by drought:',df.loc[(df.impact_drought==1)&(df.main_wage_job_ag==0),'pcwgt'].sum()/df.loc[(df.main_wage_job_ag==0),'pcwgt'].sum())
    print('\n')
    print('Frac w/ unpaid ag labor or ganyu reporting affected by drought:',(df.loc[(df.impact_drought==1)&((df.labor_ag_unpaid>0)|(df.income_ganyu>0)),'pcwgt'].sum()
                                                                             /df.loc[((df.labor_ag_unpaid>0)|(df.income_ganyu>0)),'pcwgt'].sum()))

    print('Frac w/o unpaid ag labor or ganyu reporting affected by drought:',(df.loc[(df.impact_drought==1)&(df.labor_ag_unpaid==0)&(df.income_ganyu==0),'pcwgt'].sum()
                                                                              /df.loc[(df.labor_ag_unpaid==0)&(df.income_ganyu==0),'pcwgt'].sum()))
  
    print('\n')
    print('Frac w/ unpaid ag labor/ganyu/ag wage reporting affected by drought:',(df.loc[(df.impact_drought==1)&((df.labor_ag_unpaid>0)|(df.income_ganyu>0)|(df.main_wage_job_ag==1)),'pcwgt'].sum()
                                                                                  /df.loc[((df.labor_ag_unpaid>0)|(df.income_ganyu>0)|(df.main_wage_job_ag==1)),'pcwgt'].sum()))
    print('Frac w/o unpaid ag labor/ganyu/ag wage reporting affected by drought:',(df.loc[(df.impact_drought==1)&(df.labor_ag_unpaid==0)&(df.income_ganyu==0)&(df.main_wage_job_ag==0),'pcwgt'].sum()
                                                                                   /df.loc[(df.labor_ag_unpaid==0)&(df.income_ganyu==0)&(df.main_wage_job_ag==0),'pcwgt'].sum()))  

    df = df.reset_index().set_index(['district','hhid'])
    df_dist = pd.DataFrame(index=df.sum(level='district').index)
    df_dist['frac_drought_impact'] = (df.loc[(df.impact_drought==1),'pcwgt'].sum(level='district')/df['pcwgt'].sum(level='district'))
    df_dist['frac_ganyu_income'] = (df.loc[(df.income_ganyu>0),'pcwgt'].sum(level='district')/df['pcwgt'].sum(level='district'))
    df_dist['frac_ganyu_plus'] = (df.loc[(df.income_ganyu>0)|(df.main_wage_job_ag==1),'pcwgt'].sum(level='district')/df['pcwgt'].sum(level='district'))
    df_dist['frac_ag_unpaid'] = (df.loc[(df.labor_ag_unpaid>0),'pcwgt'].sum(level='district')/df['pcwgt'].sum(level='district'))

    #print(df_dist)

    ax = df_dist.plot.scatter('frac_ganyu_income','frac_drought_impact',color='red')
    df_dist.plot.scatter('frac_ganyu_plus','frac_drought_impact',color='blue',ax=ax)
    df_dist.plot.scatter('frac_ag_unpaid','frac_drought_impact',color='green',ax=ax)

    plt.gca().get_figure().savefig('../output_plots/MW/drought_impacts.pdf',format='pdf')

    df_ag = pd.read_stata('../inputs/MW/MWI_2016_IHS-IV_v02_M_Stata/AG_MOD_M.dta',columns=['case_id','ag_m10a','ag_m10b']).rename(columns={'case_id':'hhid'}).set_index('hhid')
    print(df_ag.head())
    print(df_ag.loc[(df_ag.ag_m10a=='Drought')|(df_ag.ag_m10b=='Drought')])

    #print(df.head())
    assert(False)
