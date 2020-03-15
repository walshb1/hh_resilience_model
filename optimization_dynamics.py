import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from libraries.lib_agents import optimize_reco

#
df = pd.DataFrame({'v':np.linspace(0.10,.80,100),'reco_rate':None})

_rho = 0.06
for _pi in [0.3,0.4,0.5]:

    df['reco_rate_{}'.format(int(10*_pi))] = df.apply(lambda x:optimize_reco({},_pi,_rho,x['v'],step=0.001),axis=1)        
    df['t_reco_{}'.format(int(10*_pi))] = np.log(1/0.05)/df['reco_rate_{}'.format(int(10*_pi))]

    plt.plot(df['v'],df['t_reco_{}'.format(int(10*_pi))],label='avg prod k = {}'.format(_pi))

plt.legend()
plt.xlim(0)
plt.ylim(0)

plt.xlabel('Asset vulnerability (fraction lost)')
plt.ylabel('Reconstruction time [years]')
sns.despine()

df.to_csv('~/Desktop/tmp/df.csv')
plt.gcf().savefig('optimal_reco_times_pi.pdf')
plt.cla()
#
df = pd.DataFrame({'v':np.linspace(0.10,.80,100),'reco_rate':None})

_pi = 0.40
for _rho in [0.04,0.06,0.08]:

    df['reco_rate_{}'.format(int(1E2*_rho))] = df.apply(lambda x:optimize_reco({},_pi,_rho,x['v'],step=0.001),axis=1)        
    df['t_reco_{}'.format(int(1E2*_rho))] = np.log(1/0.05)/df['reco_rate_{}'.format(int(1E2*_rho))]

    plt.plot(df['v'],df['t_reco_{}'.format(int(1E2*_rho))],label='discount rate = {}'.format(_rho))

plt.legend()
plt.xlim(0)
plt.ylim(0)

plt.xlabel('Asset vulnerability (fraction lost)')
plt.ylabel('Reconstruction time [years]')

sns.despine()

df.to_csv('~/Desktop/tmp/df.csv')
plt.gcf().savefig('optimal_reco_times_rho.pdf')
