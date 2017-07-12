from IPython import get_ipython
get_ipython().magic('reset -f')
get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')

from multiprocessing import Pool
from compute_resilience_and_risk import launch_compute_resilience_and_risk_thread

if __name__ == '__main__':
    print('here')

    my_pool = Pool(processes=1)
    #my_pool.map(launch_compute_resilience_and_risk_thread, ['']) 
    
    launch_compute_resilience_and_risk_thread('FJ','')
