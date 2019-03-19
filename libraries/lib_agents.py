import pandas as pd
import numpy as np

def smart_savers(c,dc0,hhrr,pi,Vsav,_cttot=1):

    if dc0 == 0: return 0,10
    if hhrr == 0: return int(round(min(dc0,max(dc0-(2/3)*Vsav,0.)),0)), 1.

    gamma = dc0
    last_result = None

    while True:

        beta = gamma/dc0
        result = dc0*(1-beta)+gamma*np.log(beta)-Vsav*hhrr

        try:
            if (last_result < 0 and result > 0) or (last_result > 0 and result < 0):

                _t = -np.log(beta)/hhrr

                if _t < 0:
                    print('RESULT!:\ngamma = ',gamma,'& beta = ',beta,' & t = ',_t)
                    print('CHECK:',dc0*np.e**(-hhrr*_t),' gamma = ',gamma)

                if _t >= 10: return int(round(min(dc0,max(dc0-(2/3)*Vsav,0.)),0.)), 1.
                return int(round(gamma,0)), round(_t,3)

        except: pass

        last_result = result
        gamma -= 0.01*dc0
        if gamma <= 0: return 0,10


def optimize_reco(v_to_reco_rate, pi, rho, v, verbose=False):

    if v == 0: return 0
    v = round(float(v),2)

    if v in v_to_reco_rate: return v_to_reco_rate[v]
    ###############

    eta = 1.5

    last_integ = None
    last_lambda = None

    _l = 0.0
    while True:

        if pi-(pi+_l)*v < 0: assert(False)

        x_max = 15
        dt_step = 52*x_max

        integ = 0
        for _t in np.linspace(0,x_max,dt_step):
            integ += np.e**(-_t*(rho+_l)) * ((pi+_l)*_t-1) * (pi-(pi+_l)*v*np.e**(-_l*_t))**(-eta)

        if last_integ and ((last_integ < 0 and integ > 0) or (last_integ > 0 and integ < 0)):
            print('\n Found the Minimum!\n lambda = ',last_lambda,'--> integ = ',last_integ)
            print('lambda = ',_l,'--> integ = ',integ,'\n')

            _out = (_l+last_lambda)/2

            v_to_reco_rate[v] = _out
            return _out

        last_integ = integ
        if last_integ is None: assert(False)

        last_lambda = _l
        _l += 0.01

#print(optimize_reco())
#smart_savers(None,None,None,None)
