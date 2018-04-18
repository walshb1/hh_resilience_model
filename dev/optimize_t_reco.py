import numpy as np
import matplotlib.pyplot as plt

t_interval,_dt = np.linspace(0,10,52*10,retstep=True)

####################
# These don't change
eta = 1.5
pov_line = 10

####################
# These can change 
rng = [6.,3]
####################

pi = 0.333
pi_vals = [pi]#pi + pi/rng[0]*np.array(range(-1*rng[1],rng[1]+1))
# ^ avg productivity of cap

rho = 0.06
rho_vals = [rho]#rho + rho/rng[0]*np.array(range(-1*rng[1],rng[1]+1))
# ^ discount rate

c = 10*pov_line
c_vals = [c]#c + c/rng[0]*np.array(range(-1*rng[1],rng[1]+1))
# ^ consumption expressed as multiple of poverty line

dk0 = 0.1*(c/pi)
dk0_vals = dk0 + dk0/rng[0]*np.array(range(-1*rng[1],rng[1]+1))
# ^ expressed as fraction of assets (= c/pi)

t_reco = 4.
t_reco_vals = t_reco + t_reco/rng[0]*np.array(range(-1*rng[1],rng[1]+1))

########################
# Calculate
dw_sums = []
dw_dict = {}
dw_val_dict = {}

for _pi in pi_vals:
    for _rho in rho_vals:
        for _c in c_vals:
            for _dk0 in dk0_vals:
                for _t_reco in t_reco_vals:
                    _R = np.log(1/0.05)/_t_reco

                    dw_tot = 0

                    print([_pi,_rho,_c,_dk0,_t_reco])

                    const = -_c**(1-eta)/(1-eta)
                    
                    for _t in t_interval:
                        
                        _dw = const*((1-(_dk0/_c)*(_pi+_R)*np.exp(-_t*_R))**(1-eta)-1)*np.exp(-_t*_rho)
                        dw_tot += _dw

                    dw_dict[len(dw_sums)] = [_pi,_rho,_c,_dk0,_t_reco]
                    dw_val_dict[dw_tot] = len(dw_sums)

                    dw_sums.append(dw_tot)
                    


plt.scatter(range(len(dw_sums)),dw_sums)
plt.gcf().savefig('dw_optimization.pdf',format='pdf')

print(min(dw_sums),dw_dict[dw_val_dict[min(dw_sums)]])

