from libraries.lib_policy_scorecard import plot_impact_for_single_event
from libraries.lib_plot_recovery_times import *

plot_impact_for_single_event('RO',event=('Bucharest-Ilfov','EQ',200))
plot_impact_for_single_event('RO',event=('Bucharest-Ilfov','EQ','aal'))
plot_impact_for_single_event('RO',event=('North-East','EQ','aal'))
plot_impact_for_single_event('RO',event=('North-East','PF','aal'))
plot_impact_for_single_event('RO',event=('Bucharest-Ilfov','PF','aal'))
