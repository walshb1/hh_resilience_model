import seaborn as sns
paired_pal = sns.color_palette('Paired', n_colors=12)

pds_dict = {'unif_poor':'uniform payout',
            'unif_poor_only':'uniform payout to poorest 20%',
            'unif_poor_q12':'uniform payout to poorest 40%',
            'scaleout_samurdhi':'Samurdhi to affected for 1 month (PMT $\leq$ 887; perfect targeting)',
            'scaleout_samurdhi_universal':'Samurdhi to affected for 1 month (perfect targeting)',
            'samurdhi_scaleup':'1 month topup to Samurdhi enrollees (no targeting)',
            'samurdhi_scaleup66':'increase existing Samurdhi payments (66% incl err)',
            'samurdhi_scaleup33':'increase existing Samurdhi payments (33% incl err)',
            'samurdhi_scaleup00':'1 month topup to affected Samurdhi enrollees (perfect targeting)',
            'prop_q12':'public insurance for poorest 40%',
            'prop_q1':'public insurance for poorest 20%',
            'no':'no support'}

pds_colors = {'unif_poor':None,
              'unif_poor_only':None,
              'unif_poor_q12':None,
              'scaleout_samurdhi':paired_pal[0],
              'scaleout_samurdhi_universal':paired_pal[1],
              'samurdhi_scaleup':paired_pal[3],
              'samurdhi_scaleup66':None,
              'samurdhi_scaleup33':None,
              'samurdhi_scaleup00':paired_pal[2],
              'prop_q12':paired_pal[8],
              'prop_q1':paired_pal[9],
              'no':paired_pal[10]}

pds_crit_dict = {'scaleout_samurdhi':'Affected and PMT $\leq$ 887 (perfect targeting)',
                 'scaleout_samurdhi_universal':'All affected (perfect targeting)',
                 'samurdhi_scaleup':'All Samurdhi enrollees (no targeting)',
                 'samurdhi_scaleup66':'Affected and in Samurdhi (66% incl err)',
                 'samurdhi_scaleup33':'Affected and in Samurdhi (33% incl err)',
                 'samurdhi_scaleup00':'Samurdhi enrollee and affected (perfect targeting)'
                 }

#['unif_poor_only', 'samurdhi_scaleup33', 'scaleout_samurdhi', 'prop_q12', 'unif_poor_q12', 'samurdhi_scaleup66', 'scaleout_samurdhi_universal', 'prop_q1', 'no', 'unif_poor', 'samurdhi_scaleup', 'samurdhi_scaleup00'] 'samurdhi_scaleup','scaleout_samurdhi_universal']
