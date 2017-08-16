import moviepy.editor as mpy 
import numpy as np

movie_array = []
for nfood, dfood in enumerate(np.linspace(0.00,1.00,21)):
    dfood_str = str(5*int(nfood))

    movie_array.append('../output_plots/FJ/sectoral/food_inc'+dfood_str+'_vs_income.png')

my_clip = mpy.ImageSequenceClip(movie_array, fps=2.0)
my_clip.write_gif('../output_plots/FJ/sectoral/food_inc_vs_income.gif')

assert(False)

file_dir = '/Users/brian/Desktop/BANK/hh_resilience_model/output_plots/FJ/png/'
for haz in ['EQTS','TC']:
    file_list = [file_dir+'poverty_k_'+haz+'_1.png',
                 file_dir+'poverty_k_'+haz+'_10.png',
                 file_dir+'poverty_k_'+haz+'_22.png',
                 file_dir+'poverty_k_'+haz+'_50.png',
                 file_dir+'poverty_k_'+haz+'_72.png',
                 file_dir+'poverty_k_'+haz+'_100.png',
                 file_dir+'poverty_k_'+haz+'_224.png',
                 file_dir+'poverty_k_'+haz+'_475.png',
                 file_dir+'poverty_k_'+haz+'_975.png',
                 file_dir+'poverty_k_'+haz+'_2475.png']

    my_clip = mpy.ImageSequenceClip(file_list, fps=1.5)
    my_clip.write_gif(file_dir+'_gif_poverty_'+haz+'.gif')
    
    #myclip = mpy.ImageClip(file_list).write_gif(file_dir+'poverty_eq.gif')
