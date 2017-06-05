


###Social transfer Data


####early warning system

###Income group


###Country Ratings


#Transforms ratings letters into 1-100 numbers


###Ratings + HFA
df['borrow_abi']=(df['rating']+df['finance_pre'])/2 # Ability and willingness to improve transfers after the disaster


##Hazards data
###Vulnerability
#VULNERABILITY OF EACH HOUSEHOLD, BUT NEED A VULNERABILITY CURVE TO TRANSLATE HOUSING TYPE INTO VULNERABILITY VALUE

#EXPOSURE TO FLOOD FROM SURVEY HAZARD DATA/GLOFRIS

##Protection


##Data by income categories
cat_info =pd.DataFrame()
cat_info['n']  = concat_categories(ph,(1-ph),index= income_cats)	#number
cp=   df['share1'] /ph    *df['gdp_pc_pp']	#consumption levels, by definition.
cr=(1-df['share1'])/(1-ph)*df['gdp_pc_pp']
cat_info['c']       = concat_categories(cp,cr,index= income_cats)
cat_info['social']  = concat_categories(df.social_p,df.social_r,index= income_cats)	#diversification
cat_info['axfin'] = concat_categories(df.axfin_p,df.axfin_r,index= income_cats)	#access to finance
cat_info = cat_info.dropna()

##Taxes, redistribution, capital
df['tau_tax'],cat_info['gamma_SP'] = social_to_tx_and_gsp(economy,cat_info)	#computes tau tax and gamma_sp from socail_poor and social_nonpoor. CHECKED!
cat_info['k'] = (1-cat_info['social'])*cat_info['c']/((1-df['tau_tax'])*df['avg_prod_k']) #here k in cat_info has poor and non poor, while that from capital_data.csv has only k, regardless of poor or nonpoor



#access to early warnings





if drop_unused_data:
    cat_info= cat_info.drop(['social'],axis=1, errors='ignore').dropna()
    df_in = df.drop(['social_p', 'social_r','share1','pov_head', 'pe','vp','vr', 'axfin_p',  'axfin_r','rating','finance_pre'],axis=1, errors='ignore').dropna()
else :
    df_in = df.dropna()
df_in = df_in.drop([ 'shew','v'],axis=1, errors='ignore').dropna()

#Save all data
fa_guessed_gar.to_csv(intermediate+'/fa_guessed_from_GAR_and_PAGER_shaved.csv',encoding='utf-8', header=True)
pd.DataFrame([vp,vr,v], index=['vp','vr','v']).T.to_csv(intermediate+'/v_pr_fromPAGER_shaved_GAR.csv',encoding='utf-8', header=True)
df_in.to_csv(intermediate+'/macro.csv',encoding='utf-8', header=True)
cat_info.to_csv(intermediate+'/cat_info.csv',encoding='utf-8', header=True)
hazard_ratios.to_csv(intermediate+'/hazard_ratios.csv',encoding='utf-8', header=True)

