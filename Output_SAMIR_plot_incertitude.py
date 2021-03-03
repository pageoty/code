#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 09:42:48 2021

@author: pageot
"""

import os
import sqlite3
import geopandas as geo 
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
import csv
import numpy as np
import pandas as pd
import seaborn as sns 
import TEST_ANALYSE_SIGNATURE
import shapely.geometry as geom
import descartes
import pickle
from datetime import datetime, date, time, timezone
from scipy import *
from scipy import stats
from pylab import *
from sklearn.metrics import *
from sklearn.linear_model import LinearRegression


def predict(x):
   return slope * x + intercept

def select_color_NDVI(x):
    couleurs=[]
    for i in range(len(x)):
        if x.iloc[i].NDVI<= 0.5 : 
            couleurs.append("r")
        else : 
            couleurs.append("b")
    return couleurs

def select_color_date(x):
    couleurs=[]
    for i in range(len(x)):
        if x.date[i].strftime('%m-%d')<= "05-01" : 
            couleurs.append("r")
        else : 
            couleurs.append("b")
    return couleurs

if __name__ == '__main__':
    d={}
    # name_run="Bilan_hydrique/RUN_FERMETURE_BILAN_HYDRIQUE/RUN_vege_avec_pluie_Fcover_assimil_avec_irri_auto/"
    # name_run="RUNS_SAMIR/RUNS_PARCELLE_GRIGNON/RUN_test/"
    name_run="RUNS_SAMIR/RUN_MULTI_SITE_ICOS/RUN_OPTIMISATION_ICOS/SAMIR_OPTIMI_LAM/RUN_analyse/FAO_init_ru_optim_maxzr_irri_man_soil/"
    name_run_FAO="RUNS_SAMIR/RUN_MULTI_SITE_ICOS/RUN_OPTIMISATION_ICOS/SAMIR_OPTIMI_LAM/RUN_analyse/FAO_init_ru_optim_maxzr_irri_man_soil/"
    name_run_merlin="RUNS_SAMIR/RUN_MULTI_SITE_ICOS/RUN_OPTIMISATION_ICOS/SAMIR_OPTIMI_LAM/RUN_analyse/Merlin_init_ru_optim_maxzr_irri_man_Bruand/"
    d["PC_disk"]="/run/media/pageot/Transcend/Yann_THESE/BESOIN_EAU/BESOIN_EAU/"
    d["PC_home"]="/mnt/d/THESE_TMP/"
    d["PC_home_Wind"]="D:/THESE_TMP/"
    # d["PC_disk"]="H:/Yann_THESE/BESOIN_EAU/BESOIN_EAU/"
    d["PC_labo"]="/datalocal/vboxshare/THESE/BESOIN_EAU/"
    years=['2008','2010']
# =============================================================================
# Validation Flux ETR ICOS non Multi_sie run
# =============================================================================
# modif pour groupby lc

    for y in years:
        for lc in ["maize_irri"]: # maize_rain
            # d['Output_model_PC_labo']='/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/'+name_run+"/"+y+"/"
            # d["Output_model_PC_home"]="/mnt/d/THESE_TMP/TRAITEMENT/"+name_run+"/"+y+"/"
            d["Output_model_PC_home"]="D:/THESE_TMP/TRAITEMENT/"+name_run+"/"+y+"/"
            d["Output_model_PC_home_disk"]="/run/media/pageot/Transcend/Yann_THESE/BESOIN_EAU/BESOIN_EAU/TRAITEMENT/"+name_run+"/"+y+"/"
            d["Output_model_PC_labo"]="/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/"+name_run+"/"+y+"/"
            # d["Output_model_PC_home_disk"]="H:/Yann_THESE/BESOIN_EAU/BESOIN_EAU/TRAITEMENT/"+name_run+"/"+y+"/"
            # if lc == "maize_irri":
            #     SWC=pd.read_csv(d["PC_labo"]+"TRAITEMENT/DATA_VALIDATION/DATA_SWC/SWC_LAM/SWC_LAM_"+str(y)+".csv")
            #     SWC["Date/Time"]=pd.to_datetime(SWC["Date/Time"],format="%Y-%m-%d")
            #     meteo=pd.read_csv(d["PC_labo"]+"TRAITEMENT/INPUT_DATA/DATA_METEO_BV/PARCELLE_LAM/meteo_lam_"+str(y)+".csv",decimal=".")
            #     meteo.date=pd.to_datetime(meteo.date,format="%Y-%m-%d")
            # else:
            #     SWC=pd.read_csv(d["PC_labo"]+"TRAITEMENT/DATA_VALIDATION/DATA_SWC/SWC_GRI/SWC_GRI_2019.csv")
            #     SWC["date"]=pd.to_datetime(SWC["date"],format="%Y-%m-%d")
            #     meteo=pd.read_csv(d["PC_labo"]+"TRAITEMENT/INPUT_DATA/DATA_METEO_BV/PARCELLE_GRI/meteo_gri_2019.csv",decimal=".")
            #     meteo.date=pd.to_datetime(meteo.date,format="%Y-%m-%d")
# =============================================================================
#             Utilisation des flux corrigées
# =============================================================================
            ETR=pd.read_csv(d["PC_disk"]+"/TRAITEMENT/DATA_VALIDATION/DATA_ETR_CESBIO/DATA_ETR_corr_"+str(lc)+"/ETR_"+str(lc)+str(y)+".csv",decimal='.',sep=",")
            ETR["date"]=pd.to_datetime(ETR["date"],format="%Y-%m-%d")
            ETR_obs=ETR.loc[(ETR.date >= str(y)+"-04-01") &(ETR.date <= str(y)+"-09-30")]
            # flux non corrigés
            # ETR_nn=pd.read_csv(d["PC_labo"]+"/TRAITEMENT/DATA_VALIDATION/DATA_ETR_CESBIO/DATA_ETR_"+str(lc)+"/ETR_"+str(lc)+"_"+str(y)+".csv",decimal='.',sep=",")
            # ETR_nn["date"]=pd.to_datetime(ETR_nn["date"],format="%Y-%m-%d")
            # ETR_obs_nn=ETR_nn.loc[(ETR_nn.date >= str(y)+"-04-01") &(ETR_nn.date <= str(y)+"-09-30")]
            # Flux non corrigés
            # ETR_mod=pd.read_csv(d["Output_model_PC_home_disk"][:-5]+"/LUT_ETR"+str(y)+".csv",index_col=[0,1])
            # ETR_mod.columns=pd.to_datetime(ETR_mod.columns,format="%Y-%m-%d")
            # ETR_mod=ETR_mod.loc[:,(ETR_mod.columns >= str(y)+"-04-01") &(ETR_mod.columns <= str(y)+"-09-30")]

            # # Creation plot
            # ### plot dynamique 
            # plt.figure(figsize=(7,7))
            # plt.plot(ETR_obs.date,ETR_obs.LE_Bowen,label='ETR_obs',color="black",linewidth=1)
            # plt.plot(ETR_mod.T.index,ETR_mod.mean(),label='ETR_mod_moyenne',linewidth=1)# récupération mode 
            # plt.fill_between(ETR_mod.T.index, ETR_mod.min(),ETR_mod.max(),alpha=0.5,facecolor="red")
            # plt.ylabel("ETR")
            # plt.ylim(0,10)
            # plt.title("Dynamique ETR obs et ETR mod %s en %s"%(lc,y))
            # plt.legend()
            # plt.savefig(d["Output_model_PC_home_disk"]+"/plt_Incertitude_REW_ETR_mod_%s_%s.png"%(lc,y),dpi=330)

            # #### plot dyna cum
            # plt.figure(figsize=(7,7))
            # plt.plot(ETR_obs.date,ETR_obs.LE_Bowen.cumsum(),label='ETR_obs',color="black")
            # plt.fill_between(ETR_mod.T.index, ETR_mod.min().cumsum(),ETR_mod.max().cumsum(),alpha=0.2,facecolor="red")
            # plt.plot(ETR_mod.T.index,ETR_mod.mean().cumsum(),label='ETR_mod',color='red')
            # plt.text(ETR_obs.date.iloc[-1], ETR_obs.LE_Bowen.cumsum().iloc[-1], s=round(ETR_obs.LE_Bowen.cumsum().iloc[-1],2))
            # plt.text(ETR_mod.T.index[-1], ETR_mod.mean().cumsum().iloc[-1], s=round(ETR_mod.mean().cumsum().iloc[-1],2))
            # plt.ylabel("ETR")
            # plt.ylim(0,700)
            # plt.title("Dynamique ETR obs et ETR mod %s en %s"%(lc,y))
            # plt.legend()
            
            # plt.savefig(d["Output_model_PC_home_disk"]+"/plt_Dynamique_ETR_obs_ETR_mod_cumul_%s_%s.png"%(lc,y))
            ###########" Dynamique week #############
           # Modification récuper max et min 
            # plt.fill_between(ETR_mod.T.index, ETR_mod.xs(500.0,level=1).min().cumsum(),ETR_mod.xs(2500.0,level=1).max().cumsum(),alpha=0.2,facecolor="red")
            # =============================================================================
        # Moyenne glissante
        # =============================================================================
            # ETR_rolling=ETR_obs.rolling(5).mean()
            # ETR_rolling["date"]=ETR_obs.date
            # ETR_mod_rolling=ETR_mod.T.rolling(5).mean()
            # ETR_mod_rolling=ETR_mod_rolling.T
            # plt.figure(figsize=(7,7))
            # plt.plot(ETR_rolling.date,ETR_rolling.LE_Bowen,label='ETR_obs',color="black",linewidth=1)
            # plt.plot(ETR_mod_rolling.T.index,ETR_mod_rolling.mean(),label='ETR_mod_moyenne',linewidth=1)# récupération mode 

            # plt.fill_between(ETR_mod_rolling.T.index, ETR_mod_rolling.min(),ETR_mod_rolling.max(),alpha=0.5,facecolor="red")
            # plt.ylabel("ETR")
            # plt.ylim(0,10)
            # plt.title("Dynamique ETR obs et ETR mod moyenne glissante %s en %s"%(lc,y))
            # plt.legend()
            # plt.savefig(d["Output_model_PC_home_disk"]+"/plt_Incertitude_REW_ETR_mod_mean_rolling_%s_%s.png"%(lc,y),dpi=330)

            # plt.savefig(d["Output_model_PC_labo_disk"]+"/plt_Dynamique_ETR_obs_ETR_mod_%s_%s.png"%(lc,y))
            #### plot dyna cum
            # plt.figure(figsize=(7,7))
            # plt.plot(ETR_rolling.date,ETR_rolling.LE_Bowen.cumsum(),label='ETR_obs',color="black")
            # plt.fill_between(ETR_mod_rolling.T.index, ETR_mod_rolling.min().cumsum(),ETR_mod_rolling.max().cumsum(),alpha=0.2,facecolor="red")
            # # for REW in np.arange(0,16,1):
            # #     print(float(REW))
            # #     plt.fill_between(ETR_mod.T.index, ETR_mod.T[float(REW),500.0].cumsum(),ETR_mod.T[float(REW),2500.0].cumsum(),alpha=0.2,facecolor="red")
            # plt.plot(ETR_mod_rolling.T.index,ETR_mod_rolling.mean().cumsum(),label='ETR_mod',color='red')
            # plt.text(ETR_rolling.date.iloc[-1], ETR_rolling.LE_Bowen.cumsum().iloc[-1], s=round(ETR_rolling.LE_Bowen.cumsum().iloc[-1],2))
            # plt.text(ETR_mod_rolling.T.index[-1], ETR_mod_rolling.mean().cumsum().iloc[-1], s=round(ETR_mod_rolling.mean().cumsum().iloc[-1],2))
            # plt.ylabel("ETR")
            # plt.title("Dynamique ETR obs et ETR mod moyenne glissante %s en %s"%(lc,y))
            # plt.legend()
            # plt.savefig(d["Output_model_PC_labo_disk"]+"/plt_Dynamique_ETR_obs_ETR_mod_cumul_mean_rolling_%s_%s.png"%(lc,y))
# =============================================================================
#             Scatter plot
# =============================================================================
            # tmp=ETR_rolling.LE_Bowen.dropna()        
            # tmp_mod=ETR_mod_rolling.mean().dropna()
            # slope, intercept, r_value, p_value, std_err = stats.linregress(tmp.to_list(),tmp_mod.to_list())
            # bias=1/tmp.shape[0]*sum(np.mean(tmp_mod)-tmp) 
            # fitLine = predict(tmp_mod)
            # # Creation plot
            # plt.figure(figsize=(7,7))
            # plt.plot([0.0, 10], [0.0,10], 'black', lw=1,linestyle='--')
            # plt.xlabel("ETR OBS")
            # plt.ylabel("ETR model")
            # plt.xlim(0,10)
            # plt.ylim(0,10)
            # plt.scatter(ETR_rolling.LE_Bowen, ETR_mod_rolling.mean(), zorder = 2)
            # plt.errorbar(ETR_rolling.LE_Bowen, ETR_mod_rolling.mean(), xerr = None,yerr = ETR_mod_rolling.std(),
            #                 fmt = 'none', capsize = 0, ecolor = 'red', zorder = 1)
            # plt.title("Scatter ETR %s"%y)
            # rms = mean_squared_error(tmp,tmp_mod)
            # plt.text(8,min(tmp)+0.1,"RMSE = "+str(round(rms,2))) 
            # plt.text(8,min(tmp)+0.4,"R² = "+str(round(r_value,2)))
            # plt.text(8,min(tmp)+0.7,"Pente = "+str(round(slope,2)))
            # plt.text(8,min(tmp)+1,"Biais = "+str(round(bias,2)))
            # plt.savefig(d["Output_model_PC_home_disk"]+"/scatter_Incertitude_ETR_mod_mean_rolling_%s_%s.png"%(lc,y),dpi=330)
            
            
#    hypothèse irrigation validation 
# =============================================================================
            # récupération num_run max incertitude
            # diff=ETR_mod_rolling.max()-ETR_mod_rolling.min()
            # datefor=diff.idxmax() # forte incertitude a cette date
            # coup_parm=ETR_mod_rolling.T.loc[ETR_mod_rolling.columns==datefor].T.idxmax() # récupération couple param max 
            # coup_parm_min=ETR_mod_rolling.T.loc[ETR_mod_rolling.columns==datefor].T.idxmin()
            # #  lecture des output_LUT
            # dfmax=pickle.load(open(d["PC_disk"]+"/TRAITEMENT/"+name_run+"/"+str(y)+"/Output/maxZr/output_test_maize_irri_"+str(coup_parm.iloc[0][0])+".df","rb"))
            # dfmin=pickle.load(open(d["PC_disk"]+"/TRAITEMENT/"+name_run+"/"+str(y)+"/Output/maxZr/output_test_maize_irri_"+str(coup_parm_min.iloc[0][0])+".df","rb"))
            # print(y)
            # if "irri_man" in name_run:
            #     print(dfmax.loc[dfmax['Irrig']> 0.0][["date","Irrig"]])
            #     print(dfmin.loc[dfmin['Irrig']> 0.0][["date","Irrig"]])
            #     plt.figure(figsize=(7,7))
            #     plt.plot(ETR_rolling.date,ETR_rolling.LE_Bowen,label='ETR_obs',color="black",linewidth=1)
            #     plt.plot(ETR_mod_rolling.T.index,ETR_mod_rolling.mean(),label='ETR_mod_moyenne',linewidth=1)# récupération mode 
            #     plt.fill_between(ETR_mod_rolling.T.index, ETR_mod_rolling.min(),ETR_mod_rolling.max(),alpha=0.5,facecolor="red")
            #     plt.bar(dfmax.loc[(dfmax.date >= str(y)+'-04-01')&(dfmax.date<=str(y)+"-09-30")]["date"],dfmax.loc[(dfmax.date >= str(y)+'-04-01')&(dfmax.date<=str(y)+"-09-30")]["Irrig"]/10,label= "Irrigation",width=1)
            #     # plt.bar(dfmin.loc[(dfmin.date >= '2008-04-01')&(dfmin.date<="2008-09-30")]["date"],dfmin.loc[(dfmax.date >= '2008-04-01')&(dfmin.date<="2008-09-30")]["Ir_auto"]/10,label= "Irrigation")
            #     plt.ylabel("ETR")
            #     plt.ylim(0,10)
            #     plt.title("Dynamique ETR obs et ETR mod moyenne glissante %s en %s"%(lc,y))
            #     plt.legend()
            #     ax2=plt.twinx(ax=None)
            #     ax2.plot(dfmax.loc[(dfmax.date >= str(y)+'-04-01')&(dfmax.date<=str(y)+"-09-30")]["date"],dfmax.loc[(dfmax.date >= str(y)+'-04-01')&(dfmax.date<=str(y)+"-09-30")]["NDVI"],color="darkgreen",linestyle="--")
            #     ax2.set_ylabel("NDVI")
            #     ax2.set_ylim(0,1)
            #     plt.savefig(d["Output_model_PC_home_disk"]+"/plt_Incertitude_REW_ETR_mod_mean_rolling__irrigation%s_%s.png"%(lc,y),dpi=330)
            # else:
            #     print("ici")
            #     print(dfmax.loc[dfmax['Ir_auto']> 0.0][["date","Ir_auto"]])
            #     print(dfmin.loc[dfmin['Ir_auto']> 0.0][["date","Ir_auto"]])
            #     plt.figure(figsize=(7,7))
            #     plt.plot(ETR_rolling.date,ETR_rolling.LE_Bowen,label='ETR_obs',color="black",linewidth=1)
            #     plt.plot(ETR_mod_rolling.T.index,ETR_mod_rolling.mean(),label='ETR_mod_moyenne',linewidth=1)# récupération mode 
            #     plt.fill_between(ETR_mod_rolling.T.index, ETR_mod_rolling.min(),ETR_mod_rolling.max(),alpha=0.5,facecolor="red")
            #     plt.bar(dfmax.loc[(dfmax.date >= str(y)+'-04-01')&(dfmax.date<=str(y)+"-09-30")]["date"],dfmax.loc[(dfmax.date >= str(y)+'-04-01')&(dfmax.date<=str(y)+"-09-30")]["Ir_auto"]/10,label= "Irrigation_max",width=1,color="red")
            #     plt.bar(dfmin.loc[(dfmin.date >= str(y)+'-04-01')&(dfmin.date<=str(y)+"-09-30")]["date"],dfmin.loc[(dfmin.date >= str(y)+'-04-01')&(dfmin.date<=str(y)+"-09-30")]["Ir_auto"]/10,label= "Irrigation_min",width=1,color="green")
            #     # plt.bar(dfmin.loc[(dfmin.date >= '2008-04-01')&(dfmin.date<="2008-09-30")]["date"],dfmin.loc[(dfmax.date >= '2008-04-01')&(dfmin.date<="2008-09-30")]["Ir_auto"]/10,label= "Irrigation")
            #     plt.ylabel("ETR")
            #     plt.ylim(0,10)
            #     plt.title("Dynamique ETR obs et ETR mod moyenne glissante %s en %s"%(lc,y))
            #     plt.legend()
            #     ax2=plt.twinx(ax=None)
            #     ax2.plot(dfmax.loc[(dfmax.date >= str(y)+'-04-01')&(dfmax.date<=str(y)+"-09-30")]["date"],dfmax.loc[(dfmax.date >= str(y)+'-04-01')&(dfmax.date<=str(y)+"-09-30")]["NDVI"],color="darkgreen",linestyle="--")
            #     ax2.set_ylabel("NDVI")
            #     ax2.set_ylim(0,1)
            #     plt.savefig(d["Output_model_PC_home_disk"]+"/plt_Incertitude_REW_ETR_mod_mean_rolling__irrigation%s_%s.png"%(lc,y),dpi=330)


# =============================================================================
# Test incertitude PF_cc et Fcover
# =============================================================================
    print("============")
    print ("Incertitude plot")
    print("============")

    d["Output_model_save_fig"]="/run/media/pageot/Transcend/Yann_THESE/BESOIN_EAU/BESOIN_EAU/TRAITEMENT/RUNS_SAMIR/RUN_MULTI_SITE_ICOS/RUN_OPTIMISATION_ICOS/SAMIR_OPTIMI_LAM/RUN_analyse/Merlin_init_ru_optim_maxzr_irri_man_Bruand/"
    ETR_test_m20=None
    ETR_test_pl20=None
    ETR_test=None
    for y in ['2008','2010',"2012","2014","2015","2019"]:
        # for Fco in ["_pl20","_m20",""]:
        ####  récupération num_run max incertitude
            # d["Output_model_PC_home_disk"]="/run/media/pageot/Transcend/Yann_THESE/BESOIN_EAU/BESOIN_EAU/TRAITEMENT/RUNS_SAMIR/RUN_MULTI_SITE_ICOS/RUN_OPTIMISATION_ICOS/SAMIR_OPTIMI_LAM/RUN_analyse/Merlin_init_ru_optim_Fcover"+Fco+"_maxzr_irri_man_Bruand"
        d["Output_model_PC_home_disk"]="/run/media/pageot/Transcend/Yann_THESE/BESOIN_EAU/BESOIN_EAU/TRAITEMENT/RUNS_SAMIR/RUN_MULTI_SITE_ICOS/RUN_OPTIMISATION_ICOS/SAMIR_OPTIMI_LAM/RUN_analyse/Merlin_init_ru_optim_maxzr_irri_man_Bruand"
        ETR_mod=pd.read_csv(d["Output_model_PC_home_disk"]+"/LUT_ETR"+str(y)+".csv",index_col=[0,1])
        ETR_mod.columns=pd.to_datetime(ETR_mod.columns,format="%Y-%m-%d")
        ETR_mod=ETR_mod.loc[:,(ETR_mod.columns >= str(y)+"-04-01") &(ETR_mod.columns <= str(y)+"-09-30")]
        ETR_mod_max=pd.read_csv(d["Output_model_PC_home_disk"]+"_max"+"/LUT_ETR"+str(y)+".csv",index_col=[0,1])
        ETR_mod_max.columns=pd.to_datetime(ETR_mod_max.columns,format="%Y-%m-%d")
        ETR_mod_max=ETR_mod_max.loc[:,(ETR_mod_max.columns >= str(y)+"-04-01") &(ETR_mod_max.columns <= str(y)+"-09-30")]
        ETR_mod_min=pd.read_csv(d["Output_model_PC_home_disk"]+"_min"+"/LUT_ETR"+str(y)+".csv",index_col=[0,1])
        ETR_mod_min.columns=pd.to_datetime(ETR_mod_min.columns,format="%Y-%m-%d")
        ETR_mod_min=ETR_mod_min.loc[:,(ETR_mod_min.columns >= str(y)+"-04-01") &(ETR_mod_min.columns <= str(y)+"-09-30")]
        # globals()["ETR_test%s"%(Fco)]=pd.concat([ETR_mod_max,ETR_mod,ETR_mod_min])
    # ETR_test_merlin=pd.concat([ETR_test_pl20,ETR_test_m20,ETR_test])
        ETR_test_merlin=pd.concat([ETR_mod_max,ETR_mod,ETR_mod_min])
        # FAO
        d["Output_model_PC_home_disk"]="/run/media/pageot/Transcend/Yann_THESE/BESOIN_EAU/BESOIN_EAU/TRAITEMENT/RUNS_SAMIR/RUN_MULTI_SITE_ICOS/RUN_OPTIMISATION_ICOS/SAMIR_OPTIMI_LAM/RUN_analyse/FAO_init_ru_optim_maxzr_irri_man_soil"
        ETR_mod=pd.read_csv(d["Output_model_PC_home_disk"]+"/LUT_ETR"+str(y)+".csv",index_col=[0,1])
        ETR_mod.columns=pd.to_datetime(ETR_mod.columns,format="%Y-%m-%d")
        ETR_mod=ETR_mod.loc[:,(ETR_mod.columns >= str(y)+"-04-01") &(ETR_mod.columns <= str(y)+"-09-30")]
        ETR_mod_max=pd.read_csv(d["Output_model_PC_home_disk"]+"_max"+"/LUT_ETR"+str(y)+".csv",index_col=[0,1])
        ETR_mod_max.columns=pd.to_datetime(ETR_mod_max.columns,format="%Y-%m-%d")
        ETR_mod_max=ETR_mod_max.loc[:,(ETR_mod_max.columns >= str(y)+"-04-01") &(ETR_mod_max.columns <= str(y)+"-09-30")]
        ETR_mod_min=pd.read_csv(d["Output_model_PC_home_disk"]+"_min"+"/LUT_ETR"+str(y)+".csv",index_col=[0,1])
        ETR_mod_min.columns=pd.to_datetime(ETR_mod_min.columns,format="%Y-%m-%d")
        ETR_mod_min=ETR_mod_min.loc[:,(ETR_mod_min.columns >= str(y)+"-04-01") &(ETR_mod_min.columns <= str(y)+"-09-30")]
        # globals()["ETR_test%s"%(Fco)]=pd.concat([ETR_mod_max,ETR_mod,ETR_mod_min])
        ETR_test_FAO=pd.concat([ETR_mod_max,ETR_mod,ETR_mod_min])
        
        
        # ETR_test=pd.concat([ETR_test_pl20,ETR_test,ETR_test_m20]) # Ecrire en dur ce tableau (FAO / Merlin)
        ETR=pd.read_csv(d["PC_disk"]+"/TRAITEMENT/DATA_VALIDATION/DATA_ETR_CESBIO/DATA_ETR_corr_maize_irri/ETR_maize_irri"+str(y)+".csv",decimal='.',sep=",")
        ETR["date"]=pd.to_datetime(ETR["date"],format="%Y-%m-%d")
        ETR_obs=ETR.loc[(ETR.date >= str(y)+"-04-01") &(ETR.date <= str(y)+"-09-30")]
        ETR_rolling=ETR_obs.rolling(5).mean()
        ETR_rolling["date"]=ETR_obs.date
        # Merlin
        ETR_test_rolling=ETR_test_merlin.T.rolling(5).mean()
        ETR_test_rolling=ETR_test_rolling.T
        diff=ETR_test_rolling.max()-ETR_test_rolling.min()
        datefor=diff.idxmax() # forte incertitude a cette date
        coup_parm=ETR_test_rolling.T.loc[ETR_test_rolling.columns==datefor].T.idxmax() # récupération couple param max 
        coup_parm_min=ETR_test_rolling.T.loc[ETR_test_rolling.columns==datefor].T.idxmin()
        #  lecture des output_LUT
        dfmax=pickle.load(open(d["PC_disk"]+"/TRAITEMENT/"+name_run_merlin+"/"+str(y)+"/Output/maxZr/output_test_maize_irri_"+str(coup_parm.iloc[0][0])+".df","rb"))
        dfmin=pickle.load(open(d["PC_disk"]+"/TRAITEMENT/"+name_run_merlin+"/"+str(y)+"/Output/maxZr/output_test_maize_irri_"+str(coup_parm_min.iloc[0][0])+".df","rb"))
        dfmax.date=pd.to_datetime(dfmax.date,format="%Y-%m-%d")
        dfmin.date=pd.to_datetime(dfmin.date,format="%Y-%m-%d")
        dfmax=dfmax.loc[(dfmax.date>= str(y)+"-04-01") &(dfmax.date <= str(y)+"-09-30")]
        dfmin=dfmin.loc[(dfmin.date>= str(y)+"-04-01") &(dfmin.date <= str(y)+"-09-30")]
        KS=pd.concat([dfmax,dfmin])
        coeff_ks=KS.groupby("date").mean()["Ks"]
        # FAO 
        ETR_test_FAO_rolling=ETR_test_FAO.T.rolling(5).mean()
        ETR_test_FAO_rolling=ETR_test_FAO_rolling.T
        diff=ETR_test_FAO_rolling.max()-ETR_test_FAO_rolling.min()
        datefor=diff.idxmax() # forte incertitude a cette date
        coup_parm=ETR_test_FAO_rolling.T.loc[ETR_test_FAO_rolling.columns==datefor].T.idxmax() # récupération couple param max 
        coup_parm_min=ETR_test_FAO_rolling.T.loc[ETR_test_FAO_rolling.columns==datefor].T.idxmin()
        #  lecture des output_LUT
        dfmax=pickle.load(open(d["PC_disk"]+"/TRAITEMENT/"+name_run_FAO+"/"+str(y)+"/Output/maxZr/output_test_maize_irri_"+str(coup_parm.iloc[0][0])+".df","rb"))
        dfmin=pickle.load(open(d["PC_disk"]+"/TRAITEMENT/"+name_run_FAO+"/"+str(y)+"/Output/maxZr/output_test_maize_irri_"+str(coup_parm_min.iloc[0][0])+".df","rb"))
        dfmax.date=pd.to_datetime(dfmax.date,format="%Y-%m-%d")
        dfmin.date=pd.to_datetime(dfmin.date,format="%Y-%m-%d")
        dfmax=dfmax.loc[(dfmax.date>= str(y)+"-04-01") &(dfmax.date <= str(y)+"-09-30")]
        dfmin=dfmin.loc[(dfmin.date>= str(y)+"-04-01") &(dfmin.date <= str(y)+"-09-30")]
        KS=pd.concat([dfmax,dfmin])
        coeff_ks=KS.groupby("date").mean()["Ks"]
        # plot
        plt.figure(figsize=(7,7))
        plt.plot(ETR_rolling.date,ETR_rolling.LE_Bowen,label='ETR_obs',color="black",linewidth=1)
        plt.plot(ETR_test_rolling.T.index,ETR_test_rolling.mean(),label='ETR_mod_FcoverBvnet',linewidth=1,color='red')# récupération mode 
        plt.fill_between(ETR_test_rolling.T.index, ETR_test_rolling.min(),ETR_test_rolling.max(),alpha=0.2,facecolor="red")
        plt.plot(ETR_test_FAO_rolling.T.index,ETR_test_FAO_rolling.mean(),label='ETR_mod_FcoverSamir',linewidth=1,color="Blue")# récupération mode 
        plt.fill_between(ETR_test_FAO_rolling.T.index, ETR_test_FAO_rolling.min(),ETR_test_FAO_rolling.max(),alpha=0.2,facecolor="Blue")
        plt.ylabel("ETR")
        plt.ylim(0,10)
        plt.legend()
        # ax2 = plt.twinx()
        # ax2.plot(coeff_ks.index,coeff_ks,color='r',linestyle="--",label="Ks")
        # ax2.set_ylim(-5,1)
        # ax2.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(6))
        plt.title("Dynamique ETR moyenne PFCC incertitude %s en %s"%(lc,y))
        plt.savefig(d["Output_model_save_fig"]+"/plt_Incertitude_PFCC_ETR_mod_mean_rolling_%s_%s.png"%(lc,y),dpi=330)
        
        # Cumul
        plt.figure(figsize=(7,7))
        plt.plot(ETR_obs.date,ETR_obs.LE_Bowen.cumsum(),label='ETR_obs',color="black")
        plt.fill_between(ETR_test_merlin.T.index, ETR_test_merlin.min().cumsum(),ETR_test_merlin.max().cumsum(),alpha=0.2,facecolor="red")
        plt.plot(ETR_test_merlin.T.index,ETR_test_merlin.mean().cumsum(),label='ETR_mod',color='red')
        plt.text(ETR_obs.date.iloc[-1], ETR_obs.LE_Bowen.cumsum().iloc[-1], s=round(ETR_obs.LE_Bowen.cumsum().iloc[-1],2))
        plt.text(ETR_test_merlin.T.index[-1], ETR_test_merlin.mean().cumsum().iloc[-1], s=round(ETR_test_merlin.mean().cumsum().iloc[-1],2))
        plt.fill_between(ETR_test_FAO.T.index, ETR_test_FAO.min().cumsum(),ETR_test_FAO.max().cumsum(),alpha=0.2,facecolor="red")
        plt.plot(ETR_test_FAO.T.index,ETR_test_FAO.mean().cumsum(),label='ETR_mod',color='red')
        plt.text(ETR_test_FAO.T.index[-1], ETR_test_FAO.mean().cumsum().iloc[-1], s=round(ETR_test_FAO.mean().cumsum().iloc[-1],2))
        plt.ylabel("ETR")
        plt.ylim(0,700)
        plt.title("Dynamique ETR cumul incertitude %s en %s"%(lc,y))
        plt.legend()
        plt.savefig(d["Output_model_save_fig"]+"/plt_Dynamique_ETR_ETR_cumul_incertitude_PFCC_%s_%s.png"%(lc,y))
        
        
        # Scatter
        tmp=ETR_rolling.LE_Bowen.dropna()        
        tmp_mod=ETR_test_rolling.mean().dropna()
        tmp_mod_FAO=ETR_test_FAO_rolling.mean().dropna()
        slope, intercept, r_value, p_value, std_err = stats.linregress(tmp_mod_FAO.to_list(),tmp_mod.to_list())
        bias=1/tmp_mod_FAO.shape[0]*sum(np.mean(tmp_mod)-tmp_mod_FAO) 
        fitLine = predict(tmp_mod)
        # Creation plot
        plt.figure(figsize=(7,7))
        plt.plot([0.0, 10], [0.0,10], 'black', lw=1,linestyle='--')
        plt.xlabel("ETR Fcov Samir")
        plt.ylabel("ETR Fcov BVNet")
        plt.xlim(0,10)
        plt.ylim(0,10)
        plt.scatter(ETR_test_FAO_rolling.mean(), ETR_test_rolling.mean(), zorder = 2)
        plt.errorbar(ETR_test_FAO_rolling.mean(), ETR_test_rolling.mean(), xerr = ETR_test_FAO_rolling.std(),yerr = ETR_test_rolling.std(),
                        fmt = 'none', capsize = 0, ecolor = 'red', zorder = 1)
        plt.title("Scatter ETR %s"%y)
        rms = mean_squared_error(tmp,tmp_mod)
        plt.text(8,min(tmp_mod_FAO)+0.1,"RMSE = "+str(round(rms,2))) 
        plt.text(8,min(tmp_mod_FAO)+0.4,"R² = "+str(round(r_value,2)))
        plt.text(8,min(tmp_mod_FAO)+0.7,"Pente = "+str(round(slope,2)))
        plt.text(8,min(tmp_mod_FAO)+1,"Biais = "+str(round(bias,2)))
        plt.savefig(d["Output_model_save_fig"]+"/scatter_Incertitude_ETR_mod_mean_rolling_%s_%s.png"%(lc,y),dpi=330)
    
        
        
# =============================================================================
#         Kei merlin vs Kei FAO 
# =============================================================================
        # name_run_FAO="RUNS_SAMIR/RUN_MULTI_SITE_ICOS/RUN_OPTIMISATION_ICOS/SAMIR_OPTIMI_LAM/FAO_init_ru_optim_maxzr_irri_man"
        # name_run_Merlin="RUNS_SAMIR/RUN_MULTI_SITE_ICOS/RUN_OPTIMISATION_ICOS/SAMIR_OPTIMI_LAM/Merlin_init_ru_optim_maxzr_new_irri_man_Bruand/"
        # dfmax_FAO=pickle.load(open(d["PC_disk"]+"/TRAITEMENT/"+name_run_FAO+"/"+str(y)+"/Output/maxZr/output_test_maize_irri_"+str(coup_parm.iloc[0][0])+".df","rb"))
        # dfmin_FAO=pickle.load(open(d["PC_disk"]+"/TRAITEMENT/"+name_run_FAO+"/"+str(y)+"/Output/maxZr/output_test_maize_irri_"+str(coup_parm_min.iloc[0][0])+".df","rb"))
        # dfmax_Merlin=pickle.load(open(d["PC_disk"]+"/TRAITEMENT/"+name_run_Merlin+"/"+str(y)+"/Output/maxZr/output_test_maize_irri_"+str(coup_parm.iloc[0][0])+".df","rb"))
        # dfmin_Merlin=pickle.load(open(d["PC_disk"]+"/TRAITEMENT/"+name_run_Merlin+"/"+str(y)+"/Output/maxZr/output_test_maize_irri_"+str(coup_parm_min.iloc[0][0])+".df","rb"))
        # Ke_FAO=pd.concat([dfmax_FAO,dfmin_FAO])
        # coeff_ke_FAO=Ke_FAO.groupby("date").mean()[["Kei","NDVI"]]
        # Ke_Merlin=pd.concat([dfmax_Merlin,dfmin_Merlin])
        # coeff_ke_Merlin=Ke_Merlin.groupby("date").mean()["Kei"]
        
        # plt.figure(figsize=(7,7))
        # plt.plot(coeff_ke_FAO.index,coeff_ke_FAO.Kei,label='Ke_FAO',color="black",linewidth=1)
        # plt.plot(coeff_ke_Merlin.index,coeff_ke_Merlin,label='Ke_Merlin',linewidth=1)# récupération mode 
        # plt.plot(coeff_ke_Merlin.index,coeff_ke_FAO.NDVI,label='NDVI',linewidth=1)# récupération mode 
        # plt.legend()
        # plt.ylim(0, 1.5)
        # plt.title("Dynamique Kei")
        # plt.savefig(d["Output_model_PC_home_disk"]+"/Dyan_Kei_FAO_Merlin_%s_%s.png"%(lc,y),dpi=330)
# =============================================================================
#         Irrigation_auto vs man
# =============================================================================
# =============================================================================
#         # name_run_auto_max="RUNS_SAMIR/RUN_MULTI_SITE_ICOS/RUN_OPTIMISATION_ICOS/SAMIR_OPTIMI_LAM/Merlin_init_ru_optim_maxzr_Fcover_pl20_new_irri_auto_Bruand_max/"
#         # name_run_auto_min="RUNS_SAMIR/RUN_MULTI_SITE_ICOS/RUN_OPTIMISATION_ICOS/SAMIR_OPTIMI_LAM/Merlin_init_ru_optim_maxzr_Fcover_m20_new_irri_auto_Bruand_min/"
#         # name_run_man="RUNS_SAMIR/RUN_MULTI_SITE_ICOS/RUN_OPTIMISATION_ICOS/SAMIR_OPTIMI_LAM/Merlin_init_ru_optim_maxzr_Fcover_new_irri_man_Bruand/"
#         # # name_run_auto_max2="RUNS_SAMIR/RUN_MULTI_SITE_ICOS/RUN_OPTIMISATION_ICOS/SAMIR_OPTIMI_LAM/Merlin_init_ru_optim_maxzr_Fcover_new_irri_auto_Bruand_max/"
#         # # name_run_auto_min2="RUNS_SAMIR/RUN_MULTI_SITE_ICOS/RUN_OPTIMISATION_ICOS/SAMIR_OPTIMI_LAM/Merlin_init_ru_optim_maxzr_Fcover_new_irri_auto_Bruand_min/"
#         # # name_run_auto_max3="RUNS_SAMIR/RUN_MULTI_SITE_ICOS/RUN_OPTIMISATION_ICOS/SAMIR_OPTIMI_LAM/Merlin_init_ru_optim_maxzr_Fcover_m20_new_irri_auto_Bruand_max/"
#         # # name_run_auto_min3="RUNS_SAMIR/RUN_MULTI_SITE_ICOS/RUN_OPTIMISATION_ICOS/SAMIR_OPTIMI_LAM/Merlin_init_ru_optim_maxzr_Fcover_pl20_new_irri_auto_Bruand_min/"
#         
#         # # dfmax_2=pickle.load(open(d["PC_disk"]+"/TRAITEMENT/"+name_run_auto_max2+"/"+str(y)+"/Output/maxZr/output_test_maize_irri_"+str(coup_parm.iloc[0][0])+".df","rb"))
#         # # dfmax_3=pickle.load(open(d["PC_disk"]+"/TRAITEMENT/"+name_run_auto_min2+"/"+str(y)+"/Output/maxZr/output_test_maize_irri_"+str(coup_parm.iloc[0][0])+".df","rb"))
#         # # dfmax_4=pickle.load(open(d["PC_disk"]+"/TRAITEMENT/"+name_run_auto_max3+"/"+str(y)+"/Output/maxZr/output_test_maize_irri_"+str(coup_parm.iloc[0][0])+".df","rb"))
#         # # dfmax_5=pickle.load(open(d["PC_disk"]+"/TRAITEMENT/"+name_run_auto_max3+"/"+str(y)+"/Output/maxZr/output_test_maize_irri_"+str(coup_parm.iloc[0][0])+".df","rb"))
#         
#         
# 
#         # dfmax_Man=pickle.load(open(d["PC_disk"]+"/TRAITEMENT/"+name_run_man+"/"+str(y)+"/Output/maxZr/output_test_maize_irri_"+str(coup_parm.iloc[0][0])+".df","rb"))
#         # dfmax_Max=pickle.load(open(d["PC_disk"]+"/TRAITEMENT/"+name_run_auto_max+"/"+str(y)+"/Output/maxZr/output_test_maize_irri_"+str(coup_parm.iloc[0][0])+".df","rb"))
#         # dfmin_Max=pickle.load(open(d["PC_disk"]+"/TRAITEMENT/"+name_run_auto_max+"/"+str(y)+"/Output/maxZr/output_test_maize_irri_"+str(coup_parm_min.iloc[0][0])+".df","rb"))
#         # dfmax_Min=pickle.load(open(d["PC_disk"]+"/TRAITEMENT/"+name_run_auto_min+"/"+str(y)+"/Output/maxZr/output_test_maize_irri_"+str(coup_parm.iloc[0][0])+".df","rb"))
#         # dfmin_Min=pickle.load(open(d["PC_disk"]+"/TRAITEMENT/"+name_run_auto_min+"/"+str(y)+"/Output/maxZr/output_test_maize_irri_"+str(coup_parm_min.iloc[0][0])+".df","rb"))
#         # Irr_max=pd.concat([dfmax_Max,dfmin_Max])
#         # coeff_ke_FAO=Irr_max.groupby("date").mean()[["Ir_auto","NDVI"]]
#         # Irr_min=pd.concat([dfmax_Min,dfmin_Min])
#         # coeff_ke_Merlin=Irr_min.groupby("date").mean()[["Ir_auto","NDVI"]]
#         
#         # # test=pd.concat([dfmax_Max,dfmax_2,dfmax_3,dfmax_4,dfmax_5,dfmin_Max,dfmax_Min,dfmin_Min])
#         # # a=test.groupby("date").mean()[["NDVI","Ir_auto"]].sum()
#         # print(y)
#         # # print(a)
#         # print(coeff_ke_FAO["Ir_auto"].sum())
#         # print(coeff_ke_Merlin["Ir_auto"].sum())
#         # print("mean => %s"%(mean([coeff_ke_FAO["Ir_auto"].sum(),coeff_ke_Merlin["Ir_auto"].sum()])))
#         # print("std => %s"%(std([coeff_ke_FAO["Ir_auto"].sum(),coeff_ke_Merlin["Ir_auto"].sum()])))
#         # print(dfmax_Man["Irrig"].sum())
#         
# =============================================================================
        
        # plt.figure(figsize=(7,7))
        # plt.bar(coeff_ke_FAO.index,coeff_ke_FAO.Ir_auto,label='Irrigation',color="black")
        # plt.bar(dfmax_Man.date,dfmax_Man.Irrig,label='Irrigation',color="Blue")
        # # plt.bar(coeff_ke_Merlin.index,coeff_ke_Merlin,label='Irrigation')# récupération mode 
        # plt.legend()
        # # plt.ylim(0, 1.5)
        # plt.title("Irrigation _auto")
        # plt.savefig(d["Output_model_PC_home_disk"]+"/Dyan_Kei_FAO_Merlin_%s_%s.png"%(lc,y),dpi=330)
# =============================================================================
#         Test incertitude PF_CC
# =============================================================================

    # d["Output_model_PC_home_disk"]="H:/Yann_THESE/BESOIN_EAU/BESOIN_EAU/TRAITEMENT/RUNS_SAMIR/RUN_MULTI_SITE_ICOS/RUN_OPTIMISATION_ICOS/SAMIR_OPTIMI_LAM/Merlin_init_ru_optim_maxzr_Fcover_new_irri_man_Bruand"
    # for y in ['2008','2010','2012','2014','2015',"2019"]:
    #     ETR_mod=pd.read_csv(d["Output_model_PC_home_disk"]+"/LUT_ETR"+str(y)+".csv",index_col=[0,1])
    #     ETR_mod.columns=pd.to_datetime(ETR_mod.columns,format="%Y-%m-%d")
    #     ETR_mod=ETR_mod.loc[:,(ETR_mod.columns >= str(y)+"-04-01") &(ETR_mod.columns <= str(y)+"-09-30")]
    #     ETR_mod_max=pd.read_csv(d["Output_model_PC_home_disk"]+"_max"+"/LUT_ETR"+str(y)+".csv",index_col=[0,1])
    #     ETR_mod_max.columns=pd.to_datetime(ETR_mod_max.columns,format="%Y-%m-%d")
    #     ETR_mod_max=ETR_mod_max.loc[:,(ETR_mod_max.columns >= str(y)+"-04-01") &(ETR_mod_max.columns <= str(y)+"-09-30")]
    #     ETR_mod_min=pd.read_csv(d["Output_model_PC_home_disk"]+"_min"+"/LUT_ETR"+str(y)+".csv",index_col=[0,1])
    #     ETR_mod_min.columns=pd.to_datetime(ETR_mod_min.columns,format="%Y-%m-%d")
    #     ETR_mod_min=ETR_mod_min.loc[:,(ETR_mod_min.columns >= str(y)+"-04-01") &(ETR_mod_min.columns <= str(y)+"-09-30")]
    #     ETR_test= pd.concat([ETR_mod_max,ETR_mod,ETR_mod_min]) #♦ Concac max et mod 
    #     ETR=pd.read_csv(d["PC_disk"]+"/TRAITEMENT/DATA_VALIDATION/DATA_ETR_CESBIO/DATA_ETR_corr_maize_irri/ETR_maize_irri"+str(y)+".csv",decimal='.',sep=",")
    #     ETR["date"]=pd.to_datetime(ETR["date"],format="%Y-%m-%d")
    #     ETR_obs=ETR.loc[(ETR.date >= str(y)+"-04-01") &(ETR.date <= str(y)+"-09-30")]
    #     ETR_rolling=ETR_obs.rolling(5).mean()
    #     ETR_rolling["date"]=ETR_obs.date
    #     ETR_test_rolling=ETR_test.T.rolling(5).mean()
    #     ETR_test_rolling=ETR_test_rolling.T
    #     plt.figure(figsize=(7,7))
    #     plt.plot(ETR_rolling.date,ETR_rolling.LE_Bowen,label='ETR_obs',color="black",linewidth=1)
    #     plt.plot(ETR_test_rolling.T.index,ETR_test_rolling.mean(),label='ETR_mod_moyenne',linewidth=1)# récupération mode 
    #     plt.fill_between(ETR_test_rolling.T.index, ETR_test_rolling.min(),ETR_test_rolling.max(),alpha=0.5,facecolor="red")
    #     plt.ylabel("ETR")
    #     plt.ylim(0,10)
    #     plt.title("Dynamique ETR obs et ETR mod moyenne glissante incertitude_PFCC %s en %s"%(lc,y))
    #     plt.legend()
    #     plt.savefig(d["Output_model_PC_home_disk"]+"/plt_Incertitude_PFCC_ETR_mod_mean_rolling_%s_%s.png"%(lc,y),dpi=330)
 # =============================================================================
# #              Isole forte periode incertitude 
# # # =============================================================================
#     ETR_test=ETR_mod.iloc[:,80:120]
#     # plt.figure(figsize=(7,7))
#     plt.plot(ETR_test.T.index,ETR_test.mean(),label='ETR_mod_moyenne',linewidth=1)# récupération mode 
#     plt.fill_between(ETR_test.T.index, ETR_test.min(),ETR_test.max(),alpha=0.5,facecolor="red")
    
# #  permet de connaitre le paramétrage
#     plt.figure(figsize=(7,7))
#     # plt.plot(ETR_obs.date,ETR_obs.LE_Bowen,label='ETR_obs',color="black",linewidth=1)
#     plt.plot(ETR_mod.T.index,ETR_mod.T[7.0,1000.0],label='ETR_mod_moyenne',linewidth=1)# récupération mode 
#     # for REW in np.arange(0,16,1):
#     #     print(float(REW))
#     #     plt.fill_between(ETR_mod.T.index, ETR_mod.T[float(REW),500.0].values,ETR_mod.T[float(REW),2500.0].values,alpha=0.2,facecolor="red")
#     plt.fill_between(ETR_mod.T.index, ETR_mod.xs(800.0,level=1).min(),ETR_mod.xs(1200.0,level=1).max(),alpha=0.5,facecolor="red")
#     # permer d'identifier le couple de parametre donnant le max 
#     ETR_test.idxmin()
    
# =============================================================================
# =============================================================================
#  Validation Irri_ préparation data 
# =============================================================================
    # all_quantity=[]
    # all_number=[]
    # all_id=[]
    # all_runs=[]
    # all_date=[]
    # all_date_jj=[]
    # for r in os.listdir('D:/THESE_TMP/RUNS_SAMIR/RUN_RELATION/TARN_CACG/'):
    #     d["path_PC"]='D:/THESE_TMP/RUNS_SAMIR/RUN_RELATION/TARN_CACG/'+str(r)+'/Inputdata/'
    #     for s in os.listdir(d["path_PC"][:-10]):
    #         if "output" in s:
    #             print (s)
    #             df=pickle.load(open(d["path_PC"][:-10]+str(s),'rb'))
    #             for id in list(set(df.id)):
    #                 lam=df.loc[df.id==id]
    #                 date_irr=lam.loc[lam.Ir_auto!=0.0]["date"]
    #                 size_R=len(date_irr) # vecteur de répétiton de la variable RUNS
    #                 print(r'n° du runs : %s '% r)
    #                 print(r' n° parcelle : %s' %id)
    #                 print(r'sum irrigation in mm : %s'%lam.groupby(["LC","id"])["Ir_auto"].sum()[0])
    #                 print(r' nb irrigation : %s' %lam.Ir_auto.where(df["Ir_auto"] != 0.0).dropna().count())
    #                 all_runs.append(r)
    #                 all_id.append(id)
    #                 all_quantity.append(lam.groupby(["LC","id"])["Ir_auto"].sum()[0])
    #                 all_number.append(lam.Ir_auto.where(df["Ir_auto"] != 0.0).dropna().count())
    #                 all_date.append(date_irr.values)
    #                 for i in date_irr:
    #                     a=i.strftime("%j") # date en jj sur l'année
    #                     all_date_jj.append(a)
    # all_resu=pd.DataFrame([all_runs,all_id,all_quantity,all_number,all_date]).T
    # all_resu.columns=["runs",'id','cumul_irr',"nb_irr","date_irr"]

# =============================================================================
# Validation des Irr cumulées CACG
# =============================================================================
#     vali_cacg=pd.read_csv(d["PC_disk"]+"TRAITEMENT/DONNEES_VALIDATION_SAMIR/merge_parcelle_2017.csv")
# #    vali_cacg.dropna(subset=['id'],inplace=True)
#     vali_cacg.Date_irrigation=pd.to_datetime(vali_cacg.Date_irrigation,format='%d/%m/%y')
#     vali_cacg["Quantity(mm)"].astype(float)
#     sum_irr_cacg_val=vali_cacg.groupby("id")["Quantity(mm)"].sum()
#     nb_irr=vali_cacg.groupby("id")["Date_irrigation"].count()
    
#     for r in os.listdir('D:/THESE_TMP/RUNS_SAMIR/RUN_RELATION/TARN_CACG/'):
#         d["path_PC"]='D:/THESE_TMP/RUNS_SAMIR/RUN_RELATION/TARN_CACG/'+str(r)+'/Inputdata/'
#         slope, intercept, r_value, p_value, std_err = stats.linregress(sum_irr_cacg_val.to_list(),all_resu.loc[all_resu.runs==r]["cumul_irr"].to_list())
#         bias=1/sum_irr_cacg_val.shape[0]*sum(np.mean(all_resu.loc[all_resu.runs==r]["cumul_irr"])-sum_irr_cacg_val) 
#         fitLine = predict(sum_irr_cacg_val)
#         plt.figure(figsize=(7,7))
#         plt.title(r)
#         plt.plot([0.0, 300], [0.0, 300], 'r-', lw=2)
#         plt.plot(sum_irr_cacg_val,fitLine,linestyle="-")
#         plt.scatter(sum_irr_cacg_val,all_resu.loc[all_resu.runs==r]["cumul_irr"])
#         plt.plot()
#         plt.xlabel("cumul_irr OBS")
#         plt.ylabel("cumul irr model")
#         plt.xlim(0,300)
#         plt.ylim(0,300)
#         rms = mean_squared_error(sum_irr_cacg_val,all_resu.loc[all_resu.runs==r]["cumul_irr"],squared=False)
#         plt.text(10,min(all_resu.loc[all_resu.runs==r]["cumul_irr"])+40,"RMSE = "+str(round(rms,2)))
#         plt.text(10,min(all_resu.loc[all_resu.runs==r]["cumul_irr"])+30,"R² = "+str(round(r_value,2)))
#         plt.text(10,min(all_resu.loc[all_resu.runs==r]["cumul_irr"])+20,"Pente = "+str(round(slope,2)))
#         plt.text(10,min(all_resu.loc[all_resu.runs==r]["cumul_irr"])+10,"Biais = "+str(round(bias,2)))
#         for j in np.arange(len(sum_irr_cacg_val.index)):
#             plt.text(x = sum_irr_cacg_val.to_list()[j] + 2, y=all_resu.loc[all_resu.runs==r]["cumul_irr"].iloc[j]+ 1,s = list(all_resu.loc[all_resu.runs==r]["id"])[j],size=9)
#         plt.savefig(d["path_PC"][:-10]+"plt_scatter_quantity_irri_%s.png"%r)
# #         plot nb_irr
        
#         slope, intercept, r_value, p_value, std_err = stats.linregress(nb_irr.to_list(),all_resu.loc[all_resu.runs==r]["nb_irr"].to_list())
#         bias=1/nb_irr.shape[0]*sum(mean(all_resu.loc[all_resu.runs==r]["nb_irr"])-nb_irr) 
#         fitLine = predict(nb_irr)
#         plt.figure(figsize=(7,7))
#         plt.title(r)
#         plt.plot([0.0,20], [0.0, 20], 'r-', lw=2)
#         plt.plot(nb_irr,fitLine,linestyle="--")
#         plt.scatter(nb_irr,all_resu.loc[all_resu.runs==r]["nb_irr"])
#         plt.plot()
#         plt.xlabel("nb_irr OBS")
#         plt.ylabel("nb irr model")
#         plt.xlim(0,20)
#         plt.ylim(0,20)
#         rms = mean_squared_error(nb_irr,all_resu.loc[all_resu.runs==r]["nb_irr"],squared=False) # if False == RMSE or True == MSE
#         plt.text(10,min(all_resu.loc[all_resu.runs==r]["nb_irr"])+2,"RMSE = "+str(round(rms,2)))
#         plt.text(10,min(all_resu.loc[all_resu.runs==r]["nb_irr"])+1.5,"R² = "+str(round(r_value,2)))
#         plt.text(10,min(all_resu.loc[all_resu.runs==r]["nb_irr"])+1,"Pente = "+str(round(slope,2)))
#         plt.text(10,min(all_resu.loc[all_resu.runs==r]["nb_irr"])+0.5,"Biais = "+str(round(bias,2)))
#         for j in np.arange(len(nb_irr.index)):
#             plt.text(x = nb_irr.to_list()[j]+0.1 , y=all_resu.loc[all_resu.runs==r]["nb_irr"].iloc[j]+0.1,s = list(all_resu.loc[all_resu.runs==r]["id"])[j],size=9)
#         plt.savefig(d["path_PC"][:-10]+"plt_scatter_nb_irri_%s.png"%r)


