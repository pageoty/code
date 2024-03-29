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
    # name_run="RUNS_SAMIR/RUN_MULTI_SITE_ICOS/RUN_OPTIMISATION_ICOS/SAMIR_OPTIMI_LAM/RUN_final/Merlin_init_ru_fewi_De_Kr_Fcover_irri_auto_soil"
    name_run_FAO="RUNS_SAMIR/RUN_MULTI_SITE_ICOS/RUN_OPTIMISATION_ICOS/SAMIR_OPTIMI_LAM/RUN_final/Init_ru/Merlin_init_ru_0_optim_fewi_De_Kr_irri_man_soil"
    name_run_merlin="RUNS_SAMIR/RUN_MULTI_SITE_ICOS/RUN_OPTIMISATION_ICOS/SAMIR_OPTIMI_LAM/RUN_final/Init_ru/Merlin_init_ru_100_optim_fewi_De_Kr_irri_man_soil"
    name_run_save_fig="RUNS_SAMIR/RUN_MULTI_SITE_ICOS/RUN_OPTIMISATION_ICOS/SAMIR_OPTIMI_LAM/RUN_final/Init_ru/Merlin_init_ru_100_optim_fewi_De_Kr_irri_man_soil"
    d["PC_disk"]="/run/media/pageot/Transcend/Yann_THESE/BESOIN_EAU/BESOIN_EAU/"
    d["PC_home"]="/mnt/d/THESE_TMP/"
    d["PC_home_Wind"]="D:/THESE_TMP/"
    # d["PC_disk"]="H:/Yann_THESE/BESOIN_EAU/BESOIN_EAU/"
    d["PC_labo"]="/datalocal/vboxshare/THESE/BESOIN_EAU/"
    label_ref="Kcmax =1.15"
    label_test="Kcmax =1.00"
    years=["2008","2010","2012","2014",'2015',"2019"]
    lc="maize_irri"
    
# =============================================================================
# Validation Flux ETR ICOS non Multi_sie run
# =============================================================================
# modif pour groupby lc

    # for y in years:
    #     for lc in ["maize_irri"]: # maize_rain
    #         # d['Output_model_PC_labo']='/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/'+name_run+"/"+y+"/"
    #         # d["Output_model_PC_home"]="/mnt/d/THESE_TMP/TRAITEMENT/"+name_run+"/"+y+"/"
    #         d["Output_model_PC_home"]="D:/THESE_TMP/TRAITEMENT/"+name_run+"/"+y+"/"
    #         d["Output_model_PC_home_disk"]="/run/media/pageot/Transcend/Yann_THESE/BESOIN_EAU/BESOIN_EAU/TRAITEMENT/"+name_run+"/"+y+"/"
    #         d["Output_model_PC_labo"]="/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/"+name_run+"/"+y+"/"
    #         # d["Output_model_PC_home_disk"]="H:/Yann_THESE/BESOIN_EAU/BESOIN_EAU/TRAITEMENT/"+name_run+"/"+y+"/"
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
            # ETR=pd.read_csv(d["PC_disk"]+"/TRAITEMENT/DATA_VALIDATION/DATA_ETR_CESBIO/DATA_ETR_corr_"+str(lc)+"/ETR_"+str(lc)+str(y)+".csv",decimal='.',sep=",")
            # ETR["date"]=pd.to_datetime(ETR["date"],format="%Y-%m-%d")
            # ETR_obs=ETR.loc[(ETR.date >= str(y)+"-04-01") &(ETR.date <= str(y)+"-09-30")]

            
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
# Plot incertitude PF_cc et Fcover
# =============================================================================
    print("============")
    print ("Incertitude plot")
    print("============")

    d["Output_model_save_fig"]="/run/media/pageot/Transcend/Yann_THESE/BESOIN_EAU/BESOIN_EAU/TRAITEMENT/"+name_run_save_fig
    # d["Output_model_save_fig"]="H:/Yann_THESE/BESOIN_EAU/BESOIN_EAU/TRAITEMENT/"+name_run_save_fig

    ETR_test_m20=None
    ETR_test_pl20=None
    ETR_test=None
    
    ETR_test_FAO_m20=None
    ETR_test_FAO_pl20=None
    ETR_test_FAO=None
    
    ETR_all=pd.DataFrame()
    ETR_all_Merlin=pd.DataFrame()
    ETR_mod_all=pd.DataFrame()
    for y in years:
        # ET0=pickle.load(open("H:/Yann_THESE/BESOIN_EAU/BESOIN_EAU/TRAITEMENT/RUNS_SAMIR/RUN_MULTI_SITE_ICOS/RUN_OPTIMISATION_ICOS/SAMIR_OPTIMI_LAM/RUN_final/Merlin_init_ru_optim_fewi_De_Kr_Fcover_irri_man_soil/"+str(y)+"/Inputdata/maize_irri/meteo.df","rb"))
        ET0=pickle.load(open("/run/media/pageot/Transcend/Yann_THESE/BESOIN_EAU/BESOIN_EAU/TRAITEMENT/RUNS_SAMIR/RUN_MULTI_SITE_ICOS/RUN_OPTIMISATION_ICOS/SAMIR_OPTIMI_LAM/RUN_final/Merlin_init_ru_optim_fewi_De_Kr_Fcover_irri_man_soil/"+str(y)+"/Inputdata/maize_irri/meteo.df","rb"))
        data_E=ET0[["date",'ET0',"Prec"]]
        data_meteo=data_E.loc[(data_E.date>= str(y)+"-04-01") &(data_E.date <= str(y)+"-09-30")]
        # if "Fcover" in name_run_merlin:
        #     for Fco in ["_pl20","_m20",""]:
        #         d["Output_model_PC_home_disk"]="/run/media/pageot/Transcend/Yann_THESE/BESOIN_EAU/BESOIN_EAU/TRAITEMENT/"+name_run_merlin[0:112]+str(Fco)+name_run_merlin[112:]
        #         ####  récupération num_run max incertitude
        #         ETR_mod=pd.read_csv(d["Output_model_PC_home_disk"]+"/LUT_ETR"+str(y)+".csv",index_col=[0,1])
        #         ETR_mod.columns=pd.to_datetime(ETR_mod.columns,format="%Y-%m-%d")
        #         ETR_mod=ETR_mod.loc[:,(ETR_mod.columns >= str(y)+"-04-01") &(ETR_mod.columns <= str(y)+"-09-30")]
        #         ETR_mod_max=pd.read_csv(d["Output_model_PC_home_disk"]+"_max"+"/LUT_ETR"+str(y)+".csv",index_col=[0,1])
        #         ETR_mod_max.columns=pd.to_datetime(ETR_mod_max.columns,format="%Y-%m-%d")
        #         ETR_mod_max=ETR_mod_max.loc[:,(ETR_mod_max.columns >= str(y)+"-04-01") &(ETR_mod_max.columns <= str(y)+"-09-30")]
        #         ETR_mod_min=pd.read_csv(d["Output_model_PC_home_disk"]+"_min"+"/LUT_ETR"+str(y)+".csv",index_col=[0,1])
        #         ETR_mod_min.columns=pd.to_datetime(ETR_mod_min.columns,format="%Y-%m-%d")
        #         ETR_mod_min=ETR_mod_min.loc[:,(ETR_mod_min.columns >= str(y)+"-04-01") &(ETR_mod_min.columns <= str(y)+"-09-30")]
        #         globals()["ETR_test%s"%(Fco)]=pd.concat([ETR_mod_max,ETR_mod,ETR_mod_min])
        #     ETR_test_merlin=pd.concat([ETR_test_pl20,ETR_test_m20,ETR_test])
        # else:
        d["Output_model_PC_home_disk"]="/run/media/pageot/Transcend/Yann_THESE/BESOIN_EAU/BESOIN_EAU/TRAITEMENT/"+name_run_merlin
        ####  récupération num_run max incertitude
        ETR_mod=pd.read_csv(d["Output_model_PC_home_disk"]+"/LUT_ETR"+str(y)+".csv",index_col=[0,1])
        ETR_mod.columns=pd.to_datetime(ETR_mod.columns,format="%Y-%m-%d")
        ETR_mod=ETR_mod.loc[:,(ETR_mod.columns >= str(y)+"-04-01") &(ETR_mod.columns <= str(y)+"-09-30")]
        # Select_run_1000
        ETR_mod=ETR_mod.loc[4]
        # ETR_test_merlin=pd.concat([ETR_mod_max,ETR_mod,ETR_mod_min])
        ETR_test_merlin=ETR_mod

        ###### run de ref en X
        if "Fcover" in name_run_FAO:
            for Fco in ["_pl20","_m20",""]:
                d["Output_model_PC_home_disk"]="/run/media/pageot/Transcend/Yann_THESE/BESOIN_EAU/BESOIN_EAU/TRAITEMENT/"+name_run_FAO[0:112]+str(Fco)+name_run_FAO[112:] # 112 sans optim  or 129
                ETR_mod=pd.read_csv(d["Output_model_PC_home_disk"]+"/LUT_ETR"+str(y)+".csv",index_col=[0,1])
                ETR_mod.columns=pd.to_datetime(ETR_mod.columns,format="%Y-%m-%d")
                ETR_mod=ETR_mod.loc[:,(ETR_mod.columns >= str(y)+"-04-01") &(ETR_mod.columns <= str(y)+"-09-30")]
                ETR_mod_max=pd.read_csv(d["Output_model_PC_home_disk"]+"_max"+"/LUT_ETR"+str(y)+".csv",index_col=[0,1])
                ETR_mod_max.columns=pd.to_datetime(ETR_mod_max.columns,format="%Y-%m-%d")
                ETR_mod_max=ETR_mod_max.loc[:,(ETR_mod_max.columns >= str(y)+"-04-01") &(ETR_mod_max.columns <= str(y)+"-09-30")]
                ETR_mod_min=pd.read_csv(d["Output_model_PC_home_disk"]+"_min"+"/LUT_ETR"+str(y)+".csv",index_col=[0,1])
                ETR_mod_min.columns=pd.to_datetime(ETR_mod_min.columns,format="%Y-%m-%d")
                ETR_mod_min=ETR_mod_min.loc[:,(ETR_mod_min.columns >= str(y)+"-04-01") &(ETR_mod_min.columns <= str(y)+"-09-30")]
                globals()["ETR_test_FAO%s"%(Fco)]=pd.concat([ETR_mod_max,ETR_mod,ETR_mod_min])
            ETR_test_FAO=pd.concat([ETR_test_FAO_pl20,ETR_test_FAO_m20,ETR_test_FAO])
        else:
            d["Output_model_PC_home_disk"]="/run/media/pageot/Transcend/Yann_THESE/BESOIN_EAU/BESOIN_EAU/TRAITEMENT/"+name_run_FAO
            ETR_mod=pd.read_csv(d["Output_model_PC_home_disk"]+"/LUT_ETR"+str(y)+".csv",index_col=[0,1])
            ETR_mod.columns=pd.to_datetime(ETR_mod.columns,format="%Y-%m-%d")
            ETR_mod=ETR_mod.loc[:,(ETR_mod.columns >= str(y)+"-04-01") &(ETR_mod.columns <= str(y)+"-09-30")]
            # ETR_mod_max=pd.read_csv(d["Output_model_PC_home_disk"]+"_max"+"/LUT_ETR"+str(y)+".csv",index_col=[0,1])
            # ETR_mod_max.columns=pd.to_datetime(ETR_mod_max.columns,format="%Y-%m-%d")
            # ETR_mod_max=ETR_mod_max.loc[:,(ETR_mod_max.columns >= str(y)+"-04-01") &(ETR_mod_max.columns <= str(y)+"-09-30")]
            # ETR_mod_min=pd.read_csv(d["Output_model_PC_home_disk"]+"_min"+"/LUT_ETR"+str(y)+".csv",index_col=[0,1])
            # ETR_mod_min.columns=pd.to_datetime(ETR_mod_min.columns,format="%Y-%m-%d")
            # ETR_mod_min=ETR_mod_min.loc[:,(ETR_mod_min.columns >= str(y)+"-04-01") &(ETR_mod_min.columns <= str(y)+"-09-30")]
            # ETR_test_FAO=pd.concat([ETR_mod_max,ETR_mod,ETR_mod_min])
            ETR_mod=ETR_mod.loc[4]
            ETR_test_FAO=ETR_mod
# df=pickle.load(open("H:/Yann_THESE/BESOIN_EAU/BESOIN_EAU/TRAITEMENT/RUNS_SAMIR/RUN_MULTI_SITE_ICOS/RUN_OPTIMISATION_ICOS/SAMIR_OPTIMI_LAM/RUN_final/init_ru/Merlin_init_ru_100_optim_fewi_De_Kr_irri_man_soil_min/2015/Output/maxZr/output_test_maize_irri_0.df",'rb'))
# =============================================================================
#         Utilisation des données ETR obs 
# =============================================================================
        ETR=pd.read_csv(d["PC_disk"]+"/TRAITEMENT/DATA_VALIDATION/DATA_ETR_CESBIO/DATA_ETR_corr_maize_irri/ETR_maize_irri"+str(y)+".csv",decimal='.',sep=",")
        ETR["date"]=pd.to_datetime(ETR["date"],format="%Y-%m-%d")
        ETR_obs=ETR.loc[(ETR.date >= str(y)+"-04-01") &(ETR.date <= str(y)+"-09-30")]
        #  Calcule de la moyenne glissante
        ETR_rolling=ETR_obs.rolling(5).mean()
        ETR_rolling["date"]=ETR_obs.date
        ETR_mod_all=ETR_mod_all.append(ETR_rolling)
        
        ETR_obs_rolling_pl20=ETR_rolling.LE_Bowen+(20*ETR_rolling.LE_Bowen/100)
        ETR_obs_rolling_m20=ETR_rolling.LE_Bowen-(20*ETR_rolling.LE_Bowen/100)
        ETR_obs_m20=ETR_obs.LE_Bowen-(20*ETR_obs.LE_Bowen/100)
        ETR_obs_p20=ETR_obs.LE_Bowen+(20*ETR_obs.LE_Bowen/100)
        # Merlin
        ETR_test_rolling=ETR_test_merlin.T.rolling(5).mean()
        ETR_test_rolling=ETR_test_rolling.T
        ETR_all_Merlin=ETR_all_Merlin.append(ETR_test_rolling.T)


        
        
        # diff=ETR_test_rolling.max()-ETR_test_rolling.min()
        # datefor=diff.idxmax() # forte incertitude a cette date
        # coup_parm=ETR_test_rolling.T.loc[ETR_test_rolling.columns==datefor].T.idxmax() # récupération couple param max 
        # coup_parm_min=ETR_test_rolling.T.loc[ETR_test_rolling.columns==datefor].T.idxmin()
        #   # lecture des output_LUT
        # dfmax=pickle.load(open(d["PC_disk"]+"/TRAITEMENT/"+name_run_merlin+"/"+str(y)+"/Output/maxZr/output_test_maize_irri_"+str(coup_parm.iloc[0][0])+".df","rb"))
        # dfmin=pickle.load(open(d["PC_disk"]+"/TRAITEMENT/"+name_run_merlin+"/"+str(y)+"/Output/maxZr/output_test_maize_irri_"+str(coup_parm_min.iloc[0][0])+".df","rb"))
        # dfmax.date=pd.to_datetime(dfmax.date,format="%Y-%m-%d")
        # dfmin.date=pd.to_datetime(dfmin.date,format="%Y-%m-%d")
        # dfmax=dfmax.loc[(dfmax.date>= str(y)+"-04-01") &(dfmax.date <= str(y)+"-09-30")]
        # dfmin=dfmin.loc[(dfmin.date>= str(y)+"-04-01") &(dfmin.date <= str(y)+"-09-30")]
        # KS=pd.concat([dfmax,dfmin])
        # coeff_ks=KS.groupby("date").mean()[["Ks","Kei",'Kep',"fewi","SWC1",'Ir_auto','Dei','Dr']]
        # coeff_ks_std=KS.groupby("date").std()[["Ks","Ir_auto"]]
        # FAO 
        ETR_test_FAO_rolling=ETR_test_FAO.T.rolling(5).mean()
        ETR_test_FAO_rolling=ETR_test_FAO_rolling.T
        ETR_all=ETR_all.append(ETR_test_FAO_rolling.T)
 
        # diff=ETR_test_FAO_rolling.max()-ETR_test_FAO_rolling.min()
        # datefor=diff.idxmax() # forte incertitude a cette date
        # coup_parm=ETR_test_FAO_rolling.T.loc[ETR_test_FAO_rolling.columns==datefor].T.idxmax() # récupération couple param max 
        # coup_parm_min=ETR_test_FAO_rolling.T.loc[ETR_test_FAO_rolling.columns==datefor].T.idxmin()
        # #  lecture des output_LUT
        # dfmax2=pickle.load(open(d["PC_disk"]+"/TRAITEMENT/"+name_run_FAO+"/"+str(y)+"/Output/maxZr/output_test_maize_irri_"+str(coup_parm.iloc[0][0])+".df","rb"))
        # dfmin2=pickle.load(open(d["PC_disk"]+"/TRAITEMENT/"+name_run_FAO+"/"+str(y)+"/Output/maxZr/output_test_maize_irri_"+str(coup_parm_min.iloc[0][0])+".df","rb"))
        # dfmax2.date=pd.to_datetime(dfmax2.date,format="%Y-%m-%d")
        # dfmin2.date=pd.to_datetime(dfmin2.date,format="%Y-%m-%d")
        # dfmax2=dfmax2.loc[(dfmax2.date>= str(y)+"-04-01") &(dfmax2.date <= str(y)+"-09-30")]
        # dfmin2=dfmin2.loc[(dfmin2.date>= str(y)+"-04-01") &(dfmin2.date <= str(y)+"-09-30")]
        # KS_man=pd.concat([dfmax2,dfmin2])
        # coeff_ks_man=KS_man.groupby("date").mean()[["Ks",'Kei','Kep','NDVI',"fewi",'FCov',"SWC1",'Irrig','Dei','Dr']]
# =============================================================================
#         # plot dynamique 
# =============================================================================
        # NDVI=pickle.load(open("/run/media/pageot/Transcend/Yann_THESE/BESOIN_EAU/BESOIN_EAU/TRAITEMENT/RUNS_SAMIR/RUN_MULTI_SITE_ICOS/RUN_OPTIMISATION_ICOS/SAMIR_OPTIMI_LAM/RUN_analyse/Merlin_init_ru_Fcover_irri_man_Bruand/"+str(y)+"/Inputdata/maize_irri/NDVI"+str(y)+".df","rb"))
        # NDVI=NDVI.loc[(NDVI.date>= str(y)+"-04-01") &(NDVI.date <= str(y)+"-09-30")]
        # plt.figure(figsize=(7,7))
        # # plt.plot(NDVI.date,NDVI.NDVI*10,label='NDVI',color="green",linewidth=2,linestyle="--")
        # plt.plot(ETR_rolling.date,ETR_rolling.LE_Bowen,label='ETR obs',color="black",linewidth=1)
        # plt.fill_between(ETR_rolling.date, ETR_obs_rolling_m20,ETR_obs_rolling_pl20,alpha=0.5,facecolor="None",ec='black',linestyle="--")
        # plt.plot(ETR_test_rolling.T.index,ETR_test_rolling.mean(),label='ETR '+label_test,linewidth=1,color='red')
        # plt.fill_between(ETR_test_rolling.T.index, ETR_test_rolling.min(),ETR_test_rolling.max(),alpha=0.2,facecolor="red") # revoir pour mettre l'écart types et non min et max
        # # plt.fill_between(ETR_test_rolling.T.index, ETR_test_rolling.mean()-ETR_test_rolling.std(),ETR_test_rolling.mean()+ETR_test_rolling.std(),alpha=0.2,facecolor="green") # avec std
        # plt.plot(ETR_test_FAO_rolling.T.index,ETR_test_FAO_rolling.mean(),label='ETR ' +label_ref,linewidth=1,color="Blue")
        # plt.fill_between(ETR_test_FAO_rolling.T.index, ETR_test_FAO_rolling.min(),ETR_test_FAO_rolling.max(),alpha=0.2,facecolor="Blue")
        # if "Irrigation" in label_ref:
        #     plt.plot(dfmax.loc[dfmax.Ir_auto>0.0].date,dfmax.Ir_auto.loc[dfmax.Ir_auto>0.0]/10,color='darkgreen',label="Irrigation auto",linestyle="",marker="x")
        #     plt.plot(coeff_ks_man.loc[coeff_ks_man.Irrig>0.0].index,coeff_ks_man.Irrig.loc[coeff_ks_man.Irrig>0.0]/10,color='Blue',label="Irrigation forcée",linestyle="",marker="+")
        # else:
        #     plt.plot(coeff_ks_man.loc[coeff_ks_man.Irrig>0.0].index,coeff_ks_man.Irrig.loc[coeff_ks_man.Irrig>0.0]/10,color='Blue',label="Irrigation forcée",linestyle="",marker="+")
        # plt.ylabel("ETR")
        # plt.ylim(0,10)
        # plt.legend()
        # ax2 = plt.twinx()
        # ax2.bar(data_meteo.date,data_meteo.Prec)
        # ax2.set_ylim(0,50)
        # ax2.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(6))
        # plt.ylabel("Pluviométrie")
        # # ax2.plot(coeff_ks.index,coeff_ks.Ks,color='r',linestyle="--",label="Ks")
        # ax2.set_ylim(0,50)
        # # ax2.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(6))
        # plt.title("Dynamique ETR moyenne PFCC incertitude %s en %s"%(lc,y))
        # # plt.savefig(d["Output_model_save_fig"]+"/plt_ETR_mod_mean_rolling_%s_%s.png"%(lc,y),dpi=330)
        
        
        
# =============================================================================
#         # Cumul
# =============================================================================
        # plt.figure(figsize=(7,7))
        # plt.plot(ETR_obs.date,ETR_obs.LE_Bowen.cumsum(),label='ETR obs',color="black")
        # plt.fill_between(ETR_obs.date, ETR_obs_m20.cumsum(),ETR_obs_p20.cumsum(),alpha=0.5,facecolor="None",ec='black',linestyle="--")
        # plt.fill_between(ETR_test_merlin.T.index, ETR_test_merlin.min().cumsum(),ETR_test_merlin.max().cumsum(),alpha=0.2,facecolor="red")
        # plt.plot(ETR_test_merlin.T.index,ETR_test_merlin.mean().cumsum(),label='ETR '+ label_test,color='red')
        # plt.text(ETR_obs.date.iloc[-1], ETR_obs.LE_Bowen.cumsum().iloc[-1], s=round(ETR_obs.LE_Bowen.cumsum().iloc[-1],2))
        # plt.text(ETR_test_merlin.T.index[-1], ETR_test_merlin.mean().cumsum().iloc[-1], s=round(ETR_test_merlin.mean().cumsum().iloc[-1],2))
        # plt.fill_between(ETR_test_FAO.T.index, ETR_test_FAO.min().cumsum(),ETR_test_FAO.max().cumsum(),alpha=0.2,facecolor="Blue")
        # plt.plot(ETR_test_FAO.T.index,ETR_test_FAO.mean().cumsum(),label='ETR '+label_ref,color='Blue')
        # plt.text(ETR_test_FAO.T.index[-1], ETR_test_FAO.mean().cumsum().iloc[-1], s=round(ETR_test_FAO.mean().cumsum().iloc[-1],2))
        # plt.ylabel("ETR")
        # plt.ylim(0,700)
        # plt.title("Dynamique ETR cumul incertitude %s en %s"%(lc,y))
        # plt.legend(loc="upper left")
        # plt.savefig(d["Output_model_save_fig"]+"/plt_Dynamique_ETR_ETR_cumul_incertitude_%s_%s.png"%(lc,y))
        

# # =============================================================================
# #         Scatter 3 datas
# # =============================================================================
#         tmp=ETR_rolling.LE_Bowen.dropna()        
#         tmp_mod=ETR_test_rolling.mean().dropna()
#         tmp_mod_FAO=ETR_test_FAO_rolling.mean().dropna()
#     #  Merlin vs obs
#         slope, intercept, r_value, p_value, std_err = stats.linregress(tmp.to_list(),tmp_mod.to_list())
#         bias=1/tmp.shape[0]*sum(tmp_mod-np.mean(tmp)) 
#         fitLine = predict(tmp_mod)
# #         # Creation plot

#         plt.figure(figsize=(7,7))
#         plt.plot([0.0, 10], [0.0,10], 'black', lw=1,linestyle='--')
#         plt.xlabel("ETR Obs")
#         plt.ylabel("ETR simulée")
#         plt.xlim(0,10)
#         plt.ylim(0,10)
#         plt.scatter(ETR_rolling.LE_Bowen, ETR_test_rolling.mean(), zorder = 2,color="r",label="Obs / "+label_test)
#         plt.title("Scatter ETR %s"%y)
#         rms = mean_squared_error(tmp,tmp_mod)
#         rectangle = plt.Rectangle((0.9, 8.6),2.1,1.3, ec='red',fc='red',alpha=0.1)
#         plt.gca().add_patch(rectangle)
#         plt.text(1,9.6,"RMSE = "+str(round(rms,2)),fontsize=12) 
#         plt.text(1,9.3,"R² = "+str(round(r_value,2)),fontsize=12)
#         plt.text(1,9.0,"Pente = "+str(round(slope,2)),fontsize=12)
#         plt.text(1,8.7,"Biais = "+str(round(bias,2)),fontsize=12)
#             #  FAo vs obs
#         slope, intercept, r_value, p_value, std_err = stats.linregress(tmp.to_list(),tmp_mod_FAO.to_list())
#         bias=1/tmp.shape[0]*sum(tmp_mod_FAO-np.mean(tmp)) 
#         fitLine = predict(tmp_mod_FAO)
#         # Creation plot
#         plt.scatter(ETR_rolling.LE_Bowen, ETR_test_FAO_rolling.mean(), zorder = 2,color='b',label="Obs / "+label_ref)
#         rms = mean_squared_error(tmp,tmp_mod_FAO)
#         rectangle = plt.Rectangle((7.9, 1),2.1,1.4, ec='blue',fc='blue',alpha=0.1)
#         plt.gca().add_patch(rectangle)
#         plt.text(8,2,"RMSE = "+str(round(rms,2)),fontsize=12) 
#         plt.text(8,1.7,"R² = "+str(round(r_value,2)),fontsize=12)
#         plt.text(8,1.4,"Pente = "+str(round(slope,2)),fontsize=12)
#         plt.text(8,1.1,"Biais = "+str(round(bias,2)),fontsize=12)
#         plt.legend(loc="upper right")
#         # plt.savefig(d["Output_model_save_fig"]+"/scatter_ETR_mods_obs_mean_rolling_V2_%s_%s.png"%(lc,y),dpi=330)

    
# =============================================================================
#          Calcule flux E
# =============================================================================
        # Fcov=pickle.load(open("/run/media/pageot/Transcend/Yann_THESE/BESOIN_EAU/BESOIN_EAU/TRAITEMENT/RUNS_SAMIR/RUN_MULTI_SITE_ICOS/RUN_OPTIMISATION_ICOS/SAMIR_OPTIMI_LAM/RUN_analyse/Merlin_init_ru_Fcover_irri_man_Bruand/"+str(y)+"/Inputdata/maize_irri/FCOVER.df","rb"))
        # Fcov=Fcov.loc[(Fcov.date>= str(y)+"-04-01") &(Fcov.date <= str(y)+"-09-30")]
        # data_merge=pd.merge(coeff_ks,data_E,on="date")
        # data_merge=data_merge.rolling(5).mean()
        # data_merge_ss_Fcover=pd.merge(coeff_ks_man,data_E,on="date")
        # data_merge_ss_Fcover=data_merge_ss_Fcover.rolling(5).mean()
        # fluxE_avec_Fcover=np.maximum((data_merge.Kei+data_merge.Kep)*data_merge.ET0,0)
        # fluxE_sans_Fcover=np.maximum((data_merge_ss_Fcover.Kei+data_merge_ss_Fcover.Kep)*data_merge.ET0,0)
        # plt.figure(figsize=(7,7))
        # plt.plot(Fcov.date,fluxE_avec_Fcover,label="Eva " +label_test)
        # plt.plot(Fcov.date,fluxE_sans_Fcover,label="Eva "+label_ref)
        # plt.plot(Fcov.date,coeff_ks_man.FCov,label="Fcover Samir")
        # if "Irrigation" in label_ref:
        #     plt.plot(dfmax.loc[dfmax.Ir_auto>0.0].date,dfmax.Ir_auto.loc[dfmax.Ir_auto>0.0]/10,color='darkgreen',label="Irrigation auto",linestyle="",marker="x")
        #     plt.plot(coeff_ks_man.loc[coeff_ks_man.Irrig>0.0].index,coeff_ks_man.Irrig.loc[coeff_ks_man.Irrig>0.0]/10,color='Blue',label="Irrigation forcée",linestyle="",marker="+")
        # else:
        #     plt.plot(coeff_ks_man.loc[coeff_ks_man.Irrig>0.0].index,coeff_ks_man.Irrig.loc[coeff_ks_man.Irrig>0.0]/10,color='Blue',label="Irrigation forcée",linestyle="",marker="+")
        # # plt.plot(Fcov.date,Fcov.FCov,label='Fcover_BVNet',linewidth=1)
        # plt.legend()
        # plt.ylim(0,5)
        # plt.ylabel("Flux Evaporation")
        # ax2 = plt.twinx()
        # ax2.bar(data_meteo.date,data_meteo.Prec)
        # ax2.set_ylim(0,50)
        # ax2.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(6))
        # plt.ylabel("Pluviométrie")
        # plt.title(str(y))
        # # plt.savefig(d["Output_model_save_fig"]+"/dynamique_flux_E_Fcover_%s_%s.png"%(lc,y),dpi=330)
        
        # # plt.figure(figsize=(7,7))
        # # plt.plot(coeff_ks.index,coeff_ks.fewi,label="fewi "+label_test)
        # # plt.plot(coeff_ks_man.index,coeff_ks_man.fewi,label="fewi "+label_ref)
        # # plt.plot(Fcov.date,coeff_ks_man.FCov,label="Fcover_samir")
        # # plt.plot(Fcov.date,Fcov.FCov,label='Fcover_BVNet',linewidth=1)
        # # plt.legend()
        # # plt.ylim(0,2)
        # # plt.ylabel("fewi")
        # # plt.title(str(y))
        # # plt.savefig(d["Output_model_save_fig"]+"/dynamique_fewi_Fcover_%s_%s.png"%(lc,y),dpi=330)
        
        
        # plt.figure(figsize=(7,7))
        # plt.plot(coeff_ks.index,coeff_ks.Kei,label="Kei " +label_test,color='red')
        # plt.plot(coeff_ks_man.index,coeff_ks_man.Kei,label="Kei "+label_ref,color='blue')
        # plt.legend()
        # plt.ylim(0,2)
        # plt.ylabel("Kr")
        # plt.title(str(y))
        # ax2 = plt.twinx()
        # ax2.plot(coeff_ks.index,coeff_ks.Ks,label="Ks " +label_test,linestyle="--",color='red')
        # ax2.plot(coeff_ks_man.index,coeff_ks_man.Ks,label="Ks "+label_ref,linestyle="--",color='blue')
        # ax2.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(6))
        # ax2.set_ylim(-1,1.5)
        # ax2.set_ylabel("Ks")
        # ax2.legend(loc="upper left")
        # plt.savefig(d["Output_model_save_fig"]+"/dynamique_Ke_Ks_%s_%s.png"%(lc,y),dpi=330)
        # plt.figure(figsize=(7,7))
        # plt.plot(coeff_ks.Dei,coeff_ks.Kri,label="Kri "+label_test,marker='o',linestyle="")
        # plt.plot(coeff_ks_man.Dei,coeff_ks_man.Kri,label="Kri "+label_ref,marker='o',linestyle="")
        # plt.legend()
        # plt.ylim(0,1.5)
        # plt.ylabel("Kr")
        # plt.xlabel("De")
        # plt.title(str(y))
        # plt.savefig(d["Output_model_save_fig"]+"/dynamique_Kr_Dei_%s_%s.png"%(lc,y),dpi=330)
        
        # plt.figure(figsize=(7,7))
        # plt.plot(coeff_ks.index,coeff_ks.Dr,label="Dr " +label_test,color='blue')
        # plt.plot(coeff_ks_man.index,coeff_ks_man.Dr,label="Dr "+label_ref,color='red')
        # plt.plot(coeff_ks.index,coeff_ks.TAW*0.55,label="RAW",color='black',linestyle="--")
        # # plt.plot(coeff_ks.index,coeff_ks.TAW,label="TAW",color='black',linestyle="--")
        # plt.legend(loc="upper right")
        # plt.ylim(0,150)
        # plt.ylabel("Dr")
        # plt.title(str(y))
        # # ax2 = plt.twinx()
        # # # ax2.plot(coeff_ks.index,coeff_ks.Ks,label="Ks " +label_test,linestyle="--",color='blue')
        # # # ax2.plot(coeff_ks_man.index,coeff_ks_man.Ks,label="Ks "+label_ref,linestyle="--",color='red')
        # # ax2.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(6))
        # # ax2.set_ylim(1,-2)
        # # ax2.set_ylabel("Ks")
        # # ax2.legend(loc="upper left")
        # plt.savefig(d["Output_model_save_fig"]+"/dynamique_Dr_TAW_RU_%s_%s.png"%(lc,y),dpi=330)
        
# =============================================================================
#         Plot maxZR variation ETR + Irri auto
# ============================================================================
    #     dfmax=pickle.load(open(d["PC_disk"]+"/TRAITEMENT/"+name_run_merlin+"/"+str(y)+"/Output/maxZr/output_test_maize_irri_14.df","rb"))
    #     dfmax_p20=pickle.load(open(d["PC_disk"]+"/TRAITEMENT/"+name_run_merlin[0:129]+"_pl20"+name_run_merlin[129:]+"/"+str(y)+"/Output/maxZr/output_test_maize_irri_14.df","rb"))
    #     dfmax_m20=pickle.load(open(d["PC_disk"]+"/TRAITEMENT/"+name_run_merlin[0:129]+"_m20"+name_run_merlin[129:]+"/"+str(y)+"/Output/maxZr/output_test_maize_irri_14.df","rb"))
        
    #     dfmin=pickle.load(open(d["PC_disk"]+"/TRAITEMENT/"+name_run_merlin+"/"+str(y)+"/Output/maxZr/output_test_maize_irri_0.df","rb"))
    #     dfmin_p20=pickle.load(open(d["PC_disk"]+"/TRAITEMENT/"+name_run_merlin[0:129]+"_pl20"+name_run_merlin[129:]+"/"+str(y)+"/Output/maxZr/output_test_maize_irri_0.df","rb"))
    #     dfmin_m20=pickle.load(open(d["PC_disk"]+"/TRAITEMENT/"+name_run_merlin[0:129]+"_m20"+name_run_merlin[129:]+"/"+str(y)+"/Output/maxZr/output_test_maize_irri_0.df","rb"))
        
    #     dfmean=pickle.load(open(d["PC_disk"]+"/TRAITEMENT/"+name_run_merlin+"/"+str(y)+"/Output/maxZr/output_test_maize_irri_4.df","rb"))
    #     dfmean_p20=pickle.load(open(d["PC_disk"]+"/TRAITEMENT/"+name_run_merlin[0:129]+"_pl20"+name_run_merlin[129:]+"/"+str(y)+"/Output/maxZr/output_test_maize_irri_4.df","rb"))
    #     dfmean_m20=pickle.load(open(d["PC_disk"]+"/TRAITEMENT/"+name_run_merlin[0:129]+"_m20"+name_run_merlin[129:]+"/"+str(y)+"/Output/maxZr/output_test_maize_irri_4.df","rb"))
    #     dfmax.date=pd.to_datetime(dfmax.date,format="%Y-%m-%d")
    #     dfmax_m20.date=pd.to_datetime(dfmax_m20.date,format="%Y-%m-%d")
    #     dfmax_p20.date=pd.to_datetime(dfmax_p20.date,format="%Y-%m-%d")
    #     dfmin.date=pd.to_datetime(dfmin.date,format="%Y-%m-%d")
    #     dfmin_m20.date=pd.to_datetime(dfmin_m20.date,format="%Y-%m-%d")
    #     dfmin_p20.date=pd.to_datetime(dfmin_p20.date,format="%Y-%m-%d")
    #     dfmean.date=pd.to_datetime(dfmean.date,format="%Y-%m-%d")
    #     dfmean_m20.date=pd.to_datetime(dfmean_m20.date,format="%Y-%m-%d")
    #     dfmean_p20.date=pd.to_datetime(dfmean_p20.date,format="%Y-%m-%d")
    #     dfmax=dfmax.loc[(dfmax.date>= str(y)+"-04-01") &(dfmax.date <= str(y)+"-09-30")]
    #     dfmax_p20=dfmax_p20.loc[(dfmax_p20.date>= str(y)+"-04-01") &(dfmax_p20.date <= str(y)+"-09-30")]
    #     dfmax_m20=dfmax_m20.loc[(dfmax_m20.date>= str(y)+"-04-01") &(dfmax_m20.date <= str(y)+"-09-30")]
    #     dfmin=dfmin.loc[(dfmin.date>= str(y)+"-04-01") &(dfmin.date <= str(y)+"-09-30")]
    #     dfmin_p20=dfmin_p20.loc[(dfmin_p20.date>= str(y)+"-04-01") &(dfmin_p20.date <= str(y)+"-09-30")]
    #     dfmin_m20=dfmin_m20.loc[(dfmin_m20.date>= str(y)+"-04-01") &(dfmin_m20.date <= str(y)+"-09-30")]
    #     dfmean=dfmean.loc[(dfmean.date>= str(y)+"-04-01") &(dfmean.date <= str(y)+"-09-30")]
    #     dfmean_p20=dfmean_p20.loc[(dfmean_p20.date>= str(y)+"-04-01") &(dfmean_p20.date <= str(y)+"-09-30")]
    #     dfmean_m20=dfmean_m20.loc[(dfmean_m20.date>= str(y)+"-04-01") &(dfmean_m20.date <= str(y)+"-09-30")]
    #     # print(y)
    #     # print("==========")
    #     # print('max 1000')
    #     # print(dfmean.Ir_auto[dfmean.Ir_auto!=0.0])
    #     # print("==========")
    #     # print("==========")
    #     # print('max 800')
    #     # print(dfmin.Ir_auto[dfmin.Ir_auto!=0.0])
    #     # print("==========")
    #     # print("==========")
    #     # print('max 1500')
    #     # print(dfmax.Ir_auto[dfmax.Ir_auto!=0.0])
    #     # print("==========")
    #     #  plot
    #     plt.figure(figsize=(7,7))
    #     plt.plot(ETR_rolling.date,ETR_rolling.LE_Bowen,label='ETR obs',color="black",linewidth=1,linestyle="--")
    #     plt.plot(dfmean.date,dfmean.ET.rolling(5).mean(),label='ETR maxZr 1000',color="green",linewidth=1)
    #     plt.fill_between(dfmean.date, dfmean_m20.ET.rolling(5).mean(),dfmean_p20.ET.rolling(5).mean(),alpha=0.5,facecolor="None",ec='black',linestyle="--")
    #     plt.plot(dfmax.date,dfmax.ET.rolling(5).mean(),label='ETR maxZr 1500',linewidth=1,color='red')
    #     plt.fill_between(dfmean.date, dfmax_m20.ET.rolling(5).mean(),dfmax_p20.ET.rolling(5).mean(),alpha=0.2,facecolor="red",ec='red')
    #     plt.plot(dfmin.date,dfmin.ET.rolling(5).mean(),label='ETR maxZr 800',linewidth=1,color="Blue")
    #     plt.fill_between(dfmean.date, dfmin_m20.ET.rolling(5).mean(),dfmin_p20.ET.rolling(5).mean(),alpha=0.2,facecolor="blue",ec='blue')
    #     plt.ylabel("ETR")
    #     plt.legend()
    #     plt.ylim(0,10)
    #     ax2 = plt.twinx()
    #     ax2.plot(dfmean.date,dfmean.Ks.rolling(5).mean(),label="Ks maxZr 1000" ,linestyle="--",color='green')
    #     ax2.plot(dfmin.date,dfmin.Ks.rolling(5).mean(),label="Ks maxZr 800",linestyle="--",color='blue')
    #     ax2.plot(dfmax.date,dfmax.Ks.rolling(5).mean(),label="Ks maxZr 1500",linestyle="--",color='red')
    #     ax2.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(6))
    #     ax2.set_ylim(1,-2)
    #     ax2.set_ylabel("Ks")
    #     ax2.set_ylabel("Ks")
    #     ax2.legend(loc="upper left")
    #     plt.savefig(d["Output_model_save_fig"]+"/ETR_maxzr_vara_modif_xlabel_%s_%s.png"%(lc,y),dpi=330)
        
    #     tmp=ETR_rolling.LE_Bowen.dropna()        
    #     tmp_mod=dfmean.ET.rolling(5).mean().dropna()
    #     tmp_mod_800=dfmin.ET.rolling(5).mean().dropna()
    #     tmp_mod_1500=dfmax.ET.rolling(5).mean().dropna()
    # #  Merlin vs obs
    #     slope, intercept, r_value, p_value, std_err = stats.linregress(tmp.to_list(),tmp_mod.to_list())
    #     bias=1/tmp.shape[0]*sum(tmp_mod-np.mean(tmp)) 
    #     fitLine = predict(tmp_mod)
        
    #     plt.figure(figsize=(7,7))
    #     plt.plot([0.0, 10], [0.0,10], 'black', lw=1,linestyle='--')
    #     plt.xlabel("ETR Obs")
    #     plt.ylabel("ETR simulée")
    #     plt.xlim(0,10)
    #     plt.ylim(0,10)
    #     plt.scatter(ETR_rolling.LE_Bowen, dfmean.ET.rolling(5).mean(), zorder = 2,color="green",label="maxZr 1000")
    #     plt.title("Scatter ETR %s"%y)
    #     rms = mean_squared_error(tmp,tmp_mod)
    #     rectangle = plt.Rectangle((0.9, 8.6),2.1,1.4, ec='red',fc='green',alpha=0.1)
    #     plt.gca().add_patch(rectangle)
    #     plt.text(1,9.6,"RMSE = "+str(round(rms,2)),fontsize=12) 
    #     plt.text(1,9.3,"R² = "+str(round(r_value,2)),fontsize=12)
    #     plt.text(1,9.0,"Pente = "+str(round(slope,2)),fontsize=12)
    #     plt.text(1,8.7,"Biais = "+str(round(bias,2)),fontsize=12)
    #         #  800 vs obs
    #     slope, intercept, r_value, p_value, std_err = stats.linregress(tmp.to_list(),tmp_mod_800.to_list())
    #     bias=1/tmp.shape[0]*sum(tmp_mod_800-np.mean(tmp)) 
    #     fitLine = predict(tmp_mod_800)
    #     # Creation plot
    #     plt.scatter(ETR_rolling.LE_Bowen,dfmin.ET.rolling(5).mean(), zorder = 2,color='b',label="maxZr 800")
    #     rms = mean_squared_error(tmp,tmp_mod_800)
    #     rectangle = plt.Rectangle((7.9, 1),2.1,1.4, ec='blue',fc='blue',alpha=0.1)
    #     plt.gca().add_patch(rectangle)
    #     plt.text(8,2,"RMSE = "+str(round(rms,2)),fontsize=12) 
    #     plt.text(8,1.7,"R² = "+str(round(r_value,2)),fontsize=12)
    #     plt.text(8,1.4,"Pente = "+str(round(slope,2)),fontsize=12)
    #     plt.text(8,1.1,"Biais = "+str(round(bias,2)),fontsize=12)
        
    #     #  1500 vs obs 
    #     slope, intercept, r_value, p_value, std_err = stats.linregress(tmp.to_list(),tmp_mod_1500.to_list())
    #     bias=1/tmp.shape[0]*sum(tmp_mod_1500-np.mean(tmp)) 
    #     fitLine = predict(tmp_mod_1500)
    #     # Creation plot
    #     plt.scatter(ETR_rolling.LE_Bowen, dfmax.ET.rolling(5).mean(), zorder = 2,color='r',label="maxZr 1500")
    #     rms = mean_squared_error(tmp,tmp_mod_1500)
    #     rectangle = plt.Rectangle((7.9, 5),2.1,1.4, ec='r',fc='r',alpha=0.1)
    #     plt.gca().add_patch(rectangle)
    #     plt.text(8,6,"RMSE = "+str(round(rms,2)),fontsize=12) 
    #     plt.text(8,5.7,"R² = "+str(round(r_value,2)),fontsize=12)
    #     plt.text(8,5.4,"Pente = "+str(round(slope,2)),fontsize=12)
    #     plt.text(8,5.1,"Biais = "+str(round(bias,2)),fontsize=12)
    #     plt.legend(loc="upper right")
    #     plt.savefig(d["Output_model_save_fig"]+"/scatter_ETR_maxzr_vara_%s_%s.png"%(lc,y),dpi=330)
        
    #     #  plot Ks 
        
# =============================================================================
#          SWC 2012 et 2014 
# =============================================================================
        # if "2014" in y or "2012" in y:
        #     print('ici')
            # plt.figure(figsize=(10,7))    
            # SWC=pd.read_csv(d["PC_disk"]+"/TRAITEMENT/DATA_VALIDATION/DATA_SWC/SWC_LAM/SWC_LAM_"+str(y)+".csv")
            # SWC["Date"]=pd.to_datetime(SWC["Date/Time"],format="%Y-%m-%d")
            # SWC=SWC.loc[(SWC.Date>= str(y)+"-05-01") &(SWC.Date <= str(y)+"-09-30")]
            # data_meteo=data_meteo.loc[(data_meteo.date>= str(y)+"-05-01") &(data_meteo.date <= str(y)+"-09-30")]
            # coeff_ks=coeff_ks.loc[(coeff_ks.index>= str(y)+"-05-01") &(coeff_ks.index <= str(y)+"-09-30")]
            # plt.plot(SWC["Date"],SWC.SWC_0_moy/100,label="0 cm")
            # plt.plot(SWC["Date"],SWC.SWC_5_moy/100,label="5 cm")
            # plt.plot(SWC["Date"],SWC.SWC_10_moy/100,label="10 cm")
            # # plt.plot(SWC["Date"],SWC.SWC_30_moy/100,label="30 cm")
            # plt.ylim(0,0.5)
            # plt.ylabel("Humidité du sol [m³ . m-³]")
            # plt.legend()
            # ax2 = plt.twinx()
            # ax2.bar(data_meteo.date,data_meteo.Prec,label="Précipitation")
            # ax2.plot(coeff_ks.loc[coeff_ks.Ir_auto>0.0].index,coeff_ks.Ir_auto.loc[coeff_ks.Ir_auto>0.0],color='darkgreen',label="Irrigation",linestyle="",marker="<")
            # # ax2.plot(coeff_ks_man.loc[coeff_ks_man.Irrig>0.0].index,coeff_ks_man.Irrig.loc[coeff_ks_man.Irrig>0.0],color='Blue',label="Irrigation_man",linestyle="",marker=">")
            # ax2.set_ylim(0,50)
            # ax2.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(6))
            # plt.ylabel("Pluviométrie")
            # plt.legend(loc="upper left")
            # plt.savefig(d["Output_model_save_fig"]+"/dynamique_SWC_sonde_Irrigation_%s.png"%(y),dpi=330)
# =============================================================================
#         Kei merlin vs Kei FAO 
# =============================================================================
        # name_run_FAO="RUNS_SAMIR/RUN_MULTI_SITE_ICOS/RUN_OPTIMISATION_ICOS/SAMIR_OPTIMI_LAM/RUN_final/FAO_init_ru_fewi_De_Kr_irri_man_soil"
        # name_run_Merlin="RUNS_SAMIR/RUN_MULTI_SITE_ICOS/RUN_OPTIMISATION_ICOS/SAMIR_OPTIMI_LAM/RUN_final/Merlin_init_ru_fewi_De_Kr_irri_man_soil_test_TAW_Hsat/"
        # dfmax_FAO=pickle.load(open(d["PC_disk"]+"/TRAITEMENT/"+name_run_FAO+"/"+str(y)+"/Output/maxZr/output_test_maize_irri_"+str(coup_parm.iloc[0][0])+".df","rb"))
        # dfmin_FAO=pickle.load(open(d["PC_disk"]+"/TRAITEMENT/"+name_run_FAO+"/"+str(y)+"/Output/maxZr/output_test_maize_irri_"+str(coup_parm_min.iloc[0][0])+".df","rb"))
        # dfmax_Merlin=pickle.load(open(d["PC_disk"]+"/TRAITEMENT/"+name_run_Merlin+"/"+str(y)+"/Output/maxZr/output_test_maize_irri_"+str(coup_parm.iloc[0][0])+".df","rb"))
        # dfmin_Merlin=pickle.load(open(d["PC_disk"]+"/TRAITEMENT/"+name_run_Merlin+"/"+str(y)+"/Output/maxZr/output_test_maize_irri_"+str(coup_parm_min.iloc[0][0])+".df","rb"))
        # # Fcov=pickle.load(open("H:/Yann_THESE/BESOIN_EAU/BESOIN_EAU/TRAITEMENT/RUNS_SAMIR/RUN_MULTI_SITE_ICOS/RUN_OPTIMISATION_ICOS/SAMIR_OPTIMI_LAM/RUN_analyse/Merlin_init_ru_Fcover_irri_man_Bruand/"+str(y)+"/Inputdata/maize_irri/FCOVER.df","rb"))
        # Ke_FAO=pd.concat([dfmax_FAO,dfmin_FAO])
        # coeff_ke_FAO=Ke_FAO.groupby("date").mean()[["Kei","NDVI"]]
        # Ke_Merlin=pd.concat([dfmax_Merlin,dfmin_Merlin])
        # coeff_ke_Merlin=Ke_Merlin.groupby("date").mean()["Kei"]
        
        # plt.figure(figsize=(7,7))
        # plt.plot(coeff_ke_FAO.index,coeff_ke_FAO.Kei,label='Ke_Fcover_SAMIR',color="black",linewidth=1)
        # plt.plot(coeff_ke_Merlin.index,coeff_ke_Merlin,label='Ke_Fcover_BVnet',linewidth=1)# récupération mode 
        # plt.plot(Fcov.date,Fcov.FCov,label='Fcover_BVNet',linewidth=1)
        # plt.plot(coeff_ke_FAO.index,coeff_ke_FAO.NDVI*1.39-0.25,label='Fcover_SAMIR',linewidth=1)
        # plt.legend()
        # plt.ylim(0, 1.5)
        # plt.title("Dynamique Kei")
        # plt.savefig(d["Output_model_save_fig"]+"/Dyan_Kei_FAO_Merlin_%s_%s.png"%(lc,y),dpi=330)
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
# debug

#     prec=pickle.load(open("/run/media/pageot/Transcend/Yann_THESE/BESOIN_EAU/BESOIN_EAU/TRAITEMENT/RUNS_SAMIR/RUN_MULTI_SITE_ICOS/RUN_OPTIMISATION_ICOS/SAMIR_OPTIMI_LAM/RUN_final/Merlin_init_ru_fewi_De_Kr_irri_man_soil/2012/Inputdata/maize_irri/meteo.df",'rb'))
#     df_Merlin=pickle.load(open("/run/media/pageot/Transcend/Yann_THESE/BESOIN_EAU/BESOIN_EAU/TRAITEMENT/RUNS_SAMIR/RUN_MULTI_SITE_ICOS/RUN_OPTIMISATION_ICOS/SAMIR_OPTIMI_LAM/RUN_final/Merlin_init_ru_fewi_De_Kr_irri_man_soil/2012/Output/maxZr/output_test_maize_irri_0.df",'rb'))
#     df_FAO=pickle.load(open("/run/media/pageot/Transcend/Yann_THESE/BESOIN_EAU/BESOIN_EAU/TRAITEMENT/RUNS_SAMIR/RUN_MULTI_SITE_ICOS/RUN_OPTIMISATION_ICOS/SAMIR_OPTIMI_LAM/RUN_final/FAO_init_ru_fewi_De_Kr_irri_man_soil/2012/Output/maxZr/output_test_maize_irri_0.df",'rb'))
    
    
#     plt.plot(df_Merlin.date,df_Merlin.Kei,label="Merlin")
#     plt.plot(df_FAO.date,df_FAO.Kei,label="FAO")
#     plt.bar(prec.date,prec.Prec)
#     plt.legend()
    
#     prec=pickle.load(open("/run/media/pageot/Transcend/Yann_THESE/BESOIN_EAU/BESOIN_EAU/TRAITEMENT/RUNS_SAMIR/RUN_MULTI_SITE_ICOS/RUN_OPTIMISATION_ICOS/SAMIR_OPTIMI_LAM/RUN_final/Merlin_init_ru_fewi_De_Kr_irri_man_soil/2010/Inputdata/maize_irri/meteo.df",'rb'))
    
    
#     # SWC=pd.read_csv("D:/THESE_TMP//TRAITEMENT/DATA_VALIDATION/DATA_SWC/SWC_LAM/SWC_LAM_2010.csv")
#     df_Merlin=pickle.load(open("/run/media/pageot/Transcend/Yann_THESE/BESOIN_EAU/BESOIN_EAU/TRAITEMENT/RUNS_SAMIR/RUN_MULTI_SITE_ICOS/RUN_OPTIMISATION_ICOS/SAMIR_OPTIMI_LAM/RUN_final/Merlin_init_ru_fewi_De_Kr_irri_man_soil_test/2010/Output/maxZr/output_test_maize_irri_0.df",'rb'))
#     df_FAO=pickle.load(open("/run/media/pageot/Transcend/Yann_THESE/BESOIN_EAU/BESOIN_EAU/TRAITEMENT/RUNS_SAMIR/RUN_MULTI_SITE_ICOS/RUN_OPTIMISATION_ICOS/SAMIR_OPTIMI_LAM/RUN_final/FAO_init_ru_fewi_De_Kr_irri_man_soil/2010/Output/maxZr/output_test_maize_irri_0.df",'rb'))
#     df_Merlin.date=pd.to_datetime(df_Merlin.date,format="%Y-%m-%d")
#     df_Merlin=df_Merlin.loc[(df_Merlin.date >= "2010-04-01") &(df_Merlin.date <= "2010-09-30")]
#     df_FAO.date=pd.to_datetime(df_FAO.date,format="%Y-%m-%d")
#     df_FAO=df_FAO.loc[(df_FAO.date >= "2010-04-01") &(df_FAO.date <= "2010-09-30")]
#     # SWC["Date"]=pd.to_datetime(SWC["Date/Time"],format="%Y-%m-%d")
#     # SWC=SWC.loc[(SWC.Date >= "2010-04-01") &(SWC.Date <= "2010-09-30")]
    
#     plt.plot(df_Merlin.date,df_Merlin.Dei,label="Merlin")
#     plt.plot(df_FAO.date,df_FAO.Dei,label="FAO")
#     plt.legend()
#     plt.plot(df_Merlin.date,df_Merlin.Dr,label="Merlin")
#     plt.plot(df_FAO.date,df_FAO.Dr,label="FAO")
#     plt.legend()
#     plt.plot(df_Merlin.date,0.311/2+df_Merlin.SWC1*33/150,label="Merlin")
#     plt.plot(df_FAO.date,0.311/2+df_FAO.SWC1*33/150,label="FAO")
#     # plt.plot(SWC.Date,SWC.SWC_5_moy/100,label="OBS",marker="x",linestyle="")
#     plt.legend()
#     plt.plot(df_Merlin["date"],np.repeat(0.172,len(df_Merlin["date"])),c="r",label="WP")
#     plt.plot(df_Merlin["date"],np.repeat(0.363,len(df_Merlin["date"])),c="b", label="FC")
    
#     prec=pickle.load(open("/run/media/pageot/Transcend/Yann_THESE/BESOIN_EAU/BESOIN_EAU/TRAITEMENT/RUNS_SAMIR/RUN_MULTI_SITE_ICOS/RUN_OPTIMISATION_ICOS/SAMIR_OPTIMI_LAM/RUN_final/Merlin_init_ru_fewi_De_Kr_irri_man_soil/2015/Inputdata/maize_irri/meteo.df",'rb'))
#     df_Merlin=pickle.load(open("H:/Yann_THESE/BESOIN_EAU/BESOIN_EAU/TRAITEMENT/RUNS_SAMIR/RUN_MULTI_SITE_ICOS/RUN_OPTIMISATION_ICOS/SAMIR_OPTIMI_LAM/RUN_final/Merlin_init_ru_fewi_De_Kr_irri_man_soil/2015/Output/maxZr/output_test_maize_irri_0.df",'rb'))
#     df_FAO=pickle.load(open("H:/Yann_THESE/BESOIN_EAU/BESOIN_EAU/TRAITEMENT/RUNS_SAMIR/RUN_MULTI_SITE_ICOS/RUN_OPTIMISATION_ICOS/SAMIR_OPTIMI_LAM/RUN_final/FAO_init_ru_fewi_De_Kr_irri_man_soil/2015/Output/maxZr/output_test_maize_irri_0.df",'rb'))
#     plt.plot(df_Merlin.date,df_Merlin.SWC1,label="Merlin")
#     plt.plot(df_FAO.date,df_FAO.SWC1,label="FAO")
#     plt.legend()

# =============================================================================
# # # Plot KE/De
# =============================================================================

    # for y in ['2008','2010','2012','2014','2015',"2019"]:
    #     SWC=pd.read_csv(d["PC_labo"]+"TRAITEMENT/DATA_VALIDATION/DATA_SWC/SWC_LAM/SWC_LAM_"+str(y)+".csv")
    #     SWC["Date/Time"]=pd.to_datetime(SWC["Date/Time"],format="%Y-%m-%d")
    #     SWC=SWC.loc[(SWC["Date/Time"] >= str(y)+"-04-01") & (SWC["Date/Time"] <= str(y)+"-09-30")]
    #     SWC50=SWC.drop(SWC[(SWC.SWC_50 <= 0)].index)
    #     SWC100=SWC.drop(SWC[(SWC.SWC_100 <= 0)].index)
    #     prec=pickle.load(open("/run/media/pageot/Transcend/Yann_THESE/BESOIN_EAU/BESOIN_EAU/TRAITEMENT/RUNS_SAMIR/RUN_MULTI_SITE_ICOS/RUN_OPTIMISATION_ICOS/SAMIR_OPTIMI_LAM/RUN_final/Merlin_init_ru_fewi_De_Kr_irri_man_soil/"+y+"/Inputdata/maize_irri/meteo.df",'rb'))
    #     prec.date=pd.to_datetime(prec.date,format="%Y-%m-%d")
    #     prec=prec.loc[(prec.date >= str(y)+"-04-01") &(prec.date <= str(y)+"-09-30")]
        
    #     SWC["Half_Hum"]=0.20 + 0.28 * 0.2 - 0.16* 0.33
    #     SWC["SAT_Hum"]=0.489 - 0.126 * 0.33
    #     SWC["P_Melin"]=np.log(0.5)/np.log(0.5 - 0.5 * np.cos(np.pi*SWC.Half_Hum/SWC.SAT_Hum))
        ETR_all
    #     df_Merlin=pickle.load(open("/run/media/pageot/Transcend/Yann_THESE/BESOIN_EAU/BESOIN_EAU/TRAITEMENT/RUNS_SAMIR/RUN_MULTI_SITE_ICOS/RUN_OPTIMISATION_ICOS/SAMIR_OPTIMI_LAM/RUN_final/Merlin_init_ru_fewi_De_Kr_irri_man_soil_V3/"+y+"/Output/maxZr/output_test_maize_irri_0.df",'rb'))
    #     df_Merlin_auto=pickle.load(open("/run/media/pageot/Transcend/Yann_THESE/BESOIN_EAU/BESOIN_EAU/TRAITEMENT/RUNS_SAMIR/RUN_MULTI_SITE_ICOS/RUN_OPTIMISATION_ICOS/SAMIR_OPTIMI_LAM/RUN_final/Merlin_init_ru_optim_fewi_De_Kr_Fcover_irri_auto_soil/"+y+"/Output/maxZr/output_test_maize_irri_8.df",'rb'))

    #     df_FAO=pickle.load(open("/run/media/pageot/Transcend/Yann_THESE/BESOIN_EAU/BESOIN_EAU/TRAITEMENT/RUNS_SAMIR/RUN_MULTI_SITE_ICOS/RUN_OPTIMISATION_ICOS/SAMIR_OPTIMI_LAM/RUN_final/FAO_init_ru_fewi_De_Kr_irri_man_soil_test/"+y+"/Output/maxZr/output_test_maize_irri_0.df",'rb'))
    #     df_Merlin.date=pd.to_datetime(df_Merlin.date,format="%Y-%m-%d")
    #     df_Merlin=df_Merlin.loc[(df_Merlin.date >= str(y)+"-04-01") &(df_Merlin.date <= str(y)+"-09-30")]
    #     df_Merlin_auto.date=pd.to_datetime(df_Merlin_auto.date,format="%Y-%m-%d")
    #     df_Merlin_auto=df_Merlin_auto.loc[(df_Merlin_auto.date >= str(y)+"-04-01") &(df_Merlin_auto.date <= str(y)+"-09-30")]
    #     df_FAO.date=pd.to_datetime(df_FAO.date,format="%Y-%m-%d")
    #     df_FAO=df_FAO.loc[(df_FAO.date >= str(y)+"-04-01") &(df_FAO.date <= str(y)+"-09-30")]
    #     df_Merlin["Irri"]=0
    #     df_Merlin.loc[(df_Merlin.Irrig != 0.0),'Irri']= 0.45
    #     df_Merlin_auto["Irri"]=0
    #     df_Merlin_auto.loc[(df_Merlin_auto.Ir_auto != 0.0),'Irri']= 0.45
    #     # df_Merlin["Irri"]=0.5
        
    #     # if y =="2019":
    #     #     SWC["KR"]=np.where(SWC.SWC_10 <= SWC.SAT_Hum,np.power(1/2-1/2*np.cos(np.pi*(SWC.SWC_10/SWC.SAT_Hum)), SWC.P_Melin.values),1)
    #     # else:
    #     #     SWC["KR"]=np.where(SWC.SWC_10_moy/100 <= SWC.SAT_Hum,np.power(1/2-1/2*np.cos(np.pi*(SWC.SWC_10_moy/100/SWC.SAT_Hum)), SWC.P_Melin.values),1)
    
    #     # if y =='2015':
    #     #     SWC=SWC.loc[(SWC["Date/Time"] >= str(y)+"-05-15") &(SWC["Date/Time"] <= str(y)+"-09-06")]
    #     #     df_Merlin=df_Merlin.loc[(df_Merlin.date >= str(y)+"-05-15") &(df_Merlin.date <= str(y)+"-09-06")]
    #     # elif y =='2012':
    #     #     SWC=SWC.loc[(SWC["Date/Time"] >= str(y)+"-05-03") &(SWC["Date/Time"] <= str(y)+"-08-22")]
    #     #     df_Merlin=df_Merlin.loc[(df_Merlin.date >= str(y)+"-05-03") &(df_Merlin.date <= str(y)+"-08-22")]
            
    #     # elif y == '2014':
    #     # ETR_all    SWC=SWC.loc[(SWC["Date/Time"] >= str(y)+"-05-21") &(SWC["Date/Time"] <= str(y)+"-09-18")]
    #     #     df_Merlin=df_Merlin.loc[(df_Merlin.date >= str(y)+"-05-21") &(df_Merlin.date <= str(y)+"-09-18")]
        
    #     # elif y == '2019':
    #     #     SWC=SWC.loc[(SWC["Date/Time"] >= str(y)+"-04-29") &(SWC["Date/Time"] <= str(y)+"-08-28")]
    #     #     df_Merlin=df_Merlin.loc[(df_Merlin.date >= str(y)+"-04-29") &(df_Merlin.date <= str(y)+"-08-28")]
    #     # plt.figure(figsize=(7,7))
    #     # plt.plot(df_Merlin.Dei,df_Merlin.Kri,label="Kri Merlin",marker='o',linestyle="")
    #     # # plt.plot(df_Merlin.Dei,df_FAO.Kri,label="Kri Merlin",marker='o',linestyle="")
    #     # plt.plot(df_Merlin.Dei,SWC.KR,label="Kri Obs.",marker='o',linestyle="",c='Black')
    #     # # plt.plot(df_Merlin.date,df_Merlin.Kri,label="Kri Merlin")
    #     # # plt.plot(SWC["Date/Time"],SWC.KR.rolling(5).mean(),label="Kri Obs",linestyle="--")
    #     # plt.legend()
    #     # plt.ylim(0,1.5)
    #     # plt.ylabel("Kr")
    #     # plt.xlabel("De")
    #     # plt.title(str(y))
    #     # plt.savefig("/run/media/pageot/Transcend/Yann_THESE/BESOIN_EAU/BESOIN_EAU/TRAITEMENT/RUNS_SAMIR/RUN_MULTI_SITE_ICOS/RUN_OPTIMISATION_ICOS/SAMIR_OPTIMI_LAM/RUN_final/Merlin_V2_init_ru_optim_Kcmax30_fewi_De_Kr_irri_man_soil/dynamique_Kr_Dei_oBS_model_Merlin_%s.png"%(y),dpi=330)
        

    #     plt.figure(figsize=(15,4))
    #     # plt.plot(df_Merlin.date,df_Merlin.SWCvol1,label="Merlin",linestyle='dotted',c="Black")
    #     # plt.plot(df_Merlin.date,df.Dei,label="merlin")
    #     # plt.plot(df_Merlin.date,df_FAO.SWCvol1,linestyle="dotted",label='FAO')
    #     if y == "2019" :
    #         plt.plot(SWC["Date/Time"],np.repeat(0.172,len(SWC["Date/Time"])),c="r",label="WP",linestyle="--")
    #         plt.plot(SWC["Date/Time"],np.repeat(0.363,len(SWC["Date/Time"])),c="b", label="FC",linestyle="--")
    #         # plt.plot(SWC["Date/Time"],np.repeat(0.558,len(SWC["Date/Time"])),c="black", label="Sat",linestyle="--")
    #         plt.plot(SWC["Date/Time"],SWC.SWC_0,label="SWC prof 0 cm")
    #         plt.plot(SWC["Date/Time"],SWC.SWC_5,label="SWC prof 5 cm")
    #         plt.plot(SWC["Date/Time"],SWC.SWC_10,label="SWC prof 10 cm")
    #         # plt.plot(df_Merlin.date,df_Merlin.SWCvol1, label ="SWC Ze modélisé",linestyle="dotted")
    #         plt.plot(df_Merlin[df_Merlin.Irri!=0]["date"],df_Merlin[df_Merlin.Irri!=0]["Irri"],Marker='o',markersize=8,linewidth=1, label ="Irrigation forcée",linestyle="")
    #         plt.plot(df_Merlin_auto[df_Merlin_auto.Irri!=0]["date"],df_Merlin_auto[df_Merlin_auto.Irri!=0]["Irri"],Marker='x',markersize=8,linewidth=1, label ="Irrigation auto",linestyle="",c='green')
    #         plt.legend(loc="upper right",ncol=2)
    #         plt.ylim(0,0.7)
    #         plt.ylabel("Contenu en eau (en m3.m-3) ")
    #         ax2 = plt.twinx()
    #         ax2.bar(prec.date,prec.Prec,label="Précipitation",color='blue')
    #         ax2.set_ylim(0,100)
    #         ax2.set_ylabel("Précipitation en mm")
    #         plt.legend(loc="upper left")
    #     # plt.ylim(0,0.7)
            
    #     else:
    #         plt.plot(SWC["Date/Time"],np.repeat(0.172,len(SWC["Date/Time"])),c="r",label="WP",linestyle="--")
    #         plt.plot(SWC["Date/Time"],np.repeat(0.363,len(SWC["Date/Time"])),c="b", label="FC",linestyle="--")
    #         # plt.plot(SWC["Date/Time"],np.repeat(0.558,len(SWC["Date/Time"])),c="black", label="Sat",linestyle="--")
    #         plt.plot(SWC["Date/Time"],SWC.SWC_0_moy/100,label="SWC prof 0 cm")
    #         plt.plot(SWC["Date/Time"],SWC.SWC_5_moy/100,label="SWC prof 5 cm")
    #         plt.plot(SWC["Date/Time"],SWC.SWC_10_moy/100,label="SWC prof 10 cm")
    #         # plt.plot(df_Merlin.date,df_Merlin.SWCvol1, label ="SWC Ze modélisé",linestyle="dotted")
    #         plt.plot(df_Merlin[df_Merlin.Irri!=0]["date"],df_Merlin[df_Merlin.Irri!=0]["Irri"],Marker='o',markersize=8,linewidth=1, label ="Irrigation forcée",linestyle="")
    #         plt.plot(df_Merlin_auto[df_Merlin_auto.Irri!=0]["date"],df_Merlin_auto[df_Merlin_auto.Irri!=0]["Irri"],Marker='x',markersize=8,linewidth=1, label ="Irrigation auto",linestyle="",c='green')
    #         plt.legend(loc="upper right",ncol=2)
    #         plt.ylim(0,0.7)
    #         plt.ylabel("Contenu en eau (en m3.m-3) ")
    #         ax2 = plt.twinx()
    #         ax2.bar(prec.date,prec.Prec,label="Précipitation",color='blue')
    #         ax2.set_ylim(0,100)
    #         ax2.set_ylabel("Précipitation en mm")
    #         plt.legend(loc="upper left")
    #     # plt.ylim(0,0.7)
    #     plt.title(y)
    #     plt.savefig("/run/media/pageot/Transcend/Yann_THESE/BESOIN_EAU/BESOIN_EAU/TRAITEMENT/RUNS_SAMIR/RUN_MULTI_SITE_ICOS/RUN_OPTIMISATION_ICOS/SAMIR_OPTIMI_LAM/RUN_final/Merlin_init_ru_fewi_De_Kr_irri_man_soil_test/dynamique_SWC_prec_ze_irri_obs_%s_large_size_v2.png"%(y),dpi=330)
        
    
    #     plt.figure(figsize=(15,4))
    #     # plt.plot(df_Merlin.date,df_Merlin.SWCvol1,label="Merlin",linestyle='dotted',c="Black")
    #     # plt.plot(df_Merlin.date,df.Dei,label="merlin")
    #     # plt.plot(df_Merlin.date,df_FAO.SWCvol1,linestyle="dotted",label='FAO')
    #     if y == "2019" :
    #         plt.plot(SWC["Date/Time"],np.repeat(0.172,len(SWC["Date/Time"])),c="r",label="WP",linestyle="--")
    #         plt.plot(SWC["Date/Time"],np.repeat(0.363,len(SWC["Date/Time"])),c="b", label="FC",linestyle="--")
    #         # plt.plot(SWC["Date/Time"],np.repeat(0.558,len(SWC["Date/Time"])),c="black", label="Sat",linestyle="--")
    #         plt.plot(SWC["Date/Time"],np.repeat(0.172+0.07,len(SWC["Date/Time"])),c="green", label="RAW",linestyle="--")
    #         plt.plot(SWC["Date/Time"],SWC.SWC_30,label="SWC prof 30 cm")
    #         plt.plot(SWC50["Date/Time"],SWC50.SWC_50,label="SWC prof 50 cm")
    #         plt.plot(SWC100["Date/Time"],SWC100.SWC_100,label="SWC prof 100 cm")
    #         plt.ylabel("Contenu en eau (en m3.m-3) ")
    #         plt.legend()
    #         plt.ylim(0.15,0.65)
    #         # plt.plot(df_Merlin.date,df_Merlin.SWCvol2, label ="SWC Zr modélisé",linestyle="dotted")
    #         # plt.plot(df_Merlin.date,df_Merlin.DP, label ="SWC Zr modélisé",linestyle="solid")
    #         # plt.plot(df_Merlin.date,df_Merlin.Irrig, label ="SWC Zr modélisé",linestyle="",marker="*")
    #         ax2 = plt.twinx()
    #         ax2.plot(df_Merlin.date,df_Merlin.Ks,label="Ks",c='red')
    #         ax2.set_ylim(-1,1.1)
    #         ax2.set_ylabel("Coefficient Ks")
    #         plt.legend()
    #     else:
    #         plt.plot(SWC["Date/Time"],np.repeat(0.172,len(SWC["Date/Time"])),c="r",label="WP",linestyle="--")
    #         plt.plot(SWC["Date/Time"],np.repeat(0.363,len(SWC["Date/Time"])),c="b", label="FC",linestyle="--")
    #         # plt.plot(SWC["Date/Time"],np.repeat(0.558,len(SWC["Date/Time"])),c="black", label="Sat",linestyle="--")
    #         # plt.plot(SWC["Date/Time"],np.repeat(0.175+0.07,len(SWC["Date/Time"])),c="green", label="RAW",linestyle="--")
    #         plt.plot(SWC["Date/Time"],SWC.SWC_30_moy/100,label="SWC prof 30 cm")
    #         plt.plot(SWC["Date/Time"],SWC.SWC_50/100,label="SWC prof 50 cm")
    #         plt.plot(SWC["Date/Time"],SWC.SWC_100/100,label="SWC prof 100 cm")
    #         # plt.plot(df_Merlin.date,df_Merlin.SWCvol2, label ="SWC Zr modélisé",linestyle="dotted")
    #         plt.ylim(0.15,0.65)
    #         plt.ylabel("Contenu en eau (en m3.m-3) ")
    #         plt.legend()
    #         ax2 = plt.twinx()
    #         ax2.plot(df_Merlin.date,df_Merlin.Ks,label="Ks",c='red')
    #         ax2.set_ylim(-1,1.1)
    #         ax2.set_ylabel("Coefficient Ks")
    #         plt.legend()
    #     # plt.ylabel("Contenu en eau (en m3.m-3) ")
    #     # plt.ylim(0,0.7)
    #     plt.title(y)
    #     plt.savefig("/run/media/pageot/Transcend/Yann_THESE/BESOIN_EAU/BESOIN_EAU/TRAITEMENT/RUNS_SAMIR/RUN_MULTI_SITE_ICOS/RUN_OPTIMISATION_ICOS/SAMIR_OPTIMI_LAM/RUN_final/Merlin_init_ru_fewi_De_Kr_irri_man_soil_test/dynamique_SWC_Zr_Ks_model_obs_%s_large_size_v2.png"%(y),dpi=330)
        

# =============================================================================
# Irri auto vs man SWC2 combien
# =============================================================================
    # for y in ["2008","2010","2012","2014","2015","2019"]:
    #     SWC=pd.read_csv(d["PC_labo"]+"TRAITEMENT/DATA_VALIDATION/DATA_SWC/SWC_LAM/SWC_LAM_"+str(y)+".csv")
    #     SWC["Date/Time"]=pd.to_datetime(SWC["Date/Time"],format="%Y-%m-%d")
    #     SWC=SWC.loc[(SWC["Date/Time"] >= str(y)+"-04-01") & (SWC["Date/Time"] <= str(y)+"-09-30")]
    #     SWC50=SWC.drop(SWC[(SWC.SWC_50 <= 0)].index)
    #     SWC100=SWC.drop(SWC[(SWC.SWC_100 <= 0)].index)
    #     prec=pickle.load(open("/run/media/pageot/Transcend/Yann_THESE/BESOIN_EAU/BESOIN_EAU/TRAITEMENT/RUNS_SAMIR/RUN_MULTI_SITE_ICOS/RUN_OPTIMISATION_ICOS/SAMIR_OPTIMI_LAM/RUN_final/Merlin_init_ru_fewi_De_Kr_irri_man_soil/"+y+"/Inputdata/maize_irri/meteo.df",'rb'))
    #     prec.date=pd.to_datetime(prec.date,format="%Y-%m-%d")
    #     prec=prec.loc[(prec.date >= str(y)+"-04-01") &(prec.date <= str(y)+"-09-30")]
        
    #     df_Merlin=pickle.load(open("/run/media/pageot/Transcend/Yann_THESE/BESOIN_EAU/BESOIN_EAU/TRAITEMENT/RUNS_SAMIR/RUN_MULTI_SITE_ICOS/RUN_OPTIMISATION_ICOS/SAMIR_OPTIMI_LAM/RUN_final/Merlin_init_ru_fewi_De_Kr_Fcover_irri_man_soil_V3/"+y+"/Output/maxZr/output_test_maize_irri_4.df",'rb'))
    #     df_Merlin_auto=pickle.load(open("/run/media/pageot/Transcend/Yann_THESE/BESOIN_EAU/BESOIN_EAU/TRAITEMENT/RUNS_SAMIR/RUN_MULTI_SITE_ICOS/RUN_OPTIMISATION_ICOS/SAMIR_OPTIMI_LAM/RUN_final/Merlin_init_ru_fewi_De_Kr_Fcover_irri_auto_soil_V3/"+y+"/Output/maxZr/output_test_maize_irri_8.df",'rb'))
    #     df_Merlin.date=pd.to_datetime(df_Merlin.date,format="%Y-%m-%d")
    #     df_Merlin=df_Merlin.loc[(df_Merlin.date >= str(y)+"-04-01") &(df_Merlin.date <= str(y)+"-09-30")]
    #     df_Merlin_auto.date=pd.to_datetime(df_Merlin_auto.date,format="%Y-%m-%d")
    #     df_Merlin_auto=df_Merlin_auto.loc[(df_Merlin_auto.date >= str(y)+"-04-01") &(df_Merlin_auto.date <= str(y)+"-09-30")]

    #     # plt.figure(figsize=(7,7))
    #     # plt.plot(df_Merlin.date,df_Merlin.SWC2,label="SW plt.plot(id1.date,id1.NDVI)C man")
    #     # plt.plot(df_Merlin_auto.date,df_Merlin_auto.SWC2,label="SWC auto")
    #     # plt.legend()
    #     # plt.figure(figsize=(7,7))
    #     # plt.plot(df_Merlin.date,df_Merlin.Irrig,label="SWC man",linestyle='',marker="o")
    #     # plt.plot(df_Merlin_auto.date,df_Merlin_auto.Ir_auto,label="SWC auto",linestyle='',marker="<")
    #     # plt.legend()
    #     print(y)
    #     print(r'Irrig_man DP : %s'%df_Merlin.DP.sum())
    #     print(r'Irrig_auto DP : %s'%df_Merlin_auto.DP.sum())
    #     print(r'Irrig_man Irrig : %s'%df_Merlin.Irrig.sum())
    #     print(r'Irrig_auto Irrig : %s'%df_Merlin_auto.Ir_auto.sum())
    #     print(r'Irrig_auto NB Irrig : %s'%df_Merlin_auto[df_Merlin_auto.Ir_auto>0]["Ir_auto"].count())
    #     print(r'Prec : %s'%prec.Prec.sum())
# for y in ["2008","2010","2012","2014","2015"]:
#     plt.xlim(0,0.5)
#     plt.ylim(0,0.5)
#     SWC=pd.read_csv(d["PC_labo"]+"TRAITEMENT/DATA_VALIDATION/DATA_SWC/SWC_LAM/SWC_LAM_"+str(y)+".csv")
#     SWC["Date/Time"]=pd.to_datetime(SWC["Date/Time"],format="%Y-%m-%d")
#     SWC=SWC.loc[(SWC["Date/Time"] >= str(y)+"-04-01") &(SWC["Date/Time"] <= str(y)+"-06-30")]
#     prec=pickle.load(open("/run/media/pageot/Transcend/Yann_THESE/BESOIN_EAU/BESOIN_EAU/TRAITEMENT/RUNS_SAMIR/RUN_MULTI_SITE_ICOS/RUN_OPTIMISATION_ICOS/SAMIR_OPTIMI_LAM/RUN_final/Merlin_init_ru_fewi_De_Kr_irri_man_soil/"+y+"/Inputdata/maize_irri/meteo.df",'rb'))

#     df_Merlin=pickle.load(open("/run/media/pageot/Transcend/Yann_THESE/BESOIN_EAU/BESOIN_EAU/TRAITEMENT/RUNS_SAMIR/RUN_MULTI_SITE_ICOS/RUN_OPTIMISATION_ICOS/SAMIR_OPTIMI_LAM/RUN_final/Merlin_init_ru_fewi_De_Kr_irri_man_soil_test/"+y+"/Output/maxZr/output_test_maize_irri_8.df",'rb'))
#     df_FAO=pickle.load(open("/run/media/pageot/Transcend/Yann_THESE/BESOIN_EAU/BESOIN_EAU/TRAITEMENT/RUNS_SAMIR/RUN_MULTI_SITE_ICOS/RUN_OPTIMISATION_ICOS/SAMIR_OPTIMI_LAM/RUN_final/FAO_init_ru_fewi_De_Kr_irri_man_soil/"+y+"/Output/maxZr/output_test_maize_irri_0.df",'rb'))
#     df_Merlin.date=pd.to_datetime(df_Merlin.date,format="%Y-%m-%d")
#     df_Merlin=df_Merlin.loc[(df_Merlin.date >= str(y)+"-04-01") &(df_Merlin.date <= str(y)+"-06-30")]
#     df_FAO.date=pd.to_datetime(df_FAO.date,format="%Y-%m-%d")
#     df_FAO=df_FAO.loc[(df_FAO.date >= str(y)+"-04-01") &(df_FAO.date <= str(y)+"-06-30")]
#     if y =='2019' :
#         plt.scatter(df_Merlin.SWCvol1,SWC.SWC_10)
#     else:
#         plt.scatter(df_Merlin.SWCvol1,SWC.SWC_10_moy/100)


# =============================================================================
#  Plot soutenance
# =============================================================================
    # ETR_all.columns=ETR_all.columns.droplevel()
    ETR_all.reset_index(inplace=True)
    ETR_all.columns=["date","ETR"]
    ETR_all_FAO=ETR_all
    
    # ETR_all_Merlin.columns=ETR_all_Merlin.columns.droplevel()
    ETR_all_Merlin.reset_index(inplace=True)
    ETR_all_Merlin.columns=["date","ETR"]
    
    tmp_valid=ETR_mod_all.LE_Bowen.dropna()
    tmp_FAO=ETR_all_FAO.ETR.dropna()
    tmp_Merlin=ETR_all_Merlin.ETR.dropna()
    list_marker=[".","v","s","*","d","h"]

    plt.figure(figsize=(7,7))
    slope, intercept, r_value, p_value, std_err = stats.linregress(tmp_valid.to_list(),tmp_FAO.to_list())
    bias=1/tmp_valid.shape[0]*sum(tmp_FAO-np.mean(tmp_valid)) 
    rms = mean_squared_error(tmp_valid,tmp_FAO)
    plt.plot([0.0, 10], [0.0,10], 'black', lw=1,linestyle='--')
    plt.xlabel("ETR Obs")
    plt.ylabel("ETR simulée")
    plt.xlim(0,10)
    plt.ylim(0,10)
    for y,m in zip(["2008","2010","2012","2014","2015","2019"],list_marker):
        periode1=(ETR_mod_all['date'] > str(y)+'-04-01') & (ETR_mod_all['date'] <= str(y)+'-09-30')
        periode2=(ETR_all_FAO['date'] > str(y)+'-04-01') & (ETR_all_FAO['date'] <= str(y)+'-09-30')
        plt.scatter(ETR_mod_all.loc[periode1]["LE_Bowen"], ETR_all_FAO.loc[periode2]["ETR"], zorder = 2,color="black",marker=m,label=y)
    plt.title("Scatter ETR")
    rectangle = plt.Rectangle((0.9, 8.6),2.1,1.3, ec='red',fc='red',alpha=0.1)
    plt.gca().add_patch(rectangle)
    plt.text(1,9.6,"RMSE = "+str(round(rms,2)),fontsize=12) 
    plt.text(1,9.3,"R = "+str(round(r_value,2)),fontsize=12)
    plt.text(1,9.0,"Pente = "+str(round(slope,2)),fontsize=12)
    plt.text(1,8.7,"Biais = "+str(round(bias,2)),fontsize=12)
    plt.legend()
    plt.savefig(d["Output_model_PC_home_disk"]+"/plt_scatter_all_years_Merlin_vide.png",dpi=330)


    plt.figure(figsize=(7,7))
    slope, intercept, r_value, p_value, std_err = stats.linregress(tmp_valid.to_list(),tmp_Merlin.to_list())
    bias=1/tmp_valid.shape[0]*sum(tmp_Merlin-np.mean(tmp_valid)) 
    rms = mean_squared_error(tmp_valid,tmp_Merlin)
    plt.plot([0.0, 10], [0.0,10], 'black', lw=1,linestyle='--')
    plt.xlabel("ETR Obs")
    plt.ylabel("ETR simulée")
    plt.xlim(0,10)
    plt.ylim(0,10)
    for y,m in zip(["2008","2010","2012","2014","2015","2019"],list_marker):
        periode1=(ETR_mod_all['date'] > str(y)+'-04-01') & (ETR_mod_all['date'] <= str(y)+'-09-30')
        periode2=(ETR_all_Merlin['date'] > str(y)+'-04-01') & (ETR_all_Merlin['date'] <= str(y)+'-09-30')
        plt.scatter(ETR_mod_all.loc[periode1]["LE_Bowen"], ETR_all_Merlin.loc[periode2]["ETR"], zorder = 2,color="black",marker=m,label=y)
    plt.title("Scatter ETR")
    rectangle = plt.Rectangle((0.9, 8.6),2.1,1.3, ec='red',fc='red',alpha=0.1)
    plt.gca().add_patch(rectangle)
    plt.text(1,9.6,"RMSE = "+str(round(rms,2)),fontsize=12) 
    plt.text(1,9.3,"R = "+str(round(r_value,2)),fontsize=12)
    plt.text(1,9.0,"Pente = "+str(round(slope,2)),fontsize=12)
    plt.text(1,8.7,"Biais = "+str(round(bias,2)),fontsize=12)
    plt.legend()
    plt.savefig(d["Output_model_PC_home_disk"]+"/plt_scatter_all_years_Merlin_RU_pleine.png",dpi=330)
    
    
    # plt.figure(figsize=(14,5))
    # plt.plot(ETR_mod_all.date,ETR_mod_all.LE_Bowen)
    # plt.plot(ETR_mod_all.date,ETR_all_FAO.ETR)
    # plt.plot(ETR_mod_all.date,ETR_all_Merlin.ETR)
