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
    name_run_save_fig="RUNS_SAMIR/RUN_MULTI_SITE_ICOS/RUN_OPTIMISATION_ICOS/SAMIR_OPTIMI_LAM/RUN_final/Init_ru/Merlin_init_ru_0_optim_fewi_De_Kr_irri_man_soil"
    # d["PC_disk"]="/run/media/pageot/Transcend/Yann_THESE/BESOIN_EAU/BESOIN_EAU/"
    d["PC_home"]="/mnt/d/THESE_TMP/"
    d["PC_home_Wind"]="D:/THESE_TMP/"
    d["PC_disk"]="H:/Yann_THESE/BESOIN_EAU/BESOIN_EAU/"
    d["PC_labo"]="/datalocal/vboxshare/THESE/BESOIN_EAU/"
    label_ref="Init ru vide"
    label_test="Init ru pleine"
    years=["2008","2010","2012","2014","2015","2019"]
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

    # d["Output_model_save_fig"]="/run/media/pageot/Transcend/Yann_THESE/BESOIN_EAU/BESOIN_EAU/TRAITEMENT/"+name_run_save_fig
    d["Output_model_save_fig"]="H:/Yann_THESE/BESOIN_EAU/BESOIN_EAU/TRAITEMENT/"+name_run_save_fig

    ETR_test_m20=None
    ETR_test_pl20=None
    ETR_test=None
    
    ETR_test_FAO_m20=None
    ETR_test_FAO_pl20=None
    ETR_test_FAO=None
    for y in years:
        ET0=pickle.load(open("H:/Yann_THESE/BESOIN_EAU/BESOIN_EAU/TRAITEMENT/RUNS_SAMIR/RUN_MULTI_SITE_ICOS/RUN_OPTIMISATION_ICOS/SAMIR_OPTIMI_LAM/RUN_final/Merlin_init_ru_optim_fewi_De_Kr_Fcover_irri_man_soil/"+str(y)+"/Inputdata/maize_irri/meteo.df","rb"))
        data_E=ET0[["date",'ET0',"Prec"]]
        data_meteo=data_E.loc[(data_E.date>= str(y)+"-04-01") &(data_E.date <= str(y)+"-09-30")]
        if "Fcover" in name_run_merlin:
            for Fco in ["_pl20","_m20",""]:
                d["Output_model_PC_home_disk"]="H:/Yann_THESE/BESOIN_EAU/BESOIN_EAU/TRAITEMENT/"+name_run_merlin[0:128]+str(Fco)+name_run_merlin[128:]
                ####  récupération num_run max incertitude
                ETR_mod=pd.read_csv(d["Output_model_PC_home_disk"]+"/LUT_ETR"+str(y)+".csv",index_col=[0,1])
                ETR_mod.columns=pd.to_datetime(ETR_mod.columns,format="%Y-%m-%d")
                ETR_mod=ETR_mod.loc[:,(ETR_mod.columns >= str(y)+"-04-01") &(ETR_mod.columns <= str(y)+"-09-30")]
                ETR_mod_max=pd.read_csv(d["Output_model_PC_home_disk"]+"_max"+"/LUT_ETR"+str(y)+".csv",index_col=[0,1])
                ETR_mod_max.columns=pd.to_datetime(ETR_mod_max.columns,format="%Y-%m-%d")
                ETR_mod_max=ETR_mod_max.loc[:,(ETR_mod_max.columns >= str(y)+"-04-01") &(ETR_mod_max.columns <= str(y)+"-09-30")]
                ETR_mod_min=pd.read_csv(d["Output_model_PC_home_disk"]+"_min"+"/LUT_ETR"+str(y)+".csv",index_col=[0,1])
                ETR_mod_min.columns=pd.to_datetime(ETR_mod_min.columns,format="%Y-%m-%d")
                ETR_mod_min=ETR_mod_min.loc[:,(ETR_mod_min.columns >= str(y)+"-04-01") &(ETR_mod_min.columns <= str(y)+"-09-30")]
                globals()["ETR_test%s"%(Fco)]=pd.concat([ETR_mod_max,ETR_mod,ETR_mod_min])
            ETR_test_merlin=pd.concat([ETR_test_pl20,ETR_test_m20,ETR_test])
        else:
            d["Output_model_PC_home_disk"]="H:/Yann_THESE/BESOIN_EAU/BESOIN_EAU/TRAITEMENT/"+name_run_merlin
            ####  récupération num_run max incertitude
            ETR_mod=pd.read_csv(d["Output_model_PC_home_disk"]+"/LUT_ETR"+str(y)+".csv",index_col=[0,1])
            ETR_mod.columns=pd.to_datetime(ETR_mod.columns,format="%Y-%m-%d")
            ETR_mod=ETR_mod.loc[:,(ETR_mod.columns >= str(y)+"-04-01") &(ETR_mod.columns <= str(y)+"-09-30")]
            ETR_mod_max=pd.read_csv(d["Output_model_PC_home_disk"]+"_max"+"/LUT_ETR"+str(y)+".csv",index_col=[0,1])
            ETR_mod_max.columns=pd.to_datetime(ETR_mod_max.columns,format="%Y-%m-%d")
            ETR_mod_max=ETR_mod_max.loc[:,(ETR_mod_max.columns >= str(y)+"-04-01") &(ETR_mod_max.columns <= str(y)+"-09-30")]
            ETR_mod_min=pd.read_csv(d["Output_model_PC_home_disk"]+"_min"+"/LUT_ETR"+str(y)+".csv",index_col=[0,1])
            ETR_mod_min.columns=pd.to_datetime(ETR_mod_min.columns,format="%Y-%m-%d")
            ETR_mod_min=ETR_mod_min.loc[:,(ETR_mod_min.columns >= str(y)+"-04-01") &(ETR_mod_min.columns <= str(y)+"-09-30")]
            ETR_test_merlin=pd.concat([ETR_mod_max,ETR_mod,ETR_mod_min])
        ###### run de ref en X
        if "Fcover" in name_run_FAO:
            for Fco in ["_pl20","_m20",""]:
                d["Output_model_PC_home_disk"]="H:/Yann_THESE/BESOIN_EAU/BESOIN_EAU/TRAITEMENT/"+name_run_FAO[0:130]+str(Fco)+name_run_FAO[130:] # 112 sans optim 
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
            d["Output_model_PC_home_disk"]="H:/Yann_THESE/BESOIN_EAU/BESOIN_EAU/TRAITEMENT/"+name_run_FAO
            ETR_mod=pd.read_csv(d["Output_model_PC_home_disk"]+"/LUT_ETR"+str(y)+".csv",index_col=[0,1])
            ETR_mod.columns=pd.to_datetime(ETR_mod.columns,format="%Y-%m-%d")
            ETR_mod=ETR_mod.loc[:,(ETR_mod.columns >= str(y)+"-04-01") &(ETR_mod.columns <= str(y)+"-09-30")]
            ETR_mod_max=pd.read_csv(d["Output_model_PC_home_disk"]+"_max"+"/LUT_ETR"+str(y)+".csv",index_col=[0,1])
            ETR_mod_max.columns=pd.to_datetime(ETR_mod_max.columns,format="%Y-%m-%d")
            ETR_mod_max=ETR_mod_max.loc[:,(ETR_mod_max.columns >= str(y)+"-04-01") &(ETR_mod_max.columns <= str(y)+"-09-30")]
            ETR_mod_min=pd.read_csv(d["Output_model_PC_home_disk"]+"_min"+"/LUT_ETR"+str(y)+".csv",index_col=[0,1])
            ETR_mod_min.columns=pd.to_datetime(ETR_mod_min.columns,format="%Y-%m-%d")
            ETR_mod_min=ETR_mod_min.loc[:,(ETR_mod_min.columns >= str(y)+"-04-01") &(ETR_mod_min.columns <= str(y)+"-09-30")]
            ETR_test_FAO=pd.concat([ETR_mod_max,ETR_mod,ETR_mod_min])
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
        ETR_obs_rolling_pl20=ETR_rolling.LE_Bowen+(20*ETR_rolling.LE_Bowen/100)
        ETR_obs_rolling_m20=ETR_rolling.LE_Bowen-(20*ETR_rolling.LE_Bowen/100)
        ETR_obs_m20=ETR_obs.LE_Bowen-(20*ETR_obs.LE_Bowen/100)
        ETR_obs_p20=ETR_obs.LE_Bowen+(20*ETR_obs.LE_Bowen/100)
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
        coeff_ks=KS.groupby("date").mean()[["Ks","Kei",'Kep',"fewi","SWC1",'Ir_auto','Dei','Dr','TEW','TAW']]
        # coeff_ks_std=KS.groupby("date").std()[["Ks","Ir_auto"]]
        # FAO 
        ETR_test_FAO_rolling=ETR_test_FAO.T.rolling(5).mean()
        ETR_test_FAO_rolling=ETR_test_FAO_rolling.T
        diff=ETR_test_FAO_rolling.max()-ETR_test_FAO_rolling.min()
        datefor=diff.idxmax() # forte incertitude a cette date
        coup_parm=ETR_test_FAO_rolling.T.loc[ETR_test_FAO_rolling.columns==datefor].T.idxmax() # récupération couple param max 
        coup_parm_min=ETR_test_FAO_rolling.T.loc[ETR_test_FAO_rolling.columns==datefor].T.idxmin()
        #  lecture des output_LUT
        dfmax2=pickle.load(open(d["PC_disk"]+"/TRAITEMENT/"+name_run_FAO+"/"+str(y)+"/Output/maxZr/output_test_maize_irri_"+str(coup_parm.iloc[0][0])+".df","rb"))
        dfmin2=pickle.load(open(d["PC_disk"]+"/TRAITEMENT/"+name_run_FAO+"/"+str(y)+"/Output/maxZr/output_test_maize_irri_"+str(coup_parm_min.iloc[0][0])+".df","rb"))
        dfmax2.date=pd.to_datetime(dfmax2.date,format="%Y-%m-%d")
        dfmin2.date=pd.to_datetime(dfmin2.date,format="%Y-%m-%d")
        dfmax2=dfmax2.loc[(dfmax2.date>= str(y)+"-04-01") &(dfmax2.date <= str(y)+"-09-30")]
        dfmin2=dfmin2.loc[(dfmin2.date>= str(y)+"-04-01") &(dfmin2.date <= str(y)+"-09-30")]
        KS_man=pd.concat([dfmax2,dfmin2])
        coeff_ks_man=KS_man.groupby("date").mean()[["Ks",'Kei','Kep','NDVI',"fewi",'FCov',"SWC1",'Irrig','Dei','Dr','TEW','TAW']]
# =============================================================================
#         # plot dynamique 
# =============================================================================
        plt.figure(figsize=(7,7))
        plt.plot(ETR_rolling.date,ETR_rolling.LE_Bowen,label='ETR obs',color="black",linewidth=1)
        plt.fill_between(ETR_rolling.date, ETR_obs_rolling_m20,ETR_obs_rolling_pl20,alpha=0.5,facecolor="None",ec='black',linestyle="--")
        plt.plot(ETR_test_rolling.T.index,ETR_test_rolling.mean(),label='ETR '+label_test,linewidth=1,color='red')
        plt.fill_between(ETR_test_rolling.T.index, ETR_test_rolling.min(),ETR_test_rolling.max(),alpha=0.2,facecolor="red") # revoir pour mettre l'écart types et non min et max
        # plt.fill_between(ETR_test_rolling.T.index, ETR_test_rolling.mean()-ETR_test_rolling.std(),ETR_test_rolling.mean()+ETR_test_rolling.std(),alpha=0.2,facecolor="green") # avec std
        plt.plot(ETR_test_FAO_rolling.T.index,ETR_test_FAO_rolling.mean(),label='ETR ' +label_ref,linewidth=1,color="Blue")
        plt.fill_between(ETR_test_FAO_rolling.T.index, ETR_test_FAO_rolling.min(),ETR_test_FAO_rolling.max(),alpha=0.2,facecolor="Blue")
        if "Irrigation" in label_ref:
            plt.plot(dfmax.loc[dfmax.Ir_auto>0.0].date,dfmax.Ir_auto.loc[dfmax.Ir_auto>0.0]/10,color='darkgreen',label="Irrigation auto",linestyle="",marker="x")
            plt.plot(coeff_ks_man.loc[coeff_ks_man.Irrig>0.0].index,coeff_ks_man.Irrig.loc[coeff_ks_man.Irrig>0.0]/10,color='Blue',label="Irrigation forcée",linestyle="",marker="+")
        else:
            plt.plot(coeff_ks_man.loc[coeff_ks_man.Irrig>0.0].index,coeff_ks_man.Irrig.loc[coeff_ks_man.Irrig>0.0]/10,color='Blue',label="Irrigation forcée",linestyle="",marker="+")
        plt.ylabel("ETR")
        plt.ylim(0,10)
        plt.legend()
        ax2 = plt.twinx()
        ax2.bar(data_meteo.date,data_meteo.Prec)
        ax2.set_ylim(0,50)
        ax2.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(6))
        plt.ylabel("Pluviométrie")
        # ax2.plot(coeff_ks.index,coeff_ks.Ks,color='r',linestyle="--",label="Ks")
        ax2.set_ylim(0,50)
        # ax2.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(6))
        plt.title("Dynamique ETR moyenne PFCC incertitude %s en %s"%(lc,y))
        plt.savefig(d["Output_model_save_fig"]+"/plt_ETR_mod_mean_rolling_%s_%s.png"%(lc,y),dpi=330)
        
        
        
# =============================================================================
#         # Cumul
# =============================================================================
        plt.figure(figsize=(7,7))
        plt.plot(ETR_obs.date,ETR_obs.LE_Bowen.cumsum(),label='ETR obs',color="black")
        plt.fill_between(ETR_obs.date, ETR_obs_m20.cumsum(),ETR_obs_p20.cumsum(),alpha=0.5,facecolor="None",ec='black',linestyle="--")
        plt.fill_between(ETR_test_merlin.T.index, ETR_test_merlin.min().cumsum(),ETR_test_merlin.max().cumsum(),alpha=0.2,facecolor="red")
        plt.plot(ETR_test_merlin.T.index,ETR_test_merlin.mean().cumsum(),label='ETR '+ label_test,color='red')
        plt.text(ETR_obs.date.iloc[-1], ETR_obs.LE_Bowen.cumsum().iloc[-1], s=round(ETR_obs.LE_Bowen.cumsum().iloc[-1],2))
        plt.text(ETR_test_merlin.T.index[-1], ETR_test_merlin.mean().cumsum().iloc[-1], s=round(ETR_test_merlin.mean().cumsum().iloc[-1],2))
        plt.fill_between(ETR_test_FAO.T.index, ETR_test_FAO.min().cumsum(),ETR_test_FAO.max().cumsum(),alpha=0.2,facecolor="Blue")
        plt.plot(ETR_test_FAO.T.index,ETR_test_FAO.mean().cumsum(),label='ETR '+label_ref,color='Blue')
        plt.text(ETR_test_FAO.T.index[-1], ETR_test_FAO.mean().cumsum().iloc[-1], s=round(ETR_test_FAO.mean().cumsum().iloc[-1],2))
        plt.ylabel("ETR")
        plt.ylim(0,700)
        plt.title("Dynamique ETR cumul incertitude %s en %s"%(lc,y))
        plt.legend(loc="upper left")
        plt.savefig(d["Output_model_save_fig"]+"/plt_Dynamique_ETR_ETR_cumul_incertitude_%s_%s.png"%(lc,y))
        

# =============================================================================
#         Scatter 3 datas
# =============================================================================
        tmp=ETR_rolling.LE_Bowen.dropna()        
        tmp_mod=ETR_test_rolling.mean().dropna()
        tmp_mod_FAO=ETR_test_FAO_rolling.mean().dropna()
    #  Merlin vs obs
        slope, intercept, r_value, p_value, std_err = stats.linregress(tmp.to_list(),tmp_mod.to_list())
        bias=1/tmp.shape[0]*sum(np.mean(tmp)-tmp_mod) 
        fitLine = predict(tmp_mod)
        # Creation plot

        plt.figure(figsize=(7,7))
        plt.plot([0.0, 10], [0.0,10], 'black', lw=1,linestyle='--')
        plt.xlabel("ETR Obs")
        plt.ylabel("ETR simulée")
        plt.xlim(0,10)
        plt.ylim(0,10)
        plt.scatter(ETR_rolling.LE_Bowen, ETR_test_rolling.mean(), zorder = 2,color="r",label="Obs / "+label_test)
        plt.title("Scatter ETR %s"%y)
        rms = mean_squared_error(tmp,tmp_mod)
        rectangle = plt.Rectangle((0.9, 8.6),1.8,1.3, ec='red',fc='red',alpha=0.1)
        plt.gca().add_patch(rectangle)
        plt.text(1,9.6,"RMSE = "+str(round(rms,2))) 
        plt.text(1,9.3,"R² = "+str(round(r_value,2)))
        plt.text(1,9.0,"Pente = "+str(round(slope,2)))
        plt.text(1,8.7,"Biais = "+str(round(bias,2)))
            #  FAo vs obs
        slope, intercept, r_value, p_value, std_err = stats.linregress(tmp.to_list(),tmp_mod_FAO.to_list())
        bias=1/tmp.shape[0]*sum(np.mean(tmp)-tmp_mod_FAO) 
        fitLine = predict(tmp_mod_FAO)
        # Creation plot
        plt.scatter(ETR_rolling.LE_Bowen, ETR_test_FAO_rolling.mean(), zorder = 2,color='b',label="Obs / "+label_ref)
        rms = mean_squared_error(tmp,tmp_mod_FAO)
        rectangle = plt.Rectangle((7.9, 1),1.8,1.4, ec='blue',fc='blue',alpha=0.1)
        plt.gca().add_patch(rectangle)
        plt.text(8,2,"RMSE = "+str(round(rms,2))) 
        plt.text(8,1.7,"R² = "+str(round(r_value,2)))
        plt.text(8,1.4,"Pente = "+str(round(slope,2)))
        plt.text(8,1.1,"Biais = "+str(round(bias,2)))
        plt.legend(loc="upper right")
        plt.savefig(d["Output_model_save_fig"]+"/scatter_ETR_mods_obs_mean_rolling_%s_%s.png"%(lc,y),dpi=330)
# =============================================================================
#          Calcule flux E
# =============================================================================
        Fcov=pickle.load(open("H:/Yann_THESE/BESOIN_EAU/BESOIN_EAU/TRAITEMENT/RUNS_SAMIR/RUN_MULTI_SITE_ICOS/RUN_OPTIMISATION_ICOS/SAMIR_OPTIMI_LAM/RUN_analyse/Merlin_init_ru_Fcover_irri_man_Bruand/"+str(y)+"/Inputdata/maize_irri/FCOVER.df","rb"))
        Fcov=Fcov.loc[(Fcov.date>= str(y)+"-04-01") &(Fcov.date <= str(y)+"-09-30")]
        data_merge=pd.merge(coeff_ks,data_E,on="date")
        data_merge=data_merge.rolling(5).mean()
        data_merge_ss_Fcover=pd.merge(coeff_ks_man,data_E,on="date")
        data_merge_ss_Fcover=data_merge_ss_Fcover.rolling(5).mean()
        fluxE_avec_Fcover=np.maximum((data_merge.Kei+data_merge.Kep)*data_merge.ET0,0)
        fluxE_sans_Fcover=np.maximum((data_merge_ss_Fcover.Kei+data_merge_ss_Fcover.Kep)*data_merge.ET0,0)
        plt.figure(figsize=(7,7))
        plt.plot(Fcov.date,fluxE_avec_Fcover,label="Eva " +label_test)
        plt.plot(Fcov.date,fluxE_sans_Fcover,label="Eva "+label_ref)
        plt.plot(Fcov.date,coeff_ks_man.FCov,label="Fcover Samir")
        if "Irrigation" in label_ref:
            plt.plot(dfmax.loc[dfmax.Ir_auto>0.0].date,dfmax.Ir_auto.loc[dfmax.Ir_auto>0.0]/10,color='darkgreen',label="Irrigation auto",linestyle="",marker="x")
            plt.plot(coeff_ks_man.loc[coeff_ks_man.Irrig>0.0].index,coeff_ks_man.Irrig.loc[coeff_ks_man.Irrig>0.0]/10,color='Blue',label="Irrigation forcée",linestyle="",marker="+")
        else:
            plt.plot(coeff_ks_man.loc[coeff_ks_man.Irrig>0.0].index,coeff_ks_man.Irrig.loc[coeff_ks_man.Irrig>0.0]/10,color='Blue',label="Irrigation forcée",linestyle="",marker="+")
        # plt.plot(Fcov.date,Fcov.FCov,label='Fcover_BVNet',linewidth=1)
        plt.legend()
        plt.ylim(0,5)
        plt.ylabel("Flux Evaporation")
        ax2 = plt.twinx()
        ax2.bar(data_meteo.date,data_meteo.Prec)
        ax2.set_ylim(0,50)
        ax2.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(6))
        plt.ylabel("Pluviométrie")
        plt.title(str(y))
        plt.savefig(d["Output_model_save_fig"]+"/dynamique_flux_E_Fcover_%s_%s.png"%(lc,y),dpi=330)
        
        # plt.figure(figsize=(7,7))
        # plt.plot(coeff_ks.index,coeff_ks.fewi,label="fewi "+label_test)
        # plt.plot(coeff_ks_man.index,coeff_ks_man.fewi,label="fewi "+label_ref)
        # plt.plot(Fcov.date,coeff_ks_man.FCov,label="Fcover_samir")
        # plt.plot(Fcov.date,Fcov.FCov,label='Fcover_BVNet',linewidth=1)
        # plt.legend()
        # plt.ylim(0,2)
        # plt.ylabel("fewi")
        # plt.title(str(y))
        # plt.savefig(d["Output_model_save_fig"]+"/dynamique_fewi_Fcover_%s_%s.png"%(lc,y),dpi=330)
        
        
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
        
        plt.figure(figsize=(7,7))
        plt.plot(coeff_ks.index,coeff_ks.Dr,label="Dr " +label_test,color='blue')
        plt.plot(coeff_ks_man.index,coeff_ks_man.Dr,label="Dr "+label_ref,color='red')
        plt.plot(coeff_ks.index,coeff_ks_man.TAW*0.55,label="RAW",color='black',linestyle="--")
        plt.plot(coeff_ks.index,coeff_ks_man.TAW,label="TAW",color='black',linestyle="--")
        plt.legend(loc="upper right")
        plt.ylim(0,150)
        plt.ylabel("Dr")
        plt.title(str(y))
        ax2 = plt.twinx()
        ax2.plot(coeff_ks.index,coeff_ks.Ks,label="Ks " +label_test,linestyle="--",color='blue')
        ax2.plot(coeff_ks_man.index,coeff_ks_man.Ks,label="Ks "+label_ref,linestyle="--",color='red')
        ax2.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(6))
        ax2.set_ylim(1,-2)
        ax2.set_ylabel("Ks")
        ax2.legend(loc="upper left")
        plt.savefig(d["Output_model_save_fig"]+"/dynamique_Dr_RU_%s_%s.png"%(lc,y),dpi=330)
# =============================================================================
#          SWC 2012 et 2014 
# =============================================================================
        # if "2014" in y or "2012" in y:
        #     print('ici')
        #     plt.figure(figsize=(7,7))    
        #     SWC=pd.read_csv(d["PC_home_Wind"]+"/TRAITEMENT/DATA_VALIDATION/DATA_SWC/SWC_LAM/SWC_LAM_"+str(y)+".csv")
        #     SWC["Date"]=pd.to_datetime(SWC["Date/Time"],format="%Y-%m-%d")
        #     SWC=SWC.loc[(SWC.Date>= str(y)+"-04-01") &(SWC.Date <= str(y)+"-09-30")]
        #     plt.plot(SWC["Date"],SWC.SWC_0_moy/100,label="0 cm")
        #     plt.plot(SWC["Date"],SWC.SWC_5_moy/100,label="5 cm")
        #     plt.plot(SWC["Date"],SWC.SWC_10_moy/100,label="10 cm")
        #     plt.plot(SWC["Date"],SWC.SWC_30_moy/100,label="30 cm")
        #     plt.ylim(0,1)
        #     plt.legend()
        #     ax2 = plt.twinx()
        #     ax2.bar(data_meteo.date,data_meteo.Prec)
        #     ax2.plot(coeff_ks.loc[coeff_ks.Ir_auto>0.0].index,coeff_ks.Ir_auto.loc[coeff_ks.Ir_auto>0.0],color='darkgreen',label="Irri_auto",linestyle="",marker="<")
        #     ax2.plot(coeff_ks_man.loc[coeff_ks_man.Irrig>0.0].index,coeff_ks_man.Irrig.loc[coeff_ks_man.Irrig>0.0],color='Blue',label="Irrigation_man",linestyle="",marker=">")
        #     ax2.set_ylim(0,50)
        #     ax2.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(6))
        #     plt.ylabel("Pluviométrie")
        #     plt.savefig(d["Output_model_save_fig"]+"/dynamique_SWC_sonde_Irrigation_%s_%s.png"%(lc,y),dpi=330)
# =============================================================================
#         Kei merlin vs Kei FAO 
# =============================================================================
        # name_run_FAO="RUNS_SAMIR/RUN_MULTI_SITE_ICOS/RUN_OPTIMISATION_ICOS/SAMIR_OPTIMI_LAM/RUN_analyse/Merlin_init_ru_irri_man_Bruand"
        # name_run_Merlin="RUNS_SAMIR/RUN_MULTI_SITE_ICOS/RUN_OPTIMISATION_ICOS/SAMIR_OPTIMI_LAM/RUN_analyse/Merlin_init_ru_Fcover_irri_man_Bruand/"
        # dfmax_FAO=pickle.load(open(d["PC_disk"]+"/TRAITEMENT/"+name_run_FAO+"/"+str(y)+"/Output/maxZr/output_test_maize_irri_"+str(coup_parm.iloc[0][0])+".df","rb"))
        # dfmin_FAO=pickle.load(open(d["PC_disk"]+"/TRAITEMENT/"+name_run_FAO+"/"+str(y)+"/Output/maxZr/output_test_maize_irri_"+str(coup_parm_min.iloc[0][0])+".df","rb"))
        # dfmax_Merlin=pickle.load(open(d["PC_disk"]+"/TRAITEMENT/"+name_run_Merlin+"/"+str(y)+"/Output/maxZr/output_test_maize_irri_"+str(coup_parm.iloc[0][0])+".df","rb"))
        # dfmin_Merlin=pickle.load(open(d["PC_disk"]+"/TRAITEMENT/"+name_run_Merlin+"/"+str(y)+"/Output/maxZr/output_test_maize_irri_"+str(coup_parm_min.iloc[0][0])+".df","rb"))
        # Fcov=pickle.load(open("H:/Yann_THESE/BESOIN_EAU/BESOIN_EAU/TRAITEMENT/RUNS_SAMIR/RUN_MULTI_SITE_ICOS/RUN_OPTIMISATION_ICOS/SAMIR_OPTIMI_LAM/RUN_analyse/Merlin_init_ru_Fcover_irri_man_Bruand/"+str(y)+"/Inputdata/maize_irri/FCOVER.df","rb"))
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
df1100=pickle.load(open("H:/Yann_THESE/BESOIN_EAU/BESOIN_EAU/TRAITEMENT/RUNS_SAMIR/RUN_MULTI_SITE_ICOS/RUN_OPTIMISATION_ICOS/SAMIR_OPTIMI_LAM/RUN_final/Init_ru/Merlin_init_ru_100_optim_fewi_De_Kr_irri_man_soil/2015/Output/maxZr/output_test_maize_irri_7.df",'rb'))
df1150=pickle.load(open("H:/Yann_THESE/BESOIN_EAU/BESOIN_EAU/TRAITEMENT/RUNS_SAMIR/RUN_MULTI_SITE_ICOS/RUN_OPTIMISATION_ICOS/SAMIR_OPTIMI_LAM/RUN_final/Init_ru/Merlin_init_ru_100_optim_fewi_De_Kr_irri_man_soil/2015/Output/maxZr/output_test_maize_irri_1.df",'rb'))

diff_test=df1150-df1100

# df_Merlin=pickle.load(open("H:/Yann_THESE/BESOIN_EAU/BESOIN_EAU/TRAITEMENT/RUNS_SAMIR/RUN_MULTI_SITE_ICOS/RUN_OPTIMISATION_ICOS/SAMIR_OPTIMI_LAM/RUN_final/init_ru/Merlin_init_ru_optim_fewi_De_Kr_irri_man_soil/2015/Output/maxZr/output_test_maize_irri_2.df",'rb'))


# dfavant_modif=pickle.load(open("H:/Yann_THESE/BESOIN_EAU/BESOIN_EAU/TRAITEMENT/RUNS_SAMIR/RUN_MULTI_SITE_ICOS/RUN_OPTIMISATION_ICOS/SAMIR_OPTIMI_LAM/RUN_final/init_ru/Merlin_init_ru_100_optim_fewi_De_Kr_Fcover_irri_man_soil_min/2015/Output/maxZr/output_test_maize_irri_8.df",'rb'))

#  Probleme estiamtion SWC = 0