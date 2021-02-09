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

if __name__ == '__main__':
    d={}
    # name_run="Bilan_hydrique/RUN_FERMETURE_BILAN_HYDRIQUE/RUN_vege_avec_pluie_Fcover_assimil_avec_irri_auto/"
    # name_run="RUNS_SAMIR/RUNS_PARCELLE_GRIGNON/RUN_test/"
    name_run="RUNS_SAMIR/RUN_MULTI_SITE_ICOS/RUN_OPTIMISATION_ICOS/SAMIR_test_incertitude/test_incertitude_Fcover_v2/"
    d["PC_labo_disk"]="/run/media/pageot/Transcend/Yann_THESE/BESOIN_EAU/BESOIN_EAU/"
    d["PC_home"]="/mnt/d/THESE_TMP/"
    d["PC_home_Wind"]="D:/THESE_TMP/"
    d["PC_disk"]="H:/Yann_THESE/BESOIN_EAU/BESOIN_EAU/"
    d["PC_labo"]="/datalocal/vboxshare/THESE/BESOIN_EAU/"
    years=["2008","2010","2012","2014","2015","2019"]
# =============================================================================
# Validation Flux ETR ICOS non Multi_sie run
# =============================================================================
# modif pour groupby lc

    for y in years:
        for lc in ["maize_irri"]: # maize_rain
            # d['Output_model_PC_labo']='/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/'+name_run+"/"+y+"/"
            # d["Output_model_PC_home"]="/mnt/d/THESE_TMP/TRAITEMENT/"+name_run+"/"+y+"/"
            d["Output_model_PC_home"]="D:/THESE_TMP/TRAITEMENT/"+name_run+"/"+y+"/"
            d["Output_model_PC_labo_disk"]="/run/media/pageot/Transcend/Yann_THESE/BESOIN_EAU/BESOIN_EAU/TRAITEMENT/"+name_run+"/"+y+"/"
            d["Output_model_PC_labo"]="/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/"+name_run+"/"+y+"/"
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
            ETR=pd.read_csv(d["PC_labo"]+"/TRAITEMENT/DATA_VALIDATION/DATA_ETR_CESBIO/DATA_ETR_corr_"+str(lc)+"/ETR_"+str(lc)+str(y)+".csv",decimal='.',sep=",")
            ETR["date"]=pd.to_datetime(ETR["date"],format="%Y-%m-%d")
            ETR_obs=ETR.loc[(ETR.date >= str(y)+"-04-01") &(ETR.date <= str(y)+"-09-30")]
            # flux non corrigés
            # ETR_nn=pd.read_csv(d["PC_labo"]+"/TRAITEMENT/DATA_VALIDATION/DATA_ETR_CESBIO/DATA_ETR_"+str(lc)+"/ETR_"+str(lc)+"_"+str(y)+".csv",decimal='.',sep=",")
            # ETR_nn["date"]=pd.to_datetime(ETR_nn["date"],format="%Y-%m-%d")
            # ETR_obs_nn=ETR_nn.loc[(ETR_nn.date >= str(y)+"-04-01") &(ETR_nn.date <= str(y)+"-09-30")]
            # Flux non corrigés
            ETR_mod=pd.read_csv(d["Output_model_PC_labo_disk"][:-5]+"/LUT_ETR"+str(y)+".csv",index_col=[0,1])
            ETR_mod.columns=pd.to_datetime(ETR_mod.columns,format="%Y-%m-%d")
            ETR_mod=ETR_mod.loc[:,(ETR_mod.columns >= str(y)+"-04-01") &(ETR_mod.columns <= str(y)+"-09-30")]
            #  Moyenne tout les 5 jours 
            # jours5m=[]
            # date_moy=[]
            # for j in zip(np.arange(0,ETR_mod.shape[0],5),np.arange(4,ETR_mod.shape[0],5)):
            #     datea=ETR_mod.columns.iloc[j[0]]
            #     moy5day=dfETR_obs[["LE_Bowen",'ET']].iloc[j[0]:j[1]].mean()
            #     jours5m.append(moy5day.values)
            #     date_moy.append(datea)
            # date_ETR_moy=pd.DataFrame(date_moy)
            # ETR_week_moye=pd.DataFrame(jours5m)
            # ETR_week=pd.concat([date_ETR_moy,ETR_week_moye],axis=1)
            # ETR_week.columns=["date","LE_Bowen","ET"]
            # slope, intercept, r_value, p_value, std_err = stats.linregress(dfETR_obs.LE_Bowen.to_list(),dfETR_obs.ET.to_list())
            # bias=1/dfETR_obs.shape[0]*sum(np.mean(dfETR_obs.ET)-dfETR_obs.LE_Bowen) 
            # fitLine = predict(dfETR_obs.LE_Bowen)
            # Creation plot
            ### plot dynamique 
            plt.figure(figsize=(7,7))
            plt.plot(ETR_obs.date,ETR_obs.LE_Bowen,label='ETR_obs',color="black",linewidth=1)
            plt.plot(ETR_mod.T.index,ETR_mod.mean(),label='ETR_mod_moyenne',linewidth=1)# récupération mode 
            # for REW in np.arange(0,16,1):
            #     print(float(REW))
            #     plt.fill_between(ETR_mod.T.index, ETR_mod.T[float(REW),500.0].values,ETR_mod.T[float(REW),2500.0].values,alpha=0.2,facecolor="red")
            # plt.fill_between(ETR_mod.T.index, ETR_mod.xs(800.0,level=1).min(),ETR_mod.xs(1200.0,level=1).max(),alpha=0.5,facecolor="red")
            plt.fill_between(ETR_mod.T.index, ETR_mod.min(),ETR_mod.max(),alpha=0.5,facecolor="red")
            plt.ylabel("ETR")
            plt.ylim(0,10)
            plt.title("Dynamique ETR obs et ETR mod %s en %s"%(lc,y))
            plt.legend()
            plt.savefig(d["Output_model_PC_labo_disk"]+"/plt_Incertitude_REW_ETR_mod_%s_%s.png"%(lc,y),dpi=330)
            # ETR_test=ETR_mod.iloc[:,90:120]
            # print(ETR_test.idxmax())
            # print(ETR_test.idxmin())
            # plt.savefig(d["Output_model_PC_labo_disk"]+"/plt_Dynamique_ETR_obs_ETR_mod_%s_%s.png"%(lc,y))
            #### plot dyna cum
            plt.figure(figsize=(7,7))
            plt.plot(ETR_obs.date,ETR_obs.LE_Bowen.cumsum(),label='ETR_obs',color="black")
            plt.fill_between(ETR_mod.T.index, ETR_mod.xs(800.0,level=1).min().cumsum(),ETR_mod.xs(1200.0,level=1).max().cumsum(),alpha=0.2,facecolor="red")
            plt.fill_between(ETR_mod.T.index, ETR_mod.min().cumsum(),ETR_mod.max().cumsum(),alpha=0.2,facecolor="red")

            # for REW in np.arange(0,16,1):
            #     print(float(REW))
            #     plt.fill_between(ETR_mod.T.index, ETR_mod.T[float(REW),500.0].cumsum(),ETR_mod.T[float(REW),2500.0].cumsum(),alpha=0.2,facecolor="red")
            plt.plot(ETR_mod.T.index,ETR_mod.mean().cumsum(),label='ETR_mod',color='red')
            plt.text(ETR_obs.date.iloc[-1], ETR_obs.LE_Bowen.cumsum().iloc[-1], s=round(ETR_obs.LE_Bowen.cumsum().iloc[-1],2))
            plt.text(ETR_mod.T.index[-1], ETR_mod.mean().cumsum().iloc[-1], s=round(ETR_mod.mean().cumsum().iloc[-1],2))
            plt.ylabel("ETR")
            plt.ylim(0,700)
            plt.title("Dynamique ETR obs et ETR mod %s en %s"%(lc,y))
            plt.legend()
            plt.savefig(d["Output_model_PC_labo_disk"]+"/plt_Dynamique_ETR_obs_ETR_mod_cumul_%s_%s.png"%(lc,y))
            ###########" Dynamique week #############
           # Modification récuper max et min 
            # plt.fill_between(ETR_mod.T.index, ETR_mod.xs(500.0,level=1).min().cumsum(),ETR_mod.xs(2500.0,level=1).max().cumsum(),alpha=0.2,facecolor="red")
# =============================================================================
#              Isole forte periode incertitude 
# =============================================================================
    ETR_test=ETR_mod.iloc[:,80:120]
    plt.figure(figsize=(7,7))
    plt.plot(ETR_test.T.index,ETR_test.mean(),label='ETR_mod_moyenne',linewidth=1)# récupération mode 
    plt.fill_between(ETR_test.T.index, ETR_test.min(),ETR_test.max(),alpha=0.5,facecolor="red")
    
#  permet de connaitre le paramétrage
    plt.figure(figsize=(7,7))
    # plt.plot(ETR_obs.date,ETR_obs.LE_Bowen,label='ETR_obs',color="black",linewidth=1)
    plt.plot(ETR_mod.T.index,ETR_mod.T[7.0,1000.0],label='ETR_mod_moyenne',linewidth=1)# récupération mode 
    # for REW in np.arange(0,16,1):
    #     print(float(REW))
    #     plt.fill_between(ETR_mod.T.index, ETR_mod.T[float(REW),500.0].values,ETR_mod.T[float(REW),2500.0].values,alpha=0.2,facecolor="red")
    plt.fill_between(ETR_mod.T.index, ETR_mod.xs(800.0,level=1).min(),ETR_mod.xs(1200.0,level=1).max(),alpha=0.5,facecolor="red")
    # permer d'identifier le couple de parametre donnant le max 
    # ETR_test.idxmax().mode()
    
# =============================================================================
#    hypothèse irrigation 2008 
# =============================================================================
#  lecture des output_LUT
    # dfmax=pickle.load(open("/run/media/pageot/Transcend/Yann_THESE/BESOIN_EAU/BESOIN_EAU/TRAITEMENT/RUNS_SAMIR/RUN_MULTI_SITE_ICOS/RUN_OPTIMISATION_ICOS/SAMIR_test_incertitude/test_incertitude_Fcover_v3/2008/Output/REWmaxZr/output_test.df_maize_irri_5","rb"))
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


