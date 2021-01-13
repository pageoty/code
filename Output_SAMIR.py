#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 09:29:18 2020

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
    name_run="RUNS_SAMIR/RUN_MULTI_SITE_ICOS/Correction_Fcover/run_test_plateau_Fcover/"
    d["PC_labo"]="/datalocal/vboxshare/THESE/BESOIN_EAU/"
    d["PC_labo_disk"]="/run/media/pageot/Transcend/Yann_THESE/BESOIN_EAU/BESOIN_EAU/"
    d["PC_home"]="/mnt/d/THESE_TMP/"
    d["PC_home_Wind"]="D:/THESE_TMP/"
    d["PC_disk"]="H:/Yann_THESE/BESOIN_EAU/BESOIN_EAU/"
    sites=['GRIGNON']
    years=["2008","2010","2012","2014","2015","2019"]
# =============================================================================
# Validation Flux ETR ICOS non Multi_sie run
# =============================================================================
# modif pour groupby Lc
    for y in years:
        for lc in ["maize_irri"]: # maize_rain
            # d['Output_model_PC_labo']='/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/'+name_run+"/"+y+"/"
            # d["Output_model_PC_home"]="/mnt/d/THESE_TMP/TRAITEMENT/"+name_run+"/"+y+"/"
            d["Output_model_PC_home"]="D:/THESE_TMP/TRAITEMENT/"+name_run+"/"+y+"/"
            d["Output_model_PC_labo_disk"]="/run/media/pageot/Transcend/Yann_THESE/BESOIN_EAU/BESOIN_EAU/TRAITEMENT/"+name_run+"/"+y+"/"
            d["Output_model_PC_labo"]="/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/"+name_run+"/"+y+"/"
            if lc == "maize_irri":
                SWC=pd.read_csv(d["PC_labo"]+"TRAITEMENT/DATA_VALIDATION/DATA_SWC/SWC_LAM/SWC_LAM_"+str(y)+".csv")
                SWC["Date/Time"]=pd.to_datetime(SWC["Date/Time"],format="%Y-%m-%d")
                meteo=pd.read_csv(d["PC_labo"]+"TRAITEMENT/INPUT_DATA/DATA_METEO_BV/PARCELLE_LAM/meteo_lam_"+str(y)+".csv",decimal=".")
                meteo.date=pd.to_datetime(meteo.date,format="%Y-%m-%d")
            else:
                SWC=pd.read_csv(d["PC_labo"]+"TRAITEMENT/DATA_VALIDATION/DATA_SWC/SWC_GRI/SWC_GRI_2019.csv")
                SWC["date"]=pd.to_datetime(SWC["date"],format="%Y-%m-%d")
                meteo=pd.read_csv(d["PC_labo"]+"TRAITEMENT/INPUT_DATA/DATA_METEO_BV/PARCELLE_GRI/meteo_gri_2019.csv",decimal=".")
                meteo.date=pd.to_datetime(meteo.date,format="%Y-%m-%d")
            ETR=pd.read_csv(d["PC_labo"]+"/TRAITEMENT/DATA_VALIDATION/DATA_ETR_CESBIO/DATA_ETR_corr_"+str(lc)+"/ETR_"+str(lc)+str(y)+".csv",decimal='.',sep=",")
            ETR["date"]=pd.to_datetime(ETR["date"],format="%Y-%m-%d")
            ETR_obs=ETR.loc[(ETR.date >= str(y)+"-03-02") &(ETR.date <= str(y)+"-10-31")]
            ETR_mod=pickle.load(open( d['Output_model_PC_labo']+"Output/output.df",'rb'))
            ETR_mod_crops=ETR_mod.groupby("LC")
            ETR_mod=ETR_mod_crops.get_group(lc)
            # ETR_mod=ETR_mod.loc[(ETR_mod.date >= str(y)+"-03-02") &(ETR_mod.date <= str(y)+"-10-31")]
            dfETR_obs=pd.merge(ETR_obs,ETR_mod[["date",'ET',"NDVI"]],on=['date'])
            dfETR_obs.dropna(inplace=True)
            ETR_week=dfETR_obs.set_index('date').resample("W").asfreq() # revoir ETR week avec moyenne glissante 5 jour == tour d'un pivot irrigation
            ETR_week.dropna(inplace=True)
            slope, intercept, r_value, p_value, std_err = stats.linregress(dfETR_obs.LE_Bowen.to_list(),dfETR_obs.ET.to_list())
            bias=1/dfETR_obs.shape[0]*sum(np.mean(dfETR_obs.ET)-dfETR_obs.LE_Bowen) 
            fitLine = predict(dfETR_obs.LE_Bowen)
            # Creation plot
            plt.figure(figsize=(7,7))
            plt.plot([0.0, 10], [0.0,10], 'black', lw=1,linestyle='--')
            plt.plot(dfETR_obs.LE_Bowen,fitLine,linestyle="--")
            # plt.scatter(dfETR_obs.LE,dfETR_obs.ET,s=9,c=select_color_NDVI(dfETR_obs))
            plt.scatter(dfETR_obs.LE_Bowen,dfETR_obs.ET,s=9)
            plt.xlabel("ETR OBS")
            plt.ylabel("ETR model")
            plt.xlim(0,15)
            plt.ylim(0,15)
            plt.legend(('Soil nu','Vege'))
            plt.title("Scatter ETR obs et ETR mod %s en %s"%(lc,y))
            rms = mean_squared_error(dfETR_obs.LE_Bowen,dfETR_obs.ET)
            plt.text(8,min(dfETR_obs.ET)+0.1,"RMSE = "+str(round(rms,2))) 
            plt.text(8,min(dfETR_obs.ET)+0.4,"R² = "+str(round(r_value,2)))
            plt.text(8,min(dfETR_obs.ET)+0.6,"Pente = "+str(round(slope,2)))
            plt.text(8,min(dfETR_obs.ET)+0.8,"Biais = "+str(round(bias,2)))
            plt.savefig( d["Output_model_PC_labo"]+"/plt_scatter_ETR_%s_%s.png"%(lc,y))
            ###### SCATTER moyenne semaine ######
            slope, intercept, r_value, p_value, std_err = stats.linregress(ETR_week.LE_Bowen.to_list(),ETR_week.ET.to_list())
            bias=1/ETR_week.shape[0]*sum(np.mean(ETR_week.ET)-ETR_week.LE_Bowen) 
            fitLine = predict(ETR_week.LE_Bowen)
            # Creation plot
            plt.figure(figsize=(7,7))
            plt.plot([0.0, 10], [0.0,10], 'black', lw=1,linestyle='--')
            plt.plot(ETR_week.LE_Bowen,fitLine,linestyle="--")
            plt.scatter(ETR_week.LE_Bowen,ETR_week.ET,s=9)
            plt.xlabel("ETR OBS")
            plt.ylabel("ETR model")
            plt.xlim(0,15)
            plt.ylim(0,15)
            plt.title("Scatter ETR obs et ETR mod %s en %s"%(lc,y))
            rms = mean_squared_error(ETR_week.LE_Bowen,ETR_week.ET)
            plt.text(8,min(ETR_week.ET)+0.1,"RMSE = "+str(round(rms,2))) 
            plt.text(8,min(ETR_week.ET)+0.4,"R² = "+str(round(r_value,2)))
            plt.text(8,min(ETR_week.ET)+0.6,"Pente = "+str(round(slope,2)))
            plt.text(8,min(ETR_week.ET)+0.8,"Biais = "+str(round(bias,2)))
            plt.savefig( d["Output_model_PC_labo"]+"/plt_scatter_ETR_week_%s_%s.png"%(lc,y))
            ### plot dynamique 
            plt.figure(figsize=(7,7))
            plt.plot(dfETR_obs.date,dfETR_obs.LE_Bowen,label='ETR_obs',color="black")
            plt.plot(dfETR_obs.date,dfETR_obs.ET,label='ETR_mod',color='red')
            plt.ylabel("ETR")
            plt.ylim(0,15)
            plt.title("Dynamique ETR obs et ETR mod %s en %s"%(lc,y))
            plt.legend()
            plt.savefig(d["Output_model_PC_labo"]+"/plt_Dynamique_ETR_obs_ETR_mod_%s_%s.png"%(lc,y))
            #### plot dyna cum
            plt.figure(figsize=(7,7))
            plt.plot(dfETR_obs.date,dfETR_obs.LE_Bowen.cumsum(),label='ETR_obs',color="black")
            plt.plot(dfETR_obs.date,dfETR_obs.ET.cumsum(),label='ETR_mod',color='red')
            plt.text(dfETR_obs.date.iloc[-1], dfETR_obs.ET.cumsum().iloc[-1], s=round(dfETR_obs.ET.cumsum().iloc[-1],2))
            plt.text(dfETR_obs.date.iloc[-1], dfETR_obs.LE_Bowen.cumsum().iloc[-1], s=round(dfETR_obs.LE_Bowen.cumsum().iloc[-1],2))
            plt.ylabel("ETR")
            plt.title("Dynamique ETR obs et ETR mod %s en %s"%(lc,y))
            plt.legend()
            plt.savefig(d["Output_model_PC_labo"]+"/plt_Dynamique_ETR_obs_ETR_mod_cumul_%s_%s.png"%(lc,y))
            ###########" Dynamique week #############
            plt.figure(figsize=(7,7))
            plt.plot(ETR_week.index,ETR_week.LE_Bowen,label='ETR_obs',color="black")
            plt.plot(ETR_week.index,ETR_week.ET,label='ETR_mod',color='red')
            plt.ylabel("ETR")
            plt.ylim(0,15)
            plt.title("Dynamique ETR week obs et ETR week mod %s en %s"%(lc,y))
            plt.legend()
            plt.savefig(d["Output_model_PC_labo"]+"/plt_Dynamique_week_ETR_obs_ETR_mod_%s_%s.png"%(lc,y))
            plt.figure(figsize=(7,7))
            plt.title("Dynamique Dr, Irri et Ks %s en %s"%(lc,y))
            # # plt.plot(ETR_mod.date,ETR_mod.Dr,label='Dep racinaire')
            plt.plot(ETR_mod.date,ETR_mod.TEW,label="réservoir evapo")
            plt.plot(ETR_mod.date,(ETR_mod.Dei+ETR_mod.Dep),label='Dep evapo')
            # # plt.plot(ETR_mod.date,ETR_mod.Irrig,label="Irri")
            # # plt.plot(ETR_mod.date,ETR_mod.Ir_auto,label="Irri")
            # # plt.bar(ETR_mod.date,ETR_mod.Prec,label="Prec",width=2)
            plt.ylim(0,ETR_mod.Dr.max())
            plt.legend(loc='upper left')
            ax2 = plt.twinx()
            ax2.plot(ETR_mod.date,ETR_mod.Ks,color='r',linestyle="--",label="Ks")
            ax2.set_ylim(-5,1)
            ax2.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(6))
            plt.legend()
            plt.savefig(d["Output_model_PC_labo"]+"/plt_Dynamique_coeff_TEW_De_%s_%s.png"%(lc,y))

            plt.figure(figsize=(7,7))
            plt.title("Dynamique Dr, Irri et Ks %s en %s"%(lc,y))
            plt.plot(ETR_mod.date,ETR_mod.Dr,label='Dep racinaire')
            plt.plot(ETR_mod.date,ETR_mod.RAW,label="réservoir racinaire")
            # plt.plot(ETR_mod.date,ETR_mod.Irrig,label="Irri")
            # plt.plot(ETR_mod.date,ETR_mod.Ir_auto,label="Irri")
            # plt.bar(ETR_mod.date,ETR_mod.Prec,label="Prec",width=2)
            plt.ylim(0,ETR_mod.Dr.max()+10)
            plt.legend(loc='upper left')
            ax2 = plt.twinx()
            ax2.plot(ETR_mod.date,ETR_mod.Ks,color='r',linestyle="--",label="Ks")
            ax2.set_ylim(-5,1.1)
            ax2.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(6))
            plt.legend()
            plt.savefig(d["Output_model_PC_labo"]+"/plt_Dynamique_coeff_TAW_Dr_%s_%s.png"%(lc,y))
            
            plt.figure(figsize=(7,7))
            plt.title("Dynamique Dr, Irri et Ks %s en %s"%(lc,y))
            # # plt.plot(ETR_mod.date,ETR_mod.Kei,label='Ke')
            # # plt.plot(ETR_mod.date,ETR_mod.Kcb,label="Kcb")
            plt.plot(ETR_mod.date,ETR_mod.Tr,label="Tran")
            plt.plot(ETR_mod.date,ETR_mod.Ev,label="eva")
            plt.plot(ETR_mod.date,ETR_mod.FCov,label="Fcover")
            # # plt.plot(ETR_mod.date,ETR_mod.Irrig,label="Irri")
            # # plt.plot(ETR_mod.date,ETR_mod.Ir_auto,label="Irri")
            # # plt.bar(ETR_mod.date,ETR_mod.Prec,label="Prec",width=2)
            plt.ylim(0,ETR_mod.Tr.max()+1)
            plt.legend(loc='upper left')
            ax2 = plt.twinx()
            ax2.plot(ETR_mod.date,ETR_mod.Ks,color='r',linestyle="--",label="Ks")
            ax2.set_ylim(-5,1.1)
            ax2.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(6))
            plt.legend(loc="upper right")
            plt.savefig(d["Output_model_PC_labo"]+"/plt_Dynamique_Tr_Kcb_Eva_%s_%s.png"%(lc,y))
            
            
            
            ####print le NDVI max et le Kcb issu du modèle

            print("##############")
            print (y)
            print (ETR_mod.Ir_auto.sum())
            # kc=ETR_mod.loc[ETR_mod.date==ETR_mod.iloc[ETR_mod.LAI.idxmax()]["date"]]["Kcb"]
            # print(y)
            # print (kc)
            # plt.figure(figsize=(7,7))
            # plt.plot(ETR_mod.date,ETR_mod.Kcb)
            # plt.plot(ETR_mod.date,ETR_mod.NDVI,label="NDVI")
            # plt.legend()
            # plt.figure(figsize=(7,7))
            # plt.title("Dynamique Eva et Trans %s en %s"%(lc,y))
            # plt.plot(ETR_mod.date,ETR_mod.Ev,label='Evapo')
            # plt.plot(ETR_mod.date,ETR_mod.Tr,label="Trans")
            # plt.legend()
            # plt.figure(figsize=(7,7))
            # plt.title("Dynamique des coefficients %s en %s"%(lc,y))
            # plt.plot(ETR_mod.date,ETR_mod.Kcb,label='Kcb')
            # # plt.plot(ETR_mod.date,(ETR_mod.Kei+ETR_mod.Kep),label="Ke")
            # ## plt.plot(ETR_mod.date,ETR_mod.Kep,label="Kep")
            # # plt.plot(ETR_mod.date,ETR_mod.W,label="W capillary rise")
            # # plt.legend()
            # # ax2 = plt.twinx()
            # # ax2.grid()
            # # plt.bar(meteo.date,meteo.Prec,width=1,color='b')
            # plt.savefig(d["Output_model_PC_labo_disk"]+"/plt_Dynamique_coeff_Kcb_Ke_%s_%s.png"%(lc,y))
            # plt.figure(figsize=(7,7))
            # plt.title("Dynamique SWC with irrigation %s en %s"%(lc,y))
            # plt.plot(ETR_mod.date,ETR_mod.SWC1,label='zone Ze')
            # plt.plot(ETR_mod.date,ETR_mod.SWC2,label="zone Zr")
            # plt.ylim(-2,1.5)
            # plt.legend()
            # ax2 = plt.twinx()
            # plt.bar(meteo.date,meteo.Prec,width=1,color='b')
            # plt.plot(ETR_mod.date,ETR_mod.SWCvol3,label="zone Zd")
            # plt.legend()
            # ax2 = plt.twinx()
            # ax2.grid(axis='y')
            # ax2.bar(ETR_mod.date,ETR_mod.Ir_auto,label="Irrigation",color='r',width=5)
            # ax2.bar(ETR_mod.date,ETR_mod.Prec,label="Prec",color='b',width=1)
            # ax2.set_ylim(0,100)
            # plt.legend()
            # plt.savefig(d["Output_model_PC_labo_disk"]+"/plt_Dynamique_SWC_%s_%s.png"%(lc,y))
            # ###### SWC evaluation ######
            # SWC_select=SWC.loc[(SWC.date >= str(y)+"-06-01") &(SWC.date <= str(y)+"-10-31")]
            # ETR_mod_select=ETR_mod.loc[(ETR_mod.date >= str(y)+"-06-01")&( ETR_mod.date <= SWC_select.date.iloc[-1])]
            # plt.figure(figsize=(12,10))
            # plt.plot(ETR_mod.date,ETR_mod.SWCvol1)
            # plt.plot(SWC_select.date,SWC_select.SWC_5_moy/100)
            
            # plt.figure(figsize=(12,10))
            # slope, intercept, r_value, p_value, std_err = stats.linregress(SWC_select.SWC_5_moy/100.to_list(),ETR_mod_select.SWCvol1.to_list())
            # bias=1/SWC_select.shape[0]*sum(np.mean(ETR_mod.SWCvol1)-SWC_select.SWC_5_moy/100) 
            # fitLine = predict(SWC_select.SWC_5_moy/100)
            # Creation plot
            # plt.plot(SWC_select.SWC_5_moy/100,fitLine,linestyle="--")
            # plt.plot([0.0, 0.3], [0.0,0.3], 'black', lw=1,linestyle='--')
            # plt.scatter(SWC_select.SWC_5_moy/100,ETR_mod_select.SWCvol1)
            # plt.xlabel("SWC obs")
            # plt.ylabel("SWC mod")
            
# # =============================================================================
# #   Isolé le problème ETR sous estimier
# # =============================================================================
            # Jui=ETR_mod.loc[(ETR_mod.date>="2019-08-01")&(ETR_mod.date<="2019-10-01")]
            # ETR_o=dfETR_obs.loc[(dfETR_obs.date>="2019-08-01")&(dfETR_obs.date<="2019-10-01")]
            # ETR_o["date"]=pd.to_datetime(ETR_o["date"],format="%Y-%m-%d")
            # Jui["date"]=pd.to_datetime(Jui["date"],format="%Y-%m-%d")
            # # plt.plot(Jui.date,Jui.NDVI,label="NDVI")
            # plt.figure(figsize=(7,7))
            # plt.plot(ETR_o.date,ETR_o.NDVI,label='NDVI')
            # plt.plot(Jui.date,Jui.Kcb,label='Kcb')
            # # plt.plot(Jui.date,Jui.ET,label='ET')
            # # ax2 = plt.twinx()
            # # ax2.plot(Jui.date,Jui.Ks,label="Stress",linestyle='--',color='red')
            # # ax2.set_ylim(-5,1)
            # plt.legend()
            # # plt.figure(figsize=(7,7))
            # # plt.plot(Jui.date,Jui.Dr,label="Deep racin")
            # # plt.plot(Jui.date,Jui.Dei+Jui.Dep,label="Deep Evapo zone")
            # # plt.bar(Jui.date,Jui.Prec,label='Prec',color='Blue')
            # # plt.bar(Jui.date,Jui.Ir_auto,label='irr',color='red')
            # plt.legend()
# =============================================================================
# comparaison data ETR Grignon maize année 2019 , 2015 et 2012
# =============================================================================
    # ETR_2015=pd.read_csv('H:/Yann_THESE/BESOIN_EAU/BESOIN_EAU/DATA_ETR_CESBIO/DATA_ETR_GRIGNON/DATA_ETR_GRIGNON_ICOS/ETR_ICOS_GRI2015.csv')
    # ETR_2015["date"]=pd.to_datetime(ETR_2015["date"],format="%Y-%m-%d")
    # ETR_2015=ETR_2015.loc[(ETR_2015.date>="2015-05-01")&(ETR_2015.date<="2015-10-31")]
    # ETR_2015["month"]=ETR_2015.date.dt.strftime('%m-%d')
    # dfETR_obs["month"]=dfETR_obs.date.dt.strftime("%m-%d")
    # plt.figure(figsize=(7,7))
    # plt.plot(ETR_2015.month,ETR_2015.ETR,label="2015")
    # plt.plot(dfETR_obs.month,dfETR_obs.ET,label="2019")
    # plt.gca().xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(5))
    # plt.legend()
    # plt.savefig(d["Output_model_PC_home"]+"/plt_DynamiqueETR_2015_2019_Gri.png")
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


# =============================================================================
#   Calcul le cumul d'irrigation 
# =============================================================================
    #  Irrigation total par OS simuler et par parcelle
#    lam.groupby(["LC","id"])["Ir_auto"].sum()
#    # or
#    lam.Ir_auto.where(lam["Ir_auto"] != 0.0).dropna().count() # resultat  980.0 et ref = 944 soit 44 mm surplus