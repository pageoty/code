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


if __name__ == '__main__':
    d={}
    # name_run="Bilan_hydrique/RUN_FERMETURE_BILAN_HYDRIQUE/RUN_vege_avec_pluie_Fcover_assimil_avec_irri_auto/"
    # name_run="RUNS_SAMIR/RUNS_PARCELLE_GRIGNON/RUN_test/"
    name_run="RUNS_SAMIR/RUN_MULTI_SITE_ICOS/run_test_meteo/"
    sites=['GRIGNON']
    years=["2019"]
# =============================================================================
# Validation Flux ETR ICOS non Multi_sie run
# =============================================================================
# modif pour groupby Lc
    for y in years:
        for lc in ["maize_irri",'maize_rain']:
            d['Output_model_PC_labo']='/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/'+name_run+"/"+y+"/"
            ETR=pd.read_csv("/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/DATA_VALIDATION/DATA_ETR_CESBIO/DATA_ETR_"+str(lc)+"/ETR_"+str(lc)+"_"+str(y)+".csv",decimal='.')
            ETR["date"]=pd.to_datetime(ETR["date"],format="%Y-%m-%d")
            ETR_obs=ETR.loc[(ETR.date >= str(y)+"-03-02") &(ETR.date <= str(y)+"-10-31")]
            ETR_mod=pickle.load(open( d['Output_model_PC_labo']+"Output/output_valid_modif_meteo_itenti.df",'rb'))
            ETR_mod_crops=ETR_mod.groupby("LC")
            ETR_mod=ETR_mod_crops.get_group(lc)
            ETR_mod=ETR_mod.loc[(ETR_mod.date >= str(y)+"-03-02") &(ETR_mod.date <= str(y)+"-10-31")]
            dfETR_obs=pd.merge(ETR_obs,ETR_mod[["date",'ET']],on=['date'])
            dfETR_obs.dropna(inplace=True)
            ETR_week=dfETR_obs.set_index('date').resample("W").asfreq()
            slope, intercept, r_value, p_value, std_err = stats.linregress(dfETR_obs.LE.to_list(),dfETR_obs.ET.to_list())
            bias=1/dfETR_obs.shape[0]*sum(np.mean(dfETR_obs.ET)-dfETR_obs.LE) 
            fitLine = predict(dfETR_obs.LE)
            # Creation plot
            plt.figure(figsize=(7,7))
            plt.plot([0.0, 10], [0.0,10], 'black', lw=1,linestyle='--')
            plt.plot(dfETR_obs.LE,fitLine,linestyle="--")
            plt.scatter(dfETR_obs.LE,dfETR_obs.ET,s=9)
            plt.xlabel("ETR OBS")
            plt.ylabel("ETR model")
            plt.xlim(0,10)
            plt.ylim(0,10)
            # plt.title("Scatter ETR obs et ETR mod %s Fcover en %s"%(Fcover,y))
            rms = mean_squared_error(dfETR_obs.LE,dfETR_obs.ET,squared=False)
            plt.text(8,min(dfETR_obs.ET)+0.1,"RMSE = "+str(round(rms,2))) 
            plt.text(8,min(dfETR_obs.ET)+0.4,"R² = "+str(round(r_value,2)))
            plt.text(8,min(dfETR_obs.ET)+0.6,"Pente = "+str(round(slope,2)))
            plt.text(8,min(dfETR_obs.ET)+0.8,"Biais = "+str(round(bias,2)))
            # plt.savefig( d["Output_PC_home"]+"/plt_scatter_ETR_%s_Fcover_%s.png"%(Fcover,y))
            plt.figure(figsize=(7,7))
            plt.plot(dfETR_obs.date,dfETR_obs.LE,label='ETR_obs',color="black")
            plt.plot(dfETR_obs.date,dfETR_obs.ET,label='ETR_mod',color='red')
            plt.ylabel("ETR")
            plt.ylim(0,10)
            # plt.title("Dynamique ETR obs et ETR mod %s Fcover en %s"%(Fcover,y))
            plt.legend()
        # plt.savefig(d["Output_PC_home"]+"/plt_Dynamique_ETR_obs_ETR_mod_%s_Fcover_%s.png"%(Fcover,y))
        
            # Cumul semaine dfETR_obs.set_index('date').resample("W").asfreq().plot()
# =============================================================================
#   RUN_multi_site 
# =============================================================================
    d['Output_model_PC_labo']='/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/'+name_run+"/"+y+"/"
    ETR_GRI=pd.read_csv("/datalocal/vboxshare/THESE/BESOIN_EAU/DATA_ETR_CESBIO/DATA_ETR_GRIGNON/DATA_ETR_GRIGNON_ICOS/ETR_GRIGNON"+str(y)+".csv",decimal='.')
    ETR_GRI["date"]=pd.to_datetime(ETR_GRI["date"],format="%Y-%m-%d")
    ETR_GRI_obs=ETR_GRI.loc[(ETR_GRI.date >= str(y)+"-03-02") &(ETR_GRI.date <= str(y)+"-10-31")]
    
    ETR_LAM=pd.read_csv("/datalocal/vboxshare/THESE/BESOIN_EAU/DATA_ETR_CESBIO/DATA_ETR_LAM/DATA_ETR_LAM_ICOS/ETR_LAM"+str(y)+".csv",decimal='.')
    ETR_LAM["date"]=pd.to_datetime(ETR_LAM["date"],format="%Y-%m-%d")
    ETR_LAM_obs=ETR_LAM.loc[(ETR_LAM.date >= str(y)+"-03-02") &(ETR_LAM.date <= str(y)+"-10-31")]
    ETR_obs=pd.merge(ETR_GRI_obs,ETR_LAM_obs[["date",'LE']],on=["date"])
    ETR_obs.columns=["date",'LE_GRI','LE_LAM']
    
    ETR_mod=pickle.load(open( d['Output_model_PC_labo']+"Output/output.df",'rb'))
    # ETR_mod=Fcover_sat[["ET",'date']]
    ETR_mod=ETR_mod.loc[(ETR_mod.date >= str(y)+"-03-02") &(ETR_mod.date <= str(y)+"-10-31")]
    ETR_mod=ETR_mod[["ET",'date','id']]
    ETR_mod=ETR_mod.groupby('id')
    ETR_mod_LAM=ETR_mod.get_group(1)
    ETR_mod_GRI=ETR_mod.get_group(2)
    dfETR_Gri=pd.merge(ETR_GRI_obs,ETR_mod_GRI[['date','ET']],on=["date"])
    dfETR_Lam=pd.merge(ETR_LAM_obs,ETR_mod_LAM[['date','ET']],on=["date"])

    # dfETR=pd.concat([ETR_obs,ETR_mod],axis=1)
    # dfETR_obs=pd.merge(ETR_obs,ETR_mod[["date",'ET']],on=['date'])
    dfETR_obs.dropna(inplace=True)
    ETR_week=dfETR_obs.set_index('date').resample("W").asfreq()
    slope, intercept, r_value, p_value, std_err = stats.linregress(dfETR_obs.LE.to_list(),dfETR_obs.ET.to_list())
    bias=1/dfETR_obs.shape[0]*sum(np.mean(dfETR_obs.ET)-dfETR_obs.LE) 
    fitLine = predict(dfETR_obs.LE)
    # Creation plot
    plt.figure(figsize=(7,7))
    plt.plot([0.0, 10], [0.0,10], 'black', lw=1,linestyle='--')
    plt.plot(dfETR_obs.LE,fitLine,linestyle="--")
    plt.scatter(dfETR_obs.LE,dfETR_obs.ET,s=9)
    plt.xlabel("ETR OBS")
    plt.ylabel("ETR model")
    plt.xlim(0,10)
    plt.ylim(0,10)
    # plt.title("Scatter ETR obs et ETR mod %s Fcover en %s"%(Fcover,y))
    rms = mean_squared_error(dfETR_obs.LE,dfETR_obs.ET,squared=False)
    plt.text(8,min(dfETR_obs.ET)+0.1,"RMSE = "+str(round(rms,2))) 
    plt.text(8,min(dfETR_obs.ET)+0.4,"R² = "+str(round(r_value,2)))
    plt.text(8,min(dfETR_obs.ET)+0.6,"Pente = "+str(round(slope,2)))
    plt.text(8,min(dfETR_obs.ET)+0.8,"Biais = "+str(round(bias,2)))
    # plt.savefig( d["Output_PC_home"]+"/plt_scatter_ETR_%s_Fcover_%s.png"%(Fcover,y))
    plt.figure(figsize=(7,7))
    plt.plot(dfETR_obs.date,dfETR_obs.LE,label='ETR_obs',color="black")
    plt.plot(dfETR_obs.date,dfETR_obs.ET,label='ETR_mod',color='red')
    plt.ylabel("ETR")
    plt.ylim(0,10)
    # plt.title("Dynamique ETR obs et ETR mod %s Fcover en %s"%(Fcover,y))
    plt.legend()

# =============================================================================
#   Validation SWC
# =============================================================================
# =============================================================================
#  Validation Irri_ préparation data 
# =============================================================================
    all_quantity=[]
    all_number=[]
    all_id=[]
    all_runs=[]
    all_date=[]
    all_date_jj=[]
    for r in os.listdir('D:/THESE_TMP/RUNS_SAMIR/RUN_RELATION/TARN_CACG/'):
        d["path_PC"]='D:/THESE_TMP/RUNS_SAMIR/RUN_RELATION/TARN_CACG/'+str(r)+'/Inputdata/'
        for s in os.listdir(d["path_PC"][:-10]):
            if "output" in s:
                print (s)
                df=pickle.load(open(d["path_PC"][:-10]+str(s),'rb'))
                for id in list(set(df.id)):
                    lam=df.loc[df.id==id]
                    date_irr=lam.loc[lam.Ir_auto!=0.0]["date"]
                    size_R=len(date_irr) # vecteur de répétiton de la variable RUNS
                    print(r'n° du runs : %s '% r)
                    print(r' n° parcelle : %s' %id)
                    print(r'sum irrigation in mm : %s'%lam.groupby(["LC","id"])["Ir_auto"].sum()[0])
                    print(r' nb irrigation : %s' %lam.Ir_auto.where(df["Ir_auto"] != 0.0).dropna().count())
                    all_runs.append(r)
                    all_id.append(id)
                    all_quantity.append(lam.groupby(["LC","id"])["Ir_auto"].sum()[0])
                    all_number.append(lam.Ir_auto.where(df["Ir_auto"] != 0.0).dropna().count())
                    all_date.append(date_irr.values)
                    for i in date_irr:
                        a=i.strftime("%j") # date en jj sur l'année
                        all_date_jj.append(a)
    all_resu=pd.DataFrame([all_runs,all_id,all_quantity,all_number,all_date]).T
    all_resu.columns=["runs",'id','cumul_irr',"nb_irr","date_irr"]

# =============================================================================
# Validation des Irr cumulées CACG
# =============================================================================
    vali_cacg=pd.read_csv(d["PC_disk"]+"TRAITEMENT/DONNEES_VALIDATION_SAMIR/merge_parcelle_2017.csv")
#    vali_cacg.dropna(subset=['id'],inplace=True)
    vali_cacg.Date_irrigation=pd.to_datetime(vali_cacg.Date_irrigation,format='%d/%m/%y')
    vali_cacg["Quantity(mm)"].astype(float)
    sum_irr_cacg_val=vali_cacg.groupby("id")["Quantity(mm)"].sum()
    nb_irr=vali_cacg.groupby("id")["Date_irrigation"].count()
    
    for r in os.listdir('D:/THESE_TMP/RUNS_SAMIR/RUN_RELATION/TARN_CACG/'):
        d["path_PC"]='D:/THESE_TMP/RUNS_SAMIR/RUN_RELATION/TARN_CACG/'+str(r)+'/Inputdata/'
        slope, intercept, r_value, p_value, std_err = stats.linregress(sum_irr_cacg_val.to_list(),all_resu.loc[all_resu.runs==r]["cumul_irr"].to_list())
        bias=1/sum_irr_cacg_val.shape[0]*sum(np.mean(all_resu.loc[all_resu.runs==r]["cumul_irr"])-sum_irr_cacg_val) 
        fitLine = predict(sum_irr_cacg_val)
        plt.figure(figsize=(7,7))
        plt.title(r)
        plt.plot([0.0, 300], [0.0, 300], 'r-', lw=2)
        plt.plot(sum_irr_cacg_val,fitLine,linestyle="-")
        plt.scatter(sum_irr_cacg_val,all_resu.loc[all_resu.runs==r]["cumul_irr"])
        plt.plot()
        plt.xlabel("cumul_irr OBS")
        plt.ylabel("cumul irr model")
        plt.xlim(0,300)
        plt.ylim(0,300)
        rms = mean_squared_error(sum_irr_cacg_val,all_resu.loc[all_resu.runs==r]["cumul_irr"],squared=False)
        plt.text(10,min(all_resu.loc[all_resu.runs==r]["cumul_irr"])+40,"RMSE = "+str(round(rms,2)))
        plt.text(10,min(all_resu.loc[all_resu.runs==r]["cumul_irr"])+30,"R² = "+str(round(r_value,2)))
        plt.text(10,min(all_resu.loc[all_resu.runs==r]["cumul_irr"])+20,"Pente = "+str(round(slope,2)))
        plt.text(10,min(all_resu.loc[all_resu.runs==r]["cumul_irr"])+10,"Biais = "+str(round(bias,2)))
        for j in np.arange(len(sum_irr_cacg_val.index)):
            plt.text(x = sum_irr_cacg_val.to_list()[j] + 2, y=all_resu.loc[all_resu.runs==r]["cumul_irr"].iloc[j]+ 1,s = list(all_resu.loc[all_resu.runs==r]["id"])[j],size=9)
        plt.savefig(d["path_PC"][:-10]+"plt_scatter_quantity_irri_%s.png"%r)
#         plot nb_irr
        
        slope, intercept, r_value, p_value, std_err = stats.linregress(nb_irr.to_list(),all_resu.loc[all_resu.runs==r]["nb_irr"].to_list())
        bias=1/nb_irr.shape[0]*sum(mean(all_resu.loc[all_resu.runs==r]["nb_irr"])-nb_irr) 
        fitLine = predict(nb_irr)
        plt.figure(figsize=(7,7))
        plt.title(r)
        plt.plot([0.0,20], [0.0, 20], 'r-', lw=2)
        plt.plot(nb_irr,fitLine,linestyle="--")
        plt.scatter(nb_irr,all_resu.loc[all_resu.runs==r]["nb_irr"])
        plt.plot()
        plt.xlabel("nb_irr OBS")
        plt.ylabel("nb irr model")
        plt.xlim(0,20)
        plt.ylim(0,20)
        rms = mean_squared_error(nb_irr,all_resu.loc[all_resu.runs==r]["nb_irr"],squared=False) # if False == RMSE or True == MSE
        plt.text(10,min(all_resu.loc[all_resu.runs==r]["nb_irr"])+2,"RMSE = "+str(round(rms,2)))
        plt.text(10,min(all_resu.loc[all_resu.runs==r]["nb_irr"])+1.5,"R² = "+str(round(r_value,2)))
        plt.text(10,min(all_resu.loc[all_resu.runs==r]["nb_irr"])+1,"Pente = "+str(round(slope,2)))
        plt.text(10,min(all_resu.loc[all_resu.runs==r]["nb_irr"])+0.5,"Biais = "+str(round(bias,2)))
        for j in np.arange(len(nb_irr.index)):
            plt.text(x = nb_irr.to_list()[j]+0.1 , y=all_resu.loc[all_resu.runs==r]["nb_irr"].iloc[j]+0.1,s = list(all_resu.loc[all_resu.runs==r]["id"])[j],size=9)
        plt.savefig(d["path_PC"][:-10]+"plt_scatter_nb_irri_%s.png"%r)


# =============================================================================
#   Calcul le cumul d'irrigation 
# =============================================================================
    #  Irrigation total par OS simuler et par parcelle
#    lam.groupby(["LC","id"])["Ir_auto"].sum()
#    # or
#    lam.Ir_auto.where(lam["Ir_auto"] != 0.0).dropna().count() # resultat  980.0 et ref = 944 soit 44 mm surplus
#
