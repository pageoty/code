# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 14:11:22 2020

@author: yann P
"""

import os
import sqlite3
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
from matplotlib import cm
import csv
from scipy.optimize import minimize
from sklearn.metrics import *
from scipy.optimize import linprog
from scipy import optimize
import random
import pickle
from SAMIR_optimi import RMSE
import geopandas as geo
import shapely.geometry as geom
import descartes
from datetime import datetime, date, time, timezone
from scipy import *
from scipy import stats
from pylab import *
from sklearn.linear_model import LinearRegression
import seaborn as sns
from scipy.io import loadmat
import calendar

def JulianDate_to_MMDDYYY(y,jd):
    month = 1
    day = 0
    while jd - calendar.monthrange(y,month)[1] > 0 and month <= 12:
        jd = jd - calendar.monthrange(y,month)[1]
        month = month + 1
    print (month,jd,y)

if __name__ == "__main__":
    
    d={}
    d["PC_disk_home"]="D:/THESE_TMP/"
    d['PC_disk_unix']="/mnt/d/THESE_TMP/RUNS_SAMIR/DATA_Validation/"
    d["PC_disk"]="H:/Yann_THESE/BESOIN_EAU/BESOIN_EAU/"
    d["PC_labo"]="/datalocal/vboxshare/THESE/BESOIN_EAU/"
# =============================================================================
#  Input data 
# =============================================================================
    for y in ['2015']:
        # d["PC_labo"]="D:/THESE_TMP/"
        ITK=pd.read_csv(d["PC_labo"]+"DONNEES_RAW/PARCELLE_LABO/ITK_LAM/ITK_LAM_"+y+".csv",decimal=",")
        ITK["TIMESTAMP"]=ITK["TIMESTAMP"].apply(lambda x:x[0:10])
        ITK["TIMESTAMP"]=pd.to_datetime(ITK["TIMESTAMP"],format="%d/%m/%Y")
        NDVI=pd.read_csv(d["PC_labo"]+"/TRAITEMENT/INPUT_DATA/NDVI_parcelle/Parcelle_ref/PARCELLE_CESBIO/LAMOTHE_NDVI_"+str(y)+".csv",decimal=".")
        NDVI.date=pd.to_datetime(NDVI.date,format="%Y-%m-%d")
        LAI_sat=pd.read_csv(d["PC_labo"]+"/TRAITEMENT/INPUT_DATA/LAI_parcelle/PARCELLE_LABO/LAI_LAM_"+str(y)+".csv",sep=";")
        # LAI_sigmo=pd.read_csv(d["PC_disk_home"]+'/TRAITEMENT/INPUT_DATA/LAI_parcelle/PARCELLE_LABO/LAI_pre_inter_OTB/LAI_inter_dat_date_'+str(y)+'.csv')
        meteo_SAF=pd.read_csv(d["PC_labo"]+"/TRAITEMENT/INPUT_DATA/DATA_METEO_BV/PARCELLE_LAM/meteo_lam_"+str(y)+".csv",decimal=".")
        meteo_temp=pd.read_csv(d["PC_labo"]+"TRAITEMENT/INPUT_DATA/DATA_METEO_BV/PARCELLE_LAM/DATA_PREC_TEMP/meteo_prec_temp_"+y+".csv")
        meteo_temp.columns=["date","Ta","Prec"]
        SWC=pd.read_csv(d["PC_labo"]+"/TRAITEMENT/DATA_VALIDATION/DATA_SWC/SWC_LAM/SWC_LAM_"+str(y)+".csv")
        ETR=pd.read_csv(d["PC_labo"]+"/DATA_ETR_CESBIO/DATA_ETR_LAM/DATA_ETR_LAM_ICOS/ETR_LAM_corr"+str(y)+".csv")
        ETR_nn_corr=pd.read_csv(d["PC_labo"]+"/DATA_ETR_CESBIO/DATA_ETR_LAM/DATA_ETR_LAM_ICOS/ETR_LAM"+str(y)+".csv")
        ETR.date=pd.to_datetime(ETR.date,format="%Y-%m-%d")
        ETR_nn_corr.date=pd.to_datetime(ETR.date,format="%Y-%m-%d")
        meteo_SAF.date=pd.to_datetime(meteo_SAF.date,format="%Y-%m-%d")
        meteo_temp.date=pd.to_datetime(meteo_temp.date,format="%Y-%m-%d")
        SWC["Date"]=pd.to_datetime(SWC["Date/Time"],format="%Y-%m-%d")
        # LAI_sigmo["Date"]=pd.to_datetime(LAI_sigmo["date"],format="%Y-%m-%d")
        LAI_sat["Date"]=pd.to_datetime(LAI_sat["Date"],format="%d/%m/%Y")
# =============================================================================
#    récuper période sans stress 30 cm
# =============================================================================
        RU70= (0.363-0.17)-((0.363-0.17)*70/100)
        a=SWC["Date/Time"][SWC.SWC_30_moy/100>RU70]
        LAI_ss_stress=LAI_sat.loc[LAI_sat['Date'].isin(a)]
        
# =============================================================================
#   calculer Kcb théorique ETR/ETO
# =============================================================================
        df_FLUX=pd.merge(ETR,meteo_SAF[["date","ET0"]],on=['date'])
        kc_theo=df_FLUX.LE_Bowen/df_FLUX.ET0 # probleme avec l'ETR = Erreur lier à la date non superposée 
        df_FLUX_nn_corr=pd.merge(ETR_nn_corr,meteo_SAF[["date","ET0"]],on=['date'])
        kc_theo_nn_corr=df_FLUX_nn_corr.LE/df_FLUX_nn_corr.ET0 # probleme avec l'ETR
        Kc_theo=pd.DataFrame(kc_theo)
        Kc_theo["Date"]=ETR.date
        Kc_theo_ss_stress=Kc_theo.loc[Kc_theo['Date'].isin(a)]
        # moving_avg = Kc_theo_ss_stress.rolling(10).mean()
        # moving_avg["Date"]=Kc_theo_ss_stress["Date"]
        # plt.plot(moving_avg.index,moving_avg[0])
        data_relation=pd.merge(Kc_theo_ss_stress,LAI_ss_stress[["LAI","Date"]],on="Date")
        df=data_relation.loc[data_relation.LAI>0.15]
        print(df)
        df.to_csv("D:/THESE_TMP/TRAITEMENT/Relation_LAI_Kcb/data_"+str(y)+"_LAI_nn_interpo_KCB_ss_stress_70_RU.csv")
        
        Kc_theo_nn_corr=pd.DataFrame(kc_theo_nn_corr)
        Kc_theo_nn_corr["Date"]=ETR_nn_corr.date
        Kc_theo_nnn_corr_ss_stress=Kc_theo_nn_corr.loc[Kc_theo_nn_corr['Date'].isin(a)]
        # moving_avg = Kc_theo_ss_stress.rolling(10).mean()
        # moving_avg["Date"]=Kc_theo_ss_stress["Date"]
        # plt.plot(moving_avg.index,moving_avg[0])
        data_relation=pd.merge(Kc_theo_nnn_corr_ss_stress,LAI_ss_stress[["LAI","Date"]],on="Date")
        df=data_relation.loc[data_relation.LAI>0.15]
        print(df)
        df.to_csv("D:/THESE_TMP/TRAITEMENT/Relation_LAI_Kcb/data_"+str(y)+"_LAI_nn_interpo_KCB_nn_corr_ss_stress_70_RU.csv")
        
        plt.plot(df.Date,df.LAI)
        plt.plot(df.Date,df[0])
        # a=Kc_theo.set_index("Date")
        # # a.resample("W").interpolate()
        # moving_avg = a.rolling(12).mean()
        # plt.plot(moving_avg.index,moving_avg[0])
