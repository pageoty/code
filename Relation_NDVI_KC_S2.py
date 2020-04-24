# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 11:59:32 2020

@author: Yann Pageot
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



if __name__ == '__main__':
    """ Relation NDVI-KC issu de Sentinel 2 et SWC Lamothe """
    d={}
    d["path_labo"]="/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/"
    d["path_PC"]="D:/THESE_TMP/RUNS_SAMIR/R1/Inputdata/"
    d["PC_disk"]="G:/Yann_THESE/BESOIN_EAU/"
    
#Data SWC lamothe
    df=pd.read_csv(d["PC_disk"]+"PARCELLE_LABO/FLUX_SWC/SWC_LAM_N2_final_2018.csv")
    df["Date/Time"]=df["Date/Time"].apply(lambda x:x[0:10])
    df["Date/Time"]=pd.to_datetime(df["Date/Time"],format="%d/%m/%Y")
    SWC2017=df.loc[(df["Date/Time"] > "2016-12-31") & (df["Date/Time"] < "2017-12-31")]
    SWC2017_mean=SWC2017.groupby("Date/Time").mean()

    
    #   Selection 75 %  de la RU 
    RU75= (0.363-0.17)-((0.363-0.17)*75/100)
    RU70= (0.363-0.17)-((0.363-0.17)*70/100)
    
#   ETR 2017 
    LE_lam=pd.read_csv(d["PC_disk"]+"/DATA_ETR_CESBIO/DATA_LAM_lec_python/ECPP_2017_2018_level2_lec.csv",encoding = 'utf-8',delimiter=",")
    LE_lam["TIMESTAMP"]=LE_lam["TIMESTAMP"].apply(lambda x:x[0:10])
    LE_lam["TIMESTAMP"]=pd.to_datetime(LE_lam["TIMESTAMP"],format="%d/%m/%Y")
    LE_lam_day=LE_lam.groupby("TIMESTAMP")["LE"].mean()
    ETR_lam_day=LE_lam_day*0.0352
    ETR_lam_day[ETR_lam_day < -1]=pd.NaT
# Data NDVI sur mais Irr soir 2017
    NDVI2017=pd.read_csv(d["PC_disk"]+"TRAITEMENT/NDVI_parcelle/Parcelle_ref/PARCELLE_CESBIO/LAMOTHE_NDVI_2017.csv",header=None)
    NDVI2017[0]=pd.to_datetime(NDVI2017[0],format="%Y-%m-%d")
    NDVI2017.columns=["date","valeur"]
    
#    Rainfall lamothe
    ET0=pd.read_csv(d["PC_disk"]+"Calibration_SAMIR/DONNEES_CALIBRATION/Meteo_lam_2017.csv")
    ET0["date"]=pd.to_datetime(ET0["date"],format="%Y-%m-%d")
    ET0_Saf=ET0.sort_values(by="date",ascending=True)

# Plot 
    plt.figure(figsize=(10,10))
    sns.set(style="darkgrid")
    sns.set_context('paper')
    ax1 = plt.subplot(411)
    plt.plot(NDVI2017[0],NDVI2017[1])
    plt.ylabel("NDVI")
    ax2 = plt.subplot(412)
    plt.plot(SWC2017_mean.index,SWC2017_mean.SWC_0_moy/100)
    plt.plot(SWC2017_mean.index,np.repeat(0.172,len(SWC2017_mean.index)),c="r",label="WP")
    plt.plot(SWC2017_mean.index,np.repeat(0.363,len(SWC2017_mean.index)),c="b", label="FC")
    plt.plot(SWC2017_mean.index,np.repeat(0.363-RU75,len(SWC2017_mean.index)),c="b",linestyle='--',label='RU 75 %')
    # plt.plot(SWC2017["Date/Time"],np.repeat(0.363-RU75,len(SWC2017["Date/Time"])),c="b",linestyle='--',label='RU 75 %')
    plt.ylabel('SWC en surface')
    ax1 = plt.twinx()
    ax1.grid(axis='y')
    ax1.bar(ET0_Saf.date,ET0_Saf.Prec,width=1,color="b")

    ax2 = plt.subplot(413)
    plt.plot(SWC2017_mean.index,SWC2017_mean.SWC_5_moy/100)
    plt.plot(SWC2017_mean.index,np.repeat(0.172,len(SWC2017_mean.index)),c="r",label="WP")
    plt.plot(SWC2017_mean.index,np.repeat(0.363,len(SWC2017_mean.index)),c="b", label="FC")
    plt.plot(SWC2017_mean.index,np.repeat(0.363-RU75,len(SWC2017_mean.index)),c="b",linestyle='--',label='RU 75 %')
    plt.legend()
    plt.ylabel('SWC en profondeur')
    ax34=plt.subplot(414)
    plt.plot(ET0_Saf.date,ET0_Saf.ET0,label="ET0 ")
    plt.plot(ETR_lam_day.index,ETR_lam_day,label='ETR obs')
    plt.ylabel("ET")
    plt.legend()
    # plt.savefig(d["PC_disk"]+"RESULT/plt_dyna_relat.png")
    
    # plt.plot(ETR_lam_day.iloc[0:31],label='ETR obs')
    # plt.plot(ET0_Saf.date.iloc[0:31],ET0_Saf.ET0.iloc[0:31],label="ET0 ")
    # plt.xticks(rotation=45)
    # plt.legend()
    # plt.savefig(d["PC_disk"]+"RESULT/error_janvier_ETR.png")
    
# =============================================================================
#     Calcule KCb 2017, impossible
# =============================================================================
    # Localisation des periodes où végétation non stressé, soit SWC > 75 % RU
    date_RU75=[]
    NDVI_RU_75=pd.DataFrame()
    ETR_RU_75=pd.DataFrame()
    ET0_RU_75=pd.DataFrame()
    Kcb=[]
    for swc,date in zip(SWC2017_mean.iloc[90:334].SWC_5_moy/100,SWC2017_mean.iloc[90:334].index):
        if swc >= 0.363-RU75:
            print (True)
            print(date)
            a=NDVI2017.iloc[np.where(NDVI2017["date"]==date)]
            b= ETR_lam_day.iloc[np.where(ETR_lam_day.index==date)]
            c=ET0_Saf.iloc[np.where(ET0_Saf.date==date)]
            k=b/c
            date_RU75.append(date)
            NDVI_RU_75=NDVI_RU_75.append(a)
            ETR_RU_75=ETR_RU_75.append(b)
            ET0_RU_75=ET0_RU_75.append(c)
            Kcb.append(k)

    