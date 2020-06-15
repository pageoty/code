# -*- coding: utf-8 -*-
"""
Created on Thu May 28 10:37:58 2020

@author: Yann Pageot
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 09:29:51 2020

@author: Yann Pageot
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


if __name__ == "__main__":
    
   
    years=2018
    name_run="RUN_STOCK_DATA_"+str(years)+"_partenaire"
    d={}
    d['SAMIR_run_Wind']="D:/THESE_TMP/RUNS_SAMIR/"+name_run+"/"
    d["PC_disk_Wind"]="D:/THESE_TMP/RUNS_SAMIR/DATA_Validation/"
    d['PC_disk_unix']="/mnt/d/THESE_TMP/RUNS_SAMIR/DATA_Validation/"
    d["PC_disk"]="G:/Yann_THESE/BESOIN_EAU/"
# =============================================================================
#  Plot NDVI & meteo
# =============================================================================
    # Pluvio_all=pd.DataFrame()
    # Pluvio_cum=pd.DataFrame()
    # Pluvio_sais=pd.DataFrame()
    ITK=pd.read_csv("G:/Yann_THESE/BESOIN_EAU/TRAITEMENT/DONNEES_VALIDATION_SAMIR/merge_parcelle_"+str(years)+".csv",decimal=",")
    ITK["Date_irrigation"]=pd.to_datetime(ITK["Date_irrigation"],format="%d/%m/%Y")
    ITK["Date de semis"]=pd.to_datetime(ITK["Date de semis"],format="%d/%m/%Y")
    NDVI=pd.read_csv("G:/Yann_THESE/BESOIN_EAU/TRAITEMENT/NDVI_parcelle/Parcelle_ref/Fusion/NDVI_raw/NDVI_ref_parcelle_"+str(years)+".csv",decimal=".")
    NDVI.date=pd.to_datetime(NDVI.date,format="%Y-%m-%d")
    meteo=pickle.load(open(d["SAMIR_run_Wind"]+"/Inputdata/meteo.df",'rb'))
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.set(style="darkgrid")
    sns.set_context('paper')
    ax1 = plt.subplot(211)
    for i in list(set(NDVI.id)):
        plt.figure(figsize=(10,7))
        ax1=plt.subplot(211)
        plt.plot(NDVI.loc[NDVI.id==i]["date"],NDVI.loc[NDVI.id==i]["NDVI"],label='NDVI')
        plt.title(i)
        # plt.plot(NDVI_raw.date,NDVI_raw[0],linestyle='None', marker="o")
        # plt.plot([ITK.date.loc[ITK.ITK!="Ma"],ITK.date.loc[ITK.ITK!="Ma"]], [0.9,0.9],marker="o",color='red')
        plt.ylim(0,1)
        plt.ylabel('NDVI')
        plt.legend()
        plt.setp(ax1.get_xticklabels(),visible=False)
        ax2=plt.subplot(212)
        plt.plot(meteo.loc[meteo.id==i]["date"],meteo.loc[meteo.id==i].ET0,color='green',label="ET0")
        plt.legend(loc='upper left')
        plt.ylabel('ETO')
        plt.ylim(0,10)
        ax21 = plt.twinx()
        ax21.grid(axis='y')
    ax21.plot(Irri.date,Irri.Irrig,color="red",marker="o",linestyle="None",label="Irrigation")
    ax21.bar(meteo.date,meteo.Prec, color='b')
    plt.ylim(0,50)
    plt.title(y)
    plt.legend()
    # Pluvio cumulées
    Pluvio=Pluvio_cum.T
    x=Pluvio_all.date.dt.strftime('%m-%d')
    plt.figure(figsize=(10,7))
    sns.set(style="darkgrid")
    sns.set_context('paper')
    for y,i in zip(['2006','2008','2010','2012','2014','2015','2017','2019'],np.arange(Pluvio.shape[0])):
        plt.plot(x.iloc[0:365],Pluvio.Prec.iloc[:365,i],label=y)
        plt.xticks(rotation=90)
        plt.gca().xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(10))
        plt.legend()
        plt.ylabel("Pluvio cumul")
        plt.title("Cumul pluvio")
        plt.text(x =x.iloc[-1] , y=Pluvio.Prec.iloc[-2,i],s = y,size=9)
    plt.savefig("G:/Yann_THESE/BESOIN_EAU/Calibration_SAMIR/Analyse_data/plt_Pluvio_cumul_years_LAM_mais.png")
    Pluvio_seas=Pluvio_sais.T
    x=Pluvio_all.date.dt.strftime('%m-%d')
    plt.figure(figsize=(10,7))
    sns.set(style="darkgrid")
    sns.set_context('paper')
    for y,i in zip(['2006','2008','2010','2012','2014','2015','2017','2019'],np.arange(Pluvio.shape[0])):
        plt.plot(x.iloc[120:273],Pluvio_seas.Prec.iloc[:,i],label=y)
        plt.xticks(rotation=90)
        plt.gca().xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(10))
        plt.legend()
        plt.ylabel("Pluvio cumul végétation")
        plt.title("Cumul pluvio période végétation")
        plt.text(x =x.iloc[-1] , y=Pluvio_seas.Prec.iloc[-2,i],s = y,size=9)
    plt.savefig("G:/Yann_THESE/BESOIN_EAU/Calibration_SAMIR/Analyse_data/plt_Pluvio_cumul_sais_LAM_mais.png")

# =============================================================================
# test SM 
# =============================================================================
    for y in ["2017","2019"]:
        SM_lam=pd.DataFrame()
        date_sm=[]
        for f in os.listdir("F:/THESE/CLASSIFICATION/IMG_SAT/SM/outstat_"+y+"/"):
            if ".csv" in f:
                date=f[-23:-15]
                df=pd.read_csv("F:/THESE/CLASSIFICATION/IMG_SAT/SM/outstat_"+y+"/"+f)
                df.drop(columns=['originfid', 'ogc_fid', 'wp_0_30cm', 'wp_40_50m', 'fc_0_30cm',
                   'fc_40_50cm', 'pt_sat0_30', 'pt_sat40_5', 'ru_0_30', 'ru_40_50',
                   'ru_sg_60cm', 'sdru_sg_60', 'ru_sg_0_30', 'sdru_sg0_3', 'wp_sg_60',
                   'sdwp_sg_60', 'fc_sg_60', 'sdfc_sg_60'],inplace=True)
                lam=df.loc[df.id==0.0]
                date_sm.append(date)
                SM_lam=SM_lam.append(lam)
        SM_lam["date"]=date_sm
        SM_lam.date=pd.to_datetime(SM_lam.date,format="%Y%m%d")
        SM_lam.sort_values(by="date",ascending=True,inplace=True)
        plt.figure(figsize=(10,7))
        sns.set(style="darkgrid")
        sns.set_context('paper')
        plt.plot(SM_lam.date,SM_lam.value_0/5)
        plt.title(y)
    # df=pd.read_csv(d["PC_disk"]+"PARCELLE_LABO/FLUX_SWC/SWC_LAM_N2_final_2005-2018.csv",decimal=".")
    # df["TSTAMP"]=df["TSTAMP"].apply(lambda x:x[0:10])
    # df["TSTAMP"]=pd.to_datetime(df["TSTAMP"],format="%d/%m/%Y")
    # SWC2017=df.loc[(df["TSTAMP"] >= "2010-01-01") & (df["TSTAMP"] < "2010-12-31")]
    # SWC2017_mean=SWC2017.groupby("TSTAMP").mean()
    # SWC2017_mean.columns=['SWC_A_0_f', 'SWC_0_moy', 'SWC_0_std', 'SWC_A_5_f', 'SWC_5_moy',
    #    'SWC_5_std', 'SWC_A_10_f', 'SWC_10_moy', 'SWC_10_std', 'SWC_A_30_f',
    #    'SWC_30_moy', 'SWC_30_std', 'SWC_A_50_f', 'SWC_50', 'SWC_50_std',
    #    'SWC_A_100_f', 'SWC_100', 'SWC_100_std']
    # SWC2017_mean.to_csv("G:/Yann_THESE/BESOIN_EAU//Calibration_SAMIR/DONNEES_CALIBRATION/DATA_SWC/SWC_LAM_2010.csv")
    #   # Hauteur de végétation 
    # ITK=pd.read_csv("G:/Yann_THESE/BESOIN_EAU/PARCELLE_LABO/ITK_LAM/FR-Lam_ITK_2005-2019_hauteur_vege.csv",decimal=",")
    # ITK["TIMESTAMP"]=ITK["TIMESTAMP"].apply(lambda x:x[0:10])
    # ITK["TIMESTAMP"]=pd.to_datetime(ITK["TIMESTAMP"],format="%d/%m/%Y")
    # ITK[["TIMESTAMP","Hauteur/height"]].loc[ITK.ITK=='Ma']
    
