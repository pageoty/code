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
    
    name_run="RUN_COMPAR_Version_new_data"
    years=2006
    d={}
    d['SAMIR_run_Wind']="D:/THESE_TMP/RUNS_SAMIR/"+name_run+"/"
    d["PC_disk_Wind"]="D:/THESE_TMP/RUNS_SAMIR/DATA_Validation/"
    d['PC_disk_unix']="/mnt/d/THESE_TMP/RUNS_SAMIR/DATA_Validation/"
    d["PC_disk"]="G:/Yann_THESE/BESOIN_EAU/"
# =============================================================================
#  Plot NDVI & meteo
# =============================================================================

    for y in ['2006','2008','2010','2012','2014','2015','2017','2019']:
        # df=pd.read_csv(d["PC_disk"]+"PARCELLE_LABO/FLUX_SWC/SWC_LAM_N2_Stev_2020.csv",decimal=".")
        # df["Date/Time"]=df["Date/Time"].apply(lambda x:x[0:10])
        # df["Date/Time"]=pd.to_datetime(df["Date/Time"],format="%d/%m/%Y")
        # SWC2017=df.loc[(df["Date/Time"] > "2018-12-31") & (df["Date/Time"] < "2019-12-31")]
        # SWC2017_mean=SWC2017.groupby("Date/Time").mean()
        # SWC2017_mean.to_csv("G:/Yann_THESE/BESOIN_EAU//Calibration_SAMIR/DONNEES_CALIBRATION/DATA_SWC/SWC_LAM_2019.csv")
        NDVI = pickle.load(open(d["SAMIR_run_Wind"]+str(y)+"/Inputdata/maize/NDVI.df",'rb'))
        meteo=pickle.load(open(d["SAMIR_run_Wind"]+str(y)+"/Inputdata/meteo.df",'rb'))
        SWC=pd.read_csv("G:/Yann_THESE/BESOIN_EAU//Calibration_SAMIR/DONNEES_CALIBRATION/DATA_SWC/SWC_LAM_"+str(y)+".csv")
        ETR=pd.read_csv("G:/Yann_THESE/BESOIN_EAU/DATA_ETR_CESBIO/DATA_ETR_LAM/ETR_LAM"+str(y)+".csv")
        Fcover=pd.read_csv("G:/Yann_THESE/BESOIN_EAU/Calibration_SAMIR/DONNEES_CALIBRATION/DATA_FCOVER/FCOVER_parcelles_"+str(y)+".csv")
        Fcover.date=pd.to_datetime(Fcover.date,format="%Y-%m-%d")
        ETR.date=pd.to_datetime(ETR.date,format="%Y-%m-%d")
        SWC["Date/Time"]=pd.to_datetime(SWC["Date/Time"],format="%Y-%m-%d")
        globals()["date_irr_%s"%y]=meteo.loc[meteo.Irrig > 0.0]
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.set(style="darkgrid")
        sns.set_context('paper')
        ax1 = plt.subplot(211)
        plt.plot(NDVI.date,NDVI.NDVI,label='NDVI')
        plt.plot(Fcover.date,Fcover.Fcover,label="Fcover")
        plt.ylim(0,1)
        plt.ylabel('NDVI & Fcover')
        plt.legend()
        plt.setp(ax1.get_xticklabels(),visible=False)
        plt.title(y)
        ax2=plt.subplot(212)
        plt.plot(meteo.date,meteo.ET0,color='green',label="ET0")
        plt.plot(ETR.date,ETR.LE,color="black",label='ETR')
        plt.legend(loc='upper left')
        plt.ylabel('ETO')
        plt.ylim(0,10)
        ax21 = plt.twinx()
        ax21.grid(axis='y')
        Irri=meteo.loc[meteo.Irrig >0.0]
        ax21.plot(Irri.date,Irri.Irrig,color="red",marker="o",linestyle="None",label="Irrigation")
        ax21.bar(meteo.date,meteo.Prec, color='b')
        plt.ylim(0,50)
        plt.title(y)
        plt.legend()
        plt.savefig("G:/Yann_THESE/BESOIN_EAU/Calibration_SAMIR/Analyse_data/plt_data_"+str(y)+".png")
        # plot SWC
        if y != '2019':
            RU75= (0.363-0.17)-((0.363-0.17)*75/100)
            RU70= (0.363-0.17)-((0.363-0.17)*70/100)
            plt.figure(figsize=(10,10))
            sns.set(style="darkgrid")
            sns.set_context('paper')
            ax1 = plt.subplot(211)
            plt.plot(SWC["Date/Time"],SWC.SWC_0_moy/100)
            plt.plot(SWC["Date/Time"],np.repeat(0.172,len(SWC["Date/Time"])),c="r",label="WP")
            plt.plot(SWC["Date/Time"],np.repeat(0.363,len(SWC["Date/Time"])),c="b", label="FC")
            plt.plot(SWC["Date/Time"],np.repeat(0.363-RU75,len(SWC["Date/Time"])),c="b",linestyle='--',label='RU 75 %')
            # plt.plot(SWC2017["Date/Time"],np.repeat(0.363-RU75,len(SWC2017["Date/Time"])),c="b",linestyle='--',label='RU 75 %')
            plt.ylabel('SWC en surface')
            ax1 = plt.twinx()
            ax1.grid(axis='y')
            ax1.bar(meteo.date,meteo.Prec,width=1,color="b")
            ax1.plot(Irri.date,Irri.Irrig,color="red",marker="o",linestyle="None",label="Irrigation")
            plt.ylim(0,50)
            ax2 = plt.subplot(212)
            plt.plot(SWC["Date/Time"],SWC.SWC_30_moy/100)
            plt.plot(SWC["Date/Time"],np.repeat(0.172,len(SWC["Date/Time"])),c="r",label="WP")
            plt.plot(SWC["Date/Time"],np.repeat(0.363,len(SWC["Date/Time"])),c="b", label="FC")
            plt.plot(SWC["Date/Time"],np.repeat(0.363-RU75,len(SWC["Date/Time"])),c="b",linestyle='--',label='RU 75 %')
            plt.legend()
            plt.ylabel('SWC en profondeur 30 cm')
            plt.savefig("G:/Yann_THESE/BESOIN_EAU/Calibration_SAMIR/Analyse_data/plt_data_SWC_"+str(y)+".png")
        else :
            plt.figure(figsize=(10,10))
            sns.set(style="darkgrid")
            sns.set_context('paper')
            ax1 = plt.subplot(211)
            plt.plot(SWC["Date/Time"],SWC.SWC_0)
            plt.plot(SWC["Date/Time"],np.repeat(0.172,len(SWC["Date/Time"])),c="r",label="WP")
            plt.plot(SWC["Date/Time"],np.repeat(0.363,len(SWC["Date/Time"])),c="b", label="FC")
            plt.plot(SWC["Date/Time"],np.repeat(0.363-RU75,len(SWC["Date/Time"])),c="b",linestyle='--',label='RU 75 %')
            # plt.plot(SWC2017["Date/Time"],np.repeat(0.363-RU75,len(SWC2017["Date/Time"])),c="b",linestyle='--',label='RU 75 %')
            plt.ylabel('SWC en surface')
            ax1 = plt.twinx()
            ax1.grid(axis='y')
            ax1.bar(meteo.date,meteo.Prec,width=1,color="b")
            ax1.plot(Irri.date,Irri.Irrig,color="red",marker="o",linestyle="None",label="Irrigation")
            plt.ylim(0,50)
            ax2 = plt.subplot(212)
            plt.plot(SWC["Date/Time"],SWC.SWC_5)
            plt.plot(SWC["Date/Time"],np.repeat(0.172,len(SWC["Date/Time"])),c="r",label="WP")
            plt.plot(SWC["Date/Time"],np.repeat(0.363,len(SWC["Date/Time"])),c="b", label="FC")
            plt.plot(SWC["Date/Time"],np.repeat(0.363-RU75,len(SWC["Date/Time"])),c="b",linestyle='--',label='RU 75 %')
            plt.legend()
            plt.ylabel('SWC en profondeur 5 cm')
            plt.savefig("G:/Yann_THESE/BESOIN_EAU/Calibration_SAMIR/Analyse_data/plt_data_SWC_"+str(y)+".png")
