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
    ETR_all=pd.DataFrame()
    ETR_saison=pd.DataFrame()
    ETR_bare_soil=pd.DataFrame()
    Pluvio_all=pd.DataFrame()
    Pluvio_cum=pd.DataFrame()
    Pluvio_sais=pd.DataFrame()
    for y in ['2006','2008','2010','2012','2014','2015','2017','2019']:
        ITK=pd.read_csv("G:/Yann_THESE/BESOIN_EAU/PARCELLE_LABO/ITK_LAM/ITK_LAM_"+y+".csv",decimal=",")
        ITK["TIMESTAMP"]=ITK["TIMESTAMP"].apply(lambda x:x[0:10])
        ITK["TIMESTAMP"]=pd.to_datetime(ITK["TIMESTAMP"],format="%d/%m/%Y")
        NDVI_raw=pd.read_csv("G:/Yann_THESE/BESOIN_EAU/TRAITEMENT/NDVI_parcelle/Parcelle_ref/PARCELLE_CESBIO/NDVI_raw/NDVI_parcelles_"+str(y)+".csv",decimal=",")
        NDVI_raw=NDVI_raw.T
        NDVI_raw["date"]=NDVI_raw.index
        NDVI_raw.date=pd.to_datetime(NDVI_raw.date,format="%Y-%m-%d")
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
        plt.plot(NDVI_raw.date,NDVI_raw[0],linestyle='None', marker="o")
        plt.plot(Fcover.date,Fcover.Fcover,label="Fcover")
        plt.plot([ITK.TIMESTAMP.loc[ITK.ITK!="Ma"],ITK.TIMESTAMP.loc[ITK.ITK!="Ma"]], [0.9,0.9],marker="o",color='red')
        for t,i in zip(ITK.TIMESTAMP.loc[ITK.ITK!="Ma"],ITK.ITK.loc[ITK.ITK!="Ma"]):
            plt.text(x =t , y= 0.92,s = i,size=9)
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
        if y!='2019':
            if y != "2006" and  y != "2008"and  y !="2010":
                RU75= (0.363-0.17)-((0.363-0.17)*75/100)
                RU70= (0.363-0.17)-((0.363-0.17)*70/100)
                plt.figure(figsize=(10,10))
                sns.set(style="darkgrid")
                sns.set_context('paper')
                ax1 = plt.subplot(311)
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
                ax2 = plt.subplot(312)
                plt.plot(SWC["Date/Time"],SWC.SWC_30_moy/100)
                plt.plot(SWC["Date/Time"],np.repeat(0.172,len(SWC["Date/Time"])),c="r",label="WP")
                plt.plot(SWC["Date/Time"],np.repeat(0.363,len(SWC["Date/Time"])),c="b", label="FC")
                plt.plot(SWC["Date/Time"],np.repeat(0.363-RU75,len(SWC["Date/Time"])),c="b",linestyle='--',label='RU 75 %')
                plt.legend()
                plt.ylabel('SWC en profondeur 30 cm')
                ax3 = plt.subplot(313)
                plt.plot(SWC["Date/Time"],SWC.SWC_50/100)
                plt.plot(SWC["Date/Time"],np.repeat(0.172,len(SWC["Date/Time"])),c="r",label="WP")
                plt.plot(SWC["Date/Time"],np.repeat(0.363,len(SWC["Date/Time"])),c="b", label="FC")
                plt.plot(SWC["Date/Time"],np.repeat(0.363-RU75,len(SWC["Date/Time"])),c="b",linestyle='--',label='RU 75 %')
                plt.legend()
                plt.ylabel('SWC en profondeur 50 cm')
                plt.savefig("G:/Yann_THESE/BESOIN_EAU/Calibration_SAMIR/Analyse_data/plt_data_SWC_"+str(y)+".png")
            else:   
                RU75= (0.363-0.17)-((0.363-0.17)*75/100)
                RU70= (0.363-0.17)-((0.363-0.17)*70/100)
                plt.figure(figsize=(10,10))
                sns.set(style="darkgrid")
                sns.set_context('paper')
                ax1 = plt.subplot(311)
                plt.plot(SWC["Date/Time"],SWC.SWC_10_moy/100)
                plt.plot(SWC["Date/Time"],np.repeat(0.172,len(SWC["Date/Time"])),c="r",label="WP")
                plt.plot(SWC["Date/Time"],np.repeat(0.363,len(SWC["Date/Time"])),c="b", label="FC")
                plt.plot(SWC["Date/Time"],np.repeat(0.363-RU75,len(SWC["Date/Time"])),c="b",linestyle='--',label='RU 75 %')
                # plt.plot(SWC2017["Date/Time"],np.repeat(0.363-RU75,len(SWC2017["Date/Time"])),c="b",linestyle='--',label='RU 75 %')
                plt.ylabel('SWC en profondeur 10 cm')
                ax1 = plt.twinx()
                ax1.grid(axis='y')
                ax1.bar(meteo.date,meteo.Prec,width=1,color="b")
                ax1.plot(Irri.date,Irri.Irrig,color="red",marker="o",linestyle="None",label="Irrigation")
                plt.ylim(0,50)
                ax2 = plt.subplot(312)
                plt.plot(SWC["Date/Time"],SWC.SWC_30_moy/100)
                plt.plot(SWC["Date/Time"],np.repeat(0.172,len(SWC["Date/Time"])),c="r",label="WP")
                plt.plot(SWC["Date/Time"],np.repeat(0.363,len(SWC["Date/Time"])),c="b", label="FC")
                plt.plot(SWC["Date/Time"],np.repeat(0.363-RU75,len(SWC["Date/Time"])),c="b",linestyle='--',label='RU 75 %')
                plt.legend()
                plt.ylabel('SWC en profondeur 30 cm')
                ax3 = plt.subplot(313)
                plt.plot(SWC["Date/Time"],SWC.SWC_100/100)
                plt.plot(SWC["Date/Time"],np.repeat(0.172,len(SWC["Date/Time"])),c="r",label="WP")
                plt.plot(SWC["Date/Time"],np.repeat(0.363,len(SWC["Date/Time"])),c="b", label="FC")
                plt.plot(SWC["Date/Time"],np.repeat(0.363-RU75,len(SWC["Date/Time"])),c="b",linestyle='--',label='RU 75 %')
                plt.legend()
                plt.ylabel('SWC en profondeur 100 cm')
                plt.savefig("G:/Yann_THESE/BESOIN_EAU/Calibration_SAMIR/Analyse_data/plt_data_SWC_"+str(y)+".png")
        else :
            plt.figure(figsize=(10,10))
            sns.set(style="darkgrid")
            sns.set_context('paper')
            ax1 = plt.subplot(311)
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
            ax2 = plt.subplot(312)
            plt.plot(SWC["Date/Time"],SWC.SWC_30)
            plt.plot(SWC["Date/Time"],np.repeat(0.172,len(SWC["Date/Time"])),c="r",label="WP")
            plt.plot(SWC["Date/Time"],np.repeat(0.363,len(SWC["Date/Time"])),c="b", label="FC")
            plt.plot(SWC["Date/Time"],np.repeat(0.363-RU75,len(SWC["Date/Time"])),c="b",linestyle='--',label='RU 75 %')
            plt.legend()
            plt.ylabel('SWC en profondeur 30 cm')
            ax3 = plt.subplot(313)
            plt.plot(SWC["Date/Time"],SWC.SWC_50)
            plt.plot(SWC["Date/Time"],np.repeat(0.172,len(SWC["Date/Time"])),c="r",label="WP")
            plt.plot(SWC["Date/Time"],np.repeat(0.363,len(SWC["Date/Time"])),c="b", label="FC")
            plt.plot(SWC["Date/Time"],np.repeat(0.363-RU75,len(SWC["Date/Time"])),c="b",linestyle='--',label='RU 75 %')
            plt.legend()
            plt.ylabel('SWC en profondeur 50 cm')
            plt.savefig("G:/Yann_THESE/BESOIN_EAU/Calibration_SAMIR/Analyse_data/plt_data_SWC_"+str(y)+".png")
        # cumul ETR
        ETR_all=ETR_all.append(ETR)
        ETR_saison=ETR_saison.append(ETR.LE.iloc[120:273].cumsum())
        ETR_bare_soil=ETR_bare_soil.append(ETR.LE.iloc[0:120].cumsum())
        Pluvio_all=Pluvio_all.append(meteo)
        Pluvio_cum=Pluvio_cum.append(meteo.Prec.cumsum())
        Pluvio_sais=Pluvio_sais.append(meteo.Prec.iloc[120:273].cumsum())
        # plt.figure(figsize=(10,7))
        # sns.set(style="darkgrid")
        # sns.set_context('paper')
        # plt.plot(ETR.date,ETR.LE.cumsum(),label='ETR cumuljournalier annuel',color="black")
        # plt.plot(meteo.date,meteo.ET0.cumsum(),label='ET0 cumuljournalier annuel', color="darkgreen")
        # plt.plot(ETR.date.iloc[120:273],ETR.LE.iloc[120:273].cumsum(),label="ETR cumul journalier saison",color='black')
        # plt.plot(meteo.date.iloc[120:273],meteo.ET0.iloc[120:273].cumsum(),label="ET0 cumul journalier saison", color="green")
        # plt.legend()
        # plt.ylabel("ETR cumul")
        # plt.ylim(0,1000)
        # plt.savefig("G:/Yann_THESE/BESOIN_EAU/Calibration_SAMIR/Analyse_data/plt_data_ETR_cumul_"+str(y)+".png")
    ETR_seas=ETR_saison.T
    x=ETR_all.date.iloc[120:273].dt.strftime('%m-%d')
    plt.figure(figsize=(10,7))
    sns.set(style="darkgrid")
    sns.set_context('paper')
    for y,i in zip(['2006','2008','2010','2012','2014','2015','2017','2019'],np.arange(ETR_saison.shape[0])):
        plt.plot(x,ETR_seas.LE.iloc[:,i],label=y)
        plt.xticks(rotation=90)
        plt.gca().xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(10))
        plt.legend()
        plt.ylabel("ETR cumul")
        plt.title("Cumul ETR saison vege")
        plt.text(x =x.iloc[-1] , y=ETR_seas.LE.iloc[-2,i],s = y,size=9)
    plt.savefig("G:/Yann_THESE/BESOIN_EAU/Calibration_SAMIR/Analyse_data/plt_ETR_cumul_season_LAM_mais.png")
    # ETR cumul sol nul
    x=ETR_all.date.iloc[0:120].dt.strftime('%m-%d')
    ETR_bare=ETR_bare_soil.T
    plt.figure(figsize=(10,7))
    sns.set(style="darkgrid")
    sns.set_context('paper')
    for y,i in zip(['2006','2008','2010','2012','2014','2015','2017','2019'],np.arange(ETR_bare_soil.shape[0])):
        plt.plot(x,ETR_bare.LE.iloc[:,i],label=y)
        plt.xticks(rotation=90)
        plt.gca().xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(10))
        plt.legend()
        plt.ylabel("ETR cumul")
        plt.title("Cumul sol nu")
        plt.text(x =x.iloc[-1] , y=ETR_bare.LE.iloc[-1,i],s = y,size=9)
    plt.savefig("G:/Yann_THESE/BESOIN_EAU/Calibration_SAMIR/Analyse_data/plt_ETR_cumul_bare_soil_LAM_mais.png")
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
    
