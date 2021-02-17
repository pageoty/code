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
    
    name_run="RUN_COMPAR_Version_new_data"
    # years=2006
    d={}
    d['SAMIR_run_Wind']="D:/THESE_TMP/RUNS_SAMIR/"+name_run+"/"
    d["PC_disk_home"]="D:/THESE_TMP/"
    d['PC_disk_unix']="/mnt/d/THESE_TMP/RUNS_SAMIR/DATA_Validation/"
    d["PC_disk"]="G:/Yann_THESE/BESOIN_EAU/"
    d["PC_labo"]="/datalocal/vboxshare/THESE/BESOIN_EAU/"
# =============================================================================
#  Plot NDVI & meteo
# =============================================================================
    ETR_all=pd.DataFrame()
    ETR_saison=pd.DataFrame()
    ETR_bare_soil=pd.DataFrame()
    Pluvio_all=pd.DataFrame()
    Pluvio_cum=pd.DataFrame()
    Pluvio_sais=pd.DataFrame()
    LAI_inter_all=pd.DataFrame()
    NDVI_all=pd.DataFrame()
    Pluvio_bare_soil=pd.DataFrame()
    Ta_all=pd.DataFrame()
    Ta_cum=pd.DataFrame()
    Ta_seas=pd.DataFrame()
    Ta_soil=pd.DataFrame()
    for y in ['2015']:
        d["PC_disk_home"]="D:/THESE_TMP/"
        ITK=pd.read_csv(d["PC_disk_home"]+"DONNEES_RAW/PARCELLE_LABO/ITK_LAM/ITK_LAM_"+y+".csv",decimal=",",sep=";")
        ITK["TIMESTAMP"]=ITK["TIMESTAMP"].apply(lambda x:x[0:10])
        ITK["TIMESTAMP"]=pd.to_datetime(ITK["TIMESTAMP"],format="%d/%m/%Y")
        NDVI=pd.read_csv(d["PC_disk_home"]+"/TRAITEMENT/INPUT_DATA/NDVI_parcelle/Parcelle_ref/PARCELLE_CESBIO/LAMOTHE_NDVI_"+str(y)+".csv",decimal=".")
        NDVI.date=pd.to_datetime(NDVI.date,format="%Y-%m-%d")
        # LAI=pd.read_csv(d["PC_disk_home"]+'/TRAITEMENT/INPUT_DATA/LAI_parcelle/PARCELLE_LABO/LAI_LAM_'+y+'.csv')
        # LAI.date=pd.to_datetime(LAI.date,format="%Y-%m-%d")
        # LAI.set_index("date",inplace=True)
        # LAI_inter=LAI.resample("D").asfreq().interpolate()
        # LAI_sigmo=pd.read_csv(d["PC_disk_home"]+'/TRAITEMENT/INPUT_DATA/LAI_parcelle/PARCELLE_LABO/LAI_pre_inter_OTB/LAI_inter_sat_'+str(y)+'.csv')
        # LAI_sigmo=LAI_sigmo.T[2:]
        # LAI_sigmo["date"]=pd.date_range(str(y)+"-01-01", periods=365)
        # LAI_sigmo.columns=["LAI","date"]
        # LAI_sigmo.to_csv(d["PC_disk_home"]+'/TRAITEMENT/INPUT_DATA/LAI_parcelle/PARCELLE_LABO/LAI_pre_inter_OTB/LAI_inter_dat_date_'+str(y)+'.csv')
        
        # ############## Add relation NDVI/Kcb dans le dataframe 
        NDVI['real_Rocha']=NDVI.eval("NDVI*1.37-0.017")
        NDVI['real_Kamble_spe']=NDVI.eval("NDVI*1.457-0.1725")
        NDVI['real_Kamble_Gene']=NDVI.eval("NDVI*1.358-0.0744")
        NDVI['real_Neale']=NDVI.eval("NDVI*1.81+0.026")
        NDVI['real_Calera']=NDVI.eval("NDVI*1.56-0.1")
        NDVI['real_Demarez']=NDVI.eval("NDVI*0.99+0.08")
        NDVI['real_Toureiro']=NDVI.eval("NDVI*1.46-0.25")
        ###### ADD FCOVER_SAMIR
        NDVI["rela_fcover"]=NDVI.eval("NDVI*1.25-0.13")
        meteo_SAF=pd.read_csv(d["PC_disk_home"]+"/TRAITEMENT/INPUT_DATA/DATA_METEO_BV/PARCELLE_LAM/meteo_lam_"+str(y)+".csv",decimal=".")
        meteo_temp=pd.read_csv(d["PC_disk_home"]+"TRAITEMENT/INPUT_DATA/DATA_METEO_BV/PARCELLE_LAM/DATA_PREC_TEMP/meteo_prec_temp_"+y+".csv")
        meteo_temp.columns=["date","Ta","Prec"]
        SWC=pd.read_csv(d["PC_disk_home"]+"/TRAITEMENT/DATA_VALIDATION/DATA_SWC/SWC_LAM/SWC_LAM_"+str(y)+".csv")
        ETR=pd.read_csv(d["PC_disk_home"]+"/DATA_ETR_CESBIO/DATA_ETR_LAM/DATA_ETR_LAM_ICOS/ETR_LAM"+str(y)+".csv")
        Fcover=pd.read_csv(d["PC_disk_home"]+"TRAITEMENT/INPUT_DATA/FCOVER_parcelle/PARCELLE_CESBIO/FCOVER_parcelles_"+str(y)+".csv")
        Fcover.date=pd.to_datetime(Fcover.date,format="%Y-%m-%d")
        Fcover.columns=["date","Fcover"]
        Fcover_sigmo=pd.read_csv(d["PC_disk_home"]+"TRAITEMENT/INPUT_DATA/FCOVER_parcelle/PARCELLE_CESBIO/Fcover_sigmo_2019.csv")
        Fcover_sigmo.date=pd.to_datetime(Fcover_sigmo.date,format="%Y-%m-%d")
        ETR.date=pd.to_datetime(ETR.date,format="%Y-%m-%d")
        meteo_SAF.date=pd.to_datetime(meteo_SAF.date,format="%Y-%m-%d")
        meteo_temp.date=pd.to_datetime(meteo_temp.date,format="%Y-%m-%d")
        SWC["Date"]=pd.to_datetime(SWC["Date/Time"],format="%Y-%m-%d")
        # SWC.loc[SWC.SWC_50 == 0.000000]=np.nan
        dfnames=pd.read_csv(d["PC_disk_home"]+"/TRAITEMENT/INPUT_DATA/SAR_parcelle/PARCELLE_CESBIO/list_features_SAR31TCJ2019.txt",header=None)
        colnames=dfnames.T.iloc[0:37]
        date=colnames[0].apply(lambda x:x[-9:-1])
        # SAR=pd.read_csv(d["PC_labo"]+"/TRAITEMENT/INPUT_DATA/SAR_parcelle/PARCELLE_CESBIO/SampleExtractionVV_LAM_0112.tif.csv")
        # SAR=SAR[SAR.nom_parcel=="Lamothe"]
        # SAR.drop(columns=['num_com', 'nom_com', 'centroidx', 'centroidy','shape_leng', 'shape_area', 'labo','id','originfid',"value_37",'nom_parcel'],inplace=True)
        # SAR=SAR.T
        # SAR["date"]=date.to_list()
        # SAR.date=pd.to_datetime(SAR["date"],format="%Y%m%d")
        # SAR.set_index("date",inplace=True)
        # SAR_mean=SAR.T.mean()
        globals()["date_irr_%s"%y]=meteo_SAF.loc[meteo_SAF.Irrig > 0.0]
        # print(ETR.LE.max())
        #  # ############## plot relation NDVI/Kcb dans le dataframe 
        # plt.figure(figsize=(12,10))
        # for rela in NDVI.columns[2:-1]:
        #     print(rela)
        #     plt.plot(NDVI.date,NDVI[rela],label=rela)
        # plt.ylabel("Kcb")
        # plt.legend()
        # plt.savefig(d["PC_disk_Wind"]+"RESULT/PLOT/Analyse_Flux_ICOS/plt_RELA_NDVI_Kcb_LAM_2019.png")
        fig, ax = plt.subplots(figsize=(7, 7))
        # # sns.set(style="darkgrid")
        # # sns.set_context('paper')
        # ax1 = plt.subplot(211)
        plt.plot(NDVI.date,NDVI.NDVI,label='NDVI')
        plt.plot(Fcover.date,Fcover.Fcover,label="Fcover")
        # plt.plot(Fcover_sigmo.date,Fcover_sigmo.FCOVER,label="Fcover_sigmo")
        # plt.plot(NDVI.date,NDVI.rela_fcover,label="FCOVER_SAMIR")
        plt.plot([ITK.TIMESTAMP.loc[ITK.ITK!="Ma"],ITK.TIMESTAMP.loc[ITK.ITK!="Ma"]], [0.9,0.9],marker="o",color='red')
        for t,i in zip(ITK.TIMESTAMP.loc[ITK.ITK!="Ma"],ITK.ITK.loc[ITK.ITK!="Ma"]):
            plt.text(x =t , y= 0.92,s = i,size=9)
        plt.ylim(0,1)
        plt.ylabel('NDVI & Fcover')
        plt.legend()
        # plt.setp(ax1.get_xticklabels(),visible=False)
        plt.title(y)
        plt.savefig(d["PC_disk_home"]+"RESULT/PLOT/Analyse_Flux_ICOS/plt_data_lam_Fcover_NDVI_"+str(y)+".png")
        # ax2=plt.subplot(212)
        # plt.plot(meteo.date,meteo.ET0,color='green',label="ET0")
        # plt.plot(ETR.date,ETR.LE,color="black",label='ETR')
        # plt.legend(loc='upper left')
        # plt.ylabel('ETO')
        # plt.ylim(0,10)
        # ax21 = plt.twinx()
        # ax21.grid(axis='y')
        Irri=meteo_SAF.loc[meteo_SAF.Irrig >0.0]
        # ax21.plot(Irri.date,Irri.Irrig,color="red",marker="o",linestyle="None",label="Irrigation")
        # ax21.bar(meteo.date,meteo.Prec, color='b')
        # plt.ylim(0,50)
        # plt.title(y)
        # plt.legend()
        # plt.savefig(d["PC_disk_Wind"]+"RESULT/PLOT/Analyse_Flux_ICOS/plt_data_lam_"+str(y)+".png")
# =============================================================================
#         Calcule Kcb realtion LAI
# =============================================================================
        # kc_tab=pd.DataFrame()
        # kc=1.15*(1-np.exp(-0.34*LAI_sigmo.LAI))
        # kc_tab=kc_tab.append([kc])
        # kc_tab=kc_tab.T
        # kc_tab["date"]=pd.date_range(str(y)+"-01-01", periods=365)
        
        # # plt.plot(kc_tab.date,kc_tab.LAI)
        # LAI_inter_all=LAI_inter_all.append([LAI_inter])
        # NDVI_all=NDVI_all.append([NDVI])
        
        # # plt.figure(figsize=(12,10))
        # # plt.plot(LAI_inter.index,LAI_inter.GAI,label="GAI",color='green')
        # # plt.ylim(0,7)
        # # plt.ylabel("GAI sta")
        # # plt.legend(loc='upper left')
        # # ax1 = plt.twinx()
        # # ax1.plot(NDVI.date,NDVI.NDVI,label='NDVI',color='black')
        # # ax1.grid(axis='y')
        # # ax1.legend()
        # # plt.savefig(d["PC_disk_Wind"]+"RESULT/PLOT/Analyse_Flux_ICOS/plt_data_lam_GAI_NDVI"+str(y)+".png")
        # a=NDVI.iloc[NDVI.NDVI.idxmax()]["NDVI"]
        # b=LAI_inter.loc[LAI_inter.index==NDVI.iloc[NDVI.NDVI.idxmax()]["date"]]["GAI"]
        # print(r'NDVI :%s & GAI : %s'%(a,b[0]))
        # c=meteo.loc[meteo.date==NDVI.iloc[NDVI.NDVI.idxmax()]["date"]]["ET0"]
        # d=ETR.loc[ETR.date==NDVI.iloc[NDVI.NDVI.idxmax()]["date"]]["LE"]
        # # print (r'ETR : %s'%d)
        # # print(r'ET0 :%s'%c)
        # print(r'resultats Kc :%s'%(d/c))
        # k=kc_tab.loc[kc_tab.date==NDVI.iloc[NDVI.NDVI.idxmax()]["date"]]["LAI"]
        # print (y)
        # print(r'value kcb/lai : %s'%k)
# =============================================================================
#         plot SWC
# =============================================================================
        RU75= (0.363-0.17)-((0.363-0.17)*60/100)
        RU70= (0.363-0.17)-((0.363-0.17)*70/100)    
        if y!='2019':
            if y != "2006" and  y != "2008"and  y !="2010":
                plt.figure(figsize=(10,10))
                # sns.set(style="darkgrid")
                # sns.set_context('paper')
                ax1 = plt.subplot(311)
                plt.plot(SWC["Date"],SWC.SWC_0_moy/100)
                plt.plot(SWC["Date"],np.repeat(0.172,len(SWC["Date"])),c="r",label="WP")
                plt.plot(SWC["Date"],np.repeat(0.363,len(SWC["Date"])),c="b", label="FC")
                plt.plot(SWC["Date"],np.repeat(0.363-RU75,len(SWC["Date"])),c="b",linestyle='--',label='RU 60 %')
                # plt.plot(SWC2017["Date/Time"],np.repeat(0.363-RU75,len(SWC2017["Date/Time"])),c="b",linestyle='--',label='RU 75 %')
                plt.ylabel('SWC en surface')
                ax1 = plt.twinx()
                ax1.grid(axis='y')
                ax1.bar(meteo_SAF.date,meteo_SAF.Prec,width=1,color="b")
                ax1.plot(Irri.date,Irri.Irrig,color="red",marker="o",linestyle="None",label="Irrigation")
                plt.ylim(0,50)
                ax2 = plt.subplot(312)
                plt.plot(SWC["Date"],SWC.SWC_30_moy/100)
                plt.plot(SWC["Date"],np.repeat(0.172,len(SWC["Date"])),c="r",label="WP")
                plt.plot(SWC["Date"],np.repeat(0.363,len(SWC["Date"])),c="b", label="FC")
                plt.plot(SWC["Date"],np.repeat(0.363-RU75,len(SWC["Date"])),c="b",linestyle='--',label='RU 60 %')
                plt.legend()
                plt.ylabel('SWC en profondeur 30 cm')
                ax3 = plt.subplot(313)
                plt.plot(SWC["Date"],SWC.SWC_50/100)
                plt.plot(SWC["Date"],np.repeat(0.172,len(SWC["Date"])),c="r",label="WP")
                plt.plot(SWC["Date"],np.repeat(0.363,len(SWC["Date"])),c="b", label="FC")
                plt.plot(SWC["Date"],np.repeat(0.363-RU75,len(SWC["Date"])),c="b",linestyle='--',label='RU 60 %')
                plt.legend()
                plt.ylabel('SWC en profondeur 50 cm')
                # plt.savefig(d["PC_labo"]+"RESULT/PLOT/Analyse_Flux_ICOS/plt_data_SWC_lam_"+str(y)+".png")
                # plt.savefig("G:/Yann_THESE/BESOIN_EAU/Calibration_SAMIR/Analyse_data/plt_data_SWC_"+str(y)+".png")
            else:   
                plt.figure(figsize=(10,10))
                # sns.set(style="darkgrid")
                # sns.set_context('paper')
                ax1 = plt.subplot(311)
                plt.plot(SWC["Date"],SWC.SWC_10_moy/100)
                plt.plot(SWC["Date"],np.repeat(0.172,len(SWC["Date"])),c="r",label="WP")
                plt.plot(SWC["Date"],np.repeat(0.363,len(SWC["Date"])),c="b", label="FC")
                plt.plot(SWC["Date"],np.repeat(0.363-RU75,len(SWC["Date"])),c="b",linestyle='--',label='RU 60 %')
                # plt.plot(SWC2017["Date/Time"],np.repeat(0.363-RU75,len(SWC2017["Date/Time"])),c="b",linestyle='--',label='RU 75 %')
                plt.ylabel('SWC en profondeur 10 cm')
                ax1 = plt.twinx()
                ax1.grid(axis='y')
                ax1.bar(meteo_SAF.date,meteo_SAF.Prec,width=1,color="b")
                ax1.plot(Irri.date,Irri.Irrig,color="red",marker="o",linestyle="None",label="Irrigation")
                plt.ylim(0,50)
                ax2 = plt.subplot(312)
                plt.plot(SWC["Date"],SWC.SWC_30_moy/100)
                plt.plot(SWC["Date"],np.repeat(0.172,len(SWC["Date"])),c="r",label="WP")
                plt.plot(SWC["Date"],np.repeat(0.363,len(SWC["Date"])),c="b", label="FC")
                plt.plot(SWC["Date"],np.repeat(0.363-RU75,len(SWC["Date"])),c="b",linestyle='--',label='RU 60 %')
                plt.legend()
                plt.ylabel('SWC en profondeur 30 cm')
                ax3 = plt.subplot(313)
                plt.plot(SWC["Date"],SWC.SWC_100/100)
                plt.plot(SWC["Date"],np.repeat(0.172,len(SWC["Date"])),c="r",label="WP")
                plt.plot(SWC["Date"],np.repeat(0.363,len(SWC["Date"])),c="b", label="FC")
                plt.plot(SWC["Date"],np.repeat(0.363-RU75,len(SWC["Date"])),c="b",linestyle='--',label='RU 60 %')
                plt.legend()
                plt.ylabel('SWC en profondeur 100 cm')
                # plt.savefig(d["PC_labo"]+"RESULT/PLOT/Analyse_Flux_ICOS/plt_data_SWC_lam_"+str(y)+".png")
                # plt.savefig("G:/Yann_THESE/BESOIN_EAU/Calibration_SAMIR/Analyse_data/plt_data_SWC_"+str(y)+".png")
        else :
            plt.figure(figsize=(10,10))
            # sns.set(style="darkgrid")
            # sns.set_context('paper')
            ax1 = plt.subplot(311)
            # plt.plot(SAR_mean)
            plt.plot(SWC["Date"],np.repeat(0.172,len(SWC["Date"])),c="r",label="WP")
            plt.plot(SWC["Date"],np.repeat(0.363,len(SWC["Date"])),c="b", label="FC")
            plt.plot(SWC["Date"],np.repeat(0.363-RU75,len(SWC["Date"])),c="b",linestyle='--',label='RU 60 %')
            plt.plot(SWC["Date"],SWC.SWC_0)
            # plt.plot(SWC2017["Date/Time"],np.repeat(0.363-RU75,len(SWC2017["Date/Time"])),c="b",linestyle='--',label='RU 75 %')
            plt.ylabel('SWC en surface')
            ax1 = plt.twinx()
            ax1.grid(axis='y')
            ax1.bar(meteo_SAF.date,meteo_SAF.Prec,width=1,color="b")
            ax1.plot(Irri.date,Irri.Irrig,color="red",marker="o",linestyle="None",label="Irrigation")
            plt.ylim(0,50)
            ax2 = plt.subplot(312)
            plt.plot(SWC["Date"],SWC.SWC_30)
            plt.plot(SWC["Date"],np.repeat(0.172,len(SWC["Date/Time"])),c="r",label="WP")
            plt.plot(SWC["Date"],np.repeat(0.363,len(SWC["Date/Time"])),c="b", label="FC")
            plt.plot(SWC["Date"],np.repeat(0.363-RU75,len(SWC["Date/Time"])),c="b",linestyle='--',label='RU 60 %')
            plt.legend()
            plt.ylabel('SWC en profondeur 30 cm')
            ax3 = plt.subplot(313)
            plt.plot(SWC["Date"],SWC.SWC_50)
            plt.plot(SWC["Date"],np.repeat(0.172,len(SWC["Date/Time"])),c="r",label="WP")
            plt.plot(SWC["Date"],np.repeat(0.363,len(SWC["Date/Time"])),c="b", label="FC")
            plt.plot(SWC["Date"],np.repeat(0.363-RU75,len(SWC["Date/Time"])),c="b",linestyle='--',label='RU 60 %')
            plt.legend()
            plt.ylabel('SWC en profondeur 50 cm')
            # plt.savefig(d["PC_labo"]+"RESULT/PLOT/Analyse_Flux_ICOS/plt_data_SWC_lam_"+str(y)+".png")
# =============================================================================
#       Comparaison flux SWC et polarisation VV
# =============================================================================

        
    #     # cumul ETR
    #     ETR_all=ETR_all.append(ETR)
    #     ETR_saison=ETR_saison.append(ETR.LE.iloc[120:273].cumsum())
    #     ETR_bare_soil=ETR_bare_soil.append(ETR.LE.iloc[60:120].cumsum())
    #     Pluvio_all=Pluvio_all.append(meteo_temp)
    #     Pluvio_cum=Pluvio_cum.append(meteo_temp.Prec.cumsum())
    #     Pluvio_sais=Pluvio_sais.append(meteo_temp.Prec.iloc[120:273].cumsum())
    #     Pluvio_bare_soil=Pluvio_bare_soil.append(meteo_temp.Prec.iloc[60:120].cumsum())
    #     Ta_all=Ta_all.append(meteo_temp)
    #     Ta_cum=Ta_cum.append(meteo_temp.Ta.cumsum())
    #     Ta_seas=Ta_seas.append(meteo_temp.Ta.iloc[120:273].cumsum())
    #     Ta_soil=Ta_soil.append(meteo_temp.Ta.iloc[60:120].cumsum())
    #     plt.figure(figsize=(10,7))
    #     # sns.set(style="darkgrid")
    #     # sns.set_context('paper')
    #     plt.plot(ETR.date,ETR.LE.cumsum(),label='ETR cumuljournalier annuel',color="black")
    #     plt.plot(meteo_SAF.date,meteo_SAF.ET0.cumsum(),label='ET0 cumuljournalier annuel', color="darkgreen")
    #     plt.plot(ETR.date.iloc[120:273],ETR.LE.iloc[120:273].cumsum(),label="ETR cumul journalier saison",color='black')
    #     plt.plot(meteo_SAF.date.iloc[120:273],meteo_SAF.ET0.iloc[120:273].cumsum(),label="ET0 cumul journalier saison", color="green")
    #     plt.legend()
    #     plt.ylabel("ETR cumul")
    #     plt.ylim(0,1000)
    #     plt.savefig("H:/Yann_THESE/BESOIN_EAU/BESOIN_EAU/Calibration_SAMIR/Analyse_data/plt_data_ETR_cumul_"+str(y)+".png")
    # ETR_seas=ETR_saison.T
    # x=ETR_all.date.iloc[120:273].dt.strftime('%m-%d')
    # plt.figure(figsize=(10,7))
    # # sns.set(style="darkgrid")
    # # sns.set_context('paper')
    # for y,i in zip(['2006','2008','2010','2012','2014','2015','2019'],np.arange(ETR_seas.shape[0])):
    #     plt.plot(x,ETR_seas.LE.iloc[:,i],label=y)
    #     plt.xticks(rotation=90)
    #     plt.gca().xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(10))
    #     plt.legend()
    #     plt.ylabel("ETR cumul")
    #     plt.title("Cumul ETR saison vege")
    #     plt.text(x =x.iloc[-1] , y=ETR_seas.LE.iloc[-2,i],s = y,size=9)
    # plt.savefig("H:/Yann_THESE/BESOIN_EAU/BESOIN_EAU/Calibration_SAMIR/Analyse_data/plt_ETR_cumul_season_LAM_mais.png")
    # # # ETR cumul sol nul
    # x=ETR_all.date.iloc[60:120].dt.strftime('%m-%d')
    # ETR_bare=ETR_bare_soil.T
    # plt.figure(figsize=(10,7))
    # # sns.set(style="darkgrid")
    # # sns.set_context('paper')
    # for y,i in zip(['2006','2008','2010','2012','2014','2015','2019'],np.arange(ETR_bare_soil.shape[0])):
    #     plt.plot(x,ETR_bare.LE.iloc[:,i],label=y)
    #     plt.xticks(rotation=90)
    #     plt.gca().xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(10))
    #     plt.legend()
    #     plt.ylabel("ETR cumul")
    #     plt.title("Cumul sol nu")
    #     plt.text(x =x.iloc[-1] , y=ETR_bare.LE.iloc[-1,i],s = y,size=9)
    # plt.savefig("H:/Yann_THESE/BESOIN_EAU/BESOIN_EAU/Calibration_SAMIR/Analyse_data/plt_ETR_cumul_bare_soil_LAM_mais.png")
    # # # Pluvio cumulées
    # Pluvio=Pluvio_cum.T
    # x=Pluvio_all.date.dt.strftime('%m-%d')
    # plt.figure(figsize=(10,7))
    # # sns.set(style="darkgrid")
    # # sns.set_context('paper')
    # for y,i in zip(['2006','2008','2010','2012','2014','2015','2019'],np.arange(Pluvio.shape[0])):
    #     plt.plot(x.iloc[0:365],Pluvio.Prec.iloc[:365,i],label=y)
    #     plt.xticks(rotation=90)
    #     plt.gca().xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(10))
    #     plt.legend()
    #     plt.ylabel("Pluvio cumul")
    #     plt.title("Cumul pluvio")
    #     plt.text(x =x.iloc[-1] , y=Pluvio.Prec.iloc[-2,i],s = y,size=9)
    # plt.savefig("H:/Yann_THESE/BESOIN_EAU/BESOIN_EAU/Calibration_SAMIR/Analyse_data/plt_Pluvio_cumul_years_LAM_mais.png")
    # Pluvio_seas=Pluvio_sais.T
    # x=Pluvio_all.date.dt.strftime('%m-%d')
    # plt.figure(figsize=(10,7))
    # # sns.set(style="darkgrid")
    # # sns.set_context('paper')
    # for y,i in zip(['2006','2008','2010','2012','2014','2015','2019'],np.arange(Pluvio.shape[0])):
    #     plt.plot(x.iloc[120:273],Pluvio_seas.Prec.iloc[:,i],label=y)
    #     plt.xticks(rotation=90)
    #     plt.gca().xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(10))
    #     plt.legend()
    #     plt.ylabel("Pluvio cumul végétation")
    #     plt.title("Cumul pluvio période végétation")
    #     plt.text(x =x.iloc[-1] , y=Pluvio_seas.Prec.iloc[-2,i],s = y,size=9)
    # plt.savefig("H:/Yann_THESE/BESOIN_EAU/BESOIN_EAU/Calibration_SAMIR/Analyse_data/plt_Pluvio_cumul_sais_LAM_mais.png")
    # Pluvio_bare_soil=Pluvio_bare_soil.T
    # plt.figure(figsize=(10,7))
    # # sns.set(style="darkgrid")
    # # sns.set_context('paper')
    # for y,i in zip(['2006','2008','2010','2012','2014','2015','2019'],np.arange(Pluvio.shape[0])):
    #     plt.plot(x.iloc[60:120],Pluvio_bare_soil.Prec.iloc[:,i],label=y)
    #     plt.xticks(rotation=90)
    #     plt.gca().xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(10))
    #     plt.legend()
    #     plt.ylabel("Pluvio cumul sol nu")
    #     plt.title("Cumul pluvio période sol un")
    #     plt.text(x =x.iloc[120] , y=Pluvio_bare_soil.Prec.iloc[-2,i],s = y,size=9)
    # plt.savefig("H:/Yann_THESE/BESOIN_EAU/BESOIN_EAU/Calibration_SAMIR/Analyse_data/plt_Pluvio_cumul_sol_nu_LAM_mais.png")
    
    # # Temp cum
    # Ta=Ta_cum.T
    # x=Ta_all.date.dt.strftime('%m-%d')
    # plt.figure(figsize=(10,7))
    # # sns.set(style="darkgrid")
    # # sns.set_context('paper')
    # for y,i in zip(['2006','2008','2010','2012','2014','2015','2019'],np.arange(Ta.shape[0])):
    #     plt.plot(x.iloc[0:365],Ta.Ta.iloc[:365,i],label=y)
    #     plt.xticks(rotation=90)
    #     plt.gca().xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(10))
    #     plt.legend()
    #     plt.ylabel("température cumul")
    #     plt.title("Cumul Température")
    #     plt.text(x =x.iloc[-1] , y=Ta.Ta.iloc[-2,i],s = y,size=9)
    # plt.savefig("H:/Yann_THESE/BESOIN_EAU/BESOIN_EAU/Calibration_SAMIR/Analyse_data/plt_Ta_cumul_years_LAM_mais.png")
    # Ta_seas=Ta_seas.T
    # x=Pluvio_all.date.dt.strftime('%m-%d')
    # plt.figure(figsize=(10,7))
    # # sns.set(style="darkgrid")
    # # sns.set_context('paper')
    # for y,i in zip(['2006','2008','2010','2012','2014','2015','2019'],np.arange(Ta.shape[0])):
    #     plt.plot(x.iloc[120:273],Ta_seas.Ta.iloc[:,i],label=y)
    #     plt.xticks(rotation=90)
    #     plt.gca().xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(10))
    #     plt.legend()
    #     plt.ylabel("Température cumul végétation")
    #     plt.title("Cumul Température période végétation")
    #     plt.text(x =x.iloc[-1] , y=Ta_seas.Ta.iloc[-2,i],s = y,size=9)
    # plt.savefig("H:/Yann_THESE/BESOIN_EAU/BESOIN_EAU/Calibration_SAMIR/Analyse_data/plt_Ta_cumul_sais_LAM_mais.png")
    # Ta_soil=Ta_soil.T
    # plt.figure(figsize=(10,7))
    # # sns.set(style="darkgrid")
    # # sns.set_context('paper')
    # for y,i in zip(['2006','2008','2010','2012','2014','2015','2019'],np.arange(Ta.shape[0])):
    #     plt.plot(x.iloc[60:120],Ta_soil.Ta.iloc[:,i],label=y)
    #     plt.xticks(rotation=90)
    #     plt.gca().xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(10))
    #     plt.legend()
    #     plt.ylabel("Température cumul sol nu")
    #     plt.title("Cumul Température période sol un")
    #     plt.text(x =x.iloc[120] , y=Ta_soil.Ta.iloc[-2,i],s = y,size=9)
    # plt.savefig("H:/Yann_THESE/BESOIN_EAU/BESOIN_EAU/Calibration_SAMIR/Analyse_data/plt_Ta_cumul_sol_nu_LAM_mais.png")
    # plt.figure(figsize=(10,7))
    # for y,i in zip(['2006','2008','2010','2012','2014','2015','2019'],np.arange(ETR_seas.shape[0])):
    #     plt.scatter( ETR_seas.LE.iloc[:,i],Pluvio_seas.Prec.iloc[:,i],label=y)
    #     plt.text(ETR_seas.LE.iloc[-2,i],Pluvio_seas.Prec.iloc[-1,i],s=y)
    #     plt.legend()
    #     plt.xlabel("ETR cumulées sur la période de végétation")
    #     plt.ylabel("Pluvio cumulées sur la période de végétation")
    #     plt.savefig("H:/Yann_THESE/BESOIN_EAU/BESOIN_EAU/Calibration_SAMIR/Analyse_data/plt_scatter_cumul_seas_TER_plui_LAM_mais.png",dpi=600)
    # plt.figure(figsize=(10,7))
    # for y,i in zip(['2006','2008','2010','2012','2014','2015','2019'],np.arange(ETR_seas.shape[0])):
    #     plt.scatter( ETR_seas.LE.iloc[:,i],Ta_seas.Ta.iloc[:,i],label=y)
    #     plt.text(ETR_seas.LE.iloc[-2,i],Ta_seas.Ta.iloc[-1,i],s=y)
    #     plt.legend()
    #     plt.xlabel("ETR cumulées sur la période de végétation")
    #     plt.ylabel("Ta cumulées sur la période de végétation")
    #     plt.savefig("H:/Yann_THESE/BESOIN_EAU/BESOIN_EAU/Calibration_SAMIR/Analyse_data/plt_scatter_cumul_seas_TER_Temp_LAM_mais.png",dpi=600)
# =============================================================================
# test SM 
# =============================================================================
#     for y in ["2017","2019"]:
#         SM_lam=pd.DataFrame()
#         date_sm=[]
#         for f in os.listdir("F:/THESE/CLASSIFICATION/IMG_SAT/SM/outstat_"+y+"/"):
#             if ".csv" in f:
#                 date=f[-23:-15]
#                 df=pd.read_csv("F:/THESE/CLASSIFICATION/IMG_SAT/SM/outstat_"+y+"/"+f)
#                 df.drop(columns=['originfid', 'ogc_fid', 'wp_0_30cm', 'wp_40_50m', 'fc_0_30cm',
#                    'fc_40_50cm', 'pt_sat0_30', 'pt_sat40_5', 'ru_0_30', 'ru_40_50',
#                    'ru_sg_60cm', 'sdru_sg_60', 'ru_sg_0_30', 'sdru_sg0_3', 'wp_sg_60',
#                    'sdwp_sg_60', 'fc_sg_60', 'sdfc_sg_60'],inplace=True)
#                 lam=df.loc[df.id==0.0]
#                 date_sm.append(date)
#                 SM_lam=SM_lam.append(lam)
#         SM_lam["date"]=date_sm
#         SM_lam.date=pd.to_datetime(SM_lam.date,format="%Y%m%d")
#         SM_lam.sort_values(by="date",ascending=True,inplace=True)
#         plt.figure(figsize=(10,7))
#         sns.set(style="darkgrid")
#         sns.set_context('paper')
#         plt.plot(SM_lam.date,SM_lam.value_0/5)
#         plt.title(y)
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
# =============================================================================
#   PARCELLE_GRI
# =============================================================================
    # ITK=pd.read_csv(d["PC_labo"]+"DONNEES_RAW/DATA_PARCELLE_GRIGNON/ITK_maize_rain_Grignon.csv",decimal=",")
    # ITK["TIMESTAMP"]=pd.to_datetime(ITK["TIMESTAMP"],format="%d/%m/%Y")
    # ITK["ITK_Short"]=["Ss",'Ss',"Phy","Phy","Phy","Mn"]
    # NDVI=pd.read_csv(d["PC_labo"]+"TRAITEMENT/INPUT_DATA/NDVI_parcelle/Parcelle_ref/PARCELLE_Grignon/NDVI_Grignon_2019.csv",decimal=".")
    # NDVI.date=pd.to_datetime(NDVI.date,format="%Y-%m-%d")
    # NDVI['real_Rocha']=NDVI.eval("NDVI*1.37-0.017")
    # NDVI['real_Kamble_spe']=NDVI.eval("NDVI*1.457-0.1725")
    # NDVI['real_Kamble_Gene']=NDVI.eval("NDVI*1.358-0.0744")
    # NDVI['real_Neale']=NDVI.eval("NDVI*1.81+0.026")
    # NDVI['real_Calera']=NDVI.eval("NDVI*1.56-0.1")
    # NDVI['real_Demarez']=NDVI.eval("NDVI*0.99+0.08")
    # NDVI['real_Toureiro']=NDVI.eval("NDVI*1.46-0.25")
    # ###### ADD FCOVER_SAMIR
    # NDVI["rela_fcover"]=NDVI.eval("NDVI*1.25-0.13")
    # meteo=pd.read_csv(d["PC_labo"]+"TRAITEMENT/INPUT_DATA/DATA_METEO_BV/PARCELLE_GRI/meteo_gri_2019.csv",decimal=".")
    # SWC=pd.read_csv(d["PC_labo"]+"TRAITEMENT/DATA_VALIDATION/DATA_SWC/SWC_GRI/SWC_GRI_2019.csv")
    # ETR=pd.read_csv(d["PC_labo"]+"/DATA_ETR_CESBIO/DATA_ETR_GRIGNON/DATA_ETR_GRIGNON_ICOS/ETR_GRIGNON2019.csv")
    # Fcover=pd.read_csv(d["PC_labo"]+"TRAITEMENT/INPUT_DATA/FCOVER_parcelle/PARCELLE_Grignon/FCOVER_Grignon_2019.csv")
    # Fcover.date=pd.to_datetime(Fcover.date,format="%Y-%m-%d")
    # Fcover_sigmo=pd.read_csv(d["PC_labo"]+"TRAITEMENT/INPUT_DATA/FCOVER_parcelle/PARCELLE_Grignon/Fcover_sigmo_2019.csv")
    # Fcover_sigmo.date=pd.to_datetime(Fcover_sigmo.date,format="%Y-%m-%d")
    # ETR.date=pd.to_datetime(ETR.date,format="%Y-%m-%d")
    # meteo.date=pd.to_datetime(meteo.date,format="%Y-%m-%d")
    # SWC["date"]=pd.to_datetime(SWC["date"],format="%Y-%m-%d")
    # globals()["date_irr_%s"%y]=meteo.loc[meteo.Irrig > 0.0]
    # plt.figure(figsize=(12,10))
    # for rela in NDVI.columns[2:-1]:
    #     print(rela)
    #     plt.plot(NDVI.date,NDVI[rela],label=rela)
    # plt.ylabel("Kcb")
    # plt.legend()
    # plt.savefig(d["PC_labo"]+"RESULT/PLOT/Analyse_Flux_ICOS/plt_RELA_NDVI_Kcb_Gri_2019.png")
    # fig, ax = plt.subplots(figsize=(12, 10))
    # # sns.set(style="darkgrid")
    # # sns.set_context('paper')
    # ax1 = plt.subplot(211)
    # plt.plot(NDVI.date,NDVI.NDVI,label='NDVI')
    # plt.plot(Fcover.date,Fcover.FCOVER,label="Fcover")
    # # plt.plot(Fcover_sigmo.date,Fcover_sigmo.FCOVER,label="Fcover_sigmo")
    # plt.plot(NDVI.date,NDVI.rela_fcover,label="FCOVER_SAMIR")
    # plt.plot([ITK.TIMESTAMP.loc[ITK.ITK_Short!="Ma"],ITK.TIMESTAMP.loc[ITK.ITK_Short!="Ma"]], [0.9,0.9],marker="o",color='red')
    # for t,i in zip(ITK.TIMESTAMP.loc[ITK.ITK!="Ma"],ITK.ITK_Short.loc[ITK.ITK!="Ma"]):
    #     plt.text(x =t , y= 0.92,s = i,size=9)
    # plt.ylim(0,1)
    # plt.ylabel('NDVI & Fcover')
    # plt.legend()
    # plt.setp(ax1.get_xticklabels(),visible=False)
    # plt.title(y)
    # ax2=plt.subplot(212)
    # plt.plot(meteo.date,meteo.ET0,color='green',label="ET0")
    # plt.plot(ETR.date,ETR.LE,color="black",label='ETR')
    # plt.legend(loc='upper left')
    # plt.ylabel('ETO')
    # plt.ylim(0,10)
    # ax21 = plt.twinx()
    # ax21.grid(axis='y')
    # Irri=meteo.loc[meteo.Irrig >0.0]
    # ax21.plot(Irri.date,Irri.Irrig,color="red",marker="o",linestyle="None",label="Irrigation")
    # ax21.bar(meteo.date,meteo.Prec, color='b')
    # plt.ylim(0,50)
    # plt.title(y)
    # plt.legend()
    # plt.savefig(d["PC_labo"]+"RESULT/PLOT/Analyse_Flux_ICOS/plt_data_Gri_2019.png")
    # # plt.savefig("G:/Yann_THESE/BESOIN_EAU/Calibration_SAMIR/Analyse_data/plt_data_"+str(y)+".png")
    
    # # plot SWC
    # RU75= (0.48-0.25)-((0.48-0.25)*75/100)
    # RU70= (0.48-0.25)-((0.48-0.25)*70/100)
    # plt.figure(figsize=(10,10))
    # sns.set(style="darkgrid")
    # sns.set_context('paper')
    # ax1 = plt.subplot(311)
    # plt.plot(SWC["date"],SWC.SWC_5_moy/100)
    # plt.plot(SWC["date"],np.repeat(0.25,len(SWC["date"])),c="r",label="WP")
    # plt.plot(SWC["date"],np.repeat(0.48,len(SWC["date"])),c="b", label="FC")
    # plt.plot(SWC["date"],np.repeat(0.48-RU75,len(SWC["date"])),c="b",linestyle='--',label='RU 75 %')
    # # plt.plot(SWC2017["Date/Time"],np.repeat(0.363-RU75,len(SWC2017["Date/Time"])),c="b",linestyle='--',label='RU 75 %')
    # plt.ylabel('SWC en surface (5cm)')
    # ax1 = plt.twinx()
    # ax1.grid(axis='y')
    # ax1.bar(meteo.date.iloc[:252],meteo.Prec.iloc[:252],width=1,color="b")
    # ax1.plot(Irri.date,Irri.Irrig,color="red",marker="o",linestyle="None",label="Irrigation")
    # plt.ylim(0,50)
    # ax2 = plt.subplot(312)
    # plt.plot(SWC["date"],SWC.SWC_30_moy)
    # plt.plot(SWC["date"],np.repeat(0.25,len(SWC["date"])),c="r",label="WP")
    # plt.plot(SWC["date"],np.repeat(0.48,len(SWC["date"])),c="b", label="FC")
    # plt.plot(SWC["date"],np.repeat(0.48-RU75,len(SWC["date"])),c="b",linestyle='--',label='RU 75 %')
    # plt.legend()
    # plt.ylabel('SWC en profondeur 30 cm')
    # ax3 = plt.subplot(313)
    # plt.plot(SWC["date"],SWC.SWC_90_moy)
    # plt.plot(SWC["date"],np.repeat(0.25,len(SWC["date"])),c="r",label="WP")
    # plt.plot(SWC["date"],np.repeat(0.48,len(SWC["date"])),c="b", label="FC")
    # plt.plot(SWC["date"],np.repeat(0.48-RU75,len(SWC["date"])),c="b",linestyle='--',label='RU 75 %')
    # plt.legend()
    # plt.ylabel('SWC en profondeur 90 cm')
    # plt.savefig(d["PC_labo"]+"RESULT/PLOT/Analyse_Flux_ICOS/plt_data_SWC_gri_2019.png")
    # plt.savefig("G:/Yann_THESE/BESOIN_EAU/Calibration_SAMIR/Analyse_data/plt_data_SWC_"+str(y)+".png")
    
# Vérification Irrigation data météo Lamothe
    # for y in ["2006","2008","2010","2012","2014","2015","2019"]:
    #     ITK=pd.read_csv("H:/YANN_THESE/bESOIN_EAU/BESOIN_EAU/DONNEES_RAW/PARCELLE_LABO/ITK_LAM/ITK_LAM_"+y+".csv",sep=',')
    #     ITK["TIMESTAMP"]=ITK["TIMESTAMP"].apply(lambda x:x[0:10])
    #     ITK["TIMESTAMP"]=pd.to_datetime(ITK["TIMESTAMP"],format="%d/%m/%Y")
    #     meteo=pd.read_csv(d["PC_disk_Wind"]+"TRAITEMENT/INPUT_DATA/DATA_METEO_BV/PARCELLE_LAM/Meteo_station_"+str(y)+".csv",decimal=".")
    #     meteo=meteo.loc[(meteo.date >= str(y)+"-03-02") &(meteo.date <= str(y)+"-10-31")]
    #     meteo.date=pd.to_datetime(meteo.date,format="%Y-%m-%d")
    #     plt.figure(figsize=(7,7))
    #     plt.title("météo station")
    #     plt.plot(meteo.date,meteo.Prec)
    #     if y != "2019":
    #         plt.plot([ITK.TIMESTAMP.loc[ITK.ITK=="Irr"],ITK.TIMESTAMP.loc[ITK.ITK=="Irr"]], [ITK.dose.loc[ITK.ITK=="Irr"],ITK.dose.loc[ITK.ITK=="Irr"]],marker="o",color='red')
    #         for t,i in zip(ITK.TIMESTAMP.loc[ITK.ITK=="Irr"],ITK.dose.loc[ITK.ITK=="Irr"]):
    #             plt.text(x =t , y=i+0.5 ,s = "Irr",size=9)
    #     plt.savefig("D:/THESE_TMP/RESULT/PLOT/Analyse_Flux_ICOS/plt_pluvio_irrigation_"+str(y)+".png")
    #     meteo_SAFRAN=pd.read_csv(d["PC_disk_Wind"]+"TRAITEMENT/INPUT_DATA/DATA_METEO_BV/PARCELLE_LAM/meteo_lam_"+str(y)+".csv",decimal=".")
    #     meteo_SAFRAN.date=pd.to_datetime(meteo_SAFRAN.date,format="%Y-%m-%d")
    #     meteo_SAFRAN=meteo_SAFRAN.loc[(meteo_SAFRAN.date >= str(y)+"-03-02") &(meteo_SAFRAN.date <= str(y)+"-10-31")]
    #     plt.figure(figsize=(7,7))
    #     plt.title("météo SAFRAN")
    #     plt.plot(meteo_SAFRAN.date,meteo_SAFRAN.Prec)
    #     if y != "2019":
    #         plt.plot([ITK.TIMESTAMP.loc[ITK.ITK=="Irr"],ITK.TIMESTAMP.loc[ITK.ITK=="Irr"]], [ITK.dose.loc[ITK.ITK=="Irr"],ITK.dose.loc[ITK.ITK=="Irr"]],marker="o",color='red')
    #         for t,i in zip(ITK.TIMESTAMP.loc[ITK.ITK=="Irr"],ITK.dose.loc[ITK.ITK=="Irr"]):
    #             plt.text(x =t , y=i+0.5 ,s = "Irr",size=9)
    #     plt.figure(figsize=(7,7))
    #     plt.scatter(meteo.Prec,meteo_SAFRAN.Prec)
    #     plt.xlabel('meteo station')
    #     plt.ylabel("météo SAFRAN")
    #     plt.title(y)
    #     plt.plot([0.0, 35], [0.0,35], 'black', lw=1,linestyle='--')
    #     plt.xlim(0,35)
    #     plt.ylim(0,35)

    # LAI teste 
    # data=[]
    # annots = loadmat(d["PC_disk_home"]+'/TRAITEMENT/INPUT_DATA/LAI_parcelle/PARCELLE_LABO/LAI_sat_CESBIO/Obs_LAM_2015.mat')
    # # data = [[row.flat[0] for row in line] for line in annots['Obs_IS'][0]]
    # for line in annots['Obs'][0]:
    #     for row in line:
    #         # print(row.flat[:])
    #         data.append(row.flat[:])
    # columns = ['Mes', 'Mes_STD' , 'Date', 'Doy','Sat']
    # df_train = pd.DataFrame(data[1],index=columns)
    # df_train=df_train.T
    # print(df_train)
    # df_train["date"]=pd.to_datetime(["2015-06-02","2015-07-01","2015-07-23","2015-08-06","2015-09-02"])
    # df_train.to_csv(d["PC_disk_Wind"]+'/TRAITEMENT/INPUT_DATA/LAI_parcelle/PARCELLE_LABO/LAI_LAM_2015.csv')
    # df=pd.read_csv("D:/THESE_TMP/TRAITEMENT/INPUT_DATA/LAI_parcelle/PARCELLE_LABO/LAI_sat_CESBIO/LAI_LAM_2012.csv",sep=";")
    # df["Date"]=pd.to_datetime(df.Date,format="%d/%m/%Y")
    # df=df.T
    # df=df.iloc[0:2]
    # df.to_csv("D:/THESE_TMP/TRAITEMENT/INPUT_DATA/LAI_parcelle/PARCELLE_LABO/LAI_pre_inter_OTB/LAI_inter_2012.csv",sep=";")
# =============================================================================
#   Récupération Température station 
# =============================================================================
    # df = pd.read_csv("D:/THESE_TMP/DONNEES_RAW/PARCELLE_LABO//FLUX_ETR/eddypro_FR-Lam_biomet_2020-01-28T012345_adv.csv")
    # df=df.iloc[1:]
    # data=df[["date","Ta_1_1_1","P_1_1_1"]]
    # data_flo=data[["Ta_1_1_1","P_1_1_1"]].astype(float)
    # data_flo["date"]=data["date"]
    # data_flo['temp']=data_flo.eval("Ta_1_1_1 - 273.15")
    # data_flo["years"]=data_flo.date.apply(lambda x:x[0:4])
    # data_2019=data_flo[data_flo.years=="2019"]
    # data2019_Ta=data_2019.groupby("date")["temp"].mean()
    # data2019_P=data_2019.groupby("date")["P_1_1_1"].sum()
    # meteo=pd.merge(data2019_Ta,data2019_P*1000,on='date')
    # meteo[meteo.P_1_1_1 < 0.000]=np.nan
    # meteo[meteo.temp < -10]=np.nan
    # meteo.to_csv(d["PC_disk_home"]+"TRAITEMENT/INPUT_DATA/DATA_METEO_BV/PARCELLE_LAM/DATA_PREC_TEMP/meteo_prec_temp_2019.csv")
    
    # df["date_time"]=df["TEMPS"].apply(lambda x:x[0:10])
    # df["years"]=df["TEMPS"].apply(lambda x:x[6:10])
    # df["date_time"]=pd.to_datetime(df["date_time"],format='%d/%m/%Y')
    # data=df[['date_time','P','Ta',"Rn","years"]]
    # for y in ["2006","2008","2010","2012","2014","2015"]:
    #     data1=data[data.years==y]
    #     Ta=data1.groupby("date_time")["Ta"].mean()
    #     Prec=data1.groupby("date_time")['P'].sum()
    #     meteo=pd.merge(Ta,Prec,on='date_time')
    #     meteo.to_csv(d["PC_disk_home"]+"TRAITEMENT/INPUT_DATA/DATA_METEO_BV/PARCELLE_LAM/DATA_PREC_TEMP/meteo_prec_temp_"+y+".csv")
    
# =============================================================================
#     Analyse Météo
# ============================================================================
    # meteo_SAF=pd.read_csv(d["PC_disk_home"]+"/TRAITEMENT/INPUT_DATA/DATA_METEO_BV/PARCELLE_LAM/meteo_lam_2014.csv",decimal=".")
    # meteo_temp=pd.read_csv(d["PC_disk_home"]+"TRAITEMENT/INPUT_DATA/DATA_METEO_BV/PARCELLE_LAM/DATA_PREC_TEMP/meteo_prec_temp_2010.csv")
    # meteo_SAF.date=pd.to_datetime(meteo_SAF.date,format="%Y-%m-%d")
    # meteo_temp.date=pd.to_datetime(meteo_temp.date_time,format="%Y-%m-%d")
    # plt.scatter(meteo_SAF.Prec,meteo_temp.P)


    # labels = meteo_SAF.date[120:300]
    # SAFRAN = meteo_SAF.Prec[120:300]
    # Station = meteo_temp.P[120:300]
    # width = 0.5  # the width of the bars
    
    # plt.subplots(figsize=(10,7))
    # plt.bar(labels, SAFRAN, width, label='SAFRAN',color="red")
    # plt.bar(labels, Station, width, bottom=SAFRAN, label='Station',color="blue")
    # plt.xticks(rotation=45)
    # plt.legend()
    # r=SAFRAN.sum()
    # d=Station.sum()
    # plt.text(labels.iloc[1],max(Station+SAFRAN),s="SAFRAN: %s mm"%(round(r,2)))
    # plt.text(labels.iloc[1],max(Station+SAFRAN)-5,s="Station: %s mm"%(round(d,2)))
    # plt.savefig("H:/Yann_THESE/BESOIN_EAU/BESOIN_EAU/Calibration_SAMIR/Analyse_data/plt_data_pluvio_2010.png")
