#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 16:39:14 2019

@author: pageot
"""

import os
import sqlite3
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
import seaborn as sns
import csv
from scipy import *
from scipy import stats
from pylab import *
import STAT_ZONAL_SPECRTE
import TEST_ANALYSE_SIGNATURE
import datetime
from sklearn.metrics import *
from sklearn.linear_model import LinearRegression*

def predict(x):
  return slope * x + intercept

if __name__ == '__main__':
    df=pd.read_csv("/datalocal/vboxshare/THESE/CLASSIFICATION/TRAITEMENT/tmp/CARTE_leve/PT_MES_PARCELLES_LEVER.csv")
    df=df.groupby("originfid").mean()
    xy=df.iloc[:,0:2]
    xy.to_csv("/datalocal/vboxshare/THESE/CLASSIFICATION/TRAITEMENT/tmp/CARTE_leve/XY_MES_PARCELLES_TARN.csv")
    drop_band=['X', 'Y', 'region', 'labcroirr',"alt_band_0"]
    col_sqlite('/datalocal/vboxshare/THESE/CLASSIFICATION/TRAITEMENT/tmp/CARTE_leve/PT_MES_PARCELLES_LEVER.csv','df_lever',drop_band,"/datalocal/vboxshare/THESE/CLASSIFICATION/RESULT/list_features_SAR.txt")

    df=df_lever.iloc[:-1,:-1]/1000
    df1=df.T
    df1["band_name"]= df1.index
    df1["indice"] = df1.band_name.apply(lambda s: s[10:14])
    df1["date"] = df1.band_name.apply(lambda s: s[15:])
    df_indice = df1[df1['indice'] == "NDVI"]
    df_indice=df_indice.T
    df_indice.to_csv("/datalocal/vboxshare/THESE/CLASSIFICATION/TRAITEMENT/tmp/CARTE_leve/data_pheno_MES_PARCELLES_TARN.csv")    

# =============================================================================
# Comparaison phenotb_mes_parcelles 
# =============================================================================
    df_otb=pd.read_csv("/datalocal/vboxshare/THESE/CLASSIFICATION/TRAITEMENT/CARTE_LEVER/ZONE_TARN/data_scatter_MES_PARCELLES.csv")
    data=df_otb.iloc[:,[4,8,19]]
    data.datedebut  
    date=pd.to_datetime(data.datedebut)
    list_date_DOY=[]
    for i in date:
        a=i.strftime("%j")
        list_date_DOY.append(a)
    list_date_DOY=list(map(int,list_date_DOY))
    data["date_vaide"]=list_date_DOY
    data1=data[data["pheno_t0"] > 0]
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(data1.date_vaide,data1.pheno_t0)

    fitLine = predict(data1.date_vaide)
    bias=1/data1.shape[0]*sum(mean(data1.pheno_t0) - data1.date_vaide)
    
    plt.figure(figsize=(10,10))
    sns.set(style="darkgrid")
    sns.set_context('paper')    
    plt.scatter(data1.date_vaide,data1.pheno_t0)
    plt.plot([0.0, 360], [0.0, 360], 'r-', lw=2)
#    plt.plot(data1.date_vaide, fitLine, c='black',linestyle=":")
    plt.xlim(0,360)
    plt.ylim(0,360)
    plt.xlabel("DOY mes parcelles")
    plt.ylabel("DOY PhenOTB")
    rms = sqrt(mean_squared_error(data1.date_vaide,data1.pheno_t0))
    plt.text(1,350,"RMSE = "+str(round(rms,2)))
    plt.text(1,340,"R² = "+str(round(r_value,2)))
    plt.text(1,330,"Pente = "+str(round(slope,2)))
    plt.text(1,320,"Biais = "+str(round(bias,2)))
    plt.savefig("/datalocal/vboxshare/THESE/CLASSIFICATION/RESULT/PLOT/Validation_pheno_otb.png")

