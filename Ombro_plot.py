#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 09:30:43 2019

@author: pageot
"""

import pandas as pd
import os
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
import seaborn as sns
from scipy import *
from pylab import *
import STAT_ZONAL_SPECRTE as plot



if __name__ == "__main__":
    
    DF_OMBRO=pd.read_csv("/datalocal/vboxshare/THESE/CLASSIFICATION/TRAITEMENT/DATA_METEO/DATA_OMBRO.csv")
    DF_OMBRO.set_index("JJ",inplace=True)
    DF_OMBRO.index=pd.to_datetime(DF_OMBRO.index,format="%Y%m%d")
    
    # DF_OMBRO=pd.read_csv("F:/THESE/CLASSIFICATION/TRAITEMENT/DATA_METEO/SAFRAN_ADOUR.csv")
    DF_OMBRO.drop(columns=['X', 'Y', 'field_1', 'LAMBX', 'LAMBY', 'PRENEI_Q',
        'FF_Q', 'Q_Q', 'DLI_Q', 'SSI_Q', 'HU_Q', 'EVAP_Q', 'ETP_Q',
       'PE_Q', 'SWI_Q', 'DRAINC_Q', 'RUNC_Q', 'RESR_NEIGE', 'RESR_NEI_1',
       'HTEURNEIGE', 'HTEURNEI_1', 'HTEURNEI_2', 'SNOW_FRAC_', 'ECOULEMENT',
       'WG_RACINE_', 'WGI_RACINE', 'TINF_H_Q', 'TSUP_H_Q', 'X.1', 'Y.1'],inplace=True)
    DF_OMBRO.set_index("DATE",inplace=True)
    DF_OMBRO.index=pd.to_datetime(DF_OMBRO.index,format="%Y%m%d")
    DF_OMBRO.sort_index(ascending =True,inplace=True)
    OMBRO_mean=DF_OMBRO.groupby(DF_OMBRO.index).mean()
    OMBRO_std=DF_OMBRO.groupby(DF_OMBRO.index).std()
    ombro=OMBRO_mean.resample("M").agg({'T_Q': 'mean','PRELIQ_Q':'sum'})
    ombro_std=OMBRO_std.resample("M").agg({'T_Q': 'mean','PRELIQ_Q':'sum'})
    plt.figure(figsize=(10,10))
    sns.set(style="darkgrid")
    sns.set_context('paper')
    plt.bar(ombro.index[:12]-1,ombro.PRELIQ_Q[:12]-1,color="blue",width=1,yerr=ombro_std.PRELIQ_Q[:12]-1)
    plt.ylim(-10,200)
    plt.ylabel("Rainfall in mm")
    plt.xticks(size='large')
    plt.yticks(size='large')
    ax2 = plt.twinx()
    ax2.grid(axis='y')
    ax2.plot(ombro.index[:12]-1,ombro.T_Q[:12]-1,linewidth=5,color='r')
    ax2.fill_between(ombro.index[:12]-1,ombro_std.T_Q[:12]-1+ombro.T_Q[:12]-1,ombro.T_Q[:12]-1-ombro_std.T_Q[:12]-1, facecolor='red', alpha=0.2)
    plt.ylim(-5,100)
    plt.ylabel("Temperature in °C")
    plt.xticks(size='large')
    plt.yticks(size='large')
    # plt.savefig("F:/THESE/REDACTION/figure_paper/ombro_2017.png")
#    plt.title(i)
    
    plt.figure(figsize=(10,10))
    sns.set(style="darkgrid")
    sns.set_context('paper')
    plt.bar(ombro.plui.index[12:]-1,ombro.plui[12:]-1,color="blue",width=20,yerr=ombro_std.plui[12:]-1)
    plt.ylim(-10,200)
    plt.ylabel("Rainfall in mm")
    plt.xticks(size='large')
    plt.yticks(size='large')
    ax2 = plt.twinx()
    ax2.grid(axis='y')
    ax2.plot(ombro.index[12:]-1,ombro.T_Q[12:]-1,linewidth=5,color='r')
    ax2.fill_between(ombro.index[12:]-1,ombro_std.T_Q[12:]-1+ombro.t.T_Q[12:]-1,ombro.T_Q[12:]-ombro_std.T_Q[12:]-1, facecolor='red', alpha=0.2)
    plt.ylim(-5,100)
    plt.ylabel("Temperature in °C")
    plt.xticks(size='large')
    plt.yticks(size='large')
    # plt.savefig("F:/THESE/REDACTION/figure_paper/ombro_2018.png")
    
    
    list_name=list(set(DF_OMBRO.Nom))
    for i in list_name:
        globals()["OMBRO%s"% (i)]=DF_OMBRO[DF_OMBRO.Nom==i].resample("M").agg({'t': 'mean',"plui" : 'sum'})
        plt.figure(figsize=(10,10))
        sns.set(style="darkgrid")
        sns.set_context('paper')
        plt.bar(globals()["OMBRO%s"% (i)].index[12:24]-1,globals()["OMBRO%s"% (i)].plui[12:24],color="blue",width=20)
        plt.ylim(-10,100)
        plt.ylabel("Précipitation en mm")
        plt.xticks(size='large')
        plt.yticks(size='large')
        ax2 = plt.twinx()
        ax2.plot(globals()["OMBRO%s"% (i)].index[12:24]-1,globals()["OMBRO%s"% (i)].t[12:24],linewidth=5,color='r')
        plt.ylim(-5,50)
        plt.ylabel("Temperature en °C")
        plt.xticks(size='large')
        plt.yticks(size='large')
        plt.title(i)
        # plt.savefig("/datalocal/vboxshare/THESE/CLASSIFICATION/RESULT/PLOT/DIAGRAMME_OMBRO_SO/DIAG_OMBRO%s_2018.png"%(i))

# =============================================================================
# english version
# =============================================================================
    list_name=list(set(DF_OMBRO.Nom))
    for i in list_name:
        globals()["OMBRO%s"% (i)]=DF_OMBRO[DF_OMBRO.Nom==i].resample("M").agg({'t': 'mean',"plui" : 'sum'})
        plt.figure(figsize=(10,10))
        sns.set(style="darkgrid")
        sns.set_context('paper')
        plt.bar(globals()["OMBRO%s"% (i)].index[12:24]-1,globals()["OMBRO%s"% (i)].plui[12:24],color="blue",width=20)
        plt.ylim(-10,100)
        plt.ylabel("Rainfall in mm")
        plt.xticks(size='large')
        plt.yticks(size='large')
        ax2 = plt.twinx()
        ax2.plot(globals()["OMBRO%s"% (i)].index[12:24]-1,globals()["OMBRO%s"% (i)].t[12:24],linewidth=5,color='r')
        plt.ylim(-5,50)
        plt.ylabel("Temperature in °C")
        plt.xticks(size='large')
        plt.yticks(size='large')
#        plt.title(i)
        plt.savefig("/datalocal/vboxshare/THESE/CLASSIFICATION/RESULT/PLOT/DIAGRAMME_OMBRO_SO/DIAG_EnglishV_OMBRO%s_2017.png"%(i))
        
        plt.figure(figsize=(10,10))
        plt.bar(x[0:12],Prec[0:12],color="blue")
        plt.ylim(-10,150)
        plt.ylabel("Précipitation in mm")
        plt.xticks(size='large',rotation=45)
        plt.yticks(size='large')
        ax2 = plt.twinx()
        ax2.plot(x[0:12],tem[0:12],linewidth=5,color='r')
        plt.ylim(-5,75)
        plt.ylabel("Temperature in °C")
        plt.xticks(size='large')
        plt.yticks(size='large')
        plt.savefig("/datalocal/vboxshare/THESE/REDACTION/Manuscrit/Figure_python/DIAG_BV_OMBRO_2017.png")
# =============================================================================
#  Plot OMBRO echelle BV 
# =============================================================================
    for i in os.listdir("/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/INPUT_DATA/DATA_METEO_BV/"):
        df=pd.read_csv("/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/INPUT_DATA/DATA_METEO_BV/"+str(i))
        df.drop(columns=['X', 'Y', 'field_1', 'LAMBX', 'LAMBY','PRENEI_Q','FF_Q', 'Q_Q', 'DLI_Q', 'SSI_Q', 'HU_Q', 'EVAP_Q',
       'ETP_Q', 'PE_Q', 'SWI_Q', 'DRAINC_Q', 'RUNC_Q', 'RESR_NEIGE',
       'RESR_NEI_1', 'HTEURNEIGE', 'HTEURNEI_1', 'HTEURNEI_2', 'SNOW_FRAC_',
       'ECOULEMENT', 'WG_RACINE_', 'WGI_RACINE', 'TINF_H_Q', 'TSUP_H_Q', 'X.1',
       'Y.1'],inplace=True)
        df.set_index("DATE",inplace=True)
        df.index=pd.to_datetime(df.index,format="%Y%m%d")
        df.sort_index(ascending =True,inplace=True)
        df=df.groupby('DATE').mean()
        print (i)
        print(df.T_Q.max())
        print(df.T_Q.mean()) # moyen all years
        print(df.loc[(df.index >= str(i[7:11])+"-04-01") &(df.index <= str(i[7:11])+"-10-31")].mean())
        meteo=df.resample("M").agg({'T_Q': 'mean',"PRELIQ_Q" : 'sum'})
        meteo["date"]=meteo.index
        meteo["date_plot"]=meteo.date.dt.strftime('%Y-%m')
        print (meteo)
        plt.figure(figsize=(10,10))
        sns.set(style="whitegrid")
        sns.set_context('paper')
        plt.bar(meteo.date_plot,meteo.PRELIQ_Q,color="blue")
        plt.gca().yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(8))
        plt.gca().xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(8))
        plt.ylim(-10,200)
        plt.ylabel("Rainfall in mm")
        plt.xticks(size='large')
        plt.yticks(size='large')
        ax2 = plt.twinx()
        ax2.yaxis.grid(False) 
        ax2.plot(meteo.date_plot,meteo.T_Q,linewidth=5,color='r')
        ax2.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(8))
        ax2.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(8))
        plt.ylim(-5,100)
        plt.ylabel("Temperature in °C")
        plt.xticks(size='large')
        plt.yticks(size='large')
        plt.title("Bassin Versant : %s , Année : %s "%(i[19:-4],i[7:11]))
        plt.savefig("/datalocal/vboxshare/THESE/REDACTION/Manuscrit/Figure_python/DIAG_BV_OMBRO_%s_%s.png"%(i[19:-4],i[7:11]))
