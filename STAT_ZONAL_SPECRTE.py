#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 10:38:21 2019

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
from pylab import *

def pltemp(x):
    plt.figure(figsize=(15,10))
    sns.set(style="darkgrid")
    sns.set_context('paper')
    plt.plot(x,linestyle="-")
    plt.xticks(rotation=90)
    
def pltemplui(y,x1,x2):
    plt.figure(figsize=(20,20))
    sns.set(style="darkgrid")
    sns.set_context('paper')
    plt.bar(y,x1,color="blue",width=1)
    ax2 = plt.twinx()
    ax2.plot(y,x2,linewidth=2,color='r')
    
def pltSAR4(x,y1,y2,y3,y4):
    plt.figure(figsize=(15,15))
    ax1 = plt.subplot(411)
    plt.plot(y1.T.mean(),marker="o")
    plt.errorbar(x,y1.interpolate().T.mean(),yerr=y1.interpolate().T.std())
    plt.ylabel("db ascVV_VH")
    plt.title("CODE_CULTURE_"+str(i))
    plt.setp(ax1.get_xticklabels(), visible=False)
    ax2 = plt.subplot(412)
    plt.plot(y2.T.mean(),marker="o")
    plt.errorbar(x,y2.interpolate().T.mean(),yerr=y2.interpolate().T.std())
    plt.ylabel("db ascVV")
    plt.setp(ax2.get_xticklabels(), visible=False)
    ax3 = plt.subplot(413)
    plt.plot(y3.T.mean(),marker="o")
    plt.errorbar(x, y3.interpolate().T.mean(),yerr=y3.interpolate().T.std())
    plt.ylabel("db ascVH")
    plt.setp(ax3.get_xticklabels(), visible=False)    
    ax4 = plt.subplot(414)
    plt.bar(x, y4,width=1)
    plt.ylabel("Precipitation en mm")

    
def SAR_process_db(list_lab,data,variable_resarch): # Attention variable_resarch
     for i in list_lab:
        print(i)
        globals()['cropslab%s' % i] = pd.DataFrame(data[data.labcroirr==i]).T
        globals()['%s%s' % (variable_resarch,i)]=[]
        for index,row in globals()['cropslab%s' % i].iterrows():
            if variable_resarch in index:
                globals()['%s%s' % (variable_resarch,i)].append (row)
                globals()['df%s%s' %(variable_resarch,i)]=pd.DataFrame(globals()['%s%s' % (variable_resarch,i)])
                globals()["df%s%s"% (variable_resarch,i)].replace(to_replace =0 , value= pd.NaT,inplace=True)
                globals()["dbdf%s%s" %(variable_resarch ,i)]=10*np.log10(globals()['df%s%s' % (variable_resarch,i)])
     return  globals()["dbdf%s%s" %(variable_resarch ,i)]


def sqlite_df(path,x): # conversion BD sqlite en DataFrame pandas 
    sql=sqlite3.connect(path)
    df=pd.read_sql_query("SELECT * FROM output", sql)
    globals()["%s"%x]=df.groupby("originfid").mean()
    return globals()["%s"%x]
    

def Optique_Process(list_lab,data,variable_resarch):
    for i in list_lab:
        print(i)
        globals()['cropslab%s' % i] = pd.DataFrame(data[data.labcroirr==i]).T
        globals()['%s%s' %(variable_resarch,i)]=[]
        for index,row in globals()['cropslab%s' % i].iterrows():
            if variable_resarch in index:
                globals()['%s%s' % (variable_resarch,i)].append (row)
                globals()['df%s%s' %(variable_resarch,i)]=pd.DataFrame(globals()['%s%s' %(variable_resarch,i)])
        return globals()['df%s%s' %(variable_resarch,i)],globals()['%s%s' %(variable_resarch,i)]

if __name__ == "__main__":
    # =============================================================================
    # Gestion des datas SAFRAN 
    # =============================================================================
    # df=pd.read_csv("/datalocal/vboxshare/THESE/CLASSIFICATION/DONNES_SIG/DONNEES_METEO/SIM2_2018_201901.csv",sep=";")
    # df=pd.read_csv("D:/THESE_TMP/DONNEES_RAW/DONNES_METEO/SIM2_2010_201811.csv",sep=";")
    # df=pd.read_csv('G:/THESE/CLASSIFICATION/DONNES_SIG/DONNEES_METEO/SIM2_2000_2009.csv',sep=";")
    # LAMBX=df.LAMBX*100
    # LAMBY=df.LAMBY*100
    # df["lambX"]=LAMBX
    # df['lambY']=LAMBY
    # df=df.loc[(df.DATE >= 20090101) &(df.DATE <= 20091231)]
    # df.to_csv("D:/THESE_TMP/DONNEES_RAW/DONNES_METEO/SAFRAN2009_L2.csv")
    
    # df=pd.read_csv("/datalocal/vboxshare/THESE/CLASSIFICATION/DONNES_SIG/DONNEES_METEO/SAFRAN_TCJ.csv")
    # df1=df.drop([0])
    # preliq=df1[["DATE","PRELIQ_Q","X","Y"]]
    
    # preliq2=preliq.sort_values(by="DATE")
    # meandate=preliq2.groupby("DATE").mean()


    # =============================================================================
    # Profil TST 
    # =============================================================================
    a=set(meandate.index).intersection(set(TimeTSTNIRR.index))
    
    timeTST=pd.DataFrame()
    Years=[]
    for i in os.listdir("/datalocal/vboxshare/THESE/CLASSIFICATION/TRAITEMENT/STAT_POLY/STAT_TST/"):
        print (i)
        tile=i[35:41]
        print (tile)
        date=i[42:50]
        print (date)
        sqlite_df("/datalocal/vboxshare/THESE/CLASSIFICATION/TRAITEMENT/STAT_POLY/STAT_TST/" +i,"dfTST")
        lab=dfTST.labcroirr.astype(int)
        globals()["meanTST%s"%(date)]=round(dfTST.value_0/100-273.15,2)
        globals()["meanTST%s"%date].rename("TST_"+date,inplace=True)
        timeTST=timeTST.append(globals()["meanTST%s"%date])
        Years.append(date)
    timeTST.sort_index(inplace=True)
    timeTST[timeTST<= -1]=pd.NaT
    timeTST["date"]=sorted(Years)
    timeTST.date=pd.to_datetime(timeTST.date,format="%Y%m%d")
    TimeTST=timeTST.groupby(timeTST.date).mean()
    plt.figure(figsize=(10,10))
    plt.subplot(211)
    plt.plot(meandate.loc[a].index,TimeTST.loc[a],marker='o')
    plt.subplot(212)
    plt.bar(meandate.loc[a].index,meandate.loc[a].PRELIQ_Q,width=5)

    timeTSTNIRR=pd.DataFrame()
    Years=[]
    for i in os.listdir("/datalocal/vboxshare/THESE/CLASSIFICATION/TRAITEMENT/STAT_POLY/STAT_TST/STAT_NIRR/"):
        print (i)
        tile=i[35:41]
        print (tile)
        date=i[42:50]
        print (date)
        sqlite_df("/datalocal/vboxshare/THESE/CLASSIFICATION/TRAITEMENT/STAT_POLY/STAT_TST/STAT_NIRR/" +i,"dfTST")
        lab=dfTST.labcroirr.astype(int)
        globals()["meanTST%s"%(date)]=round(dfTST.value_0/100-273.15,2)
        globals()["meanTST%s"%date].rename("TST_"+date,inplace=True)
        timeTSTNIRR=timeTSTNIRR.append(globals()["meanTST%s"%date])
        Years.append(date)
    timeTSTNIRR.sort_index(inplace=True)
    timeTSTNIRR[timeTSTNIRR<= -1]=pd.NaT
    timeTSTNIRR["date"]=sorted(Years)
    timeTSTNIRR.date=pd.to_datetime(timeTSTNIRR.date,format="%Y%m%d")
    TimeTSTNIRR=timeTSTNIRR.groupby(timeTSTNIRR.date).mean()
    plt.figure(figsize=(10,10))
    plt.subplot(211)
    plt.plot(meandate.loc[a].index,TimeTSTNIRR.loc[a],marker='o')
    plt.subplot(212)
    plt.bar(meandate.loc[a].index,meandate.loc[a].PRELIQ_Q,width=5)

