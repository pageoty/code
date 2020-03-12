#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 09:32:58 2019

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
import STAT_ZONAL_SPECRTE

if __name__ == "__main__":

    # =============================================================================
    # SPECTRE SM
    # =============================================================================
   
    df=pd.read_csv("/datalocal/vboxshare/THESE/CLASSIFICATION/DONNES_SIG/DONNEES_METEO/SAFRAN_TCJ.csv")
    df1=df.drop([0])
    preliq=df1[["DATE","PRELIQ_Q","X","Y"]]
    
    preliq2=preliq.sort_values(by="DATE")
    meandate=preliq2.groupby("DATE").mean()
    
    
    timeSM=pd.DataFrame()
    for i in os.listdir("/datalocal/vboxshare/THESE/CLASSIFICATION/TRAITEMENT/STAT_POLY/STAT_SM_DT/"):
        print (i)
        date=i[3:11]
        sqlite_df("/datalocal/vboxshare/THESE/CLASSIFICATION/TRAITEMENT/STAT_POLY/STAT_SM_DT/"+i,"dfSM")
        lab=dfSM.labcroirr.astype(int)
        globals()["meansm%s"%date]=dfSM.value_0
        globals()["meansm%s"%date].rename(date,inplace=True)
        timeSM=timeSM.append(globals()["meansm%s"%date])
        
    timeSM=timeSM.sort_index(ascending=True)
    timesm=timeSM.T/5
    timesm["label"]=lab
    label=list(set(lab))
    for j in label:
        print(j)
        globals()['SMcropslab%s' % j] = pd.DataFrame(timesm[timesm.label==j]).T
        plt.figure(figsize=(15,10))
        plt.errorbar(globals()['SMcropslab%s' % j].index,globals()['SMcropslab%s' % j].T.mean(),yerr=globals()['SMcropslab%s' % j].T.std())
        plt.title("CODE_CULTURE"+"_"+str(j))
        plt.ylabel("soil moisture en mv")
        plt.xticks(rotation=90)
        plt.savefig("/datalocal/vboxshare/THESE/CLASSIFICATION/RESULT/PLOT/PLOT_SM/PLOT_TEMPO_SM_%s_mean.png"%(j))
            
    # =============================================================================
    # PLOT_SM_PLUVIO
    # =============================================================================
    

    meandate.index=pd.to_datetime(meandate.index,format="%Y%m%d")
    timeSM=timeSM.sort_index()
    label=list(set(lab))
    for j in label[:-1]:
        print(j)
        globals()['SMcropslab%s' % j] = pd.DataFrame(timesm[timesm.label==j]).T
        globals()["_%s"% (j)],globals()["b_sup%s"% (j)]=stats.t.interval(0.95,globals()['SMcropslab%s' % j][:-1].shape[1]-1,loc=globals()['SMcropslab%s' % j][:-1].T.mean(),scale=stats.sem(globals()['SMcropslab%s' % j][:-1].T))
        plt.figure(figsize=(15,10))
        ax1 = plt.subplot(411)
        globals()['SMcropslab%s' % j][:-1].index=pd.to_datetime(globals()['SMcropslab%s' % j][:-1].index,format="%Y%m%d")
        plt.plot(globals()['SMcropslab%s' % j][:-1].index, globals()['SMcropslab%s' % j][:-1].T.mean(), color='blue')
        plt.fill_between(globals()['SMcropslab%s' % j][:-1].index,globals()["b_sup%s"% (j)] , globals()["_%s"% (j)], facecolor='blue', alpha=0.2)
#        plt.errorbar(globals()['SMcropslab%s' % j][:-1].index,globals()['SMcropslab%s' % j][:-1].T.mean(),yerr=globals()['SMcropslab%s' % j][:-1].T.std())
        plt.title("CODE_CULTURE"+"_"+str(j))
        plt.ylabel("soil moisture en mv vol %")
        plt.setp(ax1.get_xticklabels(), visible=False)
        # share x only
        ax2 = plt.subplot(412)
        plt.bar(meandate.index,meandate.PRELIQ_Q,width=1)
        plt.ylabel("Précipitation en mm")
        plt.savefig("/datalocal/vboxshare/THESE/CLASSIFICATION/RESULT/PLOT/PLOT_SM/plot_SM_%s_SAFRAN_mean_confiance.png"%(j))