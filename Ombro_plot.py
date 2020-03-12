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
        plt.ylabel("Température en °C")
        plt.xticks(size='large')
        plt.yticks(size='large')
        plt.title(i)
        plt.savefig("/datalocal/vboxshare/THESE/CLASSIFICATION/RESULT/PLOT/DIAGRAMME_OMBRO_SO/DIAG_OMBRO%s_2018.png"%(i))

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
