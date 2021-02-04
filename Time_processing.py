#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 09:22:18 2019

@author: pageot
"""


import os
import sqlite3
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns; sns.set()
from STAT_ZONAL_SPECRTE import *
from scipy import stats
from TEST_ANALYSE_SIGNATURE import *
import PLOT_CLASSIF_BV
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
if __name__ == '__main__':
    df_ALL=pd.DataFrame()
    years="2017"
    BV="ADOUR"
    for i in os.listdir("G:/THESE/CLASSIFICATION/data_time_processing_not_cumul/"+str(years)):
        if "_full" in i:
            print(i)
            df=pd.read_csv("G:/THESE/CLASSIFICATION/data_time_processing_not_cumul/"+str(years)+"/"+str(i),header=None)
            df.columns=["step","RAM","TIME","cpuask","cpu_used","model"]
            df["config"]=np.repeat(i,df.shape[0])
            globals()["df%s"%i]=df
            df_ALL=df_ALL.append(df)
    dfADOUR=pd.DataFrame()
    for bv in [BV]:
        if bv == "ADOUR":
            globals()["df%s"%(bv)]= df_ALL[df_ALL.model=="4"]
        else:
           globals()["df%s"%(bv)]=df_ALL[df_ALL.model=="3"]
           
        Time_proc=pd.DataFrame(globals()["df%s"%(bv)].TIME*globals()["df%s"%(bv)].cpu_used)
        step=pd.Series(globals()["df%s"%(bv)].step)
        config=pd.Series(globals()["df%s"%(bv)].config)
        Time_proc["step"]=step
        Time_proc["config"]=config
        Time_proc[Time_proc.step=="classification"]
        Time_proc[Time_proc.step=="learnModel"]
        step1="learnModel"
        
        
#        Selction des élements dans un dataframe en fonction de condition. 
        # dfADOUR.loc[dfADOUR["config"]=="SAISON_S1_ASC_S2",["RAM"]]
        # dfADOUR.loc[dfADOUR['config'].str.endswith("S2") & (dfADOUR['step'] == step1)].mean()
        # Time_proc.loc[Time_proc['config'].str.endswith("S2") & (Time_proc["step"] ==step1)].mean()
        
        # Time_proc.loc[Time_proc['config'].isin(["3ind_fixe_seed"]) & (Time_proc["step"] =="classification")].mean()
        # dfADOUR.loc[dfADOUR['config'].isin(["3ind_fixe_seed"]) & (dfADOUR['step'] == "classification")].mean()
        test=dfADOUR.groupby(['step','config']).mean()
        teststd=dfADOUR.groupby(['step','config']).std()
        test.RAM.classification
# =============================================================================
# Plot RUN vs time and RAM vs performence
# =============================================================================
    test=dfADOUR.groupby(['step','config']).mean()
    teststd=dfADOUR.groupby(['step','config']).std()
    Time2017=Time_proc[Time_proc.step=="classification"].groupby("config").mean()/60
    stdtime2017=Time_proc[Time_proc.step=="classification"].groupby("config").std()/60
    Time2017_lear=Time_proc[Time_proc.step=="learnModel"].groupby("config").mean()/60
    stdtime2017_lear=Time_proc[Time_proc.step=="learnModel"].groupby("config").std()/60
    
    axes = plt.gca()
    for i in np.arange(0,10):
        y=test.RAM.classification.iloc[i]
        x=Time2017[0].iloc[i]
        stdx=stdtime2017.iloc[i]
        stdy=teststd.RAM.classification.iloc[i]
        plt.scatter(x,y,label=test.RAM.classification.index[i][:-5])
        # a=Ellipse((x,y),width=stdx,height=stdy,alpha=0.2,color="blue",zorder=5)
        # axes.add_artist(a)
        plt.legend()
        plt.xlabel("Time CPU in hours")
        plt.ylabel("Allocated RAM in Gb")
    # plt.savefig("/datalocal/vboxshare/THESE/CLASSIFICATION/TIME_PROCESSING/RAMvsCPU_stepclassif2017.png")
    
   
# =============================================================================
# Plot time/Fscore MAis 
# =============================================================================
    # F_scoreMIrr=dfMetric[dfMetric.index==0]
    # F_scoreMNIrr=dfMetric[dfMetric.index==2]
    # # MNrr2017=F_scoreMNIrr.loc[F_scoreMNIrr["step"]]
    # # M2017=F_scoreMIrr.loc[F_scoreMIrr["step"].str.endswith("2017")]
    # # M2017.set_index("step",inplace=True)
    # Time2017=Time_proc[Time_proc.step=="classification"].groupby("config").mean()/60
    # # Time2017_lea=Time_proc[Time_proc.step==step1].groupby("config").mean()/60
    # stdtime2017=Time_proc[Time_proc.step=="classification"].groupby("config").std()/60
    
    # axes = plt.gca()
    # for i in Time_proc.config:
    #     x=Time2017.loc[Time2017.index==i]
    #     y=F_scoreMIrr.loc[F_scoreMIrr.step==i[:-5]].mean_fscore
    #     # stdy=M2017.std_fscore [i]
    #     # stdx=stdtime2017.iloc[i]
    #     plt.scatter(x,y,color="blue")
    #     a=Ellipse((x,y),width=stdx,height=stdy,alpha=0.2,color="blue",zorder=5)
    #     axes.add_artist(a)
    #     plt.text(x,y+0.005,step,fontsize=9)
    # plt.ylim(0.7,0.9)
    # plt.xlim(7500,48000)
    # plt.xlabel("time processing by cpu in min")
    # plt.ylabel("F-score Mais irrigated in 2017")
#     # plt.savefig("/datalocal/vboxshare/THESE/CLASSIFICATION/TIME_PROCESSING/plot_maisirri2017.png")