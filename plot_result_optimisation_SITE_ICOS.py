#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 10:42:43 2020

@author: pageot
"""


import os
import sqlite3
import geopandas as geo
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
import csv
import numpy as np
import pandas as pd
import seaborn as sns
import TEST_ANALYSE_SIGNATURE
import shapely.geometry as geom
import descartes
import pickle
from datetime import datetime, date, time, timezone
from scipy import *
from scipy import stats
from pylab import *
from sklearn.metrics import *
from sklearn.linear_model import LinearRegression


def predict(x):
   return slope * x + intercept


if __name__ == '__main__':
    name_run="RUNS_SAMIR/RUN_MULTI_SITE_ICOS/RUN_OPTIMISATION_ICOS/OPTI_ICOS_MULTI_SITE_soil_nu/"
    y='2006'
    d={}
    d['SAMIR_run']="/mnt/d/THESE_TMP/RUNS_SAMIR/"+name_run+"/"+str(y)+"/"
    d['SAMIR_run_Wind']="D:/THESE_TMP/RUNS_SAMIR/"+name_run+"/"+str(y)+"/"
    d["PC_disk_Wind"]="D:/THESE_TMP/RUNS_SAMIR/DATA_Validation/"
    d['PC_disk_unix']="/mnt/d/THESE_TMP/RUNS_SAMIR/"
    d["PC_labo"]="/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/"+name_run+"/"+str(y)+"/"
    
    for val in ["REW","maxZr"]:
        for RU in ["1","5","0"]:
            plt.figure(figsize=(10,10))
            for file in os.listdir(d["PC_labo"][:-5]):
                if 'V1' not in file and "Fcover_sta" in file and "sta_"+RU in file and "SAF" in file:
                    rela=file[30:]
                    df=pd.read_csv(d["PC_labo"][:-5]+"/"+file+"/param_RMSE%s.csv"%val)
                    class_OS=df.groupby("OS")
                    for Os in list(set(df.OS)):
                        data=class_OS.get_group(Os)
                        data.sort_values("Param1",ascending=True,inplace=True)
                        minval=data.loc[data["RMSE"].idxmin()]
                        if Os =="maize_irri":
                            plt.plot(data.Param1,data.RMSE,linestyle="--")
                        else:
                            plt.plot(data.Param1,data.RMSE,label=rela,linewidth=2)
                        plt.plot(minval.Param1,minval.RMSE,marker="*",color="Black")
                        plt.title("%s_RU_%s_Safran_Fcover_sta"%(val,RU))
            plt.legend()
            plt.xlabel(val)
            plt.savefig("/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/RUNS_SAMIR/RUN_MULTI_SITE_ICOS/SENSI_relation_ICOS/PLOT/%s_RU_%s_Safran_Fcover_sta.png"%(val,RU))
    for val in ["REW","maxZr"]:
        for RU in ["1","5","0"]:
            plt.figure(figsize=(10,10))
            for file in os.listdir(d["PC_labo"][:-5]):
                if 'V1' not in file and "RU_"+RU in file and "SAF" in file :
                    rela=file[19:]
                    df=pd.read_csv(d["PC_labo"][:-5]+"/"+file+"/param_RMSE%s.csv"%val)
                    class_OS=df.groupby("OS")
                    for Os in list(set(df.OS)):
                        data=class_OS.get_group(Os)
                        data.sort_values("Param1",ascending=True,inplace=True)
                        minval=data.loc[data["RMSE"].idxmin()]
                        if Os =="maize_irri":
                            plt.plot(data.Param1,data.RMSE,linestyle="--")
                        else:
                            plt.plot(data.Param1,data.RMSE,label=rela,linewidth=2)
                        plt.plot(minval.Param1,minval.RMSE,marker="*",color="Black")
                        plt.title("%s_RU_%s_Safran"%(val,RU))
            plt.legend()
            plt.xlabel(val)
            plt.savefig("/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/RUNS_SAMIR/RUN_MULTI_SITE_ICOS/SENSI_relation_ICOS/PLOT/%s_RU_%s_Safran.png"%(val,RU))