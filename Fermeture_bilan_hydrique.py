# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 10:10:18 2020

@author: yanou
"""


import os
import sqlite3
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
from matplotlib import cm
import csv
#from scipy.optimize import minimize
#from sklearn.metrics import *
import seaborn as sns
import pickle
#from SAMIR_optimi import RMSE
import geopandas as geo
import shapely.geometry as geom
import descartes
from datetime import datetime, date, time, timezone
from scipy import *
from scipy import stats
from pylab import *
#from sklearn.linear_model import LinearRegression


if __name__ == "__main__":
    
    # recupérer les résultats 
    d={}
    name_run="RUN_FERMETURE_BILAN_HYDRIQUE/RUN_vege_avec_pluie_Fcover_assimil_avec_irri_auto/"
    d['Output_model_PC_home']='D:/THESE_TMP/RUNS_SAMIR/'
    d['Output_model_PC_labo']='/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/Bilan_hydrique/'

# =============================================================================
# Version excel bilan hydrique 
# # =============================================================================

# Bilan hydrique journalier 
    dif=[]
    for y in ["2006",'2008',"2010","2012","2014","2015"]:
        # print (y)
        res=pickle.load(open(d['Output_model_PC_labo']+name_run+"/"+str(y)+"/Output/output.df",'rb'))
        res_vege=res.loc[(res.date >= str(y)+"-05-01") &(res.date <= str(y)+"-08-31")]
        meteo=pickle.load(open( d['Output_model_PC_labo']+name_run+"/"+str(y)+"/Inputdata/meteo.df",'rb'))
        meteo_vege=meteo.loc[(meteo.date >= str(y)+"-05-01") &(meteo.date <= str(y)+"-08-31")]
        for j in np.arange(res_vege.shape[0]):
            # print (res_vege.date.iloc[j])
            stop_init=res_vege.TAW.iloc[j]-res_vege.Dr.iloc[j]+res_vege.TDW.iloc[j]-res_vege.Dd.iloc[j]
            if j != np.arange(res_vege.shape[0])[-1]:
                stop_fin=res_vege.TAW.iloc[int(j)+1]-res_vege.Dr.iloc[int(j)+1]+res_vege.TDW.iloc[int(j)+1]-res_vege.Dd.iloc[int(j)+1]
            else: 
                print("last day simul")
            bilan=stop_init+meteo_vege.Prec.iloc[j]-res_vege.ET.iloc[j]
            diff=bilan-stop_fin
            # print('bilan hydrique: %s' %round(bilan,2))
            # print("bilan fin de simu :%s" %round(stop_fin,2))
            # if diff > 1 or diff < -1 :
                # print(res_vege.date.iloc[j])
                # print('##DIFF= %s'%(round(diff,2)))
            # dif.append(diff)
            # print(sum(dif))
        res_vege.loc[res_vege.Ir_auto>1]
        stop_init=res_vege.TAW.iloc[0]-0.0+res_vege.TDW.iloc[0]-0.0
        stop_fin=res_vege.TAW.iloc[-1]-res_vege.Dr.iloc[-1]+res_vege.TDW.iloc[-1]-res_vege.Dd.iloc[-1]
        #  Ajout de la zone evaporatice
        bilan=stop_init+sum(meteo_vege.Prec)+sum(res_vege.Ir_auto)-sum(res_vege.ET)-sum(res_vege.DP)
        print (y)
        print('bilan hydrique: %s' %round(bilan,2))
        print("bilan fin de simu :%s" %round(stop_fin,2))

        plt.figure(figsize=(10,7))
        # plt.plot(res_vege.date,res_vege.NDVI)
        plt.plot(res_vege.date,res_vege.Dr)
        plt.plot(res_vege.date,res_vege.Ir_auto)
        plt.bar(meteo_vege.date,meteo_vege.Prec,color='Red')
        plt.plot(res_vege.date,res_vege.DP)
        