# -*- coding: utf-8 -*-
"""
Created on Tue May 26 15:25:10 2020

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
import seaborn as sns
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

if __name__ == "__main__":
    
    name_run="RUNS_optim_LUT_LAM_ETR/"
    d={}
    d['SAMIR_run']="/mnt/d/THESE_TMP/RUNS_SAMIR/"+name_run+"/"
    d['SAMIR_run_Wind']="D:/THESE_TMP/RUNS_SAMIR/"+name_run+"/"
    
    resultat=pd.read_csv(d["SAMIR_run_Wind"]+"param_RMSE.csv")
    # resultat_Zrmax=pd.read_csv(d["SAMIR_run_Wind"]+"SENSI_ZRmax/param_RMSE_ZR_max.csv")
    a=resultat.groupby(["years"])
    # b=resultat_Zrmax.groupby(["years"])
    for y in ["2006"]:
        plt.figure(figsize=(10,10))
        REW=a.get_group(int(y))
        
        
        
        # plt.scatter(REW.RMSE[0:50],Zrmax.RMSE)
        # for i,p in zip(np.arange(len(REW.RMSE[0:50])),REW.REW[0:50]):
        #     plt.text(REW.RMSE.iloc[i]+0.001 , y=Zrmax.RMSE.iloc[i], s=p,size=9)
        # for i,p in zip(np.arange(len(REW.RMSE[0:50])),Zrmax.ZR_max):
        #     plt.text(REW.RMSE.iloc[i]-0.005 , y=Zrmax.RMSE.iloc[i], s=p,size=9)

        # plt.xlabel("RSME REW")
        # plt.ylabel('RMSE Zrmax')
        # sns.heatmap(test)
        # sns.heatmap(REW.RMSE,Zrmax.RMSE)
        # plt.savefig(d['SAMIR_run_Wind']+"plt_"+str(y)+"calivra.png")
        # plt.xlim(0.75,max(REW.RMSE))
        # plt.ylim(0.75,max(REW.RMSE))
        # A,B,C=polyfit(REW.RMSE[0:50],Zrmax.RMSE,2)
        # plt.plot(REW.RMSE[0:50],A*REW.RMSE[0:50]**2+B*REW.RMSE[0:50]+C)
        min_value=REW.loc[REW['RMSE'].idxmin()]
        # print(min_value)
        # find min 
        # resultat.loc[(resultat.years==2010) &(resultat.RMSE==resultat.RMSE.min())]
        
        
        # a,b,c=polyfit(REW.RMSE[0:50],Zrmax.RMSE,2)
        # plt.plot(REW.RMSE[0:50],Zrmax.RMSE,"*")
        # plt.plot(REW.RMSE[0:50],a*REW.RMSE[0:50]**2+b*REW.RMSE[0:50]+c)
