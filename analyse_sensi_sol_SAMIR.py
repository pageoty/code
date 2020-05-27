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
    
    name_run="RUN_SENSI_SOL"
    d={}
    d['SAMIR_run']="/mnt/d/THESE_TMP/RUNS_SAMIR/"+name_run+"/"
    d['SAMIR_run_Wind']="D:/THESE_TMP/RUNS_SAMIR/"+name_run+"/"
    
    resultat_rew=pd.read_csv(d["SAMIR_run_Wind"]+"SENSI_REW/param_RMSE.csv")
    resultat_Zrmax=pd.read_csv(d["SAMIR_run_Wind"]+"SENSI_ZRmax/param_RMSE_Zrmax.csv")
    a=resultat_rew.groupby(["years"])
    b=resultat_Zrmax.groupby(["years"])
    for y in ["2006","2008","2010","2012","2014","2015","2017","2019"]:
        plt.figure(figsize=(10,10))
        REW=a.get_group(int(y))
        Zrmax=b.get_group(int(y))
        plt.scatter(REW.RMSE[0:50],Zrmax.RMSE)
        # for i,p in zip(np.arange(len(REW.RMSE[0:50])),REW.REW[0:50]):
        #     plt.text(REW.RMSE.iloc[i]+0.001 , y=Zrmax.RMSE.iloc[i], s=p,size=9)
        # for i,p in zip(np.arange(len(REW.RMSE[0:50])),Zrmax.ZR_max):
        #     plt.text(REW.RMSE.iloc[i]-0.005 , y=Zrmax.RMSE.iloc[i], s=p,size=9)
        plt.title(y)
        plt.xlabel("RSME REW")
        plt.ylabel('RMSE Zrmax')
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
