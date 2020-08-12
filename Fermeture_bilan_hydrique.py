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
from scipy.optimize import minimize
from sklearn.metrics import *
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
    
    # recupérer les résultats 
    d={}
    name_run="RUNS_optim_LUT_LAM_ETR/Run_with_optim_params/"
    d['Output_model_PC_home']='D:/THESE_TMP/RUNS_SAMIR/'
    for y in ["2006","2008","2010","2012"]:
        print (y)
        res=pickle.load(open( d['Output_model_PC_home']+name_run+str(y)+"/output_T1.df",'rb'))
        meteo=pickle.load(open( d['Output_model_PC_home']+name_run+str(y)+"/Inputdata/meteo.df",'rb'))
        meteo_periode=meteo.loc[(meteo.date >= res.date[0])&(meteo.date <= res.date.iloc[-1] )]
        #out=res.groupby(["LC","id"])['SWC1', 'SWC2', 'SWC3', 'SWCvol1', 'SWCvol2', 'SWCvol3', 'Dr', 'Dd',"Ir_auto","ET"]
        entre=pd.merge(res[["date",'Ir_auto']],meteo_periode[["Prec","date"]],on=["date"])
        entre.fillna(0,inplace=True)
        entre["sum"]=entre.eval("Ir_auto + Prec")
        sortie=res[['date','SWC1', 'SWC2', 'SWC3', 'SWCvol1', 'SWCvol2', 'SWCvol3', 'Dd','ET']]
        sortie.fillna(0,inplace=True)
        sortie["sum"]=sortie.eval("SWC1+SWC2+SWC3+ET+Dd")
        
        cloture=(entre["sum"]-sortie['sum'])
        print("bilan hydrique : %s" %sum(cloture))
        
