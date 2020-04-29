# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 09:21:57 2020

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
# from ambhas.errlib import NS

def NS(s,o):
    """
    Nash Sutcliffe efficiency coefficient
    input:
        s: simulated
        o: observed
    output:
        ns: Nash Sutcliffe efficient coefficient
    """
    # s,o = filter_nan(s,o)
    return 1 - sum((s-o)**2)/sum((o-np.mean(o))**2)
 


# def nash (s):
#     """The Nash function"""
#    # 1 - sum((s-o)**2)/sum((o-np.mean(o))**2)
#     nash=1 - sum((s-x)**2)/sum((x-np.mean(x))**2)
#     return nash 
    
def RMSE(x,*args) :
    x_data = args[0]
    y_data = args[1]
    rmse= mean_squared_error(x_data,y_data,squared=False)
    return rmse
    
def test(X):
    objective = lambda b: np.sqrt(np.mean((X, b)**2))
    return objective

if __name__ == "__main__":
    
    name_run="RUN_COMPAR_Version"
    years=2006
    d={}
    d['SAMIR_run']="/mnt/d/THESE_TMP/RUNS_SAMIR/"+name_run+"/"+years+"/"
    d['SAMIR_run_Wind']="D:/THESE_TMP/RUNS_SAMIR/"+name_run+"/"
    d["PC_disk_Wind"]="D:/THESE_TMP/RUNS_SAMIR/DATA_Validation/"
    d['PC_disk_unix']="/mnt/d/THESE_TMP/RUNS_SAMIR/DATA_Validation/"
    
   
    # REW = "-8 "
    # Zr_max="2155" 
    # A_kcb = "1.63" 
    date_start="20190103"
    date_end="20191229"
    
    # parampc=pd.read_csv("D:/THESE_TMP/RUNS_SAMIR/RUN_TEST_opi/Inputdata/param_SAMIR12_13.csv",delimiter=",",header=None)
    param=pd.read_csv("/mnt/d/THESE_TMP/RUNS_SAMIR/"+name_run+"/Inputdata/param_SAMIR12_13.csv",delimiter=",",header=None)
    # param.loc[6,13]=A_kcb # ligne 6 , colonne 13
    # param.loc[6,20]=REW
    # param.loc[6,23]=Zr_max
    param.loc[0,1]=date_start
    param.loc[0,3]=date_end
    
    param.to_csv(d["SAMIR_run"]+"/Inputdata/param_otpi_T1.csv",header=False,sep= ',',index=False,na_rep="")
    # param_op=pd.read_csv("D:/THESE_TMP/RUNS_SAMIR/RUN_TEST_opi/Inputdata/param_otpi_T1.csv",delimiter=";",header=None)

    #  Lancement du code
    os.environ["PYTHONPATH"] = "/mnt/c/users/Yann\ Pageot/Documents/code/modspa/modspa2/code/models/:$PYTHONPATH      "
    os.system('python /mnt/c/users/Yann\ Pageot/Documents/code/modspa/modspa2/code/models/main/runSAMIR.py -wd /mnt/d/THESE_TMP/RUNS_SAMIR/'+name_run+' -dd /mnt/d/THESE_TMP/RUNS_SAMIR/'+name_run+'/Inputdata/ -m meteo.df -n maize/NDVI.df -fc maize/FC.df -wp maize/WP.df -o output_T1.df -p param_otpi_T1.csv')
    
    #  Récupération des output de la simulation 
    output_sim=pickle.load(open(d["SAMIR_run"]+"/output_T1.df","rb"))
    all_quantity=[]
    all_number=[]
    all_id=[]
    all_ETR=[]
    for id in list(set(output_sim.id)):
        lam=output_sim.loc[output_sim.id==id]
        # print(r' n° parcelle : %s' %id)
        # print(r'sum irrigation in mm : %s'%lam.groupby(["LC","id"])["Ir_auto"].sum()[0])
        # print(r' nb irrigation : %s' %lam.Ir_auto.where(output_sim["Ir_auto"] != 0.0).dropna().count())
        if id == 0:
            ETRmod=lam[["ET","date"]]
        all_id.append(id)
        all_quantity.append(lam.groupby(["LC","id"])["Ir_auto"].sum()[0])
        all_number.append(lam.Ir_auto.where(output_sim["Ir_auto"] != 0.0).dropna().count())
    all_resu=pd.DataFrame([all_id,all_quantity,all_number]).T
    all_resu.columns=['id','cumul_irr',"nb_irr"]
# =============================================================================
#     validation ETR 
# =============================================================================
# FONCTION WITH Years
    # ETR_lam_2019=pd.read_csv("D:/THESE_TMP/RUNS_SAMIR/DATA_Validation/ETR_LAM2017.csv")
    ETR_lam_2019=pd.read_csv("/mnt/d/THESE_TMP/RUNS_SAMIR/DATA_Validation/ETR_LAM2017.csv")
    # ETR_lam_2019=pd.read_csv("D:/THESE_TMP/RUNS_SAMIR/DATA_Validation/ETR_LAM2019.csv",decimal='.',na_filter=True)
    ETR_lam_2019["date"]=pd.to_datetime( ETR_lam_2019["TIMESTAMP"],format="%Y-%m-%d")
    # fusion mod et obs dans dataframe
    ETR_resu=pd.concat([ETR_lam_2019["LE"].iloc[2:-4],ETRmod.ET],axis=1)
    ETR_resu["date"]=ETRmod.date
    ETR_resu_ss_nn=ETR_resu.dropna()
    val=mean_squared_error(ETR_resu_ss_nn['LE'],ETR_resu_ss_nn["ET"],squared=False)
    print(r'========')
    print(val)
    print(r'========')
    slope, intercept, r_value, p_value, std_err = stats.linregress(ETR_resu_ss_nn['LE'],ETR_resu_ss_nn["ET"])
    bias=1/ETR_resu_ss_nn['LE'].shape[0]*sum(np.mean(ETR_resu_ss_nn["ET"])-ETR_resu_ss_nn['LE']) 
    # fitLine = predict(ETR_lam_2019["LE"].iloc[2:-4])
    plt.figure(figsize=(7,7))
    plt.plot([0.0, 7], [0.0, 7], 'r-', lw=2)
    # plt.plot(ETR_lam_2019["LE"].iloc[2:-4],fitLine,linestyle="-")
    plt.scatter(ETR_resu_ss_nn['LE'],ETR_resu_ss_nn["ET"])
    plt.plot()
    plt.xlabel("ETRr OBS")
    plt.ylabel("ETR model")
    # plt.xlim(0,300)
    # plt.ylim(0,300)
    print(NS(ETR_resu_ss_nn['LE'],ETR_resu_ss_nn["ET"])) 
    plt.text(5,min(ETR_resu_ss_nn["ET"])+0.1,"RMSE = "+str(round(val,2)))
    plt.text(5,min(ETR_resu_ss_nn["ET"])+0.3,"R² = "+str(round(r_value,2)))
    plt.text(5,min(ETR_resu_ss_nn["ET"])+0.5,"Pente = "+str(round(slope,2)))
    plt.text(5,min(ETR_resu_ss_nn["ET"])+0.7,"Biais = "+str(round(bias,2)))
    plt.text(5,min(ETR_resu_ss_nn["ET"])+0.9,"Nash = "+str(round(NS(ETR_resu_ss_nn['LE'],ETR_resu_ss_nn["ET"]),2)))
    plt.savefig(d["SAMIR_run"]+"/plot_ETRobs_ETR_mod.png")
