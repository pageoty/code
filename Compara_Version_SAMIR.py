# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 09:28:49 2020

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
 


def nash (s):
    """The Nash function"""
    # 1 - sum((s-o)**2)/sum((o-np.mean(o))**2)
    nash=1 - sum((s-x)**2)/sum((x-np.mean(x))**2)
    return nash 
    
def RMSE(x,*args) :
    x_data = args[0]
    y_data = args[1]
    rmse= mean_squared_error(x_data,y_data,squared=False)
    return rmse
    

def predict(x):
  return slope * x + intercept

if __name__ == "__main__":
    for y in ["2006","2008","2010","2012","2019"]:
        name_run="RUN_COMPAR_Version"
        # years=2012
        d={}
        d['SAMIR_run']="/mnt/d/THESE_TMP/RUNS_SAMIR/"+name_run+"/"+str(y)+"/"
        d['SAMIR_run_Wind']="D:/THESE_TMP/RUNS_SAMIR/"+name_run+"/"+str(y)+"/"
        d["PC_disk_Wind"]="D:/THESE_TMP/RUNS_SAMIR/DATA_Validation/"
        d['PC_disk_unix']="/mnt/d/THESE_TMP/RUNS_SAMIR/DATA_Validation/"
        
       
        # REW = "-8 "
        # Zr_max="2155" 
        # A_kcb = "1.63" 
        # date_start="20120527"
        # date_end="20120823"
        
        # parampc=pd.read_csv("D:/THESE_TMP/RUNS_SAMIR/RUN_TEST_opi/Inputdata/param_SAMIR12_13.csv",delimiter=",",header=None)
        # param=pd.read_csv(d["SAMIR_run"]+"Inputdata/param_SAMIR12_13.csv",delimiter=",",header=None)
        # param.loc[6,13]=A_kcb # ligne 6 , colonne 13
        # param.loc[6,20]=REW
        # param.loc[6,23]=Zr_max
        # param.loc[0,1]=date_start
        # param.loc[0,3]=date_end
        
        # param.to_csv(d["SAMIR_run"]+"/Inputdata/param_otpi_T1.csv",header=False,sep= ',',index=False,na_rep="")
        # param_op=pd.read_csv("D:/THESE_TMP/RUNS_SAMIR/RUN_TEST_opi/Inputdata/param_otpi_T1.csv",delimiter=";",header=None)
    
        #  Lancement du code
        os.environ["PYTHONPATH"] = "/mnt/c/users/Yann\ Pageot/Documents/code/modspa/modspa2/code/models/:$PYTHONPATH      "
        os.system('python /mnt/c/users/Yann\ Pageot/Documents/code/modspa/modspa2/code/models/main/runSAMIR.py -wd /mnt/d/THESE_TMP/RUNS_SAMIR/'+name_run+'/'+str(y)+'/'' -dd /mnt/d/THESE_TMP/RUNS_SAMIR/'+name_run+'/'+str(y)+'/Inputdata/ -m meteo.df -n maize/NDVI.df -fc maize/FC.df -wp maize/WP.df -o output_T1.df -p param_otpi_T1.csv')
        
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
        # df=pd.read_csv("D:/THESE_TMP/RUNS_SAMIR/DATA_Validation/SAMIR_EXCEL_OUTPUT/LAM_"+str(y)+".csv",decimal=",")
        df=pd.read_csv("/mnt/d/THESE_TMP/RUNS_SAMIR/DATA_Validation/SAMIR_EXCEL_OUTPUT/LAM_"+str(y)+".csv",decimal=",")
        df["Date"]=pd.to_datetime(df["Date"],format="%d/%m/%Y")
        df.set_index("Date",inplace=True)
        output_sim.set_index("date",inplace=True)
        dict_var={"fc":"FC","Kcb":"Kcb","ET":"ET","SWC1":"SWC1","Hvol1":"SWCvol1","Zr":"Zr","TAW":"TAW","Dr":"Dr","Dep":"Dep","Dd":"Dd"}
        for c,o in zip(dict_var.keys(),dict_var.values()):
            print(c,o)
    # Plot
            val=mean_squared_error(df[c],output_sim[o],squared=False)
            # print(r'========')
            # print(val)
            # print(r'========')
            slope, intercept, r_value, p_value, std_err = stats.linregress(df[c],output_sim[o])
            bias=1/df[c].shape[0]*sum(np.mean(df[c])-output_sim[o]) 
            fitLine = predict(df[c])
            plt.figure(figsize=(7,7))
            plt.plot([0.0, max(output_sim[o])], [0.0, max(output_sim[o])], 'r-', lw=2)
            plt.plot(df[c],fitLine,linestyle="-")
            plt.scatter(df[c],output_sim[o])
            plt.plot()
            plt.xlabel("%s model_Excel"%c)
            plt.ylabel("%s model_python"%o)
            plt.title(c)
            plt.xlim(0,max(output_sim[o]))
            plt.ylim(0,max(output_sim[o]))
            # print(NS(df[c],output_sim[o])) 
            plt.text(5,min(output_sim[o])+0.1,"RMSE = "+str(round(val,2)))
            # plt.text(5,min(output_sim[o])+0.3,"R² = "+str(round(r_value,2)))
            # plt.text(5,min(output_sim[o])+0.5,"Pente = "+str(round(slope,2)))
            # plt.text(5,min(output_sim[o])+0.7,"Biais = "+str(round(bias,2)))
            # plt.text(5,min(output_sim[o])+0.9,"Nash = "+str(round(NS(df[c],output_sim[o]),2)))
            plt.savefig(d["SAMIR_run"]+"/Compar_Version_%s.png"%c)
    
    # Plot dyna tempo
        for c,o in zip(dict_var.keys(),dict_var.values()):
            print(c,o)
            fig, ax = plt.subplots(figsize=(7, 7))
            sns.set(style="darkgrid")
            sns.set_context('paper')
            plt.plot(output_sim[o],label="model_python")
            plt.plot(df[c],label="model_Excel")
            plt.legend()
            plt.title('%s'%c)
            plt.savefig(d["SAMIR_run"]+"/Compar_Version_temporelle_%s.png"%c)
