# -*- coding: utf-8 -*-
"""
Created on Mon May 25 13:53:28 2020

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

def params_update(path,output,date_start,date_end,FminNDVI=0.2,FmaxNDVI=0.9,FminFC=0,FmaxFC=1,Fslope=1.25,Foffset=-0.13,
                  Plateau=70,KminNDVI=0.1,KmaxNDVI=0.9,KminKcb=0,A_kcb=1.49,KmaxKcb=0.98,Koffset=-0.27,Zsoil=1150,Ze=150,Init_RU=0.5,
                  DiffE=5,DiffR=10,REW=6,minZr=125,maxZr=800,p=0.55,FW=100,Irrig_auto=0,Irrig_man=1,Lame_max=30,minDays=20,Kcbmin_start=0.1,
                  Kcbmax_stop=0.85):
    
    param=pd.read_csv(path,delimiter=",",header=None)
    param.loc[6,13]=A_kcb # ligne 6 , colonne 13
    param.loc[6,20]=REW
    param.loc[6,23]=maxZr
    param.loc[6,2]=FminNDVI
    param.loc[6,3]=FmaxNDVI
    param.loc[6,4]=FminFC
    param.loc[6,5]=FmaxFC
    param.loc[6,6]=Fslope
    param.loc[6,7]=Foffset
    param.loc[6,8]=Plateau
    param.loc[6,9]=KminNDVI
    param.loc[6,10]=KmaxNDVI
    param.loc[6,11]=KminKcb
    param.loc[6,12]=KmaxKcb
    param.loc[6,14]=Koffset
    param.loc[6,15]=Zsoil
    param.loc[6,16]=Ze
    param.loc[6,17]=Init_RU
    param.loc[6,18]=DiffE
    param.loc[6,19]=DiffR
    param.loc[6,22]=minZr
    param.loc[6,24]=p
    param.loc[6,25]=FW
    param.loc[6,26]=Irrig_auto
    param.loc[6,27]=Irrig_man
    param.loc[6,28]=Lame_max
    param.loc[6,29]=minDays
    param.loc[6,30]=Kcbmin_start
    param.loc[6,31]=Kcbmax_stop
    param.loc[0,1]=date_start
    param.loc[0,3]=date_end
    param.to_csv(output,header=False,sep= ',',index=False,na_rep="")
       
def select_color_date(x):
    couleurs=[]
    for i in range(len(x)):
        if x.iloc[i].date.strftime('%m-%d')<= "04-01" : 
            couleurs.append("r")
        else : 
            couleurs.append("b")
    return couleurs


if __name__ == "__main__":
    
    result=[]
    for rew in np.arange(-10,20,20):
        # print(r'==============')
        print(rew)
        # print(r'==============')
        for y in ["2006","2015"]:
            print (y)
            name_run="RUN_SENSI_SOL/SENS_Sol_V2"
            d={}
            d['SAMIR_run']="/mnt/d/THESE_TMP/RUNS_SAMIR/"+name_run+"/"+str(y)+"/"
            d['SAMIR_run_Wind']="D:/THESE_TMP/RUNS_SAMIR/"+name_run+"/"+str(y)+"/"
            d["PC_disk_Wind"]="D:/THESE_TMP/RUNS_SAMIR/DATA_Validation/"
            d['PC_disk_unix']="/mnt/d/THESE_TMP/RUNS_SAMIR/"
            params_update( d['SAMIR_run']+"/Inputdata/param_SAMIR12_13.csv",
                          d['SAMIR_run']+"/Inputdata/param_modif.csv",date_start=str(y)+str('0302'),date_end=str(y)+str('1031'),
                          Ze=125,REW=rew,maxZr=2000,Zsoil=3000,DiffE=0.00001,DiffR=0.00001)
    
            #  Lancement du code
            os.environ["PYTHONPATH"] = "/mnt/c/users/Yann\ Pageot/Documents/code/modspa/modspa2/code/models/:$PYTHONPATH      "
            os.system('python /mnt/c/users/Yann\ Pageot/Documents/code/modspa/modspa2/code/models/main/runSAMIR.py -wd /mnt/d/THESE_TMP/RUNS_SAMIR/'+name_run+'/'+str(y)+'/'' -dd /mnt/d/THESE_TMP/RUNS_SAMIR/'+name_run+'/'+str(y)+'/Inputdata/ -m meteo.df -n maize/NDVI.df -fc maize/FC.df -wp maize/WP.df -o output_T1.df -p param_modif.csv ')
    
    
              # Récupération des output de la simulation 
            output_sim=pickle.load(open(d["SAMIR_run"]+"output_T1.df","rb"))
            ETRmod=output_sim[["ET","date"]]
    
            #  Récuparation data_validation ETR
            ETR=pd.read_csv("/mnt/g/Yann_THESE/BESOIN_EAU/DATA_ETR_CESBIO/DATA_ETR_LAM/ETR_LAM"+str(y)+".csv",decimal='.')
            # localiser les nan dans ETR, les supprimer ainsi que les dates pour ensuite comparer 
            dfETR=pd.concat([ETR,ETRmod],axis=1)
            dfETR.columns=["date1",'LE','ET','date']
            dfETR.dropna(inplace=True)
            slope, intercept, r_value, p_value, std_err = stats.linregress(dfETR.LE.to_list(),dfETR.ET.to_list())
            bias=1/dfETR.shape[0]*sum(np.mean(dfETR.ET)-dfETR.LE) 
            fitLine = predict(dfETR.LE)
            # plt.figure(figsize=(7,7))
            # plt.title(zrm)
            # plt.plot([0.0, 10], [0.0,10], 'r-', lw=2)
            # plt.plot(dfETR.LE,fitLine,linestyle="-")
            # plt.scatter(dfETR.LE,dfETR.ET,c=select_color_date(dfETR),s=9)
            # plt.plot()
            # plt.xlabel("ETR OBS")
            # plt.ylabel("ETR model")
            # plt.xlim(0,10)
            # plt.ylim(0,10)
            rms = mean_squared_error(dfETR.LE,dfETR.ET,squared=False)
            # plt.text(8,min(dfETR.ET)+0.1,"RMSE = "+str(round(rms,2)))
            # plt.text(8,min(dfETR.ET)+0.3,"R² = "+str(round(r_value,2)))
            # plt.text(8,min(dfETR.ET)+0.5,"Pente = "+str(round(slope,2)))
            # plt.text(8,min(dfETR.ET)+0.7,"Biais = "+str(round(bias,2)))
            # plt.savefig(d["SAMIR_run"]+"plt_scatter_ETR_%s_%s.png"%(y,rew))
            result.append([rew,rms,bias,r_value,y])
            # print (result)
        resultat=pd.DataFrame(result,columns=["REW","RMSE",'bias','R','years'])
        resultat.to_csv(d["SAMIR_run"][:-5]+"param_RMSE_REW.csv")
    
    # result=[]
    # for y in ["2006","2008","2010","2012","2014","2015","2017","2019"]:
    #     print (y)
    #     name_run="RUNS_optim_LUT_LAM_ETR/"
    #     d={}
    #     d['SAMIR_run']="/mnt/d/THESE_TMP/RUNS_SAMIR/"+name_run+"/"+str(y)+"/"
    #     d['SAMIR_run_Wind']="D:/THESE_TMP/RUNS_SAMIR/"+name_run+"/"+str(y)+"/"
    #     d["PC_disk_Wind"]="D:/THESE_TMP/RUNS_SAMIR/DATA_Validation/"
    #     d['PC_disk_unix']="/mnt/d/THESE_TMP/RUNS_SAMIR/"
    #     params_update(d['SAMIR_run']+"/Inputdata/param_SAMIR12_13.csv",
    #                   d['SAMIR_run']+"/Inputdata/param_modif.csv",date_start=str(y)+str('0302'),date_end=str(y)+str('1031'),
    #                   Ze=125,REW='optim',maxZr='optim',Zsoil=3000,DiffE=0.00001,DiffR=0.00001)

    #     #  Lancement du code
    #     os.environ["PYTHONPATH"] = "/mnt/c/users/Yann\ Pageot/Documents/code/modspa/modspa2/code/models/:$PYTHONPATH      "
    #     os.system('python /mnt/c/users/Yann\ Pageot/Documents/code/modspa/modspa2/code/models/main/runSAMIR.py -wd /mnt/d/THESE_TMP/RUNS_SAMIR/'+name_run+'/'+str(y)+'/'' -dd /mnt/d/THESE_TMP/RUNS_SAMIR/'+name_run+'/'+str(y)+'/Inputdata/ -m meteo.df -n maize/NDVI.df -fc maize/FC.df -wp maize/WP.df -o output_test -p param_modif.csv  -optim param_SAMIR12_13_optim.csv --cal ET')
        
    #     param=pd.read_csv(d["SAMIR_run"]+"output_test_maize_param.txt",header=None,skiprows=2,sep=";")
        
    #     #  Récuparation data_validation ETR
    #     ETR=pd.read_csv("/mnt/g/Yann_THESE/BESOIN_EAU/DATA_ETR_CESBIO/DATA_ETR_LAM/ETR_LAM"+str(y)+".csv",decimal='.')

    #       # Récupération des output de la simulation 
    #     for run in os.listdir(d["SAMIR_run"]):
    #         print(run)
    #         if "maize" in run and "txt" not in run:
    #             num_run=run[18:]

    #             a=open(d["SAMIR_run"]+run,"rb")
    #             output_sim=pickle.load(a)
    #             ETRmod=output_sim[["ET","date"]]
    #             a.close()
    #             #  Récuper le couple de paramètre que je vais varier
    #             parametre=param.loc[int(num_run)]
    #             parametre1=parametre[1]
    #             parametre2=parametre[2]
    #             print (r'para %s; para2 %s'%(parametre1,parametre2))    
    #             # localiser les nan dans ETR, les supprimer ainsi que les dates pour ensuite comparer 
    #             dfETR=pd.concat([ETR,ETRmod],axis=1)
    #             dfETR.columns=["date1",'LE','ET','date']
    #             dfETR.dropna(inplace=True)
    #             slope, intercept, r_value, p_value, std_err = stats.linregress(dfETR.LE.to_list(),dfETR.ET.to_list())
    #             bias=1/dfETR.shape[0]*sum(np.mean(dfETR.ET)-dfETR.LE) 
    #             fitLine = predict(dfETR.LE)
    #             plt.figure(figsize=(7,7))
    #             plt.plot([0.0, 10], [0.0,10], 'r-', lw=2)
    #             plt.plot(dfETR.LE,fitLine,linestyle="-")
    #             plt.scatter(dfETR.LE,dfETR.ET,c=select_color_date(dfETR),s=9)
    #             plt.xlabel("ETR OBS")
    #             plt.ylabel("ETR model")
    #             plt.xlim(0,10)
    #             plt.ylim(0,10)
    #             rms = mean_squared_error(dfETR.LE,dfETR.ET,squared=False)
    #             plt.text(8,min(dfETR.ET)+0.1,"RMSE = "+str(round(rms,2)))
    #             plt.text(8,min(dfETR.ET)+0.3,"R² = "+str(round(r_value,2)))
    #             plt.text(8,min(dfETR.ET)+0.5,"Pente = "+str(round(slope,2)))
    #             plt.text(8,min(dfETR.ET)+0.7,"Biais = "+str(round(bias,2)))
    #             # plt.savefig(d["SAMIR_run"]+"plt_scatter_ETR_%s_%s.png"%(y,param))
    #             result.append([num_run,parametre1,parametre2,rms,bias,r_value,y])
    #     # print (result)
    #     resultat=pd.DataFrame(result,columns=["Num_run","Param1","Param2","RMSE",'bias','R','years'])
    #     resultat.to_csv(d["SAMIR_run"][:-5]+"param_RMSE.csv")
   
    

    