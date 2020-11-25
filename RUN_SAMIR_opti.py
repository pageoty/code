# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 10:40:05 2020

@author: Yann Pageot
"""

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
import time
from SAMIR_optimi import RMSE
import geopandas as geo
import shapely.geometry as geom
# import descartes
from datetime import datetime, date, time, timezone
from scipy import *
from scipy import stats
from pylab import *
from sklearn.linear_model import LinearRegression
import argparse
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

def params_update(path,output,date_start,date_end,ligne_OS=6,FminNDVI=0.2,FmaxNDVI=0.9,FminFC=0,FmaxFC=1,Fslope=1.25,Foffset=-0.13,
                  Plateau=70,KminNDVI=0.1,KmaxNDVI=0.9,KminKcb=0,A_kcb=1.358,KmaxKcb=0.98,Koffset=-0.27,Zsoil=1150,Ze=150,Init_RU=0.5,
                  DiffE=0.00001,DiffR=0.00001,REW=6,minZr=125,maxZr=800,p=0.55,FW=100,Irrig_auto=0,Irrig_man=1,Lame_max=30,minDays=20,Kcbmin_start=0.1,
                  Kcbmax_stop=0.85,m=0.15):
    
    param=pd.read_csv(path,delimiter=",",header=None)
    param.loc[int(ligne_OS),13]=A_kcb 
    param.loc[int(ligne_OS),20]=REW
    param.loc[int(ligne_OS),21]=m
    param.loc[int(ligne_OS),23]=maxZr
    param.loc[int(ligne_OS),2]=FminNDVI
    param.loc[int(ligne_OS),3]=FmaxNDVI
    param.loc[int(ligne_OS),4]=FminFC
    param.loc[int(ligne_OS),5]=FmaxFC
    param.loc[int(ligne_OS),6]=Fslope
    param.loc[int(ligne_OS),7]=Foffset
    param.loc[int(ligne_OS),8]=Plateau
    param.loc[int(ligne_OS),9]=KminNDVI
    param.loc[int(ligne_OS),10]=KmaxNDVI
    param.loc[int(ligne_OS),11]=KminKcb
    param.loc[int(ligne_OS),12]=KmaxKcb
    param.loc[int(ligne_OS),14]=Koffset
    param.loc[int(ligne_OS),15]=Zsoil
    param.loc[int(ligne_OS),16]=Ze
    param.loc[int(ligne_OS),17]=Init_RU
    param.loc[int(ligne_OS),18]=DiffE
    param.loc[int(ligne_OS),19]=DiffR
    param.loc[int(ligne_OS),22]=minZr
    param.loc[int(ligne_OS),24]=p
    param.loc[int(ligne_OS),25]=FW
    param.loc[int(ligne_OS),26]=Irrig_auto
    param.loc[int(ligne_OS),27]=Irrig_man
    param.loc[int(ligne_OS),28]=Lame_max
    param.loc[int(ligne_OS),29]=minDays
    param.loc[int(ligne_OS),30]=Kcbmin_start
    param.loc[int(ligne_OS),31]=Kcbmax_stop
    param.loc[0,1]=date_start
    param.loc[0,3]=date_end
    param.to_csv(output,header=False,sep= ',',index=False,na_rep="")
   
def params_opti(path,output_path,param1,value_P1,param2=None,value_P2=None,ligne_OS=1):
    opti_para=pd.read_csv(path,header=None)
    opti_para.loc[0,2]=param1
    opti_para.loc[0,3]=param2
    opti_para.loc[int(ligne_OS),2]=value_P1
    opti_para.loc[int(ligne_OS),3]=value_P2
    opti_para.to_csv(output_path,header=False,sep= ',',index=False,na_rep="")
    
def select_color_date(x):
    couleurs=[]
    for i in range(len(x)):
        if x.index[i].strftime('%m-%d')<= "04-01" : 
            couleurs.append("r")
        else : 
            couleurs.append("b")
    return couleurs

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run optimisation SAMIR ')
    parser.add_argument('-path', dest='path',nargs='+',help="path file ",required = True)
    parser.add_argument('-optim',dest='optim',nargs='+',help='optimisation value',required = True)
    parser.add_argument('-optim2',dest='optim2',nargs='+',help='optimisation value')
    parser.add_argument('-name',dest='name_run',nargs='+',help='name run',required = True)
    parser.add_argument('-REW',dest='REW',nargs='+',help='Value REW ',required = True)
    parser.add_argument('-RU_start',dest='IniRU',nargs='+',help='Value init ru',required = True)
    parser.add_argument('-meteo',dest='meteo',nargs='+',help='source meteo data')
    parser.add_argument('-REW2',dest='REW2',nargs='+',help='source meteo data')
    parser.add_argument('-A_kcb',dest='akcb',nargs='+',help='slope_relation_NDVI/Kcb')
    parser.add_argument('-B_kcb',dest='bkcb',nargs='+',help='offset_relation_NDVI/Kcb')
    args = parser.parse_args()
    print (args.optim)
    print(args.name_run)

    
    result=[]
    for y in ["2006","2008","2010","2012","2014","2015"]:# 
        print (y)
        # name_run="RUN_MULTI_SITE_ICOS/OPTI_SAF_RU_Fcover_sta_value05/"
        name_run=str(args.name_run).strip("['']")
        # optimis_val="REW"
        optimis_val=str(args.optim).strip("['']")
        if len(optimis_val) < 6:
            optimis_val=optimis_val
        else:
            optimis_val=optimis_val[0:3]+optimis_val[7:]
        print(r'===============')
        print(optimis_val)
        print(r'===============')
        d={}
        d['SAMIR_run']="/mnt/d/THESE_TMP/TRAITEMENT/RUNS_SAMIR/"+name_run+"/"+str(y)+"/"
        d['SAMIR_run_Wind']="D:/THESE_TMP/RUNS_SAMIR/"+name_run+"/"+str(y)+"/"
        d["PC_disk_Wind"]="D:/THESE_TMP/RUNS_SAMIR/DATA_Validation/"
        d['PC_disk_unix']="/mnt/d/THESE_TMP/RUNS_SAMIR/"
        # d["PC_labo"]="/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/RUNS_SAMIR/"+name_run+"/"+str(y)+"/"
        d["SAMIR_run"]=str(args.path).strip("['']")+"/"+name_run+"/"+str(y)+"/"
        print(d["SAMIR_run"])
        print(str(args.meteo))
        #  Creation file du run 
        if name_run not in os.listdir(str(args.path).strip("['']")):
            os.mkdir ("%s/%s"%(str(args.path).strip("['']"),str(args.name_run).strip("['']")))
        # os.mkdir('%s/%s/'%(d["PC_labo"],name_run))
        if str(y) not in os.listdir(d["SAMIR_run"][:-5]):
            os.mkdir ('%s/%s/%s'%(str(args.path).strip("['']"),str(args.name_run).strip("['']"),str(y)))
        #  Déplacement all file 
        if str(args.meteo).strip("['']")=="SAFRAN":
            os.system("scp -r /mnt/d/THESE_TMP/TRAITEMENT/RUNS_SAMIR/DATA_SCP_ICOS/SAFRAN/"+str(y)+"/* %s"%(d['SAMIR_run']))
        else:
            os.system("cp -r /mnt/d/THESE_TMP/TRAITEMENT/RUNS_SAMIR/DATA_SCP_ICOS/ICOS_STAT_ss_Irri/"+str(y)+"/* %s"%(d['SAMIR_run']))

        classes=["maize_irri"]
        if "LAI" in name_run:
            b=open(d["SAMIR_run"]+"/Inputdata/maize_irri/LAI"+str(y)+".df","rb")
            LAI=pickle.load(b)
            b.close()
            if optimis_val =="REW" :
                timestart=str(y)+"-07-01"
                solnu=LAI.loc[(LAI.LAI>0.2)&(LAI.date<timestart)]
                lastdate=solnu.iloc[-1]["date"].strftime('%m-%d').replace("-", "")
                if len(classes)==2:
                    params_update(d['SAMIR_run']+"/Inputdata/param_SAMIR12_13.csv",
                              d['PC_labo']+"/Inputdata/param_modif.csv",date_start=str(y)+str('0302'),date_end=str(y)+str(lastdate),
                              Ze=125,REW='optim',minZr=125,maxZr=1500,Zsoil=3000,DiffE=0.00001,DiffR=0.00001,Init_RU=float(str(args.IniRU).strip("['']")),Irrig_auto=0,Irrig_man=1,A_kcb=float(str(args.akcb).strip("['']")),m=0.15, Koffset=float(str(args.bkcb).strip("['']")))
                    params_update(d['SAMIR_run']+"/Inputdata/param_modif.csv",
                              d['SAMIR_run']+"/Inputdata/param_modif.csv",date_start=str(y)+str('0302'),date_end=str(y)+str(lastdate),
                               ligne_OS=7,Ze=125,REW='optim',minZr=125,maxZr=1500,Zsoil=3000,DiffE=5,DiffR=5,Init_RU=float(str(args.IniRU).strip("['']")),Irrig_auto=0,Irrig_man=1,A_kcb=float(str(args.akcb).strip("['']")),m=0.15, Koffset=float(str(args.bkcb).strip("['']")))
                    params_opti(d["SAMIR_run"]+"/Inputdata/param_SAMIR12_13_optim.csv",output_path=d["SAMIR_run"]+"/Inputdata/test_optim.csv",param1="REW",value_P1="-100/100/10/lin")
                    params_opti(d["SAMIR_run"]+"/Inputdata/test_optim.csv",output_path=d["SAMIR_run"]+"/Inputdata/test_optim.csv",param1="REW",value_P1="-100/100/10/lin",ligne_OS=2)
                else:
                    params_update(d['SAMIR_run']+"/Inputdata/param_SAMIR12_13.csv",
                              d['SAMIR_run']+"/Inputdata/param_modif.csv",date_start=str(y)+str('0302'),date_end=str(y)+str(lastdate),
                              Ze=125,REW='optim',minZr=125,maxZr=1500,Zsoil=3000,DiffE=0.00001,DiffR=0.00001,Init_RU=float(str(args.IniRU).strip("['']")),Irrig_auto=0,Irrig_man=1,A_kcb=float(str(args.akcb).strip("['']")),m=0.15, Koffset=float(str(args.bkcb).strip("['']")))
                    params_opti(d["SAMIR_run"]+"/Inputdata/param_SAMIR12_13_optim.csv",output_path=d["SAMIR_run"]+"/Inputdata/test_optim.csv",param1="REW",value_P1="-100/100/10/lin")
            elif optimis_val =="maxZr" :
                timestart=str(y)+"-05-01"
                vege=LAI.loc[(LAI.LAI>0.2)&(LAI.date>timestart)]
                lastdate=vege.iloc[0]["date"].strftime('%m-%d').replace("-", "")
                if len(classes)==2:
                    params_update(d['SAMIR_run']+"/Inputdata/param_SAMIR12_13.csv",
                              d['SAMIR_run']+"/Inputdata/param_modif.csv",date_start=str(y)+str(lastdate),date_end=str(y)+str('1031'),
                              Ze=125,REW=float(str(args.REW).strip("['']")),minZr=125,maxZr='optim',Zsoil=3000,DiffE=0.00001,DiffR=0.00001,Init_RU=float(str(args.IniRU).strip("['']")),Irrig_auto=0,Irrig_man=1,A_kcb=float(str(args.akcb).strip("['']")), Koffset=float(str(args.bkcb).strip("['']")))
                    params_update(d['SAMIR_run']+"/Inputdata/param_modif.csv",
                                  d['SAMIR_run']+"/Inputdata/param_modif.csv",date_start=str(y)+str(lastdate),date_end=str(y)+str('1031'),
                                  ligne_OS=7,Ze=125,REW=float(str(args.REW2).strip("['']")),minZr=125,maxZr='optim',Zsoil=3000,DiffE=5,DiffR=5,Init_RU=float(str(args.IniRU).strip("['']")),Irrig_auto=0,Irrig_man=1,A_kcb=float(str(args.akcb).strip("['']")), Koffset=float(str(args.bkcb).strip("['']")))
                    params_opti(d["SAMIR_run"]+"/Inputdata/param_SAMIR12_13_optim.csv",output_path=d["SAMIR_run"]+"/Inputdata/test_optim.csv",param1="maxZr",value_P1="500/2999/500/lin")
                    params_opti(d["SAMIR_run"]+"/Inputdata/test_optim.csv",output_path=d["SAMIR_run"]+"/Inputdata/test_optim.csv",param1="maxZr",value_P1="500/2999/100/lin",ligne_OS=2)
                else:
                    params_update(d['SAMIR_run']+"/Inputdata/param_SAMIR12_13.csv",
                             d['SAMIR_run']+"/Inputdata/param_modif.csv",date_start=str(y)+str(lastdate),date_end=str(y)+str('1026'),
                             Ze=125,REW=float(str(args.REW).strip("['']")),minZr=125,maxZr='optim',Zsoil=3000,DiffE=0.00001,DiffR=0.00001,Init_RU=float(str(args.IniRU).strip("['']")),Irrig_auto=0,Irrig_man=1,A_kcb=float(str(args.akcb).strip("['']")), Koffset=float(str(args.bkcb).strip("['']")))
                    params_opti(d["SAMIR_run"]+"/Inputdata/param_SAMIR12_13_optim.csv",output_path=d["SAMIR_run"]+"/Inputdata/test_optim.csv",param1="maxZr",value_P1="500/2999/100/lin")
            else:
                print('two optimisation')
                timestart=str(y)+"-05-01"
                vege=LAI.loc[(LAI.LAI>1)&(LAI.date>timestart)]
                lastdate=vege.iloc[0]["date"].strftime('%m-%d').replace("-", "")
                if len(classes)==2:
                    params_update(d['SAMIR_run']+"/Inputdata/param_SAMIR12_13.csv",
                              d['SAMIR_run']+"/Inputdata/param_modif.csv",date_start=str(y)+str(lastdate),date_end=str(y)+str('1026'),
                              Ze=125,REW="optim",minZr=125,maxZr='optim',Zsoil=3000,DiffE=0.00001,DiffR=0.00001,Init_RU=float(str(args.IniRU).strip("['']")),Irrig_auto=0,Irrig_man=1,A_kcb=float(str(args.akcb).strip("['']")), Koffset=float(str(args.bkcb).strip("['']")))
                    params_update(d['SAMIR_run']+"/Inputdata/param_modif.csv",
                                  d['SAMIR_run']+"/Inputdata/param_modif.csv",date_start=str(y)+str(lastdate),date_end=str(y)+str('1026'),
                                  ligne_OS=7,Ze=125,REW="optim",minZr=125,maxZr='optim',Zsoil=3000,DiffE=5,DiffR=5,Init_RU=float(str(args.IniRU).strip("['']")),Irrig_auto=0,Irrig_man=1,A_kcb=float(str(args.akcb).strip("['']")), Koffset=float(str(args.bkcb).strip("['']")))
                    params_opti(d["SAMIR_run"]+"/Inputdata/param_SAMIR12_13_optim.csv",output_path=d["SAMIR_run"]+"/Inputdata/test_optim.csv",param1="maxZr",value_P1="500/2900/250/lin",param2="REW",value_P2="-50/40/10/lin")
                    params_opti(d["SAMIR_run"]+"/Inputdata/test_optim.csv",output_path=d["SAMIR_run"]+"/Inputdata/test_optim.csv",param1="maxZr",value_P1="500/2900/250/lin",ligne_OS=2,param2="REW",value_P2="-50/40/10/lin")
                else:
                    params_update(d['SAMIR_run']+"/Inputdata/param_SAMIR12_13.csv",
                             d['SAMIR_run']+"/Inputdata/param_modif.csv",date_start=str(y)+str(lastdate),date_end=str(y)+str('1026'),
                             Ze=125,REW="optim",minZr=125,maxZr='optim',Zsoil=3000,DiffE=0.00001,DiffR=0.00001,Init_RU=float(str(args.IniRU).strip("['']")),Irrig_auto=0,Irrig_man=1,A_kcb=float(str(args.akcb).strip("['']")), Koffset=float(str(args.bkcb).strip("['']")))
                    params_opti(d["SAMIR_run"]+"/Inputdata/param_SAMIR12_13_optim.csv",output_path=d["SAMIR_run"]+"/Inputdata/test_optim.csv",param1="maxZr",value_P1="500/2900/250/lin",param2="REW",value_P2="-50/40/10/lin")
        else:
            # Définir période sol nu
            b=open(d["SAMIR_run"]+"/Inputdata/maize_irri/NDVI"+str(y)+".df","rb")
            NDVI=pickle.load(b)
            b.close()
            if optimis_val =="REW" :
                timestart=str(y)+"-07-01"
                solnu=NDVI.loc[(NDVI.NDVI>0.25)&(NDVI.date<timestart)]
                lastdate=solnu.iloc[-1]["date"].strftime('%m-%d').replace("-", "")
                if len(classes)==2:
                    params_update(d['SAMIR_run']+"/Inputdata/param_SAMIR12_13.csv",
                              d['PC_labo']+"/Inputdata/param_modif.csv",date_start=str(y)+str('0302'),date_end=str(y)+str(lastdate),
                              Ze=125,REW='optim',minZr=125,maxZr=1500,Zsoil=3000,DiffE=0.00001,DiffR=0.00001,Init_RU=float(str(args.IniRU).strip("['']")),Irrig_auto=0,Irrig_man=1,A_kcb=float(str(args.akcb).strip("['']")),m=0.15, Koffset=float(str(args.bkcb).strip("['']")))
                    params_update(d['SAMIR_run']+"/Inputdata/param_modif.csv",
                              d['SAMIR_run']+"/Inputdata/param_modif.csv",date_start=str(y)+str('0302'),date_end=str(y)+str(lastdate),
                               ligne_OS=7,Ze=125,REW='optim',minZr=125,maxZr=1500,Zsoil=3000,DiffE=5,DiffR=5,Init_RU=float(str(args.IniRU).strip("['']")),Irrig_auto=0,Irrig_man=1,A_kcb=float(str(args.akcb).strip("['']")),m=0.15, Koffset=float(str(args.bkcb).strip("['']")))
                    params_opti(d["SAMIR_run"]+"/Inputdata/param_SAMIR12_13_optim.csv",output_path=d["SAMIR_run"]+"/Inputdata/test_optim.csv",param1="REW",value_P1="-100/100/10/lin")
                    params_opti(d["SAMIR_run"]+"/Inputdata/test_optim.csv",output_path=d["SAMIR_run"]+"/Inputdata/test_optim.csv",param1="REW",value_P1="-100/100/10/lin",ligne_OS=2)
                else:
                    params_update(d['SAMIR_run']+"/Inputdata/param_SAMIR12_13.csv",
                              d['SAMIR_run']+"/Inputdata/param_modif.csv",date_start=str(y)+str('0302'),date_end=str(y)+str(lastdate),
                              Ze=125,REW='optim',minZr=125,maxZr=1500,Zsoil=3000,DiffE=0.00001,DiffR=0.00001,Init_RU=float(str(args.IniRU).strip("['']")),Irrig_auto=0,Irrig_man=1,A_kcb=float(str(args.akcb).strip("['']")),m=0.15, Koffset=float(str(args.bkcb).strip("['']")))
                    params_opti(d["SAMIR_run"]+"/Inputdata/param_SAMIR12_13_optim.csv",output_path=d["SAMIR_run"]+"/Inputdata/test_optim.csv",param1="REW",value_P1="-100/100/10/lin")
            elif optimis_val =="maxZr" :
                timestart=str(y)+"-05-01"
                vege=NDVI.loc[(NDVI.NDVI>0.25)&(NDVI.date>timestart)]
                lastdate=vege.iloc[0]["date"].strftime('%m-%d').replace("-", "")
                if len(classes)==2:
                    params_update(d['SAMIR_run']+"/Inputdata/param_SAMIR12_13.csv",
                              d['SAMIR_run']+"/Inputdata/param_modif.csv",date_start=str(y)+str(lastdate),date_end=str(y)+str('1031'),
                              Ze=125,REW=float(str(args.REW).strip("['']")),minZr=125,maxZr='optim',Zsoil=3000,DiffE=0.00001,DiffR=0.00001,Init_RU=float(str(args.IniRU).strip("['']")),Irrig_auto=0,Irrig_man=1,A_kcb=float(str(args.akcb).strip("['']")), Koffset=float(str(args.bkcb).strip("['']")))
                    params_update(d['SAMIR_run']+"/Inputdata/param_modif.csv",
                                  d['SAMIR_run']+"/Inputdata/param_modif.csv",date_start=str(y)+str(lastdate),date_end=str(y)+str('1031'),
                                  ligne_OS=7,Ze=125,REW=float(str(args.REW2).strip("['']")),minZr=125,maxZr='optim',Zsoil=3000,DiffE=5,DiffR=5,Init_RU=float(str(args.IniRU).strip("['']")),Irrig_auto=0,Irrig_man=1,A_kcb=float(str(args.akcb).strip("['']")), Koffset=float(str(args.bkcb).strip("['']")))
                    params_opti(d["SAMIR_run"]+"/Inputdata/param_SAMIR12_13_optim.csv",output_path=d["SAMIR_run"]+"/Inputdata/test_optim.csv",param1="maxZr",value_P1="500/2999/500/lin")
                    params_opti(d["SAMIR_run"]+"/Inputdata/test_optim.csv",output_path=d["SAMIR_run"]+"/Inputdata/test_optim.csv",param1="maxZr",value_P1="500/2999/500/lin",ligne_OS=2)
                else:
                    params_update(d['SAMIR_run']+"/Inputdata/param_SAMIR12_13.csv",
                             d['SAMIR_run']+"/Inputdata/param_modif.csv",date_start=str(y)+str(lastdate),date_end=str(y)+str('1026'),
                             Ze=125,REW=float(str(args.REW).strip("['']")),minZr=125,maxZr='optim',Zsoil=3000,DiffE=0.00001,DiffR=0.00001,Init_RU=float(str(args.IniRU).strip("['']")),Irrig_auto=0,Irrig_man=1,A_kcb=float(str(args.akcb).strip("['']")), Koffset=float(str(args.bkcb).strip("['']")))
                    params_opti(d["SAMIR_run"]+"/Inputdata/param_SAMIR12_13_optim.csv",output_path=d["SAMIR_run"]+"/Inputdata/test_optim.csv",param1="maxZr",value_P1="500/2999/500/lin")
            else:
                print('two optimisation')
                timestart=str(y)+"-05-01"
                vege=NDVI.loc[(NDVI.NDVI>0.25)&(NDVI.date>timestart)]
                lastdate=vege.iloc[0]["date"].strftime('%m-%d').replace("-", "")
                if len(classes)==2:
                    params_update(d['SAMIR_run']+"/Inputdata/param_SAMIR12_13.csv",
                              d['SAMIR_run']+"/Inputdata/param_modif.csv",date_start=str(y)+str(lastdate),date_end=str(y)+str('1026'),
                              Ze=125,REW="optim",minZr=125,maxZr='optim',Zsoil=3000,DiffE=0.00001,DiffR=0.00001,Init_RU=float(str(args.IniRU).strip("['']")),Irrig_auto=0,Irrig_man=1,A_kcb=float(str(args.akcb).strip("['']")), Koffset=float(str(args.bkcb).strip("['']")))
                    params_update(d['SAMIR_run']+"/Inputdata/param_modif.csv",
                                  d['SAMIR_run']+"/Inputdata/param_modif.csv",date_start=str(y)+str(lastdate),date_end=str(y)+str('1026'),
                                  ligne_OS=7,Ze=125,REW="optim",minZr=125,maxZr='optim',Zsoil=3000,DiffE=5,DiffR=5,Init_RU=float(str(args.IniRU).strip("['']")),Irrig_auto=0,Irrig_man=1,A_kcb=float(str(args.akcb).strip("['']")), Koffset=float(str(args.bkcb).strip("['']")))
                    params_opti(d["SAMIR_run"]+"/Inputdata/param_SAMIR12_13_optim.csv",output_path=d["SAMIR_run"]+"/Inputdata/test_optim.csv",param1="maxZr",value_P1="500/2900/250/lin",param2="REW",value_P2="-50/40/10/lin")
                    params_opti(d["SAMIR_run"]+"/Inputdata/test_optim.csv",output_path=d["SAMIR_run"]+"/Inputdata/test_optim.csv",param1="maxZr",value_P1="500/2900/250/lin",ligne_OS=2,param2="REW",value_P2="-50/40/10/lin")
                else:
                    params_update(d['SAMIR_run']+"/Inputdata/param_SAMIR12_13.csv",
                             d['SAMIR_run']+"/Inputdata/param_modif.csv",date_start=str(y)+str(lastdate),date_end=str(y)+str('1026'),
                             Ze=125,REW="optim",minZr=125,maxZr='optim',Zsoil=3000,DiffE=0.00001,DiffR=0.00001,Init_RU=float(str(args.IniRU).strip("['']")),Irrig_auto=0,Irrig_man=1,A_kcb=float(str(args.akcb).strip("['']")), Koffset=float(str(args.bkcb).strip("['']")))
                    params_opti(d["SAMIR_run"]+"/Inputdata/param_SAMIR12_13_optim.csv",output_path=d["SAMIR_run"]+"/Inputdata/test_optim.csv",param1="maxZr",value_P1="500/2900/250/lin",param2="REW",value_P2="-50/40/10/lin")
        #  Lancement du code
        if "Output" in os.listdir(d['SAMIR_run']):
            print ("existing file")
        else :
            os.mkdir ("%s/Output"%d['SAMIR_run']) # allows create file via call system
        #  création file Output
        if optimis_val in os.listdir(d['SAMIR_run']+"/Output/"):
            print ("existing file")
        else :
            os.mkdir ("%s/Output/%s"%(d['SAMIR_run'],optimis_val)) 
            
        if "Plot" in os.listdir(d['SAMIR_run']+"/Output/"+optimis_val+"/"):
            print ("existing file")
        else :
            os.mkdir ("%s/Output/%s/Plot"%(d['SAMIR_run'],optimis_val)) 
            
        if "Plot_dyna" in os.listdir(d['SAMIR_run']+"/Output/"+optimis_val+"/Plot/"):
            print ("existing file")
        else :
            os.mkdir ("%s/Output/%s/Plot/Plot_dyna"%(d['SAMIR_run'],optimis_val)) 
            
        if "CSV" in os.listdir(d['SAMIR_run']+"/Output/"+optimis_val+"/"):
            print ("existing file")
        else :
            os.mkdir ("%s/Output/%s/CSV"%(d['SAMIR_run'],optimis_val)) 
        # os.environ["PYTHONPATH"] = "/home/pageot/sources/modspa2/Code/models/main/:$PYTHONPATH"
        if "LAI" in name_run :
                os.system('python /home/yann/sources/modspa2_LAI/modspa2/Code/models/main/runSAMIR.py -wd '+d['SAMIR_run']+' -dd '+d['SAMIR_run']+'/Inputdata/ -m /*/meteo.df -n /*/LAI'+str(y)+'.df -fcover /*/FCOVER.df -fc /*/FC.df -wp /*/WP.df  --fc_input  -o Output/'+optimis_val+'/output_test -p param_modif.csv  -optim test_optim.csv --cal ET ')
        else:
            if "Fcover" in name_run :
                os.system('python /home/yann/sources/modspa2/Code/models/main/runSAMIR.py -wd '+d['SAMIR_run']+' -dd '+d['SAMIR_run']+'/Inputdata/ -m /*/meteo.df -n /*/NDVI'+str(y)+'.df -fcover /*/FCOVER.df -fc /*/FC.df -wp /*/WP.df  --fc_input  -o Output/'+optimis_val+'/output_test -p param_modif.csv  -optim test_optim.csv --cal ET ')
            else:
                os.system('python /home/yann/sources/modspa2/Code/models/main/runSAMIR.py -wd '+d['SAMIR_run']+' -dd '+d['SAMIR_run']+'/Inputdata/ -m /*/meteo.df -n /*/NDVI'+str(y)+'.df  -fc /*/FC.df -wp /*/WP.df  -o Output/'+optimis_val+'/output_test -p param_modif.csv  -optim test_optim.csv --cal ET')
        for classe in classes:
            if len(optimis_val) > 5:
                param=pd.read_csv(d["SAMIR_run"]+"Output/"+optimis_val+"/output_test_"+classe+"_param.txt",header=None,skiprows=2,sep=";")
                param.set_index(0,inplace=True)
            else:
                param=pd.read_csv(d["SAMIR_run"]+"Output/"+optimis_val+"/output_test_"+classe+"_param.txt",header=None,skiprows=1,sep=";")
                param.set_index(0,inplace=True)
            #  Récuparation data_validation ETR
            if classe =='maize_irri':
                ETR_lam=pd.read_csv("/mnt/d/THESE_TMP/DATA_ETR_CESBIO/DATA_ETR_LAM/DATA_ETR_LAM_ICOS/ETR_LAM"+str(y)+".csv",decimal='.')
                ETR_lam.date=pd.to_datetime(ETR_lam["date"],format='%Y-%m-%d')
            else:
                ETR_gri=pd.read_csv("/mnt/d/THESE_TMP/DATA_ETR_CESBIO/DATA_ETR_GRIGNON/DATA_ETR_GRIGNON_ICOS/ETR_GRIGNON"+str(y)+".csv",decimal='.')
                ETR_gri.date=pd.to_datetime(ETR_gri["date"],format='%Y-%m-%d')
                # Récupération des output de la simulation 
            for run in os.listdir(d["SAMIR_run"]+"Output/"+optimis_val+"/"):
                print(run)
                if classe in run and "txt" not in run:
                    num_run=run[23:]
                    a=open(d["SAMIR_run"]+"Output/"+optimis_val+"/"+run,"rb")
                    output_sim=pickle.load(a)
                    ETRmod=output_sim[["ET","date"]]
                    a.close()
                    #  Récuper le couple de paramètre que je vais varier
                    if len(optimis_val) < 6:
                        parametre=param.iloc[int(num_run)]
                        print (parametre)
                        parametre1=parametre[1]
                        # print (r'para %s;'%(parametre1)) 
                    else: 
                        parametre=param.iloc[int(num_run)]
                        parametre1=parametre[1]
                        parametre2=parametre[2]
                        # print (r'para %s; para2 %s'%(parametre1,parametre2))    
                    # localiser les nan dans ETR, les supprimer ainsi que les dates pour ensuite comparer 
                    if classe =='maize_irri':
                        # dfETR=pd.concat([ETR_lam,ETRmod],axis=1)
                        dfETR=pd.merge(ETR_lam,ETRmod,on=["date"])
                    else:
                        # dfETR=pd.concat([ETR_gri,ETRmod],axis=1)
                        dfETR=pd.merge(ETR_gri,ETRmod,on=["date"])
                    dfETR.columns=["date",'LE','ET']
                    dfETR_day=dfETR.set_index('date').resample("D").asfreq()
                    dfETR=dfETR.set_index('date').resample("W").asfreq()
                    # dfETR.to_csv(d["SAMIR_run"]+"Output/"+optimis_val+"/CSV/ETR_%s_%s.csv"%(classe,num_run))
                    dfETR.dropna(inplace=True)
                    dfETR_day.dropna(inplace=True)
                    if dfETR.shape[0]==0:
                        print("%s non utilisable " %y) # pas de date similaire entre modélisation et ETRobs
                        continue
                    else:
                        slope, intercept, r_value, p_value, std_err = stats.linregress(dfETR.LE.to_list(),dfETR.ET.to_list())
                        bias=1/dfETR_day.shape[0]*sum(np.mean(dfETR.ET)-dfETR.LE) 
                        fitLine = predict(dfETR.LE)
                    #     plt.figure(figsize=(7,7))
                    #     plt.plot([0.0, 10], [0.0,10], 'r-', lw=2)
                    #     plt.plot(dfETR.LE,fitLine,linestyle="-")
                    #     plt.scatter(dfETR.LE,dfETR.ET,c=select_color_date(dfETR),s=9)
                    #     plt.legend(('bare_soil', 'Vege'))
                    #     plt.xlabel("ETR OBS")
                    #     plt.ylabel("ETR model")
                    #     plt.xlim(0,10)
                    #     plt.ylim(0,10)
                        rms = mean_squared_error(dfETR.LE,dfETR.ET,squared=False)
                    #     plt.text(8,min(dfETR.ET)+0.1,"RMSE = "+str(round(rms,2)))
                    #     plt.text(8,min(dfETR.ET)+0.3,"R² = "+str(round(r_value,2)))
                    #     plt.text(8,min(dfETR.ET)+0.5,"Pente = "+str(round(slope,2)))
                    #     plt.text(8,min(dfETR.ET)+0.7,"Biais = "+str(round(bias,2)))
                    #     plt.savefig(d["SAMIR_run"]+"Output/"+optimis_val+"/Plot/plt_scatter_ETR_%s_%s_%s.png"%(classe,optimis_val,str(int(parametre1))))
                    #     plt.figure(figsize=(7,7))
                    #     plt.plot(dfETR.index,dfETR.LE,label='ETR_obs',color="black")
                    #     plt.plot(dfETR.index,dfETR.ET,label='ETR_mod',color='red')
                    #     plt.ylabel("ETR")
                    #     plt.ylim(0,10)
                    #     plt.legend()
                    #     plt.savefig(d["SAMIR_run"]+"Output/"+optimis_val+"/Plot/Plot_dyna/plt_dynamique_ETR_%s_%s_%s.png"%(classe,optimis_val,str(int(parametre1))))
                        if len(optimis_val) < 6:
                            result.append([num_run,parametre1,rms,bias,r_value,y,classe])
                        else: 
                            result.append([num_run,parametre1,parametre2,rms,bias,r_value,y,classe])
            if len(optimis_val) < 6:
                resultat=pd.DataFrame(result,columns=["Num_run","Param1","RMSE",'bias','R','years','OS'])
            else:
                resultat=pd.DataFrame(result,columns=["Num_run","Param1","Param2","RMSE",'bias','R','years','OS'])
            resultat.to_csv(d["SAMIR_run"][:-5]+"param_RMSE%s.csv"%(optimis_val))
    plt.figure(figsize=(7,7))
    for years in ["2006","2008","2010","2012","2014","2015"]:#       
        # all_min=[]
        df=pd.read_csv(d["SAMIR_run"][:-5]+"param_RMSE%s.csv"%optimis_val)
        class_OS=df.groupby("OS")
        for Os in classes:
            # if len(optimis_val) < 6:
                data=class_OS.get_group(Os)
                a=data.groupby("years")
                b=a.get_group(int(years))
                b.sort_values("Param1",ascending=True,inplace=True)
                minval=b.loc[b["RMSE"].idxmin()]
                print(minval)
                # all_min.append(minval)
                x=[]
                y=[]
                # for p in sorted(list(set(b.Param1))):
                #     test=b[b.Param1==p]
                #     a=b[b.Param1==p]["RMSE"].idxmin()
                #     x.append(b.loc[a].Param1)
                #     y.append(b.loc[a].RMSE)
                # plt.plot(x,y,label=str(years))
                plt.plot(b.Param1,b.RMSE,label=str(years))
                plt.plot(minval.Param1,minval.RMSE,marker="*",color="Black")
                plt.text(minval.Param1,minval.RMSE,s="min: %s"%(minval.Param1))
                plt.legend()
                plt.xlabel(optimis_val)
                plt.title(optimis_val)
                plt.ylabel("RMSE ETRobs/ETRmod")
        plt.savefig(d["SAMIR_run"][:-5]+"min_value_optimi_%s.png"%(optimis_val))
# test plot
    # x=[]
    # y=[]
    # c=[]
    # minva=pd.DataFrame()
    # plt.figure(figsize=(7,7))
    # for p in sorted(list(set(b.Param1))):
    #     test=b[b.Param1==p]
    #     minval=test.loc[test["RMSE"].idxmin()]
    #     a=b[b.Param1==p]["RMSE"].idxmin()
    #     x.append(b.loc[a].Param1)
    #     y.append(b.loc[a].RMSE)
    #     c.append(b.loc[a].Param2)
    #     minva=minva.append(minval)
    # plt.plot(x,y)
    # plt.plot(minva.Param1,minva.RMSE,marker="*",color="Black")
    # plt.text(minva.Param1,minva.RMSE,s=c)
        
        
        
        # minall=pd.DataFrame(all_min)
        # minall["Param1"].mean()
    #     # minall.to_csv( d["Output_PC_home"]+"Sans_Spin/min_value_Zrmax"+name_run[-10:]+".csv")
    # df=pd.read_csv(d["SAMIR_run"][:-5]+"param_RMSE%s.csv"%optimis_val)
    # for y in ["2006","2008","2010",'2012','2014','2019']:
    #     plt.figure(figsize=(10,10))
    #     g=test.groupby("years")
    #     test1=g.get_group(int(y))
    #     test1=test1[["Param1","Param2","RMSE"]]
    #     test1.columns=["REW","zrmax",'RMSE']
    #     test2=test1.pivot("REW",'zrmax','RMSE')
    #     sns.heatmap(test2,annot=True)
    #     plt.title(y)
    #     plt.savefig("D:/THESE_TMP/TRAITEMENT/RUNS_SAMIR/RUN_MULTI_SITE_ICOS/RUN_OPTIMISATION_ICOS/bin/OPTI_ICOS_MULTI_SITE_pluvio_stat_two_param/plot_test"+y+".png")
    #     # nump_data=np.array(test2)
    #     # plt.imshow(nump_data)
        # plt.colorbar()

