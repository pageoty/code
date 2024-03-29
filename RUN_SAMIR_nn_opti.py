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
# import descartes
from datetime import datetime, date, time, timezone
from scipy import *
from scipy import stats
from pylab import *
from sklearn.linear_model import LinearRegression
from RUN_SAMIR_opti import params_update
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
 
def RMSE(x,*args) :
    x_data = args[0]
    y_data = args[1]
    rmse= mean_squared_error(x_data,y_data,squared=False)
    return rmse
    
def test(X):
    objective = lambda b: np.sqrt(np.mean((X, b)**2))
    return objective

if __name__ == "__main__":
    result=[]
    for y in ["2008","2010","2012","2014","2015","2019"]: #"2008","2010","2012","2014","2015","2017","2019"
        print (y)
        meteo='SAFRAN'
        name_run="test_kr"
        d={}
        # d['SAMIR_run']="/mnt/d/THESE_TMP/TRAITEMENT/"+name_run+"/"+str(y)+"/"
        d['SAMIR_run']="/run/media/pageot/Transcend/Yann_THESE/BESOIN_EAU/BESOIN_EAU/TRAITEMENT/"+name_run+"/"+str(y)+"/"
        d['SAMIR_run_Wind']="D:/THESE_TMP/"+name_run+"/"+str(y)+"/"
        d["PC_disk_Wind"]="D:/THESE_TMP/RUNS_SAMIR/DATA_Validation/"
        d['PC_disk_unix']="/mnt/d/THESE_TMP/"
        d["PC_labo_short"]="/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/RUNS_SAMIR/RUN_MULTI_SITE_ICOS/RUN_OPTIMISATION_ICOS/SAMIR_test_incertitude/"
        d["PC_labo"]=d["PC_labo_short"]+name_run+"/"+str(y)+"/"#"/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/RUNS_SAMIR/RUNS_optim_LUT_LAM_ETR/"+name_run+"/"+str(y)+"/"
        
        if name_run not in os.listdir(d["PC_labo_short"]):
            os.mkdir('%s/%s/'%(d["PC_labo_short"],name_run))
        else: 
            print('existing folder')
        if str(y) not in os.listdir(d["PC_labo"][:-5]):
            os.mkdir ('%s/%s/%s'%(d["PC_labo_short"],name_run,str(y)))
        else: 
            print('existing folder')
          # Déplacement all file 
        if meteo=="SAFRAN":
            os.system("cp -r /datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/RUNS_SAMIR/DATA_SCP_ICOS/SAFRAN/* %s"%(d['PC_labo'][:-5]))
        else:
            os.system("cp -r /datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/RUNS_SAMIR/DATA_SCP_ICOS/ICOS_STAT/* %s"%(d['PC_labo'][:-5]))
                
        if "Output" in os.listdir(d['PC_labo']):
            print ("existing file")
        else :
            os.mkdir ("%s/Output"%d['PC_labo'])
        
        #### Calcule du REW via les focntions de pédotransfert d'Allen 2005 pour lam

        # Sand = mean([0.471,0.639,0.64,0.829]) # ensemble de la colonne sol 
        # Clay = mean([0.50,0.48])
        
        Sand = np.mean([0.471,0.646]) # ensemble de la colonne sol 
        Clay = 0.5026
        if Sand >= 0.80 :
            REW = 20 - 0.15 * Sand
        elif Clay >= 0.50 : 
            REW = 11 - 0.06 * Clay 
        elif (Sand < 0.80) and (Clay < 0.50):
            REW = 8 + 0.008 * Clay
        
        params_update(d['PC_labo']+"/Inputdata/param_SAMIR12_13.csv",
                      d['PC_labo']+"/Inputdata/param_modif.csv",date_start=str(y)+str('0101'),date_end=str(y)+str('1231'),Ze=150,minZr=150,REW=REW,maxZr=1500,Zsoil=2000,DiffE=0.000001,DiffR=0.000001,m=1,Irrig_auto=1,Irrig_man=0,Lame_max=30,Init_RU=1,KmaxKcb=1.15,Plateau=70,Fslope=1.39, Foffset=-0.25)
        params_update(d['PC_labo']+"/Inputdata/param_modif.csv",
                      d['PC_labo']+"/Inputdata/param_modif.csv",date_start=str(y)+str('0101'),date_end=str(y)+str('1231'),ligne_OS=7,Ze=150,REW=-10,minZr=150,maxZr=1500,A_kcb=1.358,Koffset=-0.017,Zsoil=3000,DiffE=10,DiffR=10,Irrig_auto=0,Irrig_man=1,Lame_max=0,Init_RU=1,KmaxKcb=1.15)

# =============================================================================
#     #  Lancement du code
# =============================================================================
        if "LAI" in name_run:
            os.system('python /home/yann/sources/modspa2_LAI/modspa2/Code/models/main/runSAMIR.py -wd /mnt/d/THESE_TMP/TRAITEMENT/'+name_run+'/'+str(y)+'/'' -dd /mnt/d/THESE_TMP/TRAITEMENT/'+name_run+'/'+str(y)+'/Inputdata/ -m /*/meteo.df -n /*/LAI'+str(y)+'.df -fc /*/FC.df -wp /*/WP.df -fcover /*/FCOVER.df --fc_input -o Output/output -p param_modif.csv')
        else:
            if "Fcover" not in name_run:
                # os.system('python /home/yann/sources/modspa_SAMIR/modspa/Code/models/main/runSAMIR.py -wd /mnt/d/THESE_TMP/TRAITEMENT/'+name_run+'/'+str(y)+'/'' -dd /mnt/d/THESE_TMP/TRAITEMENT/'+name_run+'/'+str(y)+'/Inputdata/ -m meteo.df -n NDVI'+str(y)+'.df -fc FC.df -wp WP.df -o Output/output.df -p param_modif.csv')
                os.system("python /home/pageot/sources/modspa_SAMIR/modspa/Code/models/main/runSAMIR.py -wd "+d['PC_labo']+" -dd /"+d['PC_labo']+"Inputdata/ -m meteo.df -n NDVI"+str(y)+".df -fc FC.df -wp WP.df -o Output/output.df -p param_modif.csv")
            else:
                os.system("python /home/pageot/sources/modspa_SAMIR/modspa/Code/models/main/runSAMIR.py -wd "+d['PC_labo']+" -dd /"+d['PC_labo']+"Inputdata/ -m meteo.df -n NDVI"+str(y)+".df -fc FC.df -wp WP.df -fcover FCOVER.df --fc_input -o Output/output.df -p param_modif.csv")
 # allows create file via call system
        # os.system('python /home/pageot/sources/modspa2/Code/models/main/runSAMIR.py -wd /datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/'+name_run+'/'+str(y)+'/'' -dd /datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/'+name_run+'/'+str(y)+'/Inputdata/ -m /*/meteo.df -n /*/NDVI'+str(y)+'.df  -fc /*/FC.df -wp /*/WP.df  -o Output/output.df -p param_modif.csv ')
        # os.system('python /home/pageot/sources/modspa2/Code/models/main/runSAMIR.py -wd /datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/'+name_run+'/'+str(y)+'/'' -dd /datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/'+name_run+'/'+str(y)+'/Inputdata/ -m /*/meteo.df -n /*/NDVI'+str(y)+'.df  -fc /*/FC.df -wp /*/WP.df -fcover /*/FCOVER.df --fc_input  -o Output/output.df -p param_modif.csv ')
        # Sans le FCOVER sat
        # os.system('python /home/pageot/sources/modspa2/Code/models/main/runSAMIR.py -wd /datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/'+name_run+'/'+str(y)+'/'' -dd /datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/'+name_run+'/'+str(y)+'/Inputdata/ -m meteo.df -n */NDVI'+str(y)+'.df  -fc */FC.df -wp */WP.df  -o Output/output.df -p param_modif.csv ')
       
        #  Récupération des output de la simulation 
        # output_sim=pickle.load(open(d["PC_labo"]+"/output_T1.df","rb"))
        # all_quantity=[]
        # all_number=[]
        # all_id=[]
        # all_ETR=[]
        # for id in list(set(output_sim.id)):
        #     lam=output_sim.loc[output_sim.id==id]
        #     # print(r' n° parcelle : %s' %id)
        #     # print(r'sum irrigation in mm : %s'%lam.groupby(["LC","id"])["Ir_auto"].sum()[0])
        #     # print(r' nb irrigation : %s' %lam.Ir_auto.where(output_sim["Ir_auto"] != 0.0).dropna().count())
        #     if id == 1:
        #         ETRmod=lam[["ET","date"]]
        #     all_id.append(id)
        #     all_quantity.append(lam.groupby(["LC","id"])["Ir_auto"].sum()[0])
        #     all_number.append(lam.Ir_auto.where(output_sim["Ir_auto"] != 0.0).dropna().count())
        # all_resu=pd.DataFrame([all_id,all_quantity,all_number]).T
        # all_resu.columns=['id','cumul_irr',"nb_irr"]
# =============================================================================
#     validation ETR 
# =============================================================================
# FONCTION WITH Years
        # ETR_lam_2019=pd.read_csv("D:/THESE_TMP/RUNS_SAMIR/DATA_Validation/ETR_LAM2017.csv")
        # ETR_lam_2019=pd.read_csv("/mnt/g/Yann_THESE/BESOIN_EAU/Calibration_SAMIR/DONNEES_CALIBRATION/DATA_ETR_LAM/ETR_LAM"+str(y)+".csv")
        # # ETR_lam_2019=pd.read_csv("D:/THESE_TMP/RUNS_SAMIR/DATA_Validation/ETR_LAM2019.csv",decimal='.',na_filter=True)
        # ETR_lam_2019["date"]=pd.to_datetime( ETR_lam_2019["TIMESTAMP"],format="%Y-%m-%d")
        # # fusion mod et obs dans dataframe
        # ETR_resu=pd.concat([ETR_lam_2019["LE"].iloc[2:-4],ETRmod.ET],axis=1)
        # ETR_resu["date"]=ETRmod.date
        # ETR_resu_ss_nn=ETR_resu.dropna()
        # val=mean_squared_error(ETR_resu_ss_nn['LE'],ETR_resu_ss_nn["ET"],squared=False)
        # print(r'========')
        # print(val)
        # print(r'========')
        # slope, intercept, r_value, p_value, std_err = stats.linregress(ETR_resu_ss_nn['LE'],ETR_resu_ss_nn["ET"])
        # bias=1/ETR_resu_ss_nn['LE'].shape[0]*sum(np.mean(ETR_resu_ss_nn["ET"])-ETR_resu_ss_nn['LE']) 
        # # fitLine = predict(ETR_lam_2019["LE"].iloc[2:-4])
        # plt.figure(figsize=(7,7))
        # plt.plot([0.0, 7], [0.0, 7], 'r-', lw=2)
        # # plt.plot(ETR_lam_2019["LE"].iloc[2:-4],fitLine,linestyle="-")
        # plt.scatter(ETR_resu_ss_nn['LE'],ETR_resu_ss_nn["ET"])
        # plt.plot()
        # plt.xlabel("ETRr OBS")
        # plt.ylabel("ETR model")
        # # plt.xlim(0,300)
        # # plt.ylim(0,300)
        # print(NS(ETR_resu_ss_nn['LE'],ETR_resu_ss_nn["ET"])) 
        # plt.text(5,min(ETR_resu_ss_nn["ET"])+0.1,"RMSE = "+str(round(val,2)))
        # plt.text(5,min(ETR_resu_ss_nn["ET"])+0.3,"R² = "+str(round(r_value,2)))
        # plt.text(5,min(ETR_resu_ss_nn["ET"])+0.5,"Pente = "+str(round(slope,2)))
        # plt.text(5,min(ETR_resu_ss_nn["ET"])+0.7,"Biais = "+str(round(bias,2)))
        # plt.text(5,min(ETR_resu_ss_nn["ET"])+0.9,"Nash = "+str(round(NS(ETR_resu_ss_nn['LE'],ETR_resu_ss_nn["ET"]),2)))
        # # plt.savefig(d["SAMIR_run"]+"/plot_ETRobs_ETR_mod.png")
