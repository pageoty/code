# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 09:37:01 2020

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
    
def sse(params, args):
    #data, psr = args[0], args[1]
    data, psr = args['dn'], args['r']
    betavar, rho, zeta = params[0], params[1], params[2]
    return sum((data - (rho + zeta*np.exp(betavar*psr))**-1)**2)

if __name__ == "__main__":

    
    d={}
    d['SAMIR_run']="/mnt/d/THESE_TMP/RUNS_SAMIR/RUN_TEST_opi/"
    d['SAMIR_run_Wind']="D:/THESE_TMP/RUNS_SAMIR/RUN_TEST_opi/"
    d["PC_disk_Wind"]="D:/THESE_TMP/RUNS_SAMIR/DATA_Validation/"
    d['PC_disk_unix']="/mnt/d/THESE_TMP/RUNS_SAMIR/DATA_Validation/"
# =============================================================================
#     Données de validation 
# =============================================================================
    vali_cacg=pd.read_csv(d["PC_disk_unix"]+"merge_parcelle_2017.csv")
#   vali_cacg.dropna(subset=['id'],inplace=True)
    vali_cacg.Date_irrigation=pd.to_datetime(vali_cacg.Date_irrigation,format='%d/%m/%y')
    vali_cacg["Quantity(mm)"].astype(float)
    sum_irr_cacg_val=vali_cacg.groupby("id")["Quantity(mm)"].sum()
    # nb_irr=vali_cacg.groupby("id")["Date_irrigation"].count()

# =============================================================================
#   Lancer SAMIR avec Paramétrage initaux 
# =============================================================================
    REW = "-8 "
    Zr_max="500" 
    A_kcb = "1.63" 
    # param=pd.read_csv("D:/THESE_TMP/RUNS_SAMIR/RUN_TEST_opi/Inputdata/param_SAMIR12_13.csv",delimiter=",",header=None)
    param=pd.read_csv("/mnt/d/THESE_TMP/RUNS_SAMIR/RUN_TEST_opi/Inputdata/param_SAMIR12_13.csv",delimiter=",",header=None)
    param.loc[6,13]=A_kcb # ligne 6 , colonne 13
    param.loc[6,20]=REW
    param.loc[6,23]=Zr_max
    
    param.to_csv(d["SAMIR_run"]+"/Inputdata/param_otpi_T1.csv",header=False,sep= ',',index=False,na_rep="")
    # param_op=pd.read_csv("D:/THESE_TMP/RUNS_SAMIR/RUN_TEST_opi/Inputdata/param_otpi_T1.csv",delimiter=";",header=None)

    #  Lancement du code
    os.environ["PYTHONPATH"] = "/mnt/c/users/Yann\ Pageot/Documents/code/modspa/modspa2/code/models/:$PYTHONPATH      "
    os.system('python /mnt/c/users/Yann\ Pageot/Documents/code/modspa/modspa2/code/models/main/runSAMIR.py -wd /mnt/d/THESE_TMP/RUNS_SAMIR/RUN_TEST_opi -dd /mnt/d/THESE_TMP/RUNS_SAMIR/RUN_TEST_opi/Inputdata/ -m meteo.df -n maize/NDVI.df -fc maize/FC.df -wp maize/WP.df -o output_T1.df -p param_otpi_T1.csv')
    
    # os.environ["PYTHONPATH"] = "C:/Users/Yann\ Pageot/Documents/code/modspa/modspa2/code/models/:$PYTHONPATH      "
    # os.system('python C:/Users/Yann\ Pageot/Documents/code/modspa/modspa2/code/models/main/runSAMIR.py -wd D:/THESE_TMP/RUNS_SAMIR/RUN_TEST_opi -dd D:/THESE_TMP/RUNS_SAMIR/RUN_TEST_opi/Inputdata/ -m meteo.df -n maize/NDVI.df -fc maize/FC.df -wp maize/WP.df -o output_T1.df -p param_otpi_T1.csv')
    
    #  Récupération des output de la simulation 
    output_sim=pickle.load(open(d["SAMIR_run"]+"/output_T1.df","rb"))
    all_quantity=[]
    all_number=[]
    all_id=[]
    all_RMSE=[]
    for id in list(set(output_sim.id)):
        lam=output_sim.loc[output_sim.id==id]
        # print(r' n° parcelle : %s' %id)
        # print(r'sum irrigation in mm : %s'%lam.groupby(["LC","id"])["Ir_auto"].sum()[0])
        # print(r' nb irrigation : %s' %lam.Ir_auto.where(output_sim["Ir_auto"] != 0.0).dropna().count())
        all_id.append(id)
        all_quantity.append(lam.groupby(["LC","id"])["Ir_auto"].sum()[0])
        all_number.append(lam.Ir_auto.where(output_sim["Ir_auto"] != 0.0).dropna().count())
    all_resu=pd.DataFrame([all_id,all_quantity,all_number]).T
    all_resu.columns=['id','cumul_irr',"nb_irr"]
    
    x_data=sum_irr_cacg_val
    y_data=all_resu.cumul_irr

    x0=[REW,Zr_max,A_kcb]
    
    Evaluation= minimize(RMSE, x0, args=(x_data, y_data),options={'xatol': 1e-10, 'disp': True})
    print(Evaluation)
    print(Evaluation.fun)
# =============================================================================
#  Optimisation des paramètres
# =============================================================================
#     while Evaluation.fun >= 70 :
#         print("continued")
#         # Paramètre initaux
#         # REW = random.uniform(-400,200) # stocker résultat du run précédnt pour converser 
#         # Zr_max=random.uniform(150,1500) 
#         # A_kcb =random.uniform(1,2)
        
        

#         # REW = Evaluation.x[0] *random.uniform(-10,10)
#         # if REW <=-400 and  REW >= 200:
#         #     REW = Evaluation.x[0] *random.uniform(-400,200)
#         # Zr_max=Evaluation.x[1] *random.uniform(150,1500) #Mais add contition car ne peut pas être < 150
#         # if Zr_max >= 1500 :
#         #     Zr_max=Evaluation.x[1] *random.uniform(150,1500)
#         # A_kcb = Evaluation.x[2] *random.uniform(1,2)
#         # if A_kcb <=0.5 and A_kcb >= 2.5:
#         #     A_kcb = Evaluation.x[2] *random.uniform(1,2)
        
#         x0=[REW,Zr_max,A_kcb]
#         print(r"========")
#         print(x0)
#         print(r"========")
#         param=pd.read_csv("/mnt/d/THESE_TMP/RUNS_SAMIR/RUN_TEST_opi/Inputdata/param_SAMIR12_13.csv",delimiter=",",header=None)
#         param.loc[6,13]=A_kcb # ligne 6 , colonne 13
#         param.loc[6,20]=REW
#         param.loc[6,23]=Zr_max
        
#         param.to_csv(d["SAMIR_run"]+"/Inputdata/param_otpi_T1.csv",header=False,sep= ',',index=False,na_rep="")
#         # param_op=pd.read_csv("D:/THESE_TMP/RUNS_SAMIR/RUN_TEST_opi/Inputdata/param_otpi_T1.csv",delimiter=";",header=None)
    
#         #  Lancement du code
#         os.environ["PYTHONPATH"] = "/mnt/c/users/Yann\ Pageot/Documents/code/modspa/modspa2/code/models/:$PYTHONPATH      "
#         os.system('python /mnt/c/users/Yann\ Pageot/Documents/code/modspa/modspa2/code/models/main/runSAMIR.py -wd /mnt/d/THESE_TMP/RUNS_SAMIR/RUN_TEST_opi -dd /mnt/d/THESE_TMP/RUNS_SAMIR/RUN_TEST_opi/Inputdata/ -m meteo.df -n maize/NDVI.df -fc maize/FC.df -wp maize/WP.df -o output_T1.df -p param_otpi_T1.csv')
        
#         #  Récupération des output de la simulation 
#         output_sim=pickle.load(open(d["SAMIR_run"]+"/output_T1.df","rb"))
#         all_quantity=[]
#         all_number=[]
#         all_id=[]
#         for id in list(set(output_sim.id)):
#             lam=output_sim.loc[output_sim.id==id]
#             all_id.append(id)
#             all_quantity.append(lam.groupby(["LC","id"])["Ir_auto"].sum()[0])
#             all_number.append(lam.Ir_auto.where(output_sim["Ir_auto"] != 0.0).dropna().count())
#         all_resu=pd.DataFrame([all_id,all_quantity,all_number]).T
#         all_resu.columns=['id','cumul_irr',"nb_irr"]
        
        
#         # Evaluation du run  à partir des paramétres
#         x_data=sum_irr_cacg_val
#         y_data=all_resu.cumul_irr
#         x0=[REW,Zr_max,A_kcb]
#         print(r"========")
#         print(x0)
#         print(r"========")
#         Evaluation= minimize(RMSE, x0, args=(x_data, y_data),method='nelder-mead',options={'xatol': 1, 'disp': True})
#         print(Evaluation)
#         print(Evaluation.fun)
#         # x0=[REW,Zr_max,A_kcb]
#         all_RMSE.append(Evaluation.fun)
        
#     OPTI=pd.DataFrame(all_RMSE)
#     OPTI.to_csv("/mnt/d/THESE_TMP/RUNS_SAMIR/resu_opt_RMSE.csv")
#     a=pd.read_csv("/mnt/d/THESE_TMP/RUNS_SAMIR/resu_opt_RMSE.csv")
#     plt.plot(a["0"])
#     # a=pd.read_csv("D:/THESE_TMP/RUNS_SAMIR/resu_opt_RMSE.csv")
    
#         # all_RMSE.to_csv("/mnt/d/THESE_TMP/RUNS_SAMIR/RMSE_opti_.csv")
#         # test= minimize(RMSE, x0, args=(x_data, y_data),method='nelder-mead')
#         # testfmin = fmin(RMSE, x0, args=(x_data, y_data), xtol=1e-8, disp=True,)


# # =============================================================================
# # Test compréhension
# # =============================================================================
    
    def fun(x, a,b,c):
       return (a*x**2 + b*x + c).sum()
    x0=[10,0,0]
    additional=(np.array([3,5,15]), 10, 10)
    t=optimize.minimize(fun,x0, args=additional)
    t

    def sse(params, args):
       #data, psr = args[0], args[1]
       data, psr = args['dn'], args['r']
       betavar, rho, zeta = params[0], params[1], params[2]
       return sum((data - (rho + zeta*np.exp(betavar*psr))**-1)**2)


    past_sample_range = [0, 1,2]
    data_known = [0.98, 0.98, 1]
    data_predic= [1 ,0.5,0.25]
    additional =[data_known,data_predic]
    # additional = {'dn': data_known, 'r':range(len(past_sample_range))}
    res = minimize(fun=sse, args=additional, x0=np.array([0, 0]),method='nelder-mead')# bounds = ((0, None), (0, None), (0, None)))
    print(res)