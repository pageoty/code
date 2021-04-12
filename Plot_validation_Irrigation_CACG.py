# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 09:50:47 2021

@author: yann 

Validation Irrigation automatique parcelle de référence
"""


import os
import sqlite3
import geopandas as geo 
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
import csv
import numpy as np
import pandas as pd
import seaborn as sns 
import TEST_ANALYSE_SIGNATURE
import shapely.geometry as geom
import descartes
import pickle
from datetime import datetime, date, time, timezone
from scipy import *
from scipy import stats
from pylab import *
from sklearn.metrics import *
from sklearn.linear_model import LinearRegression



def predict(x):
   return slope * x + intercept


if __name__ == '__main__':
    d={}
    name_run="RUNS_SAMIR/RUN_CACG/CACG_init_ru_optim_fewi_De_Kr_irri_auto_soil/"
    # name_run_save_fig="RUNS_SAMIR/RUN_MULTI_SITE_ICOS/RUN_OPTIMISATION_ICOS/SAMIR_OPTIMI_LAM/RUN_final/init_ru/Merlin_init_ru_100_optim_fewi_De_Kr_irri_auto_soil"
    # d["PC_disk"]="/run/media/pageot/Transcend/Yann_THESE/BESOIN_EAU/BESOIN_EAU/"
    d["PC_home"]="/mnt/d/THESE_TMP/"
    d["PC_home_Wind"]="D:/THESE_TMP/"
    d["PC_disk"]="H:/Yann_THESE/BESOIN_EAU/BESOIN_EAU/"
    d["PC_labo"]="/datalocal/vboxshare/THESE/BESOIN_EAU/"
    label="Init ru année n-1 + Irrigation auto"
    years="2017"
    lc="maize_irri"
    
    
# =============================================================================
#  Validation Irri_ préparation data 
# =============================================================================
    # all_quantity=[]
    # all_number=[]
    # all_id=[]
    # all_runs=[]
    # all_date=[]
    # all_date_jj=[]
    # for r in os.listdir('D:/THESE_TMP/RUNS_SAMIR/RUN_RELATION/TARN_CACG/'):
    #     d["path_PC"]='D:/THESE_TMP/RUNS_SAMIR/RUN_RELATION/TARN_CACG/'+str(r)+'/Inputdata/'
    #     for s in os.listdir(d["path_PC"][:-10]):
    #         if "output" in s:
    #             print (s)
    #             df=pickle.load(open(d["path_PC"][:-10]+str(s),'rb'))
    #             for id in list(set(df.id)):
    #                 lam=df.loc[df.id==id]
    #                 date_irr=lam.loc[lam.Ir_auto!=0.0]["date"]
    #                 size_R=len(date_irr) # vecteur de répétiton de la variable RUNS
    #                 print(r'n° du runs : %s '% r)
    #                 print(r' n° parcelle : %s' %id)
    #                 print(r'sum irrigation in mm : %s'%lam.groupby(["LC","id"])["Ir_auto"].sum()[0])
    #                 print(r' nb irrigation : %s' %lam.Ir_auto.where(df["Ir_auto"] != 0.0).dropna().count())
    #                 all_runs.append(r)
    #                 all_id.append(id)
    #                 all_quantity.append(lam.groupby(["LC","id"])["Ir_auto"].sum()[0])
    #                 all_number.append(lam.Ir_auto.where(df["Ir_auto"] != 0.0).dropna().count())
    #                 all_date.append(date_irr.values)
    #                 for i in date_irr:
    #                     a=i.strftime("%j") # date en jj sur l'année
    #                     all_date_jj.append(a)
    # all_resu=pd.DataFrame([all_runs,all_id,all_quantity,all_number,all_date]).T
    # all_resu.columns=["runs",'id','cumul_irr',"nb_irr","date_irr"]

# =============================================================================
# Validation des Irr cumulées CACG
# =============================================================================
    vali_cacg=pd.read_csv(d["PC_disk"]+"TRAITEMENT/DATA_VALIDATION/DATA_VOL_IRRIGATION/DATE_DOES_CACG_2017.csv",encoding='latin-1',decimal=',',sep=';',na_values="nan")
    vali_cacg.Date_irrigation=pd.to_datetime(vali_cacg.Date_irrigation,format='%d/%m/%Y')
    vali_cacg["Quantite"].astype(float)
    sum_irr_cacg_val=vali_cacg.groupby("ID")["Quantite"].sum()
    nb_irr=vali_cacg.groupby("ID")["Date_irrigation"].count()
    a=vali_cacg.groupby("ID")
    
    d["Output_model_PC_home_disk"]="H:/Yann_THESE/BESOIN_EAU/BESOIN_EAU/TRAITEMENT/RUNS_SAMIR/RUN_CACG/CACG_init_ru_optim_fewi_De_Kr_irri_auto_soil"
    Irri_mod=pd.read_csv(d["Output_model_PC_home_disk"]+"/LUT_2017.csv",index_col=[0,1],skipinitialspace=True)
    gro=Irri_mod.groupby("ID")
    
    NDVI=pickle.load(open(d["Output_model_PC_home_disk"]+"/2017/Inputdata/maize_irri/NDVI2017.df","rb"))
    NDVI=NDVI.loc[(NDVI.date >= str(years)+'-04-01')&(NDVI.date<=str(years)+"-09-30")]
    # inser loop parcelle Id
    
    for p in list(set(vali_cacg["ID"])):
        par1=gro.get_group(p)
        par1.reset_index(inplace=True)
        par1.num_run=pd.to_datetime(par1.num_run,format="%Y-%m-%d")
        mean_run=par1[["num_run","4"]]
        maxIr=par1[["num_run","8"]]
        minIr=par1[["num_run","0"]]
        mean_run.columns=["date","maxZr_1000"]
        maxIr.columns=["date","maxZr_1200"]
        minIr.columns=["date","maxZr_800"]
        maxIr.replace(0.0,pd.NaT,inplace=True)
        minIr.replace(0.0,pd.NaT,inplace=True)
        # print(mean_run.loc[mean_run['maxZr_1000']!=0.0])
        
        # validation
        par1_val=a.get_group(p)
        par1_val=par1_val[["Date_irrigation",'Quantite']]
        par1_val.set_index("Date_irrigation",inplace=True)
        par1_val_res=par1_val.resample("D").asfreq()
        par1_val_res.fillna(0.0,inplace=True)
        par1_val_res["date"]=par1_val_res.index
        all_res=pd.merge(par1_val_res,mean_run,on=["date"]) # fusion des sim/obs
        all_resu=all_res.replace(0.0,pd.NaT)
        print("============")
        print("parcelle :%s"%p)
        print(all_res.sum())
        print("============")
        # plot
        plt.figure(figsize=(7,7))
        plt.title(p)
        plt.plot(all_resu.date,all_resu.maxZr_1000,marker="x",linestyle="",label="Simulée")
        # plt.plot(minIr.date,minIr.maxZr_800,marker="x",linestyle="",label="Simulée_min",alpha=0.5)
        # plt.plot(maxIr.date,maxIr.maxZr_1200,marker="x",linestyle="",label="Simulée_max",alpha=0.5)
        plt.plot(all_resu.date,all_resu.Quantite,marker="o",linestyle="",label="Observée")
        plt.ylim(0,50)
        plt.ylabel("Irrigation en mm")
        plt.legend()
        ax2=plt.twinx(ax=None)
        ax2.plot(NDVI.loc[NDVI.id==p].date,NDVI.loc[NDVI.id==p].NDVI,color="darkgreen",linestyle="--")
        ax2.set_ylabel("NDVI")
        ax2.set_ylim(0,1)
        
# =============================================================================
# Volumes annuels
# =============================================================================
    

