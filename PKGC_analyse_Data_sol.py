# -*- coding: utf-8 -*-
"""
@author: yann 

Validation Irrigation automatique parcelle de référence PKGC
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
    # d["PC_disk"]="/run/media/pageot/Transcend/Yann_THESE/BESOIN_EAU/BESOIN_EAU/"
    d["PC_home"]="/mnt/d/THESE_TMP/"
    d["PC_home_Wind"]="D:/THESE_TMP/"
    d["PC_disk"]="H:/Yann_THESE/BESOIN_EAU/BESOIN_EAU/"

    d["PC_labo"]="/datalocal/vboxshare/THESE/BESOIN_EAU/"
    # label="Init ru année n-1 + Irrigation auto"
    years=["2017"]
    lc="maize_irri"

    param=pd.read_csv("H:/Yann_THESE/BESOIN_EAU/BESOIN_EAU/TRAITEMENT/RUNS_SAMIR/RUN_PKGC/PKGC_GSM_init_ru_optim_Fcover_fewi_De_Kr_days10_dose30_500_800_irri_auto_soil/2017/Output/maxZr/output_test_maize_irri_param.txt",header=None,skiprows=1,sep=";")
    dfGSM=pd.read_csv("H:/Yann_THESE/BESOIN_EAU/BESOIN_EAU/TRAITEMENT/RUNS_SAMIR/RUN_PKGC/PKGC_GSM_init_ru_optim_Fcover_fewi_De_Kr_days10_dose30_500_800_irri_auto_soil/Table_RMSE_parcelle_min.csv")
    dfRRP=pd.read_csv("H:/Yann_THESE/BESOIN_EAU/BESOIN_EAU/TRAITEMENT/RUNS_SAMIR/RUN_PKGC/PKGC_RRP_init_ru_optim_Fcover_fewi_De_Kr_days10_dose30_500_800_irri_auto_soil/Table_RMSE_parcelle_min.csv")
    dfUTS=pd.read_csv("H:/Yann_THESE/BESOIN_EAU/BESOIN_EAU/TRAITEMENT/RUNS_SAMIR/RUN_PKGC/PKGC_init_ru_optim_Fcover_fewi_De_Kr_days10_dose30_500_800_irri_auto_soil/Table_RMSE_parcelle_min.csv")
    All_TAW=[]
    Ir_all=[]
    quant_irr=[]
    for i in dfUTS.ID:
        a=param.loc[param[1].isin(dfGSM.loc[dfGSM.ID==i]["maxZr"])][0]+1
        b=param.loc[param[1].isin(dfRRP.loc[dfRRP.ID==i]["maxZr"])][0]+1
        c=param.loc[param[1].isin(dfUTS.loc[dfUTS.ID==i]["maxZr"])][0]+1
        UTS=pickle.load(open("H:/Yann_THESE/BESOIN_EAU/BESOIN_EAU/TRAITEMENT/RUNS_SAMIR/RUN_PKGC/PKGC_init_ru_optim_Fcover_fewi_De_Kr_days10_dose30_500_800_irri_auto_soil/2017/Output/maxZr/output_test_maize_irri_"+str(int(c))+".df","rb"))
        GSM=pickle.load(open("H:/Yann_THESE/BESOIN_EAU/BESOIN_EAU/TRAITEMENT/RUNS_SAMIR/RUN_PKGC/PKGC_GSM_init_ru_optim_Fcover_fewi_De_Kr_days10_dose30_500_800_irri_auto_soil/2017/Output/maxZr/output_test_maize_irri_"+str(int(a))+".df","rb"))
        RRP=pickle.load(open("H:/Yann_THESE/BESOIN_EAU/BESOIN_EAU/TRAITEMENT/RUNS_SAMIR/RUN_PKGC/PKGC_RRP_init_ru_optim_Fcover_fewi_De_Kr_days10_dose30_500_800_irri_auto_soil/2017/Output/maxZr/output_test_maize_irri_"+str(int(b))+".df","rb"))
        TAWdfGSM=GSM.loc[GSM.id==i][["date","Dr",'Ir_auto',"TAW",'Dr']]
        TAWdfRRP=RRP.loc[RRP.id==i][["date","Dr",'Ir_auto',"TAW",'Dr']]
        TAWdfUTS=UTS.loc[UTS.id==i][["date","Dr",'Ir_auto',"TAW",'Dr']]
        All_TAW.append([i,TAWdfUTS.TAW.max(),TAWdfRRP.TAW.max(),TAWdfGSM.TAW.max()])
        Ir_all.append([i,TAWdfUTS.Ir_auto[TAWdfUTS.Ir_auto!=0.0].count(),TAWdfRRP.Ir_auto[TAWdfRRP.Ir_auto!=0.0].count(),TAWdfGSM.Ir_auto[TAWdfGSM.Ir_auto!=0.0].count()])
        quant_irr.append([i,TAWdfUTS.Ir_auto[TAWdfUTS.Ir_auto!=0.0].sum(),TAWdfRRP.Ir_auto[TAWdfRRP.Ir_auto!=0.0].sum(),TAWdfGSM.Ir_auto[TAWdfGSM.Ir_auto!=0.0].sum()])
        #  plot dynamique 
        plt.figure(figsize=(7,7))
        plt.plot(TAWdfGSM.date,TAWdfGSM.TAW,c='Blue',label="GSM")
        plt.plot(TAWdfGSM.date,TAWdfGSM.Dr,c="Blue",label='Dr',linestyle="--",linewidth=0.9)
        plt.plot(TAWdfRRP.date,TAWdfRRP.TAW,c='r',label="RRP")
        plt.plot(TAWdfGSM.date,TAWdfRRP.Dr,c="r",linestyle="--",linewidth=0.9)
        plt.plot(TAWdfUTS.date,TAWdfUTS.TAW,c="Green",label="UTS")
        plt.plot(TAWdfGSM.date,TAWdfUTS.Dr,c="Green",linestyle="--",linewidth=0.9)
        plt.title(i)
        plt.legend()
        plt.savefig("H:/Yann_THESE/BESOIN_EAU/BESOIN_EAU/TRAITEMENT/RUNS_SAMIR/RUN_PKGC/PKGC_init_ru_optim_Fcover_fewi_De_Kr_days10_dose30_500_800_irri_auto_soil/dynamique_TAW_3_soil_%s.png"%int(i))
    TAW_all=pd.DataFrame(All_TAW)
    TAW_all.columns=["ID","UTS","RRP","GSM"]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax1=plt.subplot(131)
    plt.hist(TAW_all.UTS,bins=20)
    plt.gca().xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(5))
    plt.xlabel("TAW")
    plt.xlim(50,180)
    plt.title("TAW UTS")
    ax2=plt.subplot(132)
    plt.hist(TAW_all.RRP,bins=20)
    plt.xlabel("TAW")
    plt.xlim(50,180)
    plt.gca().xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(5))
    plt.title("TAW RRP")
    ax3=plt.subplot(133)
    plt.hist(TAW_all.GSM,bins=20)
    plt.gca().xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(5))
    plt.xlabel("TAW")
    plt.xlim(50,180)
    plt.title("TAW GSM")
    plt.savefig("H:/Yann_THESE/BESOIN_EAU/BESOIN_EAU/TRAITEMENT/RUNS_SAMIR/RUN_PKGC/PKGC_init_ru_optim_Fcover_fewi_De_Kr_days10_dose30_500_800_irri_auto_soil/hist_TAW_type_sol_data.png")
    ALL_nb_Irr=pd.DataFrame(Ir_all)
    ALL_nb_Irr.columns=["ID",'UTS',"RRP",'GSM']
    ALL_nb_Irr.to_csv("H:/Yann_THESE/BESOIN_EAU/BESOIN_EAU/TRAITEMENT/RUNS_SAMIR/RUN_PKGC/PKGC_init_ru_optim_Fcover_fewi_De_Kr_days10_dose30_500_800_irri_auto_soil/NB_IRR_sol_data.csv")
    Quant_irr=pd.DataFrame(quant_irr,columns=["ID","UTS","RRP","GSM"])
    Quant_irr.to_csv("H:/Yann_THESE/BESOIN_EAU/BESOIN_EAU/TRAITEMENT/RUNS_SAMIR/RUN_PKGC/PKGC_init_ru_optim_Fcover_fewi_De_Kr_days10_dose30_500_800_irri_auto_soil/Quant_IRR_sol_data.csv")
    