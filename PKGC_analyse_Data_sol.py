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
    data_prof=pd.read_csv(d["PC_disk"]+"/TRAITEMENT/SOIL/SOIL_RIGOU/Extract_RRP_Rigou_parcelle_PKCG_2017_UTS_maj.csv",index_col=[0],sep=';',encoding='latin-1',decimal=',')
    param=pd.read_csv("H:/Yann_THESE/BESOIN_EAU/BESOIN_EAU/TRAITEMENT/RUNS_SAMIR/RUN_PKGC/PKGC_GSM_init_ru_optim_Fcover_fewi_De_Kr_days10_dose50_500_800_irri_auto_soil/2017/Output/maxZr/output_test_maize_irri_param.txt",header=None,skiprows=1,sep=";")
    dfGSM=pd.read_csv("H:/Yann_THESE/BESOIN_EAU/BESOIN_EAU/TRAITEMENT/RUNS_SAMIR/RUN_PKGC/PKGC_GSM_init_ru_optim_Fcover_fewi_De_Kr_days10_dose50_500_800_irri_auto_soil/Table_RMSE_parcelle_min.csv")
    dfRRP=pd.read_csv("H:/Yann_THESE/BESOIN_EAU/BESOIN_EAU/TRAITEMENT/RUNS_SAMIR/RUN_PKGC/PKGC_RRP_init_ru_optim_Fcover_fewi_De_Kr_days10_dose50_500_800_irri_auto_soil/Table_RMSE_parcelle_min.csv")
    dfUTS=pd.read_csv("H:/Yann_THESE/BESOIN_EAU/BESOIN_EAU/TRAITEMENT/RUNS_SAMIR/RUN_PKGC/PKGC_init_ru_optim_Fcover_fewi_De_Kr_days10_dose50_500_800_irri_auto_soil/Table_RMSE_parcelle_min.csv")
    # dfUTSFAO=pd.read_csv("H:/Yann_THESE/BESOIN_EAU/BESOIN_EAU/TRAITEMENT/RUNS_SAMIR/RUN_PKGC/PKGC_FAO_init_ru_optim_Fcover_fewi_De_Kr_days10_dose30_500_800_irri_auto_soil/Table_RMSE_parcelle_min.csv")
    All_TAW=[]
    Ir_all=[]
    quant_irr=[]
    valeur_maxZr=[]
    for i in dfUTS.ID:
        a=param.loc[param[1].isin(dfGSM.loc[dfGSM.ID==i]["maxZr"])][0]+1
        b=param.loc[param[1].isin(dfRRP.loc[dfRRP.ID==i]["maxZr"])][0]+1
        c=param.loc[param[1].isin(dfUTS.loc[dfUTS.ID==i]["maxZr"])][0]+1
        # d=param.loc[param[1].isin(dfUTSFAO.loc[dfUTSFAO.ID==i]["maxZr"])][0]+1
        maxGsm=param.loc[param[1].isin(dfGSM.loc[dfGSM.ID==i]["maxZr"])][1]
        maxRRP=param.loc[param[1].isin(dfRRP.loc[dfRRP.ID==i]["maxZr"])][1]
        maxUTS=param.loc[param[1].isin(dfUTS.loc[dfUTS.ID==i]["maxZr"])][1]
        # maxUTSFAO=param.loc[param[1].isin(dfUTSFAO.loc[dfUTSFAO.ID==i]["maxZr"])][1]
        UTS=pickle.load(open("H:/Yann_THESE/BESOIN_EAU/BESOIN_EAU/TRAITEMENT/RUNS_SAMIR/RUN_PKGC/PKGC_init_ru_optim_Fcover_fewi_De_Kr_days10_dose50_500_800_irri_auto_soil/2017/Output/maxZr/output_test_maize_irri_"+str(int(c))+".df","rb"))
        GSM=pickle.load(open("H:/Yann_THESE/BESOIN_EAU/BESOIN_EAU/TRAITEMENT/RUNS_SAMIR/RUN_PKGC/PKGC_GSM_init_ru_optim_Fcover_fewi_De_Kr_days10_dose50_500_800_irri_auto_soil/2017/Output/maxZr/output_test_maize_irri_"+str(int(a))+".df","rb"))
        RRP=pickle.load(open("H:/Yann_THESE/BESOIN_EAU/BESOIN_EAU/TRAITEMENT/RUNS_SAMIR/RUN_PKGC/PKGC_RRP_init_ru_optim_Fcover_fewi_De_Kr_days10_dose50_500_800_irri_auto_soil/2017/Output/maxZr/output_test_maize_irri_"+str(int(b))+".df","rb"))
        # UTSFAO=pickle.load(open("H:/Yann_THESE/BESOIN_EAU/BESOIN_EAU/TRAITEMENT/RUNS_SAMIR/RUN_PKGC/PKGC_FAO_init_ru_optim_Fcover_fewi_De_Kr_days10_dose30_500_800_irri_auto_soil/2017/Output/maxZr/output_test_maize_irri_"+str(int(d))+".df","rb"))

        TAWdfGSM=GSM.loc[GSM.id==i][["date","Dr",'Ir_auto',"TAW"]]
        TAWdfRRP=RRP.loc[RRP.id==i][["date","Dr",'Ir_auto',"TAW"]]
        TAWdfUTS=UTS.loc[UTS.id==i][["date","Dr",'Ir_auto',"TAW"]]
        # TAWdfUTSFAO=UTSFAO.loc[UTSFAO.id==i][["date","Dr",'Ir_auto',"TAW"]]
        All_TAW.append([i,TAWdfUTS.TAW.max(),TAWdfRRP.TAW.max(),TAWdfGSM.TAW.max()])
        Ir_all.append([i,TAWdfUTS.Ir_auto[TAWdfUTS.Ir_auto!=0.0].count(),TAWdfRRP.Ir_auto[TAWdfRRP.Ir_auto!=0.0].count(),TAWdfGSM.Ir_auto[TAWdfGSM.Ir_auto!=0.0].count()])
        quant_irr.append([i,TAWdfUTS.Ir_auto[TAWdfUTS.Ir_auto!=0.0].sum(),TAWdfRRP.Ir_auto[TAWdfRRP.Ir_auto!=0.0].sum(),TAWdfGSM.Ir_auto[TAWdfGSM.Ir_auto!=0.0].sum()])
        valeur_maxZr.append([i,maxGsm.values[0],maxRRP.values[0],maxUTS.values[0]])
          # plot dynamique 
        plt.figure(figsize=(7,7))
        plt.plot(TAWdfGSM.date,TAWdfGSM.TAW,c='Blue',label="GSM")
        plt.plot(TAWdfGSM.date,TAWdfGSM.Dr,c="Blue",label='Dr',linestyle="--",linewidth=0.9)
        plt.plot(TAWdfRRP.date,TAWdfRRP.TAW,c='r',label="RRP")
        plt.plot(TAWdfGSM.date,TAWdfRRP.Dr,c="r",linestyle="--",linewidth=0.9)
        plt.plot(TAWdfUTS.date,TAWdfUTS.TAW,c="Green",label="UTS")
        plt.plot(TAWdfGSM.date,TAWdfUTS.Dr,c="g",linestyle="--",linewidth=0.9)
        plt.title(i)
        plt.legend()
        plt.savefig("H:/Yann_THESE/BESOIN_EAU/BESOIN_EAU/TRAITEMENT/RUNS_SAMIR/RUN_PKGC/PKGC_init_ru_optim_Fcover_fewi_De_Kr_days10_dose50_500_800_irri_auto_soil/dynamique_TAW_3_soil_%s.png"%int(i))
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
    plt.savefig("H:/Yann_THESE/BESOIN_EAU/BESOIN_EAU/TRAITEMENT/RUNS_SAMIR/RUN_PKGC/PKGC_init_ru_optim_Fcover_fewi_De_Kr_days10_dose50_500_800_irri_auto_soil/hist_TAW_type_sol_data.png")
    ALL_nb_Irr=pd.DataFrame(Ir_all)
    ALL_nb_Irr.columns=["ID",'UTS',"RRP",'GSM']
    ALL_nb_Irr.to_csv("H:/Yann_THESE/BESOIN_EAU/BESOIN_EAU/TRAITEMENT/RUNS_SAMIR/RUN_PKGC/PKGC_init_ru_optim_Fcover_fewi_De_Kr_days10_dose50_500_800_irri_auto_soil/NB_IRR_sol_data.csv")
    Quant_irr=pd.DataFrame(quant_irr,columns=["ID","UTS","RRP","GSM"])
    Quant_irr.to_csv("H:/Yann_THESE/BESOIN_EAU/BESOIN_EAU/TRAITEMENT/RUNS_SAMIR/RUN_PKGC/PKGC_init_ru_optim_Fcover_fewi_De_Kr_days10_dose50_500_800_irri_auto_soil/Quant_IRR_sol_data.csv")
    val_max=pd.DataFrame(valeur_maxZr,columns=["ID","GSM","RRP","UTS"])
    data_SOIL=pd.merge(val_max,data_prof["ProfRacPot"]*10,on=["ID"])
    
    slopeGSM, intercept, r_valueGSM, p_value, std_err = stats.linregress(data_SOIL.ProfRacPot.to_list(),data_SOIL.GSM.to_list())
    biasGSM=1/data_SOIL["ProfRacPot"].shape[0]*sum(data_SOIL.GSM-np.mean(data_SOIL.ProfRacPot)) 
    rmsGSM = np.sqrt(mean_squared_error(data_SOIL.ProfRacPot,data_SOIL.GSM))
    slopeRRP, intercept, r_valueRRP, p_value, std_err = stats.linregress(data_SOIL.ProfRacPot.to_list(),data_SOIL.RRP.to_list())
    biasRRP=1/data_SOIL["ProfRacPot"].shape[0]*sum(data_SOIL.RRP-np.mean(data_SOIL.ProfRacPot)) 
    rmsRRP = np.sqrt(mean_squared_error(data_SOIL.ProfRacPot,data_SOIL.RRP))
    slopeUTS, intercept, r_valueUTS, p_value, std_err = stats.linregress(data_SOIL.ProfRacPot.to_list(),data_SOIL.UTS.to_list())
    biasUTS=1/data_SOIL["ProfRacPot"].shape[0]*sum(data_SOIL.UTS-np.mean(data_SOIL.ProfRacPot)) 
    rmsUTS = np.sqrt(mean_squared_error(data_SOIL.ProfRacPot,data_SOIL.UTS))
    plt.figure(figsize=(7,7))
    plt.scatter(data_SOIL.ProfRacPot,data_SOIL.GSM,c='r',label="GSM",marker="x")
    plt.scatter(data_SOIL.ProfRacPot,data_SOIL.RRP,c='b',label="RRP",marker="v")
    plt.scatter(data_SOIL.ProfRacPot,data_SOIL.UTS,c='g',label="UTS")
    plt.xlim(0,2000)
    plt.ylim(0,2000)
    plt.legend()
    plt.xlabel("MaxZr obs.")
    plt.ylabel("MaxZr mod.")
    plt.plot([0, 2000], [0.0,2000], 'black', lw=1,linestyle='--')
    rectangle = plt.Rectangle((480, 1730),430,220, ec='r',fc='r',alpha=0.1)
    plt.gca().add_patch(rectangle)
    plt.text(500,1900,"RMSE = "+str(round(rmsGSM,2))) 
    plt.text(500,1850,"R² = "+str(round(r_valueGSM,2)))
    plt.text(500,1800,"Pente = "+str(round(slopeGSM,2)))
    plt.text(500,1750,"Biais = "+str(round(biasGSM,2)))
    rectangle = plt.Rectangle((1380, 1530),430,220, ec='b',fc='b',alpha=0.1)
    plt.gca().add_patch(rectangle)
    plt.text(1400,1700,"RMSE = "+str(round(rmsRRP,2))) 
    plt.text(1400,1650,"R² = "+str(round(r_valueRRP,2)))
    plt.text(1400,1600,"Pente = "+str(round(slopeRRP,2)))
    plt.text(1400,1550,"Biais = "+str(round(biasRRP,2)))
    rectangle = plt.Rectangle((1480, 230),430,220, ec='g',fc='g',alpha=0.1)
    plt.gca().add_patch(rectangle)
    plt.text(1500,400,"RMSE = "+str(round(rmsUTS,2))) 
    plt.text(1500,350,"R² = "+str(round(r_valueUTS,2)))
    plt.text(1500,300,"Pente = "+str(round(slopeUTS,2)))
    plt.text(1500,250,"Biais = "+str(round(biasUTS,2)))
    plt.savefig("H:/Yann_THESE/BESOIN_EAU/BESOIN_EAU/TRAITEMENT/RUNS_SAMIR/RUN_PKGC/PKGC_init_ru_optim_Fcover_fewi_De_Kr_days10_dose50_500_800_irri_auto_soil/scatter_MAXZR_type_sol_data.png")
    
    
#   