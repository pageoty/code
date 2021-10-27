#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 15:37:52 2021

@author: pageot
Validation Irrigation automatique parcelle all partenaires 
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
    d["PC_disk"]="/run/media/pageot/Transcend/Yann_THESE/BESOIN_EAU/BESOIN_EAU/"
    d["PC_labo"]="/datalocal/vboxshare/THESE/BESOIN_EAU/"
    years=["2017"]

    
    # Input data si depth == GSM et PF_CC ==  BRUAND
    df_CACG=pd.read_csv(d["PC_disk"]+"TRAITEMENT/RUNS_SAMIR/RUN_CACG/CACG_init_ru_optim_P055_Fcover_fewi_De_Kr_days10_dose30_400_2500_irri_auto_soil/tab_resu_depth_GSM_UTS_PF_CC.csv")
    df_CACG.columns=["unnamed","ID","Quant","MaxZr",'MMEAU']
    df_PKGC=pd.read_csv(d["PC_disk"]+"/TRAITEMENT/RUNS_SAMIR/RUN_PKGC/GERS/PKGC_init_ru_optim_P055_Fcover_fewi_De_Kr_days10_dose30_400_2500_irri_auto_soil/tab_resu_GERS_forcagemaxZr_p_depth_GSM.csv",sep=',',encoding='latin-1',decimal='.')
    data_valid_PKGC=pd.read_csv(d["PC_disk"]+"/TRAITEMENT/SOIL/SOIL_RIGOU/Extract_RRP_Rigou_parcelle_PKCG_GERS_2017_UTS_maj.csv",index_col=[0],sep=';',encoding='latin-1',decimal=',')
    list_drop=[7,9,10,13,25,29,34,50,54,61,83,90,98]
    data_valid_PKGC =data_valid_PKGC[-data_valid_PKGC["ID"].isin(list_drop)]
    df_PKGC["MMEAU"]=data_valid_PKGC.MMEAU.values
    # cocnat tab 
    df_plot=df_PKGC[["MMEAU","Quant"]]
    df_plot.append(df_CACG[["MMEAU","Quant"]])
    
    plt.figure(figsize=(7,7))
    slope, intercept, r_value, p_value, std_err = stats.linregress(df_plot.MMEAU.to_list(),df_plot.Quant.to_list())
    bias=1/df_plot["MMEAU"].shape[0]*sum(df_plot.Quant-np.mean(df_plot.MMEAU)) 
    rms = np.sqrt(mean_squared_error(df_plot.MMEAU,df_plot.Quant))
    plt.scatter(df_plot.MMEAU,df_plot.Quant)
    plt.xlim(-10,350)
    plt.ylim(-10,350)
    plt.xlabel("Quantité saisonnière observée en mm ")
    plt.ylabel("Quantité saisonnière modélisée en mm ")
    plt.plot([-10.0, 350], [-10.0,350], 'black', lw=1,linestyle='--')
    rectangle = plt.Rectangle((95, 245),70,45, ec='blue',fc='blue',alpha=0.1)
    plt.gca().add_patch(rectangle)
    plt.text(100,280,"RMSE = "+str(round(rms,2))) 
    plt.text(100,270,"R² = "+str(round(r_value,2)))
    plt.text(100,260,"Pente = "+str(round(slope,2)))
    plt.text(100,250,"Biais = "+str(round(bias,2)))
    plt.savefig("/run/media/pageot/Transcend/Yann_THESE/BESOIN_EAU/BESOIN_EAU/TRAITEMENT/RUNS_SAMIR/Plot_resu_all_parcelle_partenaires/scatter_depth_GSM_PF_CC_bruand.png")
    
    
    #  SI maxZr == RUM inverser et PF_CC = BRUAND
    df_CACG= pd.read_csv(d["PC_disk"]+"TRAITEMENT/RUNS_SAMIR/RUN_CACG/maxZr_rum/tab_resu_CACG_maxZr_inversion_RUM_2017.csv")
    df_CACG.columns=["unnamed","ID","Quant","MaxZr",'TAWmax','MMEAU']
    df_PKGC=pd.read_csv(d["PC_disk"]+"TRAITEMENT/RUNS_SAMIR/RUN_PKGC/GERS/maxZr_rum/tab_resu_PKGC_maxZr_inversion_RUM.csv")
    df_PKGC.columns=['Unnamed: 0', 'ID', 'Quant', 'maxzr', 'TAWMax', 'MMEAU','Class_Bruand']
    df_plot=df_PKGC[["MMEAU","Quant"]]
    df_plot.append(df_CACG[["MMEAU","Quant"]])
    
    plt.figure(figsize=(7,7))
    slope, intercept, r_value, p_value, std_err = stats.linregress(df_plot.MMEAU.to_list(),df_plot.Quant.to_list())
    bias=1/df_plot["MMEAU"].shape[0]*sum(df_plot.Quant-np.mean(df_plot.MMEAU)) 
    rms = np.sqrt(mean_squared_error(df_plot.MMEAU,df_plot.Quant))
    plt.scatter(df_plot.MMEAU,df_plot.Quant)
    plt.xlim(-10,350)
    plt.ylim(-10,350)
    plt.xlabel("Quantité saisonnière observée en mm ")
    plt.ylabel("Quantité saisonnière modélisée en mm ")
    plt.plot([-10.0, 350], [-10.0,350], 'black', lw=1,linestyle='--')
    rectangle = plt.Rectangle((95, 245),70,45, ec='blue',fc='blue',alpha=0.1)
    plt.gca().add_patch(rectangle)
    plt.text(100,280,"RMSE = "+str(round(rms,2))) 
    plt.text(100,270,"R² = "+str(round(r_value,2)))
    plt.text(100,260,"Pente = "+str(round(slope,2)))
    plt.text(100,250,"Biais = "+str(round(bias,2)))
    # for i in enumerate(df_plot.index):
    #     label = int(i[1])
    #     plt.annotate(label, # this is the text
    #           (df_plot["MMEAU"].iloc[i[0]],df_plot.Quant.iloc[i[0]]), # this is the point to label
    #           textcoords="offset points", # how to position the text
    #           xytext=(-6,2), # distance from text to points (x,y)
    #           ha='center')
    plt.savefig("/run/media/pageot/Transcend/Yann_THESE/BESOIN_EAU/BESOIN_EAU/TRAITEMENT/RUNS_SAMIR/Plot_resu_all_parcelle_partenaires/scatter_maxZr_inver_PF_CC_bruand.png")

    
    #  SI maxAr= GSM et PF_CC == GSM 
    #  Pas lancer pour CACG
    df_PKGC=pd.read_csv(d["PC_disk"]+"/TRAITEMENT/RUNS_SAMIR/RUN_PKGC/GERS/PKGC_Fcover_GSM_irri_auto/tab_resu_PKGC_depth_PF_CC_GSM.csv")
    df_CACG=pd.read_csv(d["PC_disk"]+"/TRAITEMENT/RUNS_SAMIR/RUN_CACG/CACG_Fcover_GSM_irri_auto/tab_resu_depth_GSM_UTS_PF_CC.csv")
    df_CACG.columns=["unnamed","ID","Quant","MaxZr","TAWMax",'MMEAU']
    df_plot=df_PKGC[["MMEAU","Quant"]]
    df_plot.append(df_CACG[["MMEAU","Quant"]])
    
    df_GSM_max=pd.concat([df_PKGC["maxZr"],df_CACG.MaxZr])
     
    plt.figure(figsize=(7,7))
    slope, intercept, r_value, p_value, std_err = stats.linregress(df_plot.MMEAU.to_list(),df_plot.Quant.to_list())
    bias=1/df_plot["MMEAU"].shape[0]*sum(df_plot.Quant-np.mean(df_plot.MMEAU)) 
    rms = np.sqrt(mean_squared_error(df_plot.MMEAU,df_plot.Quant))
    plt.scatter(df_plot.MMEAU,df_plot.Quant)
    plt.xlim(-10,350)
    plt.ylim(-10,350)
    plt.xlabel("Quantité saisonnière observée en mm ")
    plt.ylabel("Quantité saisonnière modélisée en mm ")
    plt.plot([-10.0, 350], [-10.0,350], 'black', lw=1,linestyle='--')
    rectangle = plt.Rectangle((95, 245),70,45, ec='blue',fc='blue',alpha=0.1)
    plt.gca().add_patch(rectangle)
    plt.text(100,280,"RMSE = "+str(round(rms,2))) 
    plt.text(100,270,"R² = "+str(round(r_value,2)))
    plt.text(100,260,"Pente = "+str(round(slope,2)))
    plt.text(100,250,"Biais = "+str(round(bias,2)))
    # for i in enumerate(df_plot.index):
    #     label = int(i[1])
    #     plt.annotate(label, # this is the text
    #           (df_plot["MMEAU"].iloc[i[0]],df_plot.Quant.iloc[i[0]]), # this is the point to label
    #           textcoords="offset points", # how to position the text
    #           xytext=(-6,2), # distance from text to points (x,y)
    #           ha='center')
    plt.savefig("/run/media/pageot/Transcend/Yann_THESE/BESOIN_EAU/BESOIN_EAU/TRAITEMENT/RUNS_SAMIR/Plot_resu_all_parcelle_partenaires/scatter_maxZr_depth_GSM_PF_CC_GSM.png")

    #  Plot de TAWmax 
    df_PKGC_UTS=pd.read_csv(d["PC_disk"]+"TRAITEMENT/RUNS_SAMIR/RUN_PKGC/GERS/maxZr_rum/tab_resu_PKGC_maxZr_inversion_RUM.csv")
    df_CACG_UTS= pd.read_csv(d["PC_disk"]+"TRAITEMENT/RUNS_SAMIR/RUN_CACG/maxZr_rum/tab_resu_CACG_maxZr_inversion_RUM_2017.csv")
    df_TAW_UTS=pd.concat([df_PKGC_UTS.TAWMax,df_CACG_UTS.TAWMax])
    
    df_PKGC_GSM=pd.read_csv(d["PC_disk"]+"/TRAITEMENT/RUNS_SAMIR/RUN_PKGC/GERS/PKGC_Fcover_GSM_irri_auto/tab_resu_PKGC_depth_PF_CC_GSM.csv")
    df_CACG_GSM=pd.read_csv(d["PC_disk"]+"/TRAITEMENT/RUNS_SAMIR/RUN_CACG/CACG_Fcover_GSM_irri_auto/tab_resu_depth_GSM_UTS_PF_CC.csv")
    df_TAW_GSM=pd.concat([df_PKGC_GSM.TAWmax,df_CACG_GSM.TAWMax])
 
    plt.figure(figsize=(7,7))
    plt.hist(df_TAW_GSM,label="RFU modélisées GSM",bins=15,color="b",ec="black",alpha=0.7)
    plt.hist(df_TAW_UTS,label="RFU modélisées UTS",bins=10,color="r",ec="black",alpha=0.7)
    plt.hist(df.TAWMax,label="RFU modélisées GSM/UTS", bins=10,ec="black",color="g",alpha=0.6,linestyle='--')
    plt.ylabel("Fréquence ")
    plt.legend()
    plt.ylim(0,30)
    plt.xlabel("Valeur de RUM en mm ")
    plt.ylabel("Fréquence ")
    rectangle = plt.Rectangle((163, 21),65,3, ec='gray',fc='gray',alpha=0.1)
    plt.gca().add_patch(rectangle)
    plt.text(165,23,"Median GSM = "+str(round(df_TAW_GSM.median(),2))) 
    plt.text(165,22,"Median UTS = "+str(round(df_TAW_UTS.median(),2)))
    plt.text(165,21,"Median GSM/UTS = "+str(round(df.TAWMax.median(),2))) 
    plt.savefig("/run/media/pageot/Transcend/Yann_THESE/BESOIN_EAU/BESOIN_EAU/TRAITEMENT/RUNS_SAMIR/Plot_resu_all_parcelle_partenaires/histo_TAW_UTS_GSM_GSM_and_UTS.png")

    #  Si maxZr est fixé à 1500 ou 1200
    df_CACG=pd.read_csv(d["PC_disk"]+"/TRAITEMENT/RUNS_SAMIR/RUN_CACG/CACG_init_ru_optim_P055_Fcover_fewi_De_Kr_days10_dose30_400_2500_irri_auto_soil/tab_FAO_CACG.csv")
    df_PKGC=pd.read_csv(d["PC_disk"]+"/TRAITEMENT/RUNS_SAMIR/RUN_PKGC/GERS/PKGC_init_ru_optim_P055_Fcover_fewi_De_Kr_days10_dose30_400_2500_irri_auto_soil/tab_resu_PKGC_FAO.csv")
    data_valid_PKGC=pd.read_csv(d["PC_disk"]+"/TRAITEMENT/SOIL/SOIL_RIGOU/Extract_RRP_Rigou_parcelle_PKCG_GERS_2017_UTS_maj.csv",index_col=[0],sep=';',encoding='latin-1',decimal=',')
    list_drop=[7,9,10,13,25,29,34,50,54,61,83,90,98]
    data_valid_PKGC =data_valid_PKGC[-data_valid_PKGC["ID"].isin(list_drop)]
    df_PKGC["MMEAU"]=data_valid_PKGC.MMEAU.values
    
    df_CACG_valid=pd.read_csv(d["PC_disk"]+"TRAITEMENT/RUNS_SAMIR/RUN_CACG/CACG_init_ru_optim_P055_Fcover_fewi_De_Kr_days10_dose30_400_2500_irri_auto_soil/tab_resu_depth_GSM_UTS_PF_CC.csv")
    df_CACG_valid.columns=["unnamed","ID","Quant","MaxZr",'MMEAU']
    df_CACG_valid=df_CACG_valid[["ID","MMEAU"]]
    df_CACG["MMEAU"]=df_CACG_valid.MMEAU.values
    
    
    df_plot=df_PKGC.drop("ID",axis=1)
    df_plot.append(df_CACG.drop('ID',axis=1))
    
    for i in df_plot.columns[5:-15]:
        plt.figure(figsize=(7,7))
        slope, intercept, r_value, p_value, std_err = stats.linregress(df_plot.MMEAU.to_list(),df_plot[i].to_list())
        bias=1/df_plot["MMEAU"].shape[0]*sum(df_plot[i]-np.mean(df_plot.MMEAU)) 
        rms = np.sqrt(mean_squared_error(df_plot.MMEAU,df_plot[i]))
        plt.scatter(df_plot.MMEAU,df_plot[i])
        plt.title(i)
        plt.xlim(-10,350)
        plt.ylim(-10,350)
        plt.xlabel("Quantité saisonnière observée en mm ")
        plt.ylabel("Quantité saisonnière modélisée en mm ")
        plt.plot([-10.0, 350], [-10.0,350], 'black', lw=1,linestyle='--')
        rectangle = plt.Rectangle((95, 245),70,45, ec='blue',fc='blue',alpha=0.1)
        plt.gca().add_patch(rectangle)
        plt.text(100,280,"RMSE = "+str(round(rms,2))) 
        plt.text(100,270,"R² = "+str(round(r_value,2)))
        plt.text(100,260,"Pente = "+str(round(slope,2)))
        plt.text(100,250,"Biais = "+str(round(bias,2)))
        plt.savefig("/run/media/pageot/Transcend/Yann_THESE/BESOIN_EAU/BESOIN_EAU/TRAITEMENT/RUNS_SAMIR/Plot_resu_all_parcelle_partenaires/scatter_maxZR_FAO_%s_PF_CC_bruand.png"%i)



#  Si PF_CC =GSM + soil et maxZr= FAO 
    df_PKGC=pd.read_csv(d["PC_disk"]+"/TRAITEMENT/RUNS_SAMIR/RUN_PKGC/GERS/PKGC_Fcover_GSM_irri_auto/tab_resu_GSM_PF_CC_FAO_maxZr.csv")
    data_valid_PKGC=pd.read_csv(d["PC_disk"]+"/TRAITEMENT/SOIL/SOIL_RIGOU/Extract_RRP_Rigou_parcelle_PKCG_GERS_2017_UTS_maj.csv",index_col=[0],sep=';',encoding='latin-1',decimal=',')
    list_drop=[7,9,10,13,25,29,34,50,54,61,83,90,98]
    data_valid_PKGC =data_valid_PKGC[-data_valid_PKGC["ID"].isin(list_drop)]
    df_PKGC["MMEAU"]=data_valid_PKGC.MMEAU.values
    
    df_CACG=pd.read_csv(d["PC_disk"]+"TRAITEMENT/RUNS_SAMIR/RUN_CACG/CACG_GSM_init_ru_optim_P055_Fcover_fewi_De_Kr_days10_dose30_400_1800_irri_auto_soil/tab_FAO_CACG.csv")
    df_CACG_valid=pd.read_csv(d["PC_disk"]+"TRAITEMENT/RUNS_SAMIR/RUN_CACG/CACG_init_ru_optim_P055_Fcover_fewi_De_Kr_days10_dose30_400_2500_irri_auto_soil/tab_resu_depth_GSM_UTS_PF_CC.csv")
    df_CACG_valid.columns=["unnamed","ID","Quant","MaxZr",'MMEAU']
    df_CACG_valid=df_CACG_valid[["ID","MMEAU"]]
    df_CACG["MMEAU"]=df_CACG_valid.MMEAU.values
    
    df_plot=df_PKGC.drop("ID",axis=1)
    df_plot.append(df_CACG.drop('ID',axis=1))

    for i in df_plot.columns[5:-3]:
       plt.figure(figsize=(7,7))
       slope, intercept, r_value, p_value, std_err = stats.linregress(df_plot.MMEAU.to_list(),df_plot[i].to_list())
       bias=1/df_plot["MMEAU"].shape[0]*sum(df_plot[i]-np.mean(df_plot.MMEAU)) 
       rms = np.sqrt(mean_squared_error(df_plot.MMEAU,df_plot[i]))
       plt.scatter(df_plot.MMEAU,df_plot[i])
       plt.title(i)
       plt.xlim(-10,350)
       plt.ylim(-10,350)
       plt.xlabel("Quantité saisonnière observée en mm ")
       plt.ylabel("Quantité saisonnière modélisée en mm ")
       plt.plot([-10.0, 350], [-10.0,350], 'black', lw=1,linestyle='--')
       rectangle = plt.Rectangle((95, 245),70,45, ec='blue',fc='blue',alpha=0.1)
       plt.gca().add_patch(rectangle)
       plt.text(100,280,"RMSE = "+str(round(rms,2))) 
       plt.text(100,270,"R² = "+str(round(r_value,2)))
       plt.text(100,260,"Pente = "+str(round(slope,2)))
       plt.text(100,250,"Biais = "+str(round(bias,2)))
       # plt.savefig("/run/media/pageot/Transcend/Yann_THESE/BESOIN_EAU/BESOIN_EAU/TRAITEMENT/RUNS_SAMIR/Plot_resu_all_parcelle_partenaires/scatter_maxZR_FAO_%s_PF_CC_GSM_soil_GSM.png"%i)

       
#  Si PF_CC = GSM et maxZr == UTS maps 
    df_PKGC=pd.read_csv(d["PC_disk"]+"/TRAITEMENT/RUNS_SAMIR/RUN_PKGC/GERS/PKGC_Fcover_GSM_irri_auto_300_1900/tab_resu_GSM_PF_CC_maxzrUTS.csv")
    data_valid_PKGC=pd.read_csv(d["PC_disk"]+"/TRAITEMENT/SOIL/SOIL_RIGOU/Extract_RRP_Rigou_parcelle_PKCG_GERS_2017_UTS_maj.csv",index_col=[0],sep=';',encoding='latin-1',decimal=',')
    list_drop=[7,9,10,13,25,29,34,50,54,61,83,90,98]
    data_valid_PKGC =data_valid_PKGC[-data_valid_PKGC["ID"].isin(list_drop)]
    data_sol_GSM=pd.read_csv(d["PC_disk"]+"/TRAITEMENT/SOIL/GSM/Extract_GSM_parcelle_PKGC_GERS_2017_GSM_PF_CC_class_name.csv",index_col=[0],sep=',',encoding='latin-1',decimal='.')
    data_sol_GSM =data_sol_GSM[-data_sol_GSM["ID"].isin(list_drop)]
    df_PKGC["MMEAU"]=data_valid_PKGC.MMEAU.values
    df_PKGC["Class_Bruand"]=data_valid_PKGC.Class_Bruand.values
    df_PKGC["maxZr_UTS"]=round(data_valid_PKGC.Zrmax_RUM,-2).values
    df_PKGC["PF_mean"]=data_sol_GSM.PF_GSM.values
    df_PKGC["CC_mean"]=data_sol_GSM.CC_GSM.values
    
    
    df_plot=[]
    for i,z in zip(df_PKGC.ID,df_PKGC.maxZr_UTS):
        a=df_PKGC[df_PKGC.ID==i][['MMEAU',str(float(z)),"Class_Bruand","PF_mean","CC_mean"]]
        a["TAW"]=a.eval("CC_mean-PF_mean")
        a["TAWMax"]=a["TAW"]*z
        df_plot.append([a.MMEAU.values[0], a.iloc[:,1].values[0],a.Class_Bruand.values[0],a.TAWMax.values[0]])
        
        
   
    df_CACG=pd.read_csv(d["PC_disk"]+"TRAITEMENT/RUNS_SAMIR/RUN_CACG/CACG_GSM_init_ru_optim_P055_Fcover_fewi_De_Kr_days10_dose30_400_1800_irri_auto_soil/tab_FAO_CACG.csv")
    df_CACG_valid= pd.read_csv(d["PC_disk"]+"TRAITEMENT/RUNS_SAMIR/RUN_CACG/maxZr_rum/tab_resu_CACG_maxZr_inversion_RUM_2017.csv")
    df_CACG_classe= pd.read_csv(d["PC_disk"]+"/TRAITEMENT/SOIL/SOIL_RIGOU/Extract_RRP_Rigou_parcelle_CACG_"+str(y)+"_UTS_maj.csv",index_col=[0],sep=';',encoding='latin-1',decimal=',')
    df_CACG_classe.reset_index(inplace=True)
    df_CACG_classe =df_CACG_classe[df_CACG_classe["ID"].isin(list(df_CACG_valid.ID))]
    data_sol_GSM_CACG=pd.read_csv(d["PC_disk"]+"/TRAITEMENT/SOIL/GSM/Extract_GSM_parcelle_CACG_2017_GSM_tri.csv",index_col=[0],sep=',',encoding='latin-1',decimal=',')
    data_sol_GSM_CACG =data_sol_GSM_CACG[data_sol_GSM_CACG["ID"].isin(list(df_CACG_valid.ID))]
    df_CACG_valid.columns=["unnamed","ID","Quant","MaxZr",'TAWmax','MMEAU']
    df_CACG_valid=df_CACG_valid[["ID","MMEAU","MaxZr"]]
    df_CACG["MMEAU"]=df_CACG_valid.MMEAU.values
    df_CACG["maxZr_UTS"]=round(df_CACG_valid.MaxZr,-2)
    df_CACG["Class_Bruand"]=df_CACG_classe.Class_Bruand.values
    df_CACG["PF_mean"]=data_sol_GSM_CACG.PF_GSM.values
    df_CACG["CC_mean"]=data_sol_GSM_CACG.CC_GSM.values
    for i,z in zip(df_CACG.ID,df_CACG.maxZr_UTS):
        if z < 600 :
            a=df_CACG[df_CACG.ID==i][['MMEAU',"600.0",'Class_Bruand',"PF_mean","CC_mean"]]
            a["TAW"]=a.eval("CC_mean-PF_mean")
            a["TAWMax"]=a["TAW"]*600
        else :
            a=df_CACG[df_CACG.ID==i][['MMEAU',str(float(z)),"Class_Bruand","PF_mean","CC_mean"]]
            a["TAW"]=a.eval("CC_mean-PF_mean")
            a["TAWMax"]=a["TAW"]*z
        df_plot.append([a.MMEAU.values[0], a.iloc[:,1].values[0],a.Class_Bruand.values[0],a.TAWMax.values[0]])
    df=pd.DataFrame(df_plot,columns=["MMEAU",'Quant',"Class_Bruand","TAWMax"])
    
    
    
    df_UTS_max=pd.concat([df_PKGC["maxZr_UTS"],df_CACG["maxZr_UTS"]])
    plt.figure(figsize=(7,7))
    slope, intercept, r_value, p_value, std_err = stats.linregress(df.MMEAU.to_list(),df.Quant.to_list())
    bias=1/df["MMEAU"].shape[0]*sum(df.Quant-np.mean(df.MMEAU)) 
    rms = np.sqrt(mean_squared_error(df.MMEAU,df.Quant))
    plt.scatter(df.MMEAU,df.Quant)
    plt.xlim(-10,350)
    plt.ylim(-10,350)
    plt.xlabel("Quantité saisonnière observée en mm ")
    plt.ylabel("Quantité saisonnière modélisée en mm ")
    plt.plot([-10.0, 350], [-10.0,350], 'black', lw=1,linestyle='--')
    rectangle = plt.Rectangle((95, 245),70,45, ec='blue',fc='blue',alpha=0.1)
    plt.gca().add_patch(rectangle)
    plt.text(100,280,"RMSE = "+str(round(rms,2))) 
    plt.text(100,270,"R² = "+str(round(r_value,2)))
    plt.text(100,260,"Pente = "+str(round(slope,2)))
    plt.text(100,250,"Biais = "+str(round(bias,2)))
    # for i in enumerate(df_plot.index):
    #     label = int(i[1])
    #     plt.annotate(label, # this is the text
    #           (df_plot["MMEAU"].iloc[i[0]],df_plot.Quant.iloc[i[0]]), # this is the point to label
    #           textcoords="offset points", # how to position the text
    #           xytext=(-6,2), # distance from text to points (x,y)
    #           ha='center')
    plt.savefig("/run/media/pageot/Transcend/Yann_THESE/BESOIN_EAU/BESOIN_EAU/TRAITEMENT/RUNS_SAMIR/Plot_resu_all_parcelle_partenaires/scatter_maxZrUTS_PF_CC_GSM.png")

    df["TEX"]="L"
    df.loc[(df.Class_Bruand == "A"),'TEX']= "A"
    df.loc[(df.Class_Bruand == "AL"),'TEX']="A"
    df.loc[(df.Class_Bruand == "ALO"),'TEX']="A"
    df.loc[(df.Class_Bruand == "SL"),'TEX']="S"
    df.loc[(df.Class_Bruand == "SA"),'TEX']="S"
    labels, index = np.unique(df["TEX"], return_inverse=True)
    plt.figure(figsize=(7,7))
    a=plt.scatter(df.MMEAU,df.Quant,c=index,cmap='coolwarm')
    plt.legend(a.legend_elements()[0],labels)
    plt.xlim(-10,350)
    plt.ylim(-10,350)
    plt.xlabel("Quantité saisonnière observée en mm ")
    plt.ylabel("Quantité saisonnière modélisée en mm ")
    plt.plot([-10.0, 350], [-10.0,350], 'black', lw=1,linestyle='--')
    rectangle = plt.Rectangle((95, 245),70,45, ec='blue',fc='blue',alpha=0.1)
    plt.gca().add_patch(rectangle)
    plt.text(100,280,"RMSE = "+str(round(rms,2))) 
    plt.text(100,270,"R² = "+str(round(r_value,2)))
    plt.text(100,260,"Pente = "+str(round(slope,2)))
    plt.text(100,250,"Biais = "+str(round(bias,2)))
    plt.savefig("/run/media/pageot/Transcend/Yann_THESE/BESOIN_EAU/BESOIN_EAU/TRAITEMENT/RUNS_SAMIR/Plot_resu_all_parcelle_partenaires/scatter_maxZrUTS_PF_CC_GSM_texture_sep.png")


    plt.figure(figsize=(7,7))
    plt.hist(df.TAWMax,label="RUM", bins=15)
    plt.text(120,13,"Median = "+str(round(df.TAWMax.median(),2))) 
    plt.xlabel("Valeur de RUM en mm ")
    plt.ylabel("Fréquence ")
    plt.savefig("/run/media/pageot/Transcend/Yann_THESE/BESOIN_EAU/BESOIN_EAU/TRAITEMENT/RUNS_SAMIR/Plot_resu_all_parcelle_partenaires/Hist_RUM_PF_CC_GSM_maxZr_UTS.png")


    #  Hsito profondeur de sol 
    plt.figure(figsize=(7,7))
    plt.hist(df_UTS_max,label="MaxZr UTS",bins=10,fc="Red",alpha=0.4,ec="Black")
    plt.hist(df_GSM_max,label="Profondeur de sol GSM",bins=15,fc="Blue",alpha=0.4,ec="Black")
    plt.ylabel("Fréquence ")
    plt.legend()
    plt.ylim(0,40)
    plt.xlabel("Profondeur de sol/ enracinement en mm")
    plt.ylabel("Fréquence ")
    rectangle = plt.Rectangle((163, 22),550,3, ec='gray',fc='gray',alpha=0.1)
    plt.gca().add_patch(rectangle)
    plt.text(165,24,"Median GSM = "+str(round(df_GSM_max.median(),2))) 
    plt.text(165,22,"Median UTS = "+str(round(df_UTS_max.median(),2)))
    plt.savefig("/run/media/pageot/Transcend/Yann_THESE/BESOIN_EAU/BESOIN_EAU/TRAITEMENT/RUNS_SAMIR/Plot_resu_all_parcelle_partenaires/histo_maxZr_sol_UTS_GSM.png")




# =============================================================================
#  PF_CC GSM et maxZr UTS
# =============================================================================
    NDVI_stat_irri=[]
    NDVI_stat_irri_auto=[]
    param=pd.read_csv(d["PC_disk"]+"/TRAITEMENT/RUNS_SAMIR/RUN_PKGC/GERS/PKGC_Fcover_GSM_irri_auto_v2_irri_date_satr_6_Juin/2017/Output/maxZr/output_test_maize_irri_param.txt",header=None,skiprows=1,sep=";")
    IRR=[]
    ids=[]
    data_prof=pd.read_csv(d["PC_disk"]+"/TRAITEMENT/SOIL/GSM/Extract_GSM_parcelle_PKGC_GERS_2017_GSM_PF_CC_class_name.csv",index_col=[0],sep=',',encoding='latin-1',decimal=',')
    list_drop=[7,9,10,13,25,29,34,50,54,61,83,90,98]
    data_prof =data_prof[-data_prof["ID"].isin(list_drop)]
    for i in data_prof.ID:
        c=param.loc[param[1].isin(data_prof.loc[data_prof.ID==i]["Zrmax_UTS"])][0]
        val=param.loc[param[1].isin(data_prof.loc[data_prof.ID==i]["Zrmax_UTS"])][1]
        UTS=pickle.load(open(d["PC_disk"]+"/TRAITEMENT/RUNS_SAMIR/RUN_PKGC/GERS/PKGC_Fcover_GSM_irri_auto_v2_irri_date_satr_6_Juin/2017/Output/maxZr/output_test_maize_irri_"+str(int(c))+".df","rb"))
        data_id=UTS.groupby("id")
        ID_data=data_id.get_group(i)
        #  Plot NDVI vs Start irrigation
        # plt.figure(figsize=(7,7))
        # plt.plot(ID_data.date,ID_data.NDVI)
        # plt.plot(date_stat_irr[date_stat_irr._ID==i]["IRRDATDEB"],1,marker="o",linestyle="")
        print(ID_data[ID_data.date==date_stat_irr[date_stat_irr._ID==i]["IRRDATDEB"].values[0]]["NDVI"].values) # check NDVI value date_start_irr
        NDVI_stat_irri.append(ID_data[ID_data.date==date_stat_irr[date_stat_irr._ID==i]["IRRDATDEB"].values[0]]["NDVI"].values[0])
        if ID_data.Ir_auto.sum() != 0: 
            NDVI_stat_irri_auto.append(ID_data[ID_data.Ir_auto!=0]["NDVI"].values[0])
        else:
            continue
        # print(ID_data[ID_data.Ir_auto!=0]) " print date irrigation automatique"
        IRR.append([i,ID_data.Ir_auto.sum(),val.values[0],ID_data.TAW.max()])
    tab_irr=pd.DataFrame(IRR,columns=["ID","conso","maxzr","TAWMax"])
    tab_irr2=pd.merge(tab_irr,data_prof[["ID","MMEAU","Classe_Bruand"]],on='ID')
    tab_irr2.drop_duplicates(inplace=True)
    df_PKGC=tab_irr2
    #  Isoler les parcelles en fonction du Sol
    list_drop=[7,9,10,13,25,29,34,50,54,61,83,90,98]
    data_valid_PKGC =data_valid_PKGC[-data_valid_PKGC["ID"].isin(list_drop)]
    df_PKGC["MMEAU"]=data_valid_PKGC.MMEAU.values

    param=pd.read_csv(d["PC_disk"]+"/TRAITEMENT/RUNS_SAMIR/RUN_PKGC/GERS/CACG_Fcover_GSM_irri_auto_v2/2017/Output/maxZr/output_test_maize_irri_param.txt",header=None,skiprows=1,sep=";")
    IRR=[]
    ids=[]
    df_CACG_valid= pd.read_csv(d["PC_disk"]+"TRAITEMENT/RUNS_SAMIR/RUN_CACG/maxZr_rum/tab_resu_CACG_maxZr_inversion_RUM_2017.csv")
    data_prof=pd.read_csv(d["PC_disk"]+"/TRAITEMENT/SOIL/SOIL_RIGOU/Extract_RRP_Rigou_parcelle_CACG_"+str(y)+"_UTS_maj.csv",sep=';',encoding='latin-1',decimal=',')
    data_prof =data_prof[data_prof["ID"].isin(list(df_CACG_valid.ID))]
    for i in data_prof.ID:
        c=param.loc[param[1].isin(data_prof.loc[data_prof.ID==i]["maxZr_UTS"])][0]
        val=param.loc[param[1].isin(data_prof.loc[data_prof.ID==i]["maxZr_UTS"])][1]
        UTS=pickle.load(open(d["PC_disk"]+"/TRAITEMENT/RUNS_SAMIR/RUN_PKGC/GERS/CACG_Fcover_GSM_irri_auto_v2/2017/Output/maxZr/output_test_maize_irri_"+str(int(c))+".df","rb"))
        data_id=UTS.groupby("id")
        ID_data=data_id.get_group(i)
        IRR.append([i,ID_data.Ir_auto.sum(),val.values[0],ID_data.TAW.max()])
    tab_irr=pd.DataFrame(IRR,columns=["ID","conso","maxzr","TAWMax"])
    df_CACG=tab_irr
    df_CACG_valid.columns=["unnamed","ID","Quant","MaxZr",'TAWmax','MMEAU']
    df_CACG_valid=df_CACG_valid[["ID","MMEAU","MaxZr"]]
    df_CACG["MMEAU"]=df_CACG_valid.MMEAU.values

    df_plot=df_PKGC.drop("ID",axis=1)
    df=df_plot.append(df_CACG.drop('ID',axis=1))
    
    # df_UTS_max=pd.concat([df_PKGC["maxZr_UTS"],df_CACG["maxZr_UTS"]])
    plt.figure(figsize=(7,7))
    slope, intercept, r_value, p_value, std_err = stats.linregress(df.MMEAU.to_list(),df.conso.to_list())
    bias=1/df["MMEAU"].shape[0]*sum(df.conso-np.mean(df.MMEAU)) 
    rms = np.sqrt(mean_squared_error(df.MMEAU,df.conso))
    plt.scatter(df.MMEAU,df.conso)
    plt.xlim(-10,350)
    plt.ylim(-10,350)
    plt.xlabel("Quantité saisonnière observée en mm ")
    plt.ylabel("Quantité saisonnière modélisée en mm ")
    plt.plot([-10.0, 350], [-10.0,350], 'black', lw=1,linestyle='--')
    rectangle = plt.Rectangle((95, 245),70,45, ec='blue',fc='blue',alpha=0.1)
    plt.gca().add_patch(rectangle)
    plt.text(100,280,"RMSE = "+str(round(rms,2))) 
    plt.text(100,270,"R² = "+str(round(r_value,2)))
    plt.text(100,260,"Pente = "+str(round(slope,2)))
    plt.text(100,250,"Biais = "+str(round(bias,2)))
    # for i in enumerate(df_plot.index):
    #     label = int(i[1])
    #     plt.annotate(label, # this is the text
    #           (df_plot["MMEAU"].iloc[i[0]],df_plot.Quant.iloc[i[0]]), # this is the point to label
    #           textcoords="offset points", # how to position the text
    #           xytext=(-6,2), # distance from text to points (x,y)
    #           ha='center')
    plt.savefig("/run/media/pageot/Transcend/Yann_THESE/BESOIN_EAU/BESOIN_EAU/TRAITEMENT/RUNS_SAMIR/Plot_resu_all_parcelle_partenaires/scatter_maxZrUTS_PF_CC_GSM_with_FCOVER.png")



# =============================================================================
# # SAs le Fcover
# =============================================================================
    param=pd.read_csv(d["PC_disk"]+"/TRAITEMENT/RUNS_SAMIR/RUN_PKGC/GERS/PKGC_GSM_irri_auto_v2__plt_30/2017/Output/maxZr/output_test_maize_irri_param.txt",header=None,skiprows=1,sep=";")
    IRR=[]
    ids=[]
    data_prof=pd.read_csv(d["PC_disk"]+"/TRAITEMENT/SOIL/GSM/Extract_GSM_parcelle_PKGC_GERS_2017_GSM_PF_CC_class_name.csv",index_col=[0],sep=',',encoding='latin-1',decimal=',')
    list_drop=[7,9,10,13,25,29,34,50,54,61,83,90,98]
    data_prof =data_prof[-data_prof["ID"].isin(list_drop)]
    for i in data_prof.ID:
        c=param.loc[param[1].isin(data_prof.loc[data_prof.ID==i]["Zrmax_UTS"])][0]
        val=param.loc[param[1].isin(data_prof.loc[data_prof.ID==i]["Zrmax_UTS"])][1]
        UTS=pickle.load(open(d["PC_disk"]+"/TRAITEMENT/RUNS_SAMIR/RUN_PKGC/GERS/PKGC_GSM_irri_auto_v2__plt_30/2017/Output/maxZr/output_test_maize_irri_"+str(int(c))+".df","rb"))
        data_id=UTS.groupby("id")
        ID_data=data_id.get_group(i)
        IRR.append([i,ID_data.Ir_auto.sum(),val.values[0],ID_data.TAW.max()])
    tab_irr=pd.DataFrame(IRR,columns=["ID","conso","maxzr","TAWMax"])
    tab_irr2=pd.merge(tab_irr,data_prof[["ID","MMEAU","Classe_Bruand"]],on='ID')
    tab_irr2.drop_duplicates(inplace=True)
    df_PKGC=tab_irr2
    #  Isoler les parcelles en fonction du Sol
    list_drop=[7,9,10,13,25,29,34,50,54,61,83,90,98]
    data_valid_PKGC =data_valid_PKGC[-data_valid_PKGC["ID"].isin(list_drop)]
    df_PKGC["MMEAU"]=data_valid_PKGC.MMEAU.values


   
    param=pd.read_csv(d["PC_disk"]+"/TRAITEMENT/RUNS_SAMIR/RUN_PKGC/GERS/CACG_GSM_irri_auto_v2_plt_30/2017/Output/maxZr/output_test_maize_irri_param.txt",header=None,skiprows=1,sep=";")
    IRR=[]
    ids=[]
    df_CACG_valid= pd.read_csv(d["PC_disk"]+"TRAITEMENT/RUNS_SAMIR/RUN_CACG/maxZr_rum/tab_resu_CACG_maxZr_inversion_RUM_2017.csv")
    data_prof=pd.read_csv(d["PC_disk"]+"/TRAITEMENT/SOIL/SOIL_RIGOU/Extract_RRP_Rigou_parcelle_CACG_"+str(y)+"_UTS_maj.csv",sep=';',encoding='latin-1',decimal=',')
    data_prof =data_prof[data_prof["ID"].isin(list(df_CACG_valid.ID))]
    for i in data_prof.ID:
        c=param.loc[param[1].isin(data_prof.loc[data_prof.ID==i]["maxZr_UTS"])][0]
        val=param.loc[param[1].isin(data_prof.loc[data_prof.ID==i]["maxZr_UTS"])][1]
        UTS=pickle.load(open(d["PC_disk"]+"/TRAITEMENT/RUNS_SAMIR/RUN_PKGC/GERS/CACG_GSM_irri_auto_v2_plt_30/2017/Output/maxZr/output_test_maize_irri_"+str(int(c))+".df","rb"))
        data_id=UTS.groupby("id")
        ID_data=data_id.get_group(i)
        IRR.append([i,ID_data.Ir_auto.sum(),val.values[0],ID_data.TAW.max()])
    tab_irr=pd.DataFrame(IRR,columns=["ID","conso","maxzr","TAWMax"])
    df_CACG=tab_irr
    df_CACG_valid.columns=["unnamed","ID","conso","maxZr",'TAWmax','MMEAU']
    df_CACG_valid=df_CACG_valid[["ID","MMEAU","maxZr"]]
    df_CACG["MMEAU"]=df_CACG_valid.MMEAU.values

    df_plot=df_PKGC.drop("ID",axis=1)
    df=df_plot.append(df_CACG.drop('ID',axis=1))
    
    # df_UTS_max=pd.concat([df_PKGC["maxZr_UTS"],df_CACG["maxZr_UTS"]])
    plt.figure(figsize=(7,7))
    slope, intercept, r_value, p_value, std_err = stats.linregress(df.MMEAU.to_list(),df.conso.to_list())
    bias=1/df["MMEAU"].shape[0]*sum(df.conso-np.mean(df.MMEAU)) 
    rms = np.sqrt(mean_squared_error(df.MMEAU,df.conso))
    plt.scatter(df.MMEAU,df.conso)
    plt.xlim(-10,350)
    plt.ylim(-10,350)
    plt.xlabel("Quantité saisonnière observée en mm ")
    plt.ylabel("Quantité saisonnière modélisée en mm ")
    plt.plot([-10.0, 350], [-10.0,350], 'black', lw=1,linestyle='--')
    rectangle = plt.Rectangle((95, 245),70,45, ec='blue',fc='blue',alpha=0.1)
    plt.gca().add_patch(rectangle)
    plt.text(100,280,"RMSE = "+str(round(rms,2))) 
    plt.text(100,270,"R² = "+str(round(r_value,2)))
    plt.text(100,260,"Pente = "+str(round(slope,2)))
    plt.text(100,250,"Biais = "+str(round(bias,2)))
    # for i in enumerate(df_plot.index):
    #     label = int(i[1])
    #     plt.annotate(label, # this is the text
    #           (df_plot["MMEAU"].iloc[i[0]],df_plot.Quant.iloc[i[0]]), # this is the point to label
    #           textcoords="offset points", # how to position the text
    #           xytext=(-6,2), # distance from text to points (x,y)
    #           ha='center')
    plt.savefig("/run/media/pageot/Transcend/Yann_THESE/BESOIN_EAU/BESOIN_EAU/TRAITEMENT/RUNS_SAMIR/Plot_resu_all_parcelle_partenaires/scatter_maxZrUTS_PF_CC_GSM_sans_FCOVER.png")