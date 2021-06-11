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
from trianglegraph import SoilTrianglePlot

def predict(x):
   return slope * x + intercept




if __name__ == '__main__':
    d={}
    d["PC_disk"]="/run/media/pageot/Transcend/Yann_THESE/BESOIN_EAU/BESOIN_EAU/"
    d["PC_home"]="/mnt/d/THESE_TMP/"
    d["PC_home_Wind"]="D:/THESE_TMP/"
    # d["PC_disk"]="H:/Yann_THESE/BESOIN_EAU/BESOIN_EAU/"
    d["PC_disk"]="/run/media/pageot/Transcend/Yann_THESE/BESOIN_EAU/BESOIN_EAU/"
    d["PC_labo"]="/datalocal/vboxshare/THESE/BESOIN_EAU/"
    years=["2017"]
    lc="maize_irri"
# =============================================================================
#     Extration model TAW et DR et maxZr
# =============================================================================
    # data_prof=pd.read_csv(d["PC_disk"]+"/TRAITEMENT/SOIL/SOIL_RIGOU/Extract_RRP_Rigou_parcelle_CACG_2017_UTS_maj.csv",index_col=[0],sep=';',encoding='latin-1',decimal='.')
    # param=pd.read_csv(d["PC_disk"]+"/TRAITEMENT/RUNS_SAMIR/RUN_CACG/CACG_GSM_init_ru_optim_P055_Fcover_fewi_De_Kr_days10_dose30_400_1800_irri_auto_soil/2018/Output/maxZr/output_test_maize_irri_param.txt",header=None,skiprows=1,sep=";")
    # dfGSM=pd.read_csv(d["PC_disk"]+"/TRAITEMENT/RUNS_SAMIR/RUN_CACG/CACG_GSM_init_ru_optim_P055_Fcover_fewi_De_Kr_days10_dose30_400_1800_irri_auto_soil/tab_CACG_mod_2018.csv",sep=",")
    # dfRRP=pd.read_csv(d["PC_disk"]+"/TRAITEMENT/RUNS_SAMIR/RUN_CACG/CACG_RRP_init_ru_optim_P055_Fcover_fewi_De_Kr_days10_dose30_400_1800_irri_auto_soil/tab_CACG_mod_2018.csv")
    # dfUTS=pd.read_csv(d["PC_disk"]+"/TRAITEMENT/RUNS_SAMIR/RUN_CACG/CACG_init_ru_optim_P055_Fcover_fewi_De_Kr_days10_dose30_400_1800_irri_auto_soil/tab_CACG_mod_2018.csv")
    # # dfUTSFAO=pd.read_csv("H:/Yann_THESE/BESOIN_EAU/BESOIN_EAU/TRAITEMENT/RUNS_SAMIR/RUN_PKGC/PKGC_FAO_init_ru_optim_Fcover_fewi_De_Kr_days10_dose30_500_800_irri_auto_soil/Table_RMSE_parcelle_min.csv")
    # All_TAW=[]
    # Ir_all=[]
    # quant_irr=[]
    # valeur_maxZr=[]
    # for i in dfUTS.ID:
    #     a=param.loc[param[1].isin(dfGSM.loc[dfGSM.ID==i]["param"])][0]+1
    #     b=param.loc[param[1].isin(dfRRP.loc[dfRRP.ID==i]["param"])][0]+1
    #     c=param.loc[param[1].isin(dfUTS.loc[dfUTS.ID==i]["param"])][0]+1
    #     # d=param.loc[param[1].isin(dfUTSFAO.loc[dfUTSFAO.ID==i]["maxZr"])][0]+1
    #     maxGsm=param.loc[param[1].isin(dfGSM.loc[dfGSM.ID==i]["param"])][1]
    #     maxRRP=param.loc[param[1].isin(dfRRP.loc[dfRRP.ID==i]["param"])][1]
    #     maxUTS=param.loc[param[1].isin(dfUTS.loc[dfUTS.ID==i]["param"])][1]
    #     print(" ID : %s ==> GSM = %s ; RRP = %s  ; UTS = %s "%(i,maxGsm.values[0],maxRRP.values[0],maxUTS.values[0]))
    #     # maxUTSFAO=param.loc[param[1].isin(dfUTSFAO.loc[dfUTSFAO.ID==i]["maxZr"])][1]
    #     UTS=pickle.load(open(d["PC_disk"]+"TRAITEMENT/RUNS_SAMIR/RUN_CACG/CACG_init_ru_optim_P055_Fcover_fewi_De_Kr_days10_dose30_400_1800_irri_auto_soil/2018/Output/maxZr/output_test_maize_irri_"+str(int(c))+".df","rb"))
    #     GSM=pickle.load(open(d["PC_disk"]+"TRAITEMENT/RUNS_SAMIR/RUN_CACG/CACG_GSM_init_ru_optim_P055_Fcover_fewi_De_Kr_days10_dose30_400_1800_irri_auto_soil/2018/Output/maxZr/output_test_maize_irri_"+str(int(a))+".df","rb"))
    #     RRP=pickle.load(open(d["PC_disk"]+"TRAITEMENT/RUNS_SAMIR/RUN_CACG/CACG_RRP_init_ru_optim_P055_Fcover_fewi_De_Kr_days10_dose30_400_1800_irri_auto_soil/2018/Output/maxZr/output_test_maize_irri_"+str(int(b))+".df","rb"))
    #     # UTSFAO=pickle.load(open("H:/Yann_THESE/BESOIN_EAU/BESOIN_EAU/TRAITEMENT/RUNS_SAMIR/RUN_PKGC/PKGC_FAO_init_ru_optim_Fcover_fewi_De_Kr_days10_dose30_500_800_irri_auto_soil/2017/Output/maxZr/output_test_maize_irri_"+str(int(d))+".df","rb"))

    #     TAWdfGSM=GSM.loc[GSM.id==i][["date","Dr",'Ir_auto',"TAW"]]
    #     TAWdfRRP=RRP.loc[RRP.id==i][["date","Dr",'Ir_auto',"TAW"]]
    #     TAWdfUTS=UTS.loc[UTS.id==i][["date","Dr",'Ir_auto',"TAW"]]
    #     # TAWdfUTSFAO=UTSFAO.loc[UTSFAO.id==i][["date","Dr",'Ir_auto',"TAW"]]
    #     All_TAW.append([i,TAWdfUTS.TAW.max(),TAWdfRRP.TAW.max(),TAWdfGSM.TAW.max()])
    #     Ir_all.append([i,TAWdfUTS.Ir_auto[TAWdfUTS.Ir_auto!=0.0].count(),TAWdfRRP.Ir_auto[TAWdfRRP.Ir_auto!=0.0].count(),TAWdfGSM.Ir_auto[TAWdfGSM.Ir_auto!=0.0].count()])
    #     quant_irr.append([i,TAWdfUTS.Ir_auto[TAWdfUTS.Ir_auto!=0.0].sum(),TAWdfRRP.Ir_auto[TAWdfRRP.Ir_auto!=0.0].sum(),TAWdfGSM.Ir_auto[TAWdfGSM.Ir_auto!=0.0].sum()])
    #     valeur_maxZr.append([i,maxGsm.values[0],maxRRP.values[0],maxUTS.values[0]])
    #       # plot dynamique 
    #     plt.figure(figsize=(7,7))
    #     plt.plot(TAWdfGSM.date,TAWdfGSM.TAW,c='Blue',label="GSM")
    #     plt.plot(TAWdfGSM.date,TAWdfGSM.Dr,c="Blue",label='Dr',linestyle="--",linewidth=0.9)
    #     plt.plot(TAWdfRRP.date,TAWdfRRP.TAW,c='r',label="RRP")
    #     plt.plot(TAWdfGSM.date,TAWdfRRP.Dr,c="r",linestyle="--",linewidth=0.9)
    #     plt.plot(TAWdfUTS.date,TAWdfUTS.TAW,c="Green",label="UTS")
    #     plt.plot(TAWdfGSM.date,TAWdfUTS.Dr,c="g",linestyle="--",linewidth=0.9)
    #     plt.title(i)
    #     plt.legend()
    #     # plt.savefig("H:/Yann_THESE/BESOIN_EAU/BESOIN_EAU/TRAITEMENT/RUNS_SAMIR/RUN_PKGC/PKGC_init_ru_optim_Fcover_fewi_De_Kr_days10_dose50_500_800_irri_auto_soil/dynamique_TAW_3_soil_%s.png"%int(i))
    # TAW_all=pd.DataFrame(All_TAW)
    # TAW_all.columns=["ID","UTS","RRP","GSM"]
    # fig, ax = plt.subplots(figsize=(10, 5))
    # ax1=plt.subplot(131)
    # plt.hist(TAW_all.UTS,bins=20)
    # plt.gca().xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(5))
    # plt.xlabel("TAW")
    # plt.xlim(50,180)
    # plt.title("TAW UTS")
    # ax2=plt.subplot(132)
    # plt.hist(TAW_all.RRP,bins=20)
    # plt.xlabel("TAW")
    # plt.xlim(50,180)
    # plt.gca().xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(5))
    # plt.title("TAW RRP")
    # ax3=plt.subplot(133)
    # plt.hist(TAW_all.GSM,bins=20)
    # plt.gca().xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(5))
    # plt.xlabel("TAW")
    # plt.xlim(50,180)
    # plt.title("TAW GSM")
    # # plt.savefig("H:/Yann_THESE/BESOIN_EAU/BESOIN_EAU/TRAITEMENT/RUNS_SAMIR/RUN_PKGC/PKGC_init_ru_optim_Fcover_fewi_De_Kr_days10_dose50_500_800_irri_auto_soil/hist_TAW_type_sol_data.png")
    # ALL_nb_Irr=pd.DataFrame(Ir_all)
    # ALL_nb_Irr.columns=["ID",'UTS',"RRP",'GSM']
    # # ALL_nb_Irr.to_csv("H:/Yann_THESE/BESOIN_EAU/BESOIN_EAU/TRAITEMENT/RUNS_SAMIR/RUN_PKGC/PKGC_init_ru_optim_Fcover_fewi_De_Kr_days10_dose50_500_800_irri_auto_soil/NB_IRR_sol_data.csv")
    # Quant_irr=pd.DataFrame(quant_irr,columns=["ID","UTS","RRP","GSM"])
    # # Quant_irr.to_csv("H:/Yann_THESE/BESOIN_EAU/BESOIN_EAU/TRAITEMENT/RUNS_SAMIR/RUN_PKGC/PKGC_init_ru_optim_Fcover_fewi_De_Kr_days10_dose50_500_800_irri_auto_soil/Quant_IRR_sol_data.csv")
    # val_max=pd.DataFrame(valeur_maxZr,columns=["ID","GSM","RRP","UTS"])
    # data_SOIL=pd.merge(val_max,data_prof["ProfRacPot"]*10,on=["ID"])
    
    # slopeGSM, intercept, r_valueGSM, p_value, std_err = stats.linregress(data_SOIL.ProfRacPot.to_list(),data_SOIL.GSM.to_list())
    # biasGSM=1/data_SOIL["ProfRacPot"].shape[0]*sum(data_SOIL.GSM-np.mean(data_SOIL.ProfRacPot)) 
    # rmsGSM = np.sqrt(mean_squared_error(data_SOIL.ProfRacPot,data_SOIL.GSM))
    # slopeRRP, intercept, r_valueRRP, p_value, std_err = stats.linregress(data_SOIL.ProfRacPot.to_list(),data_SOIL.RRP.to_list())
    # biasRRP=1/data_SOIL["ProfRacPot"].shape[0]*sum(data_SOIL.RRP-np.mean(data_SOIL.ProfRacPot)) 
    # rmsRRP = np.sqrt(mean_squared_error(data_SOIL.ProfRacPot,data_SOIL.RRP))
    # slopeUTS, intercept, r_valueUTS, p_value, std_err = stats.linregress(data_SOIL.ProfRacPot.to_list(),data_SOIL.UTS.to_list())
    # biasUTS=1/data_SOIL["ProfRacPot"].shape[0]*sum(data_SOIL.UTS-np.mean(data_SOIL.ProfRacPot)) 
    # rmsUTS = np.sqrt(mean_squared_error(data_SOIL.ProfRacPot,data_SOIL.UTS))
    # plt.figure(figsize=(7,7))
    # plt.scatter(data_SOIL.ProfRacPot,data_SOIL.GSM,c='r',label="GSM",marker="x")
    # plt.scatter(data_SOIL.ProfRacPot,data_SOIL.RRP,c='b',label="RRP",marker="+")
    # plt.scatter(data_SOIL.ProfRacPot,data_SOIL.UTS,c='g',label="UTS",marker='1')
    # plt.xlim(0,2000)
    # plt.ylim(0,2000)
    # plt.legend()
    # plt.xlabel("MaxZr obs.")
    # plt.ylabel("MaxZr mod.")
    # plt.plot([0, 2000], [0.0,2000], 'black', lw=1,linestyle='--')
    # rectangle = plt.Rectangle((480, 1730),430,220, ec='r',fc='r',alpha=0.1)
    # plt.gca().add_patch(rectangle)
    # plt.text(500,1900,"RMSE = "+str(round(rmsGSM,2))) 
    # plt.text(500,1850,"R² = "+str(round(r_valueGSM,2)))
    # plt.text(500,1800,"Pente = "+str(round(slopeGSM,2)))
    # plt.text(500,1750,"Biais = "+str(round(biasGSM,2)))
    # rectangle = plt.Rectangle((1380, 1530),430,220, ec='b',fc='b',alpha=0.1)
    # plt.gca().add_patch(rectangle)
    # plt.text(1400,1700,"RMSE = "+str(round(rmsRRP,2))) 
    # plt.text(1400,1650,"R² = "+str(round(r_valueRRP,2)))
    # plt.text(1400,1600,"Pente = "+str(round(slopeRRP,2)))
    # plt.text(1400,1550,"Biais = "+str(round(biasRRP,2)))
    # rectangle = plt.Rectangle((1480, 230),430,220, ec='g',fc='g',alpha=0.1)
    # plt.gca().add_patch(rectangle)
    # plt.text(1500,400,"RMSE = "+str(round(rmsUTS,2))) 
    # plt.text(1500,350,"R² = "+str(round(r_valueUTS,2)))
    # plt.text(1500,300,"Pente = "+str(round(slopeUTS,2)))
    # plt.text(1500,250,"Biais = "+str(round(biasUTS,2)))
    # plt.savefig(d['PC_disk']+"/TRAITEMENT/RUNS_SAMIR/RUN_CACG/Plot_result/scatter_MAXZR_type_sol_data.png")
    
# =============================================================================
#  UTS maps maize 2017/2018 RPG extart maxZr pot
# =============================================================================
    data_prof=pd.read_csv(d["PC_disk"]+"/TRAITEMENT/SOIL/SOIL_RIGOU/Extract_RRP_Rigou_parcelle_PKCG_2017_UTS_maj.csv",index_col=[0],sep=';',encoding='latin-1',decimal=',')
    
    plt.figure(figsize=(7,7))
    for y in ["2017","2018"]:
        data_prof_CACG=pd.read_csv(d["PC_disk"]+"/TRAITEMENT/SOIL/SOIL_RIGOU/Extract_RRP_Rigou_parcelle_CACG_"+str(y)+"_UTS_maj.csv",index_col=[0],sep=';',encoding='latin-1',decimal='.')
        df=pd.read_csv(d["PC_disk"]+"/TRAITEMENT/SOIL/SOIL_RIGOU/RPG_maize_"+str(y)+"_soil_gers_Rigou.csv")
        print(median(df.RUM))
        plt.hist(df.RUM,label=y,bins=10)
        plt.plot(data_prof_CACG.RUM,np.repeat(2000,np.shape(data_prof_CACG)[0]),label='CACG parcelle',marker="o",linestyle="",c='r')
        plt.legend()
    plt.plot(data_prof.RUM,np.repeat(3000,np.shape(data_prof)[0]),label="Parcelle PKGC",marker='o',linestyle="",c='b')
    plt.legend()
    plt.xlabel("Profondeur de la RUM en cm")
    plt.savefig(d["PC_disk"]+"/TRAITEMENT/SOIL/Plot_Sol/histo_RUM_RPG.png")

    
    plt.figure(figsize=(7,7))
    for y in ["2017","2018"]:
        data_prof_CACG=pd.read_csv(d["PC_disk"]+"/TRAITEMENT/SOIL/SOIL_RIGOU/Extract_RRP_Rigou_parcelle_CACG_"+str(y)+"_UTS_maj.csv",index_col=[0],sep=';',encoding='latin-1',decimal='.')
        df=pd.read_csv(d["PC_disk"]+"/TRAITEMENT/SOIL/SOIL_RIGOU/RPG_maize_"+str(y)+"_soil_gers_Rigou.csv")
        print(median(df.ProfRacPot))
        plt.hist(df.ProfRacPot,label=y,bins=10)
        plt.plot(data_prof_CACG.ProfRacPot,np.repeat(2000,np.shape(data_prof_CACG)[0]),label='CACG parcelle',marker="o",linestyle="",c='r')
        plt.legend()
    plt.plot(data_prof.ProfRacPot,np.repeat(3000,np.shape(data_prof)[0]),label="Parcelle PKGC",marker='o',linestyle="",c='b')
    plt.legend()
    plt.xlabel("Profondeur de la maxZr en cm")
    plt.savefig(d["PC_disk"]+"/TRAITEMENT/SOIL/Plot_Sol/histo_maxZr_RPG.png")
    
    plt.figure(figsize=(7,7))
    for y in ["2017","2018"]:
        plt.hist(data_prof_CACG.RUM,label=y)
        print(median(data_prof_CACG.RUM))
        plt.legend()
    plt.hist(data_prof.RUM,label='PKGC')
    print(median(data_prof.RUM))
    
    
# =============================================================================
#     Variation soil forcage p et maxZr à 600
# =============================================================================
    Valid=pd.read_csv(d["PC_disk"]+"/TRAITEMENT/RUNS_SAMIR/RUN_PKGC/PKGC_init_ru_optim_Fcover_fewi_De_Kr_days10_dose30_500_800_irri_auto_soil/Table_RMSE_parcelle_min.csv")
    dfGSM=pickle.load(open(d["PC_disk"]+"/TRAITEMENT/RUNS_SAMIR/RUN_PKGC/Sans_optim/PKGC_GSM_init_ru_optim_P055_Fcover_fewi_De_Kr_days10_dose30_1000_irri_auto_soil/2017/output_test_2017.df","rb"))
    dfRRP=pickle.load(open(d["PC_disk"]+"/TRAITEMENT/RUNS_SAMIR/RUN_PKGC/Sans_optim/PKGC_RRP_init_ru_optim_P055_Fcover_fewi_De_Kr_days10_dose30_1000_irri_auto_soil/2017/output_test_2017.df","rb"))
    dfUTS=pickle.load(open(d["PC_disk"]+"/TRAITEMENT/RUNS_SAMIR/RUN_PKGC/Sans_optim/PKGC_init_ru_optim_P055_Fcover_fewi_De_Kr_days10_dose30_1000_irri_auto_soil/2017/output_test_2017.df","rb"))
    mod_irr_GSM=dfGSM.groupby("id")["Ir_auto"].sum()
    mod_Irr_GSM=pd.DataFrame(mod_irr_GSM)
    mod_Irr_GSM = mod_Irr_GSM[mod_irr_GSM.index!=10]
    mod_irr_RRP=dfRRP.groupby("id")["Ir_auto"].sum()
    mod_Irr_RRP=pd.DataFrame(mod_irr_RRP)
    mod_Irr_RRP = mod_Irr_RRP[mod_irr_RRP.index!=10]
    mod_irr_UTS=dfUTS.groupby("id")["Ir_auto"].sum()
    mod_Irr_UTS=pd.DataFrame(mod_irr_UTS)
    mod_Irr_UTS = mod_Irr_UTS[mod_irr_UTS.index!=10]
    #  TAW
    mod_TAW_GSM=dfGSM.groupby("id")["TAW"].max()
    mod_TAW_GSM=pd.DataFrame(mod_TAW_GSM,columns=["ID","TAW"])
    mod_TAW_GSM.ID=mod_TAW_GSM.index
    mod_TAW_GSM = mod_TAW_GSM[mod_TAW_GSM.index!=10]
    mod_TAW_RRP=dfRRP.groupby("id")["TAW"].max()
    mod_TAW_RRP=pd.DataFrame(mod_TAW_RRP,columns=["ID","TAW"])
    mod_TAW_RRP.ID=mod_TAW_RRP.index
    mod_TAW_RRP = mod_TAW_RRP[mod_TAW_RRP.index!=10]
    mod_TAW_UTS=dfUTS.groupby("id")["TAW"].max()
    mod_TAW_UTS=pd.DataFrame(mod_TAW_UTS,columns=["ID","TAW"])
    mod_TAW_UTS.ID=mod_TAW_UTS.index
    mod_TAW_UTS = mod_TAW_UTS[mod_TAW_UTS.index!=10]

    slopeGSM, intercept, r_valueGSM, p_value, std_err = stats.linregress(Valid.MMEAU.to_list(),mod_Irr_GSM.Ir_auto.to_list())
    biasGSM=1/Valid["MMEAU"].shape[0]*sum(mod_Irr_GSM.Ir_auto-np.mean(Valid.MMEAU)) 
    rmsGSM = np.sqrt(mean_squared_error(Valid.MMEAU,mod_Irr_GSM.Ir_auto))
    slopeRRP, intercept, r_valueRRP, p_value, std_err = stats.linregress(Valid.MMEAU.to_list(),mod_Irr_RRP.Ir_auto.to_list())
    biasRRP=1/Valid["MMEAU"].shape[0]*sum(mod_Irr_RRP.Ir_auto-np.mean(Valid.MMEAU)) 
    rmsRRP = np.sqrt(mean_squared_error(Valid.MMEAU,mod_Irr_RRP.Ir_auto))
    slopeUTS, intercept, r_valueUTS, p_value, std_err = stats.linregress(Valid.MMEAU.to_list(),mod_Irr_UTS.Ir_auto.to_list())
    biasUTS=1/Valid["MMEAU"].shape[0]*sum(mod_Irr_UTS.Ir_auto-np.mean(Valid.MMEAU)) 
    rmsUTS = np.sqrt(mean_squared_error(Valid.MMEAU,mod_Irr_UTS.Ir_auto))
    plt.figure(figsize=(7,7))
    plt.scatter(Valid.MMEAU, mod_Irr_GSM, marker="+",label='GSM',color='r',s=50)
    plt.scatter(Valid.MMEAU, mod_Irr_RRP, marker="1",label='RRP',color='b',s=50)
    plt.scatter(Valid.MMEAU, mod_Irr_UTS, marker="x",label='UTS',color='g',s=50)
    plt.xlim(0,350)
    plt.ylim(0,350)
    plt.legend()
    plt.xlabel("Quantité annuelles observées en mm ")
    plt.ylabel("Quantité annuelles modélisées en mm ")
    plt.plot([0, 350], [0.0,350], 'black', lw=1,linestyle='--')
    rectangle = plt.Rectangle((75, 265),70,50, ec='r',fc='r',alpha=0.1)
    plt.gca().add_patch(rectangle)
    plt.text(80,300,"RMSE = "+str(round(rmsGSM,2))) 
    plt.text(80,290,"R² = "+str(round(r_valueGSM,2)))
    plt.text(80,280,"Pente = "+str(round(slopeGSM,2)))
    plt.text(80,270,"Biais = "+str(round(biasGSM,2)))
    rectangle = plt.Rectangle((215, 275),70,50, ec='b',fc='b',alpha=0.1)
    plt.gca().add_patch(rectangle)
    plt.text(220,310,"RMSE = "+str(round(rmsRRP,2))) 
    plt.text(220,300,"R² = "+str(round(r_valueRRP,2)))
    plt.text(220,290,"Pente = "+str(round(slopeRRP,2)))
    plt.text(220,280,"Biais = "+str(round(biasRRP,2)))
    rectangle = plt.Rectangle((245, 95),70,50, ec='g',fc='g',alpha=0.1)
    plt.gca().add_patch(rectangle)
    plt.text(250,130,"RMSE = "+str(round(rmsUTS,2))) 
    plt.text(250,120,"R² = "+str(round(r_valueUTS,2)))
    plt.text(250,110,"Pente = "+str(round(slopeUTS,2)))
    plt.text(250,100,"Biais = "+str(round(biasUTS,2)))
    
    # plt.savefig(d["PC_disk"]+"/TRAITEMENT/SOIL/Plot_Sol/scatter_irrigation_tot_GSM_RRP_UTS_PKGC.png")
# =============================================================================
#     plot RUM forcage 600 et p 
# =============================================================================
    data_prof=pd.read_csv(d["PC_disk"]+"/TRAITEMENT/SOIL/SOIL_RIGOU/Extract_RRP_Rigou_parcelle_PKCG_2017_UTS_maj.csv",index_col=[0],sep=';',encoding='latin-1',decimal=',')
    testGSM=pd.merge(data_prof["RUM"],mod_TAW_GSM,on='ID')
    testRRP=pd.merge(data_prof["RUM"],mod_TAW_RRP,on='ID')
    testUTS=pd.merge(data_prof["RUM"],mod_TAW_UTS,on='ID')
    
    slopeGSM, intercept, r_valueGSM, p_value, std_err = stats.linregress(testGSM.RUM.to_list(),testGSM.TAW.to_list())
    biasGSM=1/testGSM["RUM"].shape[0]*sum(testGSM.TAW-np.mean(testGSM.RUM)) 
    rmsGSM = np.sqrt(mean_squared_error(testGSM.RUM,testGSM.TAW))
    slopeRRP, intercept, r_valueRRP, p_value, std_err = stats.linregress(testGSM.RUM.to_list(),testRRP.TAW.to_list())
    biasRRP=1/testGSM["RUM"].shape[0]*sum(testRRP.TAW-np.mean(testGSM.RUM)) 
    rmsRRP = np.sqrt(mean_squared_error(testGSM.RUM,testRRP.TAW))
    slopeUTS, intercept, r_valueUTS, p_value, std_err = stats.linregress(testGSM.RUM.to_list(),testUTS.TAW.to_list())
    biasUTS=1/testGSM["RUM"].shape[0]*sum(testUTS.TAW-np.mean(testGSM.RUM)) 
    rmsUTS = np.sqrt(mean_squared_error(testGSM.RUM,testUTS.TAW))
    plt.figure(figsize=(7,7))
    plt.scatter(testGSM.RUM, mod_Irr_GSM, marker="+",label='GSM',color='r',s=50)
    plt.scatter(testGSM.RUM, mod_Irr_RRP, marker="1",label='RRP',color='b',s=50)
    plt.scatter(testGSM.RUM, mod_Irr_UTS, marker="x",label='UTS',color='g',s=50)
    plt.xlim(0,300)
    plt.ylim(0,300)
    plt.legend()
    plt.xlabel("Quantité annuelles observées en mm ")
    plt.ylabel("Quantité annuelles modélisées en mm ")
    plt.plot([0, 300], [0.0,300], 'black', lw=1,linestyle='--')
    rectangle = plt.Rectangle((55, 245),70,50, ec='r',fc='r',alpha=0.1)
    plt.gca().add_patch(rectangle)
    plt.text(60,280,"RMSE = "+str(round(rmsGSM,2))) 
    plt.text(60,270,"R² = "+str(round(r_valueGSM,2)))
    plt.text(60,260,"Pente = "+str(round(slopeGSM,2)))
    plt.text(60,250,"Biais = "+str(round(biasGSM,2)))
    rectangle = plt.Rectangle((215, 255),70,50, ec='b',fc='b',alpha=0.1)
    plt.gca().add_patch(rectangle)
    plt.text(220,290,"RMSE = "+str(round(rmsRRP,2))) 
    plt.text(220,280,"R² = "+str(round(r_valueRRP,2)))
    plt.text(220,270,"Pente = "+str(round(slopeRRP,2)))
    plt.text(220,260,"Biais = "+str(round(biasRRP,2)))
    rectangle = plt.Rectangle((225, 95),70,50, ec='g',fc='g',alpha=0.1)
    plt.gca().add_patch(rectangle)
    plt.text(230,130,"RMSE = "+str(round(rmsUTS,2))) 
    plt.text(230,120,"R² = "+str(round(r_valueUTS,2)))
    plt.text(230,110,"Pente = "+str(round(slopeUTS,2)))
    plt.text(230,100,"Biais = "+str(round(biasUTS,2)))
    # plt.savefig(d["PC_disk"]+"/TRAITEMENT/SOIL/Plot_Sol/scatter_RUM_GSM_RRP_UTS_PKGC.png")
# =============================================================================
#      Triangle texture
# =============================================================================
# data_prof=pd.read_csv(d["PC_disk"]+"/TRAITEMENT/SOIL/SOIL_RIGOU/Extract_RRP_Rigou_parcelle_PKCG_2017_UTS_maj.csv",index_col=[0],sep=';',encoding='latin-1',decimal=',')
# tex_UTS=data_prof[["Argile","Sable","Limon"]]
# tex_UTS.columns=["clay","sand",'silt']
# tex_UTS.to_csv(d["PC_disk"]+"/TRAITEMENT/SOIL/tex_UTS.csv")
plt.figure(figsize=(7,7))
fstp = SoilTrianglePlot()
fstp.soil_categories(country="Ainse")
fstp.scatter_from_csv(d["PC_disk"]+"/TRAITEMENT/SOIL/tex_RRP.csv",hue='nb', cmap=cm.copper_r, alpha=1,s=50,marker="x")
fstp.scatter_from_csv(d["PC_disk"]+"/TRAITEMENT/SOIL/tex_GSM.csv",hue='nb', cmap=cm.copper_r, alpha=1,s=50,marker="+")
fstp.scatter_from_csv(d["PC_disk"]+"/TRAITEMENT/SOIL/tex_UTS.csv",hue="nb", cmap=cm.copper_r, alpha=1,s=50,marker="o")
fstp.colorbar('nombre de parcelles')
fstp.line((0,0,0),(0, 0, 0), 'black', label='RRP', marker="x",linestyle='',linewidth=0.5) 
fstp.line((0,0,0),(0, 0, 0), 'black', label='GSM', marker="+",linestyle='',linewidth=1,alpha=0.5) 
fstp.line((0,0,0),(0, 0, 0), 'black', label='UTS', marker="o",linestyle='',linewidth=1,alpha=0.5)
plt.legend()


# plt.savefig(d["PC_disk"]+"/TRAITEMENT/SOIL/triangle_tex_PKGC_UTS.png")


# data_prof=pd.read_csv(d["PC_disk"]+"/TRAITEMENT/SOIL/RRP/Extract_RRP_GERS_parcelle_PKCG_2017_UTS_maj.csv",index_col=[0],sep=';',encoding='latin-1',decimal=',')
# tex_RRP=data_prof[["Argile","Sable","Limon"]]
# tex_RRP.columns=["clay","sand",'silt']
# tex_RRP.to_csv(d["PC_disk"]+"/TRAITEMENT/SOIL/tex_RRP.csv")
plt.figure(figsize=(7,7))
fstp = SoilTrianglePlot('Données RRP')
fstp.soil_categories(country="Ainse")
fstp.scatter_from_csv(d["PC_disk"]+"/TRAITEMENT/SOIL/tex_RRP.csv",hue='nb', cmap=cm.copper_r, alpha=1,s=50)
fstp.colorbar('nombre de parcelles')
plt.savefig(d["PC_disk"]+"/TRAITEMENT/SOIL/triangle_tex_PKGC_RRP.png")
    
# data_prof=pd.read_csv(d["PC_disk"]+"/TRAITEMENT/SOIL/GSM/Extract_GSM_parcelle_PKCG_2017_UTS_maj.csv",index_col=[0],sep=';',encoding='latin-1',decimal=',')
# tex_GSM=data_prof[["Argile","Sable","Limon"]]
# tex_GSM.columns=["clay","sand",'silt']
# tex_GSM.to_csv(d["PC_disk"]+"/TRAITEMENT/SOIL/tex_GSM.csv")
plt.figure(figsize=(7,7))
fstp = SoilTrianglePlot('Données GSM')
fstp.soil_categories(country="Ainse")
fstp.scatter_from_csv(d["PC_disk"]+"/TRAITEMENT/SOIL/tex_GSM.csv",hue='nb', cmap=cm.copper_r, alpha=1,s=50)
fstp.colorbar('nombre de parcelles')
plt.savefig(d["PC_disk"]+"/TRAITEMENT/SOIL/triangle_tex_PKGC_GSM.png")