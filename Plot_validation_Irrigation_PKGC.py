# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 10:47:32 2021

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
    name_run="RUNS_SAMIR/RUN_PKGC/PKGC_init_ru_optim_Fcover_fewi_De_Kr_days10_p06_1500_modif_irri_auto_soil/"
    name_run_save_fig="RUNS_SAMIR/RUN_PKGC/PKGC_init_ru_optim_Fcover_fewi_De_Kr_days10_p06_1500_modif_irri_auto_soil/"
    # d["PC_disk"]="/run/media/pageot/Transcend/Yann_THESE/BESOIN_EAU/BESOIN_EAU/"
    d["PC_home"]="/mnt/d/THESE_TMP/"
    d["PC_home_Wind"]="D:/THESE_TMP/"
    # d["PC_disk"]="H:/Yann_THESE/BESOIN_EAU/BESOIN_EAU/"
    d["PC_disk"]="H:/Yann_THESE/BESOIN_EAU/BESOIN_EAU/"
    d["PC_labo"]="/datalocal/vboxshare/THESE/BESOIN_EAU/"
    # label="Init ru année n-1 + Irrigation auto"
    years=["2017"]
    lc="maize_irri"
    

# =============================================================================
#   Mean flux ETR -> Lam corrigées
# =============================================================================
    # All_lam=[]
    # for i in os.listdir(d["PC_disk"]+"/TRAITEMENT/DATA_VALIDATION/DATA_ETR_CESBIO/DATA_ETR_corr_maize_irri/"):
    #     if "ETR_maize_irri" in i and "semi" not in i:
    #         ETR=pd.read_csv(d["PC_disk"]+"/TRAITEMENT/DATA_VALIDATION/DATA_ETR_CESBIO/DATA_ETR_corr_maize_irri/"+i,decimal='.',sep=",")
    #         ETR["date"]=pd.to_datetime(ETR["date"],format="%Y-%m-%d")
    #         All_lam.append(ETR.LE_Bowen)
    # lam_et=pd.DataFrame(All_lam)
    # mean_lam=pd.DataFrame(lam_et.mean())
    # std_lam=pd.DataFrame(lam_et.std())
    # mean_lam["date"]=ETR.date
# =============================================================================
# Validation des Irr cumulées CACG
# =============================================================================
    Vol_tot=pd.DataFrame()
    Id=pd.DataFrame()
    Vol_tot_min=pd.DataFrame()
    Vol_tot_max=pd.DataFrame()
    Vol_tot_min2=pd.DataFrame()
    Vol_tot_max2=pd.DataFrame()
    Vol_tot_max3=pd.DataFrame()
    for y in years: 
        vali_PKGC=pd.read_csv(d["PC_disk"]+"TRAITEMENT/DATA_VALIDATION/DATA_VOL_IRRIGATION/VOL_TOT_PKGC_"+str(y)+".csv",encoding='latin-1',decimal='.',sep=',',na_values="nan")
        d["Output_model_PC_home_disk"]=d["PC_disk"]+"/TRAITEMENT/"+name_run
        Irri_mod=pd.read_csv(d["Output_model_PC_home_disk"]+"/LUT_"+str(y)+".csv",index_col=[0,1],skipinitialspace=True)
        gro=Irri_mod.groupby("ID")
        
        NDVI=pickle.load(open(d["Output_model_PC_home_disk"]+"/"+str(y)+"/Inputdata/maize_irri/NDVI"+str(y)+".df","rb"))
        NDVI=NDVI.loc[(NDVI.date >= str(y)+'-04-01')&(NDVI.date<=str(y)+"-09-30")]
        
# =============================================================================
#          Recuperation ETR flux +SWC
# =============================================================================
#  Lecture file param 

        # All_ETR=pd.DataFrame()
        # param=pd.read_csv(d["Output_model_PC_home_disk"]+"/"+str(y)+"/Output/maxZr/output_test_maize_irri_param.txt",header=None,skiprows=1,sep=";")
        # for r in os.listdir(d["Output_model_PC_home_disk"]+"/"+str(y)+"/Output/maxZr/"):
        #     if ".df" in r:
        #         print(r)
        #         num_run=r[23:-3]
        #         df=pickle.load(open(d["Output_model_PC_home_disk"]+"/"+str(y)+"/Output/maxZr/"+r,'rb'))
        #         df["param"]=np.repeat(param.loc[param[0]==int(num_run)][1].values,df.shape[0])
        #         All_ETR=All_ETR.append(df[["date","ET","id",'param']])
        # et=All_ETR.groupby("id")
        # # inser loop parcelle Id
    
        for p in vali_PKGC.ID:
            par1=gro.get_group(p)
            par1.reset_index(inplace=True)
            par1.num_run=pd.to_datetime(par1.num_run,format="%Y-%m-%d")
            mean_run=par1[["ID","4"]]
            maxIr=par1[["ID","14"]] # 8 
            max2Ir=par1[["ID","12"]]
            minIr=par1[["ID","0"]]
            min2Ir=par1[["ID","2"]]
            max3Ir=par1[["ID","10"]]
            mean_run.columns=["ID","maxZr_1000"]
            maxIr.columns=["ID","maxZr_1500"]
            minIr.columns=["ID","maxZr_800"]
            min2Ir.columns=["ID","maxZr_900"]
            max2Ir.columns=["ID","maxZr_1400"]
            max3Ir.columns=["ID","maxZr_1200"]
            # maxIr.replace(0.0,pd.NaT,inplace=True)
            # minIr.replace(0.0,pd.NaT,inplace=True)
            # print(mean_run.loc[mean_run['maxZr_1000']!=0.0])
            
            # Pour ETR
            # paret1=et.get_group(p)
            # paret1.reset_index(inplace=True)
            # paret1.date=pd.to_datetime(paret1.date,format="%Y-%m-%d")
            # paret1=paret1.loc[(paret1.date >= str(y)+"-04-01") &(paret1.date <= str(y)+"-09-30")]
            # validation
            # all_res=pd.merge(vali_PKGC,mean_run,on=["ID"]) # fusion des sim/obs
            # all_res_min=pd.merge(vali_PKGC,minIr,on=["ID"])
            # all_res_max=pd.merge(vali_PKGC,maxIr,on=["ID"])
            # all_res_min2=pd.merge(vali_PKGC,min2Ir,on=["ID"])
            # all_res_max2=pd.merge(vali_PKGC,max2Ir,on=["ID"])
            # all_res_max3=pd.merge(vali_PKGC,max3Ir,on=["ID"])
            # all_resu=all_res.replace(0.0,pd.NaT)
            # print("============")
            # print("parcelle :%s"%p)
            # print(all_res.sum())
            # print("============")
            #### plot
            plt.figure(figsize=(7,7))
            plt.title(p)
            # plt.plot(all_resu.date,all_resu.maxZr_1000,marker="x",linestyle="",label="Simulée")
            # # plt.plot(minIr.date,minIr.maxZr_800,marker="x",linestyle="",label="Simulée_min",alpha=0.5)
            # # plt.plot(maxIr.date,maxIr.maxZr_1200,marker="x",linestyle="",label="Simulée_max",alpha=0.5)
            # plt.plot(all_resu.date,all_resu.Quantite,marker="o",linestyle="",label="Observée")
            # plt.ylim(0,50)
            # plt.ylabel("Irrigation en mm")
            # plt.legend()
            # ax2=plt.twinx(ax=None)
            plt.plot(NDVI.loc[NDVI.id==p].date,NDVI.loc[NDVI.id==p].NDVI,color="darkgreen",linestyle="--")
            plt.ylabel("NDVI")
            plt.ylim(0,1)
            plt.savefig(d["PC_disk"]+"/TRAITEMENT/"+name_run_save_fig+"/plot_Irrigation_%s_%s.png"%(p,y))
            
            #  Plot ETR comparer avec flux moyenne 6 years LAM 
            # plt.figure(figsize=(7,7))
            # plt.plot(paret1.loc[paret1.param==1000.0]["date"],mean_lam[0].rolling(5).mean(),color='black',label="ETR lam moyenne 6 years")
            # plt.fill_between(paret1.loc[paret1.param==1000.0]["date"],mean_lam[0].rolling(5).mean()-std_lam[0].rolling(5).mean(),mean_lam[0].rolling(5).mean()+std_lam[0].rolling(5).mean(),facecolor="None",ec='black',linestyle="--",alpha=0.5)
            # plt.plot(paret1.loc[paret1.param==1000.0]["date"],paret1.loc[paret1.param==1000.0]["ET"].rolling(5).mean(),label='ETR parcelle')
            # plt.fill_between(paret1.loc[paret1.param==1000.0]["date"],paret1.loc[paret1.param==800.0]["ET"].rolling(5).mean(),paret1.loc[paret1.param==1200.0]["ET"].rolling(5).mean(),alpha=0.5)
            # plt.legend()
            # plt.title(p)
            # plt.savefig(d["PC_disk"]+"/TRAITEMENT/"+name_run_save_fig+"/plot_ETR_dynamiuqe_%s_%s.png"%(p,y))
            
            
            Vol_tot=Vol_tot.append(mean_run)
            Vol_tot_min=Vol_tot_min.append(minIr)
            Vol_tot_max=Vol_tot_max.append(maxIr)
            Vol_tot_min2=Vol_tot_min2.append(min2Ir)
            Vol_tot_max2=Vol_tot_max2.append(max2Ir)
            Vol_tot_max3=Vol_tot_max3.append(max3Ir)
            Id=Id.append(list(par1.ID.values))
        
# =============================================================================
# Volumes annuels
# =============================================================================
    Vol_tot["ID"]=Id
    Vol_tot["maxZr_800"]=Vol_tot_min["maxZr_800"]
    Vol_tot["maxZr_1500"]=Vol_tot_max["maxZr_1500"]
    Vol_tot["maxZr_900"]=Vol_tot_min2["maxZr_900"]
    Vol_tot["maxZr_1400"]=Vol_tot_max2["maxZr_1400"]
    Vol_tot["maxZr_1200"]=Vol_tot_max3["maxZr_1200"]
    tot_ID=Vol_tot.groupby("ID").sum()
    tot_IRR=pd.merge(tot_ID,vali_PKGC,on=["ID"])
    for t in ["maxZr_1000","maxZr_1500","maxZr_800","maxZr_900","maxZr_1400",'maxZr_1200']:
        # stat total
        slope, intercept, r_value, p_value, std_err = stats.linregress(tot_IRR.MMEAU.to_list(),tot_IRR[t].to_list())
        bias=1/tot_IRR.shape[0]*sum(np.mean(tot_IRR.MMEAU)-tot_IRR[t]) 
        # fitLine = predict(tot_ID[t])
        rms = mean_squared_error(tot_IRR.MMEAU,tot_IRR[t],squared=False)
        plt.figure(figsize=(7,7))
        plt.scatter(tot_IRR.MMEAU,tot_IRR[t],label=y)
        plt.xlim(-10,350)
        plt.ylim(-10,350)
        plt.legend()
        plt.xlabel("Volumes annuels observés en mm ")
        plt.ylabel("Volumes annuels modélisés en mm ")
        plt.plot([-10.0, 350], [-10.0,350], 'black', lw=1,linestyle='--')
        rectangle = plt.Rectangle((95, 300),72,42, ec='blue',fc='blue',alpha=0.1)
        plt.gca().add_patch(rectangle)
        plt.text(100,330,"RMSE = "+str(round(rms,2))) 
        plt.text(100,320,"R² = "+str(round(r_value,2)))
        plt.text(100,310,"Pente = "+str(round(slope,2)))
        plt.text(100,300,"Biais = "+str(round(bias,2)))
        for i in enumerate(tot_IRR.ID):
            label = int(i[1])
            plt.annotate(label, # this is the text
                 (tot_IRR["MMEAU"].iloc[i[0]],tot_IRR[t].iloc[i[0]]), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,5), # distance from text to points (x,y)
                 ha='center')
        plt.title(t)
        plt.savefig(d["PC_disk"]+"/TRAITEMENT/"+name_run_save_fig+"/plot_scatter_volumes_%s_Irrigation.png"%t)
    
    
