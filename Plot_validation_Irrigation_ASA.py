# -*- coding: utf-8 -*-
"""
Created on Tue May 18 10:05:24 2021

@author: yann 

Validation Irrigation automatique parcelle de référence ASA
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
    name_run="RUNS_SAMIR/RUN_ASA/ASA_init_ru_optim_Fcover_fewi_De_Kr_days10_p06_1000_irri_auto_soil/"
    name_run_save_fig="RUNS_SAMIR/RUN_ASA/ASA_init_ru_optim_Fcover_fewi_De_Kr_days10_p06_1000_irri_auto_soil/"
    # d["PC_disk"]="/run/media/pageot/Transcend/Yann_THESE/BESOIN_EAU/BESOIN_EAU/"
    d["PC_home"]="/mnt/d/THESE_TMP/"
    d["PC_home_Wind"]="D:/THESE_TMP/"
    d["PC_disk"]="H:/Yann_THESE/BESOIN_EAU/BESOIN_EAU/"

    d["PC_labo"]="/datalocal/vboxshare/THESE/BESOIN_EAU/"
    # label="Init ru année n-1 + Irrigation auto"
    years=["2017"]
    lc="maize_irri"


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
        vali_ASA=pd.read_csv(d["PC_disk"]+"TRAITEMENT/DATA_VALIDATION/DATA_VOL_IRRIGATION/ConsoASA_NESTE_"+str(y)+".csv",encoding='latin-1',decimal='.',sep=';',na_values="nan")
        d["Output_model_PC_home_disk"]=d["PC_disk"]+"/TRAITEMENT/"+name_run
        Irri_mod=pd.read_csv(d["Output_model_PC_home_disk"]+"/LUT_"+str(y)+".csv",index_col=[0,1],skipinitialspace=True)
        gro=Irri_mod.groupby("ID")
        
        NDVI=pickle.load(open(d["Output_model_PC_home_disk"]+"/"+str(y)+"/Inputdata/maize_irri/NDVI"+str(y)+".df","rb"))
        NDVI=NDVI.loc[(NDVI.date >= str(y)+'-04-01')&(NDVI.date<=str(y)+"-09-30")]
        Prec=pickle.load(open(d["Output_model_PC_home_disk"]+"/"+str(y)+"/Inputdata/maize_irri/meteo.df","rb"))
        Prec=Prec.loc[(Prec.date >= str(y)+'-04-01')&(Prec.date<=str(y)+"-09-30")]
        df_date_aqui=pd.read_csv(d["PC_disk"]+"/TRAITEMENT/INPUT_DATA/NDVI_parcelle/Sentinel2_T30TYP_input_dates_"+str(y)+".txt",header=None)
        df_date_aqui[0]=pd.to_datetime(df_date_aqui[0],format='%Y%m%d')
        df_date_aqui.columns=["date"]
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
        
        for p in list(set(Irri_mod.ID))[2:]:
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
            df_aqui=pd.merge(df_date_aqui,NDVI.loc[NDVI.id==p],on="date")
            
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
    nb_irr=Vol_tot[Vol_tot!=0.0].groupby("ID").count()
    tot_ID=Vol_tot.groupby("ID").sum()
    tot_IRR=pd.merge(tot_ID,vali_ASA,on=["ID"])
    tot_IRR.dropna(inplace=True)
    asagroup=tot_IRR.groupby("nomasa")
    # loop 
    for t in ["maxZr_1000","maxZr_1500","maxZr_800","maxZr_900","maxZr_1400",'maxZr_1200']:
        conso=[]
        asa_n=[]
        data_v=[]
        for asa in list(set(tot_IRR.nomasa)):
            tet=asagroup.get_group(asa)
            tet["conso"]=tet.eval("%s*area*10"%t)
            # tet["am3"]=tet.area*10000
            # tet["conso"]=tet.eval("maxZr_1000*am3")
            conso_asa=sum(tet["conso"])
            conso.append(conso_asa)
            asa_n.append(asa)
            data_v.append(tet.data_Conso.iloc[0])
            # stat total
        slope, intercept, r_value, p_value, std_err = stats.linregress(data_v,conso)
        bias=1/len(data_v)*sum(conso-np.mean(data_v)) 
        # fitLine = predict(tot_ID[t])
        rms = mean_squared_error(data_v,conso,squared= False)
        plt.figure(figsize=(7,7))
        plt.scatter(data_v,conso)
        plt.xlim(-1000,1000000)
        plt.ylim(-1000,1000000)
        plt.xlabel("Volumes annuels observés en m3 ")
        plt.ylabel("Volumes annuels modélisés en m3 ")
        plt.plot([-1000, 1000000], [-1000,1000000], 'black', lw=1,linestyle='--')
        # rectangle = plt.Rectangle((95, 300),72,42, ec='blue',fc='blue',alpha=0.1)
        # plt.gca().add_patch(rectangle)
        # plt.text(100,330,"RMSE = "+str(round(rms,2))) 
        # plt.text(100,320,"R² = "+str(round(r_value,2)))
        # plt.text(100,310,"Pente = "+str(round(slope,2)))
        # plt.text(100,300,"Biais = "+str(round(bias,2)))
        for i in enumerate(asa_n):
            label = i[1]
            plt.annotate(label, # this is the text
                 (data_v[i[0]],conso[i[0]]), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,5), # distance from text to points (x,y)
                 ha='center')
        plt.title(t)
        plt.savefig(d["PC_disk"]+"/TRAITEMENT/"+name_run_save_fig+"/plot_scatter_volumes_%s_%s_Irrigation.png"%(t,y))
        
    # plt.figure(figsize=(10,10))
    # for p in list(set(Prec.id)):#list(set(Prec.id))
    #     plt.plot(Prec.loc[Prec.id==p].date,Prec.loc[Prec.id==p].Prec.cumsum(),label='pluie')
    #     # plt.plot(Prec.loc[Prec.id==p].date,Prec.loc[Prec.id==p].ET0.cumsum(),label="Et0")
    #     plt.ylabel("Cumul de précipitation en mm")
    #     plt.text(Prec.loc[Prec.id==p].iloc[-1].date,Prec.loc[Prec.id==p].Prec.cumsum().iloc[-1],s=p)
    #     # plt.text(Prec.loc[Prec.id==p].iloc[-1].date,Prec.loc[Prec.id==p].ET0.cumsum().iloc[-1],s=p)
    # # plt.legend()
    #     # plt.ylim(0,1)
    # plt.savefig(d["PC_disk"]+"/TRAITEMENT/"+name_run_save_fig+"/plot_PREC_cumul_%s_Irrigation.png"%t)
        
    tot_IRR.to_csv(d["PC_disk"]+"/TRAITEMENT/"+name_run_save_fig+"/tab_final_quantite_Irr.csv")
    nb_irr.to_csv(d["PC_disk"]+"/TRAITEMENT/"+name_run_save_fig+"/tab_final_nb_Irr.csv")
    # plt.savefig(d["PC_disk"]+"/TRAITEMENT/"+name_run_save_fig+"/plot_prec_cumul_parcelle_probleme.png")
