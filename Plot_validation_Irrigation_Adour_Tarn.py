#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 10:54:42 2021

@author: pageot

Validation Irrigation automatique parcelle de référence Adour Tarn
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
    name_run="RUNS_SAMIR/RUN_ADOUR_TARN/ADOUR_TARN_p055_Fcover_maxZr1000_irri_auto/"
    name_run_save_fig="RUNS_SAMIR/RUN_ADOUR_TARN/ADOUR_TARN_p055_Fcover_maxZr1000_irri_auto/"
    # d["PC_disk"]="/run/media/pageot/Transcend/Yann_THESE/BESOIN_EAU/BESOIN_EAU/"
    d["PC_home"]="/mnt/d/THESE_TMP/"
    d["PC_home_Wind"]="D:/THESE_TMP/"
    d["PC_disk"]="H:/Yann_THESE/BESOIN_EAU/BESOIN_EAU/"
    d["PC_labo"]="/datalocal/vboxshare/THESE/BESOIN_EAU/"
    years=["2017","2018"]
    lc="maize_irri"
    

# =============================================================================
# Validation des Irr cumulées Adour et Tarn
# =============================================================================
    

    for y in years: 
        data_soil=pd.read_csv(d["PC_disk"]+"/TRAITEMENT/SOIL/GSM/Extract_GSM_parcelle_Adour_Tarn_"+str(y)+"_UTS_maj.csv",index_col=[0],sep=';',encoding='latin-1',decimal=',')
        Vol_tot=pd.DataFrame()
        d["Output_model_PC_home_disk"]=d["PC_disk"]+"/TRAITEMENT/"+name_run
        df_date_aqui=pd.read_csv(d["PC_disk"]+"/TRAITEMENT/INPUT_DATA/NDVI_parcelle/Sentinel2_T30TYP_input_dates_"+str(y)+".txt",header=None)
        df_date_aqui[0]=pd.to_datetime(df_date_aqui[0],format='%Y%m%d')
        df_date_aqui.columns=["date"]
        vali_data=pd.read_csv(d["PC_disk"]+"TRAITEMENT/DATA_VALIDATION/DATA_VOL_IRRIGATION/Valid_Irrigation_"+str(y)+"_ADOUR_TARN.csv",encoding='latin-1',decimal=',',sep=';',na_values="nan")
        vali_data.Date=pd.to_datetime(vali_data.Date,format='%d/%m/%Y')
        vali_data["Dose"].astype(float)
        sum_irr_val=vali_data.groupby("id")["Dose"].sum()
        nb_irr=vali_data.groupby("id")["Dose"].count()
        BV_names=vali_data[["id","BV"]].drop_duplicates()
        if y =="2017":
            BV_names.drop(94,inplace=True)
        else:
            BV_names.drop(37,inplace=True)
        sum_irr_val_bv=pd.merge(sum_irr_val,BV_names,on=['id'])
        a=vali_data.groupby("id")
        
        Irri_mod=pickle.load(open(d["Output_model_PC_home_disk"]+"/"+str(y)+"/output_test_"+str(y)+".df","rb"))
        Irri_par=Irri_mod.groupby("id")["Ir_auto"].sum()
        
        NDVI=pickle.load(open(d["Output_model_PC_home_disk"]+"/"+str(y)+"/Inputdata/maize_irri/NDVI"+str(y)+".df","rb"))
        NDVI=NDVI.loc[(NDVI.date >= str(y)+'-04-01')&(NDVI.date<=str(y)+"-09-30")]
        Fcov=pickle.load(open(d["Output_model_PC_home_disk"]+"/"+str(y)+"/Inputdata/maize_irri/Fcover.df","rb"))
        Fcov=Fcov.loc[(Fcov.date >= str(y)+'-04-01')&(Fcov.date<=str(y)+"-09-30")]
        Prec=pickle.load(open(d["Output_model_PC_home_disk"]+"/"+str(y)+"/Inputdata/maize_irri/meteo.df","rb"))
        Prec=Prec.loc[(Prec.date >= str(y)+'-04-01')&(Prec.date<=str(y)+"-09-30")]
        
        if y =="2018":
            tot_2018=pd.merge(sum_irr_val_bv,Irri_par,on=["id"])
        else:
            tot_2017=pd.merge(sum_irr_val_bv,Irri_par,on=["id"])
# =============================================================================
# plot
# =============================================================================

    plt.figure(figsize=(7,7))
    for y in years :
        slope, intercept, r_value, p_value, std_err = stats.linregress(globals()["tot_"+y].Dose.to_list(), globals()["tot_"+y].Ir_auto.to_list())
        bias=1/len(globals()["tot_"+y].Dose.to_list())*sum(globals()["tot_"+y].Ir_auto-np.mean(globals()["tot_"+y].Dose.to_list())) 
        rms = np.sqrt(mean_squared_error(globals()["tot_"+y].Dose.to_list(), globals()["tot_"+y].Ir_auto))
        # plt.scatter(globals()["tot_"+y].Dose.to_list(),globals()["tot_"+y].Ir_auto,label=y,)
        sns.scatterplot(globals()["tot_"+y].Dose.to_list(),globals()["tot_"+y].Ir_auto,style=globals()["tot_"+y].BV,label=y,s=50)
        plt.xlim(-10,350)
        plt.ylim(-10,350)
        plt.legend()
        plt.xlabel("Quantité annuelles observées en mm ")
        plt.ylabel("Quantité annuelles modélisées en mm ")
        plt.plot([-10.0, 350], [-10.0,350], 'black', lw=1,linestyle='--')
        #  add error bar in scatter plot
        # plt.errorbar(globals()["tot_"+y].Vali,globals()["tot_"+y].conso,yerr=yerr,fmt='o',elinewidth=0.7,capsize =4)
        if "2017" in y :
            rectangle = plt.Rectangle((95, 245),70,45, ec='blue',fc='blue',alpha=0.1)
            plt.gca().add_patch(rectangle)
            plt.text(100,280,"RMSE = "+str(round(rms,2))) 
            plt.text(100,270,"R² = "+str(round(r_value,2)))
            plt.text(100,260,"Pente = "+str(round(slope,2)))
            plt.text(100,250,"Biais = "+str(round(bias,2)))
        else:
            rectangle = plt.Rectangle((225, 45),70,45, ec='orange',fc='orange',alpha=0.3)
            plt.gca().add_patch(rectangle)
            plt.text(230,80,"RMSE = "+str(round(rms,2))) 
            plt.text(230,70,"R² = "+str(round(r_value,2)))
            plt.text(230,60,"Pente = "+str(round(slope,2)))
            plt.text(230,50,"Biais = "+str(round(bias,2)))
        for i in enumerate(set(globals()["tot_"+y].id)):
            label = int(i[1])
            plt.annotate(label, # this is the text
                  (globals()["tot_"+y].Dose[i[0]],globals()["tot_"+y].Ir_auto[i[0]]), # this is the point to label
                  textcoords="offset points", # how to position the text
                  xytext=(0,5), # distance from text to points (x,y)
                  ha='center')
    plt.savefig(d["PC_disk"]+"/TRAITEMENT/"+name_run_save_fig+"/plot_scatter_volumes_Irrigation.png")
    
    # tot_2017_v2["TAWmax"]=tot_2017_v2.eval("(CC_mean-PF_mean)*param")
    # tot_2017_v2.param=tot_2017_v2.param/10
    # slope, intercept, r_value, p_value, std_err = stats.linregress(tot_2017_v2.RUM.to_list(),tot_2017_v2.TAWmax.to_list())
    # bias=1/tot_2017_v2["RUM"].shape[0]*sum(tot_2017_v2.TAWmax-np.mean(tot_2017_v2.RUM)) 
    # rms = np.sqrt(mean_squared_error(tot_2017_v2.RUM,tot_2017_v2.TAWmax))
    # slopeZr, intercept, r_valueZr, p_value, std_err = stats.linregress(tot_2017_v2.ProfRacPot.to_list(),tot_2017_v2.param.to_list())
    # biasZr=1/tot_2017_v2["ProfRacPot"].shape[0]*sum(tot_2017_v2.param-np.mean(tot_2017_v2.ProfRacPot)) 
    # rmsZr = np.sqrt(mean_squared_error(tot_2017_v2.ProfRacPot,tot_2017_v2.param))
    # plt.figure(figsize=(7,7))
    # a=plt.scatter(tot_2017_v2.RUM,tot_2017_v2.TAWmax,color='r',label='RUM')
    # plt.scatter(tot_2017_v2.ProfRacPot,tot_2017_v2.param,color='b',label="MaxZr")
    # plt.xlim(-10,300)
    # plt.ylim(-10,300)
    # plt.xlabel("RUM et maxZr observée en cm ")
    # plt.ylabel("RUM et maxZr modélisées en cm ")
    # rectangle = plt.Rectangle((190, 145),70,45, ec='r',fc='r',alpha=0.1)
    # plt.gca().add_patch(rectangle)
    # plt.text(200,180,"RMSE = "+str(round(rms,2))) 
    # plt.text(200,170,"R² = "+str(round(r_value,2)))
    # plt.text(200,160,"Pente = "+str(round(slope,2)))
    # plt.text(200,150,"Biais = "+str(round(bias,2)))
    # rectangle = plt.Rectangle((45, 195),70,45, ec='b',fc='b',alpha=0.1)
    # plt.gca().add_patch(rectangle)
    # plt.text(50,230,"RMSE = "+str(round(rmsZr,2))) 
    # plt.text(50,220,"R² = "+str(round(r_valueZr,2)))
    # plt.text(50,210,"Pente = "+str(round(slopeZr,2)))
    # plt.text(50,200,"Biais = "+str(round(biasZr,2)))
    # plt.plot([-10.0, 300], [-10.0,300], 'black', lw=1,linestyle='--')
    # plt.legend()
    # for i in enumerate(tot_2017_v2.ID):
    #     label = int(i[1])
    #     plt.annotate(label, # this is the text
    #           (tot_2017_v2["RUM"].iloc[i[0]],tot_2017_v2.TAWmax.iloc[i[0]]), # this is the point to label
    #           textcoords="offset points", # how to position the text
    #           xytext=(-6,2), # distance from text to points (x,y)
    #           ha='center')
    # for i in enumerate(tot_2017_v2.ID):
    #    label = int(i[1])
    #    plt.annotate(label, # this is the text
    #          (tot_2017_v2["ProfRacPot"].iloc[i[0]],tot_2017_v2.param.iloc[i[0]]), # this is the point to label
    #          textcoords="offset points", # how to position the text
    #          xytext=(-6,2), # distance from text to points (x,y)
    #          ha='center')
    # plt.savefig(d["PC_disk"]+"/TRAITEMENT/"+name_run_save_fig+"/plot_scatter_RUm_TAW.png")
