# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 13:32:39 2020

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
import seaborn as sns
import pickle
from SAMIR_optimi import RMSE
import geopandas as geo
import shapely.geometry as geom
import descartes
from datetime import datetime, date, time, timezone
from scipy import *
from scipy import stats
from pylab import *
from sklearn.linear_model import LinearRegression



def predict(x):
  return slope * x + intercept

def select_color_date(x):
    couleurs=[]
    for i in range(len(x)):
        if x.iloc[i].date.strftime('%m-%d')<= "04-01" : 
            couleurs.append("r")
        else : 
            couleurs.append("b")
    return couleurs

if __name__ == "__main__":
    
    d={}
    # all_min=[]
    # d["Output_PC_home"]="G:/Yann_THESE/BESOIN_EAU/RESULT/Optimisation/"
    # name_run="RUNS_optim_LUT_LAM_ETR/Zrmax_opti/Run_opti_param_ss_spin"
    # df=pd.read_csv("D:/THESE_TMP/RUNS_SAMIR/"+name_run+"/param_RMSE.csv")
    # Years=df.groupby("years")
    # plt.figure(figsize=(7,7))
    # for y in ["2006","2008","2010","2012","2014","2015"]:
    #     data=Years.get_group(int(y))
    #     data.sort_values("Param1",ascending=True,inplace=True)
    #     minval=data.loc[data["RMSE"].idxmin()]
    #     all_min.append(minval)
    #     plt.plot(data.Param1,data.RMSE,label=str(y))
    #     plt.plot(minval.Param1,minval.RMSE,marker="*",color="Black")
    #     plt.legend()
    #     plt.xlabel("Zrmax")
    #     plt.title("Sans_spin_up_sans_Fcover")
    #     plt.ylabel("RMSE ETRobs/ETRmod")    
    # plt.savefig( d["Output_PC_home"]+"Sans_Spin/plot_RMSE_Zrmax_"+name_run[-10:]+".png")
    # minall=pd.DataFrame(all_min)
    # minall["Param1"].mean()
    # minall.to_csv( d["Output_PC_home"]+"Sans_Spin/min_value_Zrmax"+name_run[-10:]+".csv")



# =============================================================================
# Test scatter Fcover
# =============================================================================
    All_Fcover=pd.DataFrame()
    All_ETR=pd.DataFrame()
    d["Output_PC_home"]="D:/THESE_TMP/RESULT/Optimisation/"
    # d['Output_model_PC_home']='D:/THESE_TMP/TRAITEMENT/RUNS_SAMIR/Compar_Fcover/'
    d["Output_model_PC_labo"]="/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/RUNS_SAMIR/Compar_Fcover"
    d['Output_model_PC_home']='D:/THESE_TMP/TRAITEMENT/RUNS_SAMIR/RUN_MULTI_SITE_ICOS/RUN_OPTIMISATION_ICOS/'
    for y in ["2006","2008","2010","2012","2014","2019"]:
        # Fcover_sat=pickle.load(open( d['Output_model_PC_home']+"/Avec_Fcover_sat/"+str(y)+"/output_T1.df",'rb'))
        # Fcover_mod=pickle.load(open( d['Output_model_PC_home']+"/Sans_Fcover_sat/"+str(y)+"/output_T1.df",'rb'))
        Fcover_mod=pickle.load(open( d['Output_model_PC_home']+"/OPTI_ICOS_MULTI_SITE_pluvio_Station_REW_Init1/"+str(y)+"/Output/output.df",'rb'))
        Fcover_sat=pickle.load(open( d['Output_model_PC_home']+"/OPTI_ICOS_MULTI_SITE_pluvio_Station_REW_Init1_Fcover/"+str(y)+"/Output/output.df",'rb'))
        Fcover_sat=Fcover_sat[["FCov","date",'ET','LC']]
        Fcover_mod=Fcover_mod[["FCov",'date','ET','LC']]
        if y =="2019":
            a=Fcover_sat.groupby("LC")
            Fcover_sat=a.get_group("maize_irri")
            b=Fcover_mod.groupby("LC")
            Fcover_mod=b.get_group("maize_irri")
        #  Plot dyna Fcover
        plt.figure(figsize=(7,7))
        plt.plot(Fcover_sat.date,Fcover_sat.FCov,label='Sat')
        plt.plot(Fcover_mod.date,Fcover_mod.FCov,label='Relat')
        plt.legend()
        plt.ylabel('Fcover')
        plt.title ("Fcover de l'année %s"%y)
        plt.xticks(rotation=45)
        plt.savefig(d["Output_PC_home"]+"/Fcover_comparaison_sat_relation_dynamique_"+y+".png")
       #  Plot dyna ETR
        plt.figure(figsize=(7,7))
        plt.plot(Fcover_sat.date,Fcover_sat.ET,label='Sat')
        plt.plot(Fcover_mod.date,Fcover_mod.ET,label='Relat',alpha=1)
        plt.legend()
        plt.ylabel('ETR')
        plt.title ("ETR de l'année %s"%y)
        plt.xticks(rotation=45)
        plt.savefig(d["Output_PC_home"]+"/ETR_comparaison_sat_relation_dynamique_"+y+".png")
        ETR_FC_sat=Fcover_sat.loc[(Fcover_sat.date >= str(y)+"-05-02") &(Fcover_sat.date <= str(y)+"-10-31")]
        ETR_FC_mod=Fcover_mod.loc[(Fcover_mod.date >= str(y)+"-05-01") & (Fcover_mod.date <= str(y)+"-10-31")]
        dfETR=pd.merge(ETR_FC_sat[["ET","date"]],ETR_FC_mod[["ET","date"]],on=["date"])
        dfETR.columns=["ETR_FCov_sat","date",'ETR_FCov_mod']
        dfETR.dropna(inplace=True)
        
        Fcover_sat=Fcover_sat.loc[(Fcover_sat.date >= str(y)+"-03-02") &(Fcover_sat.date <= str(y)+"-08-30")]
        Fcover_mod=Fcover_mod.loc[(Fcover_mod.date >= str(y)+"-03-01") & (Fcover_mod.date <= str(y)+"-08-30")] # problème date -1
        dfFcover=pd.merge(Fcover_sat[["FCov","date"]],Fcover_mod[["FCov","date"]],on=["date"])
        dfFcover.columns=["FCov_sat","date",'FCov_mod']
        dfFcover.FCov_mod.loc[dfFcover.FCov_mod == 0.0000]=pd.NaT
        dfFcover.dropna(inplace=True)
        All_Fcover=All_Fcover.append(dfFcover)
        All_ETR=All_ETR.append(dfETR)
    
    # plt.figure(figsize=(7,7))
    # slope, intercept, r_value, p_value, std_err = stats.linregress(All_Fcover.FCov_sat.to_list(),All_Fcover.FCov_mod.to_list())
    # bias=1/All_Fcover.shape[0]*sum(np.mean(All_Fcover.FCov_sat)-All_Fcover.FCov_mod) 
    # fitLine = predict(dfFcover.FCov_sat)
    # rms = mean_squared_error(All_Fcover.FCov_sat,All_Fcover.FCov_mod,squared=False)
    # plt.scatter(All_Fcover.FCov_sat.to_list(),All_Fcover.FCov_mod.to_list())
    # plt.plot([0.0, 1], [0.0,1], 'black', lw=1,linestyle='--')
    # plt.xlabel('FCov Sat')
    # plt.ylabel("Fcov_mod")
    # plt.title("Fcover Sat/ Fcover Relation NDVI sur la période 03-02 à 08-30")
    # plt.text(0.8,min(All_Fcover.FCov_sat)+0.05,"RMSE = "+str(round(rms,2)))
    # plt.text(0.8,min(All_Fcover.FCov_sat)+0.1,"R² = "+str(round(r_value,2)))
    # plt.text(0.8,min(All_Fcover.FCov_sat)+0.15,"Pente = "+str(round(slope,2)))
    # plt.text(0.8,min(All_Fcover.FCov_sat)+0.20,"Biais = "+str(round(bias,2)))
    # plt.savefig( d["Output_PC_home"]+"/Fcover_comparaison_sat_relation.png")
    
    plt.figure(figsize=(7,7))
    slope, intercept, r_value, p_value, std_err = stats.linregress(All_ETR.ETR_FCov_sat.to_list(),All_ETR.ETR_FCov_mod.to_list())
    bias=1/All_ETR.shape[0]*sum(np.mean(All_ETR.ETR_FCov_sat)-All_ETR.ETR_FCov_mod) 
    fitLine = predict(dfFcover.FCov_sat)
    rms = mean_squared_error(All_ETR.ETR_FCov_sat,All_ETR.ETR_FCov_mod,squared=False)
    plt.scatter(All_ETR.ETR_FCov_sat,All_ETR.ETR_FCov_mod)
    plt.plot([0.0, 10], [0.0,10], 'black', lw=1,linestyle='--')
    plt.xlabel('ETR FCov Sat')
    plt.ylabel("ETR Fcov_mod")
    plt.title("ETR avec Fcover Sat/ Fcover Relation NDVI sur la période 05-02 à 10-30")
    plt.text(8,min(All_ETR.ETR_FCov_sat)+1,"RMSE = "+str(round(rms,2)))
    plt.text(8,min(All_ETR.ETR_FCov_sat)+1.5,"R² = "+str(round(r_value,2)))
    plt.text(8,min(All_ETR.ETR_FCov_sat)+2,"Pente = "+str(round(slope,2)))
    plt.text(8,min(All_ETR.ETR_FCov_sat)+2.5,"Biais = "+str(round(bias,2)))
    plt.savefig( d["Output_PC_home"]+"/ETR_comparaison_sat_relation.png")
    
    
# =============================================================================
#   Relation ETR mod et ETR obs selon Fcover sat et Fcover mod pour les RMSE optimaux 
# =============================================================================
    # for Fcover in ["Sans","Avec"]:
    #     for y in ["2006","2008","2010","2012","2014","2015"]:
    #         ETR=pd.read_csv("G:/Yann_THESE/BESOIN_EAU/DATA_ETR_CESBIO/DATA_ETR_LAM/ETR_LAM"+str(y)+".csv",decimal='.')
    #         ETR["date"]=pd.to_datetime(ETR["date"],format="%Y-%m-%d")
    #         ETR_obs=ETR.loc[(ETR.date >= str(y)+"-03-02") &(ETR.date <= str(y)+"-10-31")]
    #         Fcover_sat=pickle.load(open( d['Output_model_PC_home']+"/"+Fcover+"_Fcover_sat/"+str(y)+"/output_T1.df",'rb'))
    #         Fcover_sat=Fcover_sat[["ET",'date']]
    #         Fcover_sat=Fcover_sat.loc[(Fcover_sat.date >= str(y)+"-03-02") &(Fcover_sat.date <= str(y)+"-10-31")]
    #         dfETR_obs=pd.merge(ETR_obs,Fcover_sat[["date",'ET']],on=['date'])
    #         dfETR_obs.dropna(inplace=True)
    #         slope, intercept, r_value, p_value, std_err = stats.linregress(dfETR_obs.LE.to_list(),dfETR_obs.ET.to_list())
    #         bias=1/dfETR_obs.shape[0]*sum(np.mean(dfETR_obs.ET)-dfETR_obs.LE) 
    #         fitLine = predict(dfETR_obs.LE)
    #         plt.figure(figsize=(7,7))
    #         plt.plot([0.0, 10], [0.0,10], 'black', lw=1,linestyle='--')
    #         plt.plot(dfETR_obs.LE,fitLine,linestyle="--")
    #         plt.scatter(dfETR_obs.LE,dfETR_obs.ET,s=9)
    #         plt.xlabel("ETR OBS")
    #         plt.ylabel("ETR model")
    #         plt.xlim(0,10)
    #         plt.ylim(0,10)
    #         plt.title("Scatter ETR obs et ETR mod %s Fcover en %s"%(Fcover,y))
    #         rms = mean_squared_error(dfETR_obs.LE,dfETR_obs.ET,squared=False)
    #         plt.text(8,min(dfETR_obs.ET)+0.1,"RMSE = "+str(round(rms,2))) 
    #         plt.text(8,min(dfETR_obs.ET)+0.4,"R² = "+str(round(r_value,2)))
    #         plt.text(8,min(dfETR_obs.ET)+0.6,"Pente = "+str(round(slope,2)))
    #         plt.text(8,min(dfETR_obs.ET)+0.8,"Biais = "+str(round(bias,2)))
    #         plt.savefig( d["Output_PC_home"]+"/plt_scatter_ETR_%s_Fcover_%s.png"%(Fcover,y))
    #         plt.figure(figsize=(7,7))
    #         plt.plot(dfETR_obs.date,dfETR_obs.LE,label='ETR_obs',color="black")
    #         plt.plot(dfETR_obs.date,dfETR_obs.ET,label='ETR_mod',color='red')
    #         plt.ylabel("ETR")
    #         plt.ylim(0,10)
    #         plt.title("Dynamique ETR obs et ETR mod %s Fcover en %s"%(Fcover,y))
    #         plt.legend()
    #         plt.savefig(d["Output_PC_home"]+"/plt_Dynamique_ETR_obs_ETR_mod_%s_Fcover_%s.png"%(Fcover,y))