#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 09:29:18 2020

@author: pageot
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
    d["path_labo"]="/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/"
#    d["path_PC"]="D:/THESE_TMP/RUNS_SAMIR/R12/Inputdata/"
    d["PC_disk"]="G:/Yann_THESE/BESOIN_EAU/"
#    runs=["R12"]
    all_quantity=[]
    all_number=[]
    all_id=[]
    all_runs=[]
    all_date=[]
    all_date_jj=[]
    for r in os.listdir('D:/THESE_TMP/RUNS_SAMIR/RUN_RELATION/TARN_CACG/'):
        d["path_PC"]='D:/THESE_TMP/RUNS_SAMIR/RUN_RELATION/TARN_CACG/'+str(r)+'/Inputdata/'
        for s in os.listdir(d["path_PC"][:-10]):
            if "output" in s:
                print (s)
                df=pickle.load(open(d["path_PC"][:-10]+str(s),'rb'))
                for id in list(set(df.id)):
                    lam=df.loc[df.id==id]
                    date_irr=lam.loc[lam.Ir_auto!=0.0]["date"]
                    size_R=len(date_irr) # vecteur de répétiton de la variable RUNS
                    print(r'n° du runs : %s '% r)
                    print(r' n° parcelle : %s' %id)
                    print(r'sum irrigation in mm : %s'%lam.groupby(["LC","id"])["Ir_auto"].sum()[0])
                    print(r' nb irrigation : %s' %lam.Ir_auto.where(df["Ir_auto"] != 0.0).dropna().count())
#                    plt.figure(figsize=(5,5))
#                    plt.title(r'run :%s : n° parcelle :%s '%(r,id))
#                    plt.plot(lam.date,lam.Ir_auto)
#                    plt.ylabel("quantité d'eau")
                    all_runs.append(r)
                    all_id.append(id)
                    all_quantity.append(lam.groupby(["LC","id"])["Ir_auto"].sum()[0])
                    all_number.append(lam.Ir_auto.where(df["Ir_auto"] != 0.0).dropna().count())
                    all_date.append(date_irr.values)
                    for i in date_irr:
                        a=i.strftime("%j") # date en jj sur l'année
                        all_date_jj.append(a)
    all_resu=pd.DataFrame([all_runs,all_id,all_quantity,all_number,all_date]).T
    all_resu.columns=["runs",'id','cumul_irr',"nb_irr","date_irr"]


#                    plt.savefig(d["path_PC"][:-10]+"plt_quantity_irri_%s.png"%id)
                
# =============================================================================
# Validation modelisation ETR     
# =============================================================================
#    Prépartion des datas Eddy-co au format journalière
    #  Pour 2017 et autre année
#     for y in os.listdir(d["PC_disk"]+"/DATA_ETR_CESBIO/DATA_LAM_lec_python/"):
#         years=y[5:9]
#         LE_lam=pd.read_csv(d["PC_disk"]+"/DATA_ETR_CESBIO/DATA_LAM_lec_python/"+str(y),encoding = 'utf-8',delimiter=",")
#         LE_lam["TIMESTAMP"]=LE_lam["TIMESTAMP"].apply(lambda x:x[0:10])
#         LE_lam["TIMESTAMP"]=pd.to_datetime(LE_lam["TIMESTAMP"],format="%d/%m/%Y")
#         LE_lam_day=LE_lam.groupby("TIMESTAMP")["LE"].mean()
#         ETR_lam_day=LE_lam_day*0.0352
#         ETR_lam_day[ETR_lam_day < -1]=pd.NaT
#         ETR_lam_day.plot()
#         ETR_lam_day.to_csv(d["PC_disk"]+"/DATA_ETR_CESBIO/DATA_ETR_LAM/ETR_LAM"+str(years)+".csv")
    
# #    Pour 2019
#     ETR_lam=pd.read_csv(d["PC_disk"]+"/DATA_ETR_CESBIO/DATA_LAM_lec_python/eddypro_FR-Lam_full_output_2020-01-28T012345_adv.csv")
#     ETR_lam["date"]=pd.to_datetime(ETR_lam["date"],format='%Y-%m-%d')
#     ETR_lam_day=ETR_lam.groupby("date")["LE"].mean()
#     ETR_lam_day=ETR_lam_day*0.0352
#     ETR_lam_day[ETR_lam_day < -1]=pd.NaT
#     ETR_lam_day.plot()
#     ETR_lam_day.to_csv(d["PC_disk"]+"/DATA_ETR_CESBIO/DATA_ETR_LAM/ETR_LAM2019.csv")
#    ## Récuparation date végétation sur ETR 


# =============================================================================
# Validation des Irr cumulées CACG
# =============================================================================
    vali_cacg=pd.read_csv(d["PC_disk"]+"TRAITEMENT/DONNEES_VALIDATION_SAMIR/merge_parcelle_2017.csv")
#    vali_cacg.dropna(subset=['id'],inplace=True)
    vali_cacg.Date_irrigation=pd.to_datetime(vali_cacg.Date_irrigation,format='%d/%m/%y')
    vali_cacg["Quantity(mm)"].astype(float)
    sum_irr_cacg_val=vali_cacg.groupby("id")["Quantity(mm)"].sum()
    nb_irr=vali_cacg.groupby("id")["Date_irrigation"].count()
    
    for r in os.listdir('D:/THESE_TMP/RUNS_SAMIR/RUN_RELATION/TARN_CACG/'):
        d["path_PC"]='D:/THESE_TMP/RUNS_SAMIR/RUN_RELATION/TARN_CACG/'+str(r)+'/Inputdata/'
        slope, intercept, r_value, p_value, std_err = stats.linregress(sum_irr_cacg_val.to_list(),all_resu.loc[all_resu.runs==r]["cumul_irr"].to_list())
        bias=1/sum_irr_cacg_val.shape[0]*sum(np.mean(all_resu.loc[all_resu.runs==r]["cumul_irr"])-sum_irr_cacg_val) 
        fitLine = predict(sum_irr_cacg_val)
        plt.figure(figsize=(7,7))
        plt.title(r)
        plt.plot([0.0, 300], [0.0, 300], 'r-', lw=2)
        plt.plot(sum_irr_cacg_val,fitLine,linestyle="-")
        plt.scatter(sum_irr_cacg_val,all_resu.loc[all_resu.runs==r]["cumul_irr"])
        plt.plot()
        plt.xlabel("cumul_irr OBS")
        plt.ylabel("cumul irr model")
        plt.xlim(0,300)
        plt.ylim(0,300)
        rms = mean_squared_error(sum_irr_cacg_val,all_resu.loc[all_resu.runs==r]["cumul_irr"],squared=False)
        plt.text(10,min(all_resu.loc[all_resu.runs==r]["cumul_irr"])+40,"RMSE = "+str(round(rms,2)))
        plt.text(10,min(all_resu.loc[all_resu.runs==r]["cumul_irr"])+30,"R² = "+str(round(r_value,2)))
        plt.text(10,min(all_resu.loc[all_resu.runs==r]["cumul_irr"])+20,"Pente = "+str(round(slope,2)))
        plt.text(10,min(all_resu.loc[all_resu.runs==r]["cumul_irr"])+10,"Biais = "+str(round(bias,2)))
        for j in np.arange(len(sum_irr_cacg_val.index)):
            plt.text(x = sum_irr_cacg_val.to_list()[j] + 2, y=all_resu.loc[all_resu.runs==r]["cumul_irr"].iloc[j]+ 1,s = list(all_resu.loc[all_resu.runs==r]["id"])[j],size=9)
        plt.savefig(d["path_PC"][:-10]+"plt_scatter_quantity_irri_%s.png"%r)
#         plot nb_irr
        
        slope, intercept, r_value, p_value, std_err = stats.linregress(nb_irr.to_list(),all_resu.loc[all_resu.runs==r]["nb_irr"].to_list())
        bias=1/nb_irr.shape[0]*sum(mean(all_resu.loc[all_resu.runs==r]["nb_irr"])-nb_irr) 
        fitLine = predict(nb_irr)
        plt.figure(figsize=(7,7))
        plt.title(r)
        plt.plot([0.0,20], [0.0, 20], 'r-', lw=2)
        plt.plot(nb_irr,fitLine,linestyle="--")
        plt.scatter(nb_irr,all_resu.loc[all_resu.runs==r]["nb_irr"])
        plt.plot()
        plt.xlabel("nb_irr OBS")
        plt.ylabel("nb irr model")
        plt.xlim(0,20)
        plt.ylim(0,20)
        rms = mean_squared_error(nb_irr,all_resu.loc[all_resu.runs==r]["nb_irr"],squared=False) # if False == RMSE or True == MSE
        plt.text(10,min(all_resu.loc[all_resu.runs==r]["nb_irr"])+2,"RMSE = "+str(round(rms,2)))
        plt.text(10,min(all_resu.loc[all_resu.runs==r]["nb_irr"])+1.5,"R² = "+str(round(r_value,2)))
        plt.text(10,min(all_resu.loc[all_resu.runs==r]["nb_irr"])+1,"Pente = "+str(round(slope,2)))
        plt.text(10,min(all_resu.loc[all_resu.runs==r]["nb_irr"])+0.5,"Biais = "+str(round(bias,2)))
        for j in np.arange(len(nb_irr.index)):
            plt.text(x = nb_irr.to_list()[j]+0.1 , y=all_resu.loc[all_resu.runs==r]["nb_irr"].iloc[j]+0.1,s = list(all_resu.loc[all_resu.runs==r]["id"])[j],size=9)
        plt.savefig(d["path_PC"][:-10]+"plt_scatter_nb_irri_%s.png"%r)

#    ETsum = (df.groupby(['LC', 'id'])['ET'].sum()).reset_index()
#    LCclasses = df.LC.cat.categories
#    
#    ETmin = ETsum.ET.min()
#    ETmax = ETsum.ET.max()
#    ET= {}
#    for lc in LCclasses:
#        ET[lc] = ETsum.loc[ETsum.LC == lc].reset_index()
#        ET[lc].drop(columns = ['LC', 'index'], inplace = True)
#    ETmin, ETmax
#    LCclasses = df.LC.cat.categories
#
#    gdf = {}
#    for lc in LCclasses:
#
#
#        gdf[lc] = geo.read_file(d["path_PC"]+"shapefiles/PARCELLE_LABO_ref.shp")
#        gdf[lc] = gdf[lc].merge(ET[lc], on='id') # attention ID en minicuel dans le shape
#        
#        gdf[lc].plot(column='ET',figsize=(10,10), vmin=ETmin, vmax=ETmax, cmap='RdYlGn', legend=True) # Création map evapotransipartion 
#        plt.title(lc + '   : Evapotranspiration')
#        
#    lam=df.loc[df.id==0] # 0 lam 1 aur
#    variables=['FCov', 'fewi', 'fewp', 'Zr', 'Zd', 'TEW', 'TAW', 'TDW', 'RAW', 'RUE',
#       'Dei', 'Dep', 'Dr', 'Dd', 'Ir_auto', 'ET', 'SWC1', 'SWC2', 'SWC3',
#       'SWCvol1', 'SWCvol2', 'SWCvol3']
#    for v in variables:
#        print(v)
#        plt.figure(figsize=(5,5))
#        plt.title(v)
#        plt.plot(lam.date,lam[v])
    
    
# =============================================================================
#   Calcul le cumul d'irrigation 
# =============================================================================
    #  Irrigation total par OS simuler et par parcelle
#    lam.groupby(["LC","id"])["Ir_auto"].sum()
#    # or
#    lam.Ir_auto.where(lam["Ir_auto"] != 0.0).dropna().count() # resultat  980.0 et ref = 944 soit 44 mm surplus
#
## =============================================================================
##   Vérification des Flux ETR
## =============================================================================
#    lam[["ET","date"]]
    