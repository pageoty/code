#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 14:00:05 2021

@author: pageot

Ce script permet de gérer les cartes de besoins en eau et des surfaces irriguées à partir des sorties du modèle SAMIR. 
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
# import seaborn as sns 
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
    d["PC_disk"]="/run/media/pageot/Transcend/Yann_THESE/"
    d["PC_home"]="/mnt/d/THESE_TMP/"
    d["PC_home_Wind"]="D:/THESE_TMP/"
    # d["PC_disk"]="H:/Yann_THESE/BESOIN_EAU/BESOIN_EAU/"
    d["PC_disk_water"]="/run/media/pageot/Transcend/Yann_THESE/BESOIN_EAU/BESOIN_EAU/"
    d["PC_labo"]="/datalocal/vboxshare/THESE/BESOIN_EAU/"
    years=["2017"]
    lc="maize_irri"

# Attention NESTE présente un duplication des dates
# =============================================================================
#     Cas du bassin versant de la Nesye
# =============================================================================
#  Input 
    Parcellaire= geo.read_file(d["PC_disk"]+"/CLASSIFICATION/DATA_CLASSIFICATION/RPG/RPG_BV/RPG_SUMMER_2017_NESTE_MAIZE_ONLY.shp")
    df_mod=pickle.load(open(d["PC_disk_water"]+"/TRAITEMENT/RUNS_SAMIR/RUN_NESTE_RPG/NESTE_init_ru_P055_Fcover_fewi_De_Kr_days10_dose30_1200_irri_auto_soil/2017/output_test_2017.df","rb"))
    ET= {}
    gdf = {}
    #  Gestion des Inputs
    Parcellaire["id"]=Parcellaire.ID
    df_mod_drop=df_mod.drop_duplicates(subset=["date","id"])
    
    ETsum = (df_mod_drop.groupby(['id'])['Ir_auto'].sum()).reset_index()
    ETmin = ETsum.Ir_auto.min()
    ETmax = ETsum.Ir_auto.max()
    ET = ETsum
    ET["IRR"]=1
    ET.loc[(ET.Ir_auto==0.0),'IRR']=0
    ET.loc[(ET.Ir_auto>0) & (ET.Ir_auto<=60),'IRR']=0.5

    gdf = Parcellaire
    gdf = gdf.merge(ET, on='id')
    gdf.to_file(d["PC_disk_water"]+"/TRAITEMENT/RUNS_SAMIR/RUN_NESTE_RPG/NESTE_init_ru_P055_Fcover_fewi_De_Kr_days10_dose30_1200_irri_auto_soil/2017/carte_surface_irriguée_2017_SAMIR_3classe_seuil_60mm.shp")
   
    # Création de la carte besoin en eau 
    gdf.plot(column='Ir_auto',figsize=(10,10), vmin=ETmin, vmax=ETmax, cmap='RdYlGn', legend=True)
    plt.title('Irrigation saisonnière en mm')
    plt.savefig(d["PC_disk_water"]+"/TRAITEMENT/RUNS_SAMIR/RUN_NESTE_RPG/NESTE_init_ru_P055_Fcover_fewi_De_Kr_days10_dose30_1200_irri_auto_soil/2017/carte_surface_irriguée_2017_SAMIR.png")
     
    #  Création carte surfaces irriguées
    gdf.plot(column='IRR',figsize=(10,10), cmap='RdYlGn', legend=True)
    plt.legend()
    plt.title('Irrigué / pluviale')
    plt.savefig(d["PC_disk_water"]+"/TRAITEMENT/RUNS_SAMIR/RUN_NESTE_RPG/NESTE_init_ru_P055_Fcover_fewi_De_Kr_days10_dose30_1200_irri_auto_soil/2017/carte_irrigue_pluvaile_2017_SAMIR.png")
#  Volumes totaux sur le BV = sum(gdf.Ir_auto*10*gdf.area/10000)/1000000 pour obtenir des millions de m3 résulta 13 millions contre 49 autorisées

#  surfaces totals sur le BV avec autorisation d'irriguées = sum(gdf[gdf.Ir_auto!=0].area/10000) résulats 22.3 milliers ha contre 25.3 observées
# =============================================================================
#  cas du BV Adour Amont
# =============================================================================
    y="2017"
    Parcellaire= geo.read_file(d["PC_disk"]+"/CLASSIFICATION/DATA_CLASSIFICATION/RPG/RPG_BV/RPG_SUMMER_"+y+"_ADOUR_AMONT.shp")
    df_mod=pickle.load(open(d["PC_disk_water"]+"/TRAITEMENT/RUNS_SAMIR/RUN_CLASSIF_ALL_MAIS/Classif_init_ru_P055_Fcover_fewi_De_Kr_days10_dose30_1000_irri_auto_soil_1classe/"+y+"/output_test_"+y+".df","rb"))
    ET= {}
    gdf = {}
    
    Parcellaire["id"]=Parcellaire.ID
    ETsum = (df_mod.groupby(['id'])['Ir_auto'].sum()).reset_index()
    ETsum=ETsum.merge(df_mod[["id","LC"]],on='id')
    ETsum.drop_duplicates(inplace=True)
    ET_TAW =(df_mod.groupby(['id'])['TAW'].max()).reset_index()
    ET_TAW[ET_TAW.TAW<0]=0
    
    ET = ETsum
    ET["IRR"]=1
    ET.loc[(ET.Ir_auto==0.0),'IRR']=11
    # ET.loc[(ET.Ir_auto>0) & (ET.Ir_auto<=30),'IRR']=11
    gdf = Parcellaire
    gdf = gdf.merge(ET, on='id')
    gdf = gdf.merge(ET_TAW, on='id')
    IRR=gdf[gdf.IRR==1]
    print(r'Volumes irrigué == %s'%sum(IRR.Ir_auto*10*IRR.area/10000/1000000))
    print(r'surface irrigué == %s'%sum(IRR[IRR.Ir_auto!=0].area/10000))
    
    
    sum(gdf.Ir_auto*10*gdf.area/10000)/1000000
    sum(gdf[gdf.Ir_auto!=0].area/10000)
    print(r'surface maisNirr == %s'%sum(gdf[gdf.Ir_auto==0].area/10000))
    gdf_mais=gdf.loc[(gdf.code_cultu=="MIS") | (gdf.code_cultu =="MID") |(gdf.code_cultu =="MIE") ]
    gdf_mais["LC"]=gdf_mais.LC.astype(str) ## supprimer champ dtypes caterory non accepter par file SHP
    gdf_mais.to_file(d["PC_disk_water"]+"/TRAITEMENT/RUNS_SAMIR/RUN_CLASSIF_ALL_MAIS/Classif_init_ru_P055_Fcover_fewi_De_Kr_days10_dose30_1000_irri_auto_soil_1classe/"+y+"/carte_surface_irriguee_ADOUR_"+y+"_SAMIR_maxZr1000_2classe_seuil_0.shp")
   
    # Création de la carte besoin en eau 
    # gdf_mais.plot(column='Ir_auto',figsize=(10,10), vmin=ETmin, vmax=ETmax, cmap='RdYlGn')
    # plt.legend()
    # plt.title('Irrigation s aisonnière en mm')
    # plt.savefig(d["PC_disk_water"]+"/TRAITEMENT/RUNS_SAMIR/RUN_CLASSIF_ALL_MAIS/Classif_init_ru_P055_Fcover_fewi_De_Kr_days10_dose30_1200_irri_auto_soil/2017/carte_surface_irriguee_ADOUR_2017_SAMIR.png")
    # #  Création carte surfaces irriguées
    # gdf_mais.plot(column='IRR',figsize=(10,10), cmap='RdYlGn', legend=True)
    # plt.legend()
    # plt.title('Irrigué / pluviale')
    # plt.savefig(d["PC_disk_water"]+"/TRAITEMENT/RUNS_SAMIR/RUN_CLASSIF_ALL_MAIS/Classif_init_ru_P055_Fcover_fewi_De_Kr_days10_dose30_1200_irri_auto_soil/2017/carte_irrigue_ADOUR_pluvaile_2017_SAMIR.png")

    # plt.figure(figsize=(7,7))
    # plt.plot(ID1.date,ID1.Dr,c="r")
    # plt.plot(ID1.date,ID1.RAW)
    # # plt.plot(ID1.date,ID1.TAW*(0.55+0.04*(5-ID1.ET0)),linestyle="--",c='black',linewidth=0.5)
    # plt.plot(ID1[ID1.Ir_auto!=0]["date"],ID1[ID1.Ir_auto!=0]["Ir_auto"],linestyle='',marker="o",c='green')
    # ax2=plt.twinx(ax=None)
    # ax2.plot(ID1.date,(ID1.TAW- ID1.Dr)/(ID1.TAW * (1.-(0.55+0.04))))
    # ax2.set_ylim(0,3)
    
    
    # plt.figure(figsize=(6, 6))
    # ax1=plt.subplot(121)
    # p1=geo.plotting.plot_polygon_collection(ax1,geoms=gdf.geometry,values=gdf.IRR,label="Irr")
    # ax2=plt.subplot(122)
    # p2=geo.plotting.plot_polygon_collection(ax2,geoms=gdf.geometry,values=gdf.IRR,label="Nir")
    # # plt.legend([p1.get_array()[0],p1.get_array()[2]],("Irrigated Maize","Irrigated Soybean"),loc='upper left',fontsize=12)
    # plt.legend(handles=[p1.get_label(),p2.get_label()],loc='upper left',fontsize=12)
    
    # for poly in gdf['geometry'][0:100]:
    #     geo.plotting.plot_polygon_collection(ax, poly, alpha=1, facecolor='red', linewidth=0)

# =============================================================================
#  Isoler parcelle même zone SAFRAN mais résultat différent
# =============================================================================
    # id_NIRR=df_mod[df_mod.id==9362]
    # id_IRR=df_mod[df_mod.id==844]
    # id_IRRplus=df_mod[df_mod.id==5398]
    
    # for i in ['NDVI', 'Clay', 'Sand', 'FCov', 'TAW','RAW', 'Dr', 'Ks']:
    #     plt.figure(figsize=(7,7))
    #     plt.plot(id_NIRR.date,id_NIRR[i],label='Nirr')
    #     plt.plot(id_IRR.date,id_IRR[i],label='IRR')
    #     plt.plot(id_IRRplus.date,id_IRRplus[i],label='IRR +++')
    #     plt.legend()
    #     plt.title(i)