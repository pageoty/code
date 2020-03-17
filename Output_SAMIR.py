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



if __name__ == '__main__':
    d={}
    d["path_labo"]="/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/"
    d["path_PC"]="D:/THESE_TMP/RUNS_SAMIR/R1/Inputdata/"
    d["PC_disk"]="G:/Yann_THESE/BESOIN_EAU/"
    runs=["R4",'R5']
    for r in runs:
        d["path_PC"]='D:/THESE_TMP/RUNS_SAMIR/RUN_RELATION/'+str(r)+'/Inputdata/'
        for s in os.listdir(d["path_PC"][:-10]):
            if "output" in s:
                df=pickle.load(open(d["path_PC"][:-10]+str(s),'rb'))
                for id in list(set(df.id)):
                    lam=df.loc[df.id==id]
                    print(r'n° du runs : %s '% r)
                    print(r' n° parcelle : %s' %id)
                    print(r'sum irrigation in mm : %s'%lam.groupby(["LC","id"])["Ir_auto"].sum()[0])
                    print(r' nb irrigation : %s' %lam.Ir_auto.where(df["Ir_auto"] != 0.0).dropna().count())
                    plt.figure(figsize=(5,5))
                    plt.title(r'run :%s : n° parcelle :%s '%(r,id))
                    plt.plot(lam.date,lam.Ir_auto)
                    plt.ylabel("quantité d'eau")
                    plt.savefig(d["path_PC"][:-10]+"plt_quantity_irri_%s.png"%id)
                
# =============================================================================
# Validation modelisation ETR     
# =============================================================================
##    Prépartion des datas Eddy-co au format journalière
#    ETR_lam=pd.read_csv(d["PC_disk"]+"/DATA_ETR_CESBIO/ECPP_2017_2018_level3_lec.csv",encoding = 'utf-8',delimiter=";")
#    ETR_lam["Date/Time"]=ETR_lam["Date/Time"].apply(lambda x:x[0:10])
#    ETR_lam["Date/Time"]=pd.to_datetime(ETR_lam["Date/Time"],format="%d/%m/%Y")
#    ETR_lam_day=ETR_lam.groupby("Date/Time")["ETpot"].mean()
#    
#    ETR_lam=pd.read_csv(d["PC_disk"]+"/DATA_ETR_CESBIO/eddypro_FR-Lam_full_output_2020-01-28T012345_adv.csv")
#    ETR_lam["date"]=pd.to_datetime(ETR_lam["date"],format='%Y-%m-%d')
#    ETR_lam_day=ETR_lam.groupby("date")["ET"].mean()
#    ## Récuparation date végétation sur ETR 


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
    