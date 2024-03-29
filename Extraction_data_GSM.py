# -*- coding: utf-8 -*-
"""
Created on Wed May 12 10:13:50 2021

@author: yann

Extration des données pédologiques issues des rasters de GSM à partir d'un fichier SHP contenant les inforamtions
"""


import os
import sqlite3
import geopandas as geo 
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
import csv
import shapely.geometry as geom
import descartes
import pickle


if __name__ == '__main__':
    d={}
    d["PC_home"]="/mnt/d/THESE_TMP/"
    d["PC_home_Wind"]="D:/THESE_TMP/"
    d["PC_disk"]="H:/Yann_THESE/BESOIN_EAU/BESOIN_EAU/"
    d["PC_labo"]="/datalocal/vboxshare/THESE/BESOIN_EAU/"
    d["PC_labo_disk"]="/run/media/pageot/Transcend/Yann_THESE/BESOIN_EAU/BESOIN_EAU/"


#  Inputs 
    parcelle=geo.read_file("/run/media/pageot/Transcend/Yann_THESE/BESOIN_EAU/BESOIN_EAU/TRAITEMENT/SOIL/GSM/SOIL_GSM_CALSSIF_ADOUR_2018.shp")
    PKGC=parcelle[['clay_l0_me', 'clay_l100_',
       'clay_l15_m', 'clay_l30_m', 'clay_l5_me', 'clay_l60_m', 'sand_l0_me',
       'sand_l100_', 'sand_l15_m', 'sand_l30_m', 'sand_l5_me', 'sand_l60_m',
       'silt_l0_me', 'silt_l100_', 'silt_l15_m', 'silt_l30_m', 'silt_l5_me',
       'silt_l60_m']]
    ids=parcelle.ID
    PKGC=PKGC/1000*100
    PKGC["ID"]=ids
    # PKGC.set_index("ID",inplace=True)
    
    ids=[]
    entete=[]
    data_parcelle_Texture=[]
    for i in PKGC.index:
        for v in ["clay","sand","silt"]:
            a=np.average(PKGC[[v+"_l0_me",v+"_l5_me",v+"_l15_m",v+"_l30_m",v+'_l60_m',v+'_l100_']].iloc[i],weights=[1,5,10,15,30,40])
            # print("ID : %s "%PKGC.ID.iloc[i])
            # print(v)
            # print(a)
            data_parcelle_Texture.append(a)
            ids.append(PKGC.ID.iloc[i])
            entete.append(v)
    data_parcelle_Texture=pd.DataFrame(data_parcelle_Texture)
    data_parcelle_Texture["ID"]=ids
    data_parcelle_Texture["Variable_modale"]=entete
    # data_parcelle_Texture.to_csv("H:/Yann_THESE/BESOIN_EAU/BESOIN_EAU/TRAITEMENT/SOIL/GSM/Extract_GSM_parcelle_mais_ALL_Classif_Adour_2017.csv")
    # data_parcelle_Texture=pd.read_csv("/run/media/pageot/Transcend/Yann_THESE/BESOIN_EAU/BESOIN_EAU/TRAITEMENT/SOIL/GSM/Extract_GSM_parcelle_CACG_2017.csv")
    # data_parcelle_Texture.drop(columns=["Unnamed: 0"],inplace=True)
    data=[]
    a=data_parcelle_Texture.groupby("ID")
    for g in ids:
         text=a.get_group(g)[0].to_list()
         data.append([g,text[0],text[1],text[2]])
    data_parcelle_Texture=pd.DataFrame(data,columns=["ID","Argile","Sable",'Limon'])
    data_parcelle_Texture["PF"]=data_parcelle_Texture.eval("0.08+0.00401 * Argile - 0.000293* Sable")
    data_parcelle_Texture["CC"]=data_parcelle_Texture.eval("0.278+0.00245 * Argile - 0.00135* Sable")
    data_parcelle_Texture.drop_duplicates(inplace=True,subset=["ID"],keep='first')
    data_parcelle_Texture.to_csv("/run/media/pageot/Transcend/Yann_THESE/BESOIN_EAU/BESOIN_EAU/TRAITEMENT/SOIL/GSM/Extract_GSM_parcelle_RPG_2018_ADOUR_CLASSIF.csv")


# =============================================================================
# Drop ducplicate
# =============================================================================
    df=pd.read_csv("/run/media/pageot/Transcend/Yann_THESE/BESOIN_EAU/BESOIN_EAU/TRAITEMENT/SOIL/GSM/Extract_GSM_parcelle_ASA_2017_GSM_tri.csv",decimal=',')
    df.drop_duplicates(subset=["ID"],keep='first',inplace=True)
    df.to_csv("/run/media/pageot/Transcend/Yann_THESE/BESOIN_EAU/BESOIN_EAU/TRAITEMENT/SOIL/GSM/Extract_GSM_parcelle_ASA_2017_GSM_tri.csv")

