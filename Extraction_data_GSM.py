# -*- coding: utf-8 -*-
"""
Created on Wed May 12 10:13:50 2021

@author: yann
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
    # d["PC_disk"]="/run/media/pageot/Transcend/Yann_THESE/BESOIN_EAU/BESOIN_EAU/"
    d["PC_home"]="/mnt/d/THESE_TMP/"
    d["PC_home_Wind"]="D:/THESE_TMP/"
    d["PC_disk"]="H:/Yann_THESE/BESOIN_EAU/BESOIN_EAU/"
    d["PC_labo"]="/datalocal/vboxshare/THESE/BESOIN_EAU/"
    # d["PC_labo_disk"]="/run/media/pageot/Transcend/Yann_THESE/BESOIN_EAU/BESOIN_EAU/"


    #other méthode pondération 

    parcelle=geo.read_file("/run/media/pageot/Transcend/Yann_THESE/BESOIN_EAU/BESOIN_EAU/TRAITEMENT/INPUT_DATA/NESTE_SOIL_EXTRACT.shp")
    # PKGC=parcelle[['clay_l0_me', 'clay_l0__1', 'clay_l100_', 'clay_l10_1', 'clay_l15_m',
    #     'clay_l15_1', 'clay_l30_m', 'clay_l30_1', 'clay_l5_me', 'clay_l5__1',
    #     'clay_l60_m', 'clay_l60_1', 'sand_l0_me', 'sand_l0__1', 'sand_l100_',
    #     'sand_l10_1', 'sand_l15_m', 'sand_l15_1', 'sand_l30_m', 'sand_l30_1',
    #     'sand_l5_me', 'sand_l5__1', 'sand_l60_m', 'sand_l60_1', 'silt_l0_me',
    #     'silt_l0__1', 'silt_l100_', 'silt_l10_1', 'silt_l15_m', 'silt_l15_1',
    #     'silt_l30_m', 'silt_l30_1', 'silt_l5_me', 'silt_l5__1', 'silt_l60_m',
    #     'silt_l60_1']]
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
            a=np.average(PKGC[[v+"_l0_me",v+"_l5_me",v+"_l15_m",v+"_l30_m",v+'_l60_m']].iloc[i],weights=[1,5,10,15,30])
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
    
    
    data=[]
    a=data_parcelle_Texture.groupby("ID")
    for g in PKGC.ID:
         text=a.get_group(g)[0].to_list()
         data.append([g,text[0],text[1],text[2]])
    data_parcelle_Texture=pd.DataFrame(data,columns=["ID","Argile","Sable",'Limon'])
    data_parcelle_Texture.to_csv("/run/media/pageot/Transcend/Yann_THESE/BESOIN_EAU/BESOIN_EAU/TRAITEMENT/SOIL/GSM/Extract_GSM_parcelle_mais_NESTE_2017.csv")

    
