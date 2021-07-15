#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 14:00:05 2021

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


    Parcellaire= geo.read_file(d["PC_disk"]+"/DONNEES_RAW/DONNEES_MAIS_CLASSIF/Classif_Adour_2017_maïs_all.shp")
    Parcellaire["id"]=Parcellaire.ID
    df_mod=pickle.load(open(d["PC_disk"]+"/TRAITEMENT/RUNS_SAMIR/RUN_CLASSIF_ALL_MAIS/Classif_init_ru_P055_Fcover_fewi_De_Kr_days10_dose30_1200_irri_auto_soil/2017/output_test_2017.df","rb"))
    
    
    ETsum = (df_mod.groupby(['id'])['Ir_auto'].sum()).reset_index()
    
    ETmin = ETsum.Ir_auto.min()
    ETmax = ETsum.Ir_auto.max()
    ET= {}
    
    ET = ETsum
    ET["IRR"]=1
    ET.loc[(ET.Ir_auto==0.0),'IRR']=0
    gdf = {}
    gdf = Parcellaire
    gdf = gdf.merge(ET, on='id')
    # gdf.to_file(d["PC_disk"]+"/TRAITEMENT/RUNS_SAMIR/RUN_CLASSIF_ALL_MAIS/Classif_init_ru_P055_Fcover_fewi_De_Kr_days10_dose30_1200_irri_auto_soil/2017/carte_surface_irriguée_2017_SAMIR.shp")
    gdf.plot(column='Ir_auto',figsize=(10,10), vmin=ETmin, vmax=ETmax, cmap='RdYlGn', legend=True)
    plt.title('Irrigation saisonnière en mm')
    # plt.savefig(d["PC_disk"]+"/TRAITEMENT/RUNS_SAMIR/RUN_CLASSIF_ALL_MAIS/Classif_init_ru_P055_Fcover_fewi_De_Kr_days10_dose30_1200_irri_auto_soil/2017/carte_surface_irriguée_2017_SAMIR.png")
    gdf.plot(column='IRR',figsize=(10,10), cmap='RdYlGn', legend=True)
    plt.legend()
    plt.title('Irrigué / pluviale')
    # plt.savefig(d["PC_disk"]+"/TRAITEMENT/RUNS_SAMIR/RUN_CLASSIF_ALL_MAIS/Classif_init_ru_P055_Fcover_fewi_De_Kr_days10_dose30_1200_irri_auto_soil/2017/carte_irrigue_pluvaile_2017_SAMIR.png")



# test df.ix
# df.ix[:4, 1:3]