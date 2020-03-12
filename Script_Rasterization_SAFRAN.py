#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 09:20:52 2019

@author: pageot
"""


import os
import sqlite3
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
import numpy as np 
import seaborn as sns
import csv
import otbApplication


#df=pd.read_csv("/datalocal/vboxshare/THESE/CLASSIFICATION/DONNES_SIG/DONNEES_METEO/SAFRAN_ALL.csv")
#df1=df.drop([0])
#Temp=df1[["DATE","T_Q","X","Y"]]
#Temp2=Temp.sort_values(by="DATE")
#
#DATE=Temp.DATE
#date=set(list(DATE))
#
#DATE_Temp=Temp.groupby("DATE")
#for i in DATE_Temp:
#    globals()["SAFRAN_%s"% list(i)[0]]=DATE_Temp.get_group(list(i)[0])
#    globals()["SAFRAN_%s"% list(i)[0]].to_csv("/datalocal/vboxshare/THESE/CLASSIFICATION/TRAITEMENT/DATA_METEO/Temp/SAFRAN_ALL/CSV/SAFRAN_%s.csv"% list(i)[0])
#    # Tarsnformation en SHP   
#    os.system("ogr2ogr -s_srs EPSG:2154 -t_srs EPSG:2154 -oo X_POSSIBLE_NAMES=X* -oo Y_POSSIBLE_NAMES=Y*  -f 'ESRI Shapefile' /datalocal/vboxshare/THESE/CLASSIFICATION/TRAITEMENT/DATA_METEO/Temp/SAFRAN_ALL/SHP/SAFRAN_%s.shp /datalocal/vboxshare/THESE/CLASSIFICATION/TRAITEMENT/DATA_METEO/Temp/SAFRAN_ALL/CSV/SAFRAN_%s.csv"%(list(i)[0],list(i)[0]))
#    os.system("gdal_rasterize -a T_Q -ot Float32 -of GTiff -tr 8000 8000 -co COMPRESS=DEFLATE -co PREDICTOR=1 -co ZLEVEL=6 -l SAFRAN_%s /datalocal/vboxshare/THESE/CLASSIFICATION/TRAITEMENT/DATA_METEO/Temp/SAFRAN_ALL/SHP/SAFRAN_%s.shp /datalocal/vboxshare/THESE/CLASSIFICATION/TRAITEMENT/DATA_METEO/Temp/SAFRAN_ALL/RASTER/RASTER_SAFRAN%s.tif"%(list(i)[0],list(i)[0],list(i)[0]))

## =============================================================================
## Concaterner l'ensemble des data pour otbenir une seule images
# =============================================================================


d={}
d["data_file1"]="/datalocal/vboxshare/THESE/CLASSIFICATION/TRAITEMENT/SOIL_MOISTURE/SM_TILES/"
d["data_file"]="/datalocal/vboxshare/THESE/CLASSIFICATION/TRAITEMENT/DATA_METEO/Temp/SAFRAN_ALL/"
d["RASTER"]= d["data_file"]+"/RASTER"
images=[]
with open(d["data_file1"]+"list_SM_TYP2018.txt", 'rb+') as csvfile:
    for line in csvfile:
#        print (line.rstrip())
        images.append(line.rstrip())

images = [x.decode('UTF8') for x in images]
commande="otbcli_ConcatenateImages -il"
for q in images:
    print (q)
    commande=commande + " %s/"%(d["data_file1"])+"T30_2018/" + str(q)
commande=commande + " -out /datalocal/vboxshare/THESE/CLASSIFICATION/TRAITEMENT/SOIL_MOISTURE/STACK_SM/RASTER_SM_TYP_vege2018.tif"
print (commande)
os.system(commande)


# =============================================================================
# loop SAFRAN concat tous les 10 jours
# =============================================================================
#for j in np.arange(0,len(images),10):
#    print (j)
#    print (j+9)
#    commande="otbcli_ConcatenateImages -il"
#    for i in np.arange(j,j+9):
#        print (i)
#        commande=commande + " %s/"%(d["RASTER"]) + str(images[i])
#    commande=commande + " -out /datalocal/vboxshare/THESE/CLASSIFICATION/TRAITEMENT/DATA_METEO/Temp/SAFRAN_ALL/RASTER_10J/RASTER_SAFRAN_Concatenat_%s.tif"%(images[i][13:-4])
#    print (commande)
#    os.system(commande) 


