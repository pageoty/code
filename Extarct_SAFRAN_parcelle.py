#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 13:13:23 2021

# Extraction données SAFRAN 
@author: pageot
"""

import os
import pandas as pd
import geopandas as geo
import numpy as np
from osgeo import osr

if __name__ == '__main__':
    d={}
    d["PC_labo"]="/datalocal/vboxshare/THESE/BESOIN_EAU/"
    d["PC_labo_disk"]="/run/media/pageot/Transcend/Yann_THESE/BESOIN_EAU/BESOIN_EAU/"
    d["PC_home"]="/mnt/d/THESE_TMP/"
    d["PC_home_Wind"]="D:/THESE_TMP/"
    d["PC_disk"]="H:/Yann_THESE/BESOIN_EAU/BESOIN_EAU/"


    meteo=geo.read_file("/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/tmp/SAFRAN_extrat_test_large.shp")
    parcelle=geo.read_file("/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/tmp/SHAPE_test_SAFRAN_EXTART.shp")
    meteo.DATE=meteo.DATE.astype(int)
    meteo.DATE=pd.to_datetime(meteo.DATE,format="%Y%m%d")
    meteo.set_index("field_1",inplace=True)
    parcelle.set_index("idparcelle",inplace=True)
    # meteo["geometry"].distance(parcelle["geometry"].loc[0]).min() # sort id grid SAFRAN plus proche de la parcelle 0
    # extart_meteo=meteo.loc[meteo["geometry"].distance(parcelle["geometry"].loc[0]).idxmin()][['DATE',"PRELIQ_Q","T_Q","ETP_Q"]] # récuper info météo pour parcelle 0
    # meteo.loc[meteo["geometry"].distance(parcelle["geometry"].iloc[0])==meteo["geometry"].distance(parcelle["geometry"].iloc[0]).min()][['DATE',"PRELIQ_Q","T_Q","ETP_Q"]]

    resu=pd.DataFrame()
    idgeom=[]
    # loop
    for par in parcelle.index:
         extart_meteo=meteo.loc[meteo["geometry"].distance(parcelle["geometry"].iloc[0])==meteo["geometry"].distance(parcelle["geometry"].iloc[0]).min()][['DATE',"PRELIQ_Q","T_Q","ETP_Q"]]
         idgeom.append(np.repeat(par,extart_meteo.shape[0]))
         resu=resu.append(extart_meteo)
    idpar=pd.DataFrame(idgeom).stack().to_list()
    resu["idparcelle"]=idpar
    test=pd.merge(parcelle,resu[["DATE","ETP_Q","PRELIQ_Q","T_Q",'idparcelle']],on="idparcelle")

# mettre cela en df SAMIR 